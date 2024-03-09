import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid

from transformers import (AutoProcessor, LlavaForConditionalGeneration, 
                          CLIPVisionConfig, AutoConfig, LlavaConfig,
                          AutoTokenizer, CLIPImageProcessor)

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

IMAGE_TOKEN_INDEX = 32000

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    From https://github.com/haotian-liu/LLaVA/blob/5d8f1760c08b7dfba3ae97b71cbd4c6f17d12dbd/llava/utils.py#L93
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# TODO rm if not using
processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")

def eval_model(args):
    # Model
    disable_torch_init()

    model_name = args.model_path.split('/')[-3]

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
    tokenizer.add_tokens(['<image>', '<pad>', '<|im_end|>', '<|im_start|>'], special_tokens=True)
    # tokenizer.add_tokens(['<image>', '<pad>'], special_tokens=True)
    tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_PROMPT' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '<|im_end|>\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% endif %}{% endfor %}"

    # Set up model config
    vision_config = CLIPVisionConfig.from_pretrained('openai/clip-vit-large-patch14-336')
    text_config = AutoConfig.from_pretrained('mistralai/Mistral-7B-v0.1')
    llava_config = LlavaConfig(vision_config, text_config)

    # Create model
    model = LlavaForConditionalGeneration(llava_config)
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load(args.model_path)
    print('checkpoint loaded')

    renamed_state_dict = {}
    for key in state_dict['state']['model'].keys():
        # strip 'model' from key
        new_key = '.'.join(key.split('.')[1:])
        renamed_state_dict[new_key] = state_dict['state']['model'][key]
    model.load_state_dict(renamed_state_dict)
    model = model.to('cuda')

    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = [{'role': 'system', 'content': 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'}]
        conv.append({'role': 'user', 'content': '<image>\n' + cur_prompt})

        # TODO still need to check that prompt is formed exactly right       
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            image_tensor = inputs['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
        else:
            images = None
            inputs = processor(text=prompt, images=images, return_tensors="pt")

        input_ids = inputs['input_ids'][0].unsqueeze(0).cuda()
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # images=images,
                pixel_values=images,
                # image_sizes=image_sizes,
                do_sample=True if args.temperature > 0.0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print('OUTPUTS:\n', outputs)

        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                #    "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='/mnt/workdisk/jasmine/llm-foundry/checkpoints/jasmine/llava-mistral7b-ft-chat-flashattn2-mb8/checkpoints/ep0-ba9000-rank0.pt')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/mnt/workdisk/jasmine/ScienceQA/test")
    parser.add_argument("--question-file", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--answers-file", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/scienceqa/answers/llava-v1.5-13b.jsonl") 
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer-prompter", action="store_true")
    # parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)