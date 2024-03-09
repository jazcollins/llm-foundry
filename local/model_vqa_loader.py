import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
from enum import auto

from transformers import (AutoProcessor, LlavaForConditionalGeneration, 
                          CLIPVisionConfig, AutoConfig, LlavaConfig,
                          AutoTokenizer, CLIPImageProcessor)

# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

import sys
sys.path.append("/mnt/workdisk/jasmine/LLaVA")
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import Conversation # conv_templates # , SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# # Model Constants
# IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 32000 # -200
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"
# IMAGE_PLACEHOLDER = "<image-placeholder>"

# conv_chatml_direct = Conversation(
#     system="""<|im_start|>system
# Answer the questions.""",
#     roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
#     version="mpt",
#     messages=(),
#     offset=0,
#     sep_style=auto(),
#     sep="<|im_end|>",
# )

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

def _square_pad_img(img, bg_color=(255, 255, 255)):
    '''
        img: PIL Image
    '''
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), bg_color)
        result.paste(img, (0, (width - height) // 2))
    else:
        result = Image.new(img.mode, (height, height), bg_color)
        result.paste(img, ((height - width) // 2, 0))
    return result

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# TODO rm if not using
processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        # if True: # self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_chatml_direct.copy() # conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        conv = [{'role': 'system', 'content': 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'}]
        conv.append({'role': 'user', 'content': '<image>  \n' + qs}) # dont actually think we need all these spaces
        # conv.append({'role': 'assistant', 'content': ''}) # PROVLEM - processor is adding <im_end>

        # TODO still need to check that prompt is formed exactly right       
        prompt = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        # image = _square_pad_img(image)
        # image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        inputs = processor(text=prompt, images=image, return_tensors="pt")

        
        input_ids = inputs['input_ids'][0]
        image_tensor = inputs['pixel_values'][0]

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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

    # Create image processor
    # image_processor = vision_tower.image_processor
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')


    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    # print("CUTTING QUESTIONS DOWN BC TESTING!")
    # questions = questions[0:10]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")



    # if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
    #     args.conv_mode = args.conv_mode + '_mmtag'
    #     print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                pixel_values=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                # pixel_values=image_tensor.to(dtype=torch.float16, device='cpu', non_blocking=True),
                # image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id)

        # input_ids = input_ids.to(device='cuda', non_blocking=True)
        # image_tensor = image_tensor.to(device='cuda', non_blocking=True)

        # # Generate
        # # print('input ids', input_ids)
        # # print('pixel values', image_tensor)
        # output_ids = model.generate(input_ids=input_ids, pixel_values=image_tensor, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500) 

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


        print('outputs:', outputs)
        print('-------')

        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                #    "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/mnt/workdisk/jasmine/llm-foundry/checkpoints/jasmine/llava-mistral7b-ft-chat/checkpoints/ep0-ba1000-rank0.pt")
    # parser.add_argument("--image-folder", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/test2015")
    # parser.add_argument("--question-file", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl")
    # parser.add_argument("--answers-file", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-mistral7b-ft-chat-flashattn2-mb8/answer.jsonl")

    parser.add_argument("--model-path", type=str, default='/mnt/workdisk/jasmine/llm-foundry/checkpoints/jasmine/llava-mistral7b-ft-chat-flashattn2-mb8/checkpoints/ep0-ba9000-rank0.pt')
    parser.add_argument("--image-folder", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/pope/val2014/")
    parser.add_argument("--question-file", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/pope/llava_pope_test.jsonl")
    parser.add_argument("--answers-file", type=str, default="/mnt/workdisk/jasmine/data/llava/eval/eval/pope/answers/llava-mistral7b-ft-chat-flashattn2-mb8.jsonl")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)