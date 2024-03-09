from PIL import Image
import torch
import requests
from transformers import (AutoProcessor, LlavaForConditionalGeneration, 
                          CLIPVisionConfig, AutoConfig, LlavaConfig,
                          AutoTokenizer)
from composer import Trainer
from llmfoundry.models.hf import ComposerHFLLaVa
from omegaconf import OmegaConf as om

# Set up model config
vision_config = CLIPVisionConfig.from_pretrained('openai/clip-vit-large-patch14-336')
text_config = AutoConfig.from_pretrained('mistralai/Mistral-7B-v0.1')
llava_config = LlavaConfig(vision_config, text_config)

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer.add_tokens(['<image>', '<pad>'], special_tokens=True)
tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_PROMPT' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '<|im_end|>\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% endif %}{% endfor %}"

# Create model
model = LlavaForConditionalGeneration(llava_config)
model.resize_token_embeddings(len(tokenizer))

# print('loading our model..')
# model_ckpt_path = '/mnt/workdisk/jasmine/llm-foundry/checkpoints/jasmine/llava-mistral7b-ft-chat/checkpoints/ep0-ba1000-rank0.pt'
# # model_ckpt_path = '/mnt/workdisk/jasmine/llm-foundry/checkpoints/jasmine/llava-mistral7b-pretrain-cc3m/checkpoints/ep1-ba4651-rank0.pt'
# # model_ckpt_path = '/mnt/workdisk/jasmine/llm-foundry/checkpoints/jasmine/llava-mistral7b-pretrain-cc3m/checkpoints/ep0-ba1000-rank0.pt'


# state_dict = torch.load(model_ckpt_path)
# print('checkpoint loaded')

# renamed_state_dict = {}
# for key in state_dict['state']['model'].keys():
#     # strip 'model' from key
#     new_key = '.'.join(key.split('.')[1:])
#     renamed_state_dict[new_key] = state_dict['state']['model'][key]
# model.load_state_dict(renamed_state_dict)
model.to(torch.bfloat16)
model = model.to('cuda')

# TODO not the right processor but ok for now
processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")

# prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
prompt = "What's the content of the image?\n<image>"
# prompt = """<|im_start|> system\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible.<|im_end|> \n<|im_start|> user\n<image> \nWhat's the content of the image?<|im_end|> \n<|im_start|> assistant\n"""
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")
for key in inputs:
    inputs[key] = inputs[key].to('cuda') 

# Generate
generate_ids = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=500) #  max_length=30)
res = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(res)