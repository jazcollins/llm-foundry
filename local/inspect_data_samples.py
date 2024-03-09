""" 
    Write LLaVA pretrain dataset as MDS format on OCI with applied chat formatting.
"""

import os
from streaming.base import MDSWriter
from PIL import Image
from tqdm import tqdm
from PIL import Image
import torch
import transformers
import json

from omegaconf import OmegaConf as om
from tqdm import tqdm 
from typing import Any, Dict, List, Optional, Union

import torch
import streaming
from omegaconf import DictConfig, ListConfig

import sys
sys.path.append("/mnt/workdisk/jasmine/llm-foundry")
from scripts.train.train import validate_config
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.utils.config_utils import (pop_config, update_batch_size_info)
from llmfoundry.utils.builders import build_tokenizer
import copy

fields = {'image': 'jpeg', 
          'messages': 'json'}
# remote = 'oci://mosaicml-internal-dataset-llava/LLaVA-Mix-FT-665K-Chat-v2'
remote = 'oci://mosaicml-internal-dataset-llava/LLaVA-LCS-Pretrain-558K'

SYSTEM = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'
# tokenizer = transformers.AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
# # Prepare tokenizer -- uncertain about this??
# tokenizer.add_tokens.add_tokens(['<image>', '<pad>'], special_tokens=True)
# tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
# tokenizer.add_special_tokens({'bos_token': '<|im_start|>', 'eos_token': '<|im_end|>'}) 

yaml_path = '/mnt/workdisk/jasmine/yamls/llava/mpt-7b-llava-local.yaml'
cfg = om.load(yaml_path)

# Check for incompatibilities between the model and data loaders
validate_config(cfg)

# Resolve all interpolation variables as early as possible
om.resolve(cfg)

# Create copy of config for logging
logged_cfg: DictConfig = copy.deepcopy(cfg)

# Set seed first
seed: int = pop_config(cfg, 'seed', must_exist=True)

# Get global and device batch size information from distributed/single node setting
cfg = update_batch_size_info(cfg)


tokenizer_config: Dict[str, Any] = pop_config(cfg,
                                                'tokenizer',
                                                must_exist=True,
                                                convert=True)
train_loader_config: DictConfig = pop_config(cfg,
                                                'train_loader',
                                                must_exist=True)
print(train_loader_config)

# Build tokenizer
tokenizer_name = tokenizer_config['name']
tokenizer_kwargs = tokenizer_config.get('kwargs', {})
tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

# just added
chat_template =  "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_PROMPT' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '<|im_end|>\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% endif %}{% endfor %}"
tokenizer.chat_template = chat_template
# no
# tokenizer.add_special_tokens({'bos_token': '<|im_start|>', 'eos_token': '<|im_end|>'}) 
tokenizer.add_tokens(['<image>', '<pad>', '<|im_end|>', '<|im_start|>'], special_tokens=True)
print(tokenizer)

# Dataloaders
device_train_batch_size = 1
import streaming
streaming.base.util.clean_stale_shared_memory()
train_loader = build_dataloader(
    train_loader_config,
    tokenizer,
    device_train_batch_size,
)

skip_special_tokens = False
for i, sample in enumerate(train_loader.dataloader):
    # print(sample.keys())
    # print(sample['pixel_values'].shape)
    
    # print(sample['input_ids'].shape, sample['labels'].shape)
    # input_ids = torch.where(sample['input_ids'] != -100, sample['input_ids'], tokenizer.pad_token_id)
    input_ids = sample['input_ids']
    decoded_input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)

    # print('labels', sample['labels'])
    labels = torch.where(sample['labels'] != -100, sample['labels'], tokenizer.pad_token_id)
    decoded_text = tokenizer.batch_decode(labels, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
    print('INPUT_IDS:', decoded_input_text)
    print('LABELS:', decoded_text)
    print('-----')

    if i > 10:
        break