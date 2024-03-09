from PIL import Image
import yaml
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from omegaconf import OmegaConf as om
from tqdm import tqdm 
from typing import Any, Dict, List, Optional, Union

import torch
import streaming
from omegaconf import DictConfig, ListConfig

import sys
sys.path.append("/mnt/workdisk/jasmine/llm-foundry")
from scripts.train.train import validate_config, build_composer_model
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.utils.config_utils import (log_config, pop_config,
                                           process_init_device,
                                           update_batch_size_info)
from llmfoundry.utils.builders import (add_metrics_to_eval_loaders,
                                       build_algorithm, build_callback,
                                       build_evaluators, build_logger,
                                       build_optimizer, build_scheduler,
                                       build_tokenizer)

yaml_path = '/mnt/workdisk/jasmine/yamls/llava/mpt-7b-llava-local.yaml'
cfg = om.load(yaml_path)

validate_config(cfg)
om.resolve(cfg)
print(cfg)

# Mandatory model training configs
model_config: DictConfig = pop_config(cfg, 'model', must_exist=True)
tokenizer_config: Dict[str, Any] = pop_config(cfg,
                                                'tokenizer',
                                                must_exist=True,
                                                convert=True)
optimizer_config: Dict[str, Any] = pop_config(cfg,
                                                'optimizer',
                                                must_exist=True,
                                                convert=True)
scheduler_config: Dict[str, Any] = pop_config(cfg,
                                                'scheduler',
                                                must_exist=True,
                                                convert=True)
train_loader_config: DictConfig = pop_config(cfg,
                                                'train_loader',
                                                must_exist=True)
max_duration: Union[int, str] = pop_config(cfg,
                                            'max_duration',
                                            must_exist=True)
eval_interval: Union[int, str] = pop_config(cfg,
                                            'eval_interval',
                                            must_exist=True)
precision: str = pop_config(cfg, 'precision', must_exist=True)
max_seq_len: int = pop_config(cfg, 'max_seq_len', must_exist=True)
device_train_batch_size = 1
# Build tokenizer
tokenizer_name = tokenizer_config['name']
tokenizer_kwargs = tokenizer_config.get('kwargs', {})
tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

# Build train loader
streaming.base.util.clean_stale_shared_memory()
train_loader = build_dataloader(
    train_loader_config,
    tokenizer,
    device_train_batch_size,
)


for i, batch in enumerate(train_loader.dataloader):
    # print(i)
    # print(batch)
    # new_dict ={}
    # for key in batch:
    #     new_dict[key] = batch[key]

    # import json
    # with open('batch.json', 'w') as fp:
    #     json.dump(new_dict, fp)

    # json_string = json.dumps(batch)
    # print(json_string)
    torch.save(batch, 'batch.ckpt')
    break