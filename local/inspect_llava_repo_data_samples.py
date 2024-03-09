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
from transformers import AutoTokenizer, CLIPImageProcessor
import json

from omegaconf import OmegaConf as om
from tqdm import tqdm 
from typing import Any, Dict, List, Optional, Union

import torch
import streaming
from omegaconf import DictConfig, ListConfig
from dataclasses import dataclass, field

import sys
sys.path.append("/mnt/workdisk/jasmine/LLaVA")

from llava.train.train import make_supervised_data_module

# def preprocess_plain(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         assert len(source) == 2
#         assert DEFAULT_IMAGE_TOKEN in source[0]['value']
#         source[0]['value'] = DEFAULT_IMAGE_TOKEN
#         conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
#         conversations.append(conversation)
#     # tokenize conversations
#     input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
#         target[:tokenized_len] = IGNORE_INDEX

#     return dict(input_ids=input_ids, labels=targets)


# def preprocess(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     """
#     Given a list of sources, each is a conversation list. This transform:
#     1. Add signal '### ' at the beginning each sentence, with end signal '\n';
#     2. Concatenate conversations together;
#     3. Tokenize the concatenated conversation;
#     4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
#     """
#     if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
#         return preprocess_plain(sources, tokenizer)
#     if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
#         return preprocess_llama_2(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version.startswith("v1"):
#         return preprocess_v1(sources, tokenizer, has_image=has_image)
#     if conversation_lib.default_conversation.version == "mpt":
#         return preprocess_mpt(sources, tokenizer, has_image=has_image)
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         header = f"{conversation_lib.default_conversation.system}\n\n"
#         conversation = _add_speaker_and_signal(header, source)
#         conversations.append(conversation)
#     # tokenize conversations
#     def get_tokenize_len(prompts):
#         return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

#     if has_image:
#         input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     else:
#         conversations_tokenized = _tokenize_fn(conversations, tokenizer)
#         input_ids = conversations_tokenized["input_ids"]

#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         if has_image:
#             tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
#         else:
#             tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
#         speakers = [sentence["from"] for sentence in source]
#         _mask_targets(target, tokenized_lens, speakers)

#     return dict(input_ids=input_ids, labels=targets)

# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  data_args: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         list_data_dict = json.load(open(data_path, "r"))

#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         self.list_data_dict = list_data_dict
#         self.data_args = data_args

#     def __len__(self):
#         return len(self.list_data_dict)

#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if 'image' in sources[0]:
#             image_file = self.list_data_dict[i]['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result
#                 image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in self.list_data_dict[i]))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         return data_dict


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances]
#                                   for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels,
#                                                  batch_first=True,
#                                                  padding_value=IGNORE_INDEX)
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
#         labels = labels[:, :self.tokenizer.model_max_length]
#         batch = dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

#         if 'image' in instances[0]:
#             images = [instance['image'] for instance in instances]
#             if all(x is not None and x.shape == images[0].shape for x in images):
#                 batch['images'] = torch.stack(images)
#             else:
#                 batch['images'] = images

        # return batch

# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
#                                 data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
#                                 data_path=data_args.data_path,
#                                 data_args=data_args)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset,
#                 eval_dataset=None,
#                 data_collator=data_collator)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square',


img_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-13b-v1.5')
# tokenizer.add_tokens(['<image>', '<pad>', '<|im_end|>', '<|im_start|>'], special_tokens=True)
# tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not 'system' in messages[0]['role'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_PROMPT' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% if system_message != false %}{{ '<|im_start|>system\n' + system_message.strip() + '<|im_end|>\n'}}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% else %}{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}{% endif %}{% if (add_generation_prompt == true and loop.last) %}{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}{% endif %}{% endfor %}"

# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     model_args.model_name_or_path,
#     cache_dir=training_args.cache_dir,
#     model_max_length=training_args.model_max_length,
#     padding_side="right",
#     use_fast=False,
# )

data_args = DataArguments(data_path='/mnt/workdisk/jasmine/data/llava/LLaVA-Instruct-150K/v1_5/llava_v1_5_mix665k.json',
                          lazy_preprocess=True,
                          is_multimodal=True,
                          image_folder='/mnt/workdisk/jasmine/data/llava/data',
                          image_aspect_ratio='pad')
data_args.image_processor = img_processor     
data_args.mm_vision_select_layer = -2
data_args.mm_use_im_start_end = False
data_args.mm_use_im_patch_token = False


data_module = make_supervised_data_module(tokenizer=tokenizer,
                                          data_args=data_args)

i = 0
for batch in data_module['train_dataset']:
    # print(batch)

    batch['input_ids'] = torch.where(batch['input_ids'] != -200, batch['input_ids'], tokenizer.pad_token_id) # IMAGE_TOKEN_INDEX
    decoded_input_ids = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print('DECODED INPUT IDS:', ' '.join(decoded_input_ids))

    batch['labels'] = torch.where(batch['labels'] != -100, batch['labels'], tokenizer.pad_token_id) # IGNORE_INDEX
    decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print('DECODED LABELS:', ' '.join(decoded_labels))

    import pdb;pdb.set_trace()
    print()
    print()
    
    i += 1
    if i > 5:
        break


print('done')