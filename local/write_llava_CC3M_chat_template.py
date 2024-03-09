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
from datasets import load_dataset
import json

fields = {'image': 'jpeg', 
          'prompt': 'str', 
          'response': 'str'}
remote = 'oci://mosaicml-internal-dataset-llava/LLaVA-CC3M-Pretrain-595K'

SYSTEM = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'
tok = transformers.AutoTokenizer.from_pretrained('rajammanabrolu/gpt-4-chat', trust_remote_code=True)

image_folder = '/mnt/workdisk/jasmine/data/llava/LLaVA-CC3M-Pretrain-595K/images'
dataset_path = '/mnt/workdisk/jasmine/data/llava/LLaVA-CC3M-Pretrain-595K/chat.json'
with open(dataset_path, 'r') as f:
    dataset = json.load(f)
# dataset = load_dataset(dataset_path)
# import pdb;pdb.set_trace()


with MDSWriter(out=remote, columns=fields) as out:
    # format multi-turn data

    for i in tqdm(range(len(dataset))):
        img_path = os.path.join(image_folder, dataset[i]['image'])
        image = Image.open(img_path)
        image = image.convert('RGB')

        convo = dataset[i]['conversations']
        # conv = conversation_lib.default_conversation.copy()
        # if roles[source[0]["from"]] != conv.roles[0]:
        if convo[0]['from'] != 'human':
            convo = convo[1:]
            
        formatted_convo = [{'role': 'system', 'content': SYSTEM}]
        for line in convo:
            if line['from'] == 'human':
                formatted_convo.append({'role': 'user', 'content': line['value']})
            elif line['from'] == 'gpt':
                formatted_convo.append({'role': 'assistant', 'content': line['value']})
            else:
                assert False, 'unrecognized from: %s'%line['from']

        for j in range(len(formatted_convo)//2):
            prompt = tok.apply_chat_template(formatted_convo[:2*(j+1)], tokenize=False, add_generation_prompt=True)
            response = formatted_convo[2*(j+1)]['content']
            # data_pt = {'prompt': tok.apply_chat_template(formatted_convo[:2*(j+1)], tokenize=False, add_generation_prompt=True), 
                    # 'response': formatted_convo[2*(j+1)]['content']}
            
            # print('prompt', prompt)
            # print('response', response)
            mds_sample = {'image': image, 'prompt': prompt, 'response': response}

            out.write(mds_sample)
        # if i == 5:
        #     break

# print(len(training_data))
# for i in range(20):
#     print(training_data[i])
            
print('done')

# with MDSWriter(out=remote, columns=fields) as out:
#     for path in tqdm(image_files):
#         image = Image.open(path)
#         image = image.convert('RGB')
#         captions = get_prompt(path)
#         mds_sample = {'image': image, 'caption': captions}
#         out.write(mds_sample)