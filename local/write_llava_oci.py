""" 
    Write LLaVA datasets as MDS format on OCI.
"""

import os
from streaming.base import MDSWriter
from PIL import Image
from tqdm import tqdm
from PIL import Image
import torch
import transformers
from datasets import load_dataset

fields = {'image': 'jpeg', 
          'prompt': 'str', 
          'response': 'str'}
remote = 'oci://mosaicml-internal-dataset-llava/llava_instruct_80k_multiturn'

SYSTEM = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'
image_folder = '/mnt/workdisk/jasmine/data/llava/data/train2017'
dataset = load_dataset("/mnt/workdisk/jasmine/data/llava/LLaVA-Instruct-150K/v1-ft")
tok = transformers.AutoTokenizer.from_pretrained('rajammanabrolu/gpt-4-chat', trust_remote_code=True)

with MDSWriter(out=remote, columns=fields) as out:
    # format multi-turn data
    # training_data = []
    for i in tqdm(range(dataset.num_rows['train'])):
        img_path = os.path.join(image_folder, dataset['train'][i]['image'])
        image = Image.open(img_path)
        image = image.convert('RGB')

        convo = dataset["train"][i]['conversations']
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
            
            mds_sample = {'image': image, 'prompt': prompt, 'response': response}
            out.write(mds_sample)
            # training_data.append(data_pt)

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