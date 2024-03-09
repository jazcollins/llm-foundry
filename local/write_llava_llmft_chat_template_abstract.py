""" 
    Write LLaVA finetuning dataset as MDS format on OCI in abstract chat formatting.
"""

import os
from streaming.base import MDSWriter
from PIL import Image
from tqdm import tqdm
import json

fields = {'image': 'jpeg', 
          'messages': 'json'}
remote = 'oci://mosaicml-internal-dataset-llava/LLaVA-Mix-FT-665K-Chat'
image_folder = '/mnt/workdisk/jasmine/data/llava/data'
dataset_path = '/mnt/workdisk/jasmine/data/llava/llava_v1_5_mix665k.json'
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

SYSTEM = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.'

with MDSWriter(out=remote, columns=fields) as out:
    for i in tqdm(range(len(dataset))):
        try:
            # Image is missing
            img_path = os.path.join(image_folder, dataset[i]['image'])
        except:
            continue
        try:
            # Some images may be missing
            image = Image.open(img_path)
            image = image.convert('RGB')
        except:
            continue
        
        convo = dataset[i]['conversations']
        if convo[0]['from'] != 'human':
            convo = convo[1:]
        
        # Chat formatting in foundry only creates one single-turn convo from list of messages
        # To get around this, create multiple single-turn convos from list of multi-turn statements
        for j in range(len(convo) // 2):
            formatted_convo = [{'role': 'system', 'content': SYSTEM}]
            for line in convo[:(j+1)*2]:
                if line['from'] == 'human':
                    formatted_convo.append({'role': 'user', 'content': line['value']})
                elif line['from'] == 'gpt':
                    formatted_convo.append({'role': 'assistant', 'content': line['value']})
                else:
                    assert False, 'unrecognized from: %s'%line['from']

            mds_sample = {'image': image, 'messages': formatted_convo}

            out.write(mds_sample)
