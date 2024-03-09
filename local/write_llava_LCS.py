""" 
    Write LLaVA LCS pretrain dataset as MDS format on OCI.
"""

import os
from streaming.base import MDSWriter
from PIL import Image
from tqdm import tqdm
from PIL import Image
import json

fields = {'image': 'jpeg', 
          'prompt': 'str', 
          'response': 'str'}
remote = 'oci://mosaicml-internal-dataset-llava/LLaVA-LCS-Pretrain-558K'
image_folder = '/mnt/workdisk/jasmine/data/llava/LLaVA-LCS-Pretrain-558K/images'
dataset_path = '/mnt/workdisk/jasmine/data/llava/LLaVA-LCS-Pretrain-558K/blip_laion_cc_sbu_558k.json'

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

with MDSWriter(out=remote, columns=fields) as out:
    for i in tqdm(range(len(dataset))):
        img_path = os.path.join(image_folder, dataset[i]['image'])
        image = Image.open(img_path)
        image = image.convert('RGB')

        convo = dataset[i]['conversations']
        if len(convo) != 2:
            print('poorly formatted convo')
            continue
        if not convo[0]['from'] == 'human' and convo[1]['from'] == 'gpt':
            print('poorly formatted convo')
            continue

        prompt = convo[0]['value']
        response = convo[1]['value']

        mds_sample = {'image': image, 'prompt': prompt, 'response': response}

        out.write(mds_sample)


print('done')

