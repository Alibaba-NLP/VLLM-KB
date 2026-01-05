import re
import os
import argparse
from tqdm import tqdm
from user_template import (USER_TEMPLATE_CN, USER_TEMPLATE_CN_SCORE, 
                           USER_TEMPLATE_EN, USER_TEMPLATE_EN_SCORE,
                           USER_TEMPLATE_EN_DS_INFERENCE, USER_TEMPLATE_EN_SCORE_DS_INFERENCE,
                           USER_TEMPLATE_CN_DS_INFERENCE, USER_TEMPLATE_CN_SCORE_DS_INFERENCE
)

import json
import time
from datetime import datetime
import deepseek_vl
from transformers import AutoModel, AutoTokenizer


parser = argparse.ArgumentParser(description="")
parser.add_argument('--model_id_or_path', default='')
parser.add_argument('--search_decision_type', default='hard', choices=['hard', 'soft', 'human']) # hard: {true, false}; soft: a float in 0~5
parser.add_argument('--language', default='cn', choices=['cn', 'en'])
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if args.language == 'en':
    if args.search_decision_type == 'hard':
        USER_TEMPLATE = USER_TEMPLATE_EN_DS_INFERENCE
    else:
        USER_TEMPLATE = USER_TEMPLATE_EN_SCORE_DS_INFERENCE
else:
    if args.search_decision_type == 'hard':
        USER_TEMPLATE = USER_TEMPLATE_CN
    else:
        USER_TEMPLATE = USER_TEMPLATE_CN_SCORE

print("Input TEMPLATE")
print(USER_TEMPLATE)

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch


model_type = ModelType.deepseek_vl_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

if args.search_decision_type == 'hard':
    args.model_id_or_path = 'VLLM-KnowledgeBoundary/ckpt/ds_hard_KB_model'
elif args.search_decision_type == 'soft':
    args.model_id_or_path = 'VLLM-KnowledgeBoundary/ckpt/ds_hard_KB_model'

print("model path: ", args.model_id_or_path)


# breakpoint()
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, model_id_or_path = args.model_id_or_path)

# print(model.generation_config.repetition_penalty)
model.config.seq_length = 8192
model.generation_config.max_new_tokens = 4096
template = get_template(template_type, tokenizer)
print(f'template: {template}')
seed_everything(42)

def load_jsonl(path_):
    import json
    ret = []
    with open(path_) as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f'Loading {path_}...'):
            data = json.loads(line)
            ret.append(data)
    return ret

def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))



data_paths = [ \
    ...
]

for data_path in data_paths:
    _date = datetime.now().strftime("%m%d")
    if data_path.endswith('jsonl'):
        data_path_search_decision = data_path.replace('.jsonl', f'_{_date}_{args.search_decision_type}_ds_KB_model.jsonl')
    elif data_path.endswith('json'):
        data_path_search_decision = data_path.replace('.json', f'_{_date}_{args.search_decision_type}_ds_KB_model.json')
    else:
        print("Not spicifying data path to write")
        exit()

    # print("Loading ", data_path)
    print("Writing to ", data_path_search_decision)

    data_with_search_decision = []

    eval_data = load_jsonl(data_path)

    total_cnt = len(eval_data)
    err_format_cnt = 0
    verbose = 1

    g = open(data_path_search_decision, 'w')
    for data in tqdm(eval_data, desc='inference...'):
        question = data['question']
        try:
            img_url = data['image_path']
        except KeyError:
            if 'origin_image' in data:
                img_url = data['origin_image']
            elif 'image_url' in data:
                img_url = data['image_url']
            elif 'image' in data:
                img_url = data['image']
            else:
                raise KeyError
                print(data, "image_path key not found!")


        if not has_chinese(question):
            if args.search_decision_type == 'hard':
                USER_TEMPLATE = USER_TEMPLATE_EN_DS_INFERENCE
            else:
                USER_TEMPLATE = USER_TEMPLATE_EN_SCORE_DS_INFERENCE
        else:
            if args.search_decision_type == 'hard':
                USER_TEMPLATE = USER_TEMPLATE_CN_DS_INFERENCE
            else:
                USER_TEMPLATE = USER_TEMPLATE_CN_SCORE_DS_INFERENCE


        input_ = USER_TEMPLATE.format(question, img_url)

        response, history = inference(model, template, input_)
        new_data = data.copy()

        if args.search_decision_type == 'hard':
            _post_process = response.strip().lower()
            if _post_process == 'true':
                new_data['search_decision'] = 'true'
            elif _post_process == 'false':
                new_data['search_decision'] = 'false'
            else:
                # default to false
                new_data['search_decision'] = 'false'
                err_format_cnt += 1
        
        else:
            _post_process = response.strip()
            try: 
                _post_process = float(_post_process)
                new_data['search_decision'] = _post_process
            except Exception as e:
                err_format_cnt += 1
                print(_post_process, e)
                new_data['search_decision'] = 1.0 # Default to not to search (1.0)

        data_with_search_decision.append(new_data)

        g.writelines(json.dumps(new_data, ensure_ascii=False)+'\n')

    print('\nError format: {0}'.format(err_format_cnt))

    g.close()
    
    print(f"Write to {data_path_search_decision} done.")
    print("#"*100)  
