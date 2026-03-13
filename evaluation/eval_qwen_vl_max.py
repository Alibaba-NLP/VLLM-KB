import argparse
import dashscope
from icecream import ic
import copy
import ujson as json
import os
from tqdm import tqdm

import PIL.Image
from io import BytesIO
import requests

from utils import _filter_by_url, _filter_by_text, KNOWLEDGE_PREFIX, logging


import load_dotenv
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input file')
parser.add_argument('--model', default="qwen-vl-max")
parser.add_argument('--rag', choices=['norag', 'text2text', 'image2image', 'both'])
parser.add_argument('--overwrite', default=0)
args = parser.parse_args()

output_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}.jsonl")


messages_template = [
    {
        "role": "user", 
        "content": [
            # {"text": "What is in the two pic? Describe the first and second pic respectively"},
            # {"image": "https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en/Freshqa_en_extracted_images/image_1.jpeg"}, 
            # ...
        ], 
        "X-DashScope-DataInspection": "disable"
    }
]


def call_qwen_vl_max(messages, max_retry=10):
    answer = None
    # while answer is None:
    for _ in range(max_retry):
        try:
            headers = {"X-DashScope-DataInspection": "disable"}
            
            resp = dashscope.MultiModalConversation.call(model=args.model, messages=messages, vl_image_first=False, top_k=1, headers=headers)
            code = resp['status_code']
            # breakpoint()
            if code != 200:
                # ic('inappropriate content!')
                # ic(messages)
                ic(resp)
                print(f"Retry: {_}")
                # breakpoint()
                # return (False, "", "", 0)
                continue

            elif code == 200:
                total_tokens = resp["usage"]["input_tokens"]+resp["usage"]["output_tokens"]
                message = resp['output']['choices'][0]["message"]
            
                answer = ""
                for item in message['content']:
                    if 'text' in item:
                        answer += item['text']

            return (True, message, answer, total_tokens)
        except Exception as e:
            print(e)
            continue
    return (False, "", "", 0)

# success, _, res, total_tokens = call_qwen_vl_max(messages)
# print(res)



def _make_norag_input(data):
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'image': data['image_url']}
    )
    ret[0]['content'].append(
        {'text': data['question']}
    )
    return ret


def _make_text2text_input(data):
    search_data = data.get('search_data', []) or data.get('search_data_text2text', [])
    search_data = _filter_by_text(search_data)
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'text': 'Knowledge:'}
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append({'text': knowledge['text']})

    ret[0]['content'] += [
        {'text': KNOWLEDGE_PREFIX},
        {'text': data['question']},
        {'image': data.get('image_url', False) or data.get('image', False)},
    ]
    return ret

def _make_image2image_input(data):
    search_data = _filter_by_url(data.get('search_data', []) or data.get('search_data_image2image', []))
    search_data = _filter_by_text(search_data)

    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'text': 'Knowledge:'}
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append({'text': knowledge['text']})
        ret[0]['content'].append({'image': knowledge['image']})
        if i == 4:
            break

    ret[0]['content'] += [
        {'text': KNOWLEDGE_PREFIX},
        {'text': data['question']},
        {'image': data.get('image_url', False) or data.get('image', False)},
    ]
    return ret

def _make_both_input(data):
    search_data_image2image = _filter_by_url(data['search_data_image2image'])
    search_data_image2image = _filter_by_text(search_data_image2image)

    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'text': 'Knowledge:'}
    )
    if len(search_data_image2image) > 0:
        ret[0]['content'].append({'text': search_data_image2image[0]['text']})
        ret[0]['content'].append({'image': search_data_image2image[0]['image']})
    
    for knowledge in _filter_by_text(data['search_data_text2text']):
        ret[0]['content'].append({'text': knowledge['text']})
    
    ret[0]['content'] += [
        {'text': KNOWLEDGE_PREFIX},
        {'text': data['question']},
        {'image': data.get('image_url', False) or data.get('image', False)},
    ]
    return ret


with open(args.i) as f:
    lines = f.readlines()


if args.overwrite:
    num_lines = 0
    mode_ = 'w'
else:
    mode_ = 'a'
    try:
        with open(output_file, 'r') as g:
            num_lines = len(g.readlines())
        print(output_file, num_lines, 'lines written.')
    except:
        num_lines = 0


with open(output_file, mode_, buffering=1) as g:
    for i, line in enumerate(tqdm(lines, ncols=100, desc=os.path.basename(output_file))):
        if i<num_lines:
            continue

        data = json.loads(line)

        if args.rag == 'norag':
            messages = _make_norag_input(data)
        elif args.rag == 'text2text':
            messages = _make_text2text_input(data)
        elif args.rag == 'image2image':
            messages = _make_image2image_input(data)
        elif args.rag == 'both':
            messages = _make_both_input(data)

        success, _, res, total_tokens = call_qwen_vl_max(messages)
        data['response'] = res

        g.write(
            json.dumps(data, ensure_ascii=False)+'\n'
        )