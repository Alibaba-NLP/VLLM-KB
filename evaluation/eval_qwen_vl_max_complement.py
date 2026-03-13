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

from utils import _filter_by_url, _filter_by_text


import load_dotenv
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input file')
parser.add_argument('--model', default="qwen-vl-max")
parser.add_argument('--rag', choices=['norag', 'text2text', 'image2image', 'both'])
args = parser.parse_args()

finished_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}.jsonl")
output_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}_2.jsonl")


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


def call_qwen_vl_max(messages):
    answer = None
    while answer is None:
        try:
            headers = {"X-DashScope-DataInspection": "disable"}
            
            resp = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages, vl_image_first=False, top_k=1, headers=headers)
            code = resp['status_code']
            # breakpoint()
            if code != 200:
                ic('inappropriate content!')
                ic(messages)
                ic(resp)
                # breakpoint()
                return (False, "", "", 0)

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
    search_data = data.get('search_data', False) or data.get('search_data_text2text', False)
    search_data = _filter_by_text(search_data)
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'text': 'Knowledge:'}
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append({'text': knowledge['text']})

    ret[0]['content'] += [
        {'text': 'Refer to the following knowledge to answer the question. Respond using the same language as the question. '},
        {'text': data['question']},
        {'image': data.get('image_url', False) or data.get('image', False)},
    ]
    return ret

def _make_image2image_input(data):
    search_data = _filter_by_url(data.get('search_data', False) or data.get('search_data_image2image', False))
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
        {'text': 'Refer to the following knowledge to answer the question. Respond using the same language as the question. '},
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
        {'text': 'Refer to the following knowledge to answer the question. Respond using the same language as the question. '},
        {'text': data['question']},
        {'image': data.get('image_url', False) or data.get('image', False)},
    ]
    return ret


with open(args.i) as f, open(finished_file) as f2:
    lines = list(f.readlines())
    lines_finished = list(f2.readlines())


g = open(output_file, 'w', buffering=1)

for line, finished_line in tqdm(zip(lines, lines_finished), ncols=100, total=len(lines)):
    data = json.loads(line)
    finished_data = json.loads(finished_line)
    
    if finished_data['response'] != '':
        g.write(finished_line.strip()+'\n')
    else:

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

g.close()