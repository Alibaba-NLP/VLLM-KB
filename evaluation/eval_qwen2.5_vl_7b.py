import argparse
import copy
import ujson as json
import os
from tqdm import tqdm
import torch

import PIL.Image
from io import BytesIO
import requests

from utils import _filter_by_url, _filter_by_text, KNOWLEDGE_PREFIX, logging

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input file')
parser.add_argument('--model', default="qwen25-vl-7b")
parser.add_argument('--rag', choices=['norag', 'text2text', 'image2image', 'both'])
parser.add_argument('--rewrite', default=0, type=int, help='Rewrite the result file')
args = parser.parse_args()

output_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}.jsonl")

if args.rewrite:
    print('Rewrite', output_file)
    try:
        os.remove(output_file)
    except:
        pass


model_path = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained(
    model_path, 
    max_pixels=640*28*28, 
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages_template = [
    {
        "role": "user",
        "content": [
            # {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            # {"type": "text", "text": "Describe this image."},
        ],
    }
]

@torch.no_grad()
def call_qwen25_vl(messages):
    # print("Calling qwen2.5 vl ...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        # print("Qwen2.5 vl returned")
        return output_text[0]
    
    except Exception as e:
        logging.info(str(e))
        return 





def _make_norag_input(data):
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'type': 'image', 'image': data['image_url']}
    )
    ret[0]['content'].append(
        {'type': 'text', 'text': data['question']}
    )
    return ret


def _make_text2text_input(data):
    search_data = data.get('search_data', []) or data.get('search_data_text2text', [])
    search_data = _filter_by_text(search_data)

    print("# Knowledge used:", len(search_data))
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'type': 'text', 'text': 'Knowledge:'}
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append({'type': 'text', 'text': knowledge['text']})

    ret[0]['content'] += [
        {'type': 'text', 'text': KNOWLEDGE_PREFIX},
        {'type': 'text', 'text': data['question']},
        {'type': 'image', 'image': data.get('image_url', False) or data.get('image', False)},
    ]
    # print("Input made.")
    return ret

def _make_image2image_input(data):
    # print("Making input...")
    search_data = _filter_by_url(data.get('search_data', []) or data.get('search_data_image2image', []))
    search_data = _filter_by_text(search_data)

    print("# Knowledge used:", len(search_data))
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'type': 'text', 'text': 'Knowledge:'}
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append({'type': 'text', 'text': knowledge['text']})
        ret[0]['content'].append({'type': 'image', 'image': knowledge['image']})
        if i == 4:
            break

    ret[0]['content'] += [
        {'type': 'text', 'text': KNOWLEDGE_PREFIX},
        {'type': 'text', 'text': data['question']},
        {'type': 'image', 'image': data.get('image_url', False) or data.get('image', False)},
    ]
    # print("Input made.")
    return ret

def _make_both_input(data):
    search_data_image2image = _filter_by_url(data['search_data_image2image'])
    search_data_image2image = _filter_by_text(search_data_image2image)

    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'type': 'text', 'text': 'Knowledge:'}
    )
    if len(search_data_image2image) > 0:
        ret[0]['content'].append({'type': 'text', 'text': search_data_image2image[0]['text']})
        ret[0]['content'].append({'type': 'image', 'image': search_data_image2image[0]['image']})
    
    for knowledge in _filter_by_text(data['search_data_text2text']):
        ret[0]['content'].append({'type': 'text', 'text': knowledge['text']})
    
    ret[0]['content'] += [
        {'type': 'text', 'text': KNOWLEDGE_PREFIX},
        {'type': 'text', 'text': data['question']},
        {'type': 'image', 'image': data.get('image_url', False) or data.get('image', False)},
    ]
    return ret

print('Input file:', args.i)
with open(args.i) as f:
    lines = f.readlines()

try:
    with open(output_file) as f:
        num_lines_finished = len(f.readlines())
except:
    num_lines_finished = 0

print("Output file:", output_file)
print("Skip", num_lines_finished, 'lines')

g = open(output_file, 'a', buffering=1)

for i, line in enumerate(tqdm(lines, ncols=100, desc=os.path.basename(output_file))):

    if i < num_lines_finished:
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

    # breakpoint()
    res = call_qwen25_vl(messages)
    data['response'] = res

    g.write(
        json.dumps(data, ensure_ascii=False)+'\n'
    )

g.close()
print('done', output_file)
print('='*80)