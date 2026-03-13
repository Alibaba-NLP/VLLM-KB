import argparse
import copy
import ujson as json
import os
import glob
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from modelscope import snapshot_download

import PIL.Image
from io import BytesIO
import requests

from utils import _filter_by_url, _filter_by_text, KNOWLEDGE_PREFIX, logging


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input file')
parser.add_argument('--model', default="deepseek-ai/deepseek-vl2")
parser.add_argument('--rag', choices=['norag', 'text2text', 'image2image', 'both'])
args = parser.parse_args()

print("Input file:", args.i)
output_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}.jsonl")

if os.path.exists(output_file):
    g = open(output_file, 'r')
    num_lines = len(g.readlines())
    print(output_file, num_lines, 'lines')
    if num_lines in [737, 715]:
        print(output_file, 'skip.', num_lines, '\n')
        g.close()
        exit()
    g.close()


# model_path = snapshot_download(args.model, local_dir='/mnt/nas-alinlp/zhuochen.zc/models/')
model_path = '/mnt/nas-alinlp/zhuochen.zc/models/deepseek-ai/deepseek-vl2'

vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
IMAGE_TOKEN = '<image>'
conversation_template = [
    {
        "role": "<|User|>",
        "content": "",
        "images": [],
    },
    {"role": "<|Assistant|>", "content": ""}
]

tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


@torch.no_grad()
def inference(text: str, image: list):
    conversation = copy.deepcopy(conversation_template)
    conversation[0]['content'] = text
    conversation[0]['images'] = image

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # greedy
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        use_cache=False
    )

    # sample
    # outputs = vl_gpt.language.generate(
    #     inputs_embeds=inputs_embeds,
    #     attention_mask=prepare_inputs.attention_mask,
    #     pad_token_id=tokenizer.eos_token_id,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=128,
    #     do_sample=True,        # 启用采样
    #     temperature=0.7,       # 降低温度以增加确定性
    #     top_k=50,              # 限制候选词范围
    #     repetition_penalty=1.2, # 惩罚重复 token
    #     use_cache=True
    # )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    # print(f"{prepare_inputs['sft_format'][0]}", answer)
    answer = answer.replace('<｜end▁of▁sentence｜|>', '')
    print(answer)
    return answer


def _make_text2text_input(data):
    image = data['image_url']
    q = data['question']

    knowledge = data.get('search_data', False) or data.get('search_data_text2text', False)
    knowledge = _filter_by_text(knowledge)
    k = '\n'.join([i['text'] for i in knowledge])

    text = f'''\
{KNOWLEDGE_PREFIX}
{k}

{IMAGE_TOKEN}
Question: {q}
'''
    return text, [image]


def _make_image2image_input(data):
    image = []
    q = data['question']
    search_data = _filter_by_url(data.get('search_data', False) or data.get('search_data_image2image', False))
    search_data = _filter_by_text(search_data)
    k = ''

    for i, knowledge in enumerate(search_data):
        k += knowledge['text']
        k += f'\n{IMAGE_TOKEN}\n\n'
        image.append(knowledge['image'])
        
        if i == 4:
            break
    k = k.strip()
    text = f'''\
{KNOWLEDGE_PREFIX}
{k}

{IMAGE_TOKEN}
Question: {q}
'''

    image.append(data.get('image', False) or data.get('image_path', False) or data.get('image_url', False))
    return text, image



def _make_both_input(data):
    image = []
    q = data['question']
    k = ''

    search_data_image2image = _filter_by_url(data['search_data_image2image'])
    if len(search_data_image2image) > 0:
        k += search_data_image2image[0]['text']
        k += f'\n{IMAGE_TOKEN}\n\n'
        image.append(search_data_image2image[0]['image'])

    for knowledge in _filter_by_text(data['search_data_text2text']):
        k += f"{knowledge['text']}\n"

    k = k.strip()

    text = f'''\
{KNOWLEDGE_PREFI}
{k}

{IMAGE_TOKEN}
Question: {q}
'''
    print(text)

    image.append(data.get('image', False) or data.get('image_path', False) or data.get('image_url', False))
    return text, image


with open(args.i) as f:
    lines = f.readlines()


print("(Continue) Wrting", output_file)

try:
    with open(output_file, 'r') as _f:
        lines_written = len(_f.readlines())
except:
    lines_written = 0

g = open(output_file, 'a', buffering=1)

for i, line in enumerate(tqdm(lines, ncols=100)):

    if i < lines_written:
        print(f"Skip {i+1}-th line")
        continue

    data = json.loads(line)

    if args.rag == 'norag':
        text = f"{IMAGE_TOKEN}\n{data['question']}"
        image = [data['image_url']]
    elif args.rag == 'text2text':
        text, image = _make_text2text_input(data)
    elif args.rag == 'image2image':
        text, image = _make_image2image_input(data)
    elif args.rag == 'both':
        text, image = _make_both_input(data)

    # breakpoint()
    res = inference(text, image)
    data['response'] = res

    g.write(
        json.dumps(data, ensure_ascii=False)+'\n'
    )

print('done', output_file)
g.close()