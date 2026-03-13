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

from utils import _filter_by_url, _filter_by_text, KNOWLEDGE_PREFIX
import logging
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input file')
parser.add_argument('--model', default="deepseek-ai/deepseek-vl-7b-chat")
parser.add_argument('--rag', choices=['norag', 'text2text', 'image2image', 'both'])
parser.add_argument('--overwrite', default=0, type=int)
args = parser.parse_args()

print("Input file:", args.i)
output_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}-2.jsonl")
print("Output file:", output_file)

with open(args.i) as f:
    lines = f.readlines()

if os.path.exists(output_file) and args.overwrite:
    os.remove(output_file)
    num_lines_finished = 0

try:
    with open(output_file, 'r') as g:
        num_lines_finished = len(g.readlines())
        print("Num finished lines:", num_lines_finished)
except:
    num_lines_finished = 0

if len(lines) == num_lines_finished:
    print("Skip!", output_file)
    exit()


# specify the path to the model
# model_path = snapshot_download(args.model)
model_path = '/mnt/nas-alinlp/zhuochen.zc/models/deepseek-ai/deepseek-vl-7b-chat'

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
IMAGE_TOKEN = '<image_placeholder>'
conversation_template = [
    {
        "role": "User",
        "content": "",
        "images": [],
    },
    {"role": "Assistant", "content": ""}
]


tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


def inference(text: str, image: list):
    conversation = copy.deepcopy(conversation_template)
    conversation[0]['content'] = text
    conversation[0]['images'] = image

    try:
        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the vl_gpt to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        if not answer:
            logger.warning(f"Answer is empty! outputs: {outputs}")
            logger.warning(tokenizer.decode(outputs[0].cpu().tolist()))
        return answer
    
    except Exception as e:
        logger.info(str(e))
        return ''


def _make_text2text_input(data):
    image = data.get('image_url', False) or data.get('image', False)
    q = data['question']

    knowledge = data.get('search_data', []) or data.get('search_data_text2text', [])
    knowledge = _filter_by_text(knowledge)
    k = '\n'.join([i['text'] for i in knowledge])

    text = f'''\
{KNOWLEDGE_PREFIX}
{k}

Question: {q}
{IMAGE_TOKEN}
'''
    return text, [image]


def _make_image2image_input(data):
    image = []
    q = data['question']
    search_data = _filter_by_url(data.get('search_data', []) or data.get('search_data_image2image', []))
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

Question: {q}
{IMAGE_TOKEN}
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
{KNOWLEDGE_PREFIX}
{k}

Question: {q}
{IMAGE_TOKEN}
'''

    image.append(data.get('image', False) or data.get('image_path', False) or data.get('image_url', False))
    return text, image



g = open(output_file, 'a', buffering=1)

for i, line in enumerate(tqdm(lines, ncols=100, desc=os.path.basename(output_file))):
    if i < num_lines_finished:
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

    res = inference(text, image)
    data['response'] = res

    g.write(
        json.dumps(data, ensure_ascii=False)+'\n'
    )

print('done', output_file)
g.close()