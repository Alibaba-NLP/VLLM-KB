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
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

import PIL.Image
from io import BytesIO
import requests

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input file')
parser.add_argument('--model', default="deepseek-ai/deepseek-vl2")
parser.add_argument('--rag', choices=['norag', 'text2text', 'image2image', 'both'])
args = parser.parse_args()

print("Input file:", args.i)
output_file = args.i.replace('.jsonl', f"_{os.path.basename(args.model)}.jsonl")

try:
    g = open(output_file, 'r')
    num_lines = len(g.readlines())
    print(output_file, num_lines, 'lines')
    if num_lines in [737, 715]:
        print(output_file, 'skip.', num_lines, '\n')
        g.close()
        exit()
    g.close()
except Exception as e:
    print(e)
    pass



# specify the path to the model
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



with init_empty_weights():
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Automatically infer the device map
device_map = infer_auto_device_map(
    vl_gpt,  # The model
    max_memory={i: "80GB" for i in range(torch.cuda.device_count())},  # Max memory per GPU (4 GPUs, each with 80GB)
    no_split_module_classes=["layers"],  # Specify layers that should not be split (if any)
    # dtype="bfloat16",  # Use bfloat16 for memory efficiency
)
# device_map['language.lm_head'] = device_map['language.model.embed_tokens']
print(device_map)

# Load the model checkpoint and distribute it across GPUs
vl_gpt = load_checkpoint_and_dispatch(
    vl_gpt,  # The model
    model_path,  # Path to the model checkpoint
    device_map=device_map,  # The automatically generated device map
    offload_folder="offload",  # Folder for offloading (if needed)
    dtype="bfloat16",  # Use bfloat16 for memory efficiency
)







# vl_gpt = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     trust_remote_code=True, 
#     device_map=device_map,
#     torch_dtype=torch.bfloat16
# )

vl_gpt = vl_gpt.eval()

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

    breakpoint()

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    print(f"{prepare_inputs['sft_format'][0]}", answer)

    return answer


def _make_text2text_input(data):
    image = data['image_url']
    q = data['question']

    knowledge = data['search_data']
    k = '\n'.join([i['text'] for i in knowledge])

    text = f'''\
Refer to the following knowledge to answer the question. \
Knowledge: 
{k}

Question: {q}
{IMAGE_TOKEN}
'''
    return text, [image]

def _filter_by_url(list_):
    ret = []
    for search_data in list_:
        try:
            image_path = search_data['image']
            response = requests.get(image_path, timeout=5)
            response.raise_for_status()
            # img_data = BytesIO(response.content)
            # pil_img = PIL.Image.open(img_data)
            ret.append(search_data)
        except:
            pass

        if len(ret) == 5:
            break
    return ret

def _make_image2image_input(data):
    image = []
    q = data['question']
    search_data = _filter_by_url(data['search_data'])
    k = ''

    for i, knowledge in enumerate(search_data):
        k += knowledge['text']
        k += f'\n{IMAGE_TOKEN}\n\n'
        image.append(knowledge['image'])
        
        if i == 4:
            break
    k = k.strip()
    text = f'''\
Refer to the following knowledge to answer the question. \
Knowledge: 
{k}

Question: {q}
{IMAGE_TOKEN}
'''

    image.append(data['image'])
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

    for i, knowledge in enumerate(data['search_data_text2text']):
        k += f"{knowledge}\n"

        if i >= 4:
            break

    k = k.strip()

    text = f'''\
Refer to the following knowledge to answer the question. \
Knowledge: 
{k}

Question: {q}
{IMAGE_TOKEN}
'''

    image.append(data['image'])
    return text, image


with open(args.i) as f:
    lines = f.readlines()


print("Wrting", output_file)

g = open(output_file, 'w', buffering=1)

for line in tqdm(lines):
    data = json.loads(line)

    if args.rag == 'norag':
        text = data['question']
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