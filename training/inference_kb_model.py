from training_data.prompt_template import PREFIX_MODEL, PREFIX_RAG, IN, IMAGE_OUT, QUERY_OUT, BOTH_OUT
import os
import argparse
import copy
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from datetime import datetime
import json
from modelscope import snapshot_download
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="")
parser.add_argument('--model_id_or_path', default=None)
parser.add_argument('-s', '--search_decision_file_suffix', default='')

args = parser.parse_args()

prompt_based = False
if args.model_id_or_path is None:
    prompt_based = True
    args.model_id_or_path = snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/mnt/nas-alinlp/zhuochen.zc/models')
logger.info(f'Loading: {args.model_id_or_path}')


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_id_or_path, torch_dtype="auto", device_map="auto"
)
logger.info('Model loaded')


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained(
    args.model_id_or_path, 
    max_pixels=1003520, 
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
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
        logger.info(str(e))
        return ''

def make_msg(query, image):
    msg = copy.deepcopy(messages)
    text = PREFIX_MODEL.format(query=query)
    msg[0]['content'].append(
        {"type": "image", "image": image}
    )

    msg[0]['content'].append(
        {"type": "text", "text": text}
    )
    return msg


data_paths = [
    # Reported datasets
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/lifevqa/norag/lifevqa.jsonl',
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/qwenvqa/norag/qwenvqa.jsonl',
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_ch/norag/vfreshqa_ch.jsonl',
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_en/norag/vfreshqa_en.jsonl',
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/visual7w/norag/visual7w.jsonl',
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/nocaps/norag/nocaps_v2.jsonl',
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/nocaps/norag/nocaps.jsonl',

    # Just play with it
    # '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/gaia_hle/norag/gaia_hle.jsonl',

    # For rebuttal
    '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/rebuttal_exp/qwenvqa.jsonl'

]


_date = datetime.now().strftime("%m%d")
_date = _date+'_'+args.search_decision_file_suffix if args.search_decision_file_suffix != '' else _date

for data_path in data_paths:

    if prompt_based:
        data_path_search_decision = data_path.replace('.jsonl', f'_cz_prompt_based.jsonl')
    elif data_path.endswith('jsonl'):
        data_path_search_decision = data_path.replace('.jsonl', f'_cz_{_date}_KB2.jsonl')
    elif data_path.endswith('json'):
        data_path_search_decision = data_path.replace('.json', f'_cz_{_date}_KB2.json')
    else:
        print("Not spicifying data path to write")
        exit()
    
    if os.path.exists(data_path_search_decision):
        with open(data_path_search_decision) as f, open(data_path) as f2:
            if len(f.readlines()) == len(f2.readlines()):
                print('Skip', data_path_search_decision)
                continue

    logger.info(f'Start processing: {data_path}')
    with open(data_path) as f, open(data_path_search_decision, 'w', buffering=1) as g:
        lines = f.readlines()
        for line in tqdm(lines, ncols=50):
            data = json.loads(line)
            query = data['question']
            image = data.get('image_url', False) or data.get('image_path', False) or data.get('image', False)

            msg = make_msg(query, image)
            response = call_qwen25_vl(msg)

            data['search_decison'] = response

            g.write(
                json.dumps(data, ensure_ascii=False)+'\n'
            )
    logger.info(f'Done processing {data_path}')
    
    # print(data_path_search_decision, 'done')