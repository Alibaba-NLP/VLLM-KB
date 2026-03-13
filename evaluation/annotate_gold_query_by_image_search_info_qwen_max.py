import argparse
import dashscope
from icecream import ic
import copy
import json
import jsonlines
import os
from tqdm import tqdm
import traceback
from rich import print as rprint
import PIL.Image
from io import BytesIO
import requests
from utils import _filter_by_url


import load_dotenv
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')


MAX_ATTEMPT = 5


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

def _make_input(image_url, text):
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        {'image': image_url}
    )
    ret[0]['content'].append(
        {'text': text}
    )
    return ret

def _append_text(text, msg=False):
    ret = copy.deepcopy(msg if (msg is not False) else messages_template)
    ret[0]['content'].append(
        {'text': text}
    )
    return ret

def _append_image(image, msg=False):
    ret = copy.deepcopy(msg if (msg is not False) else messages_template)
    ret[0]['content'].append(
        {'image': image}
    )
    return ret


def call_qwen_vl_max(messages):
    answer = None
    # while answer is None:
    for _ in range(MAX_ATTEMPT):

        try:
            headers = {"X-DashScope-DataInspection": "disable"}
            resp = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages, vl_image_first=False, top_k=1, headers=headers)
            code = resp['status_code']
            # breakpoint()
            if code == 400:
                # print('inappropriate content!')
                print(messages)
                print(resp)
                # breakpoint()
                # return (False, "", "", 0)
            elif code != 200:
                print(resp.message, f"code!=200, code={code}")
                continue
            else:
                total_tokens = resp["usage"]["input_tokens"]+resp["usage"]["output_tokens"]
                message = resp['output']['choices'][0]["message"]
            
                answer = ""
                for item in message['content']:
                    if 'text' in item:
                        answer += item['text']
                return (True, message, answer, total_tokens)

        except Exception as e:
            print('call_qwen_vl_max func:', e)
            continue
        print(f'Retry: {_}/{MAX_ATTEMPT}')
    
    print(f'Retries reach MAX. Return ""')
    return (False, "", "", 0)

    
def get_gold_query(question, image_url, search_data_image2image, lang):
    # 构建第一个提示词，生成gold_query

    if lang == 'en':
        prompt = f'''\
Given the following rules, return a dictionary. 
1. Based on the image search results, the original question, and the image, rewrite the original question into a clearer query known as the 'gold_query'
2. If the image search results are empty, please ignore this part. The search results for images may not be accurate. You can refer to them selectively.
3. The rewritten 'gold_query' should not contain demonstrative pronouns like "this" or "that," and should accurately include entities from the image whenever possible.

Output format:
{{"gold_query": "rewritten gold_query"}}

Example:
Image Search Result: (Photos of Zhao Liying from the web)
Image Title: Actress - Zhao Liying

Original Question: What are the works of this actress?
Original Image: (A photo of Zhao Liying)

You should output: {{"gold_query": "What are the works of Zhao Liying?"}}

'''
        prompt_q = 'Original Question: '
        prompt_o = '\n\nYour output: '

    elif lang == 'zh':
        prompt = f'''\
根据以下规则，返回一个字典。

1. 根据图搜结果、原始问题以及图片，将原始问题改写为一个更清晰的查询，即“gold_query”。
2. 图搜结果可能为空，若为空，请忽视这一部分。图搜结果不一定准确，你可以选择性参考。
3. 经过改写后的“gold_query”不应包含诸如“这个”、“那个”等指示性代词，并应尽量精确地包含图片中的实体信息。

输出格式：
{{"gold_query":"补全后的完整问题"}}

示例：
图搜图结果：（赵丽颖的照片）
相应图片标题：演员-赵丽颖

原始问题：这个演员的作品是什么？
原始图片：(一张赵丽颖的照片)

你应该输出: {{"gold_query":"赵丽颖的作品是什么？"}}

'''
        prompt_q = '原始问题：'
        prompt_o = '\n\n你的输出：'
    
    msg = _append_text(text=prompt)
    search_data_image2image = _filter_by_url(search_data_image2image, max_item=3)

    for search_data in search_data_image2image:
        msg = _append_image(search_data['image'], msg)
        msg = _append_text(search_data['text'], msg)

    msg = _append_image(image_url, msg)
    msg = _append_text(prompt_q + question + prompt_o, msg)

    (success, _, response_text, total_tokens) = call_qwen_vl_max(msg)
    print(response_text)

    if success:
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            gold_query = json.loads(response_text)['gold_query']

        except Exception as e:
            print('Post-process Error', e)
            print('Response:', response_text)
            gold_query = question
    else:
        gold_query = question

    return gold_query


def process_data(input_file, output_file, lang, rewrite):
    if rewrite:
        lines_written = 0
        mode = 'w'
    else:
        try:
            with open(output_file, "r") as f:
                lines_written = len(f.readlines())
        except:
            lines_written = 0
        mode = 'a'
    print('lines_written:', lines_written)

    with open(output_file, mode) as writer, jsonlines.open(input_file, 'r') as reader:
        for line_number, obj in enumerate(reader, start=1):
            if line_number <= lines_written:
                print('Skip', line_number)
                continue
            try:
                question = obj.get('question', '')
                image_url = obj.get('image_url', '') or obj.get('image', '')
                answer = obj.get('answer', '')

                search_data_image2image = obj.get('search_data_image2image', []) or obj.get('search_data', [])

                # 第一次API调用，生成gold_query，并提取image_entity
                gold_query = get_gold_query(question, image_url, search_data_image2image, lang)
                obj['gold_query'] = gold_query

                # 第二次API调用，生成image_query
                # image_query, image_entity = get_image_query(question, gold_query, image_url, model, processor, lang)
                # obj['image_query'] = image_query
                # obj['image_entity'] = image_entity

                # 第三次API调用，修改gold_query，确保包含image_entity
                # modified_gold_query = get_modified_gold_query(question, gold_query, image_entity, image_url, model, processor, lang)
                # obj['gold_query'] = modified_gold_query

                # 写入当前数据到output_file
                if 'search_data_image2image' in obj:
                    del obj['search_data_image2image']
                elif 'search_data' in obj:
                    del obj['search_data']

                writer.write(json.dumps(obj, ensure_ascii=False)+'\n')
                writer.flush()

            except Exception as e:
                print(f"Error processing line {line_number}: {e}")
                traceback.print_exc()
    
    print(f"数据处理完成，已成功保存到输出文件{output_file}。")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="控制数据处理的参数")
    parser.add_argument("--input_data", type=str, default="", help="输入数据的路径")
    parser.add_argument("--output_data", type=str, default=None, help="")
    parser.add_argument("--lang", type=str, default="", help="")
    parser.add_argument("--rewrite", type=int, default=0, help="")
    args = parser.parse_args()

    if not args.output_data:
        args.output_data = args.input_data.replace('.jsonl', '_w_gold_query_from_i2i.jsonl')

    process_data(args.input_data, args.output_data, args.lang, args.rewrite)
