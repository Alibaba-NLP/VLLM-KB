import argparse
import dashscope
from icecream import ic
import copy
import json
import jsonlines
import os
from tqdm import tqdm
import traceback

import PIL.Image
from io import BytesIO
import requests
from rich import print as rprint


import load_dotenv
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
MAX_ATTEMPT = 10


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

def call_qwen_vl_max(messages):
    answer = None
    for _ in range(MAX_ATTEMPT):

        try:
            headers = {"X-DashScope-DataInspection": "disable"}
            
            resp = dashscope.MultiModalConversation.call(model='qwen-vl-max', messages=messages, vl_image_first=False, top_k=1, headers=headers)
            code = resp['status_code']
            # breakpoint()
            if code == 400 and resp['code'] == 'DataInspectionFailed':
                print('Input data may contain inappropriate content.')
                rprint(messages)
                rprint(resp)
                break

            elif code != 200:
                print(resp.message, code)
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
            print(e)
            continue
        print(f'Retry: {_}/{MAX_ATTEMPT}')
    
    print(f'Retries reach MAX. Return ""')
    return (False, "", "", 0)

def get_gold_query(question, image_url, answer, lang):
    # 构建第一个提示词，生成gold_query
    if lang == 'en':
        prompt = f'''\
Task: Based on the following rules, extract keywords and return a dictionary:

**Rules**:
1. Use the information from the image and question to complete the question, forming a clear and full question known as "gold_query".
2. The parts of the "question" that typically need completion often contain demonstratives such as "this", "who", "it", "that".
3. If the part of the "question" that needs completion lacks demonstratives, identify the main subject needing completion from the image, and incorporate it into the "question".
4. Other than the completion part, the rest of the "gold_query" should strictly match the "question".
5. The "gold_query" should include necessary information from the image, allowing the VQA to be answered without viewing the image.

Output Format: 
{{"gold_query": "The complete question after completion"}}

Example 1:
Input: - question: "What are the works of this actor?" - image: (A photo of Zhao Liying)
You should output: {{"gold_query": "What are the works of Zhao Liying?"}}

Example 2:
Input: - question: Who is the sole student author presenting this type of neural network architecture? - image: (A diagram of LSTM)
You should output: {{"gold_query": "Who is the sole student author presenting the LSTM neural network architecture?"}}

Example 3:
Input: - question: When was it released? - image: (A photo of Tesla Model Z)
You should output: {{"gold_query": "When was the Tesla Model Z released?"}}

Example 4:
Input: - question: When did OpenAI release? - image: (A logo of GPT-4o)
You should output: {{"gold_query": "When did OpenAI release GPT-4o?"}}

Input:
- question: {question}

Output: 
'''

    elif lang == 'zh':
        prompt = f'''\
任务：按照下列规则，返回一个字典。

**规则**: 
1. 根据提供的"question"和"image"，利用question和image中的信息补全问题，改写出一个完整的、明确的问题，称为"gold_query"。
2. "question"的需补全部分大多数有指示词"这"，"谁"，"它"，"该..."。
3. 当"question"的需补全部分没有指示词时，需结合图片判断需补全的主体，并加入"question"。
3. 除了补全问题的部分，其他部分"gold_query"应严格与"question"相同。
4. "gold_query"应包含必要的图片中信息，使得回答VQA时不用看图片也能回答出问题。
5. 补全后的"gold_query"不应出现"这"，"谁"等指代词，补全后的"gold_query"不能与"question"完全一样。

输出格式: 
{{"gold_query": "补全后的完整问题"}}

示例1: 
输入: - question: "这个演员的作品是什么？"- image: (一张赵丽颖的照片)
你应该输出: {{"gold_query":"赵丽颖的作品是什么？"}}

示例2: 
输入: - question: 谁是介绍这种神经网络架构论文的唯一学生作者？- image: (一张LSTM的结构图)
你应该输出: {{"gold_query":"谁是介绍 LSTM 神经网络架构论文的唯一学生作者？"}}

示例3: 
输入: - question: 它什么时候推出？- image: (一张特斯拉 Model Z的汽车图片)
你应该输出: {{"gold_query":"特斯拉 Model Z 什么时候推出？"}}

示例4: 
输入: - question: OpenAI何时发布？- image: (一张GPT4o的logo图片)
你应该输出: {{"gold_query":"OpenAI何时发布GPT4o？"}}

输入: - question: {question}

你的输出: 
'''

    (success, _, response_text, total_tokens) = call_qwen_vl_max(_make_input(image_url, prompt))
    print(response_text)

    if success:
        try:
            gold_query = json.loads(response_text.strip())['gold_query']
        except Exception as e:
            print('Post-process Error', e)
            print('Response:', response_text)
            gold_query = question
    else:
        gold_query = question
    
    return gold_query


def process_data(input_file, output_file, lang):
    line_written = 0
    try:
        with open(output_file, 'r') as f:
            line_written =  sum(1 for _ in f)
    except FileNotFoundError:
        line_written = 0
    print(f"{line_written} lines in {output_file}")
        
    with open(output_file, 'a') as writer, jsonlines.open(input_file, 'r') as reader:
        for line_number, obj in enumerate(reader, start=1):

            if line_number<=line_written:
                print('Skip', line_number)
                continue
            
            try:
                question = obj.get('question', '')
                image_url = obj.get('image_url', '') or obj.get('image', '')
                answer = obj.get('answer', '')

                # 第一次API调用，生成gold_query，并提取image_entity
                gold_query = get_gold_query(question, image_url, answer, lang)
                obj['gold_query'] = gold_query

                # 第二次API调用，生成image_query
                # image_query, image_entity = get_image_query(question, gold_query, image_url, model, processor, lang)
                # obj['image_query'] = image_query
                # obj['image_entity'] = image_entity

                # 第三次API调用，修改gold_query，确保包含image_entity
                # modified_gold_query = get_modified_gold_query(question, gold_query, image_entity, image_url, model, processor, lang)
                # obj['gold_query'] = modified_gold_query

                # 写入当前数据到output_file
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
    args = parser.parse_args()

    if not args.output_data:
        args.output_data = args.input_data.replace('.jsonl', '_w_gold_query_from_origin_query_image.jsonl')

    process_data(args.input_data, args.output_data, args.lang)
