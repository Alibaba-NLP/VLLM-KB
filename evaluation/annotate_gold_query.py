import json
import jsonlines
import requests
from io import BytesIO
import argparse
import os
import time
import torch
from vllm import LLM, SamplingParams
# import dashscope
from pathlib import Path
import traceback


from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#########################################

def get_last_processed_line(temp_path):
    """简单地获取文件的总行数"""
    temp_path = Path(temp_path)
    if not temp_path.exists():
        return 0
    
    try:
        with open(temp_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"Resuming from line {line_count}")
        return line_count
    except Exception as e:
        print(f"Error reading temporary file: {e}")
        return 0

def get_gold_query(question, image_url, answer, model, processor, lang):
    # 构建第一个提示词，生成gold_query
    if lang == 'en':
        prompt = f'''\
**Task**: Based on the following rules, extract keywords and return a dictionary:

**Rules**:
1. Use the information from the "image" and "answer" to complete the "question", forming a clear and full question known as "gold_query".
2. The parts of the "question" that typically need completion often contain demonstratives such as "this", "who", "it", "that".
3. If the part of the "question" that needs completion lacks demonstratives, identify the main subject needing completion from the image, and incorporate it into the "question".
4. Other than the completion part, the rest of the "gold_query" should strictly match the "question".
5. The "gold_query" should include necessary information from the image, allowing the VQA to be answered without viewing the image.
6. After completion, the "gold_query" should not contain any demonstratives like "this", "who", etc., and must not be exactly the same as the "question".

**Output Format:**

{{"gold_query": "The complete question after completion"}}

**Examples:**

Input: - question: "What are the works of this actor?" - image: (A photo of Zhao Liying) -answer: "Zhao Liying's main works include 'The Journey of Flower', 'Story of Minglan', etc."
You should output: {{"gold_query": "What are the works of Zhao Liying?"}}

Input: - question: Who is the sole student author presenting this type of neural network architecture? - image: (A diagram of LSTM) -answer: "Sepp Hochreiter"
You should output: {{"gold_query": "Who is the sole student author presenting the LSTM neural network architecture?"}}

Input: - question: When was it released? - image: (A photo of Tesla Model Z) -answer: "Tesla Model Z is set to release in 2024"
You should output: {{"gold_query": "When was the Tesla Model Z released?"}}

Input: - question: When did OpenAI release? - image: (A logo of GPT-4o) -answer: "OpenAI released GPT-4o in May 2024"
You should output: {{"gold_query": "When did OpenAI release GPT-4o?"}}

**Input:**

- question: {question}
- answer: {answer}

**Output**: 
'''

    elif lang == 'zh':
        prompt = f'''\
你是一个用于问答聊天机器人的改写系统。

**你将获得以下信息**：

- question
- image
- answer

**任务**: 根据以下规则，提取关键词，并返回一个字典: 

**规则**: 
1. 根据提供的"question"和"image"，利用image和answer中的信息补全问题，改写出一个完整的、明确的问题，称为"gold_query"。
2. "question"的需补全部分大多数有指示词"这"，"谁"，"它"，"该"。
3. 当"question"的需补全部分没有指示词时，需结合图片判断需补全的主体，并加入"question"。
3. 除了补全问题的部分，其他部分"gold_query"应严格与"question"相同。
4. "gold_query"应包含必要的图片中信息，使得回答VQA时不用看图片也能回答出问题。
5. 补全后的"gold_query"不应出现"这"，"谁"等指代词，补全后的"gold_query"不能与"question"完全一样。

**输出格式: **

{{"gold_query":"补全后的完整问题"}}

**示例: **

输入: - question: "这个演员的作品是什么？"- image: (一张赵丽颖的照片) -answer: "赵丽颖的主要作品有《花千骨》《知否》等。"
你应该输出: {{"gold_query":"赵丽颖的作品是什么？"}}

输入: - question: 谁是介绍这种神经网络架构论文的唯一学生作者？- image: (一张LSTM的结构图) -answer: "塞普·霍赫赖特"
你应该输出: {{"gold_query":"谁是介绍 LSTM 神经网络架构论文的唯一学生作者？"}}

输入: - question: 它什么时候推出？- image: (一张特斯拉 Model Z的汽车图片) -answer: "特斯拉 Model Z 将于2024年推出"
你应该输出: {{"gold_query":"特斯拉 Model Z 什么时候推出？"}}

输入: - question: OpenAI何时发布？- image: (一张GPT4o的logo图片) -answer: "OpenAI于2024年5月发布GPT4o"
你应该输出: {{"gold_query":"OpenAI何时发布GPT4o？"}}

**输入:**

- question: {question}
- answer: {answer}

**输出**: 
'''

    response_text = None
    while response_text is None:

        # breakpoint()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response_text = response_text[0]

    print(response_text)
    if response_text is None:
        print(f"模型响应为空，返回空值")
        return ''

    try:
        lines = response_text.strip().split('\n')
        gold_query = ''
        for line in lines:
            if 'gold_query' in line:
                gold_query = line.split(': ', 1)[-1].strip().strip('【】""""')
                break
        # 如果未找到，则整个响应视为gold_query
        if not gold_query:
            gold_query = response_text.strip().strip('【】""""')
    except Exception as e:
        print(f"无法提取gold_query: {e}")
        print(f"响应文本: {response_text}")
        return ''

    return gold_query


def get_image_query(question, gold_query, image_url, model, processor, lang):
    if gold_query == '':
        return '',''
    else:
        if lang == 'en':
            prompt = f'''\
**Task**: Based on the following rules, extract keywords and return a dictionary:

**Rules**: 
1. Compare the "question" with the "gold_query" to identify information that is included in the "gold_query" but missing from the "question". Based on this missing information and the image, formulate a question about the content of the image, known as "image_query", and provide an answer called "image_entity".
2. Composition rules for "image_query": If the "question" includes the words "this"/"this"/"that" followed by a noun, form the query as "Who is this?" or "What is this?" If there is no noun following "this", the "image_query" should be "What is this?" If there are no clear demonstratives like "this" or "that", further guidance is needed.

**Input**:

- question: {question}
- gold_query: {gold_query}

**Output Format**:

{{"image_query": "", "image_entity": ""}}


**Examples**:

Input: - question: "What are this actor’s works?" - gold_query: "What are Zhao Liying’s works?" - image: (A photo of Zhao Liying)
You should output: {{"image_query": "Who is this actor?", "image_entity": "Zhao Liying"}}

Input: - question: "When did Epic Gaming first release this?" - gold_query: "When did Epic Gaming first release Minecraft?" - image: (A photo of Minecraft)
You should output: {{"image_query": "What is this?", "image_entity": "Minecraft"}}

Input: - question: "Who is the current CTO of this organization?" - gold_query: "Who is the CTO of Alibaba Cloud?" - image: (A photo of Alibaba Cloud)
You should output: {{"image_query": "What is this organization?", "image_entity": "Alibaba Cloud"}}

Input: - question: "How much bigger is 4?" - gold_query: "How much bigger is 3 than 4?" - image: (A photo of the number 3)
You should output: {{"image_query": "What is this?", "image_entity": "3"}}
'''
        elif lang == 'zh':
            prompt = f'''\
你是一个用于问答聊天机器人的改写系统。

**你将获得以下信息**：

- question
- image
- gold_query

**任务**: 根据以下规则，提取关键词，并返回一个字典: 

**规则**: 
1. 对比"question"和"gold_query"，找出"gold_query"中包含但"question"中缺少的信息。基于这些缺失的信息和图片，生成一个关于图片内容的提问，称为"image_query"，并给出这个问题的回答"image_entity"。
2. "image_query"组成规则: question中的"这"/"这个"/"该"+后面的名词+"是谁？"/"是什么？"，如果"这个"后面没有跟着名词，"image_query"应为"这是什么？"。如果没有明显的指代词"这"/"这个"

**输出格式: **

{{"image_query":"", "image_entity":""}}

**示例: **

输入: - question: 这个演员的作品是什么？- gold_query: 赵丽颖的作品是什么？ - image: (一张赵丽颖的照片)
你应该输出: {{"image_query":"这个演员是谁？", "image_entity":"赵丽颖"}}

输入: - question: Epic Gaming 何时首次发布这个？- gold_query: Epic Gaming 何时首次发布 Minecraft？ - image: (一张Minecraft的图片)
你应该输出: {{"image_query":"这是什么？", "image_entity":"Minecraft"}}

输入: - question: 这个组织现在的cto是谁来着？- gold_query: 阿里云cto是谁来着？ - image: (一张阿里云的图片)
你应该输出: {{"image_query":"这个组织是什么？", "image_entity":"阿里云"}}

输入: - question: 比4大多少？- gold_query: 3比4大多少？ - image: (一张数字3的图片)
你应该输出: {{"image_query":"这是什么？", "image_entity":"3"}}

**输入**:

- question: {question}
- gold_query: {gold_query}

**输出**: 
'''

    response_text = None
    while response_text is None:
       
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response_text = response_text[0]

        print(response_text)

        if response_text is None:
            print(f"模型响应为空，返回空值")
            return '', ''

        try:
            response_dict = json.loads(response_text.strip('```json\n').strip('```').strip())
        except json.JSONDecodeError:
            # breakpoint()
            # raise ValueError(f"模型响应格式不正确: {response_text}")
            print(f"模型响应格式不正确: {response_text}")
            return '', ''

        # 提取关键词和数量
        image_query = response_dict.get('image_query', '')
        image_entity = response_dict.get('image_entity', '')

        
        return image_query, image_entity


def get_modified_gold_query(question, gold_query, image_entity, image_url, model, processor, lang):
    # 构建第三次提示词，用来修改 gold_query
    if lang == 'en':
        prompt = f'''\
**Task**: Based on the following rules, modify the "gold_query" to ensure it is complete and clear, and includes the "image_entity":

**Rules**:
1. Check if "gold_query" contains the "image_entity". If "image_entity" is not present in "gold_query", use the "question", "gold_query", "image_entity", and information from the image to modify the "gold_query".
2. The modified "gold_query" should include contains the "image_entity", allowing the VQA to be answered without viewing the image.

**Input:**
- question: {question}
- gold_query: {gold_query}
- image_entity: {image_entity}

**Output Format:**
{{ "modified_gold_query": "The revised complete question" }}

**Examples:**

Input:
- question: "What are the works of this actor?"
- gold_query: "What are Zhao Liying’s works?"
- image_entity: "Zhao Liying"
Output: {{"modified_gold_query": "What are Zhao Liying’s works?"}}

Input:
- question: "Which book won this award for best novel?"
- gold_query: "Which book won this award for best novel?"
- image_entity: "Nebula Awards"
Output: {{"modified_gold_query": "Which book won the Nebula Awards for best novel?"}}

Input:
- question: "How old is he?"
- gold_query: "How old is he?"
- image_entity: "Donald Trump"
Output: {{"modified_gold_query": "How old is Donald Trump?"}}

Input:
- question: "How many world championships has this team won?"
- gold_query: "How many world championships has this team won?"
- image_entity: "Houston Astros"
Output: {{"modified_gold_query": "How many world championships have the Houston Astros won?"}}
'''
    
    elif lang == 'zh':
        prompt = f'''\
你是一个用于问答聊天机器人的改写系统。

**你将获得以下信息**：

- question
- image
- image_entity

**任务**: 根据以下规则，修改gold_query，确保其完整且明确，且包含image_entity：

**规则**:
1. 检查"gold_query"中是否包含"image_entity"，如果"image_entity"没有出现在"gold_query"中，使用"question"、"gold_query"和"image_entity"以及图片信息修改"gold_query"。
2. 修改后的"gold_query"必须包含正确的"image_entity"。

**输出格式: **
{{ "modified_gold_query": "修改后的完整问题" }}

**示例: **

输入: 
- question: "这个演员的作品是什么？"
- gold_query: "赵丽颖的作品是什么？"
- image_entity: "赵丽颖"
输出: {{"modified_gold_query": "赵丽颖的作品是什么？"}}

输入: 
- question: "哪本书获得这个奖项的最佳小说奖？"
- gold_query: "哪本书获得这个奖项的最佳小说奖？"
- image_entity: "Nebula Awards"
输出: {{"modified_gold_query": "哪本书获得Nebula Awards奖项的最佳小说奖"}}

输入: 
- question: "他几岁了？"
- gold_query: "他几岁了？"
- image_entity: "唐纳德·特朗普"
输出: {{"modified_gold_query": "唐纳德·特朗普几岁了？"}}

输入: 
- question: "这个队伍赢得了多少个世界大赛冠军？"
- gold_query: "这个队伍赢得了多少个世界大赛冠军？"
- image_entity: "Houston Astros"
输出: {{"modified_gold_query": "Houston Astros赢得了多少个世界大赛冠军？"}}

**输入:**
- question: {question}
- gold_query: {gold_query}
- image_entity: {image_entity}

**输出**: 
'''
    response_text = None
    while response_text is None:
       
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response_text = response_text[0]
        

    if response_text is None:
        print(f"模型响应为空，返回空值")
        return ''

    print("modified_gold_query:", response_text)
    try:
        response_dict = json.loads(response_text.strip('```json\n').strip('```').strip())
    except json.JSONDecodeError:
        print(f"模型响应格式不正确: {response_text}")
        return ''

    # 提取修改后的gold_query
    modified_gold_query = response_dict.get('modified_gold_query', gold_query)

    return modified_gold_query

def process_data(input_file, output_file, model, processor, lang):
    data_list = []
    start_line = get_last_processed_line(output_file)

    with open(output_file, 'a') as writer, jsonlines.open(input_file, 'r') as reader:
        for line_number, obj in enumerate(reader, start=1):
                try:
                    if line_number < start_line:
                        continue
                    # 处理每个数据项
                    question = obj.get('question', '')
                    image_url = obj.get('image_url', '') or obj.get('image', '')
                    answer = obj.get('answer', '')

                    # 第一次API调用，生成gold_query，并提取image_entity
                    gold_query = get_gold_query(question, image_url, answer, model, processor, lang)
                    # obj['gold_query'] = gold_query

                    # 第二次API调用，生成image_query
                    image_query, image_entity = get_image_query(question, gold_query, image_url, model, processor, lang)
                    obj['image_query'] = image_query
                    obj['image_entity'] = image_entity

                    # 第三次API调用，修改gold_query，确保包含image_entity
                    modified_gold_query = get_modified_gold_query(question, gold_query, image_entity, image_url, model, processor, lang)
                    obj['gold_query'] = modified_gold_query

                    # 写入当前数据到output_file
                    writer.write(json.dumps(obj, ensure_ascii=False)+'\n')
                    writer.flush()
                    print(f"数据处理完成，已成功保存到输出文件{output_file}。")

                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")
                    traceback.print_exc()


                    
################################
def get_template(lang):
    
    if lang == 'en':
        template_gold_query = f'''

            **Task**: Based on the following rules, extract keywords and return a dictionary:

            **Rules**:
            1. Use the information from the "image" and "answer" to complete the "question", forming a clear and full question known as "gold_query".
            2. The parts of the "question" that typically need completion often contain demonstratives such as "this", "who", "it", "that".
            3. If the part of the "question" that needs completion lacks demonstratives, identify the main subject needing completion from the image, and incorporate it into the "question".
            4. Other than the completion part, the rest of the "gold_query" should strictly match the "question".
            5. The "gold_query" should include necessary information from the image, allowing the VQA to be answered without viewing the image.
            6. After completion, the "gold_query" should not contain any demonstratives like "this", "who", etc., and must not be exactly the same as the "question".

            **Output Format:**

            {{"gold_query": "The complete question after completion"}}

            **Examples:**

            Input: - question: "What are the works of this actor?" - image: (A photo of Zhao Liying) -answer: "Zhao Liying's main works include 'The Journey of Flower', 'Story of Minglan', etc."
            You should output: {{"gold_query": "What are the works of Zhao Liying?"}}

            Input: - question: Who is the sole student author presenting this type of neural network architecture? - image: (A diagram of LSTM) -answer: "Sepp Hochreiter"
            You should output: {{"gold_query": "Who is the sole student author presenting the LSTM neural network architecture?"}}
            
            Input: - question: When was it released? - image: (A photo of Tesla Model Z) -answer: "Tesla Model Z is set to release in 2024"
            You should output: {{"gold_query": "When was the Tesla Model Z released?"}}

            Input: - question: When did OpenAI release? - image: (A logo of GPT-4o) -answer: "OpenAI released GPT-4o in May 2024"
            You should output: {{"gold_query": "When did OpenAI release GPT-4o?"}}
            '''

        template_image = f'''\
            **Task**: Based on the following rules, extract keywords and return a dictionary:

            **Rules**: 
            1. Compare the "question" with the "gold_query" to identify information that is included in the "gold_query" but missing from the "question". Based on this missing information and the image, formulate a question about the content of the image, known as "image_query", and provide an answer called "image_entity".
            2. Composition rules for "image_query": If the "question" includes the words "this"/"this"/"that" followed by a noun, form the query as "Who is this?" or "What is this?" If there is no noun following "this", the "image_query" should be "What is this?" If there are no clear demonstratives like "this" or "that", further guidance is needed.

            **Output Format**:

            {{"image_query": "", "image_entity": ""}}


            **Examples**:

            Input: - question: "What are this actor’s works?" - gold_query: "What are Zhao Liying’s works?" - image: (A photo of Zhao Liying)
            You should output: {{"image_query": "Who is this actor?", "image_entity": "Zhao Liying"}}

            Input: - question: "When did Epic Gaming first release this?" - gold_query: "When did Epic Gaming first release Minecraft?" - image: (A photo of Minecraft)
            You should output: {{"image_query": "What is this?", "image_entity": "Minecraft"}}

            Input: - question: "Who is the current CTO of this organization?" - gold_query: "Who is the CTO of Alibaba Cloud?" - image: (A photo of Alibaba Cloud)
            You should output: {{"image_query": "What is this organization?", "image_entity": "Alibaba Cloud"}}

            Input: - question: "How much bigger is 4?" - gold_query: "How much bigger is 3 than 4?" - image: (A photo of the number 3)
            You should output: {{"image_query": "What is this?", "image_entity": "3"}}
            '''
        template_modify = f'''\
            **Task**: Based on the following rules, modify the "gold_query" to ensure it is complete and clear, and includes the "image_entity":

            **Rules**:
            1. Check if "gold_query" contains the "image_entity". If "image_entity" is not present in "gold_query", use the "question", "gold_query", "image_entity", and information from the image to modify the "gold_query".
            2. The modified "gold_query" should include contains the "image_entity", allowing the VQA to be answered without viewing the image.

            **Output Format:**
            {{ "modified_gold_query": "The revised complete question" }}

            **Examples:**

            Input:
            - question: "What are the works of this actor?"
            - gold_query: "What are Zhao Liying’s works?"
            - image_entity: "Zhao Liying"
            Output: {{"modified_gold_query": "What are Zhao Liying’s works?"}}

            Input:
            - question: "Which book won this award for best novel?"
            - gold_query: "Which book won this award for best novel?"
            - image_entity: "Nebula Awards"
            Output: {{"modified_gold_query": "Which book won the Nebula Awards for best novel?"}}

            Input:
            - question: "How old is he?"
            - gold_query: "How old is he?"
            - image_entity: "Donald Trump"
            Output: {{"modified_gold_query": "How old is Donald Trump?"}}

            Input:
            - question: "How many world championships has this team won?"
            - gold_query: "How many world championships has this team won?"
            - image_entity: "Houston Astros"
            Output: {{"modified_gold_query": "How many world championships have the Houston Astros won?"}}
            '''

    elif lang == 'zh':
        template_gold_query = f'''
            你是一个用于问答聊天机器人的改写系统。
            
            **你将获得以下信息**：

            - question
            - image
            - answer

            **任务**: 根据以下规则，提取关键词，并返回一个字典: 

            **规则**: 
            1. 根据提供的"question"和"image"，利用image和answer中的信息补全问题，改写出一个完整的、明确的问题，称为"gold_query"。
            2. "question"的需补全部分大多数有指示词"这"，"谁"，"它"，"该"。
            3. 当"question"的需补全部分没有指示词时，需结合图片判断需补全的主体，并加入"question"。
            3. 除了补全问题的部分，其他部分"gold_query"应严格与"question"相同。
            4. "gold_query"应包含必要的图片中信息，使得回答VQA时不用看图片也能回答出问题。
            5. 补全后的"gold_query"不应出现"这"，"谁"等指代词，补全后的"gold_query"不能与"question"完全一样。

            **输出格式: **

            {{"gold_query":"补全后的完整问题"}}

            **示例: **

            输入: - question: "这个演员的作品是什么？"- image: (一张赵丽颖的照片) -answer: "赵丽颖的主要作品有《花千骨》《知否》等。"
            你应该输出: {{"gold_query":"赵丽颖的作品是什么？"}}

            输入: - question: 谁是介绍这种神经网络架构论文的唯一学生作者？- image: (一张LSTM的结构图) -answer: "塞普·霍赫赖特"
            你应该输出: {{"gold_query":"谁是介绍 LSTM 神经网络架构论文的唯一学生作者？"}}

            输入: - question: 它什么时候推出？- image: (一张特斯拉 Model Z的汽车图片) -answer: "特斯拉 Model Z 将于2024年推出"
            你应该输出: {{"gold_query":"特斯拉 Model Z 什么时候推出？"}}

            输入: - question: OpenAI何时发布？- image: (一张GPT4o的logo图片) -answer: "OpenAI于2024年5月发布GPT4o"
            你应该输出: {{"gold_query":"OpenAI何时发布GPT4o？"}}
            '''
        template_image =f'''\
            你是一个用于问答聊天机器人的改写系统。
            
            **你将获得以下信息**：

            - question
            - image
            - gold_query

            **任务**: 根据以下规则，提取关键词，并返回一个字典: 

            **规则**: 
            1. 对比"question"和"gold_query"，找出"gold_query"中包含但"question"中缺少的信息。基于这些缺失的信息和图片，生成一个关于图片内容的提问，称为"image_query"，并给出这个问题的回答"image_entity"。
            2. "image_query"组成规则: question中的"这"/"这个"/"该"+后面的名词+"是谁？"/"是什么？"，如果"这个"后面没有跟着名词，"image_query"应为"这是什么？"。如果没有明显的指代词"这"/"这个"

            **输出格式: **

            {{"image_query":"", "image_entity":""}}

            **示例: **

            输入: - question: 这个演员的作品是什么？- gold_query: 赵丽颖的作品是什么？ - image: (一张赵丽颖的照片)
            你应该输出: {{"image_query":"这个演员是谁？", "image_entity":"赵丽颖"}}

            输入: - question: Epic Gaming 何时首次发布这个？- gold_query: Epic Gaming 何时首次发布 Minecraft？ - image: (一张Minecraft的图片)
            你应该输出: {{"image_query":"这是什么？", "image_entity":"Minecraft"}}

            输入: - question: 这个组织现在的cto是谁来着？- gold_query: 阿里云cto是谁来着？ - image: (一张阿里云的图片)
            你应该输出: {{"image_query":"这个组织是什么？", "image_entity":"阿里云"}}

            输入: - question: 比4大多少？- gold_query: 3比4大多少？ - image: (一张数字3的图片)
            你应该输出: {{"image_query":"这是什么？", "image_entity":"3"}}
            '''
        template_modify =f'''\
            你是一个用于问答聊天机器人的改写系统。
            
            **你将获得以下信息**：

            - question
            - image
            - image_entity

            **任务**: 根据以下规则，修改gold_query，确保其完整且明确，且包含image_entity：

            **规则**:
            1. 检查"gold_query"中是否包含"image_entity"，如果"image_entity"没有出现在"gold_query"中，使用"question"、"gold_query"和"image_entity"以及图片信息修改"gold_query"。
            2. 修改后的"gold_query"必须包含正确的"image_entity"。

            **输出格式: **
            {{ "modified_gold_query": "修改后的完整问题" }}

            **示例: **

            输入: 
            - question: "这个演员的作品是什么？"
            - gold_query: "赵丽颖的作品是什么？"
            - image_entity: "赵丽颖"
            输出: {{"modified_gold_query": "赵丽颖的作品是什么？"}}

            输入: 
            - question: "哪本书获得这个奖项的最佳小说奖？"
            - gold_query: "哪本书获得这个奖项的最佳小说奖？"
            - image_entity: "Nebula Awards"
            输出: {{"modified_gold_query": "哪本书获得Nebula Awards奖项的最佳小说奖"}}

            输入: 
            - question: "他几岁了？"
            - gold_query: "他几岁了？"
            - image_entity: "唐纳德·特朗普"
            输出: {{"modified_gold_query": "唐纳德·特朗普几岁了？"}}

            输入: 
            - question: "这个队伍赢得了多少个世界大赛冠军？"
            - gold_query: "这个队伍赢得了多少个世界大赛冠军？"
            - image_entity: "Houston Astros"
            输出: {{"modified_gold_query": "Houston Astros赢得了多少个世界大赛冠军？"}}
            '''

    else:
        print('Wrong Lang!')
        return []

    return [template_gold_query, template_image, template_modify]


# def process_data(input_file, output_file, llm, sampling_params, lang):
#     with open(output_file, 'w', buffering=1) as writer, jsonlines.open(input_file, 'r') as reader:
#         for line_number, obj in enumerate(reader, start=1):
#             try:
#                 # 获取问题和图像 URL
#                 question = obj.get('question', '')
#                 image_url = obj.get('image_url', '')
#                 answer = obj.get('answer', '')
                
#                 template_list = get_template(lang)

#                 # 生成gold_query，并提取image_entity
#                 prompt_gold_query = f'''\
#                 **Input:**

#                 - question: {question}
#                 - answer: {answer}
#                 '''
#                 message_1 = [
#                     {"role": "system", "content": template_list[0]},
#                     {"role": "user", "content": [
#                         {"type": "image","image": image_url,},
#                         {"type": "text", "text": prompt_gold_query}]}]
#                 gold_query_response = llm.chat(messages=message_1, sampling_params=sampling_params, use_tqdm=True)
#                 gold_query = gold_query_response[0].outputs[0].text
#                 obj['gold_query'] = gold_query

#                 # 生成image_query
#                 prompt_image = f'''\
#                 **Input**:

#                 - question: {question}
#                 - gold_query: {gold_query}
#                 '''
#                 message_2 = [
#                     {"role": "system", "content": template_list[1]},
#                     {"role": "user", "content": [
#                         {"type": "image","image": image_url,},
#                         {"type": "text", "text": prompt_image}]}]
#                 image_query_response = llm.chat(messages=message_2, sampling_params=sampling_params, use_tqdm=True)
#                 image_query = image_query_response[0].outputs[0].text
#                 obj['image_query'] = image_query

#                 # 修改gold_query，确保包含image_entity
#                 prompt_modified = f'''\
#                 **Input:**

#                 - question: {question}
#                 - gold_query: {gold_query}
#                 - image_entity: {image_entity}
#                 '''
#                 message_3 = [
#                     {"role": "system", "content": template_list[2 ]},
#                     {"role": "user", "content": [
#                         {"type": "image","image": image_url,},
#                         {"type": "text", "text": prompt_modified}]}]
#                 modified_gold_query_response = llm.chat(messages=message_3, sampling_params=sampling_params, use_tqdm=True)
#                 modified_gold_query = modified_gold_query_response[0].outputs[0].text
#                 obj['gold_query'] = modified_gold_query

#                 # 保存当前数据到输出文件
#                 writer.write(json.dumps(obj, ensure_ascii=False) + '\n')
#                 writer.flush()
#                 print(f"数据处理完成，已成功保存到输出文件{output_file}。")

#                 torch.cuda.empty_cache()

#             except Exception as e:
#                 print(f"处理第 {line_number} 行时发生错误: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="控制数据处理的参数")
    parser.add_argument("--input_data", type=str, default="", help="输入数据的路径")
    parser.add_argument("--output_data", type=str, default=None, help="")
    parser.add_argument("--lang", type=str, default="", help="")
    args = parser.parse_args()

    if not args.output_data:
        args.output_data = args.input_data.replace('.jsonl', '_w_gold_query.jsonl')
    # model_name =  snapshot_download("qwen/Qwen2-VL-72B-Instruct", cache_dir='/mnt/nas-alinlp/zhuochen.zc/models/')


    # model_name = snapshot_download("qwen/Qwen2-VL-7B-Instruct", cache_dir='/mnt/nas-alinlp/zhuochen.zc/models/')
    model_name = snapshot_download("qwen/Qwen2-VL-72B-Instruct", cache_dir='/mnt/nas-alinlp/zhuochen.zc/models/')
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    process_data(args.input_data, args.output_data, model, processor, args.lang)
    exit()


    # llm = LLM(model='/nas-alinlp/zhuochen.zc/models/qwen/Qwen2-VL-7B-Instruct', enforce_eager=True, trust_remote_code=True)
    # sampling_params = SamplingParams(max_tokens=1024)

    # print("Begin processing data...")
    # process_data(args.input_data, args.output_data, llm, sampling_params, args.lang)

    # model_vl = '/nas-alinlp/zhuochen.zc/models/qwen/Qwen2-VL-72B-Instruct'
    # num_gpus = 4

    # llm = LLM(model=model_vl, tensor_parallel_size=num_gpus, enforce_eager=True, trust_remote_code=True)
    # sampling_params = SamplingParams(max_tokens=1024)
    # process_data(args.input_data, args.output_data, llm, sampling_params, args.lang)
