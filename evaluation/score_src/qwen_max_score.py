# TEMPLATE Ref: llama_index/core/evaluation/correctness.py
DEFAULT_SYSTEM_TEMPLATE = """\
You are an expert evaluation system for a visual question answering chatbot. The visual information is omitted and you do not need it. 

You are given the following information:
- a user query, 
- a generated answer, and
- gold answer(s)

Your job is to judge the relevance and correctness of the generated answer according to the given gold answer. 
Do not use your personal opinion.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is relevant but contains mistakes, \
you should give a score between 2 and 3.
- If the generated answer is close to the given gold answer(s), \
you should give a score between 4 and 5. 
- If there are multiple gold answers, you can use the most likely one as the reference \
and there is no need to consider all of them. 
- The score does not have to be integer.

Example Response:
4.0
"""


def _parse2score(str_):
    try:
        score = float(str_.split('\n')[0])
    except:
        score = -1.0
    return score

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Gold Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

import datetime
import json
import os
import random
import re
import sys
import time
import copy
from typing import Dict, List, Tuple, Union
from urllib.parse import urlencode
import argparse
from tqdm import tqdm

import dashscope
import requests


import load_dotenv
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')

QWEN_SERVER = os.getenv('QWEN_SERVER', default='dashscope')
QWEN_MODEL = os.getenv('QWEN_MODEL', default='qwen-max')

MESSAGE_TEMPLETE = [{"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE}, \
    {"role": "user", "content": DEFAULT_USER_TEMPLATE}]


def load_jsonl(path_):
    with open(path_, 'r') as f:
        ret = []
        for idx, line in enumerate(f.readlines()):
            try:
                ret.append(json.loads(line))
            except:
                print(idx, 'json.loads error:')
                print(line)
                print('='*30)
        return ret

def call_qwen_dash(model, message, use_raw_prompt, stop_words, top_k):
    retry_limit = 10
    count = 0
    text = ''
    
    kwargs = {}
    kwargs['debug'] = True
    kwargs['headers'] = {'X-DashScope-DataInspection': 'disable'}
    
    while count < retry_limit:
        try:
            # print("dashscope.Generation.call")
            response = dashscope.Generation.call(model=model,
                messages=message,
                # prompt=prompt,
                use_raw_prompt=use_raw_prompt,
                stop_words=stop_words,
                top_k=top_k,
                **kwargs)
            # print(response)
            # print("dashscope.Generation.call done")
            if response.output is None:
                count += 1
                print(f"response.output is None. Continuing {count}")
                print("response:", response)
                continue
            text = response.output.text
            if isinstance(text, str) and text:
                break
        except Exception as e:
            print(e)
            time.sleep(0.2)
            pass

        count += 1
        print('Generation.call failed. Retrying...%d' % count, file=sys.stderr)
        continue
    return text

def prompt_qwen_single(query: str, answers: Union[list, str], pred: str):
    message = copy.deepcopy(MESSAGE_TEMPLETE)
    message[-1]['content'] = message[-1]['content'].format(query=query, reference_answer=answers, generated_answer=pred)
    # print(message)

    response = call_qwen_dash(
        model=QWEN_MODEL,
        message=message,
        # prompt=prompt,
        use_raw_prompt=False,
        stop_words=[{
            'stop_str': 'Observation:',
            'mode': 'exclude'
        }],
        top_k=1,
    )

    return response


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', default='')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--overwrite', type=int, default=0)
    # parser.add_argument('--error_path', default='')

    args = parser.parse_args()

    if args.output_path == '':
        args.output_path = args.data_path.replace('.jsonl', '_qwen_max_eval.jsonl')
    print("Loading", args.data_path)

    assert args.output_path != args.data_path, print("Over write data_path!")
    data_ = load_jsonl(args.data_path)
    
    if args.overwrite and os.path.exists(args.output_path):
        print(f"Over write {args.output_path}")
        os.remove(args.output_path)
        num_lines_finished = 0
    else:
        try:
            with open(args.output_path, 'r') as f:
                num_lines_finished = len(f.readlines())
        except:
            num_lines_finished = 0
        if num_lines_finished == len(data_):
            print('Skip', args.data_path)
            print('='*30)
            sys.exit()
        print(f'Skip {num_lines_finished} lines')

    with open(args.output_path, 'a', buffering=1, encoding='utf-8') as g:
        for i, data in enumerate(tqdm(data_, desc='qwen-max scoring...', ncols=100)):
            
            if i < num_lines_finished:
                continue

            new_data = data.copy()

            query = data['question']
            answers = data['answer'] # str or list, both handled
            pred = data['response']

            qwen_eval_score = prompt_qwen_single(query, answers, pred)

            new_data['qwen_max_score'] = _parse2score(qwen_eval_score)

            g.writelines(json.dumps(new_data, ensure_ascii=False)+'\n')
