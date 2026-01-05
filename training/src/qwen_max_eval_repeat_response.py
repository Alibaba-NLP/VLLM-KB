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

# You personal api key to dashscope: qwen-max
dashscope.api_key = False
assert dashscope.api_key

QWEN_SERVER = os.getenv('QWEN_SERVER', default='dashscope')
QWEN_MODEL = os.getenv('QWEN_MODEL', default='qwen-max')

MESSAGE_TEMPLETE = [{"role": "system", "content": DEFAULT_SYSTEM_TEMPLATE}, \
    {"role": "user", "content": DEFAULT_USER_TEMPLATE}]



# TEMPLATE Ref: llama_index/core/evaluation/correctness.py
DEFAULT_SYSTEM_TEMPLATE = """
You are an expert evaluation system for a question answering chatbot.

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

DEFAULT_USER_TEMPLATE = """
## User Query
{query}

## Gold Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""


def _parse2score(str_):
    try:
        score = float(str_.split('\n')[0])
    except:
        score = -1.0
    return score


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
                correct_index = line.rfind('{"question"')
                ret.append(json.loads(line[correct_index:]))
                pass

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
                print("response.output is None. Continuing")
                continue
            text = response.output.text
            if isinstance(text, str) and text:
                break
        except Exception as e:
            print(e)
            time.sleep(0.3)
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

def prompt_qwen_repeat(query: str, answers: Union[list, str], pred: list):
    '''items in PRED might be repeated
    '''

    pred_dict_no_repeat = {k: None for k in set(pred)}

    for p_ in pred_dict_no_repeat:
        response = prompt_qwen_single(query, answers, p_)
        pred_dict_no_repeat[p_] = response
    
    return pred_dict_no_repeat



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
    if os.path.exists(args.output_path) and (not args.overwrite):
        import datetime
        time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.output_path = args.output_path.replace('.jsonl', f"_{time_now}.jsonl" )
    print("Writing to", args.output_path)

    # data_ = load_jsonl('./tmp/test_gpt4_eval.jsonl')
    # print(data_)

    data_ = load_jsonl(args.data_path)
    data_qwen_max_eval = []

    with open(args.output_path, 'w', buffering=1, encoding='utf-8') as g:
        for data in tqdm(data_, desc='prompt qwen-max'):
            new_data = data.copy()

            query = data['question']
            answers = data['answer'] # str or list, both handled
            pred = data['repeat_response']
            pred = [i.strip() for i in pred]

            ans2response_dict = prompt_qwen_repeat(query, answers, pred)

            # print(response)
            new_data['qwen_max_score'] = []
            new_data['qwen_max_eval'] = []
            for k in pred:
                new_data['qwen_max_score'].append(_parse2score(ans2response_dict[k]))
                new_data['qwen_max_eval'].append(ans2response_dict[k])
            
            g.writelines(json.dumps(new_data, ensure_ascii=False)+'\n')
