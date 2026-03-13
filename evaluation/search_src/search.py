import os
import argparse
from datetime import date
from datetime import datetime
from http import HTTPStatus
import requests
import json
from tqdm import tqdm
import base64
from PIL import Image
from io import BytesIO
import sys
import uuid
import argparse
import time
from icecream import ic


from utils import (
    ensure_directory_exists,
    is_english_or_chinese,
    base64_to_image,
    encode_url,
    save_image_from_url
)
import re
import multiprocessing
import random
import urllib.parse

import load_dotenv

# 参考 /mnt/nas-alinlp/lyh/codes/MMRAG/Entity/image_image_google_vfreshqa.py
# 参考 /mnt/nas-alinlp/lyh/codes/MMRAG/Qwen-VL/eval_mm/process_image_image_data.py



def search_text_by_text(text, engine= 'google', search_mode = 'complete', retry_attempt=10):

    if engine == 'google':
        url = 'https://idealab.alibaba-inc.com/api/v1/search/search'

        # 设置Headers
        headers = {
            'X-AK': os.getenv('SEARCH_TEXT_KEY')
            'Content-Type': 'application/json'
        }

        # 设置请求数据
        data = {
            "query": text,
            "num": 10,
            "extendParams": {
                # "country": "cn",
                # "locale": "zh-cn",
                # "location": "China",
                # "page": 2
            },
            "platformInput": {
                "model": "google-search"
            }
        }

            # breakpoint()
        for i in range(retry_attempt):
            try:
                # 发送POST请求
                response = requests.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()  # 如果返回的是4xx或5xx错误，这将抛出异常
                answer = response.json()

                if not answer.get('success', True):  # 检查操作是否成功
                    error_message = answer.get('message', 'Unknown error')
                    print("Operation failed:", error_message)
                    continue 

                # 只要其中的organic，其他不要
                answer = answer.get('data', {}).get('originalOutput', {}).get('organic', {})

                if search_mode == 'complete':
                    return answer
                
                elif search_mode == 'simple':
                    if len(answer) > 0:
                        outputs = []

                        for val in answer:
                            title = val.get('title')
                            snippet = val.get('snippet')
                            text = f"{title} {snippet}"  # 使用 f-string 格式化合并字符串，避免粘连
                            outputs.append({
                                'text': text.strip()  # 使用 strip() 去除可能的首尾空格
                            })
                        breakpoint()
                        return outputs
                    
                else:
                    return []

            except requests.exceptions.HTTPError as errh:
                print("An Http Error occurred:", errh)
                continue
            except requests.exceptions.ConnectionError as errc:
                print("An Error Connecting to the API occurred:", errc)
                continue
            except requests.exceptions.Timeout as errt:
                print("A Timeout Error occurred:", errt)
                continue
            except requests.exceptions.RequestException as err:
                print("An Unknown Error occurred:", err)
                continue
        return {}

       
    elif engine == 'bing':

        url = 'https://idealab.alibaba-inc.com/api/v1/search/search'

        # 设置Headers
        headers = {
            'X-AK': os.getenv('SEARCH_TEXT_KEY'),  
            'Content-Type': 'application/json'
        }

        # 设置请求数据
        data = {
            "query": text,
            "num": 10,
            "extendParams": {
                # "country": "cn",
                # "locale": "zh-cn",
                # "location": "China",
                # "page": 2
            },
            "platformInput": {
                "model": "bing-search",
                "instanceVersion": "S1"
            }
        }

        for i in range(retry_attempt):

            try:
                # 发送POST请求
                response = requests.post(url, headers=headers, data=json.dumps(data))
                response.raise_for_status()  # 如果返回的是4xx或5xx错误，这将抛出异常
                answer = response.json()

                if not answer.get('success', True):  # 检查操作是否成功
                    error_message = answer.get('message', 'Unknown error')
                    print("Operation failed:", error_message)
                    continue 
              
                answer = response.json().get('data', {}).get('originalOutput', {}).get('webPages', {}).get('value',{})
                
                # answer1 = answer.get('data', {})
                # # if not answer1:
                # #     print("0")
                # #     breakpoint()
                # answer2 =answer1.get('originalOutput', {})
                # # if not answer2:
                # #     print("1")
                # #     breakpoint()
                # answer3 = answer2.get('webPages', {})
                # # if not answer3:
                # #     print("2")
                # #     breakpoint()
                # answer = answer3.get('value',{})
                # # if not answer:
                # #     print("3")
                # #     breakpoint()
                
                if search_mode == 'complete':
                    return answer

                elif search_mode == 'simple':
                    if len(answer) > 0:
                        outputs = []
                        for val in answer:

                            name = val.get('name')
                            snippet = val.get('snippet')
                            text = f"{name} {snippet}"  # 使用 f-string 格式化合并字符串，避免粘连
                            outputs.append({'text': text})

                        # breakpoint()
                        return outputs
                    
                    else:
                        return []
            
            except requests.exceptions.HTTPError as errh:
                print("An Http Error occurred:", errh)
                continue
            except requests.exceptions.ConnectionError as errc:
                print("An Error Connecting to the API occurred:", errc)
                continue
            except requests.exceptions.Timeout as errt:
                print("A Timeout Error occurred:", errt)
                continue
            except requests.exceptions.RequestException as err:
                print("An Unknown Error occurred:", err)
                continue
            except Exception as e:
                print(e)
                # breakpoint()
        return {}

    else:
        raise RuntimeError("Search type must be 'google' or 'bing'")


def search_text_by_text(text, tool_use=False, readpage=False, max_try=10, qpMultiQueryConfig=[], onlyCache=True, readpage_topk=10):
    '''
    Returns
      - list of dict: [{'text': ...}, {'text': ...}]. Basically length=10.
    '''
    header = {
        "Content-Type": "application/json",
        "Accept-Encoding": "utf-8",
        "Authorization": "Bearer lm-/19WaNVGhRjcjYcKuOV96w== ",
    }

    template = {
            "rid": str(uuid.uuid4()),
            "scene": "dolphin_search_google_nlp",
            "uq": "",
            "debug": True,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "multiSearch": False,
                "qpMultiQueryConfig": qpMultiQueryConfig,
                "qpMultiQuery": True,
                "qpMultiQueryHistory": [],
                "qpSpellcheck": False,
                "qpEmbedding": False,
                "knnWithScript": False,
                "rerankSize": 10,
                "qpTermsWeight": False,
                "qpToolPlan": tool_use,
                "readpage": readpage,
                "readpageConfig": {"tokens": 4000, "topK": readpage_topk, "onlyCache": onlyCache},
                "pluginServiceConfig": {"qp": "mvp_search_qp_qwen"},  # v3 rewrite
            },
            "rankModelInfo": {
                "default": {
                    "features": [
                        {"name": "static_value", "field": "_weather_score", "weights": 1.0},
                        {
                            "name": "qwen_ranker",
                            "fields": ["hostname", "title", "snippet", "timestamp_format"],
                            "weights": 1,
                            "threshold": -50,
                            "max_length": 512,
                            "rank_size": 100,
                            "norm": False,
                        },
                    ],
                    "aggregate_algo": "weight_avg",
                }
            },
            "headers": {"__d_head_qto": 5000},
    }

    # if item.get("is_multi_turn", False):
    #     template["uq"] = item["messages"][-1]["content"]
    #     template["customConfigInfo"]["qpMultiQueryHistory"] = item["messages"][:-1]
    # else:
    #     template["uq"] = item["query"]

    template["uq"] = text

    for _ in range(max_try):
        try:
            r = requests.post(
                "https://nlp-cn-beijing.aliyuncs.com/gw/v1/api/msearch-sp/qwen-search",
                data=json.dumps(template),
                headers=header,
            )
            r = json.loads(r.text)
            ctxs = r["data"]["docs"]
            outputs = {
                "ctxs": ctxs,
                # "rewrite": list(r["debug"]["qpInfos"].values())[0]["multiQuery"],
                # "toolResult": r["data"]["extras"]["toolResult"],
            }
            # return outputs
            break
        except Exception as e:
            ic(r)
            tqdm.write(f"Search Error: {e}, retrying...")
            time.sleep(0.4)
            outputs = {}
            continue
    # logging.error("Failed search ctxs!")
    # return {}
    
    outputs_ = []
    for item in outputs.get('ctxs', []):
        if des := item.get('snippet', False):
            outputs_.append({
                'text': des
            })
    # breakpoint()
    return outputs_


def search_image_by_image_url(image_url, save_dir='/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/tmp', retry_attempt=10, timeout: int=30, lang='ch'):
    '''
    Returns
    -------
    list
        {'image_path': image,
         'snippet': snippet,
         'url': url}
    '''

    # url = 'https://203.119.169.238/api/v1/lens/search'
    # headers = { "Host": "idealab.alibaba-inc.com",
    #             "X-AK": "5efa5292559fa658625646980868afac", 
    #             "Content-Type": "application/json"
    #             }
    url = 'https://idealab.alibaba-inc.com/api/v1/lens/search'
    headers = { 
                "X-AK": "cb0ec9eb51b7ff849918b5f5b6be64e4", 
                "Content-Type": "application/json",
    }

    template = {
        "extendParams": {
            "url": image_url
        },
        "platformInput": {
            "model": "google-search"
        }
    }
    if lang == 'ch':
        template["extendParams"].update({
            "hl": "zh-cn",
            "gl": "cn"
        })
    
    for _ in range(retry_attempt):
        try:
            resp = requests.post(url, headers=headers, json=template, timeout=(timeout, timeout))
            rst = json.loads(resp.text)

            while isinstance(rst, str):
                try:
                    rst = json.loads(rst)
                except json.JSONDecodeError:
                    break

            docs = rst["data"]["originalOutput"]['organic']
            outputs = []
            
            for i, item in enumerate(docs):

                snippet = item.get('title', '')
                image = item.get('imageUrl', '')
                url = item.get('link', '')
                source = item.get('source', '')
                image_path = save_image_from_url(image, save_dir)

                if not image_path:
                    continue

                outputs.append({
                    'image_url': image,
                    'image_path': image_path,
                    'snippet': snippet,
                    'url': url
                })

                if i>=9:
                    break

            # print(outputs)
            return outputs
        except requests.exceptions.Timeout:
            print(f"请求超时（尝试 {_ + 1}/{retry_attempt}:{image_url}")
        except Exception as e:
            print("Meet error when search image:", e)
            print("retrying")
            continue
    return []

# res = search_image_by_image_url('https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en_zh/Freshqa_en_extracted_images/image_1.jpeg')
# breakpoint()

# exit()

from google_wxy_0321 import google as search_text_by_text


def process_line(args_list):
    if args.rag_type == 'image2image':
        line, image_save_dir, retry_attempt, lang = args_list

        val = json.loads(line)
        input_url = val['image_url']

        search_data = search_image_by_image_url(
            image_url=input_url,
            save_dir=image_save_dir,
            retry_attempt=retry_attempt,
            lang=lang
        )
        val['search_data_image2image'] = search_data
    
    elif args.rag_type == 'text2text':
        line = args_list
        val = json.loads(line)
        query = val.get('gold_query', False) or val.get('text') or val.get('query') or val.get('question', '')

        val['search_data_text2text'] = search_text_by_text(query)

    return val



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='', help='google/bing')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='simple', 
                        help='The amount of saved information, e.g., "simple" or "complete".')
    parser.add_argument('--rag_type', type=str, default='',
                        help='image2image, text2text, text2image')
    parser.add_argument('--retry_attempt', type=int, default=10)
    #当 --download_image 明确出现在命令行中时，args.download_image 才会被设置为 True。默认情况下，如果不提供这个选项，它将为 False
    parser.add_argument('--download_image', action='store_true', help='If set, download images.') 
    parser.add_argument('--lang', help='ch/en')
    parser.add_argument('--mp', type=int, default=1, help='setting to -1 using total cpu')
    parser.add_argument('--rewrite', type=int, default=0, help='Rewrite or append')
    args = parser.parse_args()

    print(args.dataset)

    if args.dataset == 'lifevqa':
        # query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/lifevqa/norag/lifevqa.jsonl'
        query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/lifevqa/norag/lifevqa_w_gold_query_from_origin_query_image.jsonl'
        args.lang = 'ch'
    elif args.dataset == 'qwenvqa':
        query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/qwenvqa/norag/qwenvqa_w_gold_query_from_origin_query_image.jsonl'
        args.lang = 'ch'
    elif args.dataset == 'vfreshqa_ch':
        # query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_ch/norag/vfreshqa_ch_w_gold_query_from_origin_query_image.jsonl'
        query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_ch/norag/vfreshqa_ch_w_gold_query_from_i2i.jsonl'
        args.lang = 'ch'
    elif args.dataset == 'vfreshqa_en':
        # query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_en/norag/vfreshqa_en_w_gold_query_from_origin_query_image.jsonl'
        query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_en/norag/vfreshqa_en_w_gold_query_from_i2i.jsonl'
        args.lang = 'en'
    elif args.dataset == 'visual7w':
        query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/visual7w/norag/visual7w_w_gold_query_from_origin_query_image.jsonl'
        args.lang == 'en'
    elif args.dataset == 'nocaps':
        # query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/nocaps/norag/nocaps.jsonl'
        query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/nocaps/norag/nocaps_w_gold_query_from_origin_query_image.jsonl'
        # query_dataset = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/nocaps/norag/nocaps_w_gold_query_from_i2i.jsonl'
        args.lang == 'en'


    # save_path = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/{args.dataset}/{args.rag_type}/{args.dataset}.jsonl'
    # save_path = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/{args.dataset}/{args.rag_type}_gold_query_from_image_search/{args.dataset}.jsonl'
    save_path = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/{args.dataset}/{args.rag_type}_gold_query_from_origin_query_image/{args.dataset}.jsonl'
    
    image_save_dir = '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/tmp'

    ensure_directory_exists(save_path)
    ensure_directory_exists(image_save_dir)
    date_ = datetime.today().strftime("%m%d")

    print('Write to', save_path)

    with open(query_dataset, 'r') as f:
        lines = f.readlines()

    if args.rag_type == 'image2image':
        args_list = [(line, image_save_dir, args.retry_attempt, args.lang) for line in lines]

    elif args.rag_type == 'text2text':
        args_list = [line for line in lines]
 
    if args.rewrite:
        mode = 'w'
        lines_written = 0
        print('Rewrite...')
    else:
        mode = 'a'
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                lines_written = len(f.readlines())
        except:
            lines_written = 0
        print(f"{lines_written} lines in {save_path}. Append...")


    args_list = args_list[lines_written:]

    if args.mp == 1:
        with open(save_path, mode=mode, encoding="utf-8", buffering=1) as out_file:
            for args_ in tqdm(args_list):
                result = process_line(args_)
                out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    else:
        cpu_num = args.mp if args.mp > 0 else multiprocessing.cpu_count()
        print('# CPU:', cpu_num)
        pool = multiprocessing.Pool(processes=cpu_num)
        with open(save_path, mode=mode, encoding="utf-8", buffering=1) as out_file:
            with multiprocessing.Pool(processes=cpu_num) as pool:
                for result in tqdm(pool.imap(process_line, args_list), total=len(args_list), ncols=100):
                    out_file.write(json.dumps(result, ensure_ascii=False) + '\n')


