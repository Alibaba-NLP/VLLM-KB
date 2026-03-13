# coding: utf8
"""
本程序适用于适配了OpenAI API风格（调用方法和返回格式）的API，除了OpenAI之外，还包括但不限于DashScope, 豆包，deepseek，OpenRouter等。
但是因为不同服务的具体解码超参和返回结构体还是有差异，所以请准确使用自己调用模型支持的解码超参，并正确解析最终返回结果。
"""
import json
import os
import sys
import time
import traceback
from typing import Any
from tqdm import tqdm
import multiprocessing
import copy

import requests
import tiktoken  # 因为OpenAI服务比较重要，且是外部供应商代理结果，存在不稳定性，所以引用tiktoken用于增加tokens校验。

from base import BaseAPIClient
from constant import SUPPORT_ARGS
from utils import truncate_long_strings, APIException, get_args, openai_ret_wrapper

import PIL.Image
from io import BytesIO
import requests

class OpenAIAPIClient(BaseAPIClient):
    def __init__(self,
                 call_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 api_key=os.environ.get("DASHSCOPE_API_KEY", None),
                 timeout=1800,
                 verbose_num=1,
                 retry_sleep=2,
                 max_try=10):
        self.call_url = call_url
        self.api_key = api_key 
        print(api_key)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.timeout = timeout
        self.verbose_num = verbose_num
        super(OpenAIAPIClient, self).__init__(
            time_out=timeout,
            verbose_num=verbose_num,
            retry_sleep=retry_sleep,
            max_try=max_try)

    @staticmethod
    def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        if 'audio' in model:
            return -1
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f"Warning: model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(
                    encoding.encode(value, disallowed_special=(encoding.special_tokens_set - {'<|endoftext|>'})))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _track_call(self, func_name: str, payload: Any) -> Any:
        if self._call_track[func_name] < self.verbose_num:
            self._call_track[func_name] += 1
            if self._call_track[func_name] == 1:
                print(f'Log Info for {self.verbose_num} CALLS:')
            print(f"CALL[{self._call_track[func_name]}]: {func_name}")
            print(f'url:' + self.call_url)
            print(f'headers: {truncate_long_strings(self.headers, max_len=30)}')
            print(f'payload: \n{json.dumps(truncate_long_strings(payload), indent=2, ensure_ascii=False)}')

    def _track_response(self, func_name: str, ret_json: dict) -> Any:
        if self._resp_track[func_name] < self.verbose_num:
            self._resp_track[func_name] += 1
            if self._resp_track[func_name] == 1:
                print(f'Log Info for {self.verbose_num} RESPONSE:')
            print(
                f"RESPONSE[{self._resp_track[func_name]}]: {func_name}, response: {json.dumps(truncate_long_strings(ret_json), indent=2, ensure_ascii=False)}")

            # assert True in [compare_dict_structure(resp_json, DATE_20241210_TEXT_FORMAT),
            #                 compare_dict_structure(resp_json, DATE_20241219_DASH_TEXT_FORMAT)]

    def _check_tokens(self, model, messages):
        raise NotImplementedError

    def call(self, **kwargs):
        tenant = kwargs.pop('tenant', None)
        kwargs.pop('pre', None)
        payload = {'tenant': tenant} if tenant else dict()
        for key, value in kwargs.items():
            if value is not None and key in SUPPORT_ARGS:
                payload[key] = value
        assert 'model' in payload
        self._track_call(func_name=payload['model'], payload=payload)
        # message_tokens = OpenAIAPIClient.num_tokens_from_messages(payload['messages'], payload['model'])
        for i in range(self.max_try):
            try:
                ret = requests.post(self.call_url, json=payload,
                                    headers=self.headers, timeout=self.timeout)
                ret_json = ret.json()
                if self.call_url.startswith('http://47.88.8.18') and payload['model'].startswith('claude'):
                    ret_json = openai_ret_wrapper(ret_json, 'mit', 'claude')
                self._track_response(func_name=payload['model'], ret_json=ret_json)
                if ret.status_code != 200:
                    raise APIException(f"http status_code: {ret.status_code}\n{ret.content}")

                if 'choices' not in ret_json:
                    raise APIException(f"Error: {ret_json}")
                for output in ret_json['choices']:
                    if 'finish_reason' not in output:
                        raise APIException(f"Error: {ret_json}")
                    if output['finish_reason'].lower() not in ['stop', 'function_call', 'eos', 'end_turn']:
                        raise APIException(f'openai finish with error...\n{ret_json}')
                # assert message_tokens == ret_json['usage']['prompt_tokens']
                return ret_json
            except APIException as e:
                print(''.join(traceback.format_exception(*sys.exc_info())))
                time.sleep(self.retry_sleep)
        raise APIException('Max Retry!!!')


def make_api(args):
    # prompt = "Could you write a literature review on the use of blockchain technology in supply chain management within industrial engineering, with a focus on enhancing transparency, efficiency, and security in logistics? The literature review should be written in a formal academic tone, suitable for an audience of researchers and industry professionals. It should adhere to a structured outline, including an introduction, thematic sections (transparency, efficiency, and security in logistics), critical analysis of existing research, gaps in the literature, and a conclusion with future research directions. Additionally, integrate any relevant real-world case studies to provide practical insights, and ensure proper citation of sources in APA format."
    kwargs = dict(
        # model='gpt4-o',
        messages=None,
        stream=False,
    )
    timeout = args.pop('timeout')
    kwargs.update(args.__dict__)
    call_url = kwargs.pop('call_url', None)
    authorization = kwargs.pop('authorization', None)

    if call_url and authorization:
        api = OpenAIAPIClient(call_url=call_url,
                              header_authorization=authorization,
                              timeout=timeout)
    else:
        api = OpenAIAPIClient(timeout=timeout)
    
    return api, kwargs


def call_gpt4o(api, kwargs_):
    from rich import print as rprint
    # rprint(kwargs_)

    response = api.call(**kwargs_)
    if kwargs_['model'].startswith('claude'):
        print(response['choices'][0]['message']['content'])
    
    response = response['choices'][0]['message']['content']
    # rprint(response)
    return response



messages_template = [
    {
        'role': 'user', 
        'content': [
            # {'type': 'text', 'text': 'What is the picture?'},
            # {
            #     'type': 'image_url', 
            #     'image_url': {
            #         'url': 'https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en_zh/Freshqa_en_extracted_images/image_1.jpeg'
            #     }
            # }
        ]
    }
]

def _text_wrapper(text):
    return {'type': 'text', 'text': text}

def _image_wrpper(image):
    return {
        'type': 'image_url', 
        'image_url': {
            'url': image
        }
    }



def _filter_by_url(list_):
    ret = []
    keys_to_check = ['text', 'title', 'snippet']

    for search_data in list_:
        try:
            image_path = search_data.get('image', False) or search_data.get('image_url', False) or search_data.get('image_path', False)
            
            response = requests.get(image_path, timeout=5)
            response.raise_for_status()
            img_data = BytesIO(response.content)
            pil_img = PIL.Image.open(img_data)
            search_data.update({'image': image_path})

            values = [search_data[key] for key in keys_to_check if key in search_data]
            text = "\n".join(values)
            search_data.update({'text': text})

            ret.append(search_data)
        except Exception as e:
            print(e)
            # pass

        if len(ret) == 5:
            break
    print("# Knowledge used:", len(ret))
    return ret



def _filter_by_text(list_):
    r'''
    Concate values of keys:['text', 'title', 'snippet'] by \n, and name it under 'text'
    '''
    keys_to_check = ['text', 'title', 'snippet']
    ret = []

    for search_data in list_:
        values = [search_data[key] for key in keys_to_check if key in search_data]
        result = "\n".join(values)
        search_data.update({'text': result})
        ret.append(search_data)
    
    return ret


def _make_norag_input(data):
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        _text_wrapper(data['question'])
    )
    ret[0]['content'].append(
        _image_wrpper(data['image_url'])
    )
    return ret


def _make_text2text_input(data):
    search_data = data.get('search_data', []) or data.get('search_data_text2text', [])
    search_data = _filter_by_text(search_data)
    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        _text_wrapper('Knowledge:')
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append(_text_wrapper(knowledge['text']))

    ret[0]['content'] += [
        _text_wrapper('Refer to the following knowledge to answer the question. Respond using the same language as the question.'),
        _text_wrapper(data['question']),
        _image_wrpper(data.get('image_url', False) or data.get('image', False)),
    ]
    return ret


def _make_image2image_input(data):
    search_data = _filter_by_url(data.get('search_data', []) or data.get('search_data_image2image', []))
    search_data = _filter_by_text(search_data)

    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        _text_wrapper('Knowledge:')
    )

    for i, knowledge in enumerate(search_data):
        ret[0]['content'].append(_text_wrapper(knowledge['text']))
        ret[0]['content'].append(_image_wrpper(knowledge['image']))
        if i == 4:
            break

    ret[0]['content'] += [
        _text_wrapper('Refer to the following knowledge to answer the question. Respond using the same language as the question. '),
        _text_wrapper(data['question']),
        _image_wrpper(data.get('image_url', False) or data.get('image', False)),
    ]
    return ret

def _make_both_input(data):
    search_data_image2image = _filter_by_url(data['search_data_image2image'])
    search_data_image2image = _filter_by_text(search_data_image2image)

    ret = copy.deepcopy(messages_template)
    ret[0]['content'].append(
        _text_wrapper('Knowledge:')
    )
    if len(search_data_image2image) > 0:
        ret[0]['content'].append(_text_wrapper(search_data_image2image[0]['text']))
        ret[0]['content'].append(_image_wrpper(search_data_image2image[0]['image']))
    
    for knowledge in _filter_by_text(data['search_data_text2text']):
        ret[0]['content'].append(_text_wrapper(knowledge['text']))
    
    ret[0]['content'] += [
        _text_wrapper('Refer to the following knowledge to answer the question. Respond using the same language as the question. '),
        _text_wrapper(data['question']),
        _image_wrpper(data.get('image_url', False) or data.get('image', False)),
    ]

    return ret


def process_line(line):
    global api, kwargs_

    kwargs_local = copy.deepcopy(kwargs_)

    data = json.loads(line)
    
    if args.rag == 'norag':
        messages = _make_norag_input(data)
    elif args.rag == 'text2text':
        messages = _make_text2text_input(data)
    elif args.rag == 'image2image':
        messages = _make_image2image_input(data)
    elif args.rag == 'both':
        messages = _make_both_input(data)

    kwargs_local['messages'] = messages
    # breakpoint()
    # for _ in range(10):

    try:
        response = call_gpt4o(api, kwargs_local)
    except Exception as e:
        response = ''
        print(e, 'response=""')
    
    data['response'] = response

    return json.dumps(data, ensure_ascii=False)+'\n'


# def read_in_chunks(file, chunk_size=1000):
#     """Lazy function to read a file in chunks."""
#     while True:
#         lines = file.readlines(chunk_size)
#         if not lines:
#             break
#         yield lines

def read_in_chunks(file, chunk_size=1000):
    chunk = []
    for line in file:
        chunk.append(line.strip())  # 去掉每行的换行符并添加到 chunk 中
        if len(chunk) == chunk_size:
            yield chunk  # 返回当前 chunk
            chunk = []  # 重置 chunk
    if chunk:  # 处理文件末尾不足 chunk_size 的部分
        yield chunk

def process_file(args):
    try:
        with open(args.out_file) as f:
            num_lines_finished = len(f.readlines())
    except:
        num_lines_finished = 0
    print('Num lines finished:', num_lines_finished)
    
    if args.num_workers > 1:
        with open(args.in_file, 'r') as f, open(args.out_file, 'w', buffering=8) as g:

            print('mp: ', args.num_workers)

            with multiprocessing.Pool(args.num_workers) as pool:
                for chunk in read_in_chunks(f, chunk_size=10_000):

                    processed_lines = list(tqdm(pool.imap(process_line, chunk), total=len(chunk), ncols=100))
                    
                    for line in processed_lines:
                        if line:
                            g.write(line)

    else:
        with open(args.in_file, 'r') as f, open(args.out_file, 'a', buffering=1) as g:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, ncols=100, desc=os.path.basename(args.out_file))):
                if i<num_lines_finished:
                    continue
                else:
                    g.write(
                        process_line(line)
                    )

args = get_args()
# args.out_file = args.in_file.replace('.jsonl', f"_chatgpt-4o-latest.jsonl")
args.out_file = args.in_file.replace('.jsonl', f"_{args.model}.jsonl")
output_file = args.out_file


global api, kwargs_
api, kwargs_ = make_api(args)

# single test
# call_gpt4o(api, kwargs_={
#     'messages': [
#         {
#             'role': 'user', 
#             'content': [
#                 {'type': 'text', 'text': 'What is the picture?'},
#                 {
#                     'type': 'image_url', 
#                     'image_url': {
#                         'url': 'https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en_zh/Freshqa_en_extracted_images/image_1.jpeg'
#                     }
#                 }
#             ]
#         }
#     ],
#     'stream': False,
#     'in_file': '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_en/norag/vfreshqa_en.jsonl',
#     'out_file': '/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_en/norag/vfreshqa_en_gpt-4o-2024-11-20.jsonl',
#     'num_workers': 1,
#     'local_cache': False,
#     'model': args.model,
#     'seed': None,
#     'top_p': None,
#     'top_k': None,
#     'temperature': None,
#     'max_tokens': None,
#     'max_completion_tokens': None,
#     'repetition_penalty': None,
#     'presence_penalty': None,
#     'system': None,
#     'stop': None,
#     'n': None,
#     'pre': False,
#     'completion': False,
#     'rag': 'norag'
# })

# exit()


process_file(args)
