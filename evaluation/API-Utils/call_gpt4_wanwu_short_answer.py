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

import requests
import tiktoken  # 因为OpenAI服务比较重要，且是外部供应商代理结果，存在不稳定性，所以引用tiktoken用于增加tokens校验。

from base import BaseAPIClient
from constant import SUPPORT_ARGS
from utils import truncate_long_strings, APIException, get_args, openai_ret_wrapper


class OpenAIAPIClient(BaseAPIClient):
    def __init__(self,
                 call_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 api_key=os.environ.get("DASHSCOPE_API_KEY", None),
                 timeout=1800,
                 verbose_num=1):
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
            verbose_num=verbose_num)

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
        messages=[{"role": "user", "content": None}],
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


def call_gpt4o(api, kwargs_, prompt):
    kwargs_['messages'][0]['content'] = prompt

    response = api.call(**kwargs_)
    if kwargs_['model'].startswith('claude'):
        print(response['choices'][0]['message']['content'])
    
    response = response['choices'][0]['message']['content']
    return response


def process_line(line):
    global api, kwargs_

    data = json.loads(line)
    query = data.get('gold_query', False)
    _rag = data.get('search_data_text', False)
    rag_text = ''

    if not query or not _rag:
        print('Broken data.')
        print(data)
        return False
    
    for i, rag in enumerate(_rag):
        rag_text += rag['title'] + '\n'
        rag_text += rag['snippet'] + '\n\n'

        if i == 4:
            break

    prompt = (
        f"请根据参考内容，回答问题，然后给出你的答案。"
        f"你应该只输出一个答案，不要输出其他内容。"
        f"你的回答应尽量简短，用一个词或一句话精准回答。\n"
        f"问题：{query}\n参考内容：{rag_text}\n你的回答："
    )

    for _ in range(10):
        try:
            response = call_gpt4o(api, kwargs_, prompt)
            break
        except Exception as e:
            response = None
            print(e)
            print('Retrying...', _)
    
    if response:
        data['answer'] = response
        return json.dumps(data, ensure_ascii=False)+'\n'

    return False


def read_in_chunks(file, chunk_size=1000):
    """Lazy function to read a file in chunks."""
    while True:
        lines = file.readlines(chunk_size)
        if not lines:
            break
        yield lines

def process_file(args):
    with open(args.in_file, 'r') as f, open(args.out_file, 'w', buffering=8) as g:

        print('mp: ', args.num_workers)

        with multiprocessing.Pool(args.num_workers) as pool:
            for chunk in read_in_chunks(f, chunk_size=1_000_000):

                processed_lines = list(tqdm(pool.imap(process_line, chunk), total=len(chunk)))
                
                for line in processed_lines:
                    if line:
                        g.write(line)


args = get_args()

# call_gpt4o(args, prompt = '1+1=?')
global api, kwargs_
api, kwargs_ = make_api(args)

process_file(args)
