"""Copyright (c) 2022, salesforce.com, inc.

All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# coding=utf-8

__author__ = 'aagrawal'

import re
# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import sys
import jieba
from collections import Counter
from nltk.tokenize import word_tokenize
import dashscope
import requests
import os
import copy
from numpy import mean
import time
from dotenv import load_dotenv, find_dotenv

# Find the .env file in the parent directory
dotenv_path = find_dotenv(filename='.env', raise_error_if_not_found=True)

# Load the .env file
load_dotenv(dotenv_path)

DASHSCOPE = os.getenv('DASHSCOPE')


dashscope.api_key = DASHSCOPE
QWEN_SERVER = os.getenv('QWEN_SERVER', default='dashscope')
QWEN_MODEL = os.getenv('QWEN_MODEL', default='qwen-max')


def process_string(s):
    s = str(s)
    words = []
    for word in ' '.join(jieba.cut(s)).split():
        if word not in '，、。 ,.《》':
            words.append(word)
    return words

def process_string_en(s):
    s = str(s).lower()
    words = []
    for word in word_tokenize(s):
        if word not in ',.?!:;\'"':
            words.append(word)
    return words

def compute_acc_single(gold_toks, pred_toks):
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    return num_same / len(gold_toks)

def compute_acc(a_golds, a_pred, lang):
    if lang == 'zh':
        if a_pred == '':
            return 0
        golds_toks = [process_string(a_gold) for a_gold in a_golds]
        pred_toks = process_string(a_pred)
    elif lang == 'en':
        if a_pred == '':
            return 0
        golds_toks = [process_string_en(a_gold) for a_gold in a_golds]
        pred_toks = process_string_en(a_pred)

    return max(
        compute_acc_single(gold_toks, pred_toks) for gold_toks in golds_toks)


def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    if type(text) is not str:
        text = str(text)
    match = pattern.search(text)
    return match is not None

def compute_acc_en_ch(a_golds, a_pred):
    if a_pred == '':
        return 0
    ch = contains_chinese(a_pred)
    ch = ch or bool(sum([contains_chinese(a_gold) for a_gold in a_golds]))
    
    if ch:
        golds_toks = [process_string(a_gold) for a_gold in a_golds]
        pred_toks = process_string(a_pred)
    else:
        golds_toks = [process_string_en(a_gold) for a_gold in a_golds]
        pred_toks = process_string_en(a_pred)
    
    return max(
        compute_acc_single(gold_toks, pred_toks) for gold_toks in golds_toks)


def qwen_evaluate(pred:str, answers:list, retry_attempt=10):
    prompt_1 = '''\
请你参考下面的参考答案，评估模型输出，并给出0到1之间的打分。
- 0表示模型输出完全错误，1表示模型输出正确。
- 你也可以打出0到1之间的小数。参考答案可能会有多个（每行一个），模型输出只会有一个。
- 请只输出数字，不要输出其他任何内容。
示例输出：0.8

'''

    prompt_2 = '''\
模型输出：
{pred}

参考答案
{answer}

输出：
'''   
    MESSAGE_TEMPLETE = [{"role": "system", "content": prompt_1}, \
                        {"role": "user", "content": prompt_2}]
    kwargs = {}
    kwargs['debug'] = True
    kwargs['headers'] = {'X-DashScope-DataInspection': 'disable'}

    score = 0.
    message = copy.deepcopy(MESSAGE_TEMPLETE)
    answers = list(map(str,answers))
    message[-1]['content'] = message[-1]['content'].format(pred=pred, answer='\n'.join(answers))

    # for _ in range(retry_attempt):
    tried = -2
    while tried < retry_attempt:
        try:
            # breakpoint()
            response = dashscope.Generation.call(model=QWEN_MODEL,
                messages=message,
                # prompt=prompt,
                use_raw_prompt=False,
                stop_words=[{
                    'stop_str': 'Observation:',
                    'mode': 'exclude'
                }],
                top_k=1,
                **kwargs)
            
            if response.status_code == 400:
                return 0.0
            if response.status_code == 429:
                time.sleep(0.1)
                continue
            if response.status_code == 500: # RequestTimeOut
                print(RequestTimeOut)
                time.sleep(0.1)
                continue
                
            if response.output is None:
                print('response.out is None. Continue')
                print(response)
                continue

            score = response.output.text.split('\n')[0].strip()
            score = float(score)
            return score

        except ValueError:
            print(f'ValueError, {QWEN_MODEL} response score:', score[:100])
            print('Return score=0')
            break
        except Exception as e:
            print('qwen_evaluate(): ', e)
            # break
        
        if retry_attempt != -1:
            tried += 1

    # print('Reached max retry, score=', score)
    return score

class VQAEval:

    def __init__(self, vqa=None, vqaRes=None, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        if vqa is not None:
            self.params = {'question_id': vqa.getQuesIds()}
        self.contractions = {
            'aint': "ain't",
            'arent': "aren't",
            'cant': "can't",
            'couldve': "could've",
            'couldnt': "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            'didnt': "didn't",
            'doesnt': "doesn't",
            'dont': "don't",
            'hadnt': "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            'hasnt': "hasn't",
            'havent': "haven't",
            'hed': "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            'hes': "he's",
            'howd': "how'd",
            'howll': "how'll",
            'hows': "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            'Im': "I'm",
            'Ive': "I've",
            'isnt': "isn't",
            'itd': "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            'itll': "it'll",
            "let's": "let's",
            'maam': "ma'am",
            'mightnt': "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            'mightve': "might've",
            'mustnt': "mustn't",
            'mustve': "must've",
            'neednt': "needn't",
            'notve': "not've",
            'oclock': "o'clock",
            'oughtnt': "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            'shant': "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            'shouldve': "should've",
            'shouldnt': "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": 'somebodyd',
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            'somebodyll': "somebody'll",
            'somebodys': "somebody's",
            'someoned': "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            'someonell': "someone'll",
            'someones': "someone's",
            'somethingd': "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            'somethingll': "something'll",
            'thats': "that's",
            'thered': "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            'therere': "there're",
            'theres': "there's",
            'theyd': "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            'theyll': "they'll",
            'theyre': "they're",
            'theyve': "they've",
            'twas': "'twas",
            'wasnt': "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            'weve': "we've",
            'werent': "weren't",
            'whatll': "what'll",
            'whatre': "what're",
            'whats': "what's",
            'whatve': "what've",
            'whens': "when's",
            'whered': "where'd",
            'wheres': "where's",
            'whereve': "where've",
            'whod': "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            'wholl': "who'll",
            'whos': "who's",
            'whove': "who've",
            'whyll': "why'll",
            'whyre': "why're",
            'whys': "why's",
            'wont': "won't",
            'wouldve': "would've",
            'wouldnt': "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            'yall': "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            'youd': "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            'youll': "you'll",
            'youre': "you're",
            'youve': "you've",
        }
        self.manualMap = {
            'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10',
        }
        self.articles = ['a', 'an', 'the']

        self.periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')
        self.commaStrip = re.compile('(\d)(,)(\d)')
        self.punct = [
            ';',
            r'/',
            '[',
            ']',
            '"',
            '{',
            '}',
            '(',
            ')',
            '=',
            '+',
            '\\',
            '_',
            '-',
            '>',
            '<',
            '@',
            '`',
            ',',
            '?',
            '!',
        ]

    def evaluate_origin(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            # for small data
            if quesId not in self.vqa.qa.keys() or quesId not in self.vqaRes.qa.keys():
                continue
            gts[quesId] = self.vqa.qa[quesId] 
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}
        correct_ques_ids = []

        print('computing accuracy')
        step = 0
        for quesId in quesIds:
            # for small data
            if quesId not in res.keys():
                continue
            resAns = res[quesId]['answer']
            if resAns is not None:
                resAns = resAns.replace('\n', ' ')
            else:
                print(f"resAns:{resAns}")
                # 处理 resAns 为 None 的情况，例如设置一个默认值或记录错误
                # resAns = '没有回答'
                continue
            # resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(
                        str(ansDic['answer']))


            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [
                    item for item in gts[quesId]['answers']
                    if item != gtAnsDatum
                ]
                matchingAns = [
                    item for item in otherGTAns if item['answer'] == resAns
                ]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)

            if avgGTAcc > 0:  # Check if the answer is correct
                correct_ques_ids.append(quesId)  # Store the question ID of correct answers

            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 20 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print('Done computing accuracy')
        
    def evaluate_en(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            # for small data
            if quesId not in self.vqa.qa.keys() or quesId not in self.vqaRes.qa.keys():
                continue
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        qwen_scoreQuesType = {}
        accAnsType = {}
        correct_ques_ids = []
        # print('computing accuracy')

        qwen_scores = []
        step = 0
        for quesId in quesIds:
            # for small data
            if quesId not in res.keys():
                continue
            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(
                        str(ansDic['answer']))

            answers = []
            for gtAnsDatum in gts[quesId]['answers']:
                answers.append(gtAnsDatum['answer'])
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']

            avgGTAcc = compute_acc(a_golds=answers, a_pred=resAns, lang='en')
            # avgGTAcc = compute_acc_en_ch(a_golds=answers, a_pred=resAns)
            accQA.append(avgGTAcc)
            
            try:
                qwen_score = qwen_evaluate(resAns, answers, 1000)
            except Exception as e:
                breakpoint()
                print(e)
                qwen_score = 0.

            qwen_scores.append(qwen_score)

            if avgGTAcc > 0:  # Check if the answer is correct
                correct_ques_ids.append(quesId)  # Store the question ID of correct answers

            if quesType not in accQuesType:
                accQuesType[quesType] = []
                qwen_scoreQuesType[quesType] = []
            
            accQuesType[quesType].append(avgGTAcc)
            qwen_scoreQuesType[quesType].append(qwen_score)

            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 20 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        # print('Done computing accuracy')
        # print('Question IDs with correct answers:', correct_ques_ids)

        # print('Qwen score:', mean(qwen_scores)*100)
        for type_ in qwen_scoreQuesType:
            print(qwen_scoreQuesType[type_])
            print('='*40)
            qwen_scoreQuesType[type_] = mean(qwen_scoreQuesType[type_])*100
        print(qwen_scoreQuesType)

        return {'acc': accQA,
                'qwen_score': qwen_scores
        }

    def evaluate(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            # for small data
            if quesId not in self.vqa.qa.keys() or quesId not in self.vqaRes.qa.keys():
                continue
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        qwen_scoreQuesType = {}
        accAnsType = {}
        correct_ques_ids = []
        # print('computing accuracy')

        qwen_scores = []
        step = 0
        for quesId in quesIds:
            # for small data
            if quesId not in res.keys():
                continue
            resAns = res[quesId]['answer']
            # breakpoint()
            # print(resAns)

            if resAns is not None:
                resAns = resAns.replace('\n', ' ')
            else:
                print(f"resAns:{resAns}")
                continue
                # resAns = '没有回答'

            # resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    # print(ansDic['answer'])
                    ansDic['answer'] = self.processPunctuation(
                        str(ansDic['answer']))
           
            answers = []
            for gtAnsDatum in gts[quesId]['answers']:
                answers.append(gtAnsDatum['answer'])
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']
            # avgGTAcc = compute_acc(a_golds=answers, a_pred=resAns, lang='zh')
            avgGTAcc = compute_acc_en_ch(a_golds=answers, a_pred=resAns)
            accQA.append(avgGTAcc)

            qwen_score = qwen_evaluate(resAns, answers, 1000)
            qwen_scores.append(qwen_score)

            if avgGTAcc > 0:  # Check if the answer is correct
                correct_ques_ids.append(quesId)  # Store the question ID of correct answers

            if quesType not in accQuesType:
                accQuesType[quesType] = []
                qwen_scoreQuesType[quesType] = []

            accQuesType[quesType].append(avgGTAcc)
            qwen_scoreQuesType[quesType].append(qwen_score)

            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 20 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        # breakpoint()
        
        self.setAccuracy(accQA, accQuesType, accAnsType)
        # print('Done computing accuracy')
        # print('Question IDs with correct answers:', correct_ques_ids)

        # print('Qwen score:', mean(qwen_scores)*100)
        for type_ in qwen_scoreQuesType:
            qwen_scoreQuesType[type_] = mean(qwen_scoreQuesType[type_])*100
        print(qwen_scoreQuesType)

        return {'acc': accQA,
                'qwen_score': qwen_scores
        }

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p
                    in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub('', outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall'] = round(100 * float(sum(accQA)) / len(accQA),
                                         self.n)
        self.accuracy['perQuestionType'] = {
            quesType: round(
                100 * float(sum(accQuesType[quesType])) /
                len(accQuesType[quesType]),
                self.n,
            )
            for quesType in accQuesType
        }
        self.accuracy['perAnswerType'] = {
            ansType: round(
                100 * float(sum(accAnsType[ansType])) /
                len(accAnsType[ansType]), self.n)
            for ansType in accAnsType
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ''
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = 'error: progress var must be float\r\n'
        if progress < 0:
            progress = 0
            status = 'Halt...\r\n'
        if progress >= 1:
            progress = 1
            status = 'Done...\r\n'
        block = int(round(barLength * progress))
        text = '\rFinshed Percent: [{0}] {1}% {2}'.format(
            '#' * block + '-' * (barLength - block), int(progress * 100),
            status)
        sys.stdout.write(text)
        sys.stdout.flush()


    def evaluate_ub_en(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            # for small data
            if quesId not in self.vqa.qa.keys() or quesId not in self.vqaRes.qa.keys():
                continue
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}
        correct_ques_ids = []
        print('computing accuracy')

        qwen_scores = []
        step = 0
        for quesId in quesIds:
            # for small data
            if quesId not in res.keys():
                continue
            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(
                        str(ansDic['answer']))

            answers = []
            for gtAnsDatum in gts[quesId]['answers']:
                answers.append(gtAnsDatum['answer'])
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']

            avgGTAcc = compute_acc(a_golds=answers, a_pred=resAns, lang='en')
            accQA.append(avgGTAcc)
            
            try:
                qwen_score = qwen_evaluate(resAns, answers)
            except Exception as e:
                breakpoint()
                print(e)
                qwen_score = 0.

            qwen_scores.append(qwen_score)

        return {
            'acc': accQA,
            'qwen_score': qwen_scores
        }

    def evaluate_ub(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params['question_id']]
        gts = {}
        res = {}
        for quesId in quesIds:
            # for small data
            if quesId not in self.vqa.qa.keys() or quesId not in self.vqaRes.qa.keys():
                continue
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}
        correct_ques_ids = []
        print('computing accuracy')

        qwen_scores = []

        step = 0
        for quesId in quesIds:
            # for small data
            if quesId not in res.keys():
                continue
            resAns = res[quesId]['answer']
            # breakpoint()
            # print(resAns)

            if resAns is not None:
                resAns = resAns.replace('\n', ' ')
            else:
                print(f"resAns:{resAns}")
                continue
                # resAns = '没有回答'

            # resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    # print(ansDic['answer'])
                    ansDic['answer'] = self.processPunctuation(
                        str(ansDic['answer']))
           
            answers = []
            for gtAnsDatum in gts[quesId]['answers']:
                answers.append(gtAnsDatum['answer'])
            quesType = gts[quesId]['question_type']
            ansType = gts[quesId]['answer_type']
            avgGTAcc = compute_acc(a_golds=answers, a_pred=resAns, lang='zh')
            accQA.append(avgGTAcc)

            qwen_score = qwen_evaluate(resAns, answers)
            qwen_scores.append(qwen_score)

        return {
            'acc': accQA,
            'qwen_score': qwen_scores
        }
