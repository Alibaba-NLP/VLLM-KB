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
import unicodedata
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

# METRICS = ['em', 'token_acc', 'bleu', 'rouge_l', 'token_f1', 'char_level']
METRICS = ['em', 'token_acc', 'token_f1']

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


def _is_chinese(text: str) -> bool:
    """通过检查是否存在中文字符来判断文本是否包含中文"""
    for char in text:
        # CJK Unified Ideographs range
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def normalize_text(text: str) -> str:
    """
    全面文本标准化处理，自动支持中英文。
    - 对于英文：转小写，移除标点，保留字母和数字。
    - 对于中文/混合文：标准化，移除标点，并在中文字符周围添加空格以便分词。
    """
    # 步骤 1: 统一化处理（对中英文都有效）
    # 转换为NFKC范式，处理全角/半角字符等
    text = unicodedata.normalize('NFKC', text)
    text = text.lower() # 转换为小写，对中文无影响，对英文有效

    if _is_chinese(text):
        # 步骤 2: 中文或中英混合文本处理
        # 移除标点，保留中文字符、英文字母和数字
        # \u4e00-\u9fa5: 中文字符范围
        # a-zA-Z0-9: 字母和数字
        # \s: 空白字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        # 在中文字符周围添加空格，实现基于字符的分词
        text = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', text)
    else:
        # 步骤 2: 纯英文文本处理
        # 移除标点，仅保留字母、数字和空格
        text = re.sub(r'[^a-z0-9\s]', '', text)

    # 步骤 3: 移除多余的空白
    text = ' '.join(text.split())
    
    return text.strip()

def _tokenize(text: str) -> List[str]:
    """内部使用的分词函数，先标准化再切分"""
    return normalize_text(text).split()

def exact_match(pred: str, gold: str) -> int:
    """精确匹配（EM）"""
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    return int(pred_norm == gold_norm)

def token_f1(pred: str, gold: str) -> float:
    """词元F1分数"""
    pred_tokens = set(_tokenize(pred))
    gold_tokens = set(_tokenize(gold))
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
        
    tp = len(pred_tokens & gold_tokens)
    
    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# 注意：token_accuracy 的定义可能存在争议，这里我们计算重叠词元数占标准答案词元数的比例（召回率）
def token_accuracy(pred: str, gold: str) -> float:
    """词元准确率（此处实现为词元召回率）"""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
        
    common_tokens = set(pred_tokens) & set(gold_tokens)
    
    # 使用 multiset 的思想，计算每个共同词元在标准答案中出现的次数
    gold_counts = {t: gold_tokens.count(t) for t in common_tokens}
    pred_counts = {t: pred_tokens.count(t) for t in common_tokens}
    
    correct = sum(min(gold_counts[t], pred_counts[t]) for t in common_tokens)
    
    return correct / len(gold_tokens)

def bleu_score(pred: str, gold: str) -> float:
    """BLEU-4 分数"""
    pred_tokens = _tokenize(pred)
    gold_tokens_list = [_tokenize(gold)] # sentence_bleu需要一个引用列表
    
    if not pred_tokens:
        return 0.0
        
    smoothie = SmoothingFunction().method4
    return sentence_bleu(gold_tokens_list, pred_tokens, smoothing_function=smoothie)

def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L 分数"""
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if not pred_norm or not gold_norm:
        return 0.0

    rouge = Rouge()
    try:
        scores = rouge.get_scores(pred_norm, gold_norm, avg=False)
        return scores[0]['rouge-l']['f']
    except (ValueError, KeyError): # 处理空字符串等异常
        return 0.0

def character_level_metrics(pred: str, gold: str) -> Dict[str, float]:
    """字符级指标"""
    # 注意：这里的标准化移除了空格和标点，用于纯粹的字符内容比较
    pred_clean = re.sub(r'\s+', '', normalize_text(pred))
    gold_clean = re.sub(r'\s+', '', normalize_text(gold))
    
    char_exact_match = int(pred_clean == gold_clean)
    char_similarity = SequenceMatcher(None, pred_clean, gold_clean).ratio()
    
    try:
        import Levenshtein
        edit_distance = Levenshtein.distance(pred_clean, gold_clean)
        max_len = max(len(pred_clean), len(gold_clean)) or 1
        normalized_edit_sim = 1 - (edit_distance / max_len)
    except ImportError:
        normalized_edit_sim = char_similarity  # 如果未安装Levenshtein，则回退到difflib
    
    return {
        'char_exact_match': float(char_exact_match),
        'char_similarity': char_similarity,
        'normalized_edit_sim': normalized_edit_sim
    }

def evaluate_vqa(pred: str, gold: str, 
                 metrics: List[str] = ['em', 'token_f1', 'bleu', 'rouge_l', 'char_level']) -> Dict[str, Union[int, float]]:
    """
    对单个预测-标准答案对进行多维度评估，自动支持中英文。

    Args:
        pred (str): 模型的预测答案。
        gold (str): 标准答案（Ground Truth）。
        metrics (List[str]): 需要计算的指标列表。
            可选值: 'em', 'token_f1', 'token_acc', 'bleu', 'rouge_l', 'char_level'。

    Returns:
        Dict[str, Union[int, float]]: 包含所选指标分数的字典。
    """
    results = {}
    
    metric_functions = {
        'em': exact_match,
        'token_f1': token_f1,
        'token_acc': token_accuracy,
        'bleu': bleu_score,
        'rouge_l': rouge_l,
    }

    for metric in metrics:
        if metric in metric_functions:
            results[metric] = metric_functions[metric](pred, gold)
        elif metric == 'char_level':
            results.update(character_level_metrics(pred, gold))
    
    return results


def test():
    test_cases = [
        ("A cat\non the mat.", "a cat on the mat"),  # 换行和标点测试
        ("Café", "cafe\u0301"),  # Unicode测试
        ("  extra   spaces  ", "extra spaces"),  # 空格测试
        ("The quick brown fox", "The slow brown fox"),  # 内容差异
        ("", ""),  # 空输入
    ]

    print("全面字符级处理测试:")
    for pred, gold in test_cases:
        scores = evaluate_vqa(pred, gold)
        print(f"\nPred: {repr(pred)}")
        print(f"Gold: {repr(gold)}")
        print("标准化后:", repr(normalize_text(pred)), "vs", repr(normalize_text(gold)))
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")


if __name__ == '__main__':
    # test()

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', default='')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--overwrite', type=int, default=0)
    # parser.add_argument('--error_path', default='')

    args = parser.parse_args()

    if args.output_path == '':
        args.output_path = args.data_path.replace('.jsonl', '_static_eval.jsonl')
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
        for i, data in enumerate(tqdm(data_, desc='static scoring...', ncols=100)):
            
            if i < num_lines_finished:
                continue

            new_data = copy.deepcopy(data)

            # query = data['question']
            answers = data['answer'] # str or list, both handled
            pred = data['response']
            
            pred = '' if not pred else pred
            answers = '' if not answers else answers

            if isinstance(answers, str):
                new_data['static_score'] = evaluate_vqa(pred, answers, metrics=METRICS)
            elif isinstance(answers, list):
                answers = [str(a) for a in answers]
                max_static_score = {k: 0.0 for k in METRICS}
                for a in answers:
                    temp_score_dict = evaluate_vqa(pred, a, metrics=METRICS)
                    for k, v in temp_score_dict.items():
                        max_static_score[k] = max(max_static_score[k], v)
                new_data['static_score'] = max_static_score

            g.writelines(json.dumps(new_data, ensure_ascii=False)+'\n')
