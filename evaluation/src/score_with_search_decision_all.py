import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional
import torch
from tqdm import tqdm
from numpy import mean, std
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

import sys
from vqa import VQA
from vqa_eval import VQAEval
from utils import extract_questions
from utils import generate_and_reorder_annotation_file
from utils import get_filename_without_extension


def _file_exists_check(paths:list):
    err = 0
    for path_ in paths:
        if not os.path.exists(path_):
            print(path_, 'Not exists!')
            err = 1
    if err:
        exit(-1)
        # raise FileNotFoundError
    

def _random_sample(a, b, p):
    '''
    Randomly sample 'a' with probability p, and 'b' with (1-p)
    '''
    probabilities = [p, 1 - p]
    
    # Sampling one item based on the defined probabilities
    sampled_item = random.choices([a, b], weights=probabilities, k=1)[0]
    
    return sampled_item

def _count_rag_ratio(search_decision_file, args):
    rag_cnt, no_rag_cnt, err_output = 0.,0.,0.
    with open(search_decision_file) as f:
        lines = f.readlines()
    for line in lines:
        s_d_dict = json.loads(line)

        if s_d_dict['search_decision'] == 'true':
            rag_cnt += 1
        elif s_d_dict['search_decision'] == 'false':
            no_rag_cnt += 1

        elif type(s_d_dict['search_decision']) is float:
            if s_d_dict['search_decision'] >= args.rag_threshold:
                rag_cnt += 1
            else:
                no_rag_cnt += 1
        
        else:
            no_rag_cnt += 1

    return rag_cnt/len(lines)


def combin_metric_randomly(metric_dict_rag, metric_dict_norag, search_decision_file, args):
    '''
    metric_dict_(no)rag: dict(). {'acc': [...], 'qwen_score': [...]}

    Return
    dict(): e.g. {'acc': 0.9, 'qwen_score': 0.8}
    '''
    rag_ratio = _count_rag_ratio(search_decision_file, args)
    new_metric = {k: None for k in metric_dict_rag}

    for k in metric_dict_rag:
        combined = []
        for i, j in zip(metric_dict_rag[k], metric_dict_norag[k]):
            combined.append(_random_sample(i, j, rag_ratio))
        new_metric[k] = mean(combined)
    return new_metric

def combin_metric(metric_dict_rag, metric_dict_norag, search_decision_list, args, random=False):
    '''
    metric_dict_(no)rag: dict(). {'acc': [...], 'qwen_score': [...]}
    search_decision_list: list of 0/1 0=no rag; 1=rag

    Return
    dict(): e.g. {'acc': 0.9, 'qwen_score': 0.8}
    '''
    if random:
        rag_ratio = float(sum(search_decision_list))/len(search_decision_list)

    new_metric = {k: None for k in metric_dict_rag}

    for k in metric_dict_rag:
        combined = []
        for i, j, s in zip(metric_dict_rag[k], metric_dict_norag[k], search_decision_list):
            if not random:
                if s==0: # no rag
                    combined.append(j)
                elif s==1:
                    combined.append(i)
                else:
                    print(s, '? Wrong decision')
                    exit(-1)
            elif random:
                combined.append(_random_sample(i, j, rag_ratio))
        new_metric[k] = mean(combined)
    return new_metric
    

def get_search_decision(search_decision_file, args):
    '''
    Return: a list of 0/1 of length search_decision_file.
    0 = no rag on this sample
    1 = rag on this sample
    '''

    ret = []
    with open(search_decision_file) as f:
        search_decision = f.readlines()
        search_decision = [json.loads(i) for i in search_decision]
    
    err_output = 0
    for s_d in search_decision:
        if s_d['search_decision'] == 'true' or s_d['search_decision'] is True:
            ret.append(1)
        elif s_d['search_decision'] == 'false' or s_d['search_decision'] is False:
            ret.append(0)

        elif type(s_d['search_decision']) is float:
            if 5 >= s_d['search_decision'] >= args.rag_threshold:
                ret.append(1)
            elif args.rag_threshold > s_d['search_decision'] >= 1:
                ret.append(0)
        
        else:
            err_output += 1
            _s = s_d['search_decision']
            print(f"Warning: search decision of wrong format: {_s}")
            print(f"Default to false")
            ret.append(0)
    return ret


def combine_results_file(results_file_rag, results_file_no_rag, search_decision_file, args):
    
    _file_exists_check([results_file_rag, results_file_no_rag, search_decision_file])

    if args.search_decision == 'all_rag':
        return results_file_rag
    elif args.search_decision == 'no_rag':
        return results_file_no_rag
    
    # search decision or random
    rag_ratio = _count_rag_ratio(search_decision_file, args)

    with open(results_file_rag) as f1, open(results_file_no_rag) as f2, open(search_decision_file) as f3:
        res_rag = f1.readlines()
        res_no_rag = f2.readlines()
        search_decision = f3.readlines()

    combined_according_to_search_decision = []
    rag_cnt, no_rag_cnt, err_output = 0,0,0

    if len(res_rag) == len(res_no_rag) == len(search_decision):
        for line_rag, line_no_rag, s_d in zip(res_rag, res_no_rag, search_decision):
            s_d_dict = json.loads(s_d)
            # breakpoint()
            if s_d_dict['search_decision'] == 'true':
                rag_cnt += 1
                selected = line_rag if args.search_decision == 'default' else _random_sample(line_rag, line_no_rag, rag_ratio)
            elif s_d_dict['search_decision'] == 'false':
                no_rag_cnt += 1
                selected = line_no_rag if args.search_decision == 'default' else _random_sample(line_rag, line_no_rag, rag_ratio)

            elif type(s_d_dict['search_decision']) is float:
                if 5 >= s_d_dict['search_decision'] >= args.rag_threshold:
                    rag_cnt += 1
                    selected = line_rag if args.search_decision == 'default' else _random_sample(line_rag, line_no_rag, rag_ratio)
                elif args.rag_threshold > s_d_dict['search_decision'] >= 1:
                    no_rag_cnt += 1
                    selected = line_no_rag if args.search_decision == 'default' else _random_sample(line_rag, line_no_rag, rag_ratio)
            
            else:
                err_output += 1
                _s = s_d_dict['search_decision']
                print(f"Warning: search decision not true or false: {_s}")
                print(f"Default to false")
                selected = line_no_rag

            combined_according_to_search_decision.append(selected)
    else:
        print("Different data len: ")
        print('({}) {}'.format(len(res_no_rag), results_file_no_rag))
        print('({}) {}'.format(len(res_rag), results_file_rag))
        print('({}) {}'.format(len(search_decision), search_decision_file))
        exit(-1)
    
    print(f"RAG cnt: {rag_cnt}/{len(res_rag)}")
    print(f"No RAG cnt: {no_rag_cnt}/{len(res_rag)}")
    print(f"Rag Threshold: {args.rag_threshold}")
    print(f"Error output (-> No RAG) cnt: {err_output}/{len(res_rag)}")

    tmp_hash = hash(results_file_rag+results_file_no_rag+search_decision_file)
    tmp_path_str = f'/mnt/nas-alinlp/zhuochen.zc/tmp/{tmp_hash}.jsonl'
    with open(tmp_path_str, 'w') as g:
        for i in combined_according_to_search_decision:
            g.writelines(i.strip()+'\n')
    return tmp_path_str
    
def compute_upper_bound(list_1, list_2, lowest_correct_threshold=0.5):
    '''
    list_1/2: list of float. list_1 from all rag, list_2 from no rag
    Return
    new list of float
    '''
    ret = []
    rag_cnt = 0
    for i, j in zip(list_1, list_2):
        # upper bound of (llm + search engine). No reference value
        # if i>j:
        #     rag_cnt += 1
        #     ret.append(i)
        # else:
        #     ret.append(j)

        if j<=lowest_correct_threshold:
            # deem as wrong answer
            ret.append(i)
            rag_cnt += 1
        else:
            # corrent without rag
            ret.append(j)

    return ret, rag_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--results_file_rag', type=str, default='')
    parser.add_argument('--results_file_no_rag', type=str, default='')

    parser.add_argument('--search_decision_file_hard', type=str, default='')
    parser.add_argument('--search_decision_file_soft', type=str, default='')
    parser.add_argument('--search_decision_file_human', type=str, default='')
    parser.add_argument('--search_decision_file_prompt', type=str, default='')
    
    parser.add_argument('--search_decision', type=str, default='default', choices=['default', 'all_rag', 'no_rag', 'random'])
    parser.add_argument('--rag_threshold', type=float, default=4.0)

    parser.add_argument('--metric', type=str, default='qwen_score')
    parser.add_argument('--lang', type=str, default=None, choices=['en', 'ch'])
    args = parser.parse_args()

    if args.lang is None:
        print('Specify --lang!')
        exit(-1)


    LIMIT = 1000000000000 # limit tokens
    random.seed(args.seed)


    anno_path = "./anno_storage/"
    q_path = "./q_storage/"
    os.makedirs(anno_path, exist_ok=True) 
    os.makedirs(q_path, exist_ok=True) 


    # breakpoint()
    print(f"evaluating data from {args.results_file_rag}")
    data_name = get_filename_without_extension(args.results_file_rag)
    results_file_rag = args.results_file_rag
    metric = args.metric
    lang = args.lang

    annotation_file = anno_path+f"{data_name}.json"
    annotation = generate_and_reorder_annotation_file(args.results_file_rag, annotation_file)

    question_file = q_path+f"{data_name}.json"
    question = extract_questions(args.results_file_rag, question_file)


    # rag scores
    vqa = VQA(annotation, question)
    results = vqa.loadRes(
        resFile=args.results_file_rag,
        quesFile=question)
    vqa_scorer = VQAEval(vqa, results, n=2)
    if lang == 'ch':
        result_rag = vqa_scorer.evaluate()
    elif lang == 'en':
        result_rag = vqa_scorer.evaluate_en()

    # no rag scores
    vqa = VQA(annotation, question)
    results = vqa.loadRes(
        resFile=args.results_file_no_rag,
        quesFile=question)
    vqa_scorer = VQAEval(vqa, results, n=2)
    if lang == 'ch':
        result_norag = vqa_scorer.evaluate()
    elif lang == 'en':
        result_norag = vqa_scorer.evaluate_en()
    # result_norag = {'acc': [...],
    #     'qwen_score': [...]
    # }

    final_print_acc = []
    final_print_qwen_score = []

    print("\nUpper bound:")
    acc_ub, rag_cnt = compute_upper_bound(result_rag['acc'], result_norag['acc'])
    print('Acc ub:', 100*mean(acc_ub))
    print(f"Acc rag ratio: {rag_cnt}/{len(result_rag['acc'])}")
    final_print_acc.append(mean(acc_ub)*100)
    final_print_acc.append(f"={rag_cnt}/{len(result_rag['acc'])}")

    score_ub, rag_cnt = compute_upper_bound(result_rag['qwen_score'], result_norag['qwen_score'])
    print('qwen_scores ub:', 100*mean(score_ub))
    print(f"qwen_scores rag ratio: {rag_cnt}/{len(result_rag['qwen_score'])}")
    print('='*50+'\n')
    final_print_qwen_score.append(mean(score_ub)*100)
    final_print_qwen_score.append(f"={rag_cnt}/{len(result_rag['qwen_score'])}")

    print("No rag")
    print('Acc:', mean(result_norag['acc']))
    print('qwen_score:', mean(result_norag['qwen_score']))
    final_print_acc.append(mean(result_norag['acc'])*100)
    final_print_qwen_score.append(mean(result_norag['qwen_score'])*100)

    print("All rag")
    print('Acc:', mean(result_rag['acc']))
    print('qwen_score:', mean(result_rag['qwen_score']))
    final_print_acc.append(mean(result_rag['acc'])*100)
    final_print_qwen_score.append(mean(result_rag['qwen_score'])*100)
    print('='*50+'\n')


    print('(Qwen-VL-Chat) Prompt')
    search_decision_prompt = get_search_decision(args.search_decision_file_prompt, args)
    result_prompt = combin_metric(result_rag, result_norag, search_decision_prompt, args, random=False)
    print('Acc:', result_prompt['acc'])
    print('qwen_score:', result_prompt['qwen_score'])
    final_print_acc.append(mean(result_prompt['acc'])*100)
    final_print_acc.append(f"={sum(search_decision_prompt)}/{len(search_decision_prompt)}")
    final_print_qwen_score.append(mean(result_prompt['qwen_score'])*100)
    final_print_qwen_score.append(f"={sum(search_decision_prompt)}/{len(search_decision_prompt)}")
    print('='*50+'\n')


    print('Human')
    search_decision_human = get_search_decision(args.search_decision_file_human, args)
    result_human = combin_metric(result_rag, result_norag, search_decision_human, args, random=False)
    print('Acc:', result_human['acc'])
    print('qwen_score:', result_human['qwen_score'])
    final_print_acc.append(mean(result_human['acc'])*100)
    final_print_acc.append(f"={sum(search_decision_human)}/{len(search_decision_human)}")
    final_print_qwen_score.append(mean(result_human['qwen_score'])*100)
    final_print_qwen_score.append(f"={sum(search_decision_human)}/{len(search_decision_human)}")
    print('='*50+'\n')


    print('Hard & random')
    search_decision_hard = get_search_decision(args.search_decision_file_hard, args)
    result_hard = combin_metric(result_rag, result_norag, search_decision_hard, args, random=False)
    print('Acc:', result_hard['acc'])
    print('qwen_score:', result_hard['qwen_score'])
    final_print_acc.append(result_hard['acc']*100)
    final_print_qwen_score.append(result_hard['qwen_score']*100)
    final_print_acc.append(f"={sum(search_decision_hard)}/{len(search_decision_hard)}")
    final_print_qwen_score.append(f"={sum(search_decision_hard)}/{len(search_decision_hard)}")

    metric_over_seed = {
        'acc': [], 'qwen_score': []
    }
    for seed in [0, 42, 1988, 30, 4321]:
        random.seed(seed)
        result_hard_random = combin_metric(result_rag, result_norag, search_decision_hard, args, random=True)
        for k in metric_over_seed:
            metric_over_seed[k].append(result_hard_random[k])
    mean_acc = mean(metric_over_seed['acc'])
    std_acc = std(metric_over_seed['acc'])
    mean_qwen_score = mean(metric_over_seed['qwen_score'])
    std_qwen_score = std(metric_over_seed['qwen_score'])

    print('Acc random: {} std: {}'.format(mean_acc, std_acc))
    print('qwen_score random: {} std: {}'.format(mean_qwen_score, std_qwen_score))
    print('='*50+'\n')
    final_print_acc.extend([mean_acc*100, std_acc*100])
    final_print_qwen_score.extend([mean_qwen_score*100, std_qwen_score*100])

    print('Soft & random')
    search_decision_soft = get_search_decision(args.search_decision_file_soft, args)
    result_soft = combin_metric(result_rag, result_norag, search_decision_soft, args, random=False)
    print('Acc:', result_soft['acc'])
    print('qwen_score:', result_soft['qwen_score'])
    final_print_acc.append(result_soft['acc']*100)
    final_print_qwen_score.append(result_soft['qwen_score']*100)
    final_print_acc.append(f"={sum(search_decision_soft)}/{len(search_decision_soft)}")
    final_print_qwen_score.append(f"={sum(search_decision_soft)}/{len(search_decision_soft)}")

    metric_over_seed = {
        'acc': [], 'qwen_score': []
    }
    for seed in [0, 42, 1988, 30, 4321]:
        random.seed(seed)
        result_soft_random = combin_metric(result_rag, result_norag, search_decision_soft, args, random=True)
        for k in metric_over_seed:
            metric_over_seed[k].append(result_soft_random[k])
    mean_acc = mean(metric_over_seed['acc'])
    std_acc = std(metric_over_seed['acc'])
    mean_qwen_score = mean(metric_over_seed['qwen_score'])
    std_qwen_score = std(metric_over_seed['qwen_score'])

    print('Acc random: {} std: {}'.format(mean_acc, std_acc))
    print('qwen_score random: {} std: {}'.format(mean_qwen_score, std_qwen_score))
    print('='*50+'\n')
    final_print_acc.extend([mean_acc*100, std_acc*100])
    final_print_qwen_score.extend([mean_qwen_score*100, std_qwen_score*100])

    # now = datetime.now()
    # now = now.strftime("%Y_%m_%d_%H_%M_%S")
    # with open(f'/mnt/nas-alinlp/zhuochen.zc/others/MMRAG/Release_Codes/out/{now}.txt', 'w') as g:
    #     g.write('\t'.join([str(i) for i in final_print_acc]))
    #     g.write('\n')
    #     g.write('\t'.join([str(i) for i in final_print_qwen_score]))

    # print('\t'.join([str(i) for i in final_print_acc]))
    # print('\t'.join([str(i) for i in final_print_qwen_score]))
    # print(f'/mnt/nas-alinlp/zhuochen.zc/others/MMRAG/Release_Codes/out/{now}.txt')
    # exit()

