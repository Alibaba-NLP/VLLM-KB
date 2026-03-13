import json
from tqdm import tqdm
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--both', default='w_image_search',
                              help='Both image and text search setting.', 
                              choices=['w_image_search', 'wo_image_search'])
args = parser.parse_args()

models = [
    'omnisearch-gpt4o'
]

datasets = [
    'lifevqa', 
    'qwenvqa', 
    'vfreshqa_ch',
    'vfreshqa_en',
    # 'visual7w',
]

settings = [
    'norag', 
    # 'image2image', 
    # 'text2text_gold_query_from_origin_query_image',
    # 'both',
    # 'prompt',
    # 'ours',
]


def combine_metric_by_search_decision(dataset, model, setting):
    score, cnt = 0., 0
    norag_cnt, i2i_cnt, t2t_cnt, both_cnt = 0.,0.,0.,0.
    global search_decision_file

    if setting == 'ours':
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0403_KB2_model.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0408_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0421_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0503_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0503_full_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0504_even_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0505_even_KB2.jsonl"
        
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0518_lora32_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0519_full_KB2.jsonl"

        # More training data
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0526_0526_more_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0527_more_full_h20_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0527_more_full_KB2.jsonl"

        # Strict even each=25k
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0528_strict_even_lora_KB2.jsonl" # -> might over-fit to label:image_search data
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0528_strict_even_full_KB2.jsonl" # -> might over-fit to label:image_search data

        search_decision_file = f"{dataset}/norag/{dataset}_cz_0529_more_dev2_full_KB2.jsonl"

    else:
        search_decision_file = f"{dataset}/norag/{dataset}_cz_prompt_based.jsonl"

    norag_file = f"{dataset}/norag/{dataset}_{model}_qwen_max_eval.jsonl"
    image2image_file = f"{dataset}/image2image/{dataset}_{model}_qwen_max_eval.jsonl"
    
    text2text_gq_from_origin_query_image_file = f"{dataset}/text2text_gold_query_from_origin_query_image/{dataset}_{model}_qwen_max_eval.jsonl"
    text2text_gq_from_image_search_file = f"{dataset}/text2text_gold_query_from_image_search/{dataset}_{model}_qwen_max_eval.jsonl"

    both_file = f"{dataset}/both/{dataset}_{model}_qwen_max_eval.jsonl"

    if 'vfreshqa' in dataset:
        image2image_file = f"{dataset}/image2image/{dataset}_bing_{model}_qwen_max_eval.jsonl"
        # text2text_gq_from_origin_query_image_file = f"{dataset}/text2text_gold_query_from_origin_query_image/{dataset}_google_{model}_qwen_max_eval.jsonl"
        # text2text_gq_from_image_search_file = f"{dataset}/text2text_gold_query_from_image_search/{dataset}_google_{model}_qwen_max_eval.jsonl"

    with open(search_decision_file) as f1, \
        open(norag_file) as f2, \
        open(image2image_file) as f3, \
        open(text2text_gq_from_origin_query_image_file) as f4, \
        open(text2text_gq_from_image_search_file) as f5, \
        open(both_file) as f6:
        
        search_decision_lines = list(f1.readlines())
        norag_lines = list(f2.readlines())
        image2image_lines = list(f3.readlines())
        text2text_gq_from_origin_query_image_lines = list(f4.readlines())
        text2text_gq_from_image_search_lines = list(f5.readlines())
        both_lines = list(f6.readlines())

        if not len(search_decision_lines)==len(norag_lines)==len(image2image_lines)==len(text2text_gq_from_origin_query_image_lines)==len(text2text_gq_from_image_search_lines)==len(both_lines):
            print('Different #lines', len(search_decision_lines),len(norag_lines),len(image2image_lines),len(text2text_gq_from_origin_query_image_lines),len(text2text_gq_from_image_search_lines),len(both_lines))
            print(text2text_gq_from_origin_query_image_file)
            return -1,-1,-1,-1,-1

        for sd, norag, i2i, t2t_o, t2t_i, both in zip(search_decision_lines, norag_lines, image2image_lines, text2text_gq_from_origin_query_image_lines, text2text_gq_from_image_search_lines, both_lines):
            sd = json.loads(sd)
            norag = json.loads(norag)
            i2i = json.loads(i2i)
            t2t_o = json.loads(t2t_o)
            t2t_i = json.loads(t2t_i)
            both = json.loads(both)
            if sd['search_decison'].startswith('A'):
                score += norag['qwen_max_score']
                norag_cnt += 1
            elif sd['search_decison'].startswith('B'):
                score += i2i['qwen_max_score']
                i2i_cnt += 1
            elif sd['search_decison'].startswith('C'):
                # breakpoint()
                score += t2t_o['qwen_max_score']
                t2t_cnt += 1
            elif sd['search_decison'].startswith('D'):

                if args.both == 'wo_image_search':
                    score += t2t_i['qwen_max_score']
                elif args.both == 'w_image_search':
                    score += both['qwen_max_score']
                else:
                    raise ValueError('--both wrong')
                both_cnt += 1

            else:
                tqdm.write(f'Error alignment output. Default to norag. {dataset}, {model}, {setting}')
                score += norag['qwen_max_score']
                norag_cnt += 1
            cnt += 1
    # breakpoint()
    return score/cnt, norag_cnt/cnt, i2i_cnt/cnt, t2t_cnt/cnt, both_cnt/cnt


scores_print = ''

for dataset in tqdm(datasets, ncols=50):
    for model in models:
        # if model == 'chatgpt-4o-latest':
        #     scores_print += '\n'
        #     continue

        for setting in settings:
            if setting in ['ours', 'prompt']:
                score, norag_ratio, i2i_ratio, t2t_ratio, both_ratio = combine_metric_by_search_decision(dataset, model, setting)
                score = "{:.2f}".format(score/5*100)
                norag_ratio = "{:.2f}%".format(norag_ratio*100)
                i2i_ratio = "{:.2f}%".format(i2i_ratio*100)
                t2t_ratio = "{:.2f}%".format(t2t_ratio*100)
                both_ratio = "{:.2f}%".format(both_ratio*100)

                scores_print += score+'\t'+norag_ratio+'\t'+i2i_ratio+'\t'+t2t_ratio+'\t'+both_ratio+'\t'
            else:
                score_file_path = f"{dataset}/{setting}/{dataset}_{model}_qwen_max_eval.jsonl"
                
                if 'vfreshqa' in dataset:
                    if setting == 'image2image':
                        score_file_path = f"{dataset}/{setting}/{dataset}_bing_{model}_qwen_max_eval.jsonl"

                if setting == 'both' and args.both == 'wo_image_search':
                    score_file_path = f"{dataset}/text2text_gold_query_from_image_search/{dataset}_{model}_qwen_max_eval.jsonl"
                    # print("Both without image search.")

                score = 0.
                with open(score_file_path) as f:
                    lines = list(f.readlines())
                    for line in lines:
                        data = json.loads(line)
                        score += data['qwen_max_score']
                    score /= len(lines)
            
                score = "{:.2f}".format(score/5*100)
                scores_print += score+'\t'
        
        scores_print += '\n'

result_txt = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/log/result_cp2excel_omni_{args.both}.txt'
with open(result_txt, 'w') as g:
    g.write(scores_print)
    print(result_txt)