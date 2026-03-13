import json
from tqdm import tqdm
import argparse
import os
import random
random.seed(30)
from numpy import mean


parser = argparse.ArgumentParser()
parser.add_argument('--both', default='w_image_search',
                              help='Both image and text search setting.', 
                              choices=['w_image_search', 'wo_image_search'])
parser.add_argument('--score_model', default='qwen_max', help='Models used to score the correctness')

args = parser.parse_args()

models = [
    'qwen25-vl-7b',
    # 'qwen2.5-vl-7b-kb',
    'deepseek-vl-7b-chat',
    # 'deepseek-vl-7b-chat-2',
    'chatgpt-4o-latest',
    # 'qwen-vl-max',
    # 'qwen-vl-max-latest',
]

datasets = [
    'lifevqa', 
    'qwenvqa', 
    # 'vfreshqa_ch',
    # 'vfreshqa_en',
    # 'visual7w',
    'nocaps'
]

settings = [
    'norag', 
    'image2image', 
    'text2text_gold_query_from_origin_query_image',
    'both',
    'prompt',
    'ours',
]

sampled_mix = {
    d: {
        m: {
            s: [] for s in settings
        } for m in models
    } for d in datasets+['mix']}
sampled_size = 100

if args.score_model == 'gpt4o':
    score_key_name = f"gpt_4o_score"
else:
    score_key_name = f"{args.score_model}_score"


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
        search_decision_file = f"{dataset}/norag/{dataset}_cz_0526_0526_more_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0527_more_full_h20_KB2.jsonl"
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0527_more_full_KB2.jsonl"

        # Strict even each=25k
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0528_strict_even_lora_KB2.jsonl" # -> might over-fit to label:image_search data
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0528_strict_even_full_KB2.jsonl" # -> might over-fit to label:image_search data

        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0529_more_dev2_full_KB2.jsonl"

        # RAG boundary
        # search_decision_file = f"{dataset}/norag/{dataset}_cz_0628_0626_ragb_lora32_RAG_KB2.jsonl"

    else:
        search_decision_file = f"{dataset}/norag/{dataset}_cz_prompt_based.jsonl"

    norag_file = f"{dataset}/norag/{dataset}_{model}_{args.score_model}_eval.jsonl"
    image2image_file = f"{dataset}/image2image/{dataset}_{model}_{args.score_model}_eval.jsonl"
    
    text2text_gq_from_origin_query_image_file = f"{dataset}/text2text_gold_query_from_origin_query_image/{dataset}_{model}_{args.score_model}_eval.jsonl"
    text2text_gq_from_image_search_file = f"{dataset}/text2text_gold_query_from_image_search/{dataset}_{model}_{args.score_model}_eval.jsonl"

    both_file = f"{dataset}/both/{dataset}_{model}_{args.score_model}_eval.jsonl"

    if 'vfreshqa' in dataset:
        image2image_file = f"{dataset}/image2image/{dataset}_bing_{model}_{args.score_model}_eval.jsonl"
        # text2text_gq_from_origin_query_image_file = f"{dataset}/text2text_gold_query_from_origin_query_image/{dataset}_google_{model}_{args.score_model}_eval.jsonl"
        # text2text_gq_from_image_search_file = f"{dataset}/text2text_gold_query_from_image_search/{dataset}_google_{model}_{args.score_model}_eval.jsonl"

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
                score += norag[score_key_name]
                norag_cnt += 1
                sampled_mix[dataset][model][setting].append((sd['search_decison'], norag[score_key_name]))
            
            elif sd['search_decison'].startswith('B'):
                score += i2i[score_key_name]
                i2i_cnt += 1
                sampled_mix[dataset][model][setting].append((sd['search_decison'], i2i[score_key_name]))
            
            elif sd['search_decison'].startswith('C'):
                # breakpoint()
                score += t2t_o[score_key_name]
                t2t_cnt += 1
                sampled_mix[dataset][model][setting].append((sd['search_decison'], t2t_o[score_key_name]))
            
            elif sd['search_decison'].startswith('D'):
                if args.both == 'wo_image_search':
                    score += t2t_i[score_key_name]
                elif args.both == 'w_image_search':
                    score += both[score_key_name]
                else:
                    raise ValueError('--both wrong')
                both_cnt += 1
                sampled_mix[dataset][model][setting].append((sd['search_decison'], both[score_key_name]))

            else:
                tqdm.write(f'Error alignment output. Default to norag. {dataset}, {model}, {setting}')
                score += norag[score_key_name]
                norag_cnt += 1
                sampled_mix[dataset][model][setting].append((sd['search_decison'], norag[score_key_name]))
            cnt += 1

    # breakpoint()
    return score/cnt, norag_cnt/cnt, i2i_cnt/cnt, t2t_cnt/cnt, both_cnt/cnt


scores_print = ''

for dataset in tqdm(datasets, ncols=50):
    for model in models:
        # if model == 'chatgpt-4o-latest':
        #     scores_print += '\n'
        #     continue

        if dataset in ['nocaps', 'visual7w'] and model == 'deepseek-vl-7b-chat-2':
            model = 'deepseek-vl-7b-chat'
        
        for setting in settings:
            if setting in ['ours', 'prompt']:
                score, norag_ratio, i2i_ratio, t2t_ratio, both_ratio = combine_metric_by_search_decision(dataset, model, setting)
                score = "{:.2f}".format(score/5*100)
                norag_ratio = "{:.1f}".format(norag_ratio*100)
                i2i_ratio = "{:.1f}".format(i2i_ratio*100)
                t2t_ratio = "{:.1f}".format(t2t_ratio*100)
                both_ratio = "{:.1f}".format(both_ratio*100)

                scores_print += score+'\t'+norag_ratio+'\t'+i2i_ratio+'\t'+t2t_ratio+'\t'+both_ratio+'\t'

            else:
                score_file_path = f"{dataset}/{setting}/{dataset}_{model}_{args.score_model}_eval.jsonl"
                
                if 'vfreshqa' in dataset:
                    if setting == 'image2image':
                        score_file_path = f"{dataset}/{setting}/{dataset}_bing_{model}_{args.score_model}_eval.jsonl"

                if setting == 'both' and args.both == 'wo_image_search':
                    score_file_path = f"{dataset}/text2text_gold_query_from_image_search/{dataset}_{model}_{args.score_model}_eval.jsonl"
                    # print("Both without image search.")

                score = 0.
                with open(score_file_path) as f:
                    lines = list(f.readlines())
                    for line in lines:
                        data = json.loads(line)
                        score += data[score_key_name]
                    score /= len(lines)
            
                score = "{:.2f}".format(score/5*100)
                scores_print += score+'\t'
                
                sampled_mix[dataset][model][setting] = [json.loads(i)[score_key_name] for i in lines]
        
        scores_print += '\n'


# Sample sample_size from each dataset and output scores
total_sample_size = 0
for dataset in datasets:
    indices = random.sample(list(range(len(sampled_mix[dataset][model][setting]))), sampled_size)
    for model in models:
        for setting in settings:
            for i in indices:
                sampled_mix['mix'][model][setting].append(sampled_mix[dataset][model][setting][i])
    
    total_sample_size += sampled_size


for model in models:
    for setting in settings:
        sampled_res = sampled_mix['mix'][model][setting]
        score, norag_cnt, i2i_cnt, t2t_cnt, both_cnt = 0., 0.,0.,0.,0.
        if setting in ['ours', 'prompt']:
            for res in sampled_res:
                if res[0].startswith('A.'):
                    norag_cnt += 1
                elif res[0].startswith('B.'):
                    i2i_cnt += 1
                elif res[0].startswith('C.'):
                    t2t_cnt += 1
                elif res[0].startswith('D.'):
                    both_cnt += 1
                else:
                    norag_cnt += 1
                score += res[1]
            
            score = "{:.2f}".format(score/5/total_sample_size*100)
            norag_ratio = "{:.1f}".format(norag_cnt/total_sample_size*100)
            i2i_ratio = "{:.1f}".format(i2i_cnt/total_sample_size*100)
            t2t_ratio = "{:.1f}".format(t2t_cnt/total_sample_size*100)
            both_ratio = "{:.1f}".format(both_cnt/total_sample_size*100)

            scores_print += score+'\t'+norag_ratio+'\t'+i2i_ratio+'\t'+t2t_ratio+'\t'+both_ratio+'\t'
        
        else:
            score = mean(sampled_res)
            score = "{:.2f}".format(score/5*100)
            scores_print += score+'\t'
    scores_print += '\n'



result_txt = f'/mnt/nas-alinlp/zhuochen.zc/others/leaderboard/log/result_cp2excel_{os.path.basename(search_decision_file)}_{args.both}.txt'
with open(result_txt, 'w') as g:
    g.write(scores_print)
    print(result_txt)