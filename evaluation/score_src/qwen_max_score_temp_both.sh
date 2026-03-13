#!/bin/bash

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate qwen;
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;


datasets=("lifevqa" "qwenvqa" "visual7w")
# datasets=("qwenvqa")

# sub_dirs=("norag" "image2image" "text2text_gold_query_from_origin_query_image" "text2text_gold_query_from_image_search" "both")
sub_dirs=("both")


models=("qwen-vl-max" "qwen25-vl-7b" "deepseek-vl-7b-chat" "chatgpt-4o-latest")
# models=("qwen-vl-max-latest")

for dir in "${datasets[@]}"; do
    for sub_dir in "${sub_dirs[@]}"; do
        for model in "${models[@]}"; do
            # 构建文件路径
            file_path="${dir}/${sub_dir}/${dir}_${model}.jsonl"

            # 检查文件是否存在
            if [ -f "$file_path" ]; then
                echo "Eval $file_path ..."
            else
                echo "File not found: $file_path" >&2
            fi

            python score_src/qwen_max_score.py --data_path $file_path
        done
    done
done



datasets=("vfreshqa_ch" "vfreshqa_en")
# datasets=( "vfreshqa_en")

# sub_dirs=("norag" "image2image" "text2text" "both")
# sub_dirs=("text2text_gold_query_from_image_search")

models=("qwen-vl-max" "qwen25-vl-7b" "deepseek-vl-7b-chat" "chatgpt-4o-latest")
# models=("qwen-vl-max-latest")


for dir in "${datasets[@]}"; do
    for sub_dir in "${sub_dirs[@]}"; do
        for model in "${models[@]}"; do

            # 构建文件路径
            if [ "$sub_dir" = "image2image" ]; then
                file_path="${dir}/${sub_dir}/${dir}_bing_${model}.jsonl"
            elif [ "$sub_dir" = "text2text" ]; then
                file_path="${dir}/${sub_dir}/${dir}_google_${model}.jsonl"
            else
                file_path="${dir}/${sub_dir}/${dir}_${model}.jsonl"
            fi
      
            # 检查文件是否存在
            if [ -f "$file_path" ]; then
                echo "Eval $file_path ..."
            else
                echo "File not found: $file_path" >&2
            fi

            python score_src/qwen_max_score.py --data_path $file_path
        done
    done
done

