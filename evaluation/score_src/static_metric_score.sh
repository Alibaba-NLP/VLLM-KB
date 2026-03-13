#!/bin/bash

PID=521508
while ps -p $PID > /dev/null; do
    echo 'waiting' "$PID"
    sleep 120  # Wait for 120 seconds before checking again
done
echo "Process $PID has finished."

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate qwen;
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;


datasets=("lifevqa" "qwenvqa" "visual7w" "nocaps")
# datasets=("lifevqa" "qwenvqa")
# datasets=()

sub_dirs=("norag" "image2image" "text2text_gold_query_from_origin_query_image" "text2text_gold_query_from_image_search" "both")

# models=("qwen25-vl-7b" "deepseek-vl-7b-chat" "chatgpt-4o-latest" "qwen-vl-max" "qwen-vl-max-latest")
# models=("qwen-vl-max-latest")
models=("qwen2.5-vl-7b-kb")
# models=("deepseek-vl-7b-chat-2")


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

            python score_src/static_metric_score.py --data_path $file_path --overwrite 1
        done
    done
done


datasets=("vfreshqa_ch" "vfreshqa_en")
# datasets=( "vfreshqa_ch")

# sub_dirs=("norag" "image2image" "text2text" "both")
# sub_dirs=("text2text_gold_query_from_image_search")

# models=("qwen-vl-max" "qwen25-vl-7b" "deepseek-vl-7b-chat" "chatgpt-4o-latest")
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

            python score_src/static_metric_score.py --data_path $file_path --overwrite 1
        done
    done
done

