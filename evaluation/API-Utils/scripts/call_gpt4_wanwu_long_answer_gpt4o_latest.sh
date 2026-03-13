#!/bin/bash

export DASHSCOPE_API_KEY='sk-933b59d5568f40de963756c66199cff5'

cd /mnt/nas-alinlp/zhuochen.zc/others/API-Utils;

# DatasetNames=("knowledge_animal")
DatasetNames=("knowledge_person" "knowledge_animal" "knowledge_plant" "knowledge_product")


# 基本参数配置
INPUT_PATH=/mnt/nas-alinlp/zhili/RAG-Evaluation/vl_boundary/search/data_storage/final/zh/wanwu
OUTPUT_PATH=/mnt/nas-alinlp/zhili/RAG-Evaluation/vl_boundary/search/data_storage/final/zh/wanwu_with_answer
NUM_WORKERS=4

start_time=$(date +%s)
echo "Script started at: $(date)"

for DatasetName in ${DatasetNames[@]}; do
    echo "Processing dataset: $DatasetName"

    INPUT_FILE=$INPUT_PATH/$DatasetName.jsonl
    OUTPUT_FILE=$OUTPUT_PATH/$DatasetName.long_answer.4o_latest.jsonl 
    LOG_FILE=./scripts/$DatasetName.long_answer.log 

    python -u call_gpt4_wanwu_long_answer.py -m chatgpt-4o-latest \
    -i $INPUT_FILE \
    -o $OUTPUT_FILE \
    --num-workers $NUM_WORKERS

done
