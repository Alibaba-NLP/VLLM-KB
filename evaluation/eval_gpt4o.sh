#!/bin/bash
set -e

export DASHSCOPE_API_KEY=

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift_ds;
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/API-Utils;

# MODEL_NAME='gpt-4.5-preview-2025-02-27'
MODEL_NAME='chatgpt-4o-latest'
# MODEL_NAME='gpt-4o'

# ====================== lifevqa ======================
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../lifevqa/norag/lifevqa.jsonl --rag norag
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../lifevqa/image2image/lifevqa.jsonl --rag image2image
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../lifevqa/text2text/lifevqa.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../lifevqa/text2text_gold_query_from_origin_query_image/lifevqa.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../lifevqa/both/lifevqa.jsonl --rag both
# ====================== lifevqa END ======================

# ====================== qwenvqa ======================
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../qwenvqa/norag/qwenvqa.jsonl --rag norag
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../qwenvqa/image2image/qwenvqa.jsonl --rag image2image
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../qwenvqa/text2text/qwenvqa.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../qwenvqa/text2text_gold_query_from_origin_query_image/qwenvqa.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../qwenvqa/both/qwenvqa.jsonl --rag both
# ====================== qwenvqa END ======================

# ====================== visual7w ======================
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../visual7w/norag/visual7w.jsonl --rag norag
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../visual7w/image2image/visual7w.jsonl --rag image2image
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../visual7w/text2text/visual7w.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../visual7w/text2text_gold_query_from_origin_query_image/visual7w.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../visual7w/both/visual7w.jsonl --rag both
# ====================== visual7w END ======================

# ====================== nocaps ======================
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../nocaps/norag/nocaps.jsonl --rag norag
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../nocaps/image2image/nocaps.jsonl --rag image2image
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../nocaps/text2text/nocaps.jsonl --rag text2text

python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../nocaps/text2text_gold_query_from_origin_query_image/nocaps.jsonl --rag text2text
# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../nocaps/text2text_gold_query_from_image_search/nocaps.jsonl --rag text2text

# python -u call_gpt4_vfreshqa.py -m $MODEL_NAME -i ../nocaps/both/nocaps.jsonl --rag both
# ====================== nocaps END ======================
exit



# ========================= vfreshqa =========================
langs=("en" "ch")

for lang in "${langs[@]}" 
do

    # no rag
    # python -u call_gpt4_vfreshqa.py \
    #     -m $MODEL_NAME \
    #     -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/norag/vfreshqa_${lang}.jsonl \
    #     --rag norag

    # image2image
    # python -u call_gpt4_vfreshqa.py \
    #     -m $MODEL_NAME \
    #     -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/image2image/vfreshqa_${lang}_bing.jsonl \
    #     --rag image2image

    # text2text
    # python -u call_gpt4_vfreshqa.py \
    #     -m $MODEL_NAME \
    #     -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/text2text/vfreshqa_${lang}_google.jsonl \
    #     --rag text2text

    # python -u call_gpt4_vfreshqa.py \
    #     -m $MODEL_NAME \
    #     -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/text2text_gold_query_from_origin_query_image/vfreshqa_${lang}.jsonl \
    #     --rag text2text

    # python -u call_gpt4_vfreshqa.py \
    #     -m $MODEL_NAME \
    #     -i ../vfreshqa_${lang}/text2text_gold_query_from_image_search/vfreshqa_${lang}.jsonl \
    #     --rag text2text

    # both
    python -u call_gpt4_vfreshqa.py \
        -m $MODEL_NAME \
        -i ../vfreshqa_${lang}/both/vfreshqa_${lang}.jsonl \
        --rag both

done

exit




# 补充没有跑出response的
for lang in "${langs[@]}" 
do

    # no rag
    python -u call_gpt4_vfreshqa_complement.py \
        -m $MODEL_NAME \
        -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/norag/vfreshqa_${lang}.jsonl \
        --rag norag

    # image2image
    python -u call_gpt4_vfreshqa_complement.py \
        -m $MODEL_NAME \
        -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/image2image/vfreshqa_${lang}_image2image_bing.jsonl \
        --rag image2image

    # both image2image, text2text
    python -u call_gpt4_vfreshqa_complement.py \
        -m $MODEL_NAME \
        -i /mnt/nas-alinlp/zhuochen.zc/others/leaderboard/vfreshqa_${lang}/both/vfreshqa_${lang}_both.jsonl \
        --rag both

done
# ========================= vfreshqa END =========================

