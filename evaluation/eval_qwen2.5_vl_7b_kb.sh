PID=346151
while ps -p $PID > /dev/null; do
    echo 'waiting' "$PID"
    sleep 120  # Wait for 120 seconds before checking again
done
echo "Process $PID has finished."


source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3;
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;

MODEL_PATH='/mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/0526_more/best-merged'


# ====================== lifevqa ======================
python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i lifevqa/norag/lifevqa.jsonl --rag norag
python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i lifevqa/image2image/lifevqa.jsonl --rag image2image
# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i lifevqa/text2text/lifevqa.jsonl --rag text2text

python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i lifevqa/text2text_gold_query_from_origin_query_image/lifevqa.jsonl --rag text2text
python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i lifevqa/text2text_gold_query_from_image_search/lifevqa.jsonl --rag text2text

python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i lifevqa/both/lifevqa.jsonl --rag both
# # ====================== lifevqa END ======================

# # # ====================== qwenvqa ======================
python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i qwenvqa/norag/qwenvqa.jsonl --rag norag
python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i qwenvqa/image2image/qwenvqa.jsonl --rag image2image
# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i qwenvqa/text2text/qwenvqa.jsonl --rag text2text

python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i qwenvqa/text2text_gold_query_from_origin_query_image/qwenvqa.jsonl --rag text2text
python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i qwenvqa/text2text_gold_query_from_image_search/qwenvqa.jsonl --rag text2text

python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i qwenvqa/both/qwenvqa.jsonl --rag both --rewrite 0
# # ====================== qwenvqa END ======================

# # ====================== visual7w ======================
# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i visual7w/norag/visual7w.jsonl --rag norag
# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i visual7w/image2image/visual7w.jsonl --rag image2image
# # python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i visual7w/text2text/visual7w.jsonl --rag text2text

# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i visual7w/text2text_gold_query_from_origin_query_image/visual7w.jsonl --rag text2text
# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i visual7w/text2text_gold_query_from_image_search/visual7w.jsonl --rag text2text

# python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i visual7w/both/visual7w.jsonl --rag both --rewrite 0
# # ====================== visual7w END ======================

# # ====================== nocaps ======================
# python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i nocaps/norag/nocaps.jsonl --rag norag
# python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i nocaps/image2image/nocaps.jsonl --rag image2image
# # python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i nocaps/text2text/nocaps.jsonl --rag text2text

# python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i nocaps/text2text_gold_query_from_origin_query_image/nocaps.jsonl --rag text2text
# python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i nocaps/text2text_gold_query_from_image_search/nocaps.jsonl --rag text2text

# python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i nocaps/both/nocaps.jsonl --rag both
# # ====================== nocaps END ======================
exit


langs=("en" "ch")
# langs=("ch")

for lang in "${langs[@]}" 
do

    # no rag
    python eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH -i vfreshqa_${lang}/norag/vfreshqa_${lang}.jsonl --rag norag

    # image2image
    python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH \
        -i vfreshqa_${lang}/image2image/vfreshqa_${lang}_bing.jsonl \
        --rag image2image

    # python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH \
    #     -i vfreshqa_${lang}/text2text/vfreshqa_${lang}_google.jsonl \
    #     --rag text2text

    python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH \
        -i vfreshqa_${lang}/text2text_gold_query_from_origin_query_image/vfreshqa_${lang}.jsonl \
        --rag text2text

    python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH \
        -i vfreshqa_${lang}/text2text_gold_query_from_image_search/vfreshqa_${lang}.jsonl \
        --rag text2text

    # both
    python -u eval_qwen2.5_vl_7b_kb.py --model $MODEL_PATH \
        -i vfreshqa_${lang}/both/vfreshqa_${lang}.jsonl \
        --rag both
done

