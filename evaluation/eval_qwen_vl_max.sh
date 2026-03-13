source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift_ds;

cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;


# ====================== lifevqa ======================
# python eval_qwen_vl_max.py -i lifevqa/norag/lifevqa.jsonl --rag norag
# python eval_qwen_vl_max.py -i lifevqa/image2image/lifevqa.jsonl --rag image2image
# python eval_qwen_vl_max_complement.py -i lifevqa/image2image/lifevqa.jsonl --rag image2image
# python eval_qwen_vl_max.py -i lifevqa/text2text/lifevqa.jsonl --rag text2text

# python eval_qwen_vl_max.py -i lifevqa/text2text_gold_query_from_origin_query_image/lifevqa.jsonl --rag text2text

# python eval_qwen_vl_max.py -i lifevqa/both/lifevqa.jsonl --rag both
# ====================== lifevqa END ======================

# ====================== qwenvqa ======================
# python eval_qwen_vl_max.py -i qwenvqa/norag/qwenvqa.jsonl --rag norag
# python eval_qwen_vl_max.py -i qwenvqa/image2image/qwenvqa.jsonl --rag image2image
# python eval_qwen_vl_max_complement.py -i qwenvqa/image2image/qwenvqa.jsonl --rag image2image
# python eval_qwen_vl_max.py -i qwenvqa/text2text/qwenvqa.jsonl --rag text2text

# python eval_qwen_vl_max.py -i qwenvqa/text2text_gold_query_from_origin_query_image/qwenvqa.jsonl --rag text2text

# python eval_qwen_vl_max.py -i qwenvqa/both/qwenvqa.jsonl --rag both
# ====================== qwenvqa END ======================

# ====================== visual7w ======================
# python eval_qwen_vl_max.py -i visual7w/norag/visual7w.jsonl --rag norag
# python eval_qwen_vl_max.py -i visual7w/image2image/visual7w.jsonl --rag image2image
# python eval_qwen_vl_max.py -i visual7w/text2text/visual7w.jsonl --rag text2text

# python eval_qwen_vl_max.py -i visual7w/text2text_gold_query_from_origin_query_image/visual7w.jsonl --rag text2text

# python eval_qwen_vl_max.py -i visual7w/both/visual7w.jsonl --rag both
# ====================== visual7w END ======================


# ====================== nocaps ======================
# python eval_qwen_vl_max.py -i nocaps/norag/nocaps.jsonl --rag norag
# python eval_qwen_vl_max.py -i nocaps/image2image/nocaps.jsonl --rag image2image
# python eval_qwen_vl_max.py -i nocaps/text2text/nocaps.jsonl --rag text2text

python eval_qwen_vl_max.py -i nocaps/text2text_gold_query_from_origin_query_image/nocaps.jsonl --rag text2text
# python eval_qwen_vl_max.py -i nocaps/text2text_gold_query_from_image_search/nocaps.jsonl --rag text2text

# python eval_qwen_vl_max.py -i nocaps/both/nocaps.jsonl --rag both
# ====================== nocaps END ======================
exit



langs=("en" "ch")
for lang in "${langs[@]}" 
do

    # no rag
    # python eval_qwen_vl_max.py -i vfreshqa_${lang}/norag/vfreshqa_${lang}.jsonl --rag norag

    # image2image
    # python eval_qwen_vl_max.py \
    #     -i vfreshqa_${lang}/image2image/vfreshqa_${lang}_bing.jsonl \
    #     --rag image2image

    # text2text
    # python eval_qwen_vl_max.py \
    #     -i vfreshqa_${lang}/text2text/vfreshqa_${lang}_google.jsonl \
    #     --rag text2text

    # python eval_qwen_vl_max.py \
    #     -i vfreshqa_${lang}/text2text_gold_query_from_origin_query_image/vfreshqa_${lang}.jsonl \
    #     --rag text2text

    # python eval_qwen_vl_max.py \
    #     -i vfreshqa_${lang}/text2text_gold_query_from_image_search/vfreshqa_${lang}.jsonl \
    #     --rag text2text

    # both 
    python eval_qwen_vl_max.py \
        -i vfreshqa_${lang}/both/vfreshqa_${lang}.jsonl \
        --rag both

done

