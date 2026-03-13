source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate deepseek;

cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;


langs=("en" "ch")


for lang in "${langs[@]}" 
do

    # no rag
    # python eval_ds_vl.py -i vfreshqa_${lang}/norag/vfreshqa_${lang}.jsonl --rag norag
    # python eval_ds_vl2.py -i vfreshqa_${lang}/norag/vfreshqa_${lang}.jsonl --rag norag

    # # image2image
    # python eval_ds_vl.py \
    #     -i vfreshqa_${lang}/image2image/vfreshqa_${lang}_bing.jsonl \
    #     --rag image2image

    # python eval_ds_vl2.py \
    #     -i vfreshqa_${lang}/image2image/vfreshqa_${lang}_bing.jsonl \
    #     --rag image2image

    # both image2image, text2text
    # python eval_ds_vl.py \
    #     -i vfreshqa_${lang}/both/vfreshqa_${lang}_both.jsonl \
    #     --rag both

    python eval_ds_vl2.py \
        -i vfreshqa_${lang}/both/vfreshqa_${lang}_both.jsonl \
        --rag both
done


# for lang in "${langs[@]}" 
# do
#     python -u eval_ds_vl2_multi_gpu.py \
#         -i vfreshqa_${lang}/both/vfreshqa_${lang}_both.jsonl \
#         --rag both
# done