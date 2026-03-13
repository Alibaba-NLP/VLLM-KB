set -e

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate qwen 
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;

# python search_src/search.py --dataset lifevqa --rag_type text2text --mp 1 > log/search_lifevqa_text2text.log
# python search_src/search.py --dataset qwenvqa --rag_type text2text --mp 1 > log/search_qwenvqa_text2text.log
# python search_src/search.py --dataset vfreshqa_ch --rag_type text2text --mp 1 > log/search_vfreshqa_ch_text2text_gold_query_from_image_search.log
# python search_src/search.py --dataset vfreshqa_en --rag_type text2text --mp 1 > log/search_vfreshqa_en_text2text_gold_query_from_image_search.log
# python search_src/search.py --dataset visual7w --rag_type text2text --mp 1 > log/search_visual7w_text2text.log

python -u search_src/search.py --dataset nocaps --rag_type text2text --mp 1 > log/search_nocaps_image2image.log

