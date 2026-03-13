set -e

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate qwen  
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;

# python annotate_gold_query_by_image_search_info_qwen_max.py --input_data vfreshqa_ch/image2image/vfreshqa_ch_image2image_bing.jsonl  --lang zh 
# python annotate_gold_query_by_image_search_info_qwen_max.py --input_data vfreshqa_en/image2image/vfreshqa_en_image2image_bing.jsonl  --lang en

python -u annotate_gold_query_by_image_search_info_qwen_max.py --input_data nocaps/image2image/nocaps.jsonl --lang en
