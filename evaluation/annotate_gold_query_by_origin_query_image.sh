set -e

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate qwen  
cd /mnt/nas-alinlp/zhuochen.zc/others/leaderboard;

py=annotate_gold_query_by_origin_query_image_qwen_max.py

# python $py --input_data lifevqa/norag/lifevqa.jsonl  --lang zh 
# python $py --input_data qwenvqa/norag/qwenvqa.jsonl  --lang zh 

# python $py --input_data vfreshqa_ch/norag/vfreshqa_ch.jsonl  --lang zh 
# python $py --input_data vfreshqa_en/norag/vfreshqa_en.jsonl  --lang en

# python $py --input_data visual7w/norag/visual7w.jsonl  --lang en

python $py --input_data nocaps/norag/nocaps.jsonl  --lang en