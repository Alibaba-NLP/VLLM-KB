# Training done on h20...



source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3_github_main

# Run KB model
cd /mnt/nas-alinlp/zhuochen.zc/others/KnowB2;
python inference_kb_model.py --model_id_or_path output/0526_more_full_h20/checkpoint-2500 -s 'more_full_h20'
