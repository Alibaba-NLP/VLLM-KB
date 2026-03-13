# PID=805132  # Replace with your process ID
# echo "Wait $PID."

# while ps -p "$PID" > /dev/null; do
#     sleep 300  # Wait for xxx before checking again
# done
# echo "Process $PID has finished."


source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3

MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 swift sft \
--model_type qwen2_5_vl \
--model /mnt/nas-alinlp/.cache/modelscope/models/Qwen/Qwen2___5-VL-7B-Instruct \
--dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0225_info_vqa_model_boundary/train.jsonl \
--val_dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0225_info_vqa_model_boundary/dev.jsonl \
--output_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output \
--max_length 1024 \
--evaluation_strategy 'steps' \
--eval_steps 1000 \
--logging_steps 200 \
--save_total_limit 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 16 \
--attn_impl 'flash_attn' \
--deepspeed /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_scripts/ds_config.json