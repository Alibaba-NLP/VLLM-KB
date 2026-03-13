# PID=805132  # Replace with your process ID
# echo "Wait $PID."

# while ps -p "$PID" > /dev/null; do
#     sleep 300  # Wait for xxx before checking again
# done
# echo "Process $PID has finished."


source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3

MAX_PIXELS=1003520 \
swift sft \
--model_type qwen2_5_vl \
--model Qwen/Qwen2.5-VL-7B-Instruct \
--dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0501_thres_updated/train.jsonl \
--val_dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0501_thres_updated/dev.jsonl \
--output_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output \
--add_output_dir_suffix true \
--add_output_dir_suffix true \
--max_length 1024 \
--evaluation_strategy 'steps' \
--eval_steps 1000 \
--logging_steps 200 \
--save_total_limit 1 \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing false 
# --merge_lora true
# --attn_impl 'flash_attn'