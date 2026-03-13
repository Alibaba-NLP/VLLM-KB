set -e

# PID=805132  # Replace with your process ID
# echo "Wait $PID."

# while ps -p "$PID" > /dev/null; do
#     sleep 300  # Wait for xxx before checking again
# done
# echo "Process $PID has finished."

MODEL_PATH=/mnt/nas-alinlp/zhuochen.zc/models/Qwen/Qwen2___5-VL-7B-Instruct
CKPT_DIR=/mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/0526_more_full
# mkdir -p $CKPT_DIR

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3_github_main


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# NPROC_PER_NODE=4 \
# MAX_PIXELS=1003520 \
# WANDB_API_KEY=e51623e527d88fc12135eb0c197e3ab40f50721f WANDB_PROJECT=KnowB2 WANDB_LOG_MODEL=false \
# swift sft \
# --model_type qwen2_5_vl \
# --model $MODEL_PATH \
# --dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0526_more/train.jsonl \
# --val_dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0526_more/dev.jsonl \
# --output_dir $CKPT_DIR \
# --create_checkpoint_symlink true \
# --add_version false \
# --max_length 1024 \
# --train_type full \
# --learning_rate 2e-5 \
# --eval_strategy 'steps' \
# --eval_steps 500 \
# --logging_steps 200 \
# --metric_for_best_model eval_loss \
# --save_total_limit 2 \
# --num_train_epochs 3 \
# --per_device_train_batch_size 1 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 16 \
# --report_to wandb \
# --run_name "0526_more_full_a100" \
# --gradient_checkpointing true \
# --deepspeed zero3


# Run inference of KB model to get boundary labels
cd /mnt/nas-alinlp/zhuochen.zc/others/KnowB2;
python inference_kb_model.py --model_id_or_path $CKPT_DIR/best -s 'more_full'