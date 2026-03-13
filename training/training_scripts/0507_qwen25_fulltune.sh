set -e

# PID=805132  # Replace with your process ID
# echo "Wait $PID."

# while ps -p "$PID" > /dev/null; do
#     sleep 300  # Wait for xxx before checking again
# done
# echo "Process $PID has finished."

source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3
export NCCL_SOCKET_TIMEOUT_MS=1200000 # =20 mins

MODEL_PATH=/mnt/nas-alinlp/zhuochen.zc/models/Qwen/Qwen2___5-VL-7B-Instruct
CKPT_DIR=/mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/0507_KB2_even_fulltune
mkdir -p $CKPT_DIR


nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
MAX_PIXELS=1003520 \
swift sft \
--model_type qwen2_5_vl \
--model $MODEL_PATH \
--dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0503_even/train.jsonl \
--val_dataset /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/training_data/0503_even/dev.jsonl \
--output_dir $CKPT_DIR \
--create_checkpoint_symlink true \
--add_version false \
--max_length 1024 \
--train_type full \
--learning_rate 3e-5 \
--evaluation_strategy 'steps' \
--eval_steps 500 \
--logging_steps 200 \
--metric_for_best_model eval_loss \
--save_total_limit 2 \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing true \
--deepspeed zero3


# Merge LoRA
# CUDA_VISIBLE_DEVICES=0 swift export \
# --ckpt_dir $CKPT_DIR/best \
# --merge_lora true \
# --model Qwen/Qwen2.5-VL-7B-Instruct

# Run inference of KB model to get boundary labels
cd /mnt/nas-alinlp/zhuochen.zc/others/KnowB2;
python inference_kb_model.py --model_id_or_path $CKPT_DIR/best