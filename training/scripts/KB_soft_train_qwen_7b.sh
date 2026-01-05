TRAIN_PATH=training_data/qwen_vl_chat_7b/soft/train.all.jsonl
DEV_PATH=training_data/qwen_vl_chat_7b/soft/dev.all.jsonl

# train qwen-vl-chat
CUDA_VISIBLE_DEVICES=0 swift sft \
--model_type qwen-vl-chat \
--dataset $TRAIN_PATH \
--val_dataset $DEV_PATH \
--model_id_or_path Qwen/Qwen-VL-Chat \
--output_dir ./output \
--max_length 8192 \
--evaluation_strategy 'steps' \
--eval_steps 1000 \
--logging_steps 200 \
--save_total_limit 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 16
