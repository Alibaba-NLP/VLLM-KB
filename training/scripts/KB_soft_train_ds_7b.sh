TRAIN_PATH=training_data/deepseek_vl_chat_7b/soft/train_soft.all.jsonl
DEV_PATH=training_data/deepseek_vl_chat_7b/soft/dev_soft.all.jsonl

CUDA_VISIBLE_DEVICES=0 swift sft \
--model_type deepseek-vl-7b-chat \
--model_id_or_path deepseek-ai/deepseek-vl-7b-chat \
--dataset $TRAIN_PATH \
--val_dataset $DEV_PATH \
--output_dir ./output \
--max_length 8192 \
--evaluation_strategy 'steps' \
--eval_steps 1000 \
--logging_steps 200 \
--save_total_limit 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 16
