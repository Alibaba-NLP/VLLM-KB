CKPT_DIR=
model_id_or_path=deepseek-ai/deepseek-vl-7b-chat
# model_id_or_path=Qwen/Qwen-VL-Chat


CUDA_VISIBLE_DEVICES=0 swift export \
--ckpt_dir $CKPT_DIR \
--merge_lora true \
--model_id_or_path $model_id_or_path