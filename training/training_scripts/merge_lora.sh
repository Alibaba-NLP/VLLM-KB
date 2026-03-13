# source /mnt/nas-alinlp/yangning/miniconda3/bin/activate swift
# source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift_ds;
source /mnt/nas-alinlp/zhuochen.zc/anaconda3/bin/activate swift3;

which swift;

# 0403 train on infoseek+vqav2+wanwu
# CUDA_VISIBLE_DEVICES=0 swift export \
# --ckpt_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/v1-20250401-094254/checkpoint-5459 \
# --merge_lora true \
# --model Qwen/Qwen2.5-VL-7B-Instruct

# 0408 less in-boundary training data
# CUDA_VISIBLE_DEVICES=0 swift export \
# --ckpt_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/v2-20250403-060544/checkpoint-4078 \
# --merge_lora true \
# --model Qwen/Qwen2.5-VL-7B-Instruct

# less wanwu data
# CUDA_VISIBLE_DEVICES=0 swift export \
# --ckpt_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/v3-20250408-213443/checkpoint-4000/ \
# --merge_lora true \
# --model Qwen/Qwen2.5-VL-7B-Instruct

# less infoseek+vqav2+wanwu animal,plant
# CUDA_VISIBLE_DEVICES=0 swift export \
# --ckpt_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/v4-20250501-112810/checkpoint-1700 \
# --merge_lora true \
# --model Qwen/Qwen2.5-VL-7B-Instruct

# full infoseek+vqav2+wanwu animal,plant
CUDA_VISIBLE_DEVICES=0 swift export \
--ckpt_dir /mnt/nas-alinlp/zhuochen.zc/others/KnowB2/output/v5-20250501-120717/checkpoint-4635 \
--merge_lora true \
--model Qwen/Qwen2.5-VL-7B-Instruct