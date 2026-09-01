#!/bin/bash

# DepthCAD + Region-Aware Inpainting Training Script

# export MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
# export OUTPUT_DIR="output/depthcad_pbrt_rad_4_20_3"
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_OFFLINE=0
# export TRANSFORMERS_OFFLINE=0
# export HF_DATASETS_OFFLINE=0

# # 可选: 从已有的 DepthCAD checkpoint 继续训练
# # export DEPTHCAD_PATH="output/depthcad_pbrt_4_16_masked/checkpoint-15000/depthcad"

# accelerate launch train_pbrt_rad.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --dataset_name="pbrt_dataset" \
#  --dataset_config="masked" \
#  --train_list_path="pbrt_dataset/train.txt" \
#  --mixed_precision="fp16" \
#  --resolution=512 \
#  --learning_rate=5e-6 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --use_8bit_adam \
#  --enable_xformers_memory_efficient_attention \
#  --set_grads_to_none \
#  --num_train_epochs=500 \
#  --lr_scheduler="constant_with_warmup" \
#  --lr_warmup_steps=500 \
#  --checkpointing_steps=2000 \
#  --resume_from_checkpoint=latest \
#  --inpaint_lr=5e-5 \
#  --freeze_depthcad

# export DEPTHCAD_PATH="output/depthcad_pbrt_4_16_masked/checkpoint-15000/depthcad"
# accelerate launch train_pbrt_rad.py \
#  --depthcad_path=$DEPTHCAD_PATH \
#  --output_dir=$OUTPUT_DIR \
#  --dataset_name="pbrt_dataset" \
#  --dataset_config="masked" \
#  --train_list_path="pbrt_dataset/train.txt" \
#  --mixed_precision="fp16" \
#  --resolution=512 \
#  --learning_rate=5e-6 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --use_8bit_adam \
#  --enable_xformers_memory_efficient_attention \
#  --set_grads_to_none \
#  --num_train_epochs=500 \
#  --lr_scheduler="constant_with_warmup" \
#  --lr_warmup_steps=500 \
#  --checkpointing_steps=2000 \
#  --resume_from_checkpoint=latest \
#  --inpaint_lr=5e-5


# MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"

# accelerate launch train_pbrt_rad.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --output_dir=output/depthcad_pbrt_rad_4_20_4 \
#  --dataset_name="pbrt_dataset" \
#  --dataset_config="masked" \
#  --train_list_path="pbrt_dataset/train.txt" \
#  --mixed_precision="fp16" \
#  --resolution=512 \
#  --learning_rate=5e-6 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
#  --use_8bit_adam \
#  --enable_xformers_memory_efficient_attention \
#  --set_grads_to_none \
#  --num_train_epochs=500 \
#  --lr_scheduler="constant_with_warmup" \
#  --lr_warmup_steps=500 \
#  --checkpointing_steps=5000 \
#  --resume_from_checkpoint=latest \
#  --inpaint_lr=1e-4 \

mkdir -p output/depthcad_pbrt_rad_inpaint_only

MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"

# 冻结 ControlNet，只训 inpaint
accelerate launch train_pbrt_rad.py \
 --pretrained_model_name_or_path="$MODEL_DIR" \
 --output_dir=output/depthcad_pbrt_rad_inpaint_only \
 --depthcad_path=output/depthcad_pbrt_4_16_masked/checkpoint-15000/depthcad \
 --dataset_name="pbrt_dataset" \
 --dataset_config="masked" \
 --train_list_path="pbrt_dataset/train.txt" \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=5e-6 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --num_train_epochs=500 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=500 \
 --checkpointing_steps=5000 \
 --resume_from_checkpoint=latest \
 --inpaint_lr=1e-4 \
 --freeze_depthcad
