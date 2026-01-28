#!/bin/bash

# Marigold-style training script for DepthCAD
# This script demonstrates two key improvements:
# 1. Enhanced ControlNet conditioning with gradient features
# 2. LoRA-based parameter-efficient fine-tuning

# ===== Method 1: Original training (baseline) =====
# python train2.py \
#     --dataset_name=flat_dataset \
#     --dataset_config=default \
#     --output_dir=model_baseline \
#     --resolution=512 \
#     --train_batch_size=4 \
#     --gradient_accumulation_steps=1 \
#     --gradient_checkpointing \
#     --mixed_precision=bf16 \
#     --num_train_epochs=1 \
#     --learning_rate=5e-6 \
#     --lr_scheduler=constant \
#     --lr_warmup_steps=500 \
#     --seed=42

# ===== Method 2: Enhanced features only (no LoRA) =====
# Use 6-channel conditioning (noise, conf, and their gradients)
# python train2.py \
#     --dataset_name=pbrt_dataset \
#     --dataset_config=default \
#     --output_dir=model_enhanced \
#     --resolution=512 \
#     --train_batch_size=4 \
#     --gradient_accumulation_steps=1 \
#     --gradient_checkpointing \
#     --mixed_precision=bf16 \
#     --num_train_epochs=1 \
#     --learning_rate=5e-6 \
#     --lr_scheduler=constant \
#     --lr_warmup_steps=500 \
#     --seed=42 \
#     --use_enhanced_features

# ===== Method 3: Full Marigold-style (enhanced features + LoRA) =====
# Use 6-channel conditioning + LoRA fine-tuning
# python train2.py \
#     --dataset_name=flat_dataset \
#     --dataset_config=default \
#     --output_dir=model_marigold \
#     --resolution=512 \
#     --train_batch_size=4 \
#     --gradient_accumulation_steps=1 \
#     --gradient_checkpointing \
#     --mixed_precision=bf16 \
#     --num_train_epochs=1 \
#     --learning_rate=1e-4 \
#     --lr_scheduler=constant \
#     --lr_warmup_steps=500 \
#     --seed=42 \
#     --use_enhanced_features \
#     --use_lora \
#     --lora_rank=16 \
#     --lora_alpha=32 \
#     --lora_dropout=0.1

# ===== Method 4: LoRA only (no enhanced features) =====
# Use original 2-channel conditioning + LoRA fine-tuning
# python train2.py \
#     --dataset_name=flat_dataset \
#     --dataset_config=default \
#     --output_dir=model_lora \
#     --resolution=512 \
#     --train_batch_size=4 \
#     --gradient_accumulation_steps=1 \
#     --gradient_checkpointing \
#     --mixed_precision=bf16 \
#     --num_train_epochs=1 \
#     --learning_rate=1e-4 \
#     --lr_scheduler=constant \
#     --lr_warmup_steps=500 \
#     --seed=42 \
#     --use_lora \
#     --lora_rank=16 \
#     --lora_alpha=32 \
#     --lora_dropout=0.1


# Use local cached model to avoid network issues
# Use direct local path to prevent transformers from looking for adapter_config.json online
export MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
export OUTPUT_DIR="output/depthcad_pbrt_marigold_1_27"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=3
accelerate launch train_marigold.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="pbrt_dataset" \
 --mixed_precision="fp16" \
  --resolution=512 \
  --learning_rate=5e-6 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --num_train_epochs=500 \
  --lr_scheduler="cosine" \
  --checkpointing_steps 5000 \
    --use_enhanced_features \
     --resume_from_checkpoint latest
 #   --use_lora \
 #   --lora_rank=16 \
 #   --lora_alpha=32 \
