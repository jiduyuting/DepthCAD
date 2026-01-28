#!/bin/bash
# Training script for DepthCAD with masked PBRT dataset
# Mask configuration: amplitude < 5th percentile OR > 99.5th percentile are masked

export MODEL_DIR="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="output/depthcad_pbrt_masked"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="pbrt_dataset" \
 --dataset_config="masked" \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=1e-4 \
 --train_batch_size=16 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none \
 --num_train_epochs=500 \
 --lr_scheduler="cosine" \
 --checkpointing_steps 5000

# Notes:
# 1. This uses the masked dataset which applies amplitude-based masking:
#    - Lower threshold: adaptive 5th percentile of amplitude
#    - Upper threshold: 99.5th percentile (removes extreme outliers)
#    - Masked regions are set to 0 in IQ data and confidence map
#
# 2. To try different mask thresholds, edit process_mask.py:
#    - amp_thresh: None for adaptive 5%, or set fixed value like 0.01
#    - upper_percentile: 99.5 for top 0.5%, or None to disable
#
# 3. After modifying thresholds, re-run process_mask.py to regenerate data
