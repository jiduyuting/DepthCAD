#!/bin/bash

# ============================================================================
# HYPIR-Enhanced DepthCAD Training Script
# ============================================================================
#
# This script trains DepthCAD using the HYPIR LoRA weights pre-merged into UNet.
# Key features:
# - HYPIR LoRA weights (HYPIR_sd2.pth) are loaded and merged into UNet backbone
# - Simple 2-channel conditioning (noise + confidence)
# - MSE loss only (no gradient loss)
# - Uses transforms.Resize for data preprocessing
#
# ============================================================================

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depthcad_zimage

# ===== Configuration =====
# Use local cached model to avoid network issues
export MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
export OUTPUT_DIR="output/depthcad_pbrt_hypir_1_27"

# HuggingFace settings (offline mode)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# GPU settings (train_hypir.py uses CUDA device '1')
export CUDA_VISIBLE_DEVICES=2

# ===== Training Parameters =====
# Mixed precision for memory efficiency
MIXED_PRECISION="fp16"

# Resolution for training (default: 512)
RESOLUTION=512

# Learning rate (HYPIR typically uses lower LR)
LEARNING_RATE=5e-6

# Batch size per GPU
TRAIN_BATCH_SIZE=16

# Gradient accumulation for effective larger batch size
GRADIENT_ACCUMULATION_STEPS=4

# Effective batch size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
# = 16 * 4 * 1 = 64

# Number of training epochs
NUM_TRAIN_EPOCHS=500

# Learning rate scheduler
LR_SCHEDULER="cosine"

# Checkpoint saving interval
CHECKPOINTING_STEPS=5000

# ===== Memory Optimization =====
# Enable gradient checkpointing to save memory
USE_GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# Use 8-bit Adam optimizer
USE_8BIT_ADAM="--use_8bit_adam"

# Enable xformers memory efficient attention
USE_XFORMERS="--enable_xformers_memory_efficient_attention"

# Set gradients to None for memory savings
SET_GRADS_NONE="--set_grads_to_none"

# ===== Resume Training =====
# Set to "latest" to resume from last checkpoint, or specify path like "checkpoint-5000"
# Comment out to start from scratch
# NOTE: Starting fresh to use fixed data loading code
# RESUME_FROM_CHECKPOINT="--resume_from_checkpoint latest"

# ===== Launch Training =====
echo "=========================================="
echo "Starting HYPIR-Enhanced DepthCAD Training"
echo "=========================================="
echo "Model Dir: $MODEL_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $NUM_TRAIN_EPOCHS"
echo "=========================================="

accelerate launch train_hypir.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name="pbrt_dataset" \
    --mixed_precision=$MIXED_PRECISION \
    --resolution=$RESOLUTION \
    --learning_rate=$LEARNING_RATE \
    --train_batch_size=$TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    $USE_GRADIENT_CHECKPOINTING \
    $USE_8BIT_ADAM \
    $USE_XFORMERS \
    $SET_GRADS_NONE \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --lr_scheduler=$LR_SCHEDULER \
    --lr_warmup_steps=500 \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    $RESUME_FROM_CHECKPOINT

echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# ===== Notes =====
# 1. HYPIR_sd2.pth should be in the current directory
# 2. The script will automatically load and merge HYPIR LoRA weights (lines 471-505 in train_hypir.py)
# 3. If you encounter OOM errors, reduce TRAIN_BATCH_SIZE or increase GRADIENT_ACCUMULATION_STEPS
# 4. Monitor training with TensorBoard: tensorboard --logdir=$OUTPUT_DIR/logs
# 5. To start fresh without resuming, comment out the RESUME_FROM_CHECKPOINT line