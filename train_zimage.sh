#!/bin/bash

# DepthCAD Training Script for Z-Image-Turbo Model
#
# This script trains DepthCAD using Tongyi-MAI/Z-Image-Turbo as the base model.
#
# ⚠️ WARNING: This is EXPERIMENTAL and may not work without further modifications.
#
# Requirements:
# - diffusers >= 0.36.0
# - torch >= 2.1.0
# - transformers (latest)
# - See requirements_zimage.txt for complete list
#
# Usage:
#   bash train_zimage.sh

# Activate the correct conda environment
source /home/lab507/anaconda3/etc/profile.d/conda.sh
conda activate depthcad_zimage

# Set model directory - using Z-Image-Turbo
export MODEL_DIR="Tongyi-MAI/Z-Image-Turbo"
export CUDA_VISIBLE_DEVICES=2
# Set output directory
export OUTPUT_DIR="output/depthcad_zimage_turbo"

# Set HuggingFace mirror for faster download in China
export HF_ENDPOINT=https://hf-mirror.com

# Set which GPU to use

# Step 1: Precompute text embeddings to avoid OOM
# This is REQUIRED for Z-Image-Turbo due to large Qwen3Model
echo "========================================"
echo "Step 1: Precomputing text embeddings..."
echo "========================================"

EMBEDDINGS_OUTPUT="${OUTPUT_DIR}/embeddings"

python precompute_embeddings_zimage.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --dataset_name="pbrt_dataset" \
    --dataset_config="default" \
    --output_dir=$EMBEDDINGS_OUTPUT \
    --batch_size=32

# Check if embeddings were created successfully
if [ ! -f "${EMBEDDINGS_OUTPUT}/text_embeddings.npy" ]; then
    echo "ERROR: Failed to create text embeddings!"
    exit 1
fi

echo "✓ Text embeddings precomputed successfully!"

# Step 2: Train with precomputed embeddings
echo ""
echo "========================================"
echo "Step 2: Starting training with precomputed embeddings..."
echo "========================================"

# Launch training with accelerate
# Note: Z-Image-Turbo requires much smaller batch size and resolution due to:
# - VAE with 16 channels (vs 4 in SD21)
# - Larger transformer model
# - We use precomputed embeddings to avoid loading Qwen3Model
# We use gradient_accumulation to maintain effective batch size
accelerate launch train_zimage.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --dataset_name="pbrt_dataset" \
    --mixed_precision="fp16" \
    --resolution=256 \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=64 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \
    --num_train_epochs=500 \
    --lr_scheduler="cosine" \
    --checkpointing_steps 5000 \
    --precomputed_embeddings_path="${EMBEDDINGS_OUTPUT}/text_embeddings.npy" \
    --resume_from_checkpoint latest

