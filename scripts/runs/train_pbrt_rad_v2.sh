#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
export OUTPUT_DIR="output/depthcad_pbrt_rad_v2"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

accelerate launch scripts/train_pbrt_rad_v2.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="pbrt_dataset" \
 --dataset_config="masked" \
 --train_list_path="pbrt_dataset/train.txt" \
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
 --checkpointing_steps=5000 \
 --resume_from_checkpoint=latest \
 --inpaint_lr=1e-4 \
 --boundary_weight=5.0 \
 --center_weight=1.0