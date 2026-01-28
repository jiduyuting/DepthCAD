#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depthcad_zimage

export MODEL_DIR="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="output/depthcad_pbrt_zimage"

# Offline mode settings
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Mirror (fallback if offline doesn't work)
export HF_ENDPOINT=https://hf-mirror.com

# GPU settings
export CUDA_VISIBLE_DEVICES=1

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="pbrt_dataset" \
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
  --checkpointing_steps 5000 \
  --resume_from_checkpoint latest


# export HF_ENDPOINT=https://hf-mirror.com
# MODEL_DIR="stabilityai/stable-diffusion-2-1"
# DEPTHCAD_PATH="/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_12_12/checkpoint-55000/depthcad"
# TEST_LIST_PATH="/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt"
# NOISE_IQ_DIR="/data/pre_student/hcy/pbrt/noise"
# NOISE_DEPTH_DIR="/data/pre_student/hcy/pbrt/noise_depth"
# OUT_DIR="/data/pre_student/GJ/DepthCAD/pbrt/data_12_12"

# mkdir -p "$OUT_DIR"

# while IFS= read -r sample_id; do
#     # 跳过空行
#     [[ -z "$sample_id" ]] && continue
#
#     noise_iq_file="$NOISE_IQ_DIR/$sample_id"
#     noise_depth_file="$NOISE_DEPTH_DIR/$sample_id.npy"
#     out_file="$OUT_DIR/$sample_id.npy"
#
#     if [[ ! -e "$noise_iq_file" ]]; then
#         echo "[WARN] noise IQ missing: $noise_iq_file" >&2
#         continue
#     fi
#     if [[ ! -e "$noise_depth_file" ]]; then
#         echo "[WARN] noise depth missing: $noise_depth_file" >&2
#         continue
#     fi
#
#     echo "[INFO] Inference for $sample_id -> $out_file"
#     # ensure per-sample output directory exists (sample_id may contain subdirs)
#     mkdir -p "$(dirname "$out_file")"
#     python inference.py \
#         --pretrained_model_name_or_path "$MODEL_DIR" \
#         --depthcad_path "$DEPTHCAD_PATH" \
#         --noise_IQ_file "$noise_iq_file" \
#         --noise_depth_file "$noise_depth_file" \
#         --out_file "$out_file"
# done < "$TEST_LIST_PATH"

# python inference.py \
#     --pretrained_model_name_or_path "$MODEL_DIR" \
#     --depthcad_path "$DEPTHCAD_PATH" \
#     --noise_IQ_file "/data/pre_student/hcy/pbrt/noise/bathroom/1/104.npy" \
#     --noise_depth_file "/data/pre_student/hcy/pbrt/noise_depth/bathroom/1/104.npy" \
#     --out_file "/data/pre_student/GJ/DepthCAD/test/bathroom/1/104_55000.npy"


# python eval.py \
#     --test_list_path "/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt" \
#     --out_dir "/data/pre_student/GJ/DepthCAD/out_pbrt" \
#     --pred_dir "/data/pre_student/GJ/DepthCAD/pbrt/data_1212_55000"

# python eval.py \
#     --test_list_path "/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt" \
#     --out_dir "/data/pre_student/GJ/DepthCAD/out_pbrt" \
#     --pred_dir "/data/pre_student/GJ/DepthCAD/pbrt/data_1212_55000" \
#     --gt_dir "/data/pre_student/hcy/pbrt/gt_depth"
