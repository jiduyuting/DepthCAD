#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"

CACHE_DIR="${CACHE_DIR:-depth_completion_cache/depth_cache_0515_n1000_plane_r12}"
TRAIN_LIST="${TRAIN_LIST:-output/splits_n1000_plane_r12_exclude_seed123/train.txt}"
VAL_LIST="${VAL_LIST:-output/splits_n1000_plane_r12_exclude_seed123/val.txt}"
REAL_RAW_DIR="${REAL_RAW_DIR:-raw}"
REAL_DEPTH_DIR="${REAL_DEPTH_DIR:-depth}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-output/real_raw9_flow_finetune_holefocus_iq6_realholes_continue_e20_lr5e6/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/synthetic_realhole_flow_pretrain_generalized_split_e20}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MASKS_PER_SAMPLE="${MASKS_PER_SAMPLE:-1}"
VAL_MASKS_PER_SAMPLE="${VAL_MASKS_PER_SAMPLE:-1}"
COMPONENT_VAL_RATIO="${COMPONENT_VAL_RATIO:-0.25}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"

"${PYTHON_BIN}" -u scripts/train_synthetic_realhole_flow_pretrain.py \
  --cache_dir "${CACHE_DIR}" \
  --train_list "${TRAIN_LIST}" \
  --val_list "${VAL_LIST}" \
  --real_raw_dir "${REAL_RAW_DIR}" \
  --real_depth_dir "${REAL_DEPTH_DIR}" \
  --pretrained_checkpoint "${PRETRAIN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --masks_per_sample "${MASKS_PER_SAMPLE}" \
  --val_masks_per_sample "${VAL_MASKS_PER_SAMPLE}" \
  --mask_mode real_hole_speckle_shapes \
  --component_val_ratio "${COMPONENT_VAL_RATIO}" \
  --real_hole_max_components 24 \
  --real_hole_min_overlap 0.6 \
  --real_speckle_component_ratio 0.6 \
  --lr 1e-5 \
  --amp \
  --device "${CUDA_DEVICE}" \
  --log_every 50 \
  --save_every 5
