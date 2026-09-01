#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}
if [[ -z "${REAL_ROOT:-}" ]]; then
  if [[ -d /hcy/datasets/pbrt/Real ]]; then
    REAL_ROOT=/hcy/datasets/pbrt/Real
  else
    REAL_ROOT=/data/pre_student/hcy/datasets/pbrt/Real
  fi
fi
if [[ ! -d "${REAL_ROOT}" && "${REAL_ROOT}" == /hcy/* && -d "/data/pre_student${REAL_ROOT}" ]]; then
  REAL_ROOT="/data/pre_student${REAL_ROOT}"
fi

RAW_DIR=${RAW_DIR:-"${REAL_ROOT}/noise"}
DEPTH_DIR=${DEPTH_DIR:-"${REAL_ROOT}/depth"}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt}
PBRT_SPLIT_JSON=${PBRT_SPLIT_JSON:-output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/split.json}
PBRT_CACHE_DIR=${PBRT_CACHE_DIR:-depth_completion_cache/depth_cache_0515_n1000_plane_r12}
REAL_SPLIT_JSON=${REAL_SPLIT_JSON:-output/pbrt_real_threshold_amp_depth_finetune_e40_p8_keepamp/split.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030}
RAW9_TRANSFORM=${RAW9_TRANSFORM:-none}

EPOCHS=${EPOCHS:-30}
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.30}
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-${BATCH_SIZE}}
REPLAY_NUM_WORKERS=${REPLAY_NUM_WORKERS:-${NUM_WORKERS}}
REPLAY_MAX_SAMPLES=${REPLAY_MAX_SAMPLES:-0}
MASK_LOSS_WEIGHT=${MASK_LOSS_WEIGHT:-1.0}
MASK_CENTER_WEIGHT=${MASK_CENTER_WEIGHT:-0.0}
VALID_LOSS_WEIGHT=${VALID_LOSS_WEIGHT:-0.05}
GRAD_LOSS_WEIGHT=${GRAD_LOSS_WEIGHT:-0.05}
HOLE_GRAD_LOSS_WEIGHT=${HOLE_GRAD_LOSS_WEIGHT:-0.0}
BOUNDARY_GRAD_LOSS_WEIGHT=${BOUNDARY_GRAD_LOSS_WEIGHT:-0.0}
BOUNDARY_L1_LOSS_WEIGHT=${BOUNDARY_L1_LOSS_WEIGHT:-0.0}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-3}
THRESHOLD_AMP_PERCENTILE=${THRESHOLD_AMP_PERCENTILE:-8.0}
THRESHOLD_MASK_CLOSE=${THRESHOLD_MASK_CLOSE:-1}
THRESHOLD_MASK_DILATE=${THRESHOLD_MASK_DILATE:-0}
THRESHOLD_MASK_MIN_COMPONENT_AREA=${THRESHOLD_MASK_MIN_COMPONENT_AREA:-8}

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-depthcad}
echo "Using REAL_ROOT=${REAL_ROOT}"
echo "Using RAW_DIR=${RAW_DIR}"
echo "Using DEPTH_DIR=${DEPTH_DIR}"
echo "Using PRETRAIN_CKPT=${PRETRAIN_CKPT}"
echo "Using PBRT_SPLIT_JSON=${PBRT_SPLIT_JSON}"
echo "Using PBRT_CACHE_DIR=${PBRT_CACHE_DIR}"
echo "Using REPLAY_WEIGHT=${REPLAY_WEIGHT}"
echo "Using RAW9_TRANSFORM=${RAW9_TRANSFORM}"
echo "Using LOSS_WEIGHTS=mask:${MASK_LOSS_WEIGHT} center:${MASK_CENTER_WEIGHT} valid:${VALID_LOSS_WEIGHT} grad:${GRAD_LOSS_WEIGHT} hole_grad:${HOLE_GRAD_LOSS_WEIGHT} boundary_grad:${BOUNDARY_GRAD_LOSS_WEIGHT} boundary_l1:${BOUNDARY_L1_LOSS_WEIGHT}"
echo "Using BOUNDARY_WIDTH=${BOUNDARY_WIDTH}"
echo "Using THRESHOLD_AMP_PERCENTILE=${THRESHOLD_AMP_PERCENTILE}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"

REAL_SPLIT_ARGS=()
if [[ -f "${REAL_SPLIT_JSON}" ]]; then
  echo "Using REAL_SPLIT_JSON=${REAL_SPLIT_JSON}"
  REAL_SPLIT_ARGS+=(--split_json "${REAL_SPLIT_JSON}")
else
  REAL_SPLIT_ARGS+=(--val_count 80 --shuffle_split)
fi

"${PYTHON_BIN}" train_real_raw9_flow_finetune.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --pretrained_checkpoint "${PRETRAIN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  "${REAL_SPLIT_ARGS[@]}" \
  --mask_mode threshold_amp_depth \
  --depth_unit auto \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.5 \
  --valid_max_depth 6.0 \
  --threshold_depth_min 0.5 \
  --threshold_depth_max 6.0 \
  --threshold_amp_percentile "${THRESHOLD_AMP_PERCENTILE}" \
  --threshold_mask_close "${THRESHOLD_MASK_CLOSE}" \
  --threshold_mask_dilate "${THRESHOLD_MASK_DILATE}" \
  --threshold_mask_min_component_area "${THRESHOLD_MASK_MIN_COMPONENT_AREA}" \
  --masks_per_sample 1 \
  --val_masks_per_sample 1 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --amplitude_mode iq6 \
  --raw9_transform "${RAW9_TRANSFORM}" \
  --hole_amplitude_mode keep_all \
  --mask_loss_weight "${MASK_LOSS_WEIGHT}" \
  --mask_center_weight "${MASK_CENTER_WEIGHT}" \
  --valid_loss_weight "${VALID_LOSS_WEIGHT}" \
  --grad_loss_weight "${GRAD_LOSS_WEIGHT}" \
  --hole_grad_loss_weight "${HOLE_GRAD_LOSS_WEIGHT}" \
  --boundary_grad_loss_weight "${BOUNDARY_GRAD_LOSS_WEIGHT}" \
  --boundary_l1_loss_weight "${BOUNDARY_L1_LOSS_WEIGHT}" \
  --boundary_width "${BOUNDARY_WIDTH}" \
  --selection_metric model_mask_mae \
  --replay_weight "${REPLAY_WEIGHT}" \
  --replay_split_json "${PBRT_SPLIT_JSON}" \
  --replay_split train \
  --replay_cache_dir "${PBRT_CACHE_DIR}" \
  --replay_batch_size "${REPLAY_BATCH_SIZE}" \
  --replay_num_workers "${REPLAY_NUM_WORKERS}" \
  --replay_max_samples "${REPLAY_MAX_SAMPLES}" \
  --amp
