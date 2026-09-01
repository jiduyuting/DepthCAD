#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
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
OUTPUT_DIR=${OUTPUT_DIR:-output/pbrt_real_threshold_amp_depth_finetune_e40_p8_keepamp}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-depthcad}
echo "Using REAL_ROOT=${REAL_ROOT}"
echo "Using RAW_DIR=${RAW_DIR}"
echo "Using DEPTH_DIR=${DEPTH_DIR}"

"${PYTHON_BIN}" scripts/flow/train_real_raw9_flow_finetune.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --pretrained_checkpoint "${PRETRAIN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --mask_mode threshold_amp_depth \
  --depth_unit auto \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.5 \
  --valid_max_depth 6.0 \
  --threshold_depth_min 0.5 \
  --threshold_depth_max 6.0 \
  --threshold_amp_percentile 8.0 \
  --threshold_mask_close 1 \
  --threshold_mask_min_component_area 8 \
  --masks_per_sample 1 \
  --val_masks_per_sample 1 \
  --val_count 80 \
  --shuffle_split \
  --epochs 40 \
  --batch_size 2 \
  --num_workers 4 \
  --lr 2e-5 \
  --amplitude_mode iq6 \
  --hole_amplitude_mode keep_all \
  --mask_loss_weight 1.0 \
  --valid_loss_weight 0.05 \
  --grad_loss_weight 0.05 \
  --selection_metric model_mask_mae \
  --amp
