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
CHECKPOINT=${CHECKPOINT:-output/pbrt_real_threshold_amp_depth_finetune_e40_p8_keepamp/best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-output/pbrt_real_threshold_amp_depth_selftest_p8_keepamp}
SPLIT_JSON=${SPLIT_JSON:-}
EVAL_SPLIT=${EVAL_SPLIT:-all}
RAW9_TRANSFORM=${RAW9_TRANSFORM:-none}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-depthcad}
echo "Using REAL_ROOT=${REAL_ROOT}"
echo "Using RAW_DIR=${RAW_DIR}"
echo "Using DEPTH_DIR=${DEPTH_DIR}"
echo "Using EVAL_SPLIT=${EVAL_SPLIT}"
echo "Using RAW9_TRANSFORM=${RAW9_TRANSFORM}"

SPLIT_ARGS=()
if [[ -n "${SPLIT_JSON}" ]]; then
  echo "Using SPLIT_JSON=${SPLIT_JSON}"
  SPLIT_ARGS+=(--split_json "${SPLIT_JSON}" --eval_split "${EVAL_SPLIT}")
elif [[ "${EVAL_SPLIT}" != "all" ]]; then
  echo "EVAL_SPLIT=${EVAL_SPLIT} requires SPLIT_JSON" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/real_raw9_masked_self_test.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  "${SPLIT_ARGS[@]}" \
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
  --num_masks_per_sample 1 \
  --amplitude_mode iq6 \
  --raw9_transform "${RAW9_TRANSFORM}" \
  --hole_amplitude_mode keep_all \
  --sampling_mode endpoint \
  --visualize \
  --vis_max_samples 24
