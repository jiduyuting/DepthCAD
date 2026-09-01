#!/usr/bin/env bash
set -euo pipefail

REAL_ROOT=${REAL_ROOT:-/data/pre_student/hcy/datasets/pbrt/Real}
VAL_SCENE=${VAL_SCENE:?Set VAL_SCENE, for example library or lab307}

PYTHON_BIN=${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-output/pbrt_real_threshold_amp_depth_finetune_replay_auto_boundary_v2b_e10_p8_keepamp_rw030/best.pt}
RAW9_TRANSFORM=${RAW9_TRANSFORM:-auto}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-scene_holdout_${VAL_SCENE}_boundary_v2b_e12_rw030}
SPLIT_JSON=${SPLIT_JSON:-output/${EXPERIMENT_NAME}/split.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/${EXPERIMENT_NAME}}

EPOCHS=${EPOCHS:-12}
LR=${LR:-3e-6}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.30}
HOLE_GRAD_LOSS_WEIGHT=${HOLE_GRAD_LOSS_WEIGHT:-0.10}
BOUNDARY_GRAD_LOSS_WEIGHT=${BOUNDARY_GRAD_LOSS_WEIGHT:-0.05}
BOUNDARY_L1_LOSS_WEIGHT=${BOUNDARY_L1_LOSS_WEIGHT:-0.10}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-3}

"${PYTHON_BIN}" make_real_scene_holdout_split.py \
  --raw_dir "${REAL_ROOT}/noise" \
  --depth_dir "${REAL_ROOT}/depth" \
  --val_scenes "${VAL_SCENE}" \
  --output_json "${SPLIT_JSON}"

export REAL_ROOT
export PRETRAIN_CKPT
export REAL_SPLIT_JSON="${SPLIT_JSON}"
export OUTPUT_DIR
export RAW9_TRANSFORM
export EPOCHS
export LR
export REPLAY_WEIGHT
export HOLE_GRAD_LOSS_WEIGHT
export BOUNDARY_GRAD_LOSS_WEIGHT
export BOUNDARY_L1_LOSS_WEIGHT
export BOUNDARY_WIDTH

bash run_pbrt_real_threshold_amp_depth_finetune_replay.sh

MODEL_DIR="${OUTPUT_DIR}" \
SPLIT_JSON="${SPLIT_JSON}" \
RAW9_TRANSFORM="${RAW9_TRANSFORM}" \
bash run_pbrt_real_boundary_v2_eval_suite.sh
