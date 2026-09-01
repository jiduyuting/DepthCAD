#!/usr/bin/env bash
set -euo pipefail

PRETRAIN_CKPT=${PRETRAIN_CKPT:-output/pbrt_real_threshold_amp_depth_finetune_replay_fliplr_e30_p8_keepamp_rw030/best.pt}
REAL_SPLIT_JSON=${REAL_SPLIT_JSON:-output/pbrt_real_threshold_amp_depth_finetune_replay_fliplr_e30_p8_keepamp_rw030/split.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/pbrt_real_threshold_amp_depth_finetune_replay_auto_boundary_v2_e15_p8_keepamp_rw030}
RAW9_TRANSFORM=${RAW9_TRANSFORM:-auto}

EPOCHS=${EPOCHS:-15}
LR=${LR:-5e-6}
REPLAY_WEIGHT=${REPLAY_WEIGHT:-0.30}

HOLE_GRAD_LOSS_WEIGHT=${HOLE_GRAD_LOSS_WEIGHT:-0.05}
BOUNDARY_GRAD_LOSS_WEIGHT=${BOUNDARY_GRAD_LOSS_WEIGHT:-0.05}
BOUNDARY_L1_LOSS_WEIGHT=${BOUNDARY_L1_LOSS_WEIGHT:-0.05}
BOUNDARY_WIDTH=${BOUNDARY_WIDTH:-3}

export PRETRAIN_CKPT
export REAL_SPLIT_JSON
export OUTPUT_DIR
export RAW9_TRANSFORM
export EPOCHS
export LR
export REPLAY_WEIGHT
export HOLE_GRAD_LOSS_WEIGHT
export BOUNDARY_GRAD_LOSS_WEIGHT
export BOUNDARY_L1_LOSS_WEIGHT
export BOUNDARY_WIDTH

exec bash run_pbrt_real_threshold_amp_depth_finetune_replay.sh
