#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"

RAW_DIR="${RAW_DIR:-raw}"
DEPTH_DIR="${DEPTH_DIR:-depth}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-output/synthetic_realhole_flow_pretrain_generalized_split_e20/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20}"

AMP_MODE="${AMP_MODE:-iq6}"
CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
DEFAULT_SPLIT_JSON="output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/split.json"
if [ -f "${DEFAULT_SPLIT_JSON}" ]; then
  SPLIT_JSON="${SPLIT_JSON:-${DEFAULT_SPLIT_JSON}}"
else
  SPLIT_JSON="${SPLIT_JSON:-}"
fi
VAL_COUNT="${VAL_COUNT:-8}"

MASK_RATIO="${MASK_RATIO:-0.08}"
SPECKLE_COMPONENT_RATIO="${SPECKLE_COMPONENT_RATIO:-0.35}"
SAT_AUG_PROB="${SAT_AUG_PROB:-0.75}"
VAL_SAT_AUG_PROB="${VAL_SAT_AUG_PROB:-1.0}"

EPOCHS="${EPOCHS:-30}"
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MASKS_PER_SAMPLE="${MASKS_PER_SAMPLE:-8}"
VAL_MASKS_PER_SAMPLE="${VAL_MASKS_PER_SAMPLE:-5}"

SPLIT_ARGS=()
if [ -n "${SPLIT_JSON}" ]; then
  SPLIT_ARGS=(--split_json "${SPLIT_JSON}")
else
  SPLIT_ARGS=(--val_count "${VAL_COUNT}")
fi

echo "Python: ${PYTHON_BIN}"
echo "Raw/depth: ${RAW_DIR} + ${DEPTH_DIR}"
echo "Pretrained: ${PRETRAIN_CKPT}"
echo "Output: ${OUTPUT_DIR}"
echo "Split args: ${SPLIT_ARGS[*]}"
echo "Saturation aug: train=${SAT_AUG_PROB} val=${VAL_SAT_AUG_PROB}"

"${PYTHON_BIN}" -u train_real_raw9_flow_finetune.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --pretrained_checkpoint "${PRETRAIN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --amplitude_mode "${AMP_MODE}" \
  "${SPLIT_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --amp \
  --device "${CUDA_DEVICE}" \
  --mask_mode real_hole_speckle_shapes \
  --mask_ratio "${MASK_RATIO}" \
  --masks_per_sample "${MASKS_PER_SAMPLE}" \
  --val_masks_per_sample "${VAL_MASKS_PER_SAMPLE}" \
  --real_hole_min_area 24 \
  --real_hole_max_area 0 \
  --real_hole_min_overlap 0.6 \
  --real_hole_max_components 10 \
  --real_hole_exclude_self \
  --max_mask_retries 8 \
  --clean_outlier_abs 0.35 \
  --clean_outlier_mad_scale 6.0 \
  --clean_median_ksize 7 \
  --clean_dilate 0 \
  --clean_min_component_area 4 \
  --speckle_window 11 \
  --speckle_density_threshold 0.10 \
  --speckle_residual_abs 0.18 \
  --speckle_link_radius 2 \
  --speckle_min_component_area 4 \
  --speckle_max_component_area 30000 \
  --speckle_max_bbox_side 260 \
  --speckle_amp_ring_radius 7 \
  --speckle_amp_ratio_min 2.5 \
  --speckle_amp_delta_min 4000.0 \
  --speckle_amp_abs_min 8000.0 \
  --real_speckle_train_min_area 6 \
  --real_speckle_train_max_area 0 \
  --real_speckle_component_ratio "${SPECKLE_COMPONENT_RATIO}" \
  --include_saturation_components \
  --sat_component_channels 2 5 8 \
  --sat_component_clip_value 65535.0 \
  --sat_component_clip_margin 1.0 \
  --sat_component_min_area 12 \
  --sat_component_max_area 30000 \
  --sat_component_dilate 1 \
  --saturation_aug_prob "${SAT_AUG_PROB}" \
  --val_saturation_aug_prob "${VAL_SAT_AUG_PROB}" \
  --saturation_aug_channels 2 5 8 \
  --saturation_aug_clip_value 65535.0 \
  --saturation_aug_jitter 0.0 \
  --saturation_aug_dilate 0 \
  --saturation_aug_depth_mode zero \
  --saturation_aug_keep_amplitude \
  --hole_amplitude_mode zero \
  --hole_depth_threshold 1.0 \
  --valid_min_depth 1.0 \
  --valid_max_depth 9.9 \
  --post_clip_mode valid_range \
  --post_clip_percentiles 0.5 99.5 \
  --mask_loss_weight 4.0 \
  --mask_center_weight 2.0 \
  --valid_loss_weight 0.02 \
  --grad_loss_weight 0.0 \
  --hole_grad_loss_weight 0.25 \
  --boundary_grad_loss_weight 0.5 \
  --boundary_l1_loss_weight 0.05 \
  --boundary_width 3 \
  --eval_component_area_threshold 700 \
  --selection_metric model_mask_mae \
  --log_every 20 \
  --save_every 10

echo
echo "Suggested real-overexposed probe:"
echo "\"${PYTHON_BIN}\" infer_real_raw9_flow.py \\"
echo "  --raw_dir ${RAW_DIR} \\"
echo "  --depth_dir ${DEPTH_DIR} \\"
echo "  --checkpoint ${OUTPUT_DIR}/best.pt \\"
echo "  --output_dir ${OUTPUT_DIR}_probe \\"
echo "  --amplitude_mode ${AMP_MODE} \\"
echo "  --hole_mask_mode amp_speckle_cleaned \\"
echo "  --hole_amplitude_mode keep_all \\"
echo "  --samples 33 34 41 42"
