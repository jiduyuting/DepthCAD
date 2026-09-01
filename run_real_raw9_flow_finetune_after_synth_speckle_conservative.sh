#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-output/synthetic_realhole_flow_pretrain_iq6_holefocus_speckle_from_after_synth_e40/best.pt}"
SPLIT_JSON="${SPLIT_JSON:-output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/split.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/real_raw9_flow_finetune_after_synth_speckle_conservative_e20_lr5e6}"

RAW_DIR="${RAW_DIR:-raw}"
DEPTH_DIR="${DEPTH_DIR:-depth}"

AMP_MODE="${AMP_MODE:-iq6}"

MASK_RATIO="${MASK_RATIO:-0.06}"
SPECKLE_COMPONENT_RATIO="${SPECKLE_COMPONENT_RATIO:-0.30}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

echo "Python: ${PYTHON_BIN}"
echo "Pretrained: ${PRETRAIN_CKPT}"
echo "Split JSON: ${SPLIT_JSON}"
echo "Output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" -u train_real_raw9_flow_finetune.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --pretrained_checkpoint "${PRETRAIN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --amplitude_mode "${AMP_MODE}" \
  --split_json "${SPLIT_JSON}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --amp \
  --mask_mode real_hole_speckle_shapes \
  --mask_ratio "${MASK_RATIO}" \
  --masks_per_sample 8 \
  --val_masks_per_sample 5 \
  --real_hole_min_area 24 \
  --real_hole_max_area 0 \
  --real_hole_min_overlap 0.6 \
  --real_hole_max_components 8 \
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
  --save_every 20

echo
echo "Suggested probe after training:"
echo "\"${PYTHON_BIN}\" infer_real_raw9_flow.py \\"
echo "  --raw_dir ${RAW_DIR} \\"
echo "  --depth_dir ${DEPTH_DIR} \\"
echo "  --checkpoint ${OUTPUT_DIR}/best.pt \\"
echo "  --output_dir ${OUTPUT_DIR}_probe \\"
echo "  --amplitude_mode ${AMP_MODE} \\"
echo "  --hole_mask_mode amp_speckle_cleaned \\"
echo "  --samples 33 34 41 42"
