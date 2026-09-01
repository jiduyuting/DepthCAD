#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"

RAW_DIR="${RAW_DIR:-raw}"
DEPTH_DIR="${DEPTH_DIR:-depth}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/best.pt}"
SPLIT_JSON="${SPLIT_JSON:-output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/split.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/real_raw9_propagation_refine_v2_iq6_e40}"

echo "Python: ${PYTHON_BIN}"
echo "Pretrained anchor source: ${PRETRAIN_CKPT}"
echo "Split JSON: ${SPLIT_JSON}"
echo "Output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" -u train_real_raw9_propagation_refine.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --pretrained_checkpoint "${PRETRAIN_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --amplitude_mode iq6 \
  --split_json "${SPLIT_JSON}" \
  --epochs 40 \
  --batch_size 4 \
  --num_workers 0 \
  --lr 2e-5 \
  --amp \
  --mask_mode real_hole_speckle_shapes \
  --mask_ratio 0.08 \
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
  --real_speckle_component_ratio 0.30 \
  --hole_depth_threshold 1.0 \
  --valid_min_depth 1.0 \
  --valid_max_depth 9.9 \
  --post_clip_mode valid_range \
  --post_clip_percentiles 0.5 99.5 \
  --base_channels 32 \
  --res_blocks 1 \
  --propagation_steps 4 \
  --propagation_hidden_scale 1.0 \
  --refine_dilate_radius 3 \
  --residual_scale 1.5 \
  --mask_loss_weight 4.0 \
  --mask_center_weight 2.0 \
  --valid_loss_weight 0.1 \
  --coarse_loss_weight 0.5 \
  --grad_loss_weight 0.0 \
  --hole_grad_loss_weight 0.25 \
  --boundary_grad_loss_weight 0.5 \
  --boundary_l1_loss_weight 0.05 \
  --boundary_width 3 \
  --refine_consistency_weight 0.05 \
  --eval_component_area_threshold 700 \
  --selection_metric model_mask_mae \
  --log_every 20 \
  --save_every 20
