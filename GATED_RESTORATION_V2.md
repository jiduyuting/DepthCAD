# Gated Residual Restoration V2

This note describes the architecture v2 added after the final v1 single restoration model.

## Motivation

The v1 model uses a residual U-Net:

```text
pred = anchor + residual
```

This is strong, but the backbone is a standard U-Net. V2 keeps the same depth-domain restoration formulation but adds an anchor-guided residual gate:

```text
residual, gate_logit = model(x)
gate = sigmoid(gate_logit)
pred = anchor + gate * residual
```

The goal is to let the model learn where it should strongly correct the NS anchor and where it should preserve it.

## Code Changes

- `train_depth_completion.py`
  - `ResidualUNet` now supports `out_channels`.
- `train_depth_restoration.py`
  - Adds `--prediction_mode residual|gated_residual`.
  - Adds gated residual prediction support.
  - Keeps default `prediction_mode=residual` for backward compatibility.
- `eval_depth_restoration.py`
  - Loads old residual checkpoints and new gated residual checkpoints automatically from checkpoint args.

## Recommended V2 Training Command

```bash
python -u train_depth_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12 \
  --train_list output/splits_n1000_plane_r12_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_exclude_seed123/val.txt \
  --output_dir output/depth_restoration_unet_noisy_ns_n1000_gated \
  --input_mode noisy \
  --anchor_mode noisy_ns \
  --anchor_inpaint_radius 15 \
  --prediction_mode gated_residual \
  --gate_bias_init 2.0 \
  --selection_metric global \
  --epochs 120 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --base_channels 32 \
  --hole_weight 5 \
  --valid_weight 1 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --amp
```

## Optional Gate Prior Ablation

Use this only after the plain gated version is evaluated. It weakly encourages larger corrections in holes and smaller corrections outside holes:

```bash
python -u train_depth_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12 \
  --train_list output/splits_n1000_plane_r12_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_exclude_seed123/val.txt \
  --output_dir output/depth_restoration_unet_noisy_ns_n1000_gated_prior002 \
  --input_mode noisy \
  --anchor_mode noisy_ns \
  --anchor_inpaint_radius 15 \
  --prediction_mode gated_residual \
  --gate_bias_init 2.0 \
  --gate_prior_weight 0.02 \
  --gate_hole_target 1.0 \
  --gate_valid_target 0.25 \
  --selection_metric global \
  --epochs 120 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --base_channels 32 \
  --hole_weight 5 \
  --valid_weight 1 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --amp
```

## Evaluation Command

```bash
python -u eval_depth_restoration.py \
  --checkpoint output/depth_restoration_unet_noisy_ns_n1000_gated/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_restoration_unet_noisy_ns_n1000_gated/eval_seed123 \
  --batch_size 8 \
  --num_workers 4 \
  --visualize \
  --vis_max_samples 20
```

Ranked visualization:

```bash
python -u eval_depth_restoration.py \
  --checkpoint output/depth_restoration_unet_noisy_ns_n1000_gated/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_restoration_unet_noisy_ns_n1000_gated/eval_seed123_ranked \
  --batch_size 8 \
  --num_workers 4 \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

## Baseline To Beat

Current v1 single restoration on `seed123`:

| Model | Global MAE | Hole MAE | Valid MAE |
|---|---:|---:|---:|
| v1 residual | 0.056383 | 0.114204 | 0.045635 |

V2 is useful as the new main method only if it improves at least one of these without a clear regression in the others. If it is similar or worse, v1 remains the final model and v2 becomes an architecture ablation.
