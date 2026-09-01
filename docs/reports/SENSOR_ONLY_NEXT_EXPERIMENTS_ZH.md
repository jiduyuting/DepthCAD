# Sensor-only Next Experiments

当前没有真实对齐 RGB，因此后续对比应该继续保持 RGB-free / sensor-only setting。

## 1. Flow Endpoint + Amplitude

现有 cache 已经包含 amplitude：

```text
noisy_amplitude
noisy_amplitude_mean
denoised_amplitude
denoised_amplitude_mean
```

主方法目前只用：

```text
noisy depth + NS anchor + hole mask + confidence
```

下一组直接加 noisy amplitude：

```bash
python -u train_depth_flow_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12 \
  --train_list output/splits_n1000_plane_r12_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_exclude_seed123/val.txt \
  --output_dir output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2 \
  --input_mode noisy_amp \
  --anchor_mode noisy_ns \
  --anchor_inpaint_radius 15 \
  --epochs 120 \
  --batch_size 8 \
  --num_workers 4 \
  --base_channels 32 \
  --sample_steps 8 \
  --eval_sampling_mode endpoint \
  --time_channels 16 \
  --bridge_noise 0.0 \
  --hole_weight 5.0 \
  --valid_weight 1.0 \
  --velocity_weight 1.0 \
  --recon_weight 1.0 \
  --endpoint_weight 2.0 \
  --endpoint_grad_weight 0.5 \
  --endpoint_smooth_weight 0.02 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --selection_metric global \
  --amp
```

Evaluation:

```bash
python -u eval_depth_flow_restoration.py \
  --checkpoint output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2/eval_seed123_endpoint \
  --batch_size 8 \
  --num_workers 4 \
  --sampling_mode endpoint \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

整理 ranked cases:

```bash
python scripts/analysis/organize_ranked_visualizations.py \
  --eval_dir output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2/eval_seed123_endpoint \
  --output_prefix output/depth_flow_restoration_summary_final/ranked_cases_endpoint_noisy_amp_w2 \
  --baseline anchor \
  --region hole \
  --top_k 6
```

## 2. Flow Endpoint + IQ / IQ + Amplitude

代码现在支持：

```text
--input_mode noisy_iq
--input_mode noisy_iq_amp
```

但是现有 n1000 cache 没有 `noisy_iq`，需要重新生成带 IQ 的 cache。

### Generate IQ Training Cache

```bash
python -u apply_kinect_holes_and_eval.py \
  --num_samples 1000 \
  --depth_fill_method plane \
  --depth_fill_radius 15 \
  --plane_max_ring_radius 12 \
  --plane_min_boundary_points 12 \
  --run_name depth_cache_0515_n1000_plane_r12_iq \
  --save_depth_completion_cache \
  --depth_cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12_iq \
  --depth_cache_save_iq \
  --seed 42
```

### Generate IQ Holdout Cache

```bash
python -u apply_kinect_holes_and_eval.py \
  --num_samples 100 \
  --depth_fill_method plane \
  --depth_fill_radius 15 \
  --plane_max_ring_radius 12 \
  --plane_min_boundary_points 12 \
  --run_name depth_cache_0514_n100_plane_r12_seed123_iq \
  --save_depth_completion_cache \
  --depth_cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq \
  --depth_cache_save_iq \
  --seed 123
```

### Create IQ Split

```bash
python make_depth_completion_splits.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12_iq \
  --holdout_cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq \
  --output_dir output/splits_n1000_plane_r12_iq_exclude_seed123 \
  --val_ratio 0.1 \
  --seed 42
```

### Train IQ + Amplitude Flow Endpoint

```bash
python -u train_depth_flow_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12_iq \
  --train_list output/splits_n1000_plane_r12_iq_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_iq_exclude_seed123/val.txt \
  --output_dir output/depth_flow_restoration_noisy_iq_amp_ns_n1000_endpoint_w2 \
  --input_mode noisy_iq_amp \
  --anchor_mode noisy_ns \
  --anchor_inpaint_radius 15 \
  --epochs 120 \
  --batch_size 8 \
  --num_workers 4 \
  --base_channels 32 \
  --sample_steps 8 \
  --eval_sampling_mode endpoint \
  --time_channels 16 \
  --bridge_noise 0.0 \
  --hole_weight 5.0 \
  --valid_weight 1.0 \
  --velocity_weight 1.0 \
  --recon_weight 1.0 \
  --endpoint_weight 2.0 \
  --endpoint_grad_weight 0.5 \
  --endpoint_smooth_weight 0.02 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --selection_metric global \
  --amp
```

Evaluation:

```bash
python -u eval_depth_flow_restoration.py \
  --checkpoint output/depth_flow_restoration_noisy_iq_amp_ns_n1000_endpoint_w2/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq \
  --split all \
  --output_dir output/depth_flow_restoration_noisy_iq_amp_ns_n1000_endpoint_w2/eval_seed123_endpoint \
  --batch_size 8 \
  --num_workers 4 \
  --sampling_mode endpoint \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

## 3. DepthCAD-HoleAware Baseline

This is a heavier baseline because it requires training/evaluating DepthCAD on noisy/holey IQ rather than using the cached depth-restoration tensors.

The purpose is to answer:

```text
If DepthCAD itself is trained with hole-corrupted inputs and mask/confidence, can it solve denoising + completion?
```

Fair input:

```text
noisy/holey IQ + confidence or hole mask
```

Target:

```text
clean/ideal IQ
```

Then convert predicted IQ to depth with the same depth estimator and evaluate with:

```text
global MAE
hole MAE
valid MAE
```

Important: do not call pseudo-RGB SD inpainting a fair modern baseline. It remains a diagnostic failure case, not the main comparison.

## Priority

Recommended order:

1. Run `Flow endpoint + amplitude` because no cache regeneration is needed.
2. Generate IQ cache and run `Flow endpoint + IQ + amplitude`.
3. Only then spend time on `DepthCAD-HoleAware`, because it is heavier and less directly comparable to the final depth-domain model.
