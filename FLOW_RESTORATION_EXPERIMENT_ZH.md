# Conditional Flow Depth Restoration

这个版本是为了替代之前的 pseudo-RGB SD inpainting，把生成模型放回正确的变量空间：

```text
condition = noisy depth + NS anchor + hole mask + confidence (+ optional amplitude)
state x_t = anchor 和 GT depth 之间的中间 depth
model(condition, x_t, t) -> velocity
sampling: 从 anchor 出发，沿 velocity 多步积分到 restored dense depth
```

它不是 `DepthCAD -> SD Inpaint`，也不是把 IQ 通道伪装成 RGB。它是一个单模型 depth-domain restoration / conditional rectified-flow baseline。

## 新增文件

```text
train_depth_flow_restoration.py
eval_depth_flow_restoration.py
```

训练脚本复用 `train_depth_restoration.py` 里的 cache dataset，所以 cache 不需要重新生成。

## 推荐第一组实验

先跑最稳的纯 rectified flow，不加随机 bridge noise：

```bash
python -u train_depth_flow_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12 \
  --train_list output/splits_n1000_plane_r12_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_exclude_seed123/val.txt \
  --output_dir output/depth_flow_restoration_noisy_ns_n1000 \
  --input_mode noisy \
  --anchor_mode noisy_ns \
  --anchor_inpaint_radius 15 \
  --epochs 120 \
  --batch_size 8 \
  --num_workers 4 \
  --base_channels 32 \
  --sample_steps 8 \
  --time_channels 16 \
  --bridge_noise 0.0 \
  --hole_weight 5.0 \
  --valid_weight 1.0 \
  --velocity_weight 1.0 \
  --recon_weight 1.0 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --selection_metric global \
  --amp
```

这里的 `sample_steps 8` 是训练期间验证用的 Euler 步数。训练完成后可以在 eval 时改成 16 或 32，看多步采样是否继续提升。

## Holdout Evaluation

和之前主结果一样，用 `seed123` holdout：

```bash
python -u eval_depth_flow_restoration.py \
  --checkpoint output/depth_flow_restoration_noisy_ns_n1000/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_flow_restoration_noisy_ns_n1000/eval_seed123_steps16 \
  --batch_size 8 \
  --num_workers 4 \
  --sample_steps 16 \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

主要和下面这个结果比较：

```text
output/depth_restoration_unet_noisy_ns_n1000/eval_seed123/summary.json
```

当前单模型 residual U-Net 主结果：

```text
Global MAE = 0.056383
Hole MAE   = 0.114204
Valid MAE  = 0.045635
```

flow 版本如果能低于这个，说明生成式 flow 确实带来了收益；如果差不多或略差，它仍然是一个重要实验，因为它回答了导师的问题：更生成式的 depth-domain 模型是否比普通 residual regression 更强。

## 第二组实验：加入 Amplitude

如果第一组没有明显超过主模型，可以跑 amplitude 条件：

```bash
python -u train_depth_flow_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12 \
  --train_list output/splits_n1000_plane_r12_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_exclude_seed123/val.txt \
  --output_dir output/depth_flow_restoration_noisy_amp_ns_n1000 \
  --input_mode noisy_amp \
  --anchor_mode noisy_ns \
  --anchor_inpaint_radius 15 \
  --epochs 120 \
  --batch_size 8 \
  --num_workers 4 \
  --base_channels 32 \
  --sample_steps 8 \
  --time_channels 16 \
  --bridge_noise 0.0 \
  --hole_weight 5.0 \
  --valid_weight 1.0 \
  --velocity_weight 1.0 \
  --recon_weight 1.0 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --selection_metric global \
  --amp
```

之前 restoration / completion 实验里 amplitude 不一定稳定增益，所以这组是 ablation，不建议一开始就作为主方法。

## 第三组实验：小 bridge noise

如果想更接近 diffusion / stochastic interpolant，可以小心尝试：

```text
--bridge_noise 0.02
```

不建议第一组就开大噪声。当前数据量只有 n1000，`0.0` 的 rectified flow 更稳；`0.02` 只能作为后续 ablation。

## 第四组实验：Flow + Endpoint Auxiliary

如果 vanilla flow 明显弱于 residual restoration，优先跑这一组。

原因是当前任务从 deterministic anchor 出发，最终只关心 restored endpoint。纯 flow 会把容量分散到整条 `anchor -> GT` 路径上，而 endpoint auxiliary 会强制同一个模型在 `t=0` 时也能直接执行 `anchor -> GT` restoration。

```bash
python -u train_depth_flow_restoration.py \
  --cache_dir depth_completion_cache/depth_cache_0515_n1000_plane_r12 \
  --train_list output/splits_n1000_plane_r12_exclude_seed123/train.txt \
  --val_list output/splits_n1000_plane_r12_exclude_seed123/val.txt \
  --output_dir output/depth_flow_restoration_noisy_ns_n1000_endpoint \
  --input_mode noisy \
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

训练后先评估 endpoint direct output：

```bash
python -u eval_depth_flow_restoration.py \
  --checkpoint output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_endpoint \
  --batch_size 8 \
  --num_workers 4 \
  --sampling_mode endpoint \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

再评估同一个 checkpoint 的 Euler flow output：

```bash
python -u eval_depth_flow_restoration.py \
  --checkpoint output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_euler16 \
  --batch_size 8 \
  --num_workers 4 \
  --sampling_mode euler \
  --sample_steps 16
```

## 判断标准

优先看 independent seed123 holdout：

```text
model_global_mae
model_hole_mae
model_valid_mae
```

然后看 ranked visualization：

```text
output/depth_flow_restoration_noisy_ns_n1000/eval_seed123_steps16/visualizations
```

需要重点检查：

- hole 边界是否比 residual U-Net 更平滑或更贴合结构；
- valid 区域有没有被过度修改；
- worst hole regression 是否比原来的 single restoration 更严重；
- `sample_steps 8/16/32` 是否有稳定趋势。

## 汇报时的定位

这个实验的定位不是继续救旧的 SD inpainting，而是：

```text
把 diffusion/flow 从 RGB image prior 改成 depth-domain physical restoration prior。
```

如果结果好：

```text
Flow-based depth restoration outperforms residual regression, showing that generative depth priors help when applied in the correct physical domain.
```

如果结果一般：

```text
Pseudo-RGB SD failed because of wrong representation; depth-domain flow is the correct generative formulation, but with n1000 data the simpler residual restoration baseline remains stronger/stabler.
```
