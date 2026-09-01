# 泛化版 Realhole 训练记录

## 目标

这次不是只把真实空洞贴到同一批图上训练，而是把真实空洞组件库按 source stem 拆开：

- 训练 mask 模板来自一批真实 raw/depth 样本。
- 验证 mask 模板来自另一批 held-out 样本。
- synthetic depth 的 train/val 仍然沿用原来的 PBRT cache split。

这样可以降低“记住当前模板形状”的风险，更接近泛化验证。

## 代码改动

修改：

```text
scripts/train_synthetic_realhole_flow_pretrain.py
```

新增参数：

```text
--component_val_ratio
```

当 `--component_val_ratio > 0` 时，真实空洞/散斑组件库会按 `source` 拆分。训练集和验证集使用不同 source 的组件。

新增脚本：

```text
scripts/runs/run_synthetic_realhole_generalized_pretrain.sh
```

默认配置：

```text
mask_mode = real_hole_speckle_shapes
component_val_ratio = 0.25
real_speckle_component_ratio = 0.6
masks_per_sample = 1
val_masks_per_sample = 1
epochs = 20
```

## 已跑结果

5 epoch pilot：

```text
output/synthetic_realhole_flow_pretrain_generalized_split_e5
```

基础信息：

```text
train synthetic samples: 858
val synthetic samples: 95
real components total: 557
train components: 429
val components: 128
train source stems: 31
val source stems: 10
```

held-out val source stems：

```text
10, 12, 13, 17, 2, 23, 24, 37, 39, 6
```

验证集上 OpenCV NS anchor：

```text
anchor_mask_mae = 0.045258 m
```

每 epoch 结果：

| epoch | model_mask_mae | improve vs anchor | large_mask_mae | small_mask_mae | model_unmasked_mae |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.038696 | 14.50% | 0.039300 | 0.036127 | 0.004133 |
| 2 | 0.035842 | 20.81% | 0.036491 | 0.033080 | 0.004008 |
| 3 | 0.034421 | 23.94% | 0.035026 | 0.031852 | 0.003890 |
| 4 | 0.033667 | 25.61% | 0.034357 | 0.030739 | 0.004383 |
| 5 | 0.033231 | 26.58% | 0.033934 | 0.030242 | 0.004032 |

结论：

```text
在 held-out realhole/speckle 模板来源上，5 epoch pilot 已经稳定优于 OpenCV NS anchor。
这说明泛化式 realhole augmentation 有效，不只是记住训练模板。
```

## 重要限制

这个 checkpoint 是 `noisy_amp` 输入：

```text
corrupted depth + mask/confidence + amplitude features
```

所以它不是旧 `scripts/benchmark_far_pic_depth_completion.py` 里那个 depth-only checkpoint 的直接替换。要在 far_pic 上公平测它，需要使用 far_pic 对齐的 raw9/amplitude，或补一个 far_pic noisy_amp benchmark adapter。

## 推荐下一步

1. 跑完整 20 epoch：

```bash
EPOCHS=20 OUTPUT_DIR=output/synthetic_realhole_flow_pretrain_generalized_split_e20 \
  bash scripts/runs/run_synthetic_realhole_generalized_pretrain.sh
```

2. 用 e20 checkpoint 作为新的 ToF-aware pretrain，再微调真实 raw9：

```text
PRETRAIN_CKPT=output/synthetic_realhole_flow_pretrain_generalized_split_e20/best.pt
```

3. 补 far_pic noisy_amp benchmark adapter，把这个 `noisy_amp` checkpoint 和 ProPainter/OpenCV NS 在同一批 `synthetic_realhole_05/10/20` 上重新比较。
