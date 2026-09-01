# Depth Completion 实验总结草稿

## 实验目标

本实验研究 Kinect-style 空洞条件下，应该在 IQ/image 域进行 inpainting，还是在 depth 域进行 completion。前期 baseline 表明，SD/IQ inpainting 对几何深度恢复并不稳定，而 `DepthCAD + plane fill` 在 depth 域内已经能显著降低空洞区域误差。因此后续实验固定 `DepthCAD + plane fill` 作为强 classical baseline，并在其上训练 learned depth completion 模型。

最终模型采用 residual U-Net。输入为 depth-only 特征，包括 `depth_base`、`depth_depthcad`、`depth_noisy`、`hole_mask` 和 `confidence`。模型只在 hole 区域预测 residual，非 hole 区域保持 baseline 不变。

## 最终模型

```text
DepthCAD + plane fill
-> depth-only residual U-Net
-> hole-only binary residual
-> n1000 training cache
```

最终 checkpoint:

```text
output/depth_completion_unet_depth_n1000_hole_binary/best.pt
```

外部测试集:

```text
depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123
```

## 主结果

所有指标均在独立 `seed123` holdout set 上评估。`Valid MAE` 保持不变是预期行为，因为最终模型采用 hole-only residual，只修改 hole 区域。

| Method | Global MAE | Hole MAE | Hole Improve vs Base | Valid MAE | Better/Worse | Worst Delta |
|---|---:|---:|---:|---:|---:|---:|
| Plane base | 0.141459 | 0.505875 | - | 0.073716 | - | - |
| n100 depth+amp | 0.114576 | 0.334381 | 33.9% | 0.073716 | 73/27 | 0.601051 |
| n500 depth+amp | 0.097376 | 0.224651 | 55.6% | 0.073716 | 87/13 | 0.186482 |
| n1000 depth+amp | 0.085487 | 0.148804 | 70.6% | 0.073716 | 93/7 | 0.125969 |
| **n1000 depth-only** | **0.082614** | **0.130479** | **74.2%** | **0.073716** | **94/6** | **0.125982** |

结果显示，随着训练样本从 n100 增加到 n1000，模型在外部 holdout 上持续提升。最终 n1000 depth-only 模型将 plane baseline 的 hole MAE 从 `0.505875` 降低到 `0.130479`，相对改善 `74.2%`；global MAE 从 `0.141459` 降低到 `0.082614`。同时，regression 样本数从 n100 的 `27/100` 降低到 n1000 depth-only 的 `6/100`，说明数据规模提升同时改善了平均精度和鲁棒性。

## 输入特征消融

n1000 规模下，depth-only 模型优于 depth+amp 模型：

```text
n1000 depth+amp:
  Global MAE = 0.085487
  Hole MAE   = 0.148804

n1000 depth-only:
  Global MAE = 0.082614
  Hole MAE   = 0.130479
```

这说明在当前设置下，主要有效信息来自 depth-domain geometry。Amplitude 特征在小数据规模下可能提供辅助线索，但在 n1000 规模下没有带来额外收益，反而可能引入噪声。因此最终模型采用 depth-only 输入。

## Residual 融合策略消融

在 n1000 depth-only checkpoint 上比较不同推理融合策略：

| Strategy | Global MAE | Hole MAE | Hole Improve | Better/Worse | Worst Delta | P95 Delta | Median Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| **binary** | **0.082614** | **0.130479** | **74.2%** | **94/6** | 0.125982 | 0.011075 | -0.181214 |
| soft_hole_distance `b1/r4` | 0.104705 | 0.271405 | 46.3% | 93/7 | 0.086222 | 0.006979 | -0.122895 |
| residual scale `0.75` | 0.092745 | 0.195105 | 61.4% | 95/5 | 0.074628 | -0.005398 | -0.153651 |

保守融合策略可以降低最坏 regression，但会显著削弱 hole 内部有效修正，导致整体 hole MAE 明显变差。因此最终采用 hole-only binary residual。该策略在平均精度和鲁棒性之间取得最佳平衡。

## 定性结果

精选定性案例保存在：

```text
output/final_depth_completion_package
```

建议展示 4 到 5 个案例：

| Case | Type | Base Hole MAE | Model Hole MAE | Delta |
|---|---|---:|---:|---:|
| `breakfast/1/174` | strong improvement | 4.3508 | 0.6613 | -3.6895 |
| `bathroom/0/1` | typical improvement | 0.3491 | 0.1167 | -0.2324 |
| `white-room/0/55` | moderate improvement | 0.2410 | 0.1553 | -0.0856 |
| `breakfast/0/151` | mild regression | 0.7821 | 0.9081 | +0.1260 |
| `contemporary-bathroom/0/173` | low-error-base regression | 0.0263 | 0.1075 | +0.0812 |

图像目录：

```text
output/final_depth_completion_package/selected_eval/visualizations
```

## 结论

实验结果表明，Kinect-style hole completion 的核心问题应在 depth 域解决，而不是在 IQ/image 域做生成式 inpainting。传统 plane fill 已经显著优于 SD inpainting 类方法；在此基础上，使用 depth-only residual U-Net 可以进一步提升空洞区域补全精度。随着训练样本从 n100 扩展到 n1000，模型在独立 seed123 holdout 上持续提升，hole MAE 从 `0.334381` 降至 `0.130479`，regression 样本数从 `27/100` 降至 `6/100`。最终 n1000 depth-only 模型相对于 plane baseline 将 hole MAE 从 `0.505875` 降至 `0.130479`，改善 `74.2%`。

## 一句话总结

```text
Depth-domain learned completion is the correct technical route: a n1000 depth-only residual U-Net on top of DepthCAD + plane fill reduces seed123 holdout hole MAE by 74.2% over the classical plane baseline while preserving non-hole regions unchanged.
```

