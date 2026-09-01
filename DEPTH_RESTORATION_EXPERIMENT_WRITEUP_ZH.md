# Single Depth Restoration 实验总结草稿

## 实验目标

前期实验已经说明，对于 Kinect-style depth holes，单纯在 IQ/image 域做 inpainting 并不是最优路线。Depth-domain 的传统补全方法，例如 NS 或 plane fill，已经明显强于 SD/IQ inpainting。进一步实验还显示，在 `DepthCAD + plane` 之后再接 learned completion 虽然有效，但会形成两个 learned stages，方法复杂度较高，也不符合“最好只有一个模型”的要求。

因此，最终方法改为 single depth restoration model：模型直接从 degraded depth 恢复 clean dense depth，同时处理空洞补全和有效区域深度恢复。

## 最终方法

最终模型采用 mask-aware residual U-Net。输入包括：

```text
noisy depth
NS depth anchor
hole mask
confidence
```

输出为：

```text
clean dense depth
```

其中 NS depth anchor 是由传统深度域 inpainting 方法得到的确定性初值，不是第二个神经网络。最终 pipeline 中只有一个 learned model。

训练损失包括：

```text
L1_hole + L1_valid + gradient consistency + smoothness regularization
```

该设计的目的不是让网络从零生成深度，而是让网络在一个粗略 dense depth anchor 上学习如何修正空洞区域、边界区域以及有效深度区域的误差。

## 实验设置

训练数据：

```text
depth_completion_cache/depth_cache_0515_n1000_plane_r12
```

训练/验证划分：

```text
output/splits_n1000_plane_r12_exclude_seed123
```

外部测试集：

```text
depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123
```

最终 checkpoint：

```text
output/depth_restoration_unet_noisy_ns_n1000/best.pt
```

最终评估目录：

```text
output/depth_restoration_unet_noisy_ns_n1000/eval_seed123
```

## 主结果

所有结果均在独立 `seed123` holdout set 上评估。

| Method | Learned Models | Uses DepthCAD | Global MAE | Hole MAE | Valid MAE |
|---|---:|---|---:|---:|---:|
| Noisy | 0 | No | 0.503548 | 2.911191 | 0.055985 |
| NS Anchor | 0 | No | 0.104833 | 0.367608 | 0.055985 |
| DepthCAD/Plane Base | 1 | Yes | 0.141459 | 0.505875 | 0.073716 |
| Two-stage Completion | 2 | Yes | 0.082614 | 0.130479 | 0.073716 |
| Ours Single Restoration | 1 | No | 0.056383 | 0.114204 | 0.045635 |

结果说明，最终 single restoration model 同时取得最低的 Global MAE、Hole MAE 和 Valid MAE。相比前一阶段的 two-stage completion，最终模型不仅更简单，而且三个指标都更好。

## 和主要 baseline 的对比

相比 Noisy 输入：

```text
Global: 0.503548 -> 0.056383，提升约 88.8%
Hole:   2.911191 -> 0.114204，提升约 96.1%
Valid:  0.055985 -> 0.045635，提升约 18.5%
```

相比 NS Anchor：

```text
Global: 0.104833 -> 0.056383，提升约 46.2%
Hole:   0.367608 -> 0.114204，提升约 68.9%
Valid:  0.055985 -> 0.045635，提升约 18.5%
```

相比 DepthCAD/Plane Base：

```text
Global: 0.141459 -> 0.056383，提升约 60.1%
Hole:   0.505875 -> 0.114204，提升约 77.4%
Valid:  0.073716 -> 0.045635，提升约 38.1%
```

相比 Two-stage Completion：

```text
Global: 0.082614 -> 0.056383，提升约 31.8%
Hole:   0.130479 -> 0.114204，提升约 12.5%
Valid:  0.073716 -> 0.045635，提升约 38.1%
```

因此，最终模型不只是补洞能力更强，也确实改善了 non-hole valid region 的深度误差。

## Ablation

| Variant | Global MAE | Hole MAE | Valid MAE | Hole Improve vs Anchor/Base | Better/Worse | Worst Delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 30-epoch pilot | 0.071754 | 0.147794 | 0.057618 | 59.8% | 97/3 | 0.005796 | 训练不足 |
| Ours main | 0.056383 | 0.114204 | 0.045635 | 68.9% | 99/1 | 0.001890 | 最优单模型 |
| valid2_anchor02 | 0.060767 | 0.117253 | 0.050266 | 68.1% | 96/4 | 0.064092 | 过强约束导致下降 |
| gated residual | 0.058415 | 0.108534 | 0.049098 | 70.5% | 98/2 | 0.001014 | hole-focused 架构变体 |
| Two-stage completion | 0.082614 | 0.130479 | 0.073716 | 74.2% | 94/6 | 0.125982 | 强两阶段 baseline |

`valid2_anchor02` 的结果说明，额外提高 valid loss 权重并加入 anchor regularization 并没有进一步改善有效区域，反而使 global、hole 和 valid 三项指标都下降。这说明主模型的损失配置已经能够较好平衡空洞补全和有效区域恢复，过强的 anchor 约束会限制模型修正 NS 初值的能力。

`gated residual` 版本将预测形式从 `pred = anchor + residual` 改为 `pred = anchor + gate * residual`。该版本取得最低的 Hole MAE（`0.108534`），但 Global MAE 和 Valid MAE 分别退化到 `0.058415` 和 `0.049098`。因此它适合作为 hole-focused 架构消融，而不是替代当前主模型。若论文主目标是整体 depth restoration，应继续使用 Ours main；若应用更关注空洞区域极限精度，可以将 gated residual 作为可选变体讨论。

## 定性样本选择

按 `model_hole_mae - anchor_hole_mae` 排序的最佳和最差样本已保存到：

```text
output/depth_restoration_summary_final/ranked_cases_vs_anchor.md
output/depth_restoration_summary_final/ranked_cases_vs_anchor.csv
```

主模型在 hole 区域相对 NS anchor 的结果是：

```text
better/worse = 99/1
worst delta  = 0.001890
```

唯一 regression 样本为：

```text
contemporary-bathroom/0/131
anchor_hole_mae = 0.003178
model_hole_mae  = 0.005068
delta           = 0.001890
```

这个 regression 的绝对误差非常小，因此从数值上看并不是严重 failure。

如需生成 ranked 可视化，可运行：

```bash
python -u eval_depth_restoration.py \
  --checkpoint output/depth_restoration_unet_noisy_ns_n1000/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_restoration_unet_noisy_ns_n1000/eval_seed123_ranked \
  --batch_size 8 \
  --num_workers 4 \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

## 方法创新性表述

可以将论文主线写成：

1. 将 Kinect-style missing depth recovery 从 IQ/image inpainting 重新表述为 depth-domain mask-aware restoration。
2. 提出 single learned restoration model，直接从 degraded depth 恢复 clean dense depth，不再依赖 `DepthCAD + learned completion` 的两阶段学习流程。
3. 使用 hole/valid 区域感知的监督，让模型同时关注空洞区域补全和非空洞有效深度恢复。
4. 使用 deterministic NS anchor 作为几何初值，降低学习难度，但最终修正由单个神经网络完成。
5. 在独立 holdout set 上同时超过 Noisy、NS Anchor、DepthCAD/Plane Base 和 Two-stage Completion。

## 一句话总结

```text
最终方法应定位为 single mask-aware depth restoration：它使用一个 learned model，从 noisy depth、hole mask、confidence 和 NS depth anchor 直接恢复 clean dense depth，并在 seed123 holdout 上同时取得最低的 global、hole 和 valid MAE。
```
