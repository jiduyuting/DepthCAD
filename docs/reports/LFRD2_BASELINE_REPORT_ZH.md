# LFRD2 Raw9 Proxy Baseline Report

Date: 2026-08-16

## 结论

`/data/pre_student/GJ/LFRD2` 的方法和当前 ToF depth restoration 方向相关，但它不能直接作为当前 `DepthCAD/raw + depth` 数据的严格同口径 baseline。

原因：

- LFRD2 real checkpoint 面向 under-display ToF raw，输入是 `8 x 180 x 240`。
- 当前 DepthCAD 真实数据是 `9 x 240 x 320` raw9 + `240 x 320` paired depth。
- LFRD2 官方代码多处写死 `.cuda()`，并依赖 `mmcv.ops.modulated_deform_conv`；当前机器没有可用 NVIDIA driver。

因此本轮做的是 **LFRD2 FracDiff proxy baseline**：

```text
DepthCAD raw9/depth
-> raw9 first 8 channels + center crop to 180x240
-> artificial block mask on paired real depth
-> NS anchor or masked depth as LFRD2 depth initialization
-> LFRD2 FracDiff real checkpoint
-> same mask MAE / global MAE / visualizations
```

这个结果可以作为 cross-sensor zero-shot 参考，不应表述为 LFRD2 官方 pipeline 的最终性能。

## 脚本

新增脚本：

```text
scripts/run_lfrd2_raw9_masked_self_test.py
```

关键实现：

- 不修改 LFRD2 源码。
- 加载 `/data/pre_student/GJ/LFRD2/checkpoint/real/net_parameter_x.pth`。
- 用 `grid_sample` 替代 CUDA-only 的 subpixel sampling / confidence sampling 路径，使 CPU 能跑。
- 输出 `.npy`、`summary.json` 和可视化。

推荐复现实验命令：

```bash
/home/lab507/anaconda3/envs/SVDC/bin/python scripts/run_lfrd2_raw9_masked_self_test.py \
  --raw_dir raw \
  --depth_dir depth \
  --output_dir output/lfrd2_raw9_masked_self_test_anchor_fliplr_vis \
  --input_mode anchor \
  --raw9_transform flip_lr \
  --device cpu \
  --torch_threads 1 \
  --visualize \
  --vis_max_samples 16
```

## 完整结果

数据：

```text
raw/:   numeric raw9/depth paired samples 2-42, total 41
depth/: paired depth maps, unit m
crop:   y=30, x=40, h=180, w=240
mask:   block mask, target ratio 10%, seed 123
```

单位：m。越低越好。

| Variant | Raw9 transform | LFRD2 depth input | Anchor mask MAE | LFRD2 mask MAE | LFRD2 / Anchor | Better cases |
|---|---|---|---:|---:|---:|---:|
| masked_none | none | corrupted masked depth | 0.119357 | 1.805330 | 15.13x | 0 / 41 |
| anchor_none | none | NS anchor | 0.119357 | 0.415971 | 3.49x | 0 / 41 |
| masked_fliplr | flip_lr | corrupted masked depth | 0.119357 | 1.809850 | 15.16x | 0 / 41 |
| anchor_fliplr | flip_lr | NS anchor | 0.119357 | 0.381199 | 3.19x | 0 / 41 |

Best proxy setting:

```text
output/lfrd2_raw9_masked_self_test_anchor_fliplr_vis/summary.json
```

Aggregate:

```text
anchor_mask_mae_mean:          0.119357
lfrd2_mask_mae_mean:           0.381199
lfrd2_hole_only_global_mae:    0.044520
lfrd2_full_global_mae:         0.386430
lfrd2_unmasked_mae:            0.388443
better_than_anchor_cases:      0 / 41
mask_improvement_vs_anchor:   -219.38%
```

可视化：

```text
output/lfrd2_raw9_masked_self_test_anchor_fliplr_vis/visualizations/
```

代表文件：

```text
output/lfrd2_raw9_masked_self_test_anchor_fliplr_vis/visualizations/2.png
output/lfrd2_raw9_masked_self_test_anchor_fliplr_vis/visualizations/10.png
output/lfrd2_raw9_masked_self_test_anchor_fliplr_vis/visualizations/17.png
```

## 与当前方法的关系

当前项目已有真实 raw9 masked self-test 结果：

```text
output/real_raw9_masked_self_test_ratio10_thr1m_iq6_finetuned_e30_best/summary.json
```

该结果在全尺寸 `240 x 320`、10% block mask 上：

| Method | Anchor mask MAE | Model mask MAE | Improve vs anchor |
|---|---:|---:|---:|
| NS anchor | 0.090745 | - | - |
| Current raw9 fine-tuned flow | 0.090745 | 0.058679 | +35.34% |

注意：这张表和 LFRD2 proxy 的 crop 口径不同，不能逐数值直接比较；但趋势很清楚：当前方法在本项目数据上已经能优于 NS anchor，而 LFRD2 zero-shot proxy 在 41/41 个 case 中都没有优于 anchor。

## 建议写法

可以把 LFRD2 放到 related work / candidate baseline：

```text
LFRD2 is highly relevant as a learned fractional reaction-diffusion framework for ToF restoration, but its released checkpoints target under-display ToF raw measurements with a different sensor format. We therefore include a cross-sensor proxy experiment by applying its FracDiff refinement module to our real raw9 masked self-test. Zero-shot transfer is unsuccessful, likely due to channel, resolution, and degradation-domain mismatch, so we do not use it as a primary quantitative baseline without retraining.
```

如果后续要做更严格的 baseline，需要：

1. 将 DepthCAD raw9 重新整理成 LFRD2 的训练格式，或改 LFRD2 输入层支持 `9 x 240 x 320`。
2. 用当前真实 raw9/depth masked self-supervision fine-tune LFRD2。
3. 再用同一个 `scripts/real_raw9_masked_self_test.py` 口径报告结果。
