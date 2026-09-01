# ToF 深度补全实验设计

目标不是继续堆更多 baseline，而是把问题拆成两个可验证问题：

1. 当前方法在“已知 mask 的深度补全”上是否真的优于传统方法和通用视频补全方法。
2. 对真实 ToF 退化，过曝、远景低信噪比、飞点、条纹和无效洞能否用同一个系统处理。

## 结论先行

当前模型应先按 depth completion 评估，不要先声称解决了完整 ToF raw/IQ 成像问题。现有 `ours_hole_only` 和 `ours_restored` 的标准输入是已经得到的深度图加 mask/条件，不是直接在 IQ/raw9 上做物理解码。

过曝和远景可以放进一个统一系统里处理，但建议系统拆成两层：

- 物理/传感器层：raw9/IQ 解码出 depth、amplitude/confidence、saturation mask、invalid mask。
- 学习补全层：输入 corrupted depth + mask/confidence/saturation 等辅助图，输出 restored depth。

如果 raw/IQ 完全饱和或远景信号完全丢失，真实几何信息已经不可逆，模型只能根据上下文补全/先验猜测，不能恢复真实细节。实验表述应叫 depth restoration/completion，而不是完整物理恢复。

## 实验 A：远景深度补全定量验证

数据：

```text
output/far_pic_benchmark/synthetic_05
output/far_pic_benchmark/synthetic_10
output/far_pic_benchmark/synthetic_20
```

这些 case 是从真实远景 depth 上人为遮挡，mask 内有 ground truth，所以可以做定量结论。

方法：

```text
opencv_ns
opencv_telea
propainter
ours_hole_only
ours_restored
DepthCAD paper/original model
RAD-LSUN，只作为弱 baseline
```

指标：

```text
eval_mask_mae_weighted     # 核心指标，mask 内 MAE，单位 m
eval_mask_rmse_mean        # 大误差敏感
eval_mask_p95_mean         # 局部严重失败
boundary_jump_m_mean_mean  # mask 边缘是否接上
hole_total_variation_m_mean_mean # 洞内是否过于抖动，但不能单独代表好坏
outside_mean_abs_change_mean     # mask 外是否被误改
zero_ratio_in_repair_mask_mean   # 是否还有没补上的无效点
```

成功标准：

- 若 ours 在 `synthetic_05/10/20` 的 `eval_mask_mae_weighted` 都优于 OpenCV NS 和 ProPainter，才可以说当前补全模型有明确优势。
- 若 ours 只在大 mask 上优于 NS，但小 mask 不优于 NS，应表述为“对大洞/结构性缺失更有潜力”，不能泛称更好。
- 若 ProPainter MAE 更差但视觉更顺，说明它更像平滑先验，不一定几何更准。
- 若 ProPainter MAE 更好，则当前 depth-flow 方法确实需要重新训练或改输入。

建议运行顺序：

```bash
python3 scripts/run_external_inpainting_far_pic.py run-propainter \
  --case output/far_pic_benchmark/synthetic_05 \
  --output_dir output/far_pic_benchmark/synthetic_05/propainter_run \
  --decode \
  --decode_name propainter_restored

python3 scripts/run_external_inpainting_far_pic.py standardize-method \
  --case output/far_pic_benchmark/synthetic_05 \
  --method propainter \
  --source_dir output/far_pic_benchmark/synthetic_05/propainter_run/restored_by_stem \
  --source_suffix propainter_restored

python3 scripts/visualize_far_pic_benchmark_compare.py \
  --case_dir output/far_pic_benchmark/synthetic_05 \
  --output_dir output/far_pic_benchmark/synthetic_05/visualizations_compare
```

对 `synthetic_10` 和 `synthetic_20` 重复同样命令。最终看每个 case 的：

```text
output/far_pic_benchmark/<case>/visualizations_compare/contact_sheet.png
output/far_pic_benchmark/<case>/visualizations_compare/summary.json
```

## 实验 B：真实远景坏点补全

数据：

```text
output/far_pic_benchmark/bad_depth_mask_v1
output/far_pic_benchmark/zero_mask
```

这两类是真实/半真实坏点 mask，没有严格 GT，不能用 mask 内 MAE 判断胜负。

可报告：

```text
outside_mean_abs_change
boundary_jump_m
hole_total_variation_m
zero_ratio_in_repair_mask
contact_sheet + per_frame visualizations
```

解释边界：

- `boundary_jump` 低通常代表边缘更连续。
- `hole_total_variation` 低通常代表更平滑，但可能是过平滑。
- 真正论文/报告里应把这些称为 no-reference diagnostics，不要称为 accuracy。

当前 `bad_depth_mask_v1` 已统一到：

```text
output/far_pic_benchmark/bad_depth_mask_v1/outputs/<method>/*.npy
output/far_pic_benchmark/bad_depth_mask_v1/visualizations_compare/
```

## 实验 C：过曝 ToF 的单独验证

目的：判断当前方法能否处理 overexposure 引起的无效深度、错误深度和饱和边缘。

数据分两类：

1. 真实过曝 raw9：

```text
/data/pre_student/GJ/DepthCAD/raw
```

其中数值型 `*.npy` 是 `(9, 240, 320)` raw9/IQ，已有饱和点。真实过曝没有 GT，只能做视觉和无参考指标。

2. 合成过曝：

从非饱和或轻微饱和 raw9/depth 出发，人工制造 saturation/corruption，再以原始解码 depth 作为 pseudo-GT。这样可以计算 mask 内误差。

建议 case：

```text
output/tof_overexposure_benchmark/synthetic_clip_05
output/tof_overexposure_benchmark/synthetic_clip_10
output/tof_overexposure_benchmark/synthetic_clip_20
output/tof_overexposure_benchmark/real_overexposed
```

合成策略：

- 在 raw9/IQ 上做局部通道 clipping，模拟强反射/过曝。
- 重新解码出 corrupted depth。
- saturation mask 作为 repair mask。
- 原始未 clipping 解码 depth 作为 GT。

如果暂时没有稳定 raw9 解码器，可以先做 depth-level proxy：

- 选 raw depth 的高反射/边缘区域或随机块。
- 将这些区域置 0、置远值、或加入局部平顶错误。
- 评估补全模型是否能恢复原 depth。

但 depth-level proxy 只能证明补洞能力，不能证明 raw/IQ 过曝物理问题被解决。

## 实验 D：远景 + 过曝联合退化

真实 ToF 里远景和过曝经常同时出现，所以最终应有联合 case。

建议先做两个层级：

### D1：Depth-level 联合 stress test

数据：

```text
far_pic/noise_depth_240x320_m
```

构造：

```text
combined_mask = synthetic_mask OR bad_depth_mask OR saturation_like_mask
corrupted_depth = raw_depth
corrupted_depth[combined_mask] = 0 或局部错误深度
```

保留每类 mask：

```text
hole_mask
far_noise_mask
saturation_like_mask
combined_mask
```

输出：

```text
output/tof_combined_benchmark/synthetic_05_bad_sat
output/tof_combined_benchmark/synthetic_10_bad_sat
output/tof_combined_benchmark/synthetic_20_bad_sat
```

评估时分别报告：

- combined mask 总 MAE
- hole mask MAE
- saturation-like mask MAE
- boundary band MAE
- outside change

### D2：Raw/IQ-level 联合 stress test

从 raw9 出发：

- 人工降低远景信号或加入低 SNR 噪声。
- 人工 clipping 局部高幅值区域。
- 解码出 corrupted depth/confidence/saturation mask。
- 用原始 raw9 解码 depth 作为 pseudo-GT。

这个实验更接近真实 ToF，但依赖 raw9 解码流程足够可信。

## 模型设计建议

不要把过曝和远景当成两个完全独立任务。它们都可以看成 ToF measurement degradation，但输入信息不同。

建议模型输入从 depth-only 升级为：

```text
corrupted_depth
valid_mask / repair_mask
confidence 或 amplitude
saturation_mask
bad_depth_mask
hole_distance_map
可选 raw9/IQ normalized channels
```

训练损失：

```text
L_mask      = mask 内 L1/Charbonnier
L_boundary  = mask 边界带加权 L1
L_outside   = mask 外小权重保持项，防止误改有效区域
L_smooth    = 只在低置信区域使用的弱平滑，不要全图过平滑
```

推荐先训练两版做 ablation：

```text
depth_only:
  input = corrupted_depth + repair_mask + distance

depth_conf_sat:
  input = corrupted_depth + repair_mask + confidence + saturation_mask + bad_depth_mask + distance
```

如果 `depth_conf_sat` 明显优于 `depth_only`，说明 ToF 退化原因标签是有价值的。之后再考虑 raw9/IQ 多通道输入。

## 报告结构建议

最终结果不要只展示一张视觉图。建议报告按下面结构组织：

1. Synthetic far-pic completion table：`synthetic_05/10/20`，有 GT，主结论。
2. Real far-pic diagnostics：`bad_depth_mask_v1/zero_mask`，视觉和无参考指标。
3. Overexposure-only：真实 visual + 合成 GT 定量。
4. Combined far + overexposure：证明统一 pipeline 对复合退化也能工作。
5. Ablation：depth-only vs depth+confidence/saturation vs raw9/IQ。

只有当第 1 和第 4 组实验都能打过 OpenCV NS/ProPainter，才适合说当前方法在 ToF 深度补全上有优势。
