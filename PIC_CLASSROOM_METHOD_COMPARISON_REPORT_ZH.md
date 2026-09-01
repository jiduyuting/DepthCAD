# Pic Classroom 新拍数据方法对比报告

日期：2026-07-07

数据目录：

```text
pic/iq_classroom/
pic/depth_classroom/
```

本轮目标：从新拍 classroom 序列中选几张代表样本，使用上一轮报告中的全部已跑通对比方法做无 GT 定性比较。

## 1. 数据检查

原始数据共 30 组：

```text
depth_classroom/depth_0000.npy ... depth_0029.npy
iq_classroom/iq_0000.npy ... iq_0029.npy
```

检查结果：

| 项 | 结果 |
|---|---|
| depth shape | `(424, 512)` |
| IQ shape | `(424, 512, 9)` |
| depth unit | mm |
| prepared depth unit | m |
| valid depth range | about `0.5m - 4.5m` |
| valid ratio | about `23% - 25%` |
| 65535 saturation | not observed in selected samples |

预览和统计输出：

```text
output/pic_classroom_selection/preview_all_depth_and_valid.png
output/pic_classroom_selection/depth_stats.csv
```

## 2. 选择样本

本轮选择 4 张：

```text
0000
0014
0028
0029
```

选择理由：

| Sample | Reason |
|---|---|
| `0000` | 序列起始样本，结构完整，适合作为标准观察样本 |
| `0014` | 中段稳定样本，和上一批报告中的常用对比样本编号一致，便于横向观察 |
| `0028` | 有效区域比例开始上升，近处结构和洞分布有变化 |
| `0029` | 有效区域比例最高，valid ratio 约 `25.3%` |

选中样本整理到：

```text
output/pic_classroom_selected_input/
```

prepared 输出：

```text
output/pic_classroom_prepared_selected/
```

## 3. 重要阈值设置

这批 classroom 深度里，很多有效深度在 `0.5m - 1.0m`。因此不能使用某些真实推理脚本默认的：

```text
hole_depth_threshold = 1.0
valid_min_depth = 1.0
```

否则会把近处有效物体误判为 hole。

本轮统一使用：

```text
hole_depth_threshold = 0.0
valid_min_depth = 0.5
valid_max_depth = 4.5
```

## 4. 已跑方法

本轮使用上一批 `new_capture` 对比中的全部已跑通方法：

| Method | Output dir | Note |
|---|---|---|
| NS anchor | `output/pic_classroom_raw9_flow_satclip/anchor/` | OpenCV/NS deterministic anchor |
| depth-only flow | `output/pic_classroom_depth_only_flow/` | depth-only endpoint flow |
| raw9 satclip | `output/pic_classroom_raw9_flow_satclip/` | overexposure/saturation fine-tuned raw9 flow |
| raw9 realholes | `output/pic_classroom_raw9_flow_realholes/` | real-hole fine-tuned raw9 flow |
| after-synth split | `output/pic_classroom_raw9_flow_after_synth_realhole/` | synthetic real-hole + split-added mask |
| propagation split | `output/pic_classroom_raw9_propagation_refine/` | propagation/refinement model + split-added mask |
| ProPainter | `output/pic_classroom_external_inpaint/propainter_run/` | external image/video inpainting baseline |
| DepthCAD depth-gray | `output/pic_classroom_depthcad_depth_gray_s5/` | diagnostic depth-as-gray DepthCAD baseline |

说明：RAD 在上一批报告中没有得到有效结果，原因是 checkpoint/显存问题，因此不计入这次“已跑通全方法”集合。

## 5. 统一对比图

对比图输出：

```text
output/pic_classroom_method_comparison/figures/0000.png
output/pic_classroom_method_comparison/figures/0014.png
output/pic_classroom_method_comparison/figures/0028.png
output/pic_classroom_method_comparison/figures/0029.png
```

汇总文件：

```text
output/pic_classroom_method_comparison/summary.json
output/pic_classroom_method_comparison/per_sample_metrics.csv
output/pic_classroom_method_comparison/figures.txt
```

## 6. 无 GT 指标汇总

注意：这批真实拍摄数据没有 GT，下面不是 accuracy，只用于观察：

1. 是否把 threshold hole 填成有效深度；
2. 是否大幅改动原始 valid 区域；
3. 相对 NS anchor 在 hole 内改动有多大；
4. 填补出来的深度尺度是否异常。

| Method | Threshold fill | Cleaned fill | Valid change | vs NS in threshold hole | Filled threshold median |
|---|---:|---:|---:|---:|---:|
| NS anchor | 100.00% | 100.00% | 0.0000 m | 0.0000 m | 3.182 m |
| depth-only flow | 100.00% | 100.00% | 0.0000 m | 0.2279 m | 3.167 m |
| raw9 satclip | 100.00% | 100.00% | 0.0000 m | 0.0316 m | 3.205 m |
| raw9 realholes | 100.00% | 100.00% | 0.0000 m | 0.0206 m | 3.182 m |
| after-synth split | 100.00% | 100.00% | 0.0138 m | 0.2516 m | 3.297 m |
| propagation split | 100.00% | 99.98% | 0.0136 m | 0.4125 m | 4.008 m |
| ProPainter | 100.00% | 100.00% | 0.0000 m | 0.4577 m | 2.726 m |
| DepthCAD depth-gray | 100.00% | 100.00% | 0.0000 m | 1.4248 m | 1.955 m |

## 7. 定性观察

### 最稳的方法

`raw9 satclip` 和 `raw9 realholes` 是本轮最稳的两个结果。

它们的共同特点：

1. hole 都能填满；
2. 输出和 NS anchor 保持接近，没有明显大面积 hallucination；
3. 相对 NS anchor 的 hole 内改动小，分别约 `0.0316m` 和 `0.0206m`；
4. valid 区域在 hole-only 输出中保持原始深度不变。

这批 classroom 样本上，`raw9 realholes` 更保守，几乎贴近 NS anchor；`raw9 satclip` 稍微更愿意调整 hole 内结构。

### depth-only flow

`depth-only flow` 可以填洞，但可视化里有明显的大面积结构幻觉和条带/波纹伪影。它在 hole 内相对 NS anchor 的平均改动约 `0.228m`，远大于 raw9 satclip/realholes。

结论：继续作为对照，不建议作为 classroom 主输出。

### after-synth split

`after-synth split` 会把 cleaned mask 额外扩大约 `2.9%`，对原始 valid 区域有约 `1.38cm` 的平均改动。视觉上比 depth-only 稳，但会更强地平滑和外推。

结论：可作为激进 mask 版本观察，但本轮不如 raw9 satclip/realholes 稳。

### propagation split

`propagation split` 在本轮明显过激。它的 threshold hole 内相对 NS anchor 平均改动约 `0.4125m`，填补中位深度约 `4.01m`，很多区域被推到更远深度。

结论：这批 classroom 数据上不建议用 propagation split 作为推荐输出。

### ProPainter

ProPainter 成功跑通并 decode。和上一批一样，最后写 mp4 时出现 `imageio` 的 `fps` 参数报错，但 PNG 帧已经完整保存，脚本已成功 decode 回 `.npy`。

视觉上 ProPainter 纹理噪声和块状/条纹伪影比较明显，hole 内相对 NS anchor 改动约 `0.458m`。

结论：可作为外部图像域 baseline，但不如 raw9 系列保结构。

### DepthCAD depth-gray

DepthCAD depth-gray 是诊断性 baseline，不是原始 IQ-domain DepthCAD。由于当前环境没有 CUDA，本轮用 `--allow_cpu` 跑了 4 张、5 step。

它能填满洞，但深度尺度明显偏近，threshold hole 中位深度约 `1.955m`，相对 NS anchor 改动约 `1.425m`。可视化上看也更像图像先验平滑，不适合作为这批数据的推荐结果。

## 8. 当前建议

本轮 classroom 新拍数据建议优先看：

```text
output/pic_classroom_raw9_flow_satclip/visualizations/
output/pic_classroom_raw9_flow_realholes/visualizations/
output/pic_classroom_method_comparison/figures/
```

推荐排序：

1. `raw9 realholes`：最保守，最接近 NS anchor，少出伪结构；
2. `raw9 satclip`：也稳定，hole 内比 realholes 略更主动；
3. `NS anchor`：可靠保守 baseline；
4. `after-synth split`：可作为激进 mask 对照；
5. `depth-only flow`、`propagation split`、`ProPainter`、`DepthCAD depth-gray`：只建议作为 baseline/failure 对照。

如果后续要扩大测试，建议先跑全部 30 张的 `raw9 satclip` 和 `raw9 realholes`，再挑出结构变化大的样本补跑外部 baseline。因为本轮 30 张基本是同一视角序列，全量跑所有外部方法的收益不高，尤其 DepthCAD CPU 很慢。
