# IQ / Depth / RGB 空洞成因与模型恢复分析

日期：2026-07-08

分析对象：

```text
picture/
output/picture_random_suite_n3_seed20260708_core/
```

本报告基于上一轮每组随机 3 张样本的结果，结合：

```text
depth
IQ/raw9
RGB
hole mask
IQ amplitude
depth edge / valid boundary
模型补全输出
```

来判断空洞大概由什么原因产生，以及当前模型恢复得如何。

## 1. 诊断输出

诊断图目录：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/
```

总览图：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/overview_hole_diagnostics.png
```

诊断指标：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/metrics.csv
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/summary.json
```

每张诊断图包含：

```text
RGB
raw depth
hole mask
IQ amplitude
low amplitude mask
boundary hole
near depth edge
NS anchor
raw9 realholes
raw9 satclip
depth-only
after-synth
propagation
ProPainter
```

## 2. 诊断方法

空洞定义：

```text
hole = non-finite depth or depth <= 0
```

IQ amplitude 计算：

```text
amp = mean_i sqrt(I_i^2 + Q_i^2)
```

其中 I/Q 使用 raw9 的前三组 ToF pair：

```text
(0, 1), (2, 3), (4, 5)
```

诊断特征：

| Feature | Meaning |
|---|---|
| `hole_ratio` | depth 中无效区域比例 |
| `hole_low_amp_fraction` | hole 中有多少像素落在低 IQ amplitude 区域 |
| `hole_boundary_fraction` | hole 中有多少靠近有效 depth 边界 |
| `hole_near_depth_edge_fraction` | hole 中有多少靠近深度梯度/几何边界 |
| `sat_hole_fraction` | hole 中有多少 IQ 接近 65535 饱和 |
| `rgb_dark_hole_fraction` / `rgb_bright_hole_fraction` | hole 是否集中在 RGB 暗/亮区域 |

注意：没有真实 GT，所以恢复效果不能用真实误差判断。这里的模型评价主要是：

1. 是否填洞；
2. 是否相对 NS anchor 改动过大；
3. 是否产生明显条纹/块状/过度外推；
4. 是否保留几何结构。

## 3. 空洞成因总览

| Group | Hole ratio | Low amp in hole | Boundary hole | Near edge | Saturation in hole | 主因判断 |
|---|---:|---:|---:|---:|---:|---|
| `pic_class2` | `0.630` | `0.145` | `0.106` | `0.113` | `0.000` | 大面积弱观测/无观测，不是单纯边界洞 |
| `pic2_l1` | `0.703` | `0.310` | `0.049` | `0.068` | `0.068` | 低信号 + 少量饱和/异常 |
| `pic2_l2` | `0.667` | `0.648` | `0.071` | `0.095` | `0.008` | 低 IQ 幅值主导 |
| `pic2_l3` | `0.196` | `0.469` | `0.376` | `0.465` | `0.000` | 几何边界/遮挡边缘 + 局部低信号 |
| `pic3_s1` | `0.795` | `0.083` | `0.090` | `0.082` | `0.001` | 大面积无观测/远距离/视场弱覆盖 |
| `pic3_s2` | `0.855` | `0.161` | `0.053` | `0.052` | `0.036` | 极大面积无观测，少量异常/饱和 |
| `pic3_s3` | `0.837` | `0.163` | `0.056` | `0.057` | `0.003` | 极大面积无观测 |
| `pic4_z1` | `0.605` | `0.418` | `0.113` | `0.133` | `0.000` | 低 IQ 幅值 + 大洞 |
| `pic4_z2` | `0.776` | `0.680` | `0.063` | `0.075` | `0.000` | 低 IQ 幅值主导，远/弱反射明显 |
| `pic4_z3` | `0.712` | `0.514` | `0.130` | `0.110` | `0.000` | 低 IQ 幅值 + 部分边界洞 |

## 4. 空洞主要来源

### 4.1 低 IQ amplitude / 弱 ToF 回波

最明显的组：

```text
pic2_l2
pic4_z1
pic4_z2
pic4_z3
```

这些组里，hole 和低 IQ amplitude 的重合比例很高：

```text
pic2_l2: 64.8%
pic4_z2: 68.0%
pic4_z3: 51.4%
pic4_z1: 41.8%
```

这说明很多空洞不是模型/脚本造成的，而是传感器原始信号已经弱：ToF 回波幅值低，depth estimator 无法稳定解算，于是输出 `depth=0`。

常见物理原因：

1. 距离较远，返回信号衰减；
2. 表面反射率低；
3. 表面角度太斜，主动光反射回相机少；
4. 透明/反光/半反光材料导致有效回波不足；
5. 场景中局部区域 IR 信号很弱。

这类空洞最适合用 raw9/amplitude-aware 方法，因为 raw9 里能看到信号强弱。

### 4.2 几何边界 / 遮挡边缘 / mixed pixel

最明显的组：

```text
pic2_l3
```

`pic2_l3` 的 hole ratio 只有约 `19.6%`，但：

```text
boundary hole: 37.6%
near depth edge: 46.5%
```

这说明它的空洞很多出现在物体边界、遮挡边缘、深度不连续附近。这类区域容易出现 ToF mixed pixel：

```text
一个像素收到前景和背景混合回波
相位不稳定
confidence 降低
depth estimator 输出无效或异常
```

这类数据非常适合做论文主图，因为：

1. 有效 depth 比例高；
2. hole 不是全图大面积缺失；
3. 边界结构清楚；
4. 可以比较方法是否保边界。

### 4.3 大面积无观测 / stress case

最明显的组：

```text
pic3_s1
pic3_s2
pic3_s3
```

这些组的 hole ratio 很高：

```text
pic3_s1: 79.5%
pic3_s2: 85.5%
pic3_s3: 83.7%
```

但 low amplitude、boundary、edge 的解释比例都不高。这说明很多 hole 是大范围无有效观测，而不是局部边界失败。

可能原因：

1. 场景大面积超出可靠测距范围；
2. 视角下 IR 投射/接收覆盖不足；
3. 远距离背景或大平面回波太弱；
4. depth estimator 对这类区域直接置零；
5. 相机姿态/曝光/环境导致有效区域很少。

这类数据不适合作主结果，只适合作为 hard/stress set。任何方法在这里都主要是在外推，不应该把视觉平滑当成真实精度。

### 4.4 饱和/过曝不是主因

大多数组里：

```text
sat_hole_fraction ~= 0
```

只有少数组有一点：

```text
pic2_l1: 6.8%
pic3_s2: 3.6%
```

所以这批数据的主因不是 65535 饱和。`raw9 satclip` 仍然能跑，但这批不是真正的 saturation/overexposure benchmark。

## 5. RGB 给出的解释

RGB 主要用于辅助看场景内容，不直接参与当前模型。

从诊断图看：

1. 很多 hole 对应墙面、远处平面、玻璃/反光/亮暗变化区域；
2. 边界洞通常和 RGB 中物体轮廓或遮挡边缘一致；
3. 一些大面积 hole 在 RGB 里是可见的背景，但 depth 完全无效，这说明不是“场景不存在”，而是 ToF 传感器没有可靠回波；
4. RGB 能解释语义和边界，但当前 raw9/depth 模型不使用 RGB，因此恢复主要依赖 depth anchor 和 IQ/amplitude 信号。

如果后续想做更强 baseline，可以考虑 RGB-guided depth completion。但要注意，这会变成另一个 setting：

```text
RGB + raw/depth -> dense depth
```

而当前主线是：

```text
RGB-free ToF depth restoration
```

论文里需要区分清楚。

## 6. 模型恢复效果

### 6.1 raw9 realholes / raw9 satclip

这两个方法整体最稳。

在大多数组里，它们相对 NS anchor 的 hole 内改动很小：

```text
raw9 realholes: mostly 0.018m - 0.027m
raw9 satclip:   mostly 0.029m - 0.051m
```

说明它们不会像 depth-only 或图像域方法那样大幅 hallucinate。

例外：

```text
pic2_l3
```

这组有效 depth 多、边界洞多，raw9 realholes/satclip 的改动达到约 `0.16m - 0.18m`。从诊断图看，这不是纯坏事，因为 `pic2_l3` 给模型的有效几何和边界信息更多，模型有更强依据修正 anchor。

推荐用途：

```text
主方法候选
论文主图优先展示
```

### 6.2 depth-only flow

depth-only 能补洞，但没有 IQ/amplitude 约束，很多时候更容易生成条带、块状平滑和大范围 hallucination。

它在部分组里相对 NS 改动明显更大：

```text
pic_class2: 0.282m
pic2_l3:   0.391m
pic4_z3:   0.171m
```

但也有例外：

```text
pic4_z2: 0.039m
```

说明 depth-only 不是完全不可用，但稳定性不如 raw9 方法。它更适合作为 ablation baseline。

### 6.3 after-synth / propagation

这两个方法普遍更激进。

很多组的相对 NS 改动达到：

```text
after-synth: 0.18m - 0.78m
propagation: 0.31m - 0.89m
```

诊断图里能看到它们经常把 hole 区域推成大平面或更远深度。对于大面积无观测场景，这种外推不可靠。

推荐用途：

```text
aggressive mask/refinement 对照
failure case 展示
不建议作为默认输出
```

### 6.4 ProPainter

ProPainter 是图像/视频 inpainting baseline，不理解 ToF 物理。

它的典型问题：

1. 网格/纹理伪影；
2. 块状平滑；
3. 深度尺度不稳定；
4. 对边界和大洞会生成视觉上平滑但 metric depth 不一定可信的结构。

推荐用途：

```text
外部 image-domain baseline
说明图像 inpainting 和 sensor depth restoration 的区别
```

## 7. 推荐看哪些诊断图

主图候选：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic2_l3_0006.png
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic4_z1_0004.png
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic_class2_0001.png
```

低 amplitude 典型：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic2_l2_0000.png
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic4_z2_0002.png
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic4_z3_0012.png
```

hard/stress 典型：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic3_s2_0013.png
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic3_s3_0006.png
```

外部 baseline failure 典型：

```text
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic2_l1_0001.png
output/picture_random_suite_n3_seed20260708_core/hole_diagnostics/figures/pic4_z2_0002.png
```

## 8. 总结

这批空洞主要不是单一原因，而是混合了：

```text
低 IQ amplitude / 弱回波
边界 mixed pixel
大面积无观测
少量饱和/异常
远距离或视角导致的 ToF 信号不足
```

其中：

1. `pic2_l3` 是最适合做主结果的数据：有效比例高，洞多在边界，能体现保结构能力；
2. `pic2_l2`、`pic4_z1/z2/z3` 体现低 amplitude 导致的真实 ToF 失效；
3. `pic3_s1/s2/s3` 是 hard/stress set，不适合宣传主效果；
4. 当前模型里 raw9 realholes / raw9 satclip 最稳；
5. depth-only、after-synth、propagation、ProPainter 都可以作为对照，但不适合作为默认主输出。

论文里可以这样表述：

```text
The observed holes are not random missing pixels. They correlate with low ToF amplitude, depth discontinuities, and large unobserved regions, indicating sensor-originated failures. Raw9-conditioned flow is more stable than depth-only or image-domain inpainting because it uses sensor reliability cues while preserving a deterministic geometric anchor.
```
