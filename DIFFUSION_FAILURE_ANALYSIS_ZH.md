# Diffusion / pseudo-RGB IQ Inpainting 失败分析

Date: 2026-05-21

本文档解释为什么当前的 Stable Diffusion / pseudo-RGB IQ inpainting 在 Kinect-style depth hole completion 上效果不好，以及如果后续还想发挥 diffusion 的能力，应该怎么改路线。

## 1. 当前 diffusion 路线做了什么

当前 SD inpainting 路线不是 RGB-guided depth completion，而是：

```text
6-channel IQ
-> 每个 IQ 通道单独取出
-> 单通道复制 3 份，变成 grayscale pseudo-RGB
-> StableDiffusionInpaint 填 hole
-> 取输出 RGB 的第 1 通道作为该 IQ 通道
-> 6 个通道分别得到填补结果
-> IQ-to-depth / DepthEstimator 转成 depth
```

对应代码：

```text
inference_sd_inpaint.py
apply_kinect_holes_and_eval.py
```

这个流程的关键问题是：它把 ToF/IQ 的物理测量问题，强行改造成了 6 次独立的自然图像 inpainting。

## 2. 现有结果说明了什么

基于已有 n1000 plane_r12 结果重新统计：

```text
diagnose_diffusion_failures.py
--eval_dir kinect_evaluation/depth_cache_0515_n1000_plane_r12
```

输出目录：

```text
output/diffusion_diagnosis_n1000_plane_r12
```

### Aggregate Metrics

| Method | Global MAE | Hole MAE | Valid MAE |
|---|---:|---:|---:|
| noisy | 0.486355 | 2.964661 | 0.026555 |
| depthcad | 0.276089 | 1.556470 | 0.038539 |
| sdinpaint | 0.257932 | 1.516326 | 0.024462 |
| full | 0.313954 | 1.798432 | 0.038539 |
| depthfill | 0.112690 | 0.512363 | 0.038539 |

结论：

```text
SD Inpaint only 比 noisy 好，说明它不是完全没用；
但它远差于 depth-domain fill；
DepthCAD + SD Inpaint 甚至比 DepthCAD only 更差。
```

### Key Comparisons

| Method | Baseline | Region | Better/Worse | Mean Delta | Worst Delta |
|---|---|---|---:|---:|---:|
| sdinpaint | depthfill | holes | 20/451 | +1.002462 | +4.071497 |
| sdinpaint | noisy | holes | 457/14 | -1.446152 | +2.498891 |
| sdinpaint | depthcad | holes | 209/262 | -0.041456 | +4.085868 |
| full | depthcad | holes | 167/304 | +0.241081 | +6.667764 |
| full | depthfill | holes | 6/465 | +1.284998 | +6.916500 |

解释：

```text
1. SD inpaint 大多数时候比 noisy hole 好，说明生成式补全有一定能力。
2. 但 SD inpaint 几乎总是比 depthfill 差：451/471 个有效样本更差。
3. DepthCAD + SD inpaint 在 304/471 个样本上比 DepthCAD only 更差，说明 SD 填 IQ 后经常破坏 DepthCAD 已经修好的物理一致性。
```

## 3. 为什么 pseudo-RGB IQ diffusion 不行

### 原因 A：伪 RGB 不是 RGB

RGB 图像中：

```text
R/G/B 是同一场景的可见光外观；
边缘、纹理、物体语义和几何边界高度相关。
```

你的 pseudo-RGB 是：

```text
一个 IQ 通道复制三份：[I30, I30, I30]
或者 [Q40, Q40, Q40]
```

它没有颜色语义，也没有自然图像纹理。Stable Diffusion 的自然图像先验并不能理解：

```text
I30/Q30/I40/Q40/I58/Q58 之间的相位关系
```

所以它生成的是“看起来像灰度图的东西”，但不一定是物理上合法的 IQ。

### 原因 B：6 个 IQ 通道被独立 inpaint，破坏 ToF 物理约束

ToF depth 不是由单个通道决定，而是由多个 I/Q 对共同决定：

```text
(I30, Q30)
(I40, Q40)
(I58, Q58)
```

深度依赖：

```text
phase = atan2(Q, I)
amplitude = sqrt(I^2 + Q^2)
multi-frequency unwrapping / estimator
```

当前 SD 流程每个通道独立生成，因此没有约束：

```text
同一频率的 I/Q pair 是否相位一致；
不同频率之间是否满足同一个 depth；
amplitude 和 confidence 是否一致；
hole 边界处是否和有效区域连续。
```

这会导致：单个 IQ 通道看起来还可以，但转成 depth 后出现大误差。

### 原因 C：SD inpaint 的优化目标不是 metric depth

Stable Diffusion inpainting 的目标是自然图像分布上的视觉合理性，不是：

```text
Depth MAE
Hole MAE
plane consistency
depth boundary continuity
I/Q phase consistency
metric scale correctness
```

所以它可能生成视觉上平滑的灰度块，但经过 DepthEstimator 后深度完全不对。

### 原因 D：RGB-guided depth completion 的 RGB 是真实几何线索

其他 RGB-guided depth completion 做得好，是因为它们通常使用：

```text
真实 RGB image + sparse/invalid depth
```

真实 RGB 中有：

```text
物体边界
语义类别
纹理
透视结构
平面/墙/地面线索
```

现代 diffusion depth completion，例如 Marigold-DC，不是把 depth 伪装成 RGB 去 inpaint，而是利用 pretrained monocular depth diffusion prior，从 RGB 生成 dense depth，并用 sparse depth guidance 约束 metric scale。

这和当前 pseudo-RGB IQ inpainting 是两类问题。

## 4. 如果还想最大化 diffusion 的功能，应改成什么路线

### 路线 1：有真实 RGB 时，优先走 RGB-guided diffusion depth completion

如果 PBRT 能导出同相机、同视角、同分辨率 RGB，最值得做：

```text
input:
  RGB
  sparse/noisy depth
  hole mask
  confidence

output:
  dense metric depth
```

可以尝试：

```text
Marigold-DC / SteeredMarigold 类方法
```

核心是：

```text
让 diffusion 在 depth latent / depth map 上工作，
用 RGB 提供语义和边界先验，
用 sparse depth / valid depth 做 metric guidance。
```

而不是：

```text
把 IQ 通道伪装成 RGB，再让图像 inpainting 生成 IQ。
```

### 路线 2：没有真实 RGB 时，用 diffusion 做 depth prior，而不是 IQ inpainting

如果没有真实 RGB，不建议继续：

```text
6 个 IQ 通道 -> pseudo RGB -> SD inpaint
```

更合理的是：

```text
训练 depth-domain diffusion / latent diffusion
```

输入：

```text
noisy depth
hole mask
confidence
NS anchor
optional amplitude
```

输出：

```text
clean dense depth
```

也就是把现在的 residual U-Net restoration 换成更强 backbone / diffusion prior，但输出仍然是 depth。

### 路线 3：如果坚持做 IQ diffusion，必须联合 6 通道生成

如果一定要在 IQ 域做 diffusion，至少要满足：

```text
1. 6 个 IQ 通道一起作为多通道 latent，不要逐通道独立 inpaint。
2. loss 里加入 IQ L1 + phase loss + amplitude loss + depth loss。
3. 生成后必须通过 differentiable / fixed DepthEstimator 约束 depth MAE。
4. hole 边界处加 continuity loss。
5. 输入还要包括 confidence/hole mask/depth anchor。
```

也就是说，IQ diffusion 应该是：

```text
physics-aware multi-channel IQ diffusion
```

而不是：

```text
single-channel pseudo-RGB SD inpainting
```

## 5. 推荐下一步实验

### 实验 A：扩展诊断，保存 SD 中间结果

已在 `apply_kinect_holes_and_eval.py` 中加入：

```bash
--save_sd_diagnostics
--sd_diagnostics_dir
```

建议只跑 10 到 20 张，不要全跑：

```bash
python -u apply_kinect_holes_and_eval.py \
  --num_samples 20 \
  --depth_fill_method plane \
  --depth_fill_radius 15 \
  --plane_max_ring_radius 12 \
  --plane_min_boundary_points 12 \
  --run_name sd_diagnosis_n20_plane_r12 \
  --save_sd_diagnostics \
  --sd_diagnostics_dir sd_diagnostics/sd_diagnosis_n20_plane_r12 \
  --seed 123 \
  --visualize
```

这会保存：

```text
ideal_iq
noisy_iq
pred_iq_denoised
filled_iq_sdinpaint
filled_iq_full
depth_noisy
depth_depthcad
depth_sdinpaint
depth_full
depth_depthfill
gt_depth
hole_mask
confidence
per-channel IQ L1
I/Q phase error
amplitude error
depth MAE
```

这样可以直接证明：

```text
SD 失败是 IQ 物理一致性破坏，还是 depth 边界/scale 破坏。
```

### 实验 B：真实 RGB 可用性检查

如果 PBRT 有 RGB，下一步应该优先确认：

```text
RGB 和 depth/IQ 是否严格同相机、同视角、同分辨率对齐。
```

如果是，可以测试：

```text
RGB + sparse depth -> Marigold-DC / Depth Anything style completion
```

### 实验 C：Depth-domain diffusion baseline

把现在的 depth restoration task 改成：

```text
condition = noisy depth + mask + confidence + NS anchor
target = clean depth
```

训练一个 diffusion / flow matching depth restoration model。这个方向能真正发挥 diffusion 的生成 prior，又不会破坏 IQ 物理关系。

## 6. 当前结论

当前 pseudo-RGB SD inpaint 不是“diffusion 没用”，而是“使用 diffusion 的接口不对”：

```text
错误接口：
  IQ channel -> pseudo RGB -> natural image inpainting -> IQ -> depth

推荐接口：
  RGB-guided depth diffusion:
    RGB + sparse depth + mask -> dense depth

  或 depth-domain diffusion:
    noisy depth + anchor + mask + confidence -> clean dense depth

  或 physics-aware IQ diffusion:
    6-channel IQ jointly generated + depth/phase/amplitude losses
```

因此，后续如果要响应导师“用 diffusion 发挥更大能力”，不要回到单通道 pseudo-RGB inpainting，而要把 diffusion 放到 depth completion / depth restoration 的正确空间里。
