## Flow Method Motivation Draft

主线是：

```text
Endpoint-aligned conditional flow for RGB-free ToF depth restoration
```



## 1. 核心问题要怎么讲

我们要解决的不是普通 RGB-D depth completion，而是 RGB-free ToF sensor depth restoration：

```text
corrupted ToF depth/raw measurements + holes/confidence/amplitude
-> restored dense metric depth
```

真实 ToF 深度中的失效区域来自低反射、过曝/饱和、多路径、远距离弱信号和传感器解算失败。这些 hole 不是随机 sparse samples，也不是自然图像中可见纹理缺失。它们带有明确的传感器物理属性：I/Q 相位、幅值、confidence、有效深度边界和 metric scale 都必须一致。

因此，论文的第一句话可以避免泛泛说 "depth completion is important"，而应直接指出现有方法的错位：

```text
Existing image-domain inpainting and RGB-guided depth completion methods are not directly designed for RGB-free ToF restoration, where the missing regions must be recovered under sensor confidence, metric scale, and ToF phase-derived geometry constraints.
```

## 2. 为什么不用 IQ / pseudo-RGB diffusion 或 DepthCAD 作为主方法

### 2.1 表示空间不对

DepthCAD / pseudo-RGB diffusion 路线大致是：

```text
ToF IQ channels -> pseudo RGB or IQ-domain generation -> depth estimator -> metric depth
```

这个路线的问题不是 "生成模型不够强"，而是生成空间与最终目标错位。ToF depth 由多频 I/Q pair 共同决定：

```text
phase = atan2(Q, I)
amplitude = sqrt(I^2 + Q^2)
multi-frequency consistency -> metric depth
```

如果把每个 IQ 通道独立当作灰度图去 inpaint，模型可以生成视觉上平滑的 IQ-like pattern，但这些 pattern 未必满足同一频率 I/Q 的相位关系，也未必满足不同频率对应同一个真实深度。最终表现就是：IQ 看起来被补上了，但解算成 depth 后误差很大。

这给出一个合理 motivation：

```text
For ToF restoration, perceptual plausibility in the IQ or pseudo-RGB space is insufficient; the restored signal must be physically consistent after projection to metric depth.
```

### 2.2 实验证据已经支持这个判断

在 seed123 100-sample PBRT holdout 上，当前结果显示：

| Method | Domain / Role | Global MAE | Hole MAE | Valid MAE |
|---|---|---:|---:|---:|
| Noisy with holes | corrupted observation | 0.503548 | 2.911191 | 0.055985 |
| DepthCAD-HoleAware | IQ-domain generative baseline | 0.141230 | 0.550396 | 0.065169 |
| NS Anchor | deterministic depth anchor | 0.104833 | 0.367608 | 0.055985 |
| ProPainter | image/video inpainting baseline | 0.097623 | 0.321613 | 0.055985 |
| Original Flow + lightweight ResUNet | depth-domain endpoint flow | 0.046417 | 0.111737 | 0.034274 |
| Flow + Large ResUNet | stronger CNN backbone | 0.038709 | 0.105873 | 0.026223 |
| Flow + Transformer Bottleneck | final model | **0.037659** | **0.104364** | **0.025259** |

这个表要服务于一个论点：

```text
Even when DepthCAD is fine-tuned on hole-aware IQ inputs, it remains much worse than depth-domain endpoint flow. This suggests that the key limitation is not merely data mismatch, but the difficulty of enforcing ToF physical consistency through IQ/image-domain generation.
```

## 3. 为什么不用传统 inpainting / NS anchor 作为最终结果

NS anchor 的价值是给出确定性的 dense initial geometry：

```text
noisy depth + hole mask -> NS inpainted anchor
```

但它有三个限制：

1. 它主要依赖局部边界传播，遇到大洞、复杂遮挡和深度 discontinuity 时缺少全局结构判断。
2. 它只填 hole，不会修复 valid region 中的传感器噪声。
3. 它没有利用 amplitude/confidence/raw9 里的可靠性线索。

实验上，NS anchor 的 hole MAE 是 0.367608，而最终 flow 是 0.104364；valid MAE 方面，NS anchor 等同 noisy valid region，为 0.055985，而最终 flow 降到 0.025259。这个结果很关键，因为它说明模型不是简单 "补洞"，而是在做 restoration：

```text
The model improves both missing regions and originally valid regions, indicating that the task is better formulated as sensor depth restoration rather than hole filling alone.
```

## 4. 为什么是 flow，而不是普通 residual UNet

### 4.1 Flow 的合理动机

这个任务天然有一个 anchor-to-clean 的结构：

```text
anchor depth x_0  ->  clean metric depth x_1
```

普通 residual UNet 直接学习：

```text
pred = anchor + residual(condition)
```

这个做法有效，但它把恢复过程压缩成一次黑盒 correction。Conditional flow 更自然地把问题表示为从确定性 anchor 到 clean depth 的连续恢复路径：

```text
x_t = (1 - t) * anchor + t * gt
v* = gt - anchor
v_theta = f_theta(x_t, t, condition)
```

condition 包括：

```text
noisy depth
NS anchor
hole mask
confidence
optional amplitude/raw9-derived signal
```

这样做有三个优势：

1. Flow 使用 anchor 作为起点，而不是从噪声或空白 depth 生成，学习难度更低。
2. Time-conditioned training 让模型在不同 corruption-to-clean 状态上学习一致的 correction field，不只是拟合单点 residual。
3. Flow 的目标是 metric depth 端点，不需要生成物理上难约束的 IQ 或 pseudo-RGB。

可以在论文中这样写：

```text
We use conditional flow not as a generic generative prior, but as an anchor-to-clean restoration model. The deterministic inpainted depth provides a physically meaningful starting point, while the learned flow predicts how this anchor should move toward sensor-consistent metric depth under hole, confidence, and amplitude conditioning.
```

### 4.2 为什么还要 endpoint auxiliary

Vanilla flow 优化整条路径，但该任务最终只关心 endpoint：

```text
clean restored depth
```

如果只训练 flow field，模型容量会被分散到整个路径。当前方法加入 endpoint-aligned auxiliary：

```text
pred_endpoint = anchor + f_theta(anchor, t=0, condition)
L_endpoint = L1(pred_endpoint, gt) + gradient + smoothness
```

总损失：

```text
L = L_flow + L_recon + lambda_endpoint * L_endpoint
    + gradient/smoothness regularization
```

推理时直接使用：

```text
pred = anchor + f_theta(anchor, t=0, condition)
```

这使方法既保留 flow 的路径建模，又对齐最终 restoration 目标。可以强调：

```text
The endpoint auxiliary is not merely an extra prediction head. It resolves a task mismatch: dense depth restoration is endpoint-dominated, while vanilla flow matching distributes supervision over the entire interpolation path.
```

### 4.3 相比 residual restoration 的数值优势

旧 single-model residual restoration：

| Method | Global MAE | Hole MAE | Valid MAE |
|---|---:|---:|---:|
| Residual restoration | 0.056383 | 0.114204 | 0.045635 |
| Depth-only endpoint flow | 0.050123 | 0.110299 | 0.038937 |
| Final transformer endpoint flow | **0.037659** | **0.104364** | **0.025259** |

这里的写法要谨慎：不要说 "flow 理论上一定优于 residual"。更稳妥的说法是：

```text
The flow formulation gives a better inductive bias for anchored restoration, and the endpoint alignment makes this bias useful for the final metric-depth prediction. Empirically, this consistently improves global, hole, and valid-region MAE over the residual restoration baseline.
```

## 5. 为什么用 transformer bottleneck

大洞和深度结构不完全是局部问题。墙面、地面、桌面、遮挡边缘和远距离区域需要更长程的几何关系。全分辨率 attention 太贵，也容易过拟合；所以当前方法只在 UNet bottleneck 的低分辨率特征上加 self-attention：

```text
CNN encoder/decoder for local geometry and boundaries
low-resolution transformer bottleneck for long-range structure
```

实验显示：

| Backbone | Global MAE | Hole MAE | Valid MAE |
|---|---:|---:|---:|
| Lightweight ResUNet | 0.046417 | 0.111737 | 0.034274 |
| Large ResUNet | 0.038709 | 0.105873 | 0.026223 |
| Transformer Bottleneck | **0.037659** | **0.104364** | **0.025259** |

结论可以写成：

```text
Most of the gain comes from moving to a stronger residual backbone, while bottleneck self-attention provides a small but consistent additional improvement by modeling long-range structure at low computational cost.
```

## 6. 怎么组织


### Contribution 1: Problem formulation

提出 RGB-free ToF depth restoration，而不是 RGB-guided completion 或 image-domain inpainting：

```text
We formulate RGB-free ToF depth restoration as conditional metric-depth generation from corrupted sensor measurements, hole masks, confidence, deterministic anchors, and optional amplitude cues.
```

### Contribution 2: Method

提出 endpoint-aligned conditional flow：

```text
We propose an endpoint-aligned conditional flow model that learns an anchor-to-clean restoration field and adds endpoint supervision to match the final dense-depth objective.
```

### Contribution 3: Physics-aware conditioning

用 NS anchor、mask、confidence、amplitude/raw9-derived signal 做条件，而不是 pseudo-RGB IQ inpainting：

```text
The model operates directly in metric depth space and uses sensor reliability cues, avoiding physically inconsistent independent IQ-channel generation.
```

### Contribution 4: Evidence

系统对比：

```text
Noisy depth
NS anchor
DepthCAD / DepthCAD-HoleAware
ProPainter / image inpainting
residual restoration
flow ablations
backbone ablations
real raw9 masked fine-tuning
```

其中真实 raw9 fine-tuning 当前可作为重要补充：

```text
After real raw9 masked self-supervised fine-tuning, the model improves masked completion over NS by 48.43% on 41 real paired samples, with 41/41 cases improved.
```

## 7. English Paper Draft

### Candidate title

```text
Endpoint-Aligned Conditional Flow for RGB-Free Time-of-Flight Depth Restoration
```

备选：

```text
Sensor-Conditioned Flow Matching for RGB-Free ToF Depth Restoration
Physics-Aware Endpoint Flow for ToF Depth Hole Restoration
From Sensor Anchors to Metric Depth: Endpoint-Aligned Flow for RGB-Free ToF Restoration
```

### Abstract draft

Time-of-flight (ToF) depth sensors often produce missing or unreliable measurements due to low reflectance, saturation, multipath interference, and weak long-range signals. While image inpainting and RGB-guided depth completion have made rapid progress, they are not directly suited to RGB-free ToF restoration, where the recovered signal must preserve metric scale and sensor-derived geometric consistency. In particular, inpainting ToF I/Q measurements as pseudo-RGB images can produce visually plausible signals that violate multi-frequency phase consistency and lead to large depth errors after sensor decoding.

We propose an endpoint-aligned conditional flow model for RGB-free ToF depth restoration. Instead of generating I/Q or pseudo-RGB images, our method operates directly in metric depth space. A deterministic depth inpainting algorithm first provides a dense anchor from the corrupted depth map. The proposed model then learns a conditional flow from this anchor to the clean depth, using the corrupted depth, hole mask, confidence, and amplitude-derived cues as conditions. To align flow matching with the restoration objective, we introduce an endpoint auxiliary loss that directly supervises the final anchor-to-clean prediction, together with gradient and smoothness regularization. A transformer bottleneck further captures long-range scene structure while preserving efficient convolutional decoding.

On a held-out synthetic ToF benchmark, the proposed method substantially outperforms deterministic inpainting, image/video inpainting, IQ-domain generative restoration, and residual depth restoration baselines. The final model reduces hole MAE from 0.3676 m with deterministic inpainting and 0.5504 m with a hole-aware DepthCAD baseline to 0.1044 m, while also reducing valid-region MAE from 0.0560 m to 0.0253 m. On real raw9 ToF measurements, masked self-supervised fine-tuning improves masked completion over deterministic inpainting by 48.4% across 41 paired samples. These results suggest that physically grounded, endpoint-aligned depth-domain flow is a more effective formulation for RGB-free ToF restoration than image-domain or IQ-domain generation.

### Introduction draft

Depth sensing is a core component of robotics, augmented reality, human-computer interaction, and 3D scene understanding. Among active depth sensors, time-of-flight cameras are attractive because they provide metric depth without relying on ambient texture. However, ToF depth maps are often incomplete and noisy. Invalid measurements appear around low-reflectance surfaces, saturated regions, object boundaries, distant areas, and multipath-contaminated regions. Recovering dense and accurate depth from such measurements is therefore an important sensor restoration problem.

Most recent depth completion methods assume access to an aligned RGB image or sparse but reliable depth samples. This assumption is not always valid for compact ToF systems or raw sensor pipelines, where RGB may be absent, misaligned, or unavailable at the required exposure and frame rate. A natural alternative is to treat the corrupted ToF measurements as images and apply image inpainting or diffusion-based generation. However, this is a poor match to ToF physics. Multi-frequency ToF depth is derived from coupled I/Q measurements, and the final quantity of interest is metric depth. Independently inpainting I/Q channels or converting them to pseudo-RGB images may produce visually plausible intermediate signals, but it does not enforce phase, amplitude, confidence, or metric-depth consistency.

This motivates a different formulation: RGB-free ToF restoration should be performed directly in depth space, while still using sensor reliability cues that explain where and why the observation is corrupted. Traditional depth-domain inpainting provides a useful starting point. For example, Navier-Stokes style inpainting can propagate nearby depth into holes and produce a dense anchor. Yet such deterministic anchors are limited by local propagation, cannot reliably infer large missing regions, and leave valid-region sensor noise unchanged. The key question is therefore how to combine the stability of a deterministic geometric anchor with the expressive power of a learned restoration model.

We propose endpoint-aligned conditional flow for this task. The method starts from a dense anchor obtained by deterministic depth inpainting and learns how this anchor should move toward the clean metric depth. During training, we sample intermediate states between the anchor and the ground-truth depth and supervise a time-conditioned velocity field. This turns restoration into an anchor-to-clean flow rather than unconstrained image generation. Because the final application only needs the clean endpoint, we further introduce an endpoint auxiliary loss that directly supervises the prediction obtained by applying the learned correction to the anchor. This aligns the flow objective with the endpoint-dominated nature of depth restoration.

Our method is conditioned on the corrupted depth, hole mask, confidence map, and amplitude-derived cues from raw ToF measurements. This conditioning allows the network to distinguish unreliable missing regions from reliable observed regions, while operating in metric depth space avoids the physical inconsistency of pseudo-RGB or independently generated I/Q channels. To handle larger holes and global structures, we add self-attention only at the low-resolution UNet bottleneck, combining long-range reasoning with efficient convolutional reconstruction.

We evaluate the method on a held-out synthetic ToF restoration benchmark and on real raw9 ToF measurements. The proposed model outperforms deterministic depth inpainting, image/video inpainting, IQ-domain generative restoration, and direct residual restoration. Importantly, it improves not only hole-region errors but also valid-region errors, supporting the view that ToF recovery should be treated as full sensor depth restoration rather than hole filling alone.

Our contributions are:

1. We formulate RGB-free ToF recovery as sensor-conditioned metric depth restoration, avoiding pseudo-RGB and independently generated I/Q inpainting.
2. We propose an endpoint-aligned conditional flow model that learns an anchor-to-clean restoration field and aligns flow matching with the final dense-depth endpoint.
3. We incorporate deterministic depth anchors, hole masks, confidence, and amplitude cues into a single learned restoration model with a transformer bottleneck for long-range structure.
4. We provide systematic experiments against deterministic inpainting, image/video inpainting, IQ-domain restoration, residual restoration, and flow/backbone ablations, including real raw9 masked self-supervised adaptation.

### Method draft

Let \(d^n\) denote the corrupted depth observation, \(m\) the hole mask, \(c\) the confidence map, and \(a\) an amplitude-derived reliability cue from raw ToF measurements when available. We first compute a deterministic dense anchor

```text
d^a = Phi_NS(d^n, m),
```

where \(Phi_NS\) is a non-learned depth-domain inpainting operator. The target is the clean metric depth \(d\).

The conditional flow is trained between the anchor and the clean depth. For a sampled time \(t ~ U(0, 1)\), we construct

```text
x_t = (1 - t) d^a + t d,
v* = d - d^a.
```

The network predicts

```text
v_theta = f_theta(x_t, t, d^n, d^a, m, c, a),
```

and is optimized with a masked velocity loss over hole and valid regions. We also reconstruct the endpoint from an intermediate state:

```text
hat_d_t = x_t + (1 - t) v_theta.
```

This gives the flow and reconstruction losses:

```text
L_flow = ||v_theta - v*||_1
L_recon = ||hat_d_t - d||_1
```

with separate weighting over missing and valid regions.

To align training with the restoration endpoint, we add direct endpoint supervision at \(t=0\):

```text
hat_d = d^a + f_theta(d^a, 0, d^n, d^a, m, c, a).
```

The endpoint loss is

```text
L_endpoint = L1(hat_d, d) + lambda_grad L_grad(hat_d, d)
             + lambda_smooth L_smooth(hat_d, d^a).
```

The full objective is

```text
L = L_flow + L_recon + lambda_endpoint L_endpoint.
```

At inference time, we use direct endpoint prediction:

```text
hat_d = d^a + f_theta(d^a, 0, d^n, d^a, m, c, a).
```

Euler integration of the learned flow can be retained as an ablation, but endpoint prediction is the main output because restoration quality is measured at the clean dense depth endpoint.

### Experiments draft

We evaluate on an independent 100-sample PBRT/ToF holdout set with Kinect-style holes and report global MAE, hole-region MAE, and valid-region MAE in meters. The baseline set includes corrupted input, deterministic NS inpainting, image/video inpainting through ProPainter, DepthCAD-HoleAware as an IQ-domain generative baseline, residual depth restoration, and flow/backbone variants.

Main synthetic results:

| Method | Global MAE | Hole MAE | Valid MAE |
|---|---:|---:|---:|
| Noisy with holes | 0.503548 | 2.911191 | 0.055985 |
| NS Anchor | 0.104833 | 0.367608 | 0.055985 |
| ProPainter | 0.097623 | 0.321613 | 0.055985 |
| DepthCAD-HoleAware | 0.141230 | 0.550396 | 0.065169 |
| Residual restoration | 0.056383 | 0.114204 | 0.045635 |
| Depth-only endpoint flow | 0.050123 | 0.110299 | 0.038937 |
| Flow + Large ResUNet | 0.038709 | 0.105873 | 0.026223 |
| Flow + Transformer Bottleneck | **0.037659** | **0.104364** | **0.025259** |

The final model reduces hole MAE by 71.6% relative to NS inpainting and by 81.0% relative to DepthCAD-HoleAware. It also reduces valid-region MAE by 54.9% relative to the noisy observation, showing that it restores the entire depth map rather than filling holes only.

For real raw9 ToF data, we use masked self-supervised fine-tuning on 41 paired samples. Synthetic-trained models do not transfer zero-shot reliably, indicating a real raw distribution gap. After real-domain adaptation, the stronger fine-tuned model reduces masked MAE from 0.09074 m with NS inpainting to 0.04680 m, improving over NS by 48.43% and improving all 41/41 cases. With real-hole-shaped masks, the model still improves over NS by 25.63%, indicating that it is not limited to simple random masks.

### Discussion draft

The results support three conclusions. First, depth-domain restoration is a better interface for RGB-free ToF recovery than pseudo-RGB or independently generated IQ inpainting. Second, deterministic anchors are useful but insufficient: they stabilize the task, while the learned flow corrects anchor errors and denoises valid regions. Third, endpoint alignment is important because restoration is evaluated only at the final clean depth, whereas vanilla flow matching distributes supervision over intermediate states.

The main limitation is domain transfer. Synthetic-trained raw9 models do not directly generalize to real ToF raw distributions. However, masked self-supervised fine-tuning on real raw9 data substantially improves performance, suggesting a practical adaptation path. Future work should strengthen real-hole-shaped training, evaluate on larger real datasets, and compare against RGB-guided depth completion when strictly aligned RGB is available.

## 8. 写作上要避免的坑

1. 不要把文章写成 "我们用了 flow matching，所以先进"。要写成 "ToF depth restoration 有 anchor-to-clean 结构，flow 是这个结构的合适建模方式"。
2. 不要声称 IQ-domain 方法一定不行。更准确是：独立 IQ/pseudo-RGB generation 很难保证 ToF phase-to-depth consistency；在当前公平 baseline 中，hole-aware IQ generation 仍显著落后。
3. 不要把 NS anchor 说成缺点。它是方法的重要物理先验；创新点是用 learned flow 修正 deterministic anchor。
4. 不要只报 hole MAE。可能问 valid region 有没有被破坏，所以 global/hole/valid 三个指标都要保留。
5. 不要缺少外部 baseline。ProPainter 已经有了，但如果有严格对齐 RGB，最好补一个 RGB-guided depth completion/diffusion baseline；如果没有，就必须明确本文是 RGB-free setting。
6. 不要把真实数据结论说过头。当前真实结果是 masked self-test 和 fine-tuning 结果，不是带 GT 的真实 observed-hole benchmark。可以作为 adaptation evidence，不宜包装成完整真实 GT SOTA。

## 9. 下一版论文需要补的材料

优先级最高：

1. 图 1：问题设定和方法 pipeline。显示 corrupted depth/raw9 -> NS anchor -> endpoint flow -> restored depth。
2. 图 2：为什么 pseudo-RGB/IQ inpainting 会失败。显示 IQ visually plausible 但 decoded depth 错误。
3. 主表：synthetic holdout global/hole/valid MAE。
4. Ablation 表：residual vs vanilla flow vs endpoint flow vs large ResUNet vs transformer bottleneck。
5. Real raw9 masked self-supervised table：zero-shot failure -> fine-tuning success。
6. Failure cases：ProPainter 比我们好的 3 个样本，以及 final flow worst cases。
7. Reproducibility supplement：数据生成、mask 生成、训练 split、checkpoint、evaluation scripts。

主文保留 problem/method/main table/ablation/real adaptation，pseudo-RGB failure 详细诊断放 supplement。
