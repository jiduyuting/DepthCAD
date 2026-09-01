# Flow Endpoint Depth Restoration 最终结果整理

日期：2026-05-22

本文档整理当前最强结果、方法实现、创新点、对比方法和后续外部 SOTA 对比计划。

## 一句话结论

当前最强方法是：

```text
Conditional Flow + Endpoint Restoration Auxiliary
```

它不是 `DepthCAD -> SD Inpaint`，也不是 pseudo-RGB diffusion inpainting，而是在 depth restoration 空间中训练一个单模型：

```text
noisy depth + NS anchor + hole mask + confidence
-> time-conditioned flow model with endpoint auxiliary
-> dense restored depth
```

当前主结果：

```text
Checkpoint:
output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt

Eval:
output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_endpoint
```

## 最终主表

所有结果都在独立 `seed123` holdout set 上评估：

```text
depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123
```

| Method | Learned Models | Uses DepthCAD | Global MAE | Hole MAE | Valid MAE |
|---|---:|---|---:|---:|---:|
| Noisy input | 0 | No | 0.503548 | 2.911191 | 0.055985 |
| NS anchor | 0 | No | 0.104833 | 0.367608 | 0.055985 |
| DepthCAD/Plane base | 1 | Yes | 0.141459 | 0.505875 | 0.073716 |
| Two-stage learned completion | 2 | Yes | 0.082614 | 0.130479 | 0.073716 |
| Residual restoration | 1 | No | 0.056383 | 0.114204 | 0.045635 |
| Gated restoration | 1 | No | 0.058415 | 0.108534 | 0.049098 |
| **Flow + endpoint, w=2** | **1** | **No** | **0.050123** | 0.110299 | **0.038937** |
| Flow + endpoint, w=4 | 1 | No | 0.051335 | **0.108501** | 0.040708 |

推荐主方法：

```text
Flow + endpoint, endpoint_weight=2
```

理由：

- global MAE 最低；
- valid MAE 最低；
- hole MAE 也优于 previous residual restoration；
- 仍然是单 learned model；
- 不依赖 DepthCAD；
- 不依赖 pseudo-RGB SD inpainting。

hole-focused 变体：

```text
Flow + endpoint, endpoint_weight=4
```

理由：hole MAE 最低。

## Ranked Visualization

主方法 `Flow + endpoint, w=2` 的 ranked visualization 已整理：

```text
output/depth_flow_restoration_summary_final/ranked_cases_endpoint_w2.md
output/depth_flow_restoration_summary_final/ranked_cases_endpoint_w2.csv
```

原始 PNG 目录：

```text
output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_endpoint_ranked/visualizations
```

排序方式：

```text
model_hole_mae - anchor_hole_mae
```

负数表示模型优于 NS anchor。

统计结果：

```text
100 samples
improved / worsened / tied = 99 / 1 / 0
mean delta = -0.257228
mean improvement = 0.257228
```

前 6 个 best cases：

```text
white-room/1/139
white-room/1/165
white-room/1/127
white-room/1/145
white-room/1/76
white-room/1/1
```

后 6 个 weakest/worst cases：

```text
contemporary-bathroom/0/131
contemporary-bathroom/0/129
bathroom/1/214
bathroom/1/217
contemporary-bathroom/1/10
bathroom/1/212
```

唯一真正比 anchor 更差的 case 是：

```text
contemporary-bathroom/0/131
anchor_hole_mae = 0.003178
model_hole_mae  = 0.005263
delta           = 0.002085
```

这个 regression 的绝对误差非常小，说明主方法在 holdout set 上没有明显灾难性失败样本。

## 相比旧主模型的提升

旧主模型：

```text
Residual restoration:
Global 0.056383
Hole   0.114204
Valid  0.045635
```

新主模型：

```text
Flow + endpoint, w=2:
Global 0.050123
Hole   0.110299
Valid  0.038937
```

相对提升：

| Metric | Previous | New | Improvement |
|---|---:|---:|---:|
| Global MAE | 0.056383 | 0.050123 | 11.1% |
| Hole MAE | 0.114204 | 0.110299 | 3.4% |
| Valid MAE | 0.045635 | 0.038937 | 14.7% |

## 方法实现

### 输入

当前主方法的输入是：

```text
anchor_norm
noisy_norm
hole_mask
confidence
x_t
t embedding
```

其中：

- `noisy_norm`：带噪声和空洞的 noisy depth；
- `anchor_norm`：对 noisy depth 做 deterministic NS inpainting 得到的 depth anchor；
- `hole_mask`：空洞区域；
- `confidence`：观测可信度；
- `x_t`：flow 中间状态；
- `t embedding`：时间条件。

NS anchor 不是 learned model，只是确定性的 depth-domain 初始化。

### Flow 训练目标

训练时采样一个时间：

```text
t ~ Uniform(0, 1)
```

构造中间 depth：

```text
x_t = (1 - t) * anchor + t * gt
```

目标 velocity：

```text
v* = gt - anchor
```

模型学习：

```text
v_theta = model(condition, x_t, t)
```

基本 flow loss：

```text
L_flow = L1(v_theta, v*)
```

并加入 one-step reconstruction：

```text
pred_x1 = x_t + (1 - t) * v_theta
L_recon = L1(pred_x1, gt)
```

### Endpoint Auxiliary

纯 flow 效果不够好，因为它把模型容量分散在 `anchor -> gt` 的整条路径上；但这个任务最终只关心 clean endpoint。

所以加入 endpoint auxiliary：

```text
pred_endpoint = anchor + model(condition, anchor, t=0)
L_endpoint = L1(pred_endpoint, gt) + gradient/smoothness terms
```

最终 loss：

```text
L = L_flow + L_recon + lambda_endpoint * L_endpoint + gradient + smoothness
```

当前主模型：

```text
lambda_endpoint = 2.0
```

hole-focused 变体：

```text
lambda_endpoint = 4.0
```

### 推理

主推理方式使用 endpoint direct prediction：

```text
pred = anchor + model(condition, anchor, t=0)
```

Euler 多步采样作为 ablation：

```text
x_0 = anchor
x_{t+dt} = x_t + dt * model(condition, x_t, t)
```

实验显示：

| Variant | Sampling | Global | Hole | Valid |
|---|---|---:|---:|---:|
| Vanilla flow | Euler 32 | 0.086673 | 0.237337 | 0.058666 |
| Flow + endpoint | Euler 16 | 0.061741 | 0.135089 | 0.048106 |
| Flow + endpoint | Endpoint | **0.050123** | **0.110299** | **0.038937** |

这说明 endpoint auxiliary 不只是给了一个 direct head，它也改善了 flow field；但最终任务是 endpoint-dominated，所以 direct endpoint prediction 最适合作为主输出。

## 创新点

### 1. 从 pseudo-RGB diffusion 转向 depth-domain physical restoration

之前 SD inpainting 失败的核心原因不是 diffusion 没用，而是表示空间错了：

```text
6-channel IQ -> each channel copied to RGB -> Stable Diffusion inpaint
```

这个过程会破坏 ToF/IQ 相位和幅值一致性，导致 IQ 看似被补全，但解算出来的 depth 错误更大。

新的方法把 generative modeling 放回 depth restoration 空间：

```text
degraded depth observation -> restored metric depth
```

### 2. 单模型同时去噪和补洞

新方法不是：

```text
DepthCAD denoise -> SD inpaint -> depth
```

而是：

```text
noisy/holey depth -> one learned model -> restored dense depth
```

它在 hole 区域补全，在 valid 区域也降低噪声：

```text
Valid MAE: 0.055985 noisy -> 0.038937 restored
```

### 3. Endpoint-aligned conditional flow

vanilla flow 只优化整条路径，不完全匹配 restoration 任务。

本文方法提出 endpoint auxiliary，使 flow 模型同时满足：

```text
learn a time-conditioned restoration flow
learn the final anchor -> clean depth endpoint
```

这个设计使 flow/diffusion 真正服务于 sensor restoration，而不是做泛化插值。

### 4. 不依赖 DepthCAD，减少 learned-stage 数量

旧 two-stage completion：

```text
DepthCAD + learned completion = 2 learned models
```

新方法：

```text
NS anchor + conditional flow model = 1 learned model
```

NS anchor 是 deterministic algorithm，不是 learned model。

## 对比方法应该怎么组织

建议论文里分两类表。

### 表 1：路线诊断与内部对比

这张表必须保留，因为它回答“为什么不用原来的 DepthCAD/SD 路线”：

- Noisy input；
- DepthCAD only；
- SD Inpaint only；
- DepthCAD + SD Inpaint；
- DepthCAD + depth fill；
- Two-stage learned completion；
- residual restoration；
- gated restoration；
- proposed flow endpoint。

这张表的作用不是展示 SOTA，而是展示技术路线转变：

```text
pseudo-RGB image-domain inpainting -> depth-domain restoration
```

### 表 2：外部 SOTA 对比

如果有真实严格对齐 RGB，应该补做最新 RGB-D depth completion 方法：

- Marigold-DC；
- SteeredMarigold；
- OMNI-DC；
- CompletionFormer；
- LingBot-Depth / masked depth modeling 类方法。

这张表才回答：

```text
和当前 depth completion 最新方法比怎么样？
```

## 要不要和最新方法对比？

要，但前提是公平。

如果你有真实对齐 RGB：

```text
RGB + noisy/sparse depth + mask -> dense depth
```

那么必须至少跑一个 diffusion depth completion baseline，例如 Marigold-DC 或 SteeredMarigold。

如果没有真实 RGB，只有 IQ/depth：

```text
不要把 IQ 复制成 RGB 后声称和 RGB-D SOTA 公平比较。
```

这种情况下，Marigold-DC / SteeredMarigold / OMNI-DC 可以作为 related work 讨论，不能作为主表公平对比。

## 下一步最值得做的实验

优先级：

1. `endpoint_weight=8` 已完成，结果开始退化，因此 endpoint 权重扫描可以停止。
2. final model ranked best/worst visualization 和 failure case analysis 已完成。
3. 下一步优先跑 `Flow endpoint + amplitude`，因为现有 cache 已经支持，不需要重新生成数据。
4. 然后生成带 IQ 的 cache，跑 `Flow endpoint + IQ + amplitude`。
5. 最后再做 `DepthCAD-HoleAware`，因为它需要重新训练/评估 DepthCAD，成本更高。

具体命令见：

```text
SENSOR_ONLY_NEXT_EXPERIMENTS_ZH.md
```

## 最终汇报话术

可以这样讲：

```text
我们先验证了 pseudo-RGB SD inpainting 在 ToF/IQ 数据上失败，原因是自然图像 diffusion prior 破坏了 I/Q 相位和幅值一致性。随后我们把生成式建模从 image domain 转到 depth restoration domain，提出 conditional flow with endpoint auxiliary。纯 flow 虽然能改善 anchor，但不如 direct restoration；加入 endpoint auxiliary 后，模型同时学习 time-conditioned restoration flow 和最终 clean-depth endpoint，在独立 holdout set 上超过 residual restoration baseline。
```

最简版本：

```text
不是 diffusion 没用，而是之前把 diffusion 用错了空间。现在将 flow/diffusion 用在 depth-domain restoration，并通过 endpoint auxiliary 对齐最终恢复目标，得到当前最优结果。
```
