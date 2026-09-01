# RGB-free ToF Depth Restoration: Final Flow Backbone Results

Date: 2026-06-09

## 1. Current Final Model

Current best model:

```text
Physics-aware endpoint-aligned conditional flow
+ noisy_amp input
+ noisy_ns anchor
+ transformer_bottleneck backbone
+ endpoint_weight = 2.0
```

Checkpoint:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt
```

Seed123 holdout result:

| Metric | Value |
|---|---:|
| Global MAE | 0.037659 |
| Hole MAE | 0.104364 |
| Valid MAE | 0.025259 |

This is the current strongest result across global, hole, and valid regions.

## 2. Main Comparison Table

All rows below use the seed123 100-sample holdout unless noted.

| Method | Domain / Role | Global MAE | Hole MAE | Valid MAE |
|---|---|---:|---:|---:|
| Noisy with holes | Raw corrupted observation | 0.503548 | 2.911191 | 0.055985 |
| DepthCAD-HoleAware | IQ-domain generative baseline | 0.141230 | 0.550396 | 0.065169 |
| NS Anchor | Deterministic depth-domain anchor | 0.104833 | 0.367608 | 0.055985 |
| Depth fill base | Traditional depth fill baseline | 0.141459 | 0.505875 | 0.073716 |
| Original Flow + lightweight ResUNet | Depth-domain endpoint flow | 0.046417 | 0.111737 | 0.034274 |
| Flow + Large ResUNet | Larger residual CNN backbone | 0.038709 | 0.105873 | 0.026223 |
| Flow + Transformer Bottleneck | Final model | **0.037659** | **0.104364** | **0.025259** |

## 3. What Each Result Shows

### DepthCAD-HoleAware

DepthCAD-HoleAware was fine-tuned on Kinect-style hole IQ data:

```text
input  = noisy IQ with Kinect-style holes + confidence
output = clean IQ
then IQ -> depth
```

It substantially improves over raw noisy IQ:

```text
Hole MAE: 2.911191 -> 0.550396
```

However, it remains much worse than the proposed depth-domain flow:

```text
DepthCAD-HoleAware hole MAE: 0.550396
Final flow hole MAE:         0.104364
```

This supports the conclusion that the limitation is not merely that the original DepthCAD was not trained with holes. Even after hole-aware IQ-domain fine-tuning, pseudo-RGB/IQ generation struggles to enforce multi-frequency ToF physical consistency inside missing regions.

### Original Flow + Lightweight ResUNet

The original endpoint flow already strongly outperforms DepthCAD-HoleAware:

```text
DepthCAD-HoleAware global/hole/valid = 0.141230 / 0.550396 / 0.065169
Original Flow global/hole/valid      = 0.046417 / 0.111737 / 0.034274
```

This validates the main route: depth-domain conditional restoration is more suitable than IQ-domain pseudo-RGB inpainting for this RGB-free ToF problem.

### Large ResUNet Backbone

Large ResUNet improves the original lightweight backbone:

```text
Original Flow:
  global = 0.046417
  hole   = 0.111737
  valid  = 0.034274

Large ResUNet:
  global = 0.038709
  hole   = 0.105873
  valid  = 0.026223
```

Interpretation:

```text
The original lightweight UNet had a capacity ceiling.
Increasing residual CNN capacity improves both hole completion and valid-region restoration.
```

Large ResUNet changes:

```text
base_channels: 32 -> 48
blocks: ordinary conv blocks -> residual conv blocks
stage depth: more residual blocks per encoder/decoder stage
```

### Transformer Bottleneck Backbone

Transformer Bottleneck gives the best final result:

```text
Large ResUNet:
  global = 0.038709
  hole   = 0.105873
  valid  = 0.026223

Transformer Bottleneck:
  global = 0.037659
  hole   = 0.104364
  valid  = 0.025259
```

Interpretation:

```text
Most of the gain comes from using a stronger backbone.
Transformer bottleneck gives a smaller but consistent additional gain.
```

Transformer Bottleneck changes:

```text
CNN encoder/decoder remains.
Self-attention is added only at the low-resolution UNet bottleneck.
transformer_pool=2 reduces the bottleneck tokens before attention.
```

This gives long-range structural reasoning without the cost of full-resolution attention.

## 4. Best-Hole Checkpoint Check

Additional seed123 evaluation:

| Model | Checkpoint | Epoch | Global MAE | Hole MAE | Valid MAE |
|---|---|---:|---:|---:|---:|
| Large ResUNet | best.pt | 115 | **0.038709** | **0.105873** | **0.026223** |
| Large ResUNet | best_hole.pt | 120 | 0.039408 | 0.107556 | 0.026739 |
| Transformer Bottleneck | best.pt | 118 | **0.037659** | **0.104364** | **0.025259** |
| Transformer Bottleneck | best_hole.pt | 118 | **0.037659** | **0.104364** | **0.025259** |

Conclusion:

```text
Large ResUNet best_hole.pt does not generalize better on seed123.
Transformer best.pt and best_hole.pt are the same checkpoint.
Use transformer_bottleneck best.pt as the final model.
```

## 5. Recommended Figures For Report

### Final Transformer Best / Worst Cases

Directory:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations
```

Recommended examples:

```text
Best:
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations/vis_000_best_anchor_hole_white-room_1_139.png
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations/vis_001_best_anchor_hole_white-room_1_165.png
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations/vis_002_best_anchor_hole_white-room_1_1.png

Worst:
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations/vis_006_worst_anchor_hole_breakfast_0_151.png
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations/vis_007_worst_anchor_hole_contemporary-bathroom_0_129.png
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations/vis_010_worst_anchor_hole_contemporary-bathroom_0_173.png
```

### Large ResUNet Comparison Cases

Directory:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_large_res48_b2/eval_seed123_endpoint/visualizations
```

Recommended examples:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_large_res48_b2/eval_seed123_endpoint/visualizations/vis_000_best_anchor_hole_white-room_1_139.png
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_large_res48_b2/eval_seed123_endpoint/visualizations/vis_006_worst_anchor_hole_contemporary-bathroom_0_129.png
```

### DepthCAD-HoleAware Qualitative Baseline

Directory:

```text
output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123/visualizations
```

Recommended examples:

```text
output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123/visualizations/vis_bathroom_1_214.png
output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123/visualizations/vis_bathroom_1_217.png
output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123/visualizations/vis_bathroom_0_121.png
```

Use these to show:

```text
DepthCAD-HoleAware can look visually plausible,
but its hole MAE remains much higher than depth-domain flow.
```

## 6. Suggested Verbal Summary

Chinese:

```text
我们首先补充了一个更公平的 IQ-domain baseline：DepthCAD-HoleAware。
它直接用带 Kinect 空洞的 noisy IQ 和 confidence fine-tune DepthCAD，输出 clean IQ 后再转 depth。
结果显示它确实能显著改善 noisy 输入，hole MAE 从 2.91 降到 0.55。
但是相比 depth-domain endpoint flow 的 0.10 左右仍然差距很大。
这说明问题不是旧 DepthCAD 没见过空洞，而是 IQ-domain pseudo-RGB generation 难以保证 ToF 多频相位物理一致性。

在 flow 主方法上，我们进一步验证了 backbone 容量。
原始轻量 ResUNet flow 已经达到 0.1117 的 hole MAE。
换成 large residual backbone 后，global/hole/valid 全部提升。
再加入 transformer bottleneck 后获得当前最优结果：global 0.0377，hole 0.1044，valid 0.0253。
这说明增强 backbone 和低分辨率长程结构建模对 RGB-free ToF depth restoration 有帮助。
```

English:

```text
We first introduce a fairer IQ-domain baseline, DepthCAD-HoleAware, which is fine-tuned directly on noisy IQ with Kinect-style holes and confidence maps. It substantially improves over raw noisy IQ, reducing hole MAE from 2.91 to 0.55. However, it remains far worse than the proposed depth-domain endpoint flow, whose hole MAE is around 0.10. This suggests that the limitation is not simply that the original DepthCAD was not trained with holes; rather, IQ-domain pseudo-RGB generation struggles to preserve multi-frequency ToF phase consistency in missing regions.

For the proposed flow method, we further study backbone capacity. The original lightweight ResUNet flow already strongly outperforms DepthCAD-HoleAware. Replacing it with a larger residual UNet improves all metrics, and adding a transformer bottleneck achieves the best result, with global MAE 0.0377, hole MAE 0.1044, and valid MAE 0.0253. This indicates that stronger backbone capacity and low-resolution long-range structural reasoning are beneficial for RGB-free ToF depth restoration.
```

## 7. Current Recommendation

Use this as the final method:

```text
Flow + Transformer Bottleneck
input_mode = noisy_amp
anchor_mode = noisy_ns
endpoint_weight = 2.0
sampling_mode = endpoint
```

Use Large ResUNet as backbone ablation.

Use DepthCAD-HoleAware as the fair IQ-domain baseline.

