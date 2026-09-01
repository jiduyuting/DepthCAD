# PBRT 与过曝/真实 Raw9 方法对比梳理

Date: 2026-07-03

## 口径说明

本文只整理当前本机已有结果，不把远景 `far_pic` 照片纳入主比较。

需要分清三类评测口径：

```text
1. PBRT seed123 holdout:
   有 GT，MAE/RMSE 是真正监督指标，可以横向比较。

2. 真实 raw9 / 过曝 raw:
   没有 GT，只能看 mask ratio、model-anchor/raw 改变量、可视化。
   这些指标不能当作准确度，只能当作无参考诊断。

3. real masked self-test:
   在真实 valid depth 上人工挖洞，有伪 GT，可以报告 mask MAE。
   这是目前判断真实泛化最可靠的代理指标。
```

## 1. PBRT Seed123 有监督结果

数据集：

```text
depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123
```

单位：米。越低越好。

| 方法 | 类型 | Global MAE | Hole MAE | Valid MAE | 结果/可视化位置 |
|---|---|---:|---:|---:|---|
| Noisy input | 原始退化输入 | 0.503548 | 2.911191 | 0.055985 | `output/pbrt_propainter_seed123/evaluation/summary.json` |
| OpenCV NS r15 | 传统 depth inpaint | 0.108885 | 0.393456 | 0.055985 | ProPainter 对比图里包含 NS：`output/pbrt_propainter_seed123/evaluation/visualizations` |
| ProPainter | 外部 video/RGB inpaint baseline，depth-as-gray | 0.097623 | 0.321613 | 0.055985 | `output/pbrt_propainter_seed123/evaluation/visualizations` |
| DepthCAD-HoleAware | IQ-domain generative baseline | 0.141230 | 0.550396 | 0.065169 | `output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123/visualizations` |
| Residual restoration | 早期单模型 depth restoration | 0.056383 | 0.114204 | 0.045635 | `output/depth_restoration_unet_noisy_ns_n1000/eval_seed123_ranked/visualizations` |
| Depth-only endpoint flow | depth-only flow 主线 | 0.050123 | 0.110299 | 0.038937 | `output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_endpoint/visualizations` |
| Large ResUNet endpoint flow | noisy_amp + larger CNN backbone | 0.038709 | 0.105873 | 0.026223 | `output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_large_res48_b2/eval_seed123_endpoint/visualizations` |
| Final transformer endpoint flow | 当前最终模型 | **0.037659** | **0.104364** | **0.025259** | `output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/visualizations` |

结论：

```text
PBRT 上当前最强的是 final transformer endpoint flow。
ProPainter 比 OpenCV NS 强，但 hole MAE 仍约为 final flow 的 3.08x。
DepthCAD-HoleAware 说明 IQ-domain 生成式路线有改善，但仍明显弱于 depth-domain flow。
```

## 2. PBRT 外部/候选方法状态

| 方法 | 当前状态 | 是否有同口径 PBRT seed123 表 | 备注 |
|---|---|---|---|
| ProPainter | 已跑完、已解码、已量化 | 有 | `output/pbrt_propainter_seed123/evaluation/summary.json` |
| RAD official | 适配脚本存在，但没有完成同口径结果 | 无 | 本地缺 RAD 官方 checkpoint；`depthcad_pbrt_rad*` 不是 RAD 官方 checkpoint 结构 |
| DepthCAD RAD-inspired | 有大量训练 checkpoint | 无统一 seed123 summary | 这是本项目内 RAD-inspired 模块，不等同于 official RAD baseline |
| Stable Diffusion / SD inpaint | 有旧诊断 | 非完整 seed123 主表 | `kinect_evaluation/sd_diagnosis_n20_plane_r12`，样本少且旧口径 |
| Marigold / HYPIR | 有旧 `out_pbrt` 结果 | 非当前主表 | 多数只覆盖 4-26 张或旧 split，不建议和 seed123 主表直接比较 |
| TurboFill / IMFine / InpDiffusion 等 | 未跑 | 无 | 不是当前最直接可复现实验；部分没有公开代码或任务不匹配 |

## 3. 过曝 Raw / 真实 Raw9 无 GT 结果

这部分没有真实 GT，所以不要解读成“谁准确率更高”。主要看：

```text
hole_ratio: 需要修复的区域占比
mean_abs_model_anchor_hole: 模型在 hole 内相对 anchor 改了多少
mean_abs_model_raw_valid: 模型在 valid 区域相对 raw 改了多少
visualizations: 主观判断是否合理
```

| 方法/实验 | 数据 | 样本数 | 关键无参考指标 | 可视化位置 | 备注 |
|---|---|---:|---|---|---|
| DepthCAD raw overexposed | `raw/*.npy` 中 9ch raw，饱和像素置低 confidence | 5 | saturation ratio 约 0.86%-1.90%；无 GT MAE | `output/raw_overexposed_depthcad/visualizations`；准备图：`output/raw_overexposed_depthcad/visualizations_prepare` | IQ-domain baseline，已实际出图 |
| Overexposure satclip flow probe | 过曝/真实 raw9 probe | 4 | hole_ratio mean 0.1233；cleaned_added mean 0.0542；model-anchor hole diff mean 0.0567；valid diff mean 0.00558 | `output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20_probe/visualizations` | 无 GT，只说明模型改动幅度 |
| Overexposure guarded probe | 同上，加 hybrid/guard | 4 | hybrid_model_hole_ratio mean 0.2524；hybrid-anchor hole diff mean 0.00247 | `output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20_probe_guarded/visualizations` | guard 很保守，主要接近 anchor |
| Overexposure safe probe | 同上，更宽松 safe hybrid | 4 | hybrid_model_hole_ratio mean 0.4676；hybrid-anchor hole diff mean 0.0495 | `output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20_probe_safe/visualizations` | 比 guarded 更愿意用模型输出 |
| Overexposure plane/split probe | 同上，plane/split 变体 | 4 | plane-anchor hole diff mean 0.2422；split-anchor hole diff mean 0.1949 | `output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20_probe_plane/visualizations` | plane/split 改动大，不一定更准 |
| Real raw9 cleaned-plane recommended | 真实 raw9，非 far_pic | 41 | hole_ratio mean 0.0730；model-anchor hole diff mean 0.0528；valid diff mean 0.00325 | `output/real_raw9_flow_infer_cleaned_plane_recommended/visualizations` | 推荐真实 raw9 推理版本之一 |
| Real raw9 cleaned-plane strong | 真实 raw9，非 far_pic | 41 | hole_ratio mean 0.0985；model-anchor hole diff mean 0.0373；valid diff mean 0.00250 | `output/real_raw9_flow_infer_cleaned_plane_strong_after_synth/visualizations` | mask 更强，hole 更大，但模型改动更小 |

当前判断：

```text
过曝/真实 raw9 上还不能说哪个方法“准确率最好”，因为没有 GT。
能说的是：DepthCAD overexposed 已有图；flow satclip/guarded/safe 已有无参考诊断和图。
要得出准确度，需要用 masked self-test 或采集 GT/伪 GT。
```

## 4. 真实 Raw9 Masked Self-Test

这部分在真实 valid depth 上人工挖洞，因此有伪 GT。它比真实 hole 的无参考诊断更能反映真实泛化。

单位：米。越低越好。

| 方法/实验 | mask 类型 | Anchor Mask MAE | Model/Hole-only Mask MAE | Mask improvement vs anchor | 可视化位置 |
|---|---|---:|---:|---:|---|
| `after_synth_realhole_e20_lr5e6` | real-hole shapes | 0.047578 | **0.031510** | **33.8%** | `output/real_raw9_masked_self_test_after_synth_realhole_e20_lr5e6/visualizations` |
| `iq6_finetuned_e100_m8_best` | real-hole shapes | 0.051713 | 0.039925 | 22.8% | `output/real_raw9_masked_self_test_realholes_ratio10_thr1m_iq6_finetuned_e100_m8_best/visualizations` |

注意：

```text
full model 的 global MAE 可能变差，是因为它会动 unmasked valid 区域。
真实部署更应该看 hole_only/masked MAE 和 outside preservation。
```

## 5. 过曝数据上 ProPainter / RAD 的状态

| 方法 | 过曝 raw/real raw9 是否已有结果 | 指标 | 可视化 | 下一步 |
|---|---|---|---|---|
| ProPainter | 当前未找到过曝 raw9 同口径输出 | 无 | 无 | 可以复用 PBRT exporter/decoder 思路，把过曝 raw depth + saturation/repair mask 导出为 PNG/mask 后跑 |
| RAD official | 当前未找到过曝 raw9 输出 | 无 | 无 | 需要先解决 official checkpoint；本地适配脚本可准备输入，但缺可用官方 checkpoint |
| SD inpaint | 只有旧 PBRT/Kinect 诊断，不是过曝 raw9 主表 | 不纳入 | `kinect_evaluation/sd_diagnosis_n20_plane_r12` | 可作为历史失败参考，不建议优先 |
| Marigold / HYPIR | 只有旧 `out_pbrt` 小样本结果 | 非过曝 | `out_pbrt/*` | 不是 hole completion 主线；可放 survey，不建议主线投入 |

## 6. 当前最清楚的结论

```text
1. PBRT 有监督：我们的 final transformer endpoint flow 是当前最强。
2. ProPainter 在 PBRT 上不是更好，只比 OpenCV NS 强。
3. 过曝 raw9 上目前没有 GT，不能做“准确率”结论。
4. 真实泛化最可靠的当前证据来自 real masked self-test：
   after_synth_realhole_e20_lr5e6 在 mask MAE 上优于 anchor 33.8%。
5. 过曝/真实 raw9 下一步需要补 ProPainter/RAD 同口径输出，或者优先建立过曝 masked self-test。
```

## 7. 建议下一步表格补齐顺序

优先补这三件事：

```text
1. 在过曝 raw9 上构建 masked self-test：
   使用 saturation/real-hole-like mask 人工挖真实 valid depth，得到可量化 MAE。

2. 跑 ProPainter on overexposed raw9：
   depth-as-gray + saturation/repair mask -> decode -> hole-only merge -> 同一套无参考/伪 GT 指标。

3. RAD 暂缓：
   当前本地缺 official checkpoint；先把 ProPainter 和我们模型在过曝 masked self-test 上对齐。
```
