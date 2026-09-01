# 新拍数据黑洞填补方法对比

数据：`data/prepared_new_capture/all/`

目标：测试已有方法能否填补真实拍摄 depth 中的黑色空洞，并做无 GT 的定性比较。

## 测试方法

| 方法 | 输入 | 输出目录 | 说明 |
|---|---|---|---|
| NS anchor | depth | `output/new_capture_raw9_flow_satclip_all/anchor/` | OpenCV/NS 类传统 inpaint anchor，作为基线 |
| depth-only flow | depth | `output/new_capture_depth_only_flow_all/` | 只用 depth 的 flow 修复模型 |
| raw9 satclip | depth + IQ | `output/new_capture_raw9_flow_satclip_all/` | 过曝/饱和增强 finetune 版本 |
| raw9 realholes | depth + IQ | `output/new_capture_raw9_flow_realholes_all/` | realholes finetune 版本 |
| after-synth-realhole split | depth + IQ | `output/new_capture_raw9_flow_after_synth_realhole_all/` | 合成 realhole 后 finetune，带 cleaned/split-added mask |
| propagation refine split | depth + IQ | `output/new_capture_raw9_propagation_refine_all/` | 传感器传播/细化模型，带 cleaned/split-added mask |
| ProPainter | depth-as-gray PNG + mask | `output/new_capture_external_inpaint/propainter_run/` | 外部视频/图像 inpainting baseline，不使用 raw9 |
| DepthCAD depth-gray | depth-as-gray + confidence | `output/new_capture_depthcad_depth_hole_all_s5/` | 诊断性 DepthCAD/ControlNet depth-hole baseline，不是原始 IQ-domain DepthCAD |

统一对比图：

- `output/new_capture_method_comparison/figures/*.png`
- 含 ProPainter 版本：`output/new_capture_method_comparison_with_propainter/figures/*.png`
- 含 ProPainter 和 DepthCAD depth-gray 版本：`output/new_capture_method_comparison_with_external_depthcad/figures/*.png`
- 统计表：`output/new_capture_method_comparison/per_sample_metrics.csv`
- 汇总：`output/new_capture_method_comparison/summary.json`

## 汇总指标

下面只统计 good 样本，也就是原始 depth 有效比例不低于 10% 的 15 张。由于没有 GT，这不是准确度，只是检查黑洞是否被填上、以及输出对原图有效区域的扰动。

| 方法 | threshold 空洞填补率 | cleaned 空洞填补率 | 对原始有效区域平均改变量 | threshold 空洞填补中位深度 |
|---|---:|---:|---:|---:|
| NS anchor | 100.00% | 100.00% | 0.0000 m | 1.705 m |
| depth-only flow | 100.00% | 100.00% | 0.0000 m | 2.036 m |
| raw9 satclip | 100.00% | 100.00% | 0.0000 m | 1.719 m |
| raw9 realholes | 100.00% | 100.00% | 0.0000 m | 1.721 m |
| after-synth split | 100.00% | 100.00% | 0.0116 m | 1.868 m |
| propagation split | 100.00% | 99.99% | 0.0114 m | 1.820 m |
| ProPainter | 100.00% | 100.00% | 0.0000 m | 1.990 m |
| DepthCAD depth-gray | 100.00% | 100.00% | 0.0000 m | 1.872 m |

填补率都是 100% 左右，说明“黑色空洞能否被填上”不是主要问题。主要差异在输出结构是否可信。

## 定性结论

### 最稳的结果

`raw9 satclip` 和 `raw9 realholes` 最稳。它们基本都能把黑洞补上，同时保留物体/墙面/边界结构，不会像 depth-only 那样在大洞里造出明显块状伪结构。`NS anchor` 也很稳，很多帧上和 raw9 两个版本很接近。

重点看：

- `output/new_capture_method_comparison/figures/0000.png`
- `output/new_capture_method_comparison/figures/0014.png`
- `output/new_capture_method_comparison/figures/0017.png`

这些图里，`raw9 satclip`、`raw9 realholes` 基本保留了场景结构；`depth-only flow` 在一些区域会出现较明显的块状或条带状伪结构。

含 ProPainter 的新版对比图：

- `output/new_capture_method_comparison_with_propainter/figures/0000.png`
- `output/new_capture_method_comparison_with_propainter/figures/0014.png`
- `output/new_capture_method_comparison_with_propainter/figures/0017.png`

含 ProPainter 和 DepthCAD depth-gray 的新版对比图：

- `output/new_capture_method_comparison_with_external_depthcad/figures/0000.png`
- `output/new_capture_method_comparison_with_external_depthcad/figures/0014.png`
- `output/new_capture_method_comparison_with_external_depthcad/figures/0017.png`

### depth-only flow 的问题

depth-only 确实能补洞，但在这批真实数据上有较明显的幻觉倾向。比如 `0000` 的下方区域会生成突兀的亮块，`0014` 会把黑洞区域补成形状不稳定的结构。

结论：它可以作为对照，但不建议作为当前真实黑洞的主结果。

### after-synth 和 propagation 的问题

这两个版本开启了 `amp_speckle_cleaned` 和 `split_added_fill`，会把额外异常区域并入修复 mask。它们可以处理更多黑色异常区域，但在这批数据上偏激进：

- `after-synth split` 容易把真实结构抹成大平面。
- `propagation split` 在一些帧会明显改变墙面/物体边界，甚至产生过强的深度外推。

在 `0014`、`0017` 上这个问题比较明显。它们更像压力测试或后续调参对象，不适合作为当前推荐输出。

### ProPainter 的问题

ProPainter 已经在这批新数据上跑通。运行过程中最后写 mp4 报了 `imageio` 参数错误，但 PNG 帧已经完整保存，并已 decode 回米制 depth。

输出：

- PNG 帧：`output/new_capture_external_inpaint/propainter_run/frames/frames/`
- decoded depth：`output/new_capture_external_inpaint/propainter_run/restored_by_stem/`
- 含 ProPainter 对比图：`output/new_capture_method_comparison_with_propainter/figures/`

定性看，ProPainter 可以把黑洞补上，但它是图像域 inpainting，容易产生平滑、条纹和结构抹除。在 `0000` 下方区域能看到条纹伪影，在 `0014` 里会把原本应有的几何结构抹得比较平。它适合作为外部 baseline，但不如 raw9 satclip / raw9 realholes 保结构。

## 外部方法准备状态

### DepthCAD

DepthCAD prepare-only 已完成：

- `output/new_capture_depthcad_prepare/`

这一步把新 raw9 转成 DepthCAD 需要的 IQ/归一化 IQ/置信度/饱和 mask 预览。后续检查发现，这个 prepare 里的 `raw IQ -> depth` 不可信：它调用的是 PBRT/DepthCAD 的固定 `DepthEstimator` 和 `load_raw_pbrt` 通道解释，而这批新数据更像 Kinect 424x512 triplet raw9。PBRT 解算会把通道顺序/相位模型用错，所以会出现锯齿、错位和 0-10m 异常深度。

诊断输出：

- 错误 prepare 图：`output/new_capture_depthcad_prepare/visualizations_prepare/`
- Kinect raw9 解算候选：`output/new_capture_kinect_raw9_depth_candidate/`
- Kinect 候选预览：`output/new_capture_kinect_raw9_depth_candidate_preview/`
- 三方对比诊断：`output/new_capture_depthcad_prepare_diagnosis/`

结论：`output/new_capture_depthcad_prepare/prepared/raw_depth/` 不能作为有效 DepthCAD baseline 的 raw depth。完整 IQ-domain DepthCAD 推理暂时不应基于这批 prepare 结果继续跑。若要公平测原始 DepthCAD，需要先重写适配：使用正确的 Kinect raw9 解算或直接使用相机 depth 作为 corrupted depth，并明确 DepthCAD 要修的是 depth=0 hole 还是 raw/IQ saturation。

为了和 ProPainter 一样提供一个 depth-hole 外部 baseline，额外实现并跑通了一个诊断性版本：

- 脚本：`scripts/run_depthcad_depth_hole_baseline.py`
- 输出：`output/new_capture_depthcad_depth_hole_all_s5/`

这个版本把 sensor depth 当成灰度图，confidence/hole mask 当成第二个 conditioning channel，送入 DepthCAD ControlNet，再把输出灰度 decode 回 depth。它可以补洞，但它不是原始 IQ-domain DepthCAD 方法，只能叫 `DepthCAD depth-gray` 或 `DepthCAD depth-hole diagnostic baseline`。定性看，它仍然偏图像先验和平滑，在 `0014` 这类结构样本上不如 `raw9 satclip` / `raw9 realholes` 保结构。

### RAD

RAD 输入已准备：

- `output/new_capture_external_inpaint/rad_val/`

RAD 脚本存在：`/data/pre_student/GJ/RAD/examples/unconditional_image_generation/inpaint.py`。

默认 FFHQ checkpoint 缺失；本机存在的是 `lsun-bedrooms` checkpoint：

- `/data/pre_student/GJ/RAD/ddpm-model-pcuenq/lsun-bedrooms-256/Local-lsun-bedrooms-2000-1000_lora_rank_16/checkpoint-300000`

当前没有 CUDA，RAD diffusion 全量 CPU 推理预计很慢，所以本轮没有跑出 RAD 结果。后续可在 GPU 环境上用已准备好的 `rad_val` 目录补跑。

后续尝试为 RAD 创建了新的唯一输出目录，避免混用旧结果：

- `/data/pre_student/GJ/RAD/ddpm-model-pcuenq/lsun-bedrooms-256/new_capture_20260706_Local-lsun-bedrooms-2000-1000_lora_rank_16/`

该目录通过 symlink 复用了现有 `lsun-bedrooms` checkpoint，并成功进入推理，但在第一步 CUDA OOM：

- GPU 总显存约 23.68GB
- 运行时只剩约 185MB free
- 另一个进程占用约 19.81GB

因此该新目录目前没有有效 RAD 输出。之前误 decode 的 `rad_lsun_bedroom_s5_smoke` 是旧 PNG 文件，已标记无效：

- `output/new_capture_external_inpaint/outputs/rad_lsun_bedroom_s5_smoke/INVALID_DO_NOT_USE.txt`

### 极端坏帧

`0012/0013/0015/0016` 的原始 depth 几乎全空。所有方法都可以给出图，但本质是大范围外推，不能当正常效果。尤其 `0013`：

- 原始 depth 有效比例约 `0.005%`
- IQ 有明显 65535 饱和
- 输出只是根据极少数有效点和模型先验生成，不适合判断方法优劣

参考图：

- `output/new_capture_method_comparison/figures/0013.png`

## 当前建议

1. 当前这批真实数据上，首选看 `raw9 satclip` 和 `raw9 realholes`。
2. 如果目标是“保守、少出伪结构”，`raw9 satclip` 更合适。
3. 如果目标是“尽量填满黑洞但仍保结构”，`raw9 realholes` 也值得保留。
4. ProPainter 可以作为外部图像域 baseline，但新数据上不如 raw9 系列保结构。
5. `depth-only flow` 只作为对照，不建议作为主方法。
6. `after-synth` 和 `propagation refine` 需要重新调 mask 策略，否则在真实场景上过度平滑/过度外推。
7. DepthCAD/RAD 可以继续补，但必须明确它们是外部/IQ-domain baseline，不要和 raw9 flow 主方法混成同一类模型。

下一步应该围绕 raw9 satclip/realholes 做针对性调参，而不是换掉模型路线。重点调：

- hole mask 不要过度扩张；
- 对大面积真实黑洞使用 raw9 特征约束；
- 对几乎全空的帧单独标记为 invalid/stress，不并入正常评测。
