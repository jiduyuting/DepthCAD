# 新拍 Depth/IQ 数据检查报告

检查对象：`data/depth/depth_*.npy` 与 `data/iq/iq_*.npy`

结论：原始数据配对是完整的，但不能原样直接送入当前推理脚本。已经生成可直接测试的预处理版本，并完成 depth-only 与 raw9+depth smoke test。

## 关键结论

- 样本数：19 对，`0000` 到 `0018`，没有缺失配对。
- 原始 depth：`(424, 512)`，`float32`，无 NaN/Inf。
- 原始 IQ：`(424, 512, 9)`，`float32`，无 NaN/Inf。
- depth 单位是毫米，数值主要在 `500` 到 `4500`，当前推理脚本按米处理，所以必须除以 `1000`。
- IQ 原始布局是 HWC，即 `(H,W,9)`；`scripts/infer_real_raw9_flow.py` 要求 `(9,H,W)`，所以必须转置。
- raw9+depth 脚本按同名 stem 配对；原始文件名是 `iq_0000.npy` 与 `depth_0000.npy`，原样会配不到，需要统一成 `0000.npy`。
- `data/__MACOSX` 是压缩包带来的 macOS 元数据，测试时应忽略。

## 已生成目录

| 用途 | 路径 |
|---|---|
| 所有样本 depth，单位米 | `data/prepared_new_capture/all/depth_m/` |
| 所有样本 IQ/raw9，CHW 布局 | `data/prepared_new_capture/all/raw9_chw/` |
| 首轮推荐测试 depth | `data/prepared_new_capture/good/depth_m/` |
| 首轮推荐测试 raw9 | `data/prepared_new_capture/good/raw9_chw/` |
| 每帧 QA 预览 | `data/prepared_new_capture/qa/*_preview.png` |
| 统计 JSON | `data/prepared_new_capture/summary.json` |
| 统计 CSV | `data/prepared_new_capture/summary.csv` |

## 每帧数据质量

`valid_ratio` 是 depth 中大于 0 的比例。`sat_pixel_ratio_65535` 是任一 IQ 通道达到 65535 的像素比例。

| 样本 | 建议 | valid_ratio | depth 中位数(m) | depth p99(m) | IQ max | sat_pixel_ratio_65535 |
|---|---:|---:|---:|---:|---:|---:|
| 0000 | good | 66.34% | 1.879 | 4.304 | 2860.42 | 0.0000% |
| 0001 | good | 66.35% | 1.881 | 4.311 | 2811.09 | 0.0000% |
| 0002 | good | 57.76% | 1.745 | 4.380 | 3828.69 | 0.0000% |
| 0003 | good | 57.80% | 1.746 | 4.381 | 3648.27 | 0.0000% |
| 0004 | good | 59.59% | 1.861 | 4.367 | 1965.33 | 0.0000% |
| 0005 | good | 59.61% | 1.860 | 4.364 | 1861.94 | 0.0000% |
| 0006 | good | 58.32% | 1.933 | 4.348 | 4939.55 | 0.0000% |
| 0007 | good | 58.35% | 1.934 | 4.351 | 4987.36 | 0.0000% |
| 0008 | good | 56.57% | 2.283 | 4.275 | 3238.63 | 0.0000% |
| 0009 | good | 56.54% | 2.284 | 4.274 | 3171.57 | 0.0000% |
| 0010 | good | 58.70% | 2.169 | 4.378 | 65535.00 | 0.0018% |
| 0011 | good | 58.83% | 2.169 | 4.379 | 65535.00 | 0.0018% |
| 0012 | bad_depth | 0.49% | 0.522 | 2.623 | 65535.00 | 0.0921% |
| 0013 | bad_depth | 0.005% | 1.211 | 2.859 | 65535.00 | 1.8292% |
| 0014 | good | 57.77% | 1.037 | 1.483 | 1054.91 | 0.0000% |
| 0015 | bad_depth | 0.065% | 2.273 | 4.443 | 1096.18 | 0.0000% |
| 0016 | bad_depth | 0.064% | 2.266 | 4.461 | 1288.34 | 0.0000% |
| 0017 | good | 17.40% | 1.601 | 3.490 | 1347.33 | 0.0000% |
| 0018 | good | 38.79% | 1.208 | 2.771 | 1454.34 | 0.0000% |

首轮建议使用 `good` 子集：`0000-0011, 0014, 0017, 0018`。`0012, 0013, 0015, 0016` 的 depth 几乎全空，最多作为极端失败/过曝压力样本，不适合作为正常定性或定量评测样本。

## 已跑通的测试

### Depth-only good 子集

```bash
/home/lab507/anaconda3/envs/depthcad_zimage/bin/python scripts/infer_real_depth_flow.py \
  --input_dir data/prepared_new_capture/good/depth_m \
  --checkpoint output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt \
  --output_dir output/new_capture_depth_only_flow_good \
  --sampling_mode endpoint \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.1 \
  --valid_max_depth 4.5
```

输出：

- 可视化：`output/new_capture_depth_only_flow_good/visualizations/*.png`
- 结果深度：`output/new_capture_depth_only_flow_good/restored/*_restored.npy`
- summary：`output/new_capture_depth_only_flow_good/summary.json`

结果摘要：

- 样本数：15
- 平均 hole ratio：44.75%
- `|model-anchor|_hole` 平均：0.3240 m
- `|model-raw|_valid` 平均：0.0072 m

### Raw9+depth good 子集

```bash
/home/lab507/anaconda3/envs/depthcad_zimage/bin/python scripts/infer_real_raw9_flow.py \
  --raw_dir data/prepared_new_capture/good/raw9_chw \
  --depth_dir data/prepared_new_capture/good/depth_m \
  --checkpoint output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20/best.pt \
  --output_dir output/new_capture_raw9_flow_satclip_good \
  --sampling_mode endpoint \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.1 \
  --valid_max_depth 4.5 \
  --amplitude_mode iq6 \
  --hole_amplitude_mode keep_all
```

输出：

- 可视化：`output/new_capture_raw9_flow_satclip_good/visualizations/*.png`
- 结果深度：`output/new_capture_raw9_flow_satclip_good/restored/*_restored.npy`
- summary：`output/new_capture_raw9_flow_satclip_good/summary.json`

结果摘要：

- 样本数：15
- 平均 hole ratio：44.75%
- `|model-anchor|_hole` 平均：0.0469 m
- `|model-raw|_valid` 平均：0.00194 m

### Raw9+depth 全量样本

```bash
/home/lab507/anaconda3/envs/depthcad_zimage/bin/python scripts/infer_real_raw9_flow.py \
  --raw_dir data/prepared_new_capture/all/raw9_chw \
  --depth_dir data/prepared_new_capture/all/depth_m \
  --checkpoint output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20/best.pt \
  --output_dir output/new_capture_raw9_flow_satclip_all \
  --sampling_mode endpoint \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.1 \
  --valid_max_depth 4.5 \
  --amplitude_mode iq6 \
  --hole_amplitude_mode keep_all
```

输出：

- 可视化：`output/new_capture_raw9_flow_satclip_all/visualizations/*.png`
- 结果深度：`output/new_capture_raw9_flow_satclip_all/restored/*_restored.npy`
- summary：`output/new_capture_raw9_flow_satclip_all/summary.json`

全量样本也能跑通，但 `0012/0013/0015/0016` 的 hole ratio 接近 1，模型输出主要是外推，不建议直接拿来判断方法优劣。

## 判断

这批数据的文件组织和数值内容基本正确，但原始目录不是当前脚本的直接输入格式。经过单位转换、布局转换、同名配对后可以直接测试。当前最稳妥的测试入口是：

- 正常首轮定性：`data/prepared_new_capture/good/*`
- 极端/过曝压力观察：`data/prepared_new_capture/all/*`
- 优先看 raw9+depth 输出：`output/new_capture_raw9_flow_satclip_good/visualizations/`
- 对照坏帧：`data/prepared_new_capture/qa/0013_preview.png` 与 `output/new_capture_raw9_flow_satclip_all/visualizations/0013.png`
