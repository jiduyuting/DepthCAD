# PBRT Real 新数据选样与测试记录

数据源：`/data/pre_student/hcy/datasets/pbrt/Real`

## 数据检查

- `depth/noise/depth_png/list` 中 503 个样本配对完整。
- `noise/*.npy` 是 `(9, 424, 512)` raw9/IR 条件输入，可直接给 `scripts/flow/infer_real_raw9_flow.py`。
- `depth/*.npy` 单位混合：`ceiling/desk/plant` 已是米，`lab307_*` 和 `library_*` 是毫米。测试输入已统一转换成米。
- 全部 503 张形状和基础质量检查通过；本轮没有直接全量跑，先选 15 张代表样本做 smoke test。

## 本轮选中样本

输入已准备到：

- depth：`output/pbrt_real_new_selection/selected/depth_m/`
- raw9：`output/pbrt_real_new_selection/selected/raw9_chw/`
- 样本列表：`output/pbrt_real_new_selection/selected_samples.txt`
- QA 总览：`output/pbrt_real_new_selection/selected_qa_montage.png`

| 样本 | 用途 | valid ratio | hole ratio | depth median(m) | sat pixel ratio |
|---|---|---:|---:|---:|---:|
| `ceiling` | named/meter-unit 场景 | 92.1% | 7.8% | 2.591 | 0.011% |
| `desk` | named/meter-unit 场景 | 91.7% | 8.2% | 1.942 | 0.000% |
| `plant` | named/meter-unit 场景 | 95.5% | 4.4% | 1.520 | 0.000% |
| `lab307_1` | 低空洞、轻饱和 sanity | 99.1% | 0.9% | 1.363 | 0.023% |
| `lab307_48` | 人物/中等空洞 | 93.2% | 6.0% | 1.797 | 0.000% |
| `lab307_87` | 人物/较远深度 | 90.7% | 8.0% | 2.497 | 0.000% |
| `lab307_135` | 人物/中等空洞 | 91.9% | 7.3% | 2.241 | 0.000% |
| `lab307_165` | 脏边界/压力样本 | 87.8% | 10.4% | 2.821 | 0.000% |
| `lab307_250` | 后段人物样本 | 92.0% | 7.3% | 2.195 | 0.000% |
| `library_71` | 桌椅/低空洞 | 96.9% | 3.0% | 1.798 | 0.000% |
| `library_97` | 台灯/低空洞 | 98.1% | 1.9% | 1.387 | 0.000% |
| `library_144` | 圆桌/中等空洞 | 93.6% | 6.1% | 1.747 | 0.000% |
| `library_115` | 大空洞压力样本 | 83.3% | 15.6% | 1.869 | 0.000% |
| `library_173` | 圆桌/中高空洞 | 90.9% | 8.6% | 1.689 | 0.000% |
| `library_244` | 桌椅/轻饱和 | 95.6% | 4.0% | 1.504 | 0.004% |

## 已运行测试

Raw9+depth satclip：

```bash
/home/lab507/anaconda3/envs/depthcad_zimage/bin/python scripts/flow/infer_real_raw9_flow.py \
  --raw_dir output/pbrt_real_new_selection/selected/raw9_chw \
  --depth_dir output/pbrt_real_new_selection/selected/depth_m \
  --checkpoint output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20/best.pt \
  --output_dir output/pbrt_real_new_selection/raw9_flow_satclip_selected \
  --sampling_mode endpoint \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.1 \
  --valid_max_depth 9.9 \
  --amplitude_mode iq6 \
  --hole_amplitude_mode keep_all
```

Depth-only baseline：

```bash
/home/lab507/anaconda3/envs/depthcad_zimage/bin/python scripts/flow/infer_real_depth_flow.py \
  --input_dir output/pbrt_real_new_selection/selected/depth_m \
  --checkpoint output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt \
  --output_dir output/pbrt_real_new_selection/depth_only_flow_selected \
  --sampling_mode endpoint \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.1 \
  --valid_max_depth 9.9
```

## 结果摘要

| 方法 | 样本数 | mean hole ratio | mean valid ratio | mean `|model-anchor|` in hole | mean `|model-raw|` on valid |
|---|---:|---:|---:|---:|---:|
| raw9 satclip | 15 | 6.64% | 92.83% | 0.3246 m | 0.0304 m |
| depth-only flow | 15 | 6.64% | - | 1.3031 m | 0.0960 m |

raw9 satclip 在这组新样本上没有整图发散，valid 区域改动明显小于 depth-only。`lab307_165` 和 `library_115` 的 hole 内差异最大，适合作为压力样本；`lab307_1`、`library_97`、`library_71` 更适合作为正常 sanity check。

主要输出：

- raw9 summary：`output/pbrt_real_new_selection/raw9_flow_satclip_selected/summary.json`
- raw9 可视化：`output/pbrt_real_new_selection/raw9_flow_satclip_selected/visualizations/`
- depth-only summary：`output/pbrt_real_new_selection/depth_only_flow_selected/summary.json`
- 对比图：`output/pbrt_real_new_selection/comparison_figures/`
- 对比索引：`output/pbrt_real_new_selection/comparison_index.csv`

## 后续建议

如果这 15 张的可视化确认可用，下一步建议直接全量跑 503 张 raw9 satclip；但要沿用本次的单位归一化逻辑，不能直接把原始 `depth` 目录送进推理脚本。
