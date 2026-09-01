# picture 随机抽样方法对比报告

日期：2026-07-08

数据根目录：

```text
/data/pre_student/GJ/DepthCAD/picture
```

本轮任务：忽略 `__MACOSX`，对 `picture/` 下每个有效 depth/IQ 文件夹组随机抽取 3 张样本，运行 core 方法并生成统一可视化。

## 1. 有效数据组

共发现 10 个有效 depth/IQ 组：

| Group | Depth dir | IQ dir | Pairs | Valid ratio mean/min/max | Median depth mean |
|---|---|---|---:|---:|---:|
| `pic_class2` | `picture/pic/depth_class2` | `picture/pic/iq_class2` | 30 | `0.364 / 0.326 / 0.460` | `790 mm` |
| `pic2_l1` | `picture/pic2/depth_l1` | `picture/pic2/iq_l1` | 30 | `0.293 / 0.274 / 0.342` | `2716 mm` |
| `pic2_l2` | `picture/pic2/depth_l2` | `picture/pic2/iq_l2` | 30 | `0.314 / 0.223 / 0.502` | `2622 mm` |
| `pic2_l3` | `picture/pic2/depth_l3` | `picture/pic2/iq_l3` | 30 | `0.827 / 0.493 / 0.937` | `2589 mm` |
| `pic3_s1` | `picture/pic3/depth_s1` | `picture/pic3/iq_s1` | 30 | `0.205 / 0.196 / 0.212` | `3056 mm` |
| `pic3_s2` | `picture/pic3/depth_s2` | `picture/pic3/iq_s2` | 30 | `0.127 / 0.094 / 0.149` | `1973 mm` |
| `pic3_s3` | `picture/pic3/depth_s3` | `picture/pic3/iq_s3` | 30 | `0.161 / 0.153 / 0.178` | `1733 mm` |
| `pic4_z1` | `picture/pic4/depth_z1` | `picture/pic4/iq_z1` | 30 | `0.366 / 0.319 / 0.401` | `3494 mm` |
| `pic4_z2` | `picture/pic4/depth_z2` | `picture/pic4/iq_z2` | 30 | `0.251 / 0.224 / 0.284` | `3804 mm` |
| `pic4_z3` | `picture/pic4/depth_z3` | `picture/pic4/iq_z3` | 30 | `0.291 / 0.259 / 0.334` | `1726 mm` |

数据质量初判：

1. `pic2_l3` 最好，有效比例最高，适合做主图候选。
2. `pic_class2`、`pic4_z1` 也比较可用，有效比例约 36%。
3. `pic3_s1/s2/s3` 有效比例偏低，尤其 `pic3_s2` 平均只有 12.7%，更像 hard/stress set。
4. `pic4_z2` 距离较远，valid ratio 约 25%，也偏难。

## 2. 新增批处理脚本

新增脚本：

```text
scripts/run_picture_random_suite.py
```

它会：

1. 自动发现 `picture/` 下所有 depth/IQ 成对组；
2. 每组按 seed 随机抽样；
3. 调用 `scripts/run_real_capture_method_suite.py`；
4. 每组生成独立输出；
5. 汇总 batch summary。

本轮运行命令：

```bash
/home/lab507/anaconda3/envs/control/bin/python scripts/run_picture_random_suite.py \
  --picture_root picture \
  --output_root output/picture_random_suite_n3_seed20260708_core \
  --samples_per_group 3 \
  --seed 20260708 \
  --methods core \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.5 \
  --valid_max_depth 4.5 \
  --skip_existing \
  --continue_on_error
```

`core` 方法包含：

```text
depth_only
raw9_satclip
raw9_realholes
after_synth
propagation
propainter
```

DepthCAD depth-gray 本轮没有批量跑，因为当前环境下 DepthCAD 只能 CPU，速度太慢。需要时可以对少量候选样本单独跑。

## 3. 随机样本

随机 seed：

```text
20260708
```

每组随机选中样本：

| Group | Selected samples |
|---|---|
| `pic_class2` | `0001 0017 0025` |
| `pic2_l1` | `0001 0011 0013` |
| `pic2_l2` | `0000 0008 0018` |
| `pic2_l3` | `0006 0020 0022` |
| `pic3_s1` | `0007 0016 0027` |
| `pic3_s2` | `0013 0018 0024` |
| `pic3_s3` | `0006 0014 0018` |
| `pic4_z1` | `0004 0010 0015` |
| `pic4_z2` | `0002 0003 0006` |
| `pic4_z3` | `0012 0015 0019` |

输出根目录：

```text
output/picture_random_suite_n3_seed20260708_core/
```

总览图：

```text
output/picture_random_suite_n3_seed20260708_core/overview_first_figures.png
```

批处理汇总：

```text
output/picture_random_suite_n3_seed20260708_core/batch_summary.json
output/picture_random_suite_n3_seed20260708_core/batch_summary.csv
output/picture_random_suite_n3_seed20260708_core/aggregate_method_metrics.csv
```

每组详细日志：

```text
output/picture_random_suite_n3_seed20260708_core/<group>/run.log
```

每组统一对比图：

```text
output/picture_random_suite_n3_seed20260708_core/<group>/comparison/figures/
```

## 4. 无 GT 指标汇总

下面的 `vs NS` 是各方法在 threshold hole 内相对 NS anchor 的平均绝对差异。因为没有 GT，它不是精度，只表示方法相对保守或激进。

| Group | raw9 realholes | raw9 satclip | depth-only | after-synth | propagation | ProPainter |
|---|---:|---:|---:|---:|---:|---:|
| `pic_class2` | `0.027` | `0.044` | `0.282` | `0.292` | `0.511` | `0.419` |
| `pic2_l1` | `0.020` | `0.033` | `0.127` | `0.470` | `0.555` | `0.887` |
| `pic2_l2` | `0.027` | `0.042` | `0.126` | `0.645` | `0.734` | `0.703` |
| `pic2_l3` | `0.159` | `0.175` | `0.391` | `0.760` | `0.893` | `0.484` |
| `pic3_s1` | `0.018` | `0.033` | `0.133` | `0.287` | `0.493` | `0.348` |
| `pic3_s2` | `0.018` | `0.029` | `0.147` | `0.183` | `0.310` | `0.506` |
| `pic3_s3` | `0.019` | `0.030` | `0.121` | `0.500` | `0.548` | `0.405` |
| `pic4_z1` | `0.026` | `0.051` | `0.147` | `0.497` | `0.661` | `0.707` |
| `pic4_z2` | `0.026` | `0.038` | `0.039` | `0.777` | `0.793` | `0.777` |
| `pic4_z3` | `0.026` | `0.040` | `0.171` | `0.462` | `0.576` | `0.414` |

## 5. 定性观察

### raw9 realholes / raw9 satclip

这两个方法最稳定。大多数组里，hole 内相对 NS anchor 的改动只有 `2cm - 5cm`。

例外是 `pic2_l3`，raw9 realholes/satclip 改动约 `16cm - 18cm`。这组有效比例很高，场景结构和深度分布更完整，模型更愿意修正 anchor。

推荐主看：

```text
raw9 realholes
raw9 satclip
NS anchor
```

### depth-only

depth-only 在不少组里仍然明显比 raw9 激进，尤其 `pic_class2` 和 `pic2_l3`。可视化中会出现条带、大片平滑或局部 hallucination。

不过在 `pic4_z2` 这组，depth-only 相对 NS 只有 `0.039m`，说明这组里 depth-only 没有特别发散。

### after-synth / propagation

这两个在很多组里都明显激进：

```text
after-synth: 0.18m - 0.78m
propagation: 0.31m - 0.89m
```

其中 `pic2_l2`、`pic2_l3`、`pic4_z2` 特别明显。当前不建议作为默认主结果，只适合作为 aggressive mask/refine 对照。

### ProPainter

ProPainter 都跑通了，但在多数组里相对 NS 改动很大，并且可视化中有明显图像域纹理、网格或块状伪影。

它仍然适合作为外部 image/video inpainting baseline，但不建议作为主方法。

## 6. 数据集推荐用途

推荐分组：

| 用途 | 推荐组 |
|---|---|
| 主定性候选 | `pic2_l3`, `pic4_z1`, `pic_class2` |
| 中等难度对比 | `pic2_l1`, `pic2_l2`, `pic4_z3` |
| hard/stress set | `pic3_s1`, `pic3_s2`, `pic3_s3`, `pic4_z2` |

如果要为论文挑图，优先从这些目录看：

```text
output/picture_random_suite_n3_seed20260708_core/pic2_l3/comparison/figures/
output/picture_random_suite_n3_seed20260708_core/pic4_z1/comparison/figures/
output/picture_random_suite_n3_seed20260708_core/pic_class2/comparison/figures/
```

如果要看 failure/stress：

```text
output/picture_random_suite_n3_seed20260708_core/pic3_s2/comparison/figures/
output/picture_random_suite_n3_seed20260708_core/pic4_z2/comparison/figures/
```

## 7. 下一步建议

1. 对 `pic2_l3`、`pic4_z1`、`pic_class2` 全量跑 `flows`，先不跑外部方法：

```bash
/home/lab507/anaconda3/envs/control/bin/python scripts/run_real_capture_method_suite.py \
  --data_root picture/pic2 \
  --depth_dir picture/pic2/depth_l3 \
  --iq_dir picture/pic2/iq_l3 \
  --output_root output/picture_pic2_l3_flows_all \
  --sample_mode all \
  --methods flows \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.5 \
  --valid_max_depth 4.5 \
  --skip_existing
```

2. 从全量 flow 结果里挑出最有代表性的 6-8 张，再补跑 `core` 或 `all`。

3. DepthCAD depth-gray 只建议对最终候选图单独跑，不建议批量跑全部组。
