# picture/pic class2 数据检查与一键对比脚本

日期：2026-07-07

新数据目录：

```text
picture/pic/
  depth_class2/
  iq_class2/
```

## 1. 数据质量

共 30 组成对数据：

```text
depth_class2/depth_0000.npy ... depth_0029.npy
iq_class2/iq_0000.npy ... iq_0029.npy
```

检查结果：

| 项 | 结果 |
|---|---:|
| depth shape | `(424, 512)` |
| IQ shape | `(424, 512, 9)` |
| depth unit | mm |
| valid ratio mean | `0.3642` |
| valid ratio min | `0.3261` |
| valid ratio max | `0.4600` |
| median depth mean | about `790 mm` |
| p99 depth | about `4.45 m` |
| 65535 saturation | `0.0` |

QA 输出：

```text
output/picture_pic_class2_qa/preview_depth_valid_sat.png
output/picture_pic_class2_qa/stats.csv
```

判断：

1. 这批比上一批 `pic/iq_classroom + depth_classroom` 更好。上一批有效比例约 `23%-25%`，这批提升到 `32.6%-46.0%`。
2. 这批仍然偏难，因为 threshold hole 仍然大约 `54%-67%`。
3. 没有 65535 饱和，因此它不是典型 overexposure/saturation 测试集，更像普通大洞/弱观测 classroom 测试。
4. 深度仍有不少 `0.5m-1.0m` 的近距离有效点，所以推理时不能使用 `hole_depth_threshold=1.0`。

本批推荐阈值：

```text
hole_depth_threshold = 0.0
valid_min_depth = 0.5
valid_max_depth = 4.5
depth_scale = 1000
```

## 2. 新增一键脚本

新增脚本：

```text
scripts/run_real_capture_method_suite.py
```

它会自动完成：

1. 自动寻找 `depth*` 和 `iq*` 子目录；
2. 自动或指定选择样本；
3. 复制选中样本到标准 `input/depth` 和 `input/iq`；
4. 调用 `scripts/prepare_new_capture_data.py` 做单位转换和 raw9 CHW 转换；
5. 跑选定方法；
6. 导出 ProPainter 输入并 decode；
7. 生成统一对比图；
8. 输出无 GT 统计表。

默认推荐用法：

```bash
/home/lab507/anaconda3/envs/control/bin/python scripts/run_real_capture_method_suite.py \
  --data_root picture/pic \
  --output_root output/picture_pic_class2_suite_auto4 \
  --sample_mode auto \
  --auto_count 4 \
  --methods core \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.5 \
  --valid_max_depth 4.5 \
  --skip_existing
```

`--methods core` 包含：

```text
depth_only
raw9_satclip
raw9_realholes
after_synth
propagation
propainter
```

如果只想跑 flow，不跑 ProPainter：

```bash
--methods flows
```

如果要加上 DepthCAD depth-gray：

```bash
--methods all --allow_depthcad_cpu
```

注意：当前环境下 DepthCAD 检测不到 CUDA，`--allow_depthcad_cpu` 会很慢。建议只对少量样本用。

如果要跑全 30 张：

```bash
--sample_mode all
```

建议先全量跑：

```bash
--methods flows --sample_mode all
```

然后挑有代表性的样本再补跑 ProPainter / DepthCAD。

## 3. 本次 auto4 运行

本次自动选择了 4 张：

```text
0000
0009
0015
0029
```

输出根目录：

```text
output/picture_pic_class2_suite_auto4/
```

统一对比图：

```text
output/picture_pic_class2_suite_auto4/comparison/figures/
```

统计：

```text
output/picture_pic_class2_suite_auto4/comparison/summary.json
output/picture_pic_class2_suite_auto4/comparison/per_sample_metrics.csv
```

## 4. auto4 无 GT 指标

无 GT 指标只用于观察填洞程度、相对 NS anchor 改动和输出尺度，不代表真实 accuracy。

| Method | Threshold fill | Cleaned fill | Valid change | vs NS in threshold hole | Filled median |
|---|---:|---:|---:|---:|---:|
| NS anchor | 100.00% | 100.00% | 0.0000 m | 0.0000 m | 3.159 m |
| depth-only flow | 100.00% | 100.00% | 0.0000 m | 0.2664 m | 3.296 m |
| raw9 satclip | 100.00% | 100.00% | 0.0000 m | 0.0423 m | 3.203 m |
| raw9 realholes | 100.00% | 100.00% | 0.0000 m | 0.0261 m | 3.161 m |
| after-synth split | 100.00% | 100.00% | 0.0138 m | 0.2716 m | 3.569 m |
| propagation split | 100.00% | 99.99% | 0.0132 m | 0.4795 m | 4.434 m |
| ProPainter | 100.00% | 100.00% | 0.0000 m | 0.4495 m | 3.102 m |

## 5. 定性结论

这批数据更适合继续测试，但仍然偏难。

推荐优先看：

```text
output/picture_pic_class2_suite_auto4/methods/raw9_satclip/visualizations/
output/picture_pic_class2_suite_auto4/methods/raw9_realholes/visualizations/
output/picture_pic_class2_suite_auto4/comparison/figures/
```

当前排序：

1. `raw9 realholes`：最保守，和 NS anchor 最接近，洞内平均改动约 `2.6 cm`；
2. `raw9 satclip`：也稳定，洞内平均改动约 `4.2 cm`；
3. `NS anchor`：可靠保守 baseline；
4. `after-synth split`：更激进，会额外扩大 mask，填补深度偏远；
5. `depth-only flow`：仍有较明显大面积纹理/条带伪影；
6. `propagation split`：本批仍然过激，填补中位深度接近 `4.43m`；
7. `ProPainter`：外部图像域 baseline，可跑通，但纹理噪声和伪结构明显。

建议下一步：

1. 用新增一键脚本全量跑 `flows`：

```bash
/home/lab507/anaconda3/envs/control/bin/python scripts/run_real_capture_method_suite.py \
  --data_root picture/pic \
  --output_root output/picture_pic_class2_suite_flows_all \
  --sample_mode all \
  --methods flows \
  --hole_depth_threshold 0.0 \
  --valid_min_depth 0.5 \
  --valid_max_depth 4.5 \
  --skip_existing
```

2. 从全量结果里挑 6-8 张最有代表性的，再跑：

```bash
--methods core
```

3. DepthCAD depth-gray 只对最终挑出的少量样本补跑：

```bash
--methods all --allow_depthcad_cpu
```
