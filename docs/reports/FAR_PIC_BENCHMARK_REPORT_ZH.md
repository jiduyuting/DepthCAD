# Far Pic Depth Completion Benchmark Report

生成时间：2026-07-01

## 运行内容

脚本：

```bash
/home/lab507/anaconda3/envs/control/bin/python scripts/benchmark_far_pic_depth_completion.py
```

输入：

```text
far_pic/noise_depth_240x320_m
```

输出：

```text
output/far_pic_benchmark
```

已测试 case：

```text
zero_mask          真实 depth==0 空洞
synthetic_05       在有效 depth 上人工挖洞 5%，用于伪真值评估
synthetic_10       在有效 depth 上人工挖洞 10%，用于伪真值评估
synthetic_20       在有效 depth 上人工挖洞 20%，用于伪真值评估
bad_depth_mask_v1  zero mask + 远景/局部残差异常点，用于真实坏点修复尝试
```

已比较方法：

```text
OpenCV NS
OpenCV Telea
ours_hole_only
ours_restored
ProPainter，仅 zero_mask 使用已有结果并入
```

注意：`ours_restored` 会改动 mask 外有效 depth，不建议作为最终输出；实际比较应主要看 `ours_hole_only`。

## 核心结论

当前 depth-flow 方法在 far_pic 这批真实远景 ToF depth 上没有比传统方法更好。

在有伪真值的人工挖洞测试中，OpenCV NS 稳定优于当前 `ours_hole_only`：

```text
synthetic_05:
  OpenCV NS MAE:      0.209 m
  ours_hole_only MAE: 0.370 m

synthetic_10:
  OpenCV NS MAE:      0.265 m
  ours_hole_only MAE: 0.472 m

synthetic_20:
  OpenCV NS MAE:      0.201 m
  ours_hole_only MAE: 0.299 m
```

在真实 zero mask 上没有 GT，只能看无参考指标。ProPainter 和 NS 都不弱于当前方法：

```text
zero_mask boundary median jump:
  ProPainter:      0.179 m
  OpenCV NS:       0.200 m
  ours_hole_only:  0.273 m

zero_mask hole total variation median:
  ProPainter:      0.043 m
  OpenCV NS:       0.099 m
  ours_hole_only:  0.136 m
```

在 bad-depth mask v1 上，OpenCV NS 仍比当前方法边界更稳：

```text
bad_depth_mask_v1 boundary median jump:
  OpenCV NS:       0.063 m
  ours_hole_only:  0.086 m

bad_depth_mask_v1 hole total variation median:
  OpenCV NS:       0.042 m
  ours_hole_only:  0.062 m
```

## 目录结构

总 summary：

```text
output/far_pic_benchmark/summary.json
```

每个 case 都有：

```text
output/far_pic_benchmark/<case>/summary.json
output/far_pic_benchmark/<case>/masks
output/far_pic_benchmark/<case>/corrupted
output/far_pic_benchmark/<case>/outputs/opencv_ns
output/far_pic_benchmark/<case>/outputs/opencv_telea
output/far_pic_benchmark/<case>/outputs/ours_hole_only
output/far_pic_benchmark/<case>/outputs/ours_restored
output/far_pic_benchmark/<case>/visualizations
output/far_pic_benchmark/<case>/external_inputs
```

`external_inputs` 是为 ProPainter/RAD 等外部方法准备的标准输入包：

```text
external_inputs/depth_npy      # corrupted depth .npy
external_inputs/mask_npy       # mask .npy，1 表示待补区域
external_inputs/export/frames  # 灰度 PNG depth，可给图像/视频 inpainting 方法
external_inputs/export/masks   # PNG mask，白色表示待补区域
external_inputs/source_mapping.json
external_inputs/export/depth_meta.json
```

## 下一步建议

1. 短期结果不要使用当前 `ours_restored`。
2. 如果需要一版可用 depth，优先看：

```text
output/far_pic_benchmark/bad_depth_mask_v1/outputs/opencv_ns
```

3. 继续比较 ProPainter 时，直接拿每个 case 的：

```text
output/far_pic_benchmark/<case>/external_inputs/export/frames
output/far_pic_benchmark/<case>/external_inputs/export/masks
```

4. RAD 可以作为 RGB/image inpainting baseline 尝试，但需要注意它不是 ToF 物理模型，结果只能说明 depth-as-image 补洞效果。
5. 如果要真正验证 DepthCAD/IQ 方向，必须重新保存正确的 6 通道 IQ 或 9 通道 raw9；仅有 depth 图时，所有方法都只是 depth/image 补洞。

