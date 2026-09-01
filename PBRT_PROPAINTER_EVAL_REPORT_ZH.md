# PBRT ProPainter Baseline Evaluation

Date: 2026-07-03

## Setup

Evaluation set:

```text
depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123
```

This is the same independent 100-sample PBRT holdout used by the current depth restoration reports.

Exported ProPainter case:

```text
output/pbrt_propainter_seed123
```

ProPainter is evaluated as a depth-as-grayscale external inpainting baseline:

```text
depth_noisy + hole_mask
-> grayscale PNG frames + binary masks
-> ProPainter
-> decode grayscale back to metric depth
-> merge only inside hole mask
```

The valid/outside-hole pixels in the ProPainter output are copied from `depth_noisy`, so ProPainter is only credited for hole completion.

Current final model used for comparison:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt
```

Existing final-flow eval:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint
```

## Commands

Export PBRT holdout to ProPainter format:

```bash
python export_pbrt_propainter_case.py \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --output_dir output/pbrt_propainter_seed123
```

Run ProPainter and decode metric depth:

```bash
python run_external_inpainting_far_pic.py run-propainter \
  --case output/pbrt_propainter_seed123 \
  --output_dir output/pbrt_propainter_seed123/propainter_run \
  --height 256 \
  --width 256 \
  --mask_dilation 0 \
  --neighbor_length 10 \
  --ref_stride 10 \
  --subvideo_length 80 \
  --decode
```

Note: ProPainter completed the PNG inference and decode successfully. Its own mp4 write step returned a non-zero `imageio` error, but the adapter found all 100 output PNG frames and decoded them:

```text
output/pbrt_propainter_seed123/propainter_run/restored_by_stem
```

Evaluate and compare:

```bash
python eval_pbrt_external_inpainting.py \
  --case_dir output/pbrt_propainter_seed123 \
  --output_dir output/pbrt_propainter_seed123/evaluation \
  --existing_eval final_flow:output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint \
  --existing_eval depth_only_flow:output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_endpoint \
  --existing_eval large_resunet:output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_large_res48_b2/eval_seed123_endpoint \
  --existing_eval residual_restoration:output/depth_restoration_unet_noisy_ns_n1000/eval_seed123
```

Main result file:

```text
output/pbrt_propainter_seed123/evaluation/summary.json
```

## Main Metrics

All metrics are in meters. Lower is better.

| Method | Global MAE | Hole MAE | Valid MAE | Hole RMSE |
|---|---:|---:|---:|---:|
| Noisy with holes | 0.503548 | 2.911191 | 0.055985 | 3.140677 |
| OpenCV NS r15 | 0.108885 | 0.393456 | 0.055985 | 0.778730 |
| ProPainter | 0.097623 | 0.321613 | 0.055985 | 0.674999 |
| Residual restoration | 0.056383 | 0.114204 | 0.045635 | - |
| Depth-only endpoint flow | 0.050123 | 0.110299 | 0.038937 | - |
| Large ResUNet endpoint flow | 0.038709 | 0.105873 | 0.026223 | - |
| Final transformer endpoint flow | **0.037659** | **0.104364** | **0.025259** | - |

## Key Comparison

ProPainter improves over deterministic OpenCV NS on hole completion:

```text
OpenCV NS hole MAE: 0.393456
ProPainter hole MAE: 0.321613
Relative improvement: 18.3%
```

But ProPainter is much worse than our final model:

```text
ProPainter hole MAE: 0.321613
Final flow hole MAE: 0.104364
Delta: +0.217249 m
Ratio: 3.08x worse
```

Global MAE is also worse:

```text
ProPainter global MAE: 0.097623
Final flow global MAE: 0.037659
Ratio: 2.59x worse
```

Valid-region MAE is worse because ProPainter only fills holes and keeps noisy valid pixels:

```text
ProPainter valid MAE: 0.055985
Final flow valid MAE: 0.025259
```

## Paired Sample-Level Result

Against the final transformer endpoint flow on the same 100 samples:

```text
Paired samples: 100
Our final flow better: 97
ProPainter better: 3
Tied: 0
Mean hole MAE delta, ProPainter - ours: +0.217059 m
Median hole MAE delta, ProPainter - ours: +0.139039 m
```

The few samples where ProPainter is better:

| Sample | ProPainter Hole MAE | Final Flow Hole MAE | Delta |
|---|---:|---:|---:|
| breakfast/0/151 | 0.561329 | 1.065552 | -0.504223 |
| white-room/0/107 | 0.756170 | 0.928565 | -0.172396 |
| pavilion/0/106 | 0.328368 | 0.392955 | -0.064587 |

Worst ProPainter regressions against final flow:

| Sample | ProPainter Hole MAE | Final Flow Hole MAE | Delta |
|---|---:|---:|---:|
| breakfast/1/227 | 1.018016 | 0.126646 | +0.891370 |
| breakfast/1/174 | 1.236134 | 0.414391 | +0.821743 |
| breakfast/1/125 | 0.887295 | 0.169024 | +0.718272 |
| white-room/1/165 | 0.673330 | 0.020116 | +0.653214 |
| white-room/1/178 | 0.670600 | 0.022781 | +0.647818 |

## Conclusion

ProPainter is useful as an external RGB/video inpainting baseline and does beat a simple OpenCV NS fill on this PBRT holdout. However, it is not better than our model. On the main hole-region metric, it is roughly 3.1x worse than the current final transformer endpoint flow.

The next generalization work should therefore not pivot toward ProPainter as the main method. It is more useful as a failure-case probe: inspect the three samples where ProPainter beats our final flow, then target those cases in the depth-domain model or data generation pipeline.

The immediate targets are:

```text
breakfast/0/151
white-room/0/107
pavilion/0/106
```

These should be used as the first diagnostic set for targeted generalization.
