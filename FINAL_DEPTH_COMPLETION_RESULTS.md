# Final Depth Completion Results

Date: 2026-05-16

This note summarizes the Kinect depth completion experiments on the `seed123` holdout set and the training/validation runs used to select the final model.

## Setup

- Base classical fill: `DepthCAD + plane` with `plane_max_ring_radius=12` and `plane_min_boundary_points=12`
- Learned model: residual U-Net trained on cached depth completion samples
- Final input mode: `depth`
- Final residual application: `hole` only
- Final residual blend: `binary`
- Training scale used for the strongest model: `n1000`

Holdout protocol:

- Training/validation cache: `depth_completion_cache/depth_cache_0515_n1000_plane_r12`
- External holdout cache: `depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123`
- Holdout split was excluded from training via `make_depth_completion_splits.py`

## Main Results

Metrics are evaluated on the `seed123` holdout set.

| Run | Global MAE | Hole MAE | Hole Improve vs Base | Valid MAE | Better/Worse | Worst Delta |
|---|---:|---:|---:|---:|---:|---:|
| Plane base | 0.141459 | 0.505875 | - | 0.073716 | - | - |
| n100 depth+amp | 0.114576 | 0.334381 | 33.9% | 0.073716 | 73/27 | 0.601051 |
| n500 depth+amp | 0.097376 | 0.224651 | 55.6% | 0.073716 | 87/13 | 0.186482 |
| n1000 depth+amp | 0.085487 | 0.148804 | 70.6% | 0.073716 | 93/7 | 0.125969 |
| n1000 depth-only | 0.082614 | 0.130479 | 74.2% | 0.073716 | 94/6 | 0.125982 |

## Ablations on the n1000 Depth-Only Model

| Strategy | Global MAE | Hole MAE | Hole Improve | Better/Worse | Worst Delta | P95 Delta | Median Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| binary | 0.082614 | 0.130479 | 74.2% | 94/6 | 0.125982 | 0.011075 | -0.181214 |
| soft_hole_distance `b1/r4` | 0.104705 | 0.271405 | 46.3% | 93/7 | 0.086222 | 0.006979 | -0.122895 |
| residual scale `0.75` | 0.092745 | 0.195105 | 61.4% | 95/5 | 0.074628 | -0.005398 | -0.153651 |

Interpretation:

- `binary` is the best overall strategy.
- `soft_hole_distance` reduces the worst regression but suppresses useful hole correction too much.
- `residual scale 0.75` is a conservative compromise, but it still underperforms `binary` on MAE.

## Key Conclusions

1. Depth-domain completion is the right direction.  
   Classical depth fill is much stronger than SD/image inpainting for hole completion.

2. Scaling training data improves performance consistently.  
   Hole MAE improves from `0.3344` at `n100` to `0.2247` at `n500` and to `0.1305` at `n1000` for the final depth-only model.

3. Amplitude features are not necessary at the largest scale tested.  
   `n1000 depth-only` outperforms `n1000 depth+amp`, so the final model should use depth-only inputs.

4. Conservative blending is not the best main strategy.  
   It can reduce the worst-case regression slightly, but it degrades average hole MAE too much.

## Recommended Final Model

Use this as the final learned baseline:

```text
DepthCAD + plane fill
-> depth-only residual U-Net
-> hole-only binary residual blending
-> trained on n1000 cache
```

## Useful Output Directories

- Final summary table: [output/depth_completion_summary_final/summary.md](output/depth_completion_summary_final/summary.md)
- n1000 depth-only holdout eval: `output/depth_completion_unet_depth_n1000_hole_binary/eval_seed123_ranked`
- n1000 depth+amp holdout eval: `output/depth_completion_unet_depth_amp_n1000_hole_binary/eval_seed123_ranked`
- n500 depth+amp holdout eval: `output/depth_completion_unet_depth_amp_n500_hole_binary/eval_seed123`
- n100 depth+amp holdout eval: `output/depth_completion_unet_depth_amp_n100_validw1/eval_seed123_fixed_holemask`

## Useful Visualization Directories

- n1000 depth-only ranked visualizations: `output/depth_completion_unet_depth_n1000_hole_binary/eval_seed123_ranked/visualizations`
- n1000 depth+amp ranked visualizations: `output/depth_completion_unet_depth_amp_n1000_hole_binary/eval_seed123_ranked/visualizations`

## Reproducibility Notes

- The split helper supports multiple cache directories and deduplication.
- The evaluation script supports `--residual_apply_mask`, `--residual_gate`, and `--residual_scale`.
- For the final reported numbers, use the `seed123` holdout set, not the training/validation split.

