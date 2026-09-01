# Final Depth Restoration Results

Date: 2026-05-18

This note summarizes the final single-model depth restoration experiments on the `seed123` holdout set. It supersedes the earlier two-stage depth completion result as the recommended main method.

## Setup

Final method:

```text
noisy depth
+ NS depth anchor
+ hole mask
+ confidence
-> single mask-aware residual U-Net
-> clean dense depth
```

The NS anchor is a deterministic depth-domain inpainting prior, not a learned model. The final learned component is a single restoration network.

Training and evaluation:

- Training cache: `depth_completion_cache/depth_cache_0515_n1000_plane_r12`
- Train/val split: `output/splits_n1000_plane_r12_exclude_seed123`
- External holdout cache: `depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123`
- Final checkpoint: `output/depth_restoration_unet_noisy_ns_n1000/best.pt`
- Final eval directory: `output/depth_restoration_unet_noisy_ns_n1000/eval_seed123`
- Final summary directory: `output/depth_restoration_summary_final`

## Main Results

All metrics are evaluated on the independent `seed123` holdout set.

| Method | Learned Models | Uses DepthCAD | Global MAE | Hole MAE | Valid MAE |
|---|---:|---|---:|---:|---:|
| Noisy | 0 | No | 0.503548 | 2.911191 | 0.055985 |
| NS Anchor | 0 | No | 0.104833 | 0.367608 | 0.055985 |
| DepthCAD/Plane Base | 1 | Yes | 0.141459 | 0.505875 | 0.073716 |
| Two-stage Completion | 2 | Yes | 0.082614 | 0.130479 | 0.073716 |
| Ours Single Restoration | 1 | No | 0.056383 | 0.114204 | 0.045635 |

## Improvements

Compared with the raw noisy input:

- Global MAE: `0.503548 -> 0.056383`, `88.8%` improvement
- Hole MAE: `2.911191 -> 0.114204`, `96.1%` improvement
- Valid MAE: `0.055985 -> 0.045635`, `18.5%` improvement

Compared with the deterministic NS anchor:

- Global MAE: `0.104833 -> 0.056383`, `46.2%` improvement
- Hole MAE: `0.367608 -> 0.114204`, `68.9%` improvement
- Valid MAE: `0.055985 -> 0.045635`, `18.5%` improvement

Compared with the earlier DepthCAD/plane base:

- Global MAE: `0.141459 -> 0.056383`, `60.1%` improvement
- Hole MAE: `0.505875 -> 0.114204`, `77.4%` improvement
- Valid MAE: `0.073716 -> 0.045635`, `38.1%` improvement

Compared with the earlier two-stage learned completion model:

- Global MAE: `0.082614 -> 0.056383`, `31.8%` improvement
- Hole MAE: `0.130479 -> 0.114204`, `12.5%` improvement
- Valid MAE: `0.073716 -> 0.045635`, `38.1%` improvement

## Ablations

| Variant | Global MAE | Hole MAE | Valid MAE | Hole Improve vs Anchor/Base | Better/Worse | Worst Delta | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| 30-epoch pilot | 0.071754 | 0.147794 | 0.057618 | 59.8% | 97/3 | 0.005796 | undertrained single restoration |
| Ours main | 0.056383 | 0.114204 | 0.045635 | 68.9% | 99/1 | 0.001890 | best single-model result |
| valid2_anchor02 | 0.060767 | 0.117253 | 0.050266 | 68.1% | 96/4 | 0.064092 | over-constrained valid/anchor regularization |
| gated residual | 0.058415 | 0.108534 | 0.049098 | 70.5% | 98/2 | 0.001014 | hole-focused gated architecture variant |
| Two-stage completion | 0.082614 | 0.130479 | 0.073716 | 74.2% | 94/6 | 0.125982 | strong two-stage baseline |

Interpretation:

- The 120-epoch main model improves both hole completion and valid-region restoration.
- Stronger valid loss plus anchor regularization is not beneficial here. It reduces global, hole, and valid performance on the holdout set.
- The gated residual variant achieves the lowest hole MAE (`0.108534`) but worsens global and valid MAE compared with the main residual model. It is useful as a hole-focused architecture ablation, but it should not replace the main model unless the target application prioritizes hole MAE over overall restoration.
- The single restoration model outperforms the two-stage DepthCAD plus learned completion pipeline while using fewer learned components.

## Ranked Qualitative Cases

Top and bottom cases ranked by `model_hole_mae - anchor_hole_mae` are saved here:

```text
output/depth_restoration_summary_final/ranked_cases_vs_anchor.md
output/depth_restoration_summary_final/ranked_cases_vs_anchor.csv
```

The worst regression against NS anchor is very small:

```text
contemporary-bathroom/0/131
anchor_hole_mae = 0.003178
model_hole_mae  = 0.005068
delta           = 0.001890
```

This means the main model improves `99/100` holdout samples in the hole region, and the only hole-region regression is negligible in absolute MAE.

## Ranked Visualization Command

The restoration evaluator supports ranked visualization:

```bash
python -u eval_depth_restoration.py \
  --checkpoint output/depth_restoration_unet_noisy_ns_n1000/best.pt \
  --cache_dir depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123 \
  --split all \
  --output_dir output/depth_restoration_unet_noisy_ns_n1000/eval_seed123_ranked \
  --batch_size 8 \
  --num_workers 4 \
  --visualize \
  --vis_max_samples 12 \
  --vis_rank best_worst_hole \
  --vis_rank_baseline anchor
```

This saves the six strongest and six weakest cases under:

```text
output/depth_restoration_unet_noisy_ns_n1000/eval_seed123_ranked/visualizations
```

## Key Conclusion

The final technical route should be single-model, mask-aware depth restoration rather than two-stage DepthCAD completion. The proposed model directly restores dense depth from degraded depth, hole mask, confidence, and a deterministic NS anchor. On the external `seed123` holdout set, it improves global, hole, and valid MAE over raw noisy depth, NS anchor, DepthCAD/plane, and the previous two-stage learned completion baseline.
