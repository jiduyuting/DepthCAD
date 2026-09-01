# Real Depth Inference Report

## Data

- Input directory: `depth/`
- Samples: 41 `.npy` depth maps
- Shape: `240 x 320`
- Channels: 1 depth channel only, not 9-channel ToF/IQ data
- Dtype: `float32`
- Single-file size: `307200` bytes, matching `240 x 320 x 4`; a 9-channel `float32` file would be `2764800` bytes
- Depth range: approximately `0-8 m`
- Hole definition: non-finite depth or `depth <= 0`
- Mean hole ratio: `4.02%`
- Hole ratio range: `0.64% - 19.25%`

Important correction: many visually invalid pixels are not exactly zero, but tiny positive
depths such as `0.0016m`, `0.0048m`, `0.04m`, or `0.19m`. The first run only used
`depth <= 0` as holes, so those near-zero bad pixels were preserved. For this real sequence,
the scene depth is mostly `4-8m`, so a practical real-data hole threshold is `depth <= 1.0m`.

## Model

- Checkpoint: `output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt`
- Reason for using this checkpoint: the real files only contain depth, no IQ/amplitude. The final best `noisy_amp` transformer checkpoint needs amplitude/IQ-derived channels, so applying it with fake amplitude would be an unfair domain-mismatched test.
- Real-data condition construction:
  - `raw depth`: loaded from `.npy`
  - `hole_mask`: `depth <= 0` or non-finite
  - `confidence`: binary proxy, `1` on valid depth and `0` in holes
  - `anchor`: Navier-Stokes depth inpainting on the raw holes
  - model input channels: `[anchor_norm, noisy_norm, hole_mask, confidence]`

## Outputs

Saved to:

```text
output/real_depth_flow_noisy_ns_endpoint/
```

The corrected real-data run with `depth <= 1.0m` treated as holes is saved to:

```text
output/real_depth_flow_noisy_ns_endpoint_thr1m/
```

Important subdirectories:

- `restored/`: full flow-restored depth, clipped to the valid depth range of each real frame
- `restored_raw/`: raw model output before physical clipping
- `hole_only/`: recommended conservative output, uses model only inside holes and preserves original valid pixels
- `anchor/`: NS inpainting anchor
- `hole_mask/`: binary hole masks
- `visualizations/`: per-sample comparison figures
- `summary.json`: per-sample and aggregate statistics

## Aggregate Statistics

These are not accuracy metrics because there is no ground-truth depth for the real data.
They only measure how strongly the model changes the input/anchor.

Corrected run, `hole_depth_threshold=1.0`:

| Statistic | Value |
|---|---:|
| Samples | 41 |
| Mean hole ratio | 0.0417 |
| Min / max hole ratio | 0.0073 / 0.1940 |
| Mean `|model - anchor|` in holes | 0.4054 m |
| Min / max `|model - anchor|` in holes | 0.1260 / 0.8499 m |
| Mean `|model - raw|` on valid pixels | 0.0233 m |
| Min / max `|model - raw|` on valid pixels | 0.0120 / 0.0560 m |

In the corrected `hole_only/` output, the remaining ratio of pixels `<= 1.0m` is `0.0`.

Original run, `hole_depth_threshold=0.0`:

| Statistic | Value |
|---|---:|
| Samples | 41 |
| Mean hole ratio | 0.0402 |
| Min / max hole ratio | 0.0064 / 0.1925 |
| Mean `|model - anchor|` in holes | 0.5237 m |
| Min / max `|model - anchor|` in holes | 0.1865 / 1.2184 m |
| Mean `|model - raw|` on valid pixels | 0.0267 m |
| Min / max `|model - raw|` on valid pixels | 0.0143 / 0.0562 m |

Before clipping, the raw model occasionally produced physically unreasonable outliers:

| Output | Min | Max |
|---|---:|---:|
| Raw model output | -2.3865 m | 12.3971 m |
| Clipped model output | 0.1000 m | 8.0000 m |

This suggests that real depth-only inference has some domain shift. For reliable use, prefer `hole_only/` plus physical clipping unless we add real amplitude/IQ or fine-tune on real depth.

## Visual Inspection

Representative files:

- `visualizations/33.png`: largest hole ratio, about `19.25%`
- `visualizations/32.png`: second largest hole ratio, about `17.20%`
- `visualizations/22.png`: high model correction in holes
- `visualizations/34.png`: largest valid-region change
- `visualizations/5.png`: ordinary smaller-hole case

Observed behavior:

- In large contiguous missing regions, the model output is smoother and more globally coherent than plain NS anchor.
- The model can fill many holes into plausible continuous depth surfaces.
- The full restored result changes valid pixels slightly across the image. The average change is small, about `2.7 cm`, but the max per-sample average reaches about `5.6 cm`.
- The model is more aggressive in holes than NS anchor, which is expected, but some real samples show strong corrections and need visual checking.

## Current Conclusion

The method can run on real depth-only data, but this is not yet the strongest final model because the available real files do not include amplitude/IQ. The safest current real-data output is:

```text
output/real_depth_flow_noisy_ns_endpoint_thr1m/hole_only/
```

For a stronger and fairer test, collect or export real IQ/amplitude/confidence together with depth, then run the final `noisy_amp + transformer_bottleneck` flow model or fine-tune it on real/sim-real mixed data.
