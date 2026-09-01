# Real Depth Masked Self-Test Report

## Purpose

This test checks whether the current depth-only flow model can fill holes on the
real depth distribution.

Because there is no ground-truth depth for the real holes, the test masks valid
pixels in the real depth maps and uses the original valid values as pseudo-GT.

## Data

- Input directory: `depth/`
- Samples: 41
- File shape: `(240, 320)`
- File dtype: `float32`
- Channels: 1 depth channel only, not 9-channel ToF/IQ data
- Real invalid threshold: `depth <= 1.0m`

## Model

- Checkpoint: `output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt`
- Input mode: `noisy`
- This is the depth-only flow endpoint model, not the final `noisy_amp` transformer model.
- The final `noisy_amp` model cannot be fairly used here because the real files do not contain amplitude/IQ channels.

## Protocol

For each real depth map:

1. Select reliable valid pixels: finite depth and `depth > 1.0m`.
2. Randomly mask about `10%` of reliable pixels with block/ellipse masks.
3. Set masked pixels to zero to create corrupted input.
4. Build condition:
   - corrupted depth
   - artificial + original invalid mask
   - binary confidence
   - NS inpaint anchor
5. Run flow endpoint restoration.
6. Evaluate only on artificially masked pixels, where pseudo-GT is known.

Metrics:

- `anchor_mask_mae`: NS anchor error on artificial mask
- `model_mask_mae`: flow model error on artificial mask
- `hole_only_mask_mae`: same as model on artificial mask, because hole-only uses model in holes
- `model_unmasked_mae`: how much full model changes unmasked reliable pixels
- `hole_only_unmasked_mae`: should be zero, because hole-only preserves valid pixels

## Results

Single-mask run:

```text
output/real_depth_masked_self_test_ratio10_thr1m/
```

Five-repeat run:

```text
output/real_depth_masked_self_test_ratio10_thr1m_rep5/
```

Robust five-repeat aggregate:

| Metric | Value |
|---|---:|
| Cases | 205 |
| Anchor masked MAE | 0.08463 m |
| Flow model masked MAE | 0.09813 m |
| Hole-only masked MAE | 0.09813 m |
| Flow vs anchor masked improvement | -15.95% |
| Flow full unmasked MAE | 0.01006 m |
| Hole-only unmasked MAE | 0.00000 m |
| Anchor global MAE | 0.00960 m |
| Flow full global MAE | 0.02005 m |
| Hole-only global MAE | 0.01113 m |

Case-level comparison:

| Outcome | Count |
|---|---:|
| Flow better than NS anchor | 49 |
| Flow worse or equal | 156 |
| Total | 205 |

Improvement distribution:

| Statistic | Value |
|---|---:|
| Mean improvement | -33.48% |
| Median improvement | -12.74% |
| Best cases | about +27% to +41% |
| Worst cases | about -293% to -604% |

## Interpretation

The masked self-test shows that the current depth-only flow model does not
generalize well enough to real depth holes. On average, it is worse than a simple
NS depth inpainting anchor.

This does not prove that the full physics-aware method is invalid, because this
test is not using the intended 9-channel ToF/amplitude/IQ input. It does show
that the current depth-only real transfer is not sufficient for a strong real
hole-completion claim.

Main likely reasons:

- The real files are single-channel depth maps, not 9-channel ToF data.
- The final best `noisy_amp` model cannot be used without amplitude/IQ channels.
- The model was trained on synthetic hole/noise distributions.
- Real depth maps are often smooth/planar, where NS inpainting is already a very strong baseline.
- The model can help in some hard NS-failure cases, but it introduces bias in many easy/smooth regions.

## Current Conclusion

The honest conclusion is:

> The current depth-only flow endpoint model is not reliable enough for real
> depth-only hole completion. It works on the synthetic benchmark, but real-domain
> masked self-test shows worse average masked MAE than NS inpainting.

For the project, the next useful step is not to keep tuning this depth-only real
inference. The next step should be one of:

1. Get the real 9-channel ToF/IQ/amplitude files and test the physics-aware model.
2. Fine-tune on real depth using masked self-supervision.
3. Train a real-depth-specific model where artificial masks on real valid pixels are part of the training objective.
