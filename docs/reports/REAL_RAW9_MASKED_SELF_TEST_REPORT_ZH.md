# Real 9-Channel Raw Masked Self-Test Report

## Data Check

Real raw directory:

```text
/data/pre_student/GJ/DepthCAD/raw
```

Numeric files are true 9-channel tensors:

| Files | Count | Shape | Dtype |
|---|---:|---|---|
| `1.npy` - `42.npy` | 42 | `(9, 240, 320)` | `float32` |

The paired depth directory is:

```text
/data/pre_student/GJ/DepthCAD/depth
```

Depth files:

| Files | Count | Shape | Dtype |
|---|---:|---|---|
| `2.npy` - `42.npy` | 41 | `(240, 320)` | `float32` |

Therefore the strict paired set is `2-42`, total 41 samples.

There are also 23 timestamped files in `raw/`:

```text
20260610_..._raw.npy
```

These have shape `(2984960,)`, dtype `uint8`, so they are undecoded byte streams and were not used in this test.

## Model

Checkpoint:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt
```

This is the final `noisy_amp + transformer_bottleneck + endpoint flow` checkpoint.

The model expects:

```text
[anchor_norm, noisy_norm, hole_mask, confidence,
 amplitude_0, amplitude_1, amplitude_2, amplitude_mean]
```

plus the flow state and time channels internally.

## Raw9 to Amplitude Variants

Because the exact real raw channel semantics were not documented, two reasonable interpretations were tested.

### Variant A: `iq6`

This matches the synthetic training cache construction:

```python
I = raw[[0, 2, 4]]
Q = raw[[1, 3, 5]]
amplitude = sqrt(I^2 + Q^2)
amplitude_mean = mean(amplitude)
```

Output:

```text
output/real_raw9_masked_self_test_ratio10_thr1m_iq6/
```

### Variant B: `raw_258`

This uses the three nonnegative-looking raw channels directly:

```python
amplitude = raw[[2, 5, 8]]
amplitude_mean = mean(amplitude)
```

Output:

```text
output/real_raw9_masked_self_test_ratio10_thr1m_raw258/
```

## Protocol

Same as the real depth masked self-test:

1. Use paired real depth as pseudo-GT.
2. Treat `depth <= 1.0m` as invalid.
3. Randomly mask about 10% of reliable valid pixels.
4. Set masked depth to zero.
5. Set amplitude to zero in masked/invalid regions.
6. Construct NS anchor from corrupted depth.
7. Run noisy_amp flow model.
8. Compute MAE on artificially masked pixels.

## Results

Single-mask comparison on the same 41 paired samples:

| Method | Anchor Masked MAE | Model Masked MAE | Improvement vs Anchor | Better Cases |
|---|---:|---:|---:|---:|
| Depth-only flow endpoint | 0.09074 | 0.10038 | -10.62% | 7 / 41 |
| Raw9 noisy_amp flow, `iq6` | 0.09074 | 0.13020 | -43.48% | 5 / 41 |
| Raw9 noisy_amp flow, `raw_258` | 0.09074 | 0.13091 | -44.27% | 5 / 41 |

Full aggregate for `iq6`:

| Metric | Value |
|---|---:|
| Cases | 41 |
| Anchor masked MAE | 0.09074 m |
| Model masked MAE | 0.13020 m |
| Flow vs anchor masked improvement | -43.48% |
| Model unmasked MAE | 0.01970 m |
| Hole-only unmasked MAE | 0.00000 m |
| Anchor global MAE | 0.01057 m |
| Model full global MAE | 0.03257 m |
| Hole-only global MAE | 0.01517 m |

Full aggregate for `raw_258`:

| Metric | Value |
|---|---:|
| Cases | 41 |
| Anchor masked MAE | 0.09074 m |
| Model masked MAE | 0.13091 m |
| Flow vs anchor masked improvement | -44.27% |
| Model unmasked MAE | 0.01939 m |
| Hole-only unmasked MAE | 0.00000 m |
| Anchor global MAE | 0.01057 m |
| Model full global MAE | 0.03238 m |
| Hole-only global MAE | 0.01525 m |

## Interpretation

Using real 9-channel raw data directly with the synthetic-trained noisy_amp flow
does not improve real masked hole completion. It is worse than both:

1. NS inpainting anchor
2. The depth-only flow endpoint model

This strongly suggests a real/synthetic raw-domain mismatch:

- The real raw channel semantics may not match the synthetic `noisy_iq` layout.
- The dynamic range is very different. Synthetic IQ in cache is roughly `[-1, 1]`, while real raw channels are thousands or up to `65535`.
- The final model was trained on synthetic amplitude statistics, not real sensor amplitude statistics.
- The model can interpret real amplitude as a misleading condition, creating biased fills.

## Current Conclusion

The real 9-channel data is available, but direct zero-shot use of the synthetic
noisy_amp flow checkpoint is not reliable.

The next meaningful experiment should be real-domain adaptation:

1. Build a paired real raw9-depth masked training dataset from `raw/2-42.npy` and `depth/2-42.npy`.
2. Fine-tune the noisy_amp flow model using masked self-supervision.
3. Re-run this same masked self-test after fine-tuning.

This is now a stronger conclusion than the depth-only test:

> The issue is not merely missing raw channels. Real raw channels exist, but the
> synthetic-trained model does not understand their real distribution without
> calibration or fine-tuning.
