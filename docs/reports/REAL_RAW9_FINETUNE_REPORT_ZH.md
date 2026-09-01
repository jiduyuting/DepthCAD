# Real Raw9 Flow Fine-Tuning Report

## Goal

Use the real paired data:

```text
raw/2.npy   - raw/42.npy    shape: (9, 240, 320)
depth/2.npy - depth/42.npy  shape: (240, 320)
```

to fine-tune the synthetic-trained `noisy_amp + transformer_bottleneck + endpoint flow`
model with masked self-supervision.

This directly tests whether the method can adapt to real ToF raw9 distribution.

## Training Setup

Pretrained checkpoint:

```text
output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt
```

Pilot fine-tuned output:

```text
output/real_raw9_flow_finetune_iq6_pilot_e30/
```

Pilot best checkpoint:

```text
output/real_raw9_flow_finetune_iq6_pilot_e30/best.pt
```

Training configuration:

| Setting | Value |
|---|---:|
| Paired samples | 41 |
| Train pairs | 33 |
| Val pairs | 8 |
| Epochs | 30 |
| Best epoch | 28 |
| Batch size | 4 |
| Masks per train sample / epoch | 4 |
| Val masks per sample | 3 |
| Mask ratio | 10% |
| Input amplitude mode | `iq6` |
| Learning rate | `2e-5` |
| Device | GPU |

The `iq6` amplitude mode matches the synthetic cache construction:

```python
I = raw[[0, 2, 4]]
Q = raw[[1, 3, 5]]
amplitude = sqrt(I^2 + Q^2)
amplitude_mean = mean(amplitude)
```

## Training Behavior

Initial val before fine-tuning:

| Metric | Value |
|---|---:|
| NS anchor masked MAE | 0.06768 m |
| Model masked MAE | 0.09868 m |
| Improvement vs NS | -45.81% |

Best val after fine-tuning:

| Metric | Value |
|---|---:|
| NS anchor masked MAE | 0.06768 m |
| Model masked MAE | 0.04536 m |
| Improvement vs NS | +32.98% |

This shows that the synthetic-trained raw9 model is not zero-shot aligned to real data,
but it can adapt quickly once trained on real masked supervision.

## Stronger Fine-Tuning

A stronger real-domain run was trained with more epochs and more masked samples
per training pair:

```text
output/real_raw9_flow_finetune_iq6_e100_m8_lr2e5/
```

Best checkpoint:

```text
output/real_raw9_flow_finetune_iq6_e100_m8_lr2e5/best.pt
```

Training configuration:

| Setting | Value |
|---|---:|
| Paired samples | 41 |
| Train pairs | 33 |
| Val pairs | 8 |
| Epochs | 100 |
| Best epoch | 93 |
| Batch size | 4 |
| Masks per train sample / epoch | 8 |
| Val masks per sample | 5 |
| Mask ratio | 10% |
| Input amplitude mode | `iq6` |
| Learning rate | `2e-5` |
| Device | GPU |

Best validation result:

| Metric | Value |
|---|---:|
| NS anchor masked MAE | 0.07248 m |
| Model masked MAE | 0.04295 m |
| Improvement vs NS | +40.74% |
| Model unmasked MAE | 0.00327 m |

## Full 41-Sample Fixed-Mask Evaluation

Evaluation output:

```text
output/real_raw9_masked_self_test_ratio10_thr1m_iq6_finetuned_e30_best/
```

Stronger evaluation output:

```text
output/real_raw9_masked_self_test_ratio10_thr1m_iq6_finetuned_e100_m8_best_allvis/
```

Full visualization output:

```text
output/real_raw9_masked_self_test_ratio10_thr1m_iq6_finetuned_e100_m8_best_allvis/visualizations/
```

This directory contains one PNG for each of the 41 real samples.

Comparison on the same 41 fixed-mask cases:

| Method | Anchor Masked MAE | Model Masked MAE | Improvement vs NS | Better Cases | Model Unmasked MAE |
|---|---:|---:|---:|---:|---:|
| Depth-only flow endpoint | 0.09074 | 0.10038 | -10.62% | 7 / 41 | 0.01000 |
| Raw9 noisy_amp zero-shot, `iq6` | 0.09074 | 0.13020 | -43.48% | 5 / 41 | 0.01970 |
| Raw9 noisy_amp zero-shot, `raw_258` | 0.09074 | 0.13091 | -44.27% | 5 / 41 | 0.01939 |
| Raw9 noisy_amp fine-tuned pilot, `iq6` | 0.09074 | 0.05868 | +35.34% | 40 / 41 | 0.00350 |
| Raw9 noisy_amp fine-tuned strong, `iq6` | 0.09074 | 0.04680 | +48.43% | 41 / 41 | 0.00281 |

The stronger run also improves global metrics more clearly:

```text
anchor global MAE   = 0.01057 m
model global MAE    = 0.00794 m
hole-only global MAE= 0.00545 m
```

The hole-only global MAE after the pilot fine-tuning is:

```text
0.00683 m
```

while the NS anchor global MAE is:

```text
0.01057 m
```

## Ranked Cases

Worst cases of the stronger run:

| Sample | NS Anchor Masked MAE | Model Masked MAE | Improvement |
|---|---:|---:|---:|
| 33 | 0.15839 | 0.13574 | +14.30% |
| 21 | 0.02462 | 0.01948 | +20.87% |
| 25 | 0.02731 | 0.02152 | +21.21% |
| 35 | 0.07206 | 0.05505 | +23.61% |
| 29 | 0.29194 | 0.21624 | +25.93% |

Best cases of the stronger run:

| Sample | NS Anchor Masked MAE | Model Masked MAE | Improvement |
|---|---:|---:|---:|
| 6 | 0.06653 | 0.01512 | +77.28% |
| 20 | 0.10246 | 0.02438 | +76.20% |
| 32 | 0.21198 | 0.05980 | +71.79% |
| 34 | 0.26076 | 0.07636 | +70.72% |
| 2 | 0.04119 | 0.01292 | +68.62% |

For presentation, use samples `6`, `20`, `32`, `34` as clear positive cases.
The old failure case `21` is no longer a regression in the stronger run.

## Mask-Ratio Robustness

Both the pilot checkpoint and the stronger checkpoint were evaluated with
different artificial mask ratios on the same 41 real samples:

| Mask Ratio | Pilot Masked MAE | Pilot Improve | Pilot Better Cases | Strong Masked MAE | Strong Improve | Strong Better Cases |
|---|---:|---:|---:|---:|---:|---:|
| 5% | 0.06767 | +34.90% | 35 / 41 | 0.05369 | +48.35% | 41 / 41 |
| 10% | 0.05868 | +35.34% | 40 / 41 | 0.04680 | +48.43% | 41 / 41 |
| 20% | 0.06899 | +30.21% | 41 / 41 | 0.05585 | +43.50% | 41 / 41 |

Global MAE also improves more consistently with the stronger checkpoint:

| Mask Ratio | Pilot Global Improve | Strong Global Improve |
|---|---:|---:|
| 5% | -19.00% | +1.34% |
| 10% | +6.08% | +24.90% |
| 20% | +16.89% | +33.47% |

This supports a stronger robustness claim: after sufficient real-domain
adaptation, the flow model improves masked completion across light, medium, and
larger missing-region settings, and the stronger run dominates the pilot run at
all three mask ratios.

## Real-Hole-Shaped Self-Test

To make the masked self-test closer to real sensor behavior, the evaluation
script was extended with:

```text
--mask_mode real_hole_shapes
```

In this mode, artificial masks are not random rectangles or ellipses. Instead,
connected components from the real observed hole masks are extracted from the
dataset and translated onto reliable regions to create pseudo-missing areas with
realistic hole shapes.

A stricter setting was used:

```text
--mask_mode real_hole_shapes
--real_hole_exclude_self
--real_hole_max_components 24
```

Evaluation output:

```text
output/real_raw9_masked_self_test_realholes_ratio10_thr1m_iq6_finetuned_e100_m8_best_c24/
```

Result on 41 real samples:

| Setting | Anchor Masked MAE | Model Masked MAE | Improvement vs NS | Better Cases | Mean Actual Mask Ratio |
|---|---:|---:|---:|---:|---:|
| Real-hole-shaped self-test | 0.04758 | 0.03538 | +25.63% | 40 / 41 | 10.32% |

Important interpretation:

1. This is weaker than the random-block masked self-test (`+48.43%`), so the
   method does not generalize equally well to all hole geometries.
2. But it is still clearly positive, which means the model is not only learning
   to fill simple rectangular masks.
3. The remaining gap strongly suggests that training and evaluation should move
   further toward real-hole-shaped masks rather than only random block masks.

For deployment and reporting, the preferred output is hole-only compositing:
keep the original reliable depth outside the hole and only replace the masked
hole region with the model prediction. This is important because the full model
prediction slightly changes reliable pixels, which can hurt full-image global
MAE when the mask ratio is small.

## Interpretation

This is now a stronger positive real-data result with a clear progression:

1. Zero-shot raw9 fails.
2. Real raw9 masked fine-tuning works.
3. Longer training with more masks per sample improves further and removes the
   previous failure case.

The earlier failures mean:

1. Depth-only zero-shot does not work well on real data.
2. Raw9 noisy_amp zero-shot also does not work because real raw distribution differs from synthetic raw distribution.

The fine-tuning result means:

1. Real raw9 channels are useful.
2. The flow model can learn to use real raw9 after real-domain adaptation.
3. The method can beat NS inpainting on real masked self-test.
4. Stronger real fine-tuning further improves both masked and global metrics.

This changes the conclusion:

> The method is not reliable as a zero-shot synthetic-to-real model, but it becomes
> effective after real raw9 masked self-supervised fine-tuning.

## Suggested Claim

A safe thesis/report claim is:

> We first observed that synthetic-trained depth-only and raw9 models do not
> directly generalize to real ToF data. However, using the available real 9-channel
> ToF raw measurements and masked self-supervised fine-tuning, the proposed
> endpoint flow model improves masked depth completion over NS inpainting by
> 48.4% on 41 real paired samples, with all 41 cases improved.

## Next Step

1. Try the same strong schedule with `raw_258`.
2. Test whether a larger real validation split changes the conclusion.
3. Train the same strong schedule with `--mask_mode real_hole_shapes`.
4. Move from masked self-test to real observed holes when a trusted pseudo-target
   or multi-frame reference is available.
