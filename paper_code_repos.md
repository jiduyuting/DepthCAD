# Paper Code Repositories And Applicability

Checked on 2026-07-01 for papers under `paper/`.

## Directly Runnable Or Worth Trying

| Paper file | Repository / project | Status | Fit for current far_pic depth test |
| --- | --- | --- | --- |
| `Zhou 等 - 2023 - ProPainter Improving Propagation and Transformer for Video Inpainting.pdf` | https://github.com/sczhou/ProPainter | Official code and pretrained weights are public. Already tested locally on `far_pic/noise_depth_240x320_m_zero_mask`. | High as a depth-as-video-image baseline. It is not ToF-physical, but it preserves mask outside and uses temporal propagation. |
| `Kim 等 - 2025 - RAD Region-Aware Diffusion Models for Image Inpainting.pdf` | https://github.com/srk1995/RAD | Official PyTorch code is public. Checkpoints are linked from the repo. | Medium. It is RGB image inpainting, single-image oriented. Can be tested by colorizing/normalizing depth to PNG, then decoding back, but metric depth fidelity is not guaranteed. |

## Has A Page Or Repo But Not Useful As A Runnable Baseline Now

| Paper file | Repository / project | Status | Reason |
| --- | --- | --- | --- |
| `Shi 等 - IMFine 3D Inpainting via Geometry-guided Multi-view Refinement.pdf` | https://github.com/zhshi0816/IMFine and https://xinxinzuo2353.github.io/imfine/ | Official GitHub exists, but README says the main algorithm code is not released due to IP policy. | Not runnable for our purpose. Also requires multi-view / 3D scene setup, not single ToF depth maps. |
| `Xie 等 - 2025 - TurboFill Adapting Few-step Text-to-image Model for Fast Image Inpainting.pdf` | https://liangbinxie.github.io/projects/TurboFill/ | Project page and paper found; no public code link found on the project page. | Not runnable unless code/weights are later released. RGB text-guided inpainting, not ToF-physical. |

## Not Appropriate For Current Completion Comparison

| Paper file | Found page | Status | Reason |
| --- | --- | --- | --- |
| `Wang 等 - 2025 - InpDiffusion Image Inpainting Localization via Conditional Diffusion Models.pdf` | https://arxiv.org/abs/2501.02816 | No official code link found on arXiv. | This is inpainting localization / forensics mask prediction, not image/depth completion. |
| `Mo 等 - 2025 - Query-efficient Attack for Black-box Image Inpainting Forensics via Reinforcement Learning.pdf` | No reliable public code or project page found from title search. | Unknown. | This is black-box attack / forensics, not completion. |
| `Zhang 等 - 2026 - GBR Generative Bundle Refinement for High-Fidelity Gaussian Splatting With Enhanced Mesh Reconstruc.pdf` | https://arxiv.org/abs/2412.05908 | No official code link found on arXiv. | Sparse-view 3D Gaussian splatting / mesh refinement, not single ToF depth completion. |
| `Yang 等 - 2026 - LUCID Learning Unified Control for Image Deflaring and Exposure Mastery in Nighttime Photography.pdf` | https://arxiv.org/abs/2606.06901 | No official code link found on arXiv. | Nighttime photo deflaring / exposure restoration, not depth completion. |
| `Wang 等 - 2026 - Image Inpainting Methods A Review of Deep Learning Approaches.pdf` | Review paper | Not a method repo. | Useful for survey/background only. |

## Recommended Benchmark Direction

Do not use `restored/` from the current depth-flow model as a final result on `far_pic`; it modifies observed valid pixels heavily. Use `hole_only/` only if comparing this model.

The most defensible next benchmark is:

1. Start from `far_pic/noise_depth_240x320_m`.
2. Build two masks:
   - `zero_mask`: `depth == 0`.
   - `bad_depth_mask`: `zero_mask` plus far-depth speckles, local jump outliers, small noisy components, and temporal-inconsistent pixels.
3. Compare:
   - OpenCV NS / Telea.
   - ProPainter.
   - RAD, if installed.
   - Our current depth-flow `hole_only`.
4. For pseudo-GT evaluation, artificially mask valid depth pixels and report mask-region MAE against the original valid depth.
5. For real zero/bad masks, report no-reference diagnostics only: boundary jump, hole total variation, temporal consistency, and outside-mask preservation.

