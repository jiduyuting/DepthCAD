#!/usr/bin/env python3
"""
DepthCAD Inference Script for Z-Image-Turbo Model

This script is adapted from inference.py to work with Tongyi-MAI/Z-Image-Turbo.

Key differences from original inference.py:
1. Uses ZImagePipeline instead of StableDiffusionControlNetPipeline
2. Potentially uses QwenImageControlNetModel for ControlNet
3. May require different image generation parameters

⚠️ WARNING: This is EXPERIMENTAL and may not work without further modifications.
The ZImagePipeline may not support ControlNet natively.
"""

import cv2
import os
import torch
import argparse
import numpy as np

# Try to import Z-Image-Turbo specific pipeline
try:
    from diffusers import ZImagePipeline
    ZIMAGE_PIPELINE_AVAILABLE = True
except ImportError:
    ZIMAGE_PIPELINE_AVAILABLE = False
    print("Warning: ZImagePipeline not available. Update diffusers to >=0.36.0")

# Fallback to standard pipeline if ZImagePipeline not available
if not ZIMAGE_PIPELINE_AVAILABLE:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Support both flat_dataset and pbrt_dataset formats
try:
    from pbrt_dataset.preprocess import load_raw as load_raw_pbrt, compute_gradient_confidence
    PBRT_AVAILABLE = True
except ImportError:
    PBRT_AVAILABLE = False

try:
    from flat_dataset.preprocess import load_raw as load_raw_flat, compute_gradient_confidence as compute_gradient_confidence_flat
    FLAT_AVAILABLE = True
except ImportError:
    FLAT_AVAILABLE = False

from IQToDepth import IQ_to_depth


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo"  # Changed default
    )
    parser.add_argument(
        "--depthcad_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--noise_IQ_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--noise_depth_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="pbrt",
        choices=["pbrt", "flat"],
        help="Dataset type: 'pbrt' for .npy files (9 channels) or 'flat' for binary files"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[240, 320],
        help="Target size for resizing (height, width)"
    )

    args = parser.parse_args()
    return args


def inference(pipe, noise, conf, scale, target_size=(240, 320)):
    """
    Run inference to predict IQ channels.
    内部强制使用 512 进行推理，最后还原回 target_size
    """
    # 1. 定义模型推理需要的尺寸 (Stable Diffusion 需要是 64 的倍数)
    infer_h, infer_w = 512, 512

    # 2. 预处理：将置信度图 resize 到推理尺寸
    # cv2.resize 参数顺序是 (width, height)
    conf_resized = cv2.resize(conf, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

    # 初始化用于存储推理结果的数组 (使用推理尺寸 512x512)
    pred_IQs_infer = np.zeros((6, infer_h, infer_w))

    for i in range(6):
        # 3. 预处理：将当前通道的 noise resize 到推理尺寸
        noise_resized = cv2.resize(noise[i], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

        # 现在的 shape 统一了，可以 stack
        guidance = np.stack([noise_resized, conf_resized], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0)

        prompt = ""

        # generate image
        generator = torch.manual_seed(42)

        # ⚠️ WARNING: ZImagePipeline may have different parameters
        try:
            # Try ZImagePipeline interface first
            pred_IQ = pipe(
                prompt,
                num_inference_steps=20,
                generator=generator,
                image=guidance,
                height=infer_h,
                width=infer_w
            ).images[0]
        except Exception as e:
            print(f"Error using pipeline: {e}")
            print("Trying fallback method...")
            # Fallback: try different parameter combinations
            try:
                pred_IQ = pipe(
                    prompt,
                    num_inference_steps=20,
                    generator=generator,
                    controlnet_conditioning_image=guidance,  # Alternative parameter name
                    height=infer_h,
                    width=infer_w
                ).images[0]
            except:
                raise RuntimeError("Failed to generate image. Pipeline may not support ControlNet.")

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0    # convert to (0, 1)
        pred_IQ = 2 * pred_IQ - 1   # (-1, 1)

        # 将结果存入推理尺寸的数组中
        pred_IQs_infer[i] = pred_IQ * scale

    # 4. 后处理：将结果 Resize 回用户指定的目标尺寸 (240, 320)
    target_h, target_w = target_size
    reshaped_IQs = np.zeros((6, target_h, target_w), dtype=np.float32)

    for i in range(6):
        reshaped_IQs[i, :, :] = cv2.resize(
            pred_IQs_infer[i, :, :],
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    return reshaped_IQs


if __name__ == '__main__':
    args = parse_args()
    base_model_path = args.pretrained_model_name_or_path
    depthcad_path = args.depthcad_path
    noise_file = args.noise_IQ_file
    noise_depth_file = args.noise_depth_file
    out_file = args.out_file
    dataset_type = args.dataset_type
    target_size = tuple(args.target_size)

    print("=" * 60)
    print("DepthCAD Inference (Z-Image-Turbo)")
    print("=" * 60)
    print(f"Dataset type: {dataset_type}")
    print(f"Target size: {target_size}")
    print(f"Noise IQ file: {noise_file}")
    print(f"Noise depth file: {noise_depth_file}")
    print(f"Output file: {out_file}")

    # Load data based on dataset type
    if dataset_type == "pbrt":
        if not PBRT_AVAILABLE:
            raise ImportError("pbrt_dataset.preprocess not available. Please check your imports.")

        print("\nLoading PBRT data...")
        # Load IQ data from .npy file (shape: 9, 240, 320)
        noise_result = load_raw_pbrt(noise_file, target_size=target_size, sqrt_in=True,
                                      amplitude_threshold=None, upper_percentile=99.5)
        # load_raw_pbrt returns (tof_IQs, amp_mask)
        if isinstance(noise_result, tuple) or isinstance(noise_result, list):
            noise, amp_mask = noise_result
        else:
            noise = noise_result
            amp_mask = None
        print(f"Loaded noise IQ shape: {noise.shape}")

        # Apply amplitude mask to noise (zero out low amplitude regions)
        if amp_mask is not None:
            masked_pct = 100.0 * np.count_nonzero(amp_mask) / amp_mask.size
            print(f"Applying amplitude mask: {masked_pct:.2f}% pixels masked")
            for c in range(noise.shape[0]):
                noise[c][amp_mask] = 0.0

        # Load depth for confidence computation
        noise_depth = np.load(noise_depth_file)
        # cv2.resize expects (width, height), but target_size is (height, width)
        noise_depth = cv2.resize(noise_depth.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        confidence = compute_gradient_confidence(noise_depth)

        # Apply amplitude mask to confidence map (zero out low amplitude regions)
        if amp_mask is not None:
            confidence[amp_mask] = 0.0
            print(f"Applied mask to confidence map")

        print(f"Computed confidence map shape: {confidence.shape}")

        # Scale normalization
        scale = max(noise.max(), abs(noise.min()), 1e-8)
        print(f"Scale factor: {scale:.4f}")
        noise /= scale

    elif dataset_type == "flat":
        if not FLAT_AVAILABLE:
            raise ImportError("flat_dataset.preprocess not available. Please check your imports.")

        print("\nLoading FLAT data...")
        # Load IQ data from binary file
        noise_result = load_raw_flat(noise_file)
        if isinstance(noise_result, tuple) or isinstance(noise_result, list):
            noise, amp_mask = noise_result
        else:
            noise = noise_result
            amp_mask = None
        print(f"Loaded noise IQ shape: {noise.shape}")

        # Resize to target size if needed
        if noise.shape[1:] != target_size:
            reshaped_noise = np.zeros((6, target_size[0], target_size[1]), dtype=np.float32)
            for i in range(6):
                reshaped_noise[i] = cv2.resize(noise[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            noise = reshaped_noise
            print(f"Resized noise IQ to: {noise.shape}")

        # Load depth for confidence computation
        noise_depth = np.load(noise_depth_file)
        noise_depth = cv2.resize(noise_depth.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        confidence = compute_gradient_confidence_flat(noise_depth)
        print(f"Computed confidence map shape: {confidence.shape}")

        # Scale normalization
        scale = max(noise.max(), abs(noise.min()), 1e-8)
        print(f"Scale factor: {scale:.4f}")
        noise /= scale
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load pipeline
    print("\nLoading DepthCAD model...")
    print(f"Base model: {base_model_path}")
    print(f"DepthCAD: {depthcad_path}")

    # Try to use ZImagePipeline if available
    if ZIMAGE_PIPELINE_AVAILABLE:
        print("Using ZImagePipeline...")

        # Try to load appropriate ControlNet type
        try:
            from diffusers.models.controlnets import QwenImageControlNetModel
            depthcad = QwenImageControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)
            print("Loaded QwenImageControlNetModel")
        except:
            # Fallback to standard ControlNet
            depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)
            print("Loaded ControlNetModel (fallback)")

        try:
            pipe = ZImagePipeline.from_pretrained(
                base_model_path,
                controlnet=depthcad,
                torch_dtype=torch.float16
            )
            print("✓ ZImagePipeline loaded successfully")
        except Exception as e:
            print(f"✗ ZImagePipeline loading failed: {e}")
            print("This may indicate that ZImagePipeline does not support ControlNet")
            print("\nPossible solutions:")
            print("1. Check if ZImagePipeline supports controlnet parameter")
            print("2. Use a custom pipeline that adds ControlNet support")
            print("3. Consider using a different model that supports ControlNet")
            raise

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
    else:
        # Fallback to standard StableDiffusionControlNetPipeline
        print("Warning: ZImagePipeline not available, using StableDiffusionControlNetPipeline as fallback")
        print("This may not work correctly with Z-Image-Turbo!")

        from diffusers import ControlNetModel

        depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path, controlnet=depthcad, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()

    print("Model loaded successfully!")

    # Run inference
    print("\nRunning inference...")
    try:
        pred_IQs = inference(pipe, noise, confidence, scale, target_size=target_size)
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        print("\nThis error indicates that the pipeline is not compatible with the current setup.")
        print("Please check:")
        print("1. Does ZImagePipeline support ControlNet?")
        print("2. Are the input dimensions correct?")
        print("3. Is the model architecture compatible?")
        raise

    print(f"Predicted IQ shape: {pred_IQs.shape}")
    print(f"Predicted IQ range: [{pred_IQs.min():.4f}, {pred_IQs.max():.4f}]")

    # Convert IQ to depth
    print("\nConverting IQ to depth...")
    # Ensure pred_IQs is in the correct format: (6, h, w) with order I30 Q30 I40 Q40 I58 Q58
    # The IQ_to_depth function expects this order
    depth = IQ_to_depth(pred_IQs, corr_save_path=None, depth_save_path=None)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}] meters")

    # Save output
    print(f"\nSaving depth to {out_file}...")
    np.save(out_file, depth)
    print("Done!")
