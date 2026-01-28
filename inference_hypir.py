#!/usr/bin/env python3
"""
DepthCAD Inference Script for HYPIR-Enhanced Model

This script is designed to work with models trained using train_hypir.py.
Key features:
- HYPIR LoRA weights are pre-merged into UNet during training
- Uses simple 2-channel conditioning (noise + confidence)
- Uses transforms.Resize for data preprocessing (consistent with training)
- Compatible with standard StableDiffusionControlNetPipeline
"""

import cv2
import os
import sys
import torch
import argparse
import numpy as np
from torchvision import transforms

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Support both flat_dataset and pbrt_dataset formats
try:
    from pbrt_dataset.preprocess import load_raw as load_raw_pbrt, compute_gradient_confidence
    PBRT_AVAILABLE = True
    print("Successfully imported from pbrt_dataset.preprocess")
except ImportError as e:
    print(f"Failed to import from pbrt_dataset.preprocess: {e}")
    PBRT_AVAILABLE = False

try:
    from flat_dataset.preprocess import load_raw as load_raw_flat, compute_gradient_confidence as compute_gradient_confidence_flat
    FLAT_AVAILABLE = True
except ImportError:
    FLAT_AVAILABLE = False

from IQToDepth import IQ_to_depth


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="DepthCAD Inference for HYPIR-Enhanced Model"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained SD2.1 model"
    )
    parser.add_argument(
        "--depthcad_path",
        type=str,
        required=True,
        help="Path to trained DepthCAD ControlNet model"
    )
    parser.add_argument(
        "--noise_IQ_file",
        type=str,
        required=True,
        help="Path to noisy IQ file (.npy for PBRT or binary for FLAT)"
    )
    parser.add_argument(
        "--noise_depth_file",
        type=str,
        required=True,
        help="Path to noisy depth file (.npy)"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="Output path for predicted depth (.npy)"
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
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps (default: 20)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for diffusion (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args(input_args)
    return args


def inference(pipe, noise, conf, scale, target_size=(240, 320),
              num_inference_steps=20, guidance_scale=1.0, seed=42):
    """
    Run inference to predict IQ channels using HYPIR-enhanced DepthCAD.

    This version uses simple 2-channel conditioning (noise + confidence),
    consistent with the training approach in train_hypir.py.

    Args:
        pipe: StableDiffusionControlNetPipeline
        noise: numpy array of shape (6, H, W) - 6 IQ channels
        conf: numpy array of shape (H, W) - confidence map
        scale: float - scaling factor for denormalization
        target_size: tuple - target (height, width) for output
        num_inference_steps: int - number of denoising steps
        guidance_scale: float - guidance scale for diffusion
        seed: int - random seed

    Returns:
        pred_IQs: numpy array of shape (6, target_h, target_w)
    """
    # Use training resolution (512) for inference
    infer_h, infer_w = 512, 512

    # Resize confidence to inference size
    conf_resized = cv2.resize(conf, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

    # Initialize array for inference results (512x512)
    pred_IQs_infer = np.zeros((6, infer_h, infer_w))

    # Process each IQ channel
    for i in range(6):
        # Resize current channel to inference size
        noise_resized = cv2.resize(noise[i], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

        # Build simple 2-channel guidance (HYPIR-style)
        # guidance: [noise, conf] - shape [2, H, W]
        guidance = np.stack([noise_resized, conf_resized], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0)  # [1, 2, H, W]

        # Empty prompt for unconditional generation
        prompt = ""

        # Set random seed for reproducibility
        generator = torch.manual_seed(seed)

        # Run diffusion
        print(f"  Processing channel {i+1}/6...")
        pred_IQ = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            image=guidance,
            height=infer_h,
            width=infer_w
        ).images[0]

        # Post-process prediction
        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0    # convert to (0, 1)
        pred_IQ = 2 * pred_IQ - 1   # (-1, 1)

        # Denormalize and store
        pred_IQs_infer[i] = pred_IQ * scale

    # Resize predictions back to target size
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
    print("DepthCAD Inference for HYPIR-Enhanced Model")
    print("=" * 60)
    print(f"Dataset type: {dataset_type}")
    print(f"Target size: {target_size}")
    print(f"Noise IQ file: {noise_file}")
    print(f"Noise depth file: {noise_depth_file}")
    print(f"Output file: {out_file}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Random seed: {args.seed}")

    # Load data based on dataset type
    if dataset_type == "pbrt":
        if not PBRT_AVAILABLE:
            raise ImportError("pbrt_dataset.preprocess not available. Please check your imports.")

        print("\nLoading PBRT data...")
        # Load IQ data from .npy file (shape: 9, 240, 320)
        noise_result = load_raw_pbrt(
            noise_file,
            target_size=target_size,
            sqrt_in=True,
            amplitude_threshold=None,
            upper_percentile=99.5
        )

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
        noise_depth = cv2.resize(
            noise_depth.astype(np.float32),
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        confidence = compute_gradient_confidence(noise_depth)

        # Apply amplitude mask to confidence map
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
                reshaped_noise[i] = cv2.resize(
                    noise[i],
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            noise = reshaped_noise
            print(f"Resized noise IQ to: {noise.shape}")

        # Load depth for confidence computation
        noise_depth = np.load(noise_depth_file)
        noise_depth = cv2.resize(
            noise_depth.astype(np.float32),
            target_size,
            interpolation=cv2.INTER_LINEAR
        )
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
    print(f"ControlNet: {depthcad_path}")

    depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)

    # Verify ControlNet expects 2 input channels (HYPIR-style)
    num_channels = depthcad.controlnet_cond_embedding.conv_in.in_channels
    print(f"Detected ControlNet input channels: {num_channels}")

    if num_channels != 2:
        print(f"\n[WARNING] Expected 2 input channels for HYPIR-style model, but got {num_channels}")
        print("This may indicate the model was trained with enhanced features (6 channels)")
        print("Consider using inference_marigold.py instead\n")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=depthcad,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    print("Model loaded successfully!")

    # Run inference
    print("\nRunning inference...")
    pred_IQs = inference(
        pipe,
        noise,
        confidence,
        scale,
        target_size=target_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )

    print(f"\nPredicted IQ shape: {pred_IQs.shape}")
    print(f"Predicted IQ range: [{pred_IQs.min():.4f}, {pred_IQs.max():.4f}]")

    # Convert IQ to depth
    print("\nConverting IQ to depth...")
    depth = IQ_to_depth(pred_IQs, corr_save_path=None, depth_save_path=None)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}] meters")

    # Save output
    print(f"\nSaving depth to {out_file}...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.save(out_file, depth)
    print("Done!")
    print("=" * 60)
