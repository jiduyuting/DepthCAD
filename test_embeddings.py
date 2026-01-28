#!/usr/bin/env python3
"""Test script to verify precomputed embeddings setup works"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import torch

print("Testing precomputed embeddings setup...")

# Test 1: Check if text_encoder loads without OOM
print("\n[Test 1] Loading Qwen3Model text encoder...")
try:
    from transformers import Qwen3Model
    text_encoder = Qwen3Model.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="text_encoder",
    )
    print(f"✓ Text encoder loaded successfully")
    print(f"  Device: {next(text_encoder.parameters()).device}")
    print(f"  Parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to load text encoder: {e}")
    sys.exit(1)

# Test 2: Check if it fits on GPU with VAE + Transformer
print("\n[Test 2] Checking GPU memory with all models...")
try:
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers import ZImageTransformer2DModel

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  No GPU available, skipping memory test")
        sys.exit(0)

    print(f"  Using device: {device}")

    # Load VAE
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="vae",
    ).to(device)

    # Load Transformer
    print("  Loading Transformer...")
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="transformer",
    ).to(device)

    # Check memory
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"  GPU memory used: {allocated_gb:.2f} GB / {total_gb:.2f} GB")

        # Try to move text_encoder to GPU
        print("  Attempting to move text_encoder to GPU...")
        text_encoder.to(device)
        allocated_after = torch.cuda.memory_allocated(device) / (1024**3)
        print(f"  After text_encoder: {allocated_after:.2f} GB")

        if allocated_after > total_gb * 0.95:
            print("  ✗ Text encoder causes near OOM!")
            print("  → Precomputed embeddings approach is required")
        else:
            print("  ✓ All models fit in GPU memory")

except torch.cuda.OutOfMemoryError:
    print("  ✗ CUDA OOM! This confirms we need precomputed embeddings.")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n✓ Test complete!")
