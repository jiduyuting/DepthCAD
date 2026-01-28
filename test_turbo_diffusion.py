#!/usr/bin/env python3
"""
测试 TurboDiffusion 模型与当前 DepthCAD 代码的兼容性
"""
import sys
import os

print("=" * 80)
print("Testing TurboDiffusion Model Compatibility")
print("=" * 80)

# 测试多个可能的 TurboDiffusion 变体
model_variants = [
    # "turbo_diffusion/TurboDiffusion",
    # "TurboDiffusion/TurboDiffusion",
    # "apple/TurboDiffusion",
    # "thibaud/turbo-diffusion",
    "TurboDiffusion/TurboWan2.2-I2V-A14B-720P"
]

for model_id in model_variants:
    print(f"\n{'=' * 80}")
    print(f"Testing model: {model_id}")
    print('=' * 80)

    # 测试 1: 检查 text_encoder 类型
    print("\n[1/5] Testing text_encoder type...")
    try:
        from transformers import PretrainedConfig
        text_encoder_config = PretrainedConfig.from_pretrained(
            model_id,
            subfolder="text_encoder",
        )
        model_class = text_encoder_config.architectures[0] if text_encoder_config.architectures else "Unknown"
        print(f"✓ Text encoder loaded successfully")
        print(f"  Model class: {model_class}")

        if model_class == "CLIPTextModel":
            print(f"  ✓ Compatible with CLIP (like SD-2.1)")
        else:
            print(f"  ⚠ Different text encoder: {model_class}")

    except Exception as e:
        print(f"✗ Error loading text_encoder: {e}")

    # 测试 2: 检查 VAE 兼容性
    print("\n[2/5] Testing VAE compatibility...")
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
        )
        print(f"✓ VAE (AutoencoderKL) loaded successfully")
        print(f"  Config: latent_channels={vae.config.latent_channels}")
    except Exception as e:
        print(f"✗ Error loading VAE: {e}")

    # 测试 3: 检查 UNet 兼容性
    print("\n[3/5] Testing UNet compatibility...")
    try:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
        )
        print(f"✓ UNet2DConditionModel loaded successfully")
        print(f"  Sample size: {unet.config.sample_size}")
    except Exception as e:
        print(f"✗ Error loading UNet: {e}")

    # 测试 4: 检查 tokenizer
    print("\n[4/5] Testing tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            use_fast=False,
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")

    # 测试 5: 检查 scheduler
    print("\n[5/5] Testing scheduler...")
    try:
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        print(f"✓ Scheduler loaded successfully")
        print(f"  Type: {scheduler.config._class_name}")
    except Exception as e:
        print(f"✗ Error loading scheduler: {e}")

    # 如果所有测试都通过，尝试加载完整 pipeline
    print("\n[Bonus] Testing full pipeline...")
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import torch

        # 创建一个简单的 ControlNet
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=2)

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        print(f"✓ StableDiffusionControlNetPipeline loaded successfully")
        print(f"  → {model_id} is FULLY COMPATIBLE!")
        print(f"  → You can simply change MODEL_DIR to use this model")
        break  # 找到兼容的模型就停止
    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("If you see 'FULLY COMPATIBLE!' above, you can just change MODEL_DIR")
print("Otherwise, the model is not compatible with the current code")
