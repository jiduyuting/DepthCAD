#!/usr/bin/env python3
"""
测试 Z-Image-Turbo 模型与当前 DepthCAD 代码的兼容性
"""
import sys

print("=" * 80)
print("Testing Z-Image-Turbo Model Compatibility")
print("=" * 80)

# 测试 1: 检查 text_encoder 类型
print("\n[1/5] Testing text_encoder type...")
try:
    from transformers import PretrainedConfig
    text_encoder_config = PretrainedConfig.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0] if text_encoder_config.architectures else "Unknown"
    print(f"✓ Text encoder loaded successfully")
    print(f"  Model class: {model_class}")

    if model_class == "CLIPTextModel":
        print(f"  ✓ Compatible with CLIP (like SD-2.1)")
    else:
        print(f"  ⚠ Different text encoder: {model_class}")
        print(f"  → Need to modify import_model_class_from_model_name_or_path() in train.py")

except Exception as e:
    print(f"✗ Error loading text_encoder: {e}")
    print("  → Model may not have 'text_encoder' subfolder")

# 测试 2: 检查 VAE 兼容性
print("\n[2/5] Testing VAE compatibility...")
try:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="vae",
    )
    print(f"✓ VAE (AutoencoderKL) loaded successfully")
    print(f"  Config: {vae.config}")
except Exception as e:
    print(f"✗ Error loading VAE: {e}")

# 测试 3: 检查 UNet 兼容性
print("\n[3/5] Testing UNet compatibility...")
try:
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="unet",
    )
    print(f"✓ UNet2DConditionModel loaded successfully")
    print(f"  Sample size: {unet.config.sample_size}")
    print(f"  Attention resolution: {unet.config.attention_head_dim}")
except Exception as e:
    print(f"✗ Error loading UNet: {e}")

# 测试 4: 检查 tokenizer
print("\n[4/5] Testing tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="tokenizer",
        use_fast=False,
    )
    print(f"✓ Tokenizer loaded successfully")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Max length: {tokenizer.model_max_length}")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")

# 测试 5: 检查 scheduler
print("\n[5/5] Testing scheduler...")
try:
    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="scheduler",
    )
    print(f"✓ Scheduler loaded successfully")
    print(f"  Type: {scheduler.config._class_name}")
    print(f"  Timesteps: {scheduler.config.num_train_timesteps}")
except Exception as e:
    print(f"✗ Error loading scheduler: {e}")

# 测试 6: 尝试加载完整 pipeline
print("\n[6/6] Testing full pipeline...")
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    import torch

    # 创建一个简单的 ControlNet (仅测试兼容性)
    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=2)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    print(f"✓ StableDiffusionControlNetPipeline loaded successfully")
    print(f"  → Model is compatible with current code!")
except Exception as e:
    print(f"✗ Error loading pipeline: {e}")
    print(f"  → Cannot use StableDiffusionControlNetPipeline")

# 总结
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("If all tests passed with ✓, you can simply change MODEL_DIR")
print("If you see ⚠ or ✗, you need to modify the code accordingly")
