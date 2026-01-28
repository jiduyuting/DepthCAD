#!/usr/bin/env python3
"""
测试多个模型与 DepthCAD 的兼容性
评估直接替换的可行性和难易程度
"""
import sys
import os

# 设置 HuggingFace 镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 要测试的模型列表
MODELS_TO_TEST = [
    "lxq007/HYPIR",
    "camenduru/SUPIR",
    "TurboDiffusion/TurboWan2.2-I2V-A14B-720P",
    "Tongyi-MAI/Z-Image-Turbo",
]

print("=" * 100)
print("DepthCAD 模型兼容性测试")
print("=" * 100)
print(f"\n待测试模型：")
for i, model in enumerate(MODELS_TO_TEST, 1):
    print(f"  {i}. {model}")

results = []

for model_id in MODELS_TO_TEST:
    print(f"\n{'=' * 100}")
    print(f"[{MODELS_TO_TEST.index(model_id) + 1}/{len(MODELS_TO_TEST)}] 测试模型: {model_id}")
    print('=' * 100)

    result = {
        "model": model_id,
        "compatible": False,
        "issues": [],
        "modifications_needed": [],
        "difficulty": "未知",
        "architecture": "未知",
    }

    # 测试 1: 尝试确定模型类型
    print("\n[1/6] 检测模型架构类型...")
    try:
        from transformers import PretrainedConfig
        try:
            # 检查是否有 model_index.json (diffusers 模型的标志)
            from huggingface_hub import hf_hub_download
            idx_file = hf_hub_download(repo_id=model_id, filename="model_index.json")
            print(f"  ✓ 检测到 Diffusers 格式模型")

            import json
            with open(idx_file) as f:
                index = json.load(f)
            print(f"  模型类别: {index.get('_class_name', 'Unknown')}")
            result["architecture"] = index.get('_class_name', 'Unknown')

            if "StableDiffusion" in index.get('_class_name', ''):
                print(f"  ✓ 基于 Stable Diffusion 架构")
                result["architecture"] = "Stable Diffusion"
            else:
                print(f"  ⚠ 非 Stable Diffusion 架构")
                result["issues"].append(f"模型类型: {index.get('_class_name', 'Unknown')}")

        except Exception as e:
            print(f"  ⚠ 不是标准 Diffusers 格式: {str(e)[:50]}")
            result["issues"].append("非标准 Diffusers 格式")

            # 尝试检查是否有其他配置文件
            try:
                config_file = hf_hub_download(repo_id=model_id, filename="config.json")
                print(f"  → 有 config.json，可能是自定义模型")
                result["architecture"] = "自定义模型"
            except:
                result["architecture"] = "未知架构"

    except Exception as e:
        print(f"  ✗ 无法检测模型类型: {e}")
        result["issues"].append("无法检测模型类型")

    # 测试 2: 检查 text_encoder
    print("\n[2/6] 测试 text_encoder 兼容性...")
    try:
        from transformers import PretrainedConfig
        text_encoder_config = PretrainedConfig.from_pretrained(
            model_id,
            subfolder="text_encoder",
        )
        model_class = text_encoder_config.architectures[0] if text_encoder_config.architectures else "Unknown"
        print(f"  ✓ text_encoder 存在")
        print(f"  类型: {model_class}")

        if model_class == "CLIPTextModel":
            print(f"  ✓ 使用 CLIP (与 SD-2.1 兼容)")
        else:
            print(f"  ⚠ 使用不同的 encoder: {model_class}")
            result["issues"].append(f"text_encoder 类型: {model_class}")
            result["modifications_needed"].append("修改 import_model_class_from_model_name_or_path 函数")

    except Exception as e:
        print(f"  ✗ 无法加载 text_encoder: {str(e)[:60]}")
        result["issues"].append("缺少 text_encoder 或结构不同")
        result["modifications_needed"].append("重写 text_encoder 加载逻辑")

    # 测试 3: 检查 VAE
    print("\n[3/6] 测试 VAE 兼容性...")
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
        )
        print(f"  ✓ VAE (AutoencoderKL) 兼容")
        print(f"  潜空间通道数: {vae.config.latent_channels}")
    except Exception as e:
        print(f"  ✗ 无法加载标准 VAE: {str(e)[:60]}")
        result["issues"].append("VAE 不兼容")
        result["modifications_needed"].append("修改 VAE 加载逻辑")

    # 测试 4: 检查 UNet
    print("\n[4/6] 测试 UNet 兼容性...")
    try:
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
        )
        print(f"  ✓ UNet2DConditionModel 兼容")
        print(f"  输入通道数: {unet.config.in_channels}")
        print(f"  采样尺寸: {unet.config.sample_size}")
    except Exception as e:
        print(f"  ✗ 无法加载标准 UNet: {str(e)[:60]}")
        result["issues"].append("UNet 不兼容")
        result["modifications_needed"].append("修改 UNet 加载逻辑")

    # 测试 5: 检查 scheduler
    print("\n[5/6] 测试 scheduler...")
    try:
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
        )
        print(f"  ✓ Scheduler 兼容")
    except Exception as e:
        print(f"  ⚠ 无法加载 scheduler: {str(e)[:60]}")
        result["issues"].append("scheduler 可能需要调整")

    # 测试 6: 尝试加载 ControlNet Pipeline
    print("\n[6/6] 测试 ControlNet Pipeline 兼容性...")
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import torch

        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=2)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        print(f"  ✓✓✓ StableDiffusionControlNetPipeline 成功加载!")
        print(f"  → 模型与当前代码完全兼容!")
        result["compatible"] = True
        result["difficulty"] = "简单 - 只需修改 MODEL_DIR"

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ 无法加载 ControlNet Pipeline")
        print(f"  错误: {error_msg[:80]}")

        # 根据错误类型判断难度
        if "text_encoder" in error_msg.lower():
            result["modifications_needed"].append("重写 text_encoder 加载逻辑")
        if "vae" in error_msg.lower():
            result["modifications_needed"].append("重写 VAE 加载逻辑")
        if "unet" in error_msg.lower():
            result["modifications_needed"].append("重写 UNet 加载逻辑")

    # 评估难度
    if not result["compatible"]:
        issue_count = len(result["issues"])
        if issue_count <= 1:
            result["difficulty"] = "中等 - 需要小幅修改"
        elif issue_count <= 3:
            result["difficulty"] = "困难 - 需要大量修改"
        else:
            result["difficulty"] = "极难 - 需要重构代码"

    results.append(result)

# 输出总结报告
print("\n" + "=" * 100)
print("兼容性测试总结报告")
print("=" * 100)

for i, result in enumerate(results, 1):
    print(f"\n{'=' * 100}")
    print(f"模型 {i}: {result['model']}")
    print('=' * 100)
    print(f"架构类型:     {result['architecture']}")
    print(f"兼容性:       {'✓ 完全兼容' if result['compatible'] else '✗ 不兼容'}")
    print(f"替换难度:     {result['difficulty']}")

    if result['issues']:
        print(f"\n发现的问题:")
        for issue in result['issues']:
            print(f"  • {issue}")

    if result['modifications_needed']:
        print(f"\n需要的修改:")
        for mod in set(result['modifications_needed']):  # 去重
            print(f"  • {mod}")

    if result['compatible']:
        print(f"\n✓ 这个模型可以直接替换，只需修改 MODEL_DIR")
    else:
        print(f"\n✗ 这个模型需要代码修改才能使用")

# 最终推荐
print("\n" + "=" * 100)
print("推荐建议")
print("=" * 100)

compatible_models = [r for r in results if r['compatible']]
if compatible_models:
    print("\n✓ 推荐使用以下模型（可直接替换）:")
    for r in compatible_models:
        print(f"  • {r['model']}")
else:
    print("\n⚠ 没有找到完全兼容的模型")
    print("\n如果你需要更换模型，建议:")
    print("  1. 选择问题最少的模型")
    print("  2. 或者继续使用当前的 stable-diffusion-2-1")

print("\n" + "=" * 100)
