import torch

# 替换为你实际的文件路径
ckpt_path = "/data/pre_student/GJ/DepthCAD/HYPIR_sd2.pth"

try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # 有些 checkpoint 会把权重放在 "state_dict" 键下，有些直接就是字典
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    keys = list(state_dict.keys())
    print(f"文件名: {ckpt_path}")
    print(f"包含键值数量: {len(keys)}")
    print("前 10 个键名 (Key Names):")
    for k in keys[:10]:
        print(f"  - {k}")

    # 智能判断逻辑
    has_text_encoder = any("cond_stage_model" in k for k in keys)
    has_vae = any("first_stage_model" in k for k in keys)
    has_unet = any("model.diffusion_model" in k for k in keys)

    print("\n[--- 诊断结果 ---]")
    if has_unet and not has_text_encoder and not has_vae:
        print("结论：这看起来是一个【仅包含 UNet】的权重文件。")
        print("建议：你需要把它'插入'到一个标准的 SD 2.1 框架中，而不是直接转换。")
    elif has_unet and has_text_encoder and has_vae:
        print("结论：这是一个【完整】的 Checkpoint 文件。")
        print("建议：可以使用通用的转换脚本。")
    else:
        print("结论：结构特殊，可能是非标准命名或特定组件。")

except Exception as e:
    print(f"读取失败: {e}")