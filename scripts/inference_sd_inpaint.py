"""
用 StableDiffusionInpaint 直接填补 IQ 数据的空洞
不需要训练，直接用预训练模型

使用方法:
    python inference_sd_inpaint.py --noise_IQ_file /path/to/noise_IQ.npy \
                                   --conf_file /path/to/conf.npy \
                                   --out_file /path/to/output.npy
"""
import cv2
import os
import torch
import argparse
import numpy as np
from glob import glob

from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise_IQ_file",
        type=str,
        default=None,
        help="Path to noise IQ .npy file (shape: 6 x H x W)"
    )
    parser.add_argument(
        "--conf_file",
        type=str,
        default=None,
        help="Path to confidence .npy file (shape: H x W)"
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Output path for filled IQ .npy file"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[240, 320],
        help="Target size (height, width)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    return args


def iq_to_rgb_triplet(channel_data, conf_mask=None):
    """单通道复制 3 份变成 RGB

    Args:
        channel_data: shape (H, W), 单通道 IQ 数据
        conf_mask: shape (H, W), 置信度图，用于归一化

    Returns:
        rgb: shape (H, W, 3), RGB 图片, 值范围 [0, 1]
    """
    if conf_mask is not None:
        valid_mask = conf_mask >= 0.5
        if valid_mask.sum() > 0:
            ch_min = channel_data[valid_mask].min()
            ch_max = channel_data[valid_mask].max()
        else:
            ch_min, ch_max = channel_data.min(), channel_data.max()
    else:
        ch_min, ch_max = channel_data.min(), channel_data.max()

    ch_norm = ((channel_data - ch_min) / (ch_max - ch_min + 1e-8)).astype(np.float32)

    # 空洞区域设为 0
    if conf_mask is not None:
        ch_norm = ch_norm * (conf_mask >= 0.5).astype(np.float32)

    rgb = np.stack([ch_norm, ch_norm, ch_norm], axis=-1)
    return rgb


def create_mask(conf, resolution=None):
    """创建 mask (0=保留, 1=空洞)

    Args:
        conf: 置信度图, shape (H, W)
        resolution: 目标分辨率，如果为 None 则不 resize

    Returns:
        mask: 二值 mask, shape (H, W) 或 (resolution, resolution)
    """
    if resolution is not None and conf.shape[0] != resolution:
        mask = cv2.resize(conf, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    else:
        mask = conf.copy()

    mask = (mask < 0.5).astype(np.float32)
    return mask


def fill_channel_with_sd_inpaint(rgb_image, mask, pipe, seed=42):
    """用 SD Inpainting 填充单个通道

    Args:
        rgb_image: shape (H, W, 3), RGB 图片, 值范围 [0, 1]
        mask: shape (H, W), 二值 mask (1=空洞)
        pipe: StableDiffusionInpaintPipeline
        seed: 随机种子

    Returns:
        filled: shape (H, W, 3), 填充后的 RGB 图片, 值范围 [0, 1]
    """
    # 转为 PIL Image
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(rgb_uint8)

    # mask: 白色=空洞 (SD 期望), 黑色=保留
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_uint8, mode='L')

    # 生成
    generator = torch.manual_seed(seed)
    result = pipe(
        prompt="",  # 空 prompt
        image=pil_image,
        mask_image=pil_mask,
        num_inference_steps=20,
        guidance_scale=1.0,  # =1 最有效率
        generator=generator,
    ).images[0]

    # 转回 numpy
    filled = np.array(result).astype(np.float32) / 255.0
    return filled


def load_pbrt_sample(noise_path, conf_path, idx):
    """加载 PBRT 样本

    Args:
        noise_path: noise_IQ 文件目录, e.g., "/path/to/noise_IQ_masked/bathroom/1"
        conf_path: confidence 文件目录, e.g., "/path/to/confidence_masked/bathroom/1"
        idx: 样本 ID, e.g., "100"

    Returns:
        noise_IQ: shape (6, H, W)
        conf: shape (H, W)
    """
    import glob as glob_module

    npy_files = sorted(glob_module.glob(f"{noise_path}/{idx}_*.npy"))
    noise_data = []
    for f in npy_files:
        data = np.load(f)
        noise_data.append(data)
    noise_IQ = np.stack(noise_data, axis=0)  # (6, H, W)

    conf = np.load(f"{conf_path}/{idx}.npy")

    return noise_IQ, conf


def main():
    args = parse_args()

    # 从文件路径解析目录和索引
    # 假设 noise_IQ_file 格式: /path/to/.../{scene}/1/{idx}_A.npy
    noise_dir = os.path.dirname(args.noise_IQ_file)
    conf_dir = os.path.dirname(args.conf_file)

    # 解析 idx (去掉 _A.npy 后缀)
    basename = os.path.basename(args.noise_IQ_file)
    idx = basename.split('_')[0]

    print(f"Loading sample {idx} from {noise_dir}")

    # 加载数据
    noise_IQ, conf = load_pbrt_sample(noise_dir, conf_dir, idx)

    print(f"IQ shape: {noise_IQ.shape}, dtype: {noise_IQ.dtype}")
    print(f"Conf shape: {conf.shape}, dtype: {conf.dtype}")
    print(f"Conf range: [{conf.min():.4f}, {conf.max():.4f}]")

    target_h, target_w = args.target_size

    # 推理尺寸 (SD 需要是 8 的倍数)
    infer_size = 512
    if target_h > 256 or target_w > 256:
        infer_size = 512
    else:
        infer_size = 256

    # 加载 SD Inpainting 模型
    print("Loading StableDiffusionInpaintPipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    # 创建 mask
    mask = create_mask(conf, resolution=infer_size)
    print(f"Mask coverage: {mask.mean():.2%}")

    # 填充每个通道
    filled_IQs = np.zeros((6, infer_size, infer_size), dtype=np.float32)

    for i in range(6):
        print(f"Filling channel {i}...")

        # resize 到推理尺寸
        if noise_IQ.shape[1] != infer_size or noise_IQ.shape[2] != infer_size:
            ch_resized = cv2.resize(noise_IQ[i], (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
            conf_resized = cv2.resize(conf, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        else:
            ch_resized = noise_IQ[i]
            conf_resized = conf

        # IQ -> RGB
        rgb = iq_to_rgb_triplet(ch_resized, conf_resized)

        # SD Inpaint 填充
        rgb_filled = fill_channel_with_sd_inpaint(rgb, mask, pipe, seed=args.seed)

        # 取灰度 (因为三通道一样)
        filled_IQs[i] = rgb_filled[:, :, 0]

    # resize 回目标尺寸
    if target_h != infer_size or target_w != infer_size:
        output_IQs = np.zeros((6, target_h, target_w), dtype=np.float32)
        for i in range(6):
            output_IQs[i] = cv2.resize(filled_IQs[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:
        output_IQs = filled_IQs

    # 保存
    np.save(args.out_file, output_IQs)
    print(f"Saved to {args.out_file}")

    # 可视化对比
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))

        for i in range(6):
            # 原始
            orig_ch = noise_IQ[i]
            if orig_ch.shape[0] != target_h or orig_ch.shape[1] != target_w:
                orig_ch = cv2.resize(orig_ch, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            axes[0, i].imshow(orig_ch, cmap='gray')
            axes[0, i].set_title(f'Ch{i} Input')
            axes[0, i].axis('off')

            # 填充后
            axes[1, i].imshow(output_IQs[i], cmap='gray')
            axes[1, i].set_title(f'Ch{i} Filled')
            axes[1, i].axis('off')

        axes[0, 3].set_title('Input (with holes)', fontsize=14)
        axes[1, 3].set_title('SD Inpaint Filled', fontsize=14)

        plt.tight_layout()
        out_png = args.out_file.replace('.npy', '_comparison.png')
        plt.savefig(out_png, dpi=150)
        print(f"Saved comparison to {out_png}")
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("Done!")


if __name__ == "__main__":
    main()