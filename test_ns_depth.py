"""
测试 OpenCV NS Inpainting + ControlNet 的完整流程

流程:
  noise_IQ[6] → NS inpaint 填补空洞 → ControlNet(2ch) → 6通道IQ → depth

对比:
  1. 原始 ControlNet (无 inpaint)
  2. ControlNet + NS inpaint (每通道填补)

Usage:
    python test_ns_depth.py
"""

import argparse
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from depth_estimator import DepthEstimatorTorch


def ns_inpaint_channel(img, mask, inpaintRadius=5):
    """对单通道图像进行 NS inpaint填补"""
    # img: (H, W) float, mask: (H, W) float (1=空洞, 0=有效)
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    filled = cv2.inpaint(img_uint8, mask_uint8, inpaintRadius=inpaintRadius, flags=cv2.INPAINT_NS)
    return filled.astype(np.float32) / 255.0


def inference_with_ns(pipe, noise, conf, scale, target_size=(240, 320), device="cuda"):
    """
    ControlNet 推理 (加入 NS inpaint 填补)
    noise: (6, H, W)
    conf: (H, W)
    """
    infer_h, infer_w = 512, 512

    # 空洞掩码: 1=空洞, 0=有效
    hole_mask = (conf < 0.5).astype(np.float32)

    # 1. NS inpaint 填补每通道 (处理在 512x512)
    noise_inpainted_512 = np.zeros((6, infer_h, infer_w), dtype=np.float32)
    for i in range(6):
        # 先 resize 到 512
        noise_i = cv2.resize(noise[i], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        hole_i = cv2.resize(hole_mask, (infer_w, infer_h), interpolation=cv2.INTER_NEAREST)
        # NS 填补
        noise_i_filled = ns_inpaint_channel(noise_i, hole_i, inpaintRadius=5)
        noise_inpainted_512[i] = noise_i_filled

    # 2. ControlNet 推理 (用填补后的 noise)
    pred_IQs_infer = np.zeros((6, infer_h, infer_w))
    conf_resized = cv2.resize(conf, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

    for i in range(6):
        noise_resized = noise_inpainted_512[i]  # 已经是 512x512，填补后的
        guidance = np.stack([noise_resized, conf_resized], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0)

        prompt = ""
        generator = torch.manual_seed(42)

        pred_IQ = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=guidance,
            height=infer_h,
            width=infer_w
        ).images[0]

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0
        pred_IQ = 2 * pred_IQ - 1
        pred_IQs_infer[i] = pred_IQ * scale

    # 3. Resize 回目标尺寸
    target_h, target_w = target_size
    reshaped_IQs = np.zeros((6, target_h, target_w), dtype=np.float32)
    for i in range(6):
        reshaped_IQs[i, :, :] = cv2.resize(
            pred_IQs_infer[i, :, :],
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    return reshaped_IQs


def inference_without_ns(pipe, noise, conf, scale, target_size=(240, 320), device="cuda"):
    """ControlNet 推理 (原始，无 inpaint)"""
    infer_h, infer_w = 512, 512

    pred_IQs_infer = np.zeros((6, infer_h, infer_w))
    conf_resized = cv2.resize(conf, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

    for i in range(6):
        noise_resized = cv2.resize(noise[i], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        guidance = np.stack([noise_resized, conf_resized], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0)

        prompt = ""
        generator = torch.manual_seed(42)

        pred_IQ = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=guidance,
            height=infer_h,
            width=infer_w
        ).images[0]

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0
        pred_IQ = 2 * pred_IQ - 1
        pred_IQs_infer[i] = pred_IQ * scale

    target_h, target_w = target_size
    reshaped_IQs = np.zeros((6, target_h, target_w), dtype=np.float32)
    for i in range(6):
        reshaped_IQs[i, :, :] = cv2.resize(
            pred_IQs_infer[i, :, :],
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    return reshaped_IQs


def depth_from_iqs(iqs, target_size, device):
    estimator = DepthEstimatorTorch(device=device)
    iqs_tensor = torch.from_numpy(iqs).unsqueeze(0).to(device)
    depth = estimator.process(iqs_tensor).squeeze(0).cpu().numpy()
    return depth


def main():
    # ============ 参数 ============
    MODEL_DIR = "/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
    DEPTHCAD_PATH = "/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_4_16_masked/checkpoint-15000/depthcad"
    TARGET_SIZE = (240, 320)

    noise_file = "/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/noise_IQ_masked/bathroom/1/100_A.npy"
    noise_depth_file = "/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/confidence_masked/bathroom/1/100.npy"
    gt_depth_file = "/data/pre_student/GJ/DepthCAD/output/visualization/bathroom/1/100/gt_depth.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ 加载数据 ============
    print("Loading data...")
    from pbrt_dataset.preprocess import load_raw as load_raw_pbrt, compute_gradient_confidence

    noise_result = load_raw_pbrt(noise_file, target_size=TARGET_SIZE, sqrt_in=True)
    if isinstance(noise_result, (tuple, list)):
        noise = noise_result[0]
    else:
        noise = noise_result

    noise_depth = np.load(noise_depth_file)
    noise_depth = cv2.resize(noise_depth.astype(np.float32), TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    confidence = compute_gradient_confidence(noise_depth)

    scale = max(noise.max(), abs(noise.min()), 1e-8)
    noise /= scale

    gt_depth = np.load(gt_depth_file)

    # ============ 加载模型 ============
    print("Loading ControlNet...")
    depthcad = ControlNetModel.from_pretrained(DEPTHCAD_PATH, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_DIR, controlnet=depthcad, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.to(device)
    print("ControlNet loaded!")

    # ============ 推理 ============
    print("\n=== Inference without NS inpaint ===")
    iqs_no_ns = inference_without_ns(pipe, noise, confidence, scale, TARGET_SIZE, device)
    depth_no_ns = depth_from_iqs(iqs_no_ns, TARGET_SIZE, device)
    rmse_no_ns = np.sqrt(((depth_no_ns - gt_depth)**2).mean())

    print(f"Depth range: [{depth_no_ns.min():.4f}, {depth_no_ns.max():.4f}]")
    print(f"RMSE vs GT: {rmse_no_ns:.4f}")

    print("\n=== Inference with NS inpaint ===")
    iqs_with_ns = inference_with_ns(pipe, noise, confidence, scale, TARGET_SIZE, device)
    depth_with_ns = depth_from_iqs(iqs_with_ns, TARGET_SIZE, device)
    rmse_with_ns = np.sqrt(((depth_with_ns - gt_depth)**2).mean())

    print(f"Depth range: [{depth_with_ns.min():.4f}, {depth_with_ns.max():.4f}]")
    print(f"RMSE vs GT: {rmse_with_ns:.4f}")

    print(f"\n=== 对比 ===")
    print(f"无 NS: RMSE = {rmse_no_ns:.4f}")
    print(f"有 NS: RMSE = {rmse_with_ns:.4f}")
    print(f"改善: {(rmse_no_ns - rmse_with_ns) / rmse_no_ns * 100:.1f}%")

    # ============ 分析 ============
    conf_resized = cv2.resize(confidence, (TARGET_SIZE[1], TARGET_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    low_conf = conf_resized < 0.5
    high_conf = ~low_conf

    if low_conf.sum() > 0:
        rmse_no_ns_low = np.sqrt(((depth_no_ns[low_conf] - gt_depth[low_conf])**2).mean())
        rmse_with_ns_low = np.sqrt(((depth_with_ns[low_conf] - gt_depth[low_conf])**2).mean())
        print(f"\n低置信区域 RMSE:")
        print(f"  无 NS: {rmse_no_ns_low:.4f}")
        print(f"  有 NS: {rmse_with_ns_low:.4f}")

    if high_conf.sum() > 0:
        rmse_no_ns_high = np.sqrt(((depth_no_ns[high_conf] - gt_depth[high_conf])**2).mean())
        rmse_with_ns_high = np.sqrt(((depth_with_ns[high_conf] - gt_depth[high_conf])**2).mean())
        print(f"\n高置信区域 RMSE:")
        print(f"  无 NS: {rmse_no_ns_high:.4f}")
        print(f"  有 NS: {rmse_with_ns_high:.4f}")

    # ============ 可视化 ============
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(gt_depth, cmap='turbo')
    axes[0, 0].set_title('GT Depth')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(depth_no_ns, cmap='turbo')
    axes[0, 1].set_title(f'Without NS\nRMSE={rmse_no_ns:.4f}')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(depth_with_ns, cmap='turbo')
    axes[0, 2].set_title(f'With NS\nRMSE={rmse_with_ns:.4f}')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(np.abs(depth_with_ns - gt_depth), cmap='hot')
    axes[0, 3].set_title('|With NS - GT|')
    axes[0, 3].axis('off')

    axes[1, 0].imshow(conf_resized, cmap='gray')
    axes[1, 0].set_title('Confidence')
    axes[1, 0].axis('off')

    # 空洞区域的 IQ 对比
    hole_mask = (conf_resized < 0.5).astype(np.float32)
    # 用 NS 填补空洞
    iq0_ns = cv2.resize(iqs_with_ns[0], (TARGET_SIZE[1], TARGET_SIZE[0]))
    iq0_no_ns = cv2.resize(iqs_no_ns[0], (TARGET_SIZE[1], TARGET_SIZE[0]))
    axes[1, 1].imshow(iq0_no_ns * hole_mask, cmap='gray')
    axes[1, 1].set_title('IQ[0] no NS\n(hole region)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(iq0_ns * hole_mask, cmap='gray')
    axes[1, 2].set_title('IQ[0] with NS\n(hole region)')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(np.abs(iq0_ns - iqs_with_ns[0].mean()) * hole_mask, cmap='hot')
    axes[1, 3].set_title('IQ NS diff\n(hole region)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    out_png = '/data/pre_student/GJ/DepthCAD/output/ns_depth_comparison.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n可视化保存到: {out_png}")


if __name__ == '__main__':
    main()
