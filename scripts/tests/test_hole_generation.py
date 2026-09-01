"""
批量测试 Kinect 风格空洞生成效果

基于 amplitude 生成空洞：当某一局部区域的大部分像素 amplitude 小于阈值时，该区域设为空洞。
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.insert(0, '/data/pre_student/GJ/DepthCAD')
from depth_estimator import DepthEstimator


def compute_amplitude_from_iq(iq_data):
    """
    从 IQ 数据计算 amplitude
    6通道格式: [I0, Q0, I1, Q1, I2, Q2]
    """
    I = np.stack([iq_data[0], iq_data[2], iq_data[4]], axis=0)  # I0, I1, I2
    Q = np.stack([iq_data[1], iq_data[3], iq_data[5]], axis=0)  # Q0, Q1, Q2
    amplitude = np.sqrt(I**2 + Q**2)  # (3, H, W)
    amplitude_mean = amplitude.mean(axis=0)  # (H, W)
    return amplitude_mean


def generate_kinect_holes(depth, amplitude, hole_ratio=0.15, block_size=8, amp_threshold=0.1, low_amp_ratio=0.6):
    """
    基于 amplitude 生成空洞

    Args:
        depth: 深度图 (H, W) in mm
        amplitude: amplitude 图 (H, W)
        hole_ratio: 目标空洞比例
        block_size: 局部区域大小
        amp_threshold: amplitude 阈值
        low_amp_ratio: block 内超过该比例的像素 amplitude < 阈值时，整个 block 设为空洞
    """
    H, W = depth.shape[:2]
    depth = depth.astype(np.float32)
    amplitude = amplitude.astype(np.float32)

    # 计算置信度
    conf_amp = np.clip(amplitude / (amp_threshold + 1e-8), 0, 1)
    conf_range = np.ones_like(depth)
    conf_range[depth < 500] = 0
    conf_range[depth > 6000] = 0
    valid_mask = (depth >= 500) & (depth <= 6000)
    conf_range[valid_mask] = 1.0

    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)
    conf_edge = 1 - np.clip(grad_norm * 3, 0, 1)

    confidence = conf_amp * conf_range * conf_edge

    # 生成空洞
    hole_mask = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block_amp = amplitude[i:i+block_size, j:j+block_size]
            block_amp_valid = block_amp[depth[i:i+block_size, j:j+block_size] > 0]

            if len(block_amp_valid) > 0:
                # 只要 block 内超过 low_amp_ratio 的像素 amplitude 小，就设为空洞
                actual_low_ratio = (block_amp_valid < amp_threshold).mean()
                if actual_low_ratio > low_amp_ratio:
                    hole_mask[i:i+block_size, j:j+block_size] = 1

    current_ratio = hole_mask.mean()
    print(f"    Initial hole ratio: {current_ratio:.4f}, target: {hole_ratio:.4f}")

    # 如果空洞太少，降低阈值
    if current_ratio < hole_ratio * 0.5:
        amp_threshold_lower = amp_threshold * 0.5
        for i in range(0, H - block_size + 1, block_size):
            for j in range(0, W - block_size + 1, block_size):
                block_amp = amplitude[i:i+block_size, j:j+block_size]
                block_amp_valid = block_amp[depth[i:i+block_size, j:j+block_size] > 0]

                if len(block_amp_valid) > 0:
                    actual_low_ratio = (block_amp_valid < amp_threshold_lower).mean()
                    if actual_low_ratio > low_amp_ratio * 0.8:
                        hole_mask[i:i+block_size, j:j+block_size] = 1

    # 如果空洞太多，随机减少
    current_ratio = hole_mask.mean()
    if current_ratio > hole_ratio * 1.5:
        mask_flat = hole_mask.flatten()
        n_to_remove = int(len(mask_flat) * (current_ratio - hole_ratio) * 0.5)
        zero_indices = np.where(mask_flat == 0)[0]
        if len(zero_indices) > n_to_remove:
            remove_indices = np.random.choice(zero_indices, n_to_remove, replace=False)
            mask_flat[remove_indices] = 1
            hole_mask = mask_flat.reshape((H, W))

    hole_mask = np.clip(hole_mask, 0, 1)

    # 形态学处理：平滑空洞边缘
    kernel = np.ones((3, 3), dtype=np.uint8)
    hole_mask = cv2.morphologyEx(hole_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
    hole_mask = hole_mask.astype(np.float32)

    return hole_mask, confidence


def process_and_visualize(ideal_dir, output_dir, num_samples=10):
    """批量处理并可视化"""
    os.makedirs(output_dir, exist_ok=True)

    # 找到所有样本
    # 目录结构: ideal_dir/scene_idx/sampleName_A.npy
    samples = []
    for scene_dir in os.listdir(ideal_dir):
        scene_path = os.path.join(ideal_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        for f in os.listdir(scene_path):
            if f.endswith('_A.npy'):
                sample_name = f.replace('_A.npy', '')
                # scene_dir 就是 scene (如 "0", "1")
                samples.append((scene_dir, sample_name))

    print(f"Found {len(samples)} samples, processing {min(num_samples, len(samples))}...")

    # 随机选择样本
    np.random.seed(42)
    selected = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)

    for cnt, idx in enumerate(selected):
        scene, sample_name = samples[idx]
        print(f"\n[{cnt+1}/{len(selected)}] Processing {scene}/{sample_name}")

        # 加载 IQ 数据
        channels = []
        for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
            fpath = os.path.join(ideal_dir, scene, f"{sample_name}_{ch}.npy")
            if os.path.exists(fpath):
                channels.append(np.load(fpath))
            else:
                print(f"    File not found: {fpath}")
                continue

        if len(channels) != 6:
            print(f"    Skipping, only {len(channels)} channels found")
            continue

        iq_data = np.stack(channels, axis=0)  # (6, H, W)
        H, W = iq_data.shape[1:]

        # 调整大小到 512x512
        target_size = 512
        if (H, W) != (target_size, target_size):
            iq_data_resized = np.zeros((6, target_size, target_size), dtype=np.float32)
            for i in range(6):
                iq_data_resized[i] = cv2.resize(iq_data[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            iq_data = iq_data_resized
            H, W = target_size, target_size

        # 计算 amplitude
        amplitude = compute_amplitude_from_iq(iq_data)
        print(f"    Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}]")

        # 估计深度 (DepthEstimator 返回米，转换为毫米)
        if 'estimator' not in globals():
            global estimator
            estimator = DepthEstimator()
        depth_est = estimator.process(iq_data) * 1000
        print(f"    Depth range: [{depth_est.min():.4f}, {depth_est.max():.4f}] mm")

        # 生成空洞 (减小 block_size 使边缘更平滑)
        hole_mask, confidence = generate_kinect_holes(depth_est, amplitude, hole_ratio=0.15, block_size=4, amp_threshold=0.3, low_amp_ratio=0.4)
        print(f"    Hole ratio: {hole_mask.mean():.4f}")
        print(f"    Confidence range: [{confidence.min():.4f}, {confidence.max():.4f}]")

        # 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # IQ A 通道
        axes[0, 0].imshow(iq_data[0], cmap='gray')
        axes[0, 0].set_title('IQ Channel A')
        axes[0, 0].axis('off')

        # Amplitude
        im1 = axes[0, 1].imshow(amplitude, cmap='gray')
        axes[0, 1].set_title('Amplitude')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        # Depth
        im2 = axes[0, 2].imshow(depth_est, cmap='turbo')
        axes[0, 2].set_title('Estimated Depth')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

        # Confidence
        im3 = axes[0, 3].imshow(confidence, cmap='gray')
        axes[0, 3].set_title('Confidence')
        axes[0, 3].axis('off')
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

        # Hole Mask
        axes[1, 0].imshow(hole_mask, cmap='gray')
        axes[1, 0].set_title(f'Hole Mask ({hole_mask.mean():.2%})')
        axes[1, 0].axis('off')

        # Amplitude with hole overlay
        axes[1, 1].imshow(amplitude, cmap='gray')
        axes[1, 1].imshow(hole_mask, cmap='Reds', alpha=0.5)
        axes[1, 1].set_title('Amplitude + Holes')
        axes[1, 1].axis('off')

        # Depth with hole overlay
        axes[1, 2].imshow(depth_est, cmap='turbo')
        axes[1, 2].imshow(hole_mask, cmap='Reds', alpha=0.5)
        axes[1, 2].set_title('Depth + Holes')
        axes[1, 2].axis('off')

        # Confidence with hole overlay
        axes[1, 3].imshow(confidence, cmap='gray')
        axes[1, 3].imshow(hole_mask, cmap='Reds', alpha=0.5)
        axes[1, 3].set_title('Confidence + Holes')
        axes[1, 3].axis('off')

        plt.tight_layout()
        out_png = os.path.join(output_dir, f'holes_{scene}_{sample_name}.png')
        plt.savefig(out_png, dpi=150)
        print(f"    Saved to {out_png}")
        plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ideal_dir", type=str, default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/ideal_IQ_masked/bathroom")
    parser.add_argument("--output_dir", type=str, default="./hole_test_output")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    process_and_visualize(args.ideal_dir, args.output_dir, args.num_samples)