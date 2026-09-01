"""
批量生成 Kinect 风格空洞并保存

预处理流程：
1. 读取 noise_IQ 图像
2. 根据 amplitude 生成空洞
3. 保存带空洞的图像和空洞 mask

用法:
    python prepare_kinect_holes.py \
        --noise_iq_dir /path/to/noise_IQ \
        --ideal_iq_dir /path/to/ideal_IQ \
        --output_dir /path/to/noise_IQ_with_holes \
        --num_samples 100
"""

import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import time


def compute_amplitude_from_iq(iq_data):
    """从 IQ 数据计算 amplitude"""
    I = np.stack([iq_data[0], iq_data[2], iq_data[4]], axis=0)  # I0, I1, I2
    Q = np.stack([iq_data[1], iq_data[3], iq_data[5]], axis=0)  # Q0, Q1, Q2
    amplitude = np.sqrt(I**2 + Q**2)  # (3, H, W)
    amplitude_mean = amplitude.mean(axis=0)  # (H, W)
    return amplitude_mean


def _select_mask_to_target(mask, confidence, valid_mask, target_ratio):
    target_count = int(round(target_ratio * mask.size))
    if target_count <= 0:
        return np.zeros_like(mask, dtype=np.float32)

    flat_conf = confidence.reshape(-1)
    flat_mask = mask.reshape(-1) > 0.5
    flat_valid = valid_mask.reshape(-1) > 0.5

    current_indices = np.flatnonzero(flat_mask & flat_valid)
    target_count = min(target_count, int(flat_valid.sum()))

    if len(current_indices) > target_count:
        keep_order = np.argsort(flat_conf[current_indices], kind="stable")
        keep_indices = current_indices[keep_order[:target_count]]
        new_mask = np.zeros_like(flat_mask, dtype=bool)
        new_mask[keep_indices] = True
        return new_mask.reshape(mask.shape).astype(np.float32)

    if len(current_indices) < target_count:
        available_indices = np.flatnonzero(flat_valid & ~flat_mask)
        add_count = min(target_count - len(current_indices), len(available_indices))
        if add_count > 0:
            add_order = np.argsort(flat_conf[available_indices], kind="stable")
            add_indices = available_indices[add_order[:add_count]]
            flat_mask[add_indices] = True

    return flat_mask.reshape(mask.shape).astype(np.float32)


def generate_kinect_holes(depth, amplitude, hole_ratio=0.15, block_size=4,
                          amp_threshold=None, amp_percentile=5.0, low_amp_ratio=0.4):
    """
    基于 amplitude 生成空洞
    """
    H, W = depth.shape[:2]
    depth = depth.astype(np.float32)
    amplitude = amplitude.astype(np.float32)
    finite_depth = depth[np.isfinite(depth)]
    depth_is_meters = finite_depth.size > 0 and np.nanmax(finite_depth) < 100.0
    min_depth = 0.5 if depth_is_meters else 500.0
    max_depth = 6.0 if depth_is_meters else 6000.0
    valid_mask = (depth >= min_depth) & (depth <= max_depth) & np.isfinite(depth)

    valid_amp = amplitude[valid_mask & np.isfinite(amplitude)]
    if valid_amp.size == 0:
        return np.zeros_like(depth, dtype=np.float32), np.zeros_like(depth, dtype=np.float32)

    adaptive_threshold = np.percentile(valid_amp, amp_percentile)
    if amp_threshold is None:
        amp_threshold = adaptive_threshold

    # 计算置信度
    amp_high = np.percentile(valid_amp, 95.0)
    conf_amp = np.clip((amplitude - amp_threshold) / (amp_high - amp_threshold + 1e-8), 0, 1)
    conf_range = np.ones_like(depth)
    conf_range[depth < min_depth] = 0
    conf_range[depth > max_depth] = 0
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
            block_valid = valid_mask[i:i+block_size, j:j+block_size]
            block_amp_valid = block_amp[block_valid]

            if len(block_amp_valid) > 0:
                actual_low_ratio = (block_amp_valid < amp_threshold).mean()
                if actual_low_ratio > low_amp_ratio:
                    hole_mask[i:i+block_size, j:j+block_size] = 1

    # 形态学处理：平滑空洞边缘
    kernel = np.ones((3, 3), dtype=np.uint8)
    hole_mask = cv2.morphologyEx(hole_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
    hole_mask = hole_mask.astype(np.float32)
    hole_mask = _select_mask_to_target(hole_mask, confidence, valid_mask, hole_ratio)

    return hole_mask, confidence


def load_iq_sample(data_dir, scene, idx, sample_name):
    """加载 IQ 样本"""
    channels = []
    for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
        fpath = os.path.join(data_dir, scene, idx, f"{sample_name}_{ch}.npy")
        if os.path.exists(fpath):
            channels.append(np.load(fpath))
        else:
            return None
    return np.stack(channels, axis=0)


def main():
    parser = argparse.ArgumentParser(description='批量生成 Kinect 风格空洞')
    parser.add_argument('--noise_iq_dir', type=str, required=True,
                        help='noise_IQ 数据目录')
    parser.add_argument('--ideal_iq_dir', type=str, required=True,
                        help='ideal_IQ 数据目录（用于计算 amplitude 和深度）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录（带空洞的 noise_IQ）')
    parser.add_argument('--hole_mask_dir', type=str, default=None,
                        help='空洞 mask 输出目录（可选）')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='处理的样本数量，默认为全部')
    parser.add_argument('--scene', type=str, default=None,
                        help='只处理指定场景')
    parser.add_argument('--target_size', type=int, default=256,
                        help='目标图像尺寸')
    parser.add_argument('--hole_ratio', type=float, default=0.15,
                        help='目标空洞比例')
    parser.add_argument('--amp_threshold', type=float, default=None,
                        help='固定 amplitude 阈值；默认使用自适应百分位。')
    parser.add_argument('--amp_percentile', type=float, default=5.0,
                        help='自适应 amplitude 阈值的百分位。')
    parser.add_argument('--block_size', type=int, default=4,
                        help='局部 block 大小。')
    parser.add_argument('--low_amp_ratio', type=float, default=0.4,
                        help='block 内低 amplitude 像素比例阈值。')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    hole_mask_dir = args.hole_mask_dir or args.output_dir.replace('noise_IQ', 'hole_masks')
    os.makedirs(hole_mask_dir, exist_ok=True)

    # 查找所有样本
    print("收集样本列表...")
    samples = []
    scenes = [args.scene] if args.scene else os.listdir(args.noise_iq_dir)

    for scene in scenes:
        scene_path = os.path.join(args.noise_iq_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        for idx in os.listdir(scene_path):
            idx_path = os.path.join(scene_path, idx)
            if not os.path.isdir(idx_path):
                continue
            for f in os.listdir(idx_path):
                if f.endswith('_A.npy'):
                    sample_name = f.replace('_A.npy', '')
                    samples.append((scene, idx, sample_name))

    print(f"找到 {len(samples)} 个样本")
    if args.num_samples:
        np.random.seed(args.seed)
        indices = np.random.choice(len(samples), min(args.num_samples, len(samples)), replace=False)
        samples = [samples[i] for i in indices]
        print(f"随机选择 {len(samples)} 个样本")

    # 导入 DepthEstimator
    from depth_estimator import DepthEstimator
    print("初始化 DepthEstimator...")
    depth_estimator = DepthEstimator()

    # 处理每个样本
    hole_ratios = []
    for scene, idx, sample_name in tqdm(samples, desc="生成空洞"):
        # 加载 ideal IQ（用于计算 amplitude 和深度）
        ideal_iq = load_iq_sample(args.ideal_iq_dir, scene, idx, sample_name)
        if ideal_iq is None:
            print(f"  跳过 {scene}/{idx}/{sample_name}: 无法加载 ideal_IQ")
            continue

        # 加载 noise IQ
        noise_iq = load_iq_sample(args.noise_iq_dir, scene, idx, sample_name)
        if noise_iq is None:
            print(f"  跳过 {scene}/{idx}/{sample_name}: 无法加载 noise_IQ")
            continue

        H, W = ideal_iq.shape[1:]

        # 调整大小
        target_size = args.target_size
        if (H, W) != (target_size, target_size):
            ideal_iq_resized = np.zeros((6, target_size, target_size), dtype=np.float32)
            noise_iq_resized = np.zeros((6, target_size, target_size), dtype=np.float32)
            for i in range(6):
                ideal_iq_resized[i] = cv2.resize(ideal_iq[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                noise_iq_resized[i] = cv2.resize(noise_iq[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            ideal_iq = ideal_iq_resized
            noise_iq = noise_iq_resized

        # 计算 amplitude
        amplitude = compute_amplitude_from_iq(ideal_iq)

        # 估计深度
        depth = depth_estimator.process(ideal_iq)

        # 生成空洞
        hole_mask, confidence = generate_kinect_holes(
            depth, amplitude,
            hole_ratio=args.hole_ratio,
            block_size=args.block_size,
            amp_threshold=args.amp_threshold,
            amp_percentile=args.amp_percentile,
            low_amp_ratio=args.low_amp_ratio
        )

        hole_ratios.append(hole_mask.mean())

        # 应用空洞到 noise IQ
        hole_3d = np.stack([hole_mask] * 6, axis=0)
        noisy_with_holes = noise_iq.copy()
        noisy_with_holes[hole_3d > 0.5] = 0

        # 保存带空洞的 noise IQ
        output_scene_dir = os.path.join(args.output_dir, scene, idx)
        os.makedirs(output_scene_dir, exist_ok=True)
        for ch_idx in range(6):
            ch_name = ['A', 'B', 'C', 'D', 'E', 'F'][ch_idx]
            output_path = os.path.join(output_scene_dir, f"{sample_name}_{ch_name}.npy")
            np.save(output_path, noisy_with_holes[ch_idx])

        # 保存空洞 mask
        mask_scene_dir = os.path.join(hole_mask_dir, scene, idx)
        os.makedirs(mask_scene_dir, exist_ok=True)
        mask_path = os.path.join(mask_scene_dir, f"{sample_name}.npy")
        np.save(mask_path, hole_mask)

    print(f"\n完成！处理了 {len(hole_ratios)} 个样本")
    print(f"平均空洞比例: {np.mean(hole_ratios):.4f} (+/- {np.std(hole_ratios):.4f})")
    print(f"输出目录: {args.output_dir}")
    print(f"空洞 mask 目录: {hole_mask_dir}")


if __name__ == "__main__":
    main()
