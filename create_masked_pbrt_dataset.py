"""
为 PBRT 数据集生成带空洞的 masked 版本

生成:
- ideal_IQ_masked: GT IQ (6通道) - 保持不变
- noise_IQ_masked: 带空洞的 IQ 输入 (空洞区域置0)
- confidence_masked: 置信度图 (hole=0, valid=1)

用法:
    python create_masked_pbrt_dataset.py

注意: 需要先确保 pbrt_dataset/data 下有 ideal_IQ, noise_IQ, confidence 目录
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import random

# 数据路径
PBRT_BASE = Path("/data/pre_student/GJ/DepthCAD/pbrt_dataset/data")

# 场景列表
SCENES = ["bathroom", "breakfast", "contemporary-bathroom", "pavilion", "white-room"]


def generate_mask(img_size, hole_ratio=0.15):
    """
    生成随机空洞掩码 (与 iq_inpaint_small 一致)

    Args:
        img_size: (H, W)
        hole_ratio: 目标空洞比例
    """
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.uint8)

    choice = random.random()

    if choice < 0.4:
        # 多个矩形块
        num_boxes = random.randint(2, 5)
        for _ in range(num_boxes):
            x1 = random.randint(0, max(0, W - 20))
            y1 = random.randint(0, max(0, H - 20))
            w = random.randint(20, min(80, W - x1))
            h = random.randint(20, min(80, H - y1))
            mask[y1:y1+h, x1:x1+w] = 1

    elif choice < 0.7:
        # 边缘缺失
        edge_type = random.randint(0, 3)
        edge_width = random.randint(5, 20)
        if edge_type == 0:
            mask[:edge_width, :] = 1
        elif edge_type == 1:
            mask[-edge_width:, :] = 1
        elif edge_type == 2:
            mask[:, :edge_width] = 1
        else:
            mask[:, -edge_width:] = 1

    else:
        # 随机椭圆形
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        rx = random.randint(10, min(50, W // 4))
        ry = random.randint(10, min(50, H // 4))
        # 用网格方法画椭圆
        y, x = np.ogrid[:H, :W]
        ellipse_mask = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
        mask[ellipse_mask] = 1

    # 检查空洞比例，太小就加大
    current_ratio = mask.sum() / mask.size
    if current_ratio < hole_ratio * 0.5:
        # 加一个大的随机块
        x1 = random.randint(0, max(0, W - 60))
        y1 = random.randint(0, max(0, H - 60))
        w = random.randint(40, min(100, W - x1))
        h = random.randint(40, min(100, H - y1))
        mask[y1:y1+h, x1:x1+w] = 1

    return mask.astype(np.float32)  # 0=valid, 1=hole


def load_iq_stack(base_dir, scene, idx, sample_name):
    """加载 IQ 6个通道 (I30, Q30, I40, Q40, I58, Q58) -> (6, H, W)"""
    channels = []
    for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
        fpath = base_dir / scene / idx / f"{sample_name}_{ch}.npy"
        if fpath.exists():
            channels.append(np.load(fpath))
        else:
            return None
    return np.stack(channels, axis=0)


def create_masked_dataset():
    """为 PBRT 创建 masked 数据集"""

    # 检查原始数据是否存在
    ideal_base = PBRT_BASE / 'ideal_IQ'
    noise_base = PBRT_BASE / 'noise_IQ'
    conf_base = PBRT_BASE / 'confidence'

    if not ideal_base.exists():
        print(f"ERROR: {ideal_base} does not exist!")
        return

    # 创建输出目录 (scene/index 层级)
    for subdir in ['ideal_IQ_masked', 'noise_IQ_masked', 'confidence_masked']:
        for scene in SCENES:
            scene_path = PBRT_BASE / subdir / scene
            # 获取所有可能的 index 目录
            for idx_dir in (ideal_base / scene).glob('*'):
                if idx_dir.is_dir():
                    (scene_path / idx_dir.name).mkdir(parents=True, exist_ok=True)

    # 遍历所有 IQ 文件 (scene/index/ 下的所有 _A.npy)
    processed = 0
    ideal_files = sorted(ideal_base.glob(f'*/*/*_A.npy'))
    print(f"Found {len(ideal_files)} IQ files")

    for ideal_path in tqdm(ideal_files, desc="Creating masked data"):
        # 解析路径: .../ideal_IQ/scene/idx/sample_A.npy
        parts = ideal_path.parts
        scene = parts[-3]  # e.g. "bathroom"
        idx = parts[-2]  # e.g. "0"
        sample_name = ideal_path.stem.replace('_A', '')  # e.g. "100"

        # 加载 GT IQ
        gt_iq = load_iq_stack(ideal_base, scene, idx, sample_name)
        if gt_iq is None:
            continue

        # 加载 noise IQ
        noise_iq = load_iq_stack(noise_base, scene, idx, sample_name)
        if noise_iq is None:
            continue

        H, W = gt_iq.shape[1:]

        # 生成随机 mask
        mask = generate_mask((H, W), hole_ratio=0.15)

        # 应用 mask: hole 区域置 0
        mask_3d = (mask == 1)
        masked_noise_iq = noise_iq.copy()
        masked_noise_iq[:, mask_3d] = 0

        # 创建 confidence map: valid=1, hole=0
        confidence = (mask == 0).astype(np.float32)

        # 保存 (格式要与原始 PBRT 数据集一致: 100_A.npy ~ 100_F.npy)
        channel_names = ['A', 'B', 'C', 'D', 'E', 'F']
        for c, ch in enumerate(channel_names):
            # ideal_IQ_masked: 每个通道单独保存
            np.save(PBRT_BASE / 'ideal_IQ_masked' / scene / idx / f"{sample_name}_{ch}.npy", gt_iq[c])
            # noise_IQ_masked: 每个通道单独保存
            np.save(PBRT_BASE / 'noise_IQ_masked' / scene / idx / f"{sample_name}_{ch}.npy", masked_noise_iq[c])

        # confidence_masked: 保存为单个文件 (H, W)
        np.save(PBRT_BASE / 'confidence_masked' / scene / idx / f"{sample_name}.npy", confidence)

        processed += 1

    print(f"Processed {processed} samples")


def main():
    print("Creating masked PBRT dataset...")
    print(f"Source: {PBRT_BASE}")
    print(f"Output: {PBRT_BASE}")

    # 检查必要的数据是否存在
    for subdir in ['ideal_IQ', 'noise_IQ', 'confidence']:
        path = PBRT_BASE / subdir
        if not path.exists():
            print(f"ERROR: {path} does not exist!")
            return
        print(f"  Found: {subdir}")

    # 创建 masked 数据 (所有数据一起处理，train/val split 通过 train_list_path 控制)
    create_masked_dataset()

    print("\nDone! Masked dataset created at:")
    print(f"  {PBRT_BASE}/ideal_IQ_masked")
    print(f"  {PBRT_BASE}/noise_IQ_masked")
    print(f"  {PBRT_BASE}/confidence_masked")


if __name__ == '__main__':
    main()