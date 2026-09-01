"""
测试 v3 模型 (带 GAN 损失 + 更大网络)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import torch

import sys
sys.path.insert(0, '/data/pre_student/GJ/DepthCAD')
from DepthCAD.old.train_iq_inpaint_v3 import TwoStageIQInpaintNetV3, IQAwareLossV2
from depth_estimator import DepthEstimator

# GT - 6个通道分别是 100_A 到 100_F (I30, Q30, I40, Q40, I58, Q58)
gt_dir = '/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/ideal_IQ/bathroom/0/'
gt_iq = np.stack([
    np.load(f'{gt_dir}100_A.npy'),
    np.load(f'{gt_dir}100_B.npy'),
    np.load(f'{gt_dir}100_C.npy'),
    np.load(f'{gt_dir}100_D.npy'),
    np.load(f'{gt_dir}100_E.npy'),
    np.load(f'{gt_dir}100_F.npy'),
], axis=0)  # (6, 512, 512)
gt_depth = np.load('/data/pre_student/hcy/pbrt/gt_depth/bathroom/0/100.npy')

# v3 模型输出 (epoch 19 best_model)
v3_dir = '/data/pre_student/GJ/DepthCAD/output/iq_inpaint_v3/inference_test/'
v3_depth = np.load(f'{v3_dir}bathroom_1_100_depth.npy')
v3_iq = np.load(f'{v3_dir}bathroom_1_100_iq_filled.npy')

# v2_small 模型输出 (旧模型对比)
v2_dir = '/data/pre_student/GJ/DepthCAD/output/iq_inpaint_v2_small/test_result/'
v2_depth = np.load(f'{v2_dir}bathroom_1_100_depth.npy')
v2_iq = np.load(f'{v2_dir}bathroom_1_100_iq_filled.npy')

print(f"GT IQ: {gt_iq.shape}, depth: {gt_depth.shape}")
print(f"V3 模型 IQ: {v3_iq.shape}, depth: {v3_depth.shape}")
print(f"V2_small 模型 IQ: {v2_iq.shape}, depth: {v2_depth.shape}")
print(f"GT depth range: [{gt_depth.min():.3f}, {gt_depth.max():.3f}]")

# resize GT 到模型输出的尺寸以便比较
target_h, target_w = v3_depth.shape
gt_depth_resized = zoom(gt_depth, (target_h / gt_depth.shape[0], target_w / gt_depth.shape[1]), order=1)
gt_iq_resized = np.zeros((6, target_h, target_w), dtype=gt_iq.dtype)
for c in range(6):
    gt_iq_resized[c] = zoom(gt_iq[c], (target_h / gt_iq.shape[1], target_w / gt_iq.shape[2]), order=1)

print(f"\nresize后 GT IQ: {gt_iq_resized.shape}, depth: {gt_depth_resized.shape}")

# 计算和GT的误差 (使用resize后的GT)
v3_depth_err = np.abs(v3_depth - gt_depth_resized)
v2_depth_err = np.abs(v2_depth - gt_depth_resized)
v3_iq_err = np.abs(v3_iq - gt_iq_resized)
v2_iq_err = np.abs(v2_iq - gt_iq_resized)

print(f"\n深度图 MAE:")
print(f"  V2_small (旧): {v2_depth_err.mean():.6f}")
print(f"  V3 (新, GAN): {v3_depth_err.mean():.6f}")

print(f"\nIQ MAE (各通道平均):")
for c, name in enumerate(['I30', 'Q30', 'I40', 'Q40', 'I58', 'Q58']):
    print(f"  {name} - V2_small: {v2_iq_err[c].mean():.6f}, V3: {v3_iq_err[c].mean():.6f}")

# 创建对比图
fig, axes = plt.subplots(3, 6, figsize=(20, 10))

# 第一行: GT (使用resize后的)
axes[0, 0].imshow(gt_depth_resized, cmap='viridis')
axes[0, 0].set_title('GT Depth')
axes[0, 0].axis('off')

for c, name in enumerate(['I30', 'Q30', 'I40', 'Q40', 'I58', 'Q58']):
    axes[0, c+1 if c < 5 else 5].imshow(gt_iq_resized[c], cmap='viridis')
    axes[0, c+1 if c < 5 else 5].set_title(f'GT {name}')
    axes[0, c+1 if c < 5 else 5].axis('off')

# 第二行: V2_small 旧模型
axes[1, 0].imshow(v2_depth, cmap='viridis')
axes[1, 0].set_title(f'V2_small Depth\nMAE={v2_depth_err.mean():.4f}')
axes[1, 0].axis('off')

for c, name in enumerate(['I30', 'Q30', 'I40', 'Q40', 'I58', 'Q58']):
    col = c + 1 if c < 5 else 5
    if col < 6:
        axes[1, col].imshow(v2_iq[c], cmap='viridis')
        axes[1, col].set_title(f'V2_small {name}')
        axes[1, col].axis('off')

# 第三行: V3 新模型
axes[2, 0].imshow(v3_depth, cmap='viridis')
axes[2, 0].set_title(f'V3 (GAN) Depth\nMAE={v3_depth_err.mean():.4f}')
axes[2, 0].axis('off')

for c, name in enumerate(['I30', 'Q30', 'I40', 'Q40', 'I58', 'Q58']):
    col = c + 1 if c < 5 else 5
    if col < 6:
        axes[2, col].imshow(v3_iq[c], cmap='viridis')
        axes[2, col].set_title(f'V3 {name}')
        axes[2, col].axis('off')

plt.suptitle('GT vs V2_small vs V3 (GAN) - bathroom_1_100')
plt.tight_layout()
plt.savefig('/data/pre_student/GJ/DepthCAD/output/iq_inpaint_v3/inference_test/bathroom_1_100_comparison.png', dpi=150)
plt.close()
print("\n已保存对比图到 output/iq_inpaint_v3/compare_v2_vs_v3.png")