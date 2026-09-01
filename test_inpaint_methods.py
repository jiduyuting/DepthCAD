"""
测试不同 inpaint 方法对 IQ 空洞的填补效果

测试三种方法:
1. OpenCV Navier-Stokes (无需安装)
2. OpenCV Telea (无需安装)
3. LaMa (需额外安装)

Usage:
    python test_inpaint_methods.py
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


def opencv_ns_inpaint(img, mask):
    """OpenCV Navier-Stokes inpainting"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    return cv2.inpaint(img, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_NS)


def opencv_telea_inpaint(img, mask):
    """OpenCV Telea inpainting"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    return cv2.inpaint(img, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ideal_path", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/ideal_IQ_masked/bathroom/1/100_A.npy")
    parser.add_argument("--noise_path", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/noise_IQ_masked/bathroom/1/100_A.npy")
    parser.add_argument("--conf_path", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/confidence_masked/bathroom/1/100.npy")
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载数据
    ideal = np.load(args.ideal_path)
    noise = np.load(args.noise_path)
    conf = np.load(args.conf_path)

    # Resize
    if ideal.shape[:2] != (args.resolution, args.resolution):
        ideal = cv2.resize(ideal, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
    if noise.shape[:2] != (args.resolution, args.resolution):
        noise = cv2.resize(noise, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
    if conf.shape[:2] != (args.resolution, args.resolution):
        conf = cv2.resize(conf, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)

    # 归一化到 0-1 (OpenCV inpaint 需要)
    scale_n = max(noise.max(), abs(noise.min()), 1e-8)
    noise_norm = (noise / scale_n + 1) / 2  # [-1,1] -> [0,1]
    ideal_norm = (ideal / scale_n + 1) / 2
    conf_norm = conf

    # 空洞掩码: 1=空洞, 0=有效
    hole_mask = (conf_norm < 0.5).astype(np.float32)

    # 先用噪声图填补（模拟实际场景）
    # 在噪声图上，inpaint 是在带噪声的像素上填充
    print("=== 噪声图填补测试 ===")
    print(f"空洞像素数: {hole_mask.sum():.0f} / {hole_mask.size}")
    print(f"noise vs ideal (空洞区): {np.sqrt(((noise_norm[hole_mask > 0.5] - ideal_norm[hole_mask > 0.5])**2).mean()):.6f}")

    # OpenCV NS
    filled_ns = opencv_ns_inpaint(noise_norm, hole_mask)
    rmse_ns = np.sqrt(((filled_ns[hole_mask > 0.5] - ideal_norm[hole_mask > 0.5])**2).mean())
    print(f"OpenCV NS inpaint RMSE: {rmse_ns:.6f}")

    # OpenCV Telea
    filled_te = opencv_telea_inpaint(noise_norm, hole_mask)
    rmse_te = np.sqrt(((filled_te[hole_mask > 0.5] - ideal_norm[hole_mask > 0.5])**2).mean())
    print(f"OpenCV Telea RMSE: {rmse_te:.6f}")

    # 对比: 完全不填补（保持噪声）
    rmse_raw = np.sqrt(((noise_norm[hole_mask > 0.5] - ideal_norm[hole_mask > 0.5])**2).mean())
    print(f"原始 noise RMSE (baseline): {rmse_raw:.6f}")

    print(f"\n改善比例:")
    print(f"  NS:    {(rmse_raw - rmse_ns) / rmse_raw * 100:.1f}%")
    print(f"  Telea: {(rmse_raw - rmse_te) / rmse_raw * 100:.1f}%")

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(ideal_norm, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ideal (GT)')
    axes[0].axis('off')

    axes[1].imshow(noise_norm, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Noisy IQ\n(holes visible)')
    axes[1].axis('off')

    axes[2].imshow(filled_ns, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'OpenCV NS\nRMSE={rmse_ns:.4f}')
    axes[2].axis('off')

    axes[3].imshow(filled_te, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'OpenCV Telea\nRMSE={rmse_te:.4f}')
    axes[3].axis('off')

    axes[4].imshow(hole_mask, cmap='gray')
    axes[4].set_title('Hole Mask')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig('/data/pre_student/GJ/DepthCAD/output/inpaint_opencv_test.png', dpi=150)
    print("\nSaved to /data/pre_student/GJ/DepthCAD/output/inpaint_opencv_test.png")


if __name__ == '__main__':
    main()
