import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def convert_npy_to_png(input_dir, output_dir, cmap='inferno', channel='mean'):
    """
    将指定目录下的所有 .npy 文件转换为 PNG 图像

    Args:
        input_dir: 包含 .npy 文件的输入目录
        output_dir: PNG 输出目录
        cmap: matplotlib 颜色映射 (默认 'inferno')
        channel: 多通道处理方式: 'mean'(平均), 'first'(第一通道), int(指定通道)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有 .npy 文件
    npy_files = list(input_path.rglob("*.npy"))
    total = len(npy_files)

    if total == 0:
        print(f"在 {input_dir} 中未找到 .npy 文件")
        return

    print(f"找到 {total} 个 .npy 文件")
    print(f"输出目录: {output_dir}")
    print(f"通道处理: {channel}")
    print("=" * 50)

    success_count = 0
    fail_count = 0

    for idx, npy_file in enumerate(npy_files, 1):
        try:
            # 加载数据
            data = np.load(npy_file)

            # 处理 (C, H, W) 格式 -> 转置为 (H, W, C)
            if data.ndim == 3 and data.shape[0] < data.shape[2]:
                # 可能是 (C, H, W) 格式
                data = np.transpose(data, (1, 2, 0))
                print(f"  [调试] 转置为 (H, W, C): {data.shape}")

            # 处理多通道: 如果是 (H, W, C) 格式但 C > 4，取平均或指定通道
            if data.ndim == 3 and data.shape[2] > 4:
                if channel == 'mean':
                    data = data.mean(axis=2)
                elif channel == 'first':
                    data = data[:, :, 0]
                elif isinstance(channel, int):
                    data = data[:, :, channel]

            # 处理 inf 和 NaN
            if np.isinf(data).any():
                max_val = data[~np.isinf(data)].max()
                data[np.isinf(data)] = max_val

            data = np.nan_to_num(data, nan=0.0)

            # 计算相对路径，保持目录结构
            rel_path = npy_file.relative_to(input_path)
            png_file = output_path / rel_path.with_suffix('.png')

            # 创建输出子目录
            png_file.parent.mkdir(parents=True, exist_ok=True)

            # 绘图并保存
            plt.figure(figsize=(8, 8))
            plt.imshow(data, cmap=cmap)
            plt.colorbar(label='Depth Value')
            plt.title(f'{rel_path} {data.shape}')
            plt.axis('off')
            plt.savefig(png_file, bbox_inches='tight', dpi=150)
            plt.close()

            print(f"[{idx}/{total}] ✓ {rel_path}")
            success_count += 1

        except Exception as e:
            print(f"[{idx}/{total}] ✗ {rel_path}: {e}")
            fail_count += 1

    print("=" * 50)
    print(f"转换完成! 成功: {success_count}, 失败: {fail_count}")


if __name__ == "__main__":
    # 配置
    INPUT_DIR = "/data/pre_student/hcy/pbrt/gt/bathroom/0"
    OUTPUT_DIR = "/data/pre_student/GJ/DepthCAD/pbrt/bathroom/0"

    # 可选颜色映射: 'viridis', 'inferno', 'plasma', 'jet', 'gray', 'hot', 'cool'
    CMAP = 'inferno'

    convert_npy_to_png(INPUT_DIR, OUTPUT_DIR, cmap=CMAP)