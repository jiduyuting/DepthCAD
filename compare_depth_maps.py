import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_depth_map(data, name):
    """分析深度图的统计特性"""
    # 过滤掉0值（通常是无效像素）
    valid_mask = data > 0
    valid_data = data[valid_mask]

    if len(valid_data) == 0:
        return {
            'name': name,
            'shape': data.shape,
            'min': 0, 'max': 0, 'mean': 0, 'std': 0,
            'valid_ratio': 0,
            'has_inf': np.isinf(data).any(),
            'has_nan': np.isnan(data).any()
        }

    return {
        'name': name,
        'shape': data.shape,
        'min': float(valid_data.min()),
        'max': float(valid_data.max()),
        'mean': float(valid_data.mean()),
        'std': float(valid_data.std()),
        'median': float(np.median(valid_data)),
        'valid_ratio': float(valid_mask.sum() / data.size),
        'has_inf': np.isinf(data).any(),
        'has_nan': np.isnan(data).any()
    }

def compare_and_visualize(pred_npy_dir, gt_npy_dir, output_dir, sample_indices=[100, 101, 102]):
    """
    对比预测深度图和GT深度图

    Args:
        pred_npy_dir: 预测的.npy文件目录
        gt_npy_dir: GT的.npy文件目录
        output_dir: 输出对比结果目录
        sample_indices: 要对比的样本编号列表
    """
    pred_path = Path(pred_npy_dir)
    gt_path = Path(gt_npy_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("深度图对比分析")
    print("=" * 70)
    print(f"预测目录: {pred_path}")
    print(f"GT目录:   {gt_path}")
    print(f"输出目录: {out_path}")
    print("=" * 70)

    for idx in sample_indices:
        npy_file = f"bathroom/1/{idx}.npy"

        pred_file = pred_path / npy_file
        gt_file = gt_path / npy_file

        if not pred_file.exists() or not gt_file.exists():
            print(f"跳过 {idx}: 文件不存在")
            continue

        # 加载数据
        pred_data = np.load(pred_file)
        gt_data = np.load(gt_file)

        # 分析统计特性
        pred_stats = analyze_depth_map(pred_data, f"Prediction {idx}")
        gt_stats = analyze_depth_map(gt_data, f"Ground Truth {idx}")

        # 打印统计信息
        print(f"\n{'='*70}")
        print(f"样本: {idx}")
        print(f"{'='*70}")
        print(f"{'指标':<15} {'预测值':>20} {'GT值':>20} {'差异':>20}")
        print(f"{'-'*70}")

        metrics = ['shape', 'min', 'max', 'mean', 'std', 'median', 'valid_ratio']
        metric_names = ['形状', '最小值', '最大值', '平均值', '标准差', '中位数', '有效像素比']

        for metric, name in zip(metrics, metric_names):
            pred_val = pred_stats[metric]
            gt_val = gt_stats[metric]

            if metric == 'shape':
                diff_str = f"{pred_val} vs {gt_val}"
            else:
                diff = pred_val - gt_val if isinstance(pred_val, (int, float)) else 0
                diff_pct = (diff / gt_val * 100) if gt_val != 0 else 0
                diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"

            print(f"{name:<15} {str(pred_val):>20} {str(gt_val):>20} {diff_str:>20}")

        print(f"{'-'*70}")
        print(f"预测值包含inf: {pred_stats['has_inf']}, 包含NaN: {pred_stats['has_nan']}")
        print(f"GT值包含inf:   {gt_stats['has_inf']}, 包含NaN: {gt_stats['has_nan']}")

        # 计算误差指标
        # 只比较有效的像素
        valid_mask = (gt_data > 0.001) & (gt_data < 9)
        if valid_mask.sum() > 0:
            pred_valid = pred_data[valid_mask]
            gt_valid = gt_data[valid_mask]

            mae = np.abs(pred_valid - gt_valid).mean()
            rmse = np.sqrt(((pred_valid - gt_valid) ** 2).mean())

            rel_error = (np.abs(pred_valid - gt_valid) / (gt_valid + 1e-8)).mean() * 100

            print(f"\n误差指标:")
            print(f"  MAE (平均绝对误差):  {mae:.4f}")
            print(f"  RMSE (均方根误差):   {rmse:.4f}")
            print(f"  相对误差:           {rel_error:.2f}%")

        # 可视化对比
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 处理数据用于可视化（替换inf和nan）
        pred_vis = pred_data.copy()
        gt_vis = gt_data.copy()

        if np.isinf(pred_vis).any():
            max_val = pred_vis[~np.isinf(pred_vis)].max() if (~np.isinf(pred_vis)).any() else 1
            pred_vis[np.isinf(pred_vis)] = max_val
        if np.isinf(gt_vis).any():
            max_val = gt_vis[~np.isinf(gt_vis)].max() if (~np.isinf(gt_vis)).any() else 1
            gt_vis[np.isinf(gt_vis)] = max_val

        pred_vis = np.nan_to_num(pred_vis, nan=0)
        gt_vis = np.nan_to_num(gt_vis, nan=0)

        # 统一颜色范围
        vmin = min(pred_vis[pred_vis > 0].min(), gt_vis[gt_vis > 0].min()) if (pred_vis > 0).any() and (gt_vis > 0).any() else 0
        vmax = max(pred_vis.max(), gt_vis.max())

        # 第一行: 预测深度图
        im1 = axes[0, 0].imshow(pred_vis, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f'预测深度图 {idx}\nRange: [{pred_vis.min():.3f}, {pred_vis.max():.3f}]')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], label='Depth')

        # 第二行第一个: GT深度图
        im2 = axes[1, 0].imshow(gt_vis, cmap='inferno', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'GT深度图 {idx}\nRange: [{gt_vis.min():.3f}, {gt_vis.max():.3f}]')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], label='Depth')

        # 误差图
        error_map = np.abs(pred_vis - gt_vis)
        im3 = axes[0, 1].imshow(error_map, cmap='hot')
        axes[0, 1].set_title(f'绝对误差图\nMAE: {np.abs(pred_vis - gt_vis).mean():.4f}')
        axes[0, 1].axis('off')
        plt.colorbar(im3, ax=axes[0, 1], label='Absolute Error')

        # 相对误差图
        rel_error_map = np.abs(pred_vis - gt_vis) / (gt_vis + 1e-8) * 100
        rel_error_map[gt_vis <= 0.001] = 0  # 忽略无效像素
        im4 = axes[1, 1].imshow(rel_error_map, cmap='hot', vmin=0, vmax=100)
        axes[1, 1].set_title(f'相对误差图 (%)')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], label='Relative Error (%)')

        # 预测值的分布直方图
        axes[0, 2].hist(pred_vis.flatten(), bins=50, alpha=0.7, label='预测', color='blue')
        axes[0, 2].set_xlabel('Depth Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title(f'预测值分布\nMean: {pred_vis[pred_vis>0].mean():.4f}')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # GT值的分布直方图
        axes[1, 2].hist(gt_vis.flatten(), bins=50, alpha=0.7, label='GT', color='orange')
        axes[1, 2].set_xlabel('Depth Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title(f'GT值分布\nMean: {gt_vis[gt_vis>0].mean():.4f}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_file = out_path / f"comparison_{idx}.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n对比图已保存: {comparison_file}")

    print(f"\n{'='*70}")
    print("分析完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 配置
    PRED_NPY_DIR = "/data/pre_student/GJ/DepthCAD/pbrt/data_1_5_5000"  # 预测的.npy文件
    GT_NPY_DIR = "/data/pre_student/hcy/pbrt/gt_depth"               # GT的.npy文件
    OUTPUT_DIR = "/data/pre_student/GJ/DepthCAD/pbrt/comparison_results"

    # 要对比的样本编号
    SAMPLES = [100, 101, 102, 103, 104, 105]

    compare_and_visualize(PRED_NPY_DIR, GT_NPY_DIR, OUTPUT_DIR, SAMPLES)
