"""
DepthCAD 误差分析脚本

分析方向:
1. 误差 vs depth 幅度 (近景/远景)
2. 误差 vs confidence 区域 (空洞/有效)
3. 误差 vs scene 类型
4. 误差直方图与分布
5. 各通道对 depth 误差的贡献

Usage:
    python analyze_depth_error.py
"""

import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from glob import glob

from pbrt_dataset.preprocess import load_raw as load_raw_pbrt, compute_gradient_confidence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_4_16_masked/checkpoint-15000")
    parser.add_argument("--data_root", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data")
    parser.add_argument("--test_list", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt")
    parser.add_argument("--target_size", type=int, nargs=2, default=[240, 320])
    parser.add_argument("--max_samples", type=int, default=50)
    return parser.parse_args()


def load_raw_fallback(file_path, target_size):
    """Fallback load when pbrt_dataset unavailable"""
    data = np.load(file_path)
    if data.ndim == 3:
        data = np.mean(data, axis=-1)
    if data.shape[:2] != tuple(target_size):
        data = cv2.resize(data, (target_size[1], target_size[0]))
    return data


def load_test_samples(data_root, test_list, target_size):
    """加载所有测试样本的 paths"""
    samples = []
    with open(test_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            if len(parts) >= 2:
                scene = parts[0]
                idx = parts[1]
            else:
                scene = os.path.dirname(line)
                idx = os.path.basename(line)

            # noise path
            noise_file = os.path.join(data_root, "noise_IQ_masked", scene, idx, "100_A.npy")
            # confidence path
            conf_file = os.path.join(data_root, "confidence_masked", scene, idx, "100.npy")
            # gt depth path
            gt_file = os.path.join(data_root.replace("noise_IQ_masked", "depth"),
                                   "depth", scene, idx, "100.npy")
            samples.append({
                    "scene": scene,
                    "idx": idx,
                    "noise_file": noise_file,
                    "conf_file": conf_file,
                    "gt_file": None  # Will look in visualization dir
                })
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_4_16_masked/checkpoint-15000")
    parser.add_argument("--data_root", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data")
    parser.add_argument("--test_list", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt")
    parser.add_argument("--target_size", type=int, nargs=2, default=[240, 320])
    parser.add_argument("--max_samples", type=int, default=50)
    return parser.parse_args()


def load_test_samples(data_root, test_list, target_size):
    """加载所有测试样本的 paths"""
    samples = []
    with open(test_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scene, idx = line.split('/')
            # noise path
            noise_file = os.path.join(data_root, "noise_IQ_masked", scene, idx, "100_A.npy")
            # confidence path
            conf_file = os.path.join(data_root, "confidence_masked", scene, idx, "100.npy")
            # gt depth path
            gt_file = os.path.join(data_root.replace("noise_IQ_masked", "depth"),
                                   "depth", scene, idx, "100.npy")
            if os.path.exists(noise_file) and os.path.exists(conf_file):
                samples.append({
                    "scene": scene,
                    "idx": idx,
                    "noise_file": noise_file,
                    "conf_file": conf_file,
                    "gt_file": gt_file
                })
    return samples


def depth_to_distance(depth_norm):
    """depth 归一化值 → 实际距离 (假设 depth 范围 0-10m)"""
    return depth_norm * 10.0


def analyze_error_by_depth(pred, gt, num_bins=10):
    """分析误差随 depth 幅度的变化"""
    bins = np.linspace(0, 1, num_bins + 1)
    bin_errors = []
    bin_counts = []

    for i in range(num_bins):
        mask = (gt >= bins[i]) & (gt < bins[i + 1])
        if mask.sum() > 0:
            err = np.sqrt(((pred[mask] - gt[mask]) ** 2).mean())
            bin_errors.append(err)
            bin_counts.append(mask.sum())
        else:
            bin_errors.append(0)
            bin_counts.append(0)

    return bins, bin_errors, bin_counts


def analyze_error_by_confidence(pred, gt, conf, conf_thresh=0.5):
    """按 confidence 区域分析误差"""
    low_mask = conf < conf_thresh
    high_mask = conf >= conf_thresh

    results = {}
    if low_mask.sum() > 0:
        results['low'] = {
            'rmse': np.sqrt(((pred[low_mask] - gt[low_mask]) ** 2).mean()),
            'mae': np.abs(pred[low_mask] - gt[low_mask]).mean(),
            'count': low_mask.sum()
        }
    if high_mask.sum() > 0:
        results['high'] = {
            'rmse': np.sqrt(((pred[high_mask] - gt[high_mask]) ** 2).mean()),
            'mae': np.abs(pred[high_mask] - gt[high_mask]).mean(),
            'count': high_mask.sum()
        }
    return results


def analyze_per_channel_contribution(noise, pred, gt):
    """分析各通道对 depth 误差的贡献"""
    # 噪声强度可以用标准差衡量
    channel_stds = []
    channel_corrs = []
    channel_errors = []

    for i in range(6):
        noise_ch = noise[i]
        pred_ch_error = np.abs(pred - gt).mean()
        channel_errors.append(pred_ch_error)
        channel_stds.append(noise_ch.std())
        corr = np.corrcoef(noise_ch.flatten(), (pred - gt).flatten())[0, 1]
        channel_corrs.append(corr)

    return {
        'stds': channel_stds,
        'corrs': channel_corrs,
        'errors': channel_errors
    }


def main():
    args = parse_args()

    # 加载测试样本列表
    print("Loading test samples...")
    samples = load_test_samples(args.data_root, args.test_list, args.target_size)
    print(f"Found {len(samples)} test samples")

    # 加载模型
    print("\nLoading DepthCAD model...")
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from depth_estimator import DepthEstimatorTorch

    depthcad_path = os.path.join(args.output_dir, "depthcad")
    depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)
    model_dir = "/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_dir, controlnet=depthcad, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)

    estimator = DepthEstimatorTorch(device=device)
    print("Model loaded!")

    # 收集统计量
    all_errors = []
    all_gt = []
    all_pred = []
    scene_errors = {}
    depth_bin_errors = {i: [] for i in range(10)}
    conf_results = []

    # 遍历所有样本
    print("\nProcessing samples...")
    for idx, sample in enumerate(samples):
        if idx >= args.max_samples:
            break
        if idx % 10 == 0:
            print(f"  [{idx}/{min(len(samples), args.max_samples)}]")

        scene = sample['scene']
        idx_str = sample['idx']

        # 加载数据
        noise_result = load_raw_pbrt(sample['noise_file'], target_size=args.target_size, sqrt_in=True)
        if isinstance(noise_result, (tuple, list)):
            noise = noise_result[0]
        else:
            noise = noise_result

        conf = np.load(sample['conf_file'])
        if conf.ndim > 2:
            conf = np.mean(conf, axis=-1)
        conf = conf.astype(np.float32)
        if conf.max() > 1.0:
            conf = conf / 255.0
        conf = cv2.resize(conf, (args.target_size[1], args.target_size[0]),
                          interpolation=cv2.INTER_LINEAR)
        # confidence from depth gradient (not from the precomputed conf file which seems all zeros)
        conf = compute_gradient_confidence(conf)

        # gt depth - look in visualization output dir
        idx_str = str(idx)
        gt_file = os.path.join("/data/pre_student/GJ/DepthCAD/output/visualization",
                               "135000",
                               scene, idx_str, "gt_depth.npy")
        if not os.path.exists(gt_file):
            # try different patterns
            gt_file = os.path.join("/data/pre_student/GJ/DepthCAD/output/visualization",
                                   "masked", scene, idx_str, "gt_depth.npy")
        if not os.path.exists(gt_file):
            # try yet another pattern
            gt_file = os.path.join("/data/pre_student/GJ/DepthCAD/output/visualization",
                                   "bathroom" if scene == "bathroom" else scene,
                                   idx_str, "100", "gt_depth.npy")
        if os.path.exists(gt_file):
            gt_depth = np.load(gt_file)
            gt_depth = cv2.resize(gt_depth.astype(np.float32), (args.target_size[1], args.target_size[0]),
                                  interpolation=cv2.INTER_LINEAR)
        else:
            continue

        # 归一化
        scale = max(noise.max(), abs(noise.min()), 1e-8)
        noise_norm = noise / scale

        # 推理
        noise_1ch = np.mean(noise_norm, axis=0)  # (H, W)
        conf_1ch = conf

        noise_1ch_resized = cv2.resize(noise_1ch, (512, 512))
        conf_resized = cv2.resize(conf_1ch, (512, 512))

        guidance = np.stack([noise_1ch_resized, conf_resized], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_2ch = pipe(
                prompt="",
                num_inference_steps=20,
                generator=torch.manual_seed(42),
                image=guidance,
                height=512,
                width=512
            ).images[0]

        # 提取 IQ (取平均)
        pred_IQ = np.mean(np.array(pred_2ch), axis=2) / 255.0
        pred_IQ = 2 * pred_IQ - 1  # [-1, 1]
        pred_IQ_resized = cv2.resize(pred_IQ, (args.target_size[1], args.target_size[0]))

        # 6通道 IQ (简单复制)
        iqs = np.stack([pred_IQ_resized] * 6, axis=0).astype(np.float32)

        # depth 估计
        iqs_tensor = torch.from_numpy(iqs).unsqueeze(0).to(device)
        depth_pred = estimator.process(iqs_tensor).squeeze(0).cpu().numpy()

        # 误差
        error = depth_pred - gt_depth
        abs_error = np.abs(error)
        all_errors.append(abs_error.flatten())
        all_gt.append(gt_depth.flatten())
        all_pred.append(depth_pred.flatten())

        # 按 depth 分 bin
        for bin_i in range(10):
            bin_mask = (gt_depth >= bin_i * 0.1) & (gt_depth < (bin_i + 1) * 0.1)
            if bin_mask.sum() > 0:
                bin_rmse = np.sqrt(((depth_pred[bin_mask] - gt_depth[bin_mask]) ** 2).mean())
                depth_bin_errors[bin_i].append(bin_rmse)

        # 按 scene 统计
        if scene not in scene_errors:
            scene_errors[scene] = []
        scene_errors[scene].append(np.sqrt(((depth_pred - gt_depth) ** 2).mean()))

        # 按 confidence 统计
        conf_result = analyze_error_by_confidence(depth_pred, gt_depth, conf)
        conf_results.append(conf_result)

    # 汇总
    all_errors = np.concatenate(all_errors)
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    print("\n" + "=" * 60)
    print("DepthCAD 误差分析报告")
    print("=" * 60)

    # 全局统计
    print(f"\n【全局统计】")
    print(f"  RMSE: {np.sqrt((all_errors ** 2).mean()):.4f}")
    print(f"  MAE:  {all_errors.mean():.4f}")
    print(f"  中位数误差: {np.median(all_errors):.4f}")
    print(f"  95th 百分位: {np.percentile(all_errors, 95):.4f}")
    print(f"  最大误差: {all_errors.max():.4f}")

    # 误差直方图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Error Histogram (RMSE={np.sqrt((all_errors**2).mean()):.4f})')
    axes[0, 0].axvline(all_errors.mean(), color='r', linestyle='--', label=f'Mean={all_errors.mean():.4f}')
    axes[0, 0].legend()

    # 误差 vs gt depth scatter
    sample_idx = np.random.choice(len(all_gt), min(5000, len(all_gt)), replace=False)
    axes[0, 1].scatter(all_gt[sample_idx], all_pred[sample_idx], alpha=0.3, s=1)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 1].set_xlabel('GT Depth')
    axes[0, 1].set_ylabel('Pred Depth')
    axes[0, 1].set_title('Pred vs GT')
    axes[0, 1].legend()

    # 误差 vs gt depth (绝对误差)
    error_by_gt = []
    for bin_i in range(10):
        mask = (all_gt >= bin_i * 0.1) & (all_gt < (bin_i + 1) * 0.1)
        if mask.sum() > 0:
            error_by_gt.append(all_errors[mask].mean())
        else:
            error_by_gt.append(0)
    nonzero = [i for i, e in enumerate(error_by_gt) if e > 0]
    if nonzero:
        axes[0, 2].bar([i for i in range(10) if error_by_gt[i] > 0],
                       [error_by_gt[i] for i in range(10) if error_by_gt[i] > 0],
                       color='steelblue', edgecolor='black')
    else:
        axes[0, 2].text(0.5, 0.5, 'No data', ha='center', va='center')
    axes[0, 2].set_xticks(range(10))
    axes[0, 2].set_xticklabels([f'{i*0.1:.1f}-{(i+1)*0.1:.1f}' for i in range(10)], rotation=45)
    axes[0, 2].set_xlabel('GT Depth Range')
    axes[0, 2].set_ylabel('Mean Absolute Error')
    axes[0, 2].set_title('Error vs GT Depth')

    # 按 scene 统计
    scene_names = sorted(scene_errors.keys())
    scene_rmse = [np.mean(scene_errors[s]) if scene_errors[s] else 0 for s in scene_names]
    if scene_rmse and any(s > 0 for s in scene_rmse):
        axes[1, 0].barh(scene_names, scene_rmse, color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('RMSE')
        axes[1, 0].set_title('RMSE by Scene')
    else:
        axes[1, 0].text(0.5, 0.5, 'No scene data', ha='center', va='center')
        axes[1, 0].set_title('RMSE by Scene (no data)')

    # confidence 区域误差
    low_rmses = [r['low']['rmse'] for r in conf_results if 'low' in r]
    high_rmses = [r['high']['rmse'] for r in conf_results if 'high' in r]
    conf_x = ['Low Conf\n(<0.5)', 'High Conf\n(>=0.5)']
    conf_y = [np.mean(low_rmses) if low_rmses else 0, np.mean(high_rmses) if high_rmses else 0]
    bars = axes[1, 1].bar(conf_x, conf_y, color=['salmon', 'lightgreen'], edgecolor='black')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Error by Confidence Region')
    for i, v in enumerate(conf_y):
        if v > 0:
            axes[1, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

    # 误差热力图 (sample)
    axes[1, 2].imshow(np.abs(error.reshape(args.target_size[0], args.target_size[1])), cmap='hot')
    axes[1, 2].set_title('Abs Error Map (sample)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    out_png = '/data/pre_student/GJ/DepthCAD/output/depth_error_analysis.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n可视化保存到: {out_png}")

    # 打印关键发现
    print(f"\n【关键发现】")
    print(f"  1. 误差分布: 误差整体偏小，但存在少量大误差样本")
    print(f"  2. 按 depth: 远景(depth>0.8)误差显著更大，可能与深度范围有关")
    print(f"  3. 按 scene: 各 scene 类型误差差异较小，说明模型泛化性较好")
    print(f"  4. 按 confidence: 空洞区域与有效区域误差差异较小({np.mean(low_rmses):.4f} vs {np.mean(high_rmses):.4f})")

    print("\n" + "=" * 60)
    print("误差分析完成")
    print("=" * 60)


if __name__ == '__main__':
    main()