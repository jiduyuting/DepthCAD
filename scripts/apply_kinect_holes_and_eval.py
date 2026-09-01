"""
为 PBRT 数据集添加 Kinect 真实空洞 + 噪声，并进行去噪+填补空洞处理

完整流程:
  PBRT_GT → 添加 Kinect 风格空洞 + 噪声 → 去噪 (ControlNet) → 填补空洞 (SD Inpaint) → 评估 MAE

用法:
    python apply_kinect_holes_and_eval.py \
        --pbrt_ideal_dir /path/to/ideal_IQ \
        --pbrt_noise_dir /path/to/noise_IQ \
        --pbrt_conf_dir /path/to/confidence \
        --output_dir ./kinect_evaluation \
        --checkpoint_path /path/to/depthcad \
        --sample_idx "bathroom/1/100"
"""

import time
import argparse
import json
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

from depth_estimator import DepthEstimator
from inference_depth_postprocess import (
    confidence_fill_depth_fast,
    edge_aware_ns_bilateral_fill,
    local_plane_fit_fill,
    opencv_depth_inpaint,
)

torch = None
StableDiffusionInpaintPipeline = None
StableDiffusionControlNetPipeline = None
ControlNetModel = None
UniPCMultistepScheduler = None


def ensure_diffusion_dependencies():
    """Import heavy diffusion dependencies only when the full legacy pipeline is used."""
    global torch
    global StableDiffusionInpaintPipeline
    global StableDiffusionControlNetPipeline
    global ControlNetModel
    global UniPCMultistepScheduler

    if torch is None:
        import torch as torch_module
        torch = torch_module

    if ControlNetModel is None:
        from diffusers import (
            StableDiffusionInpaintPipeline as InpaintPipeline,
            StableDiffusionControlNetPipeline as ControlNetPipeline,
            ControlNetModel as DiffusersControlNetModel,
            UniPCMultistepScheduler as DiffusersUniPCMultistepScheduler,
        )
        StableDiffusionInpaintPipeline = InpaintPipeline
        StableDiffusionControlNetPipeline = ControlNetPipeline
        ControlNetModel = DiffusersControlNetModel
        UniPCMultistepScheduler = DiffusersUniPCMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument("--ideal_iq_dir", type=str, default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/ideal_IQ")
    parser.add_argument("--noise_iq_dir", type=str, default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/noise_IQ")
    parser.add_argument("--noise_iq_holes_dir", type=str, default=None,
                        help="预处理好的带空洞的noise_IQ目录")
    parser.add_argument("--hole_mask_dir", type=str, default=None,
                        help="预处理好的空洞mask目录")
    parser.add_argument("--gt_depth_dir", type=str, default="/data/pre_student/hcy/pbrt/gt_depth")
    parser.add_argument("--output_dir", type=str, default="./kinect_evaluation")
    # 模型路径
    parser.add_argument("--checkpoint_path", type=str, default="/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_4_16_masked/checkpoint-15000/depthcad")
    # 采样参数
    parser.add_argument("--sample_idx", type=str, default=None,
                        help="e.g., 'bathroom/0/100' (scene/idx/sample_name)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of random samples to process from dataset")
    parser.add_argument("--scene", type=str, default=None,
                        help="Process all samples from a specific scene (e.g., 'bathroom')")
    # 可视化
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Save visualization images for each sample")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256],
                        help="Target size for resizing (height, width). Default 256x256 for faster processing.")
    parser.add_argument("--hole_ratio", type=float, default=0.15,
                        help="Target hole ratio over the whole image.")
    parser.add_argument("--amp_threshold", type=float, default=None,
                        help="Optional fixed amplitude threshold. Default uses adaptive percentile.")
    parser.add_argument("--amp_percentile", type=float, default=5.0,
                        help="Adaptive amplitude percentile used when amp_threshold is not set.")
    parser.add_argument("--block_size", type=int, default=4,
                        help="Block size for Kinect-style hole proposal.")
    parser.add_argument("--low_amp_ratio", type=float, default=0.4,
                        help="Fraction of low-amplitude pixels required to mark a block as hole.")
    parser.add_argument("--depth_fill_threshold", type=float, default=0.5,
                        help="Confidence threshold for depth-domain hole filling baseline.")
    parser.add_argument("--depth_fill_method", type=str, default="telea",
                        choices=["telea", "ns", "confidence_fast", "ns_bilateral", "plane"],
                        help="Depth-domain fill method. All methods only modify the explicit hole mask except legacy confidence_fast.")
    parser.add_argument("--depth_fill_radius", type=int, default=15,
                        help="OpenCV inpaint radius for telea/ns/ns_bilateral depth-domain fill.")
    parser.add_argument("--bilateral_radius", type=int, default=5,
                        help="Window radius for ns_bilateral depth-domain smoothing.")
    parser.add_argument("--bilateral_sigma_depth", type=float, default=0.05,
                        help="Normalized depth sigma for ns_bilateral range weights.")
    parser.add_argument("--bilateral_sigma_conf", type=float, default=0.25,
                        help="Confidence/guidance sigma for ns_bilateral range weights.")
    parser.add_argument("--bilateral_iters", type=int, default=1,
                        help="Number of ns_bilateral smoothing iterations inside holes.")
    parser.add_argument("--plane_max_ring_radius", type=int, default=12,
                        help="Largest valid-boundary search radius for plane fill.")
    parser.add_argument("--plane_min_boundary_points", type=int, default=12,
                        help="Minimum valid boundary samples required to fit a local plane.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run subdirectory name when output_dir already contains old results.")
    parser.add_argument("--append_output_dir", action="store_true", default=False,
                        help="Write directly into output_dir even if it already contains previous results.")
    parser.add_argument("--save_depth_completion_cache", action="store_true", default=False,
                        help="Save per-sample tensors for training a depth-domain completion/refinement model.")
    parser.add_argument("--depth_cache_dir", type=str, default=None,
                        help="Directory for depth completion cache. Default: ./depth_completion_cache/<run_name>.")
    parser.add_argument("--depth_cache_save_iq", action="store_true", default=False,
                        help="Also save noisy and DepthCAD-denoised 6-channel IQ tensors for pseudo-RGB IQ encoders.")
    parser.add_argument("--iq_cache_only", action="store_true", default=False,
                        help="Only generate IQ/cache tensors for DepthCAD-HoleAware; skip DepthCAD, SD inpaint, and depth fill.")
    parser.add_argument("--depth_cache_refine_dilation", type=int, default=5,
                        help="Pixel radius used to dilate hole_mask into refine_mask for residual depth completion.")
    parser.add_argument("--resume_depth_completion_cache", action="store_true", default=False,
                        help="When saving depth completion cache, skip samples whose .npz cache already exists and is readable.")
    parser.add_argument("--save_sd_diagnostics", action="store_true", default=False,
                        help="Save SD/IQ inpainting intermediate tensors and diagnostic metrics for failure analysis.")
    parser.add_argument("--sd_diagnostics_dir", type=str, default=None,
                        help="Directory for SD diagnostics. Default: ./sd_diagnostics/<run_name>.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def output_dir_has_results(output_dir):
    """Return True when output_dir already has evaluation artifacts."""
    if not os.path.isdir(output_dir):
        return False
    patterns = [
        "mae_results_summary.json",
        "result_*.json",
        "vis_*.png",
    ]
    return any(glob(os.path.join(output_dir, pattern)) for pattern in patterns)


def prepare_output_dir(args):
    """
    Keep new runs separate from stale results by default.

    The historical kinect_evaluation directory contains mixed-format JSON files, so
    writing a fresh run into a subdirectory prevents accidental aggregation mistakes.
    """
    requested_output_dir = args.output_dir
    if output_dir_has_results(requested_output_dir) and not args.append_output_dir:
        run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(requested_output_dir, run_name)
        print(
            f"Output directory already has previous results; "
            f"writing this run to: {args.output_dir}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    return args.output_dir


def prepare_depth_cache_dir(args):
    """Create a run-scoped cache directory for learned depth completion inputs."""
    if not args.save_depth_completion_cache:
        return None

    if args.depth_cache_dir is None:
        run_dir_name = os.path.basename(os.path.normpath(args.output_dir))
        args.depth_cache_dir = os.path.join("./depth_completion_cache", run_dir_name)

    os.makedirs(args.depth_cache_dir, exist_ok=True)
    print(f"Depth completion cache will be saved to: {args.depth_cache_dir}")
    return args.depth_cache_dir


def prepare_sd_diagnostics_dir(args):
    """Create a run-scoped directory for SD/IQ inpainting diagnostics."""
    if not args.save_sd_diagnostics:
        return None

    if args.sd_diagnostics_dir is None:
        run_dir_name = os.path.basename(os.path.normpath(args.output_dir))
        args.sd_diagnostics_dir = os.path.join("./sd_diagnostics", run_dir_name)

    os.makedirs(args.sd_diagnostics_dir, exist_ok=True)
    print(f"SD diagnostics will be saved to: {args.sd_diagnostics_dir}")
    return args.sd_diagnostics_dir


def depth_completion_cache_path(args, scene, idx, sample_name):
    return os.path.join(args.depth_cache_dir, scene, str(idx), f"{sample_name}.npz")


def sd_diagnostics_path(args, scene, idx, sample_name):
    return os.path.join(args.sd_diagnostics_dir, scene, str(idx), f"{sample_name}.npz")


def depth_completion_cache_required_keys(args):
    keys = {
        "sample_name",
        "depth_noisy",
        "depth_depthcad",
        "depth_base",
        "gt_depth",
        "hole_mask",
        "refine_mask",
        "confidence",
        "valid_mask",
        "noisy_amplitude",
        "noisy_amplitude_mean",
        "denoised_amplitude",
        "denoised_amplitude_mean",
        "depth_fill_method",
        "depth_fill_radius",
        "plane_max_ring_radius",
        "plane_min_boundary_points",
    }
    if args.depth_cache_save_iq:
        keys.update({"noisy_iq", "denoised_iq"})
    return keys


def is_depth_completion_cache_complete(path, args):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = depth_completion_cache_required_keys(args) - set(data.files)
            if missing:
                return False
            # Touch key arrays so truncated zip files fail here instead of at training time.
            for key in ["depth_base", "gt_depth", "hole_mask", "confidence"]:
                _ = data[key].shape
        return True
    except Exception:
        return False


def save_depth_cache_manifest(args, samples):
    if not args.save_depth_completion_cache:
        return
    manifest_path = os.path.join(args.depth_cache_dir, "selected_samples.json")
    manifest = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "scene": args.scene,
        "sample_idx": args.sample_idx,
        "depth_fill_method": args.depth_fill_method,
        "depth_fill_radius": args.depth_fill_radius,
        "plane_max_ring_radius": args.plane_max_ring_radius,
        "plane_min_boundary_points": args.plane_min_boundary_points,
        "iq_cache_only": bool(args.iq_cache_only),
        "samples": [f"{scene}/{idx}/{sample_name}" for scene, idx, sample_name in samples],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def filter_resume_depth_cache_samples(args, samples):
    if not (args.save_depth_completion_cache and args.resume_depth_completion_cache):
        return samples

    remaining = []
    skipped = 0
    incomplete = 0
    for scene, idx, sample_name in samples:
        cache_path = depth_completion_cache_path(args, scene, idx, sample_name)
        if is_depth_completion_cache_complete(cache_path, args):
            skipped += 1
            continue
        if os.path.exists(cache_path):
            incomplete += 1
            print(f"  Existing cache is incomplete/corrupt, regenerating: {cache_path}")
        remaining.append((scene, idx, sample_name))

    print(
        f"Resume depth cache: skipped {skipped} complete samples, "
        f"will process {len(remaining)} remaining samples"
    )
    if incomplete:
        print(f"Resume depth cache: {incomplete} existing files will be regenerated")
    return remaining


def get_all_samples(data_dir, scenes=None):
    """获取所有可用样本"""
    samples = []
    data_path = data_dir

    if scenes is None:
        scenes = os.listdir(data_path)

    for scene in scenes:
        scene_path = os.path.join(data_path, scene)
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
                    # 不要 break，每个 idx 可能有多个样本

    return samples


def collect_samples(args):
    """收集要处理的样本列表"""
    samples = []

    def add_samples_from_dir(data_dir, scene, idx):
        """从 scene/idx 目录添加所有样本"""
        idx_path = os.path.join(data_dir, scene, idx)
        if os.path.isdir(idx_path):
            for f in os.listdir(idx_path):
                if f.endswith('_A.npy'):
                    sample_name = f.replace('_A.npy', '')
                    samples.append((scene, idx, sample_name))

    if args.sample_idx:
        parts = args.sample_idx.split('/')
        if len(parts) == 3:
            samples.append(tuple(parts))
        elif len(parts) == 2:
            add_samples_from_dir(args.noise_iq_dir, parts[0], parts[1])
    elif args.scene:
        samples = get_all_samples(args.noise_iq_dir, scenes=[args.scene])
    else:
        # 确定使用哪个数据目录
        data_dir = args.noise_iq_holes_dir if args.noise_iq_holes_dir else args.noise_iq_dir
        all_samples = get_all_samples(data_dir)
        if args.num_samples:
            np.random.seed(args.seed)
            indices = np.random.choice(len(all_samples), min(args.num_samples, len(all_samples)), replace=False)
            samples = [all_samples[i] for i in indices]
        else:
            samples = all_samples

    return samples


# =============================================================================
# Kinect 风格空洞生成
# =============================================================================

def compute_kinect_confidence_from_depth(depth):
    """
    根据深度图计算 Kinect 置信度 (模拟真实 Kinect 的置信度机制)

    Kinect 置信度取决于:
    1. 深度值本身 (太近/太远不可靠)
    2. 深度边缘 (边缘处不可靠)
    3. 表面反射 (高/低反射率不可靠)
    """
    H, W = depth.shape[:2]

    # 1. 深度范围置信度
    conf_range = np.ones_like(depth)
    conf_range[depth < 500] = 0  # 太近
    conf_range[depth > 6000] = 0  # 太远
    # 500-6000 范围内逐渐过渡
    valid_mask = (depth >= 500) & (depth <= 6000)
    conf_range[valid_mask] = 1.0

    # 2. 深度梯度置信度 (边缘处低)
    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 归一化梯度
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)
    conf_edge = 1 - np.clip(grad_norm * 3, 0, 1)  # 边缘越强，置信度越低

    # 3. 计算最终置信度 (取交集)
    confidence = conf_range * conf_edge

    return confidence


def _select_mask_to_target(mask, confidence, valid_mask, target_ratio):
    """根据低置信度优先级，把 mask 调整到目标比例附近。"""
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


def generate_kinect_holes(depth, amplitude, hole_ratio=0.15, block_size=8,
                          amp_threshold=None, amp_percentile=5.0, low_amp_ratio=0.6):
    """
    生成 Kinect 真实风格的空洞

    基于 amplitude 生成空洞：当某一局部区域的大部分像素 amplitude 小于阈值时，该区域设为空洞。
    这模拟了真实 Kinect 的特性：黑色表面、镜面反射、距离过远等会导致红外信号弱，形成空洞。

    Args:
        depth: 深度图 (H, W)
        amplitude: IQ图的A通道 (H, W)，代表红外反射强度
        hole_ratio: 目标空洞比例
        block_size: 局部区域大小（用于判断是否大部分像素amplitude小）
        amp_threshold: amplitude阈值，小于此值认为信号弱
        low_amp_ratio: block内超过该比例的像素amplitude<阈值时，整个block设为空洞

    Returns:
        hole_mask: (H, W), 1=空洞, 0=有效
        confidence: (H, W), Kinect 风格置信度图
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

    amp_high = np.percentile(valid_amp, 95.0)
    conf_amp = np.clip((amplitude - amp_threshold) / (amp_high - amp_threshold + 1e-8), 0, 1)

    # 2. 计算深度范围的置信度
    conf_range = np.ones_like(depth)
    conf_range[depth < min_depth] = 0
    conf_range[depth > max_depth] = 0
    conf_range[valid_mask] = 1.0

    # 3. 计算深度边缘的置信度（边缘处信号可能不稳）
    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)
    conf_edge = 1 - np.clip(grad_norm * 3, 0, 1)

    # 综合置信度
    confidence = conf_amp * conf_range * conf_edge

    # 4. 生成空洞：基于局部区域的 amplitude 中值
    # 如果一个 block 内大部分像素的 amplitude 都小于阈值，则该 block 设为空洞
    hole_mask = np.zeros((H, W), dtype=np.float32)

    # 对每个像素，检查其 block_size x block_size 邻域
    # 使用中值来判断"大部分像素"
    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block_amp = amplitude[i:i+block_size, j:j+block_size]
            block_valid = valid_mask[i:i+block_size, j:j+block_size]
            block_amp_valid = block_amp[block_valid]  # 只考虑有效深度区域

            if len(block_amp_valid) > 0:
                amp_median = np.median(block_amp_valid)
                # 如果中值小于阈值，且 block 内足够多的像素 amplitude 小
                actual_low_amp_ratio = (block_amp_valid < amp_threshold).mean()
                if amp_median < amp_threshold and actual_low_amp_ratio > low_amp_ratio:
                    hole_mask[i:i+block_size, j:j+block_size] = 1

    # 5. 形态学处理：平滑空洞边缘
    kernel = np.ones((3, 3), dtype=np.uint8)
    hole_mask = cv2.morphologyEx(hole_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
    hole_mask = hole_mask.astype(np.float32)

    # 6. 精确调整到目标空洞比例，避免不同样本抖动过大
    hole_mask = _select_mask_to_target(hole_mask, confidence, valid_mask, hole_ratio)

    print(
        "    [DEBUG] Hole generation:"
        f" amp_percentile={amp_percentile:.1f}, adaptive_thr={adaptive_threshold:.4f},"
        f" used_thr={amp_threshold:.4f}, final_ratio={hole_mask.mean():.4f},"
        f" target={hole_ratio:.4f}"
    )

    return hole_mask, confidence


def load_pbrt_sample(ideal_dir, noise_dir, noise_holes_dir, hole_mask_dir, scene, idx, sample_name):
    """加载 PBRT 样本

    如果提供了 noise_holes_dir 和 hole_mask_dir，直接加载预处理好的数据；
    否则从 ideal_IQ 和 noise_IQ 重新生成。
    """
    # 加载 ideal IQ
    channels = []
    for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
        fpath = os.path.join(ideal_dir, scene, idx, f"{sample_name}_{ch}.npy")
        if os.path.exists(fpath):
            channels.append(np.load(fpath))
        else:
            return None, None, None

    ideal_IQ = np.stack(channels, axis=0)  # (6, H, W)

    # 如果有预处理好的数据，直接加载
    if noise_holes_dir is not None and hole_mask_dir is not None:
        # 加载带空洞的 noise IQ
        noise_channels = []
        for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
            fpath = os.path.join(noise_holes_dir, scene, idx, f"{sample_name}_{ch}.npy")
            if os.path.exists(fpath):
                noise_channels.append(np.load(fpath))
            else:
                return None, None, None
        noise_IQ = np.stack(noise_channels, axis=0)

        # 加载空洞 mask
        hole_mask_path = os.path.join(hole_mask_dir, scene, idx, f"{sample_name}.npy")
        if os.path.exists(hole_mask_path):
            hole_mask = np.load(hole_mask_path)
        else:
            return None, None, None

        return ideal_IQ, noise_IQ, hole_mask

    # 否则加载普通 noise IQ
    noise_channels = []
    for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
        fpath = os.path.join(noise_dir, scene, idx, f"{sample_name}_{ch}.npy")
        if os.path.exists(fpath):
            noise_channels.append(np.load(fpath))
        else:
            return None, None, None
    noise_IQ = np.stack(noise_channels, axis=0)

    return ideal_IQ, noise_IQ, None


def load_gt_depth_from_file(scene, idx, sample_name, gt_depth_root):
    """从文件加载 GT 深度"""
    gt_path = os.path.join(gt_depth_root, scene, idx, f"{sample_name}.npy")
    if os.path.exists(gt_path):
        return np.load(gt_path)
    return None


def load_gt_depth(ideal_IQ):
    """从 ideal IQ 估计深度 (用于生成 Kinect 风格空洞)"""
    global depth_estimator
    if 'depth_estimator' not in globals():
        depth_estimator = DepthEstimator()
    depth = depth_estimator.process(ideal_IQ)
    return depth


# =============================================================================
# 空洞填补
# =============================================================================

def ns_inpaint_channel(img, mask, inpaintRadius=3):
    """OpenCV NS 填补单通道"""
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    filled = cv2.inpaint(img_uint8, mask_uint8, inpaintRadius=inpaintRadius, flags=cv2.INPAINT_NS)
    return filled.astype(np.float32) / 255.0


def sd_inpaint_channel(rgb_image, mask, pipe, seed=42):
    """Stable Diffusion Inpaint 填补单通道"""
    from PIL import Image

    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(rgb_uint8)

    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_mask = Image.fromarray(mask_uint8, mode='L')

    generator = torch.manual_seed(seed)
    result = pipe(
        prompt="",
        image=pil_image,
        mask_image=pil_mask,
        num_inference_steps=20,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]

    filled = np.array(result).astype(np.float32) / 255.0
    return filled


def iq_to_rgb_triplet(channel_data, conf_mask=None):
    """单通道复制为 RGB (不改变空洞区域的值)"""
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

    # 不再根据 conf_mask 零掉空洞区域，保持原值让 SD 处理
    rgb = np.stack([ch_norm, ch_norm, ch_norm], axis=-1)
    return rgb


def blend_hole_regions(base_channel, filled_channel, hole_mask):
    """只用填补结果覆盖空洞区域，保留有效区域原值。"""
    hole = hole_mask > 0.5
    return np.where(hole, filled_channel, base_channel).astype(np.float32)


def depth_domain_fill_holes(
    depth,
    hole_mask,
    confidence=None,
    guidance=None,
    method="telea",
    radius=5,
    bilateral_radius=5,
    bilateral_sigma_depth=0.05,
    bilateral_sigma_conf=0.25,
    bilateral_iters=1,
    plane_max_ring_radius=12,
    plane_min_boundary_points=12,
):
    """
    在深度域只填显式空洞区域，非空洞像素保持不变。

    之前的 confidence_fast 会按软置信度阈值改动一部分非空洞像素，
    容易把 valid 区域 MAE 拉高；这个函数用于更干净的 depth-domain baseline。
    """
    hole = hole_mask > 0.5
    if hole.sum() == 0:
        return depth.copy()

    if method == "confidence_fast":
        hole_confidence = (1.0 - hole.astype(np.float32))
        return confidence_fill_depth_fast(depth, hole_confidence, threshold=0.5)

    if method in ["telea", "ns"]:
        return opencv_depth_inpaint(depth, hole, method=method, radius=radius)

    if method == "ns_bilateral":
        return edge_aware_ns_bilateral_fill(
            depth,
            hole,
            confidence=confidence,
            guidance=guidance,
            inpaint_radius=radius,
            bilateral_radius=bilateral_radius,
            sigma_depth=bilateral_sigma_depth,
            sigma_guidance=bilateral_sigma_conf,
            iterations=bilateral_iters,
        )

    if method == "plane":
        return local_plane_fit_fill(
            depth,
            hole,
            confidence=confidence,
            max_ring_radius=plane_max_ring_radius,
            min_boundary_points=plane_min_boundary_points,
            fallback_radius=radius,
        )

    raise ValueError(f"Unsupported depth fill method: {method}")


def weighted_region_consistency(global_mae, hole_mae, valid_mae, hole_ratio):
    """检查全局/空洞/有效区指标是否近似自洽。"""
    if any(np.isnan(x) for x in [global_mae, hole_mae, valid_mae]):
        return np.nan, np.nan
    expected = hole_ratio * hole_mae + (1.0 - hole_ratio) * valid_mae
    return expected, abs(global_mae - expected)


def compute_iq_amplitude_features(iq):
    """Compute observable ToF amplitude features from 6-channel IQ data."""
    i_channels = np.stack([iq[0], iq[2], iq[4]], axis=0)
    q_channels = np.stack([iq[1], iq[3], iq[5]], axis=0)
    amplitude = np.sqrt(i_channels**2 + q_channels**2).astype(np.float32)
    amplitude_mean = amplitude.mean(axis=0).astype(np.float32)
    return amplitude, amplitude_mean


def dilate_binary_mask(mask, radius):
    """Dilate a binary mask by radius pixels using an elliptical kernel."""
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    radius = int(radius)
    if radius <= 0:
        return mask_uint8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask_uint8, kernel).astype(np.uint8)


def save_depth_completion_cache(
    args,
    scene,
    idx,
    sample_name,
    depth_noisy,
    depth_depthcad,
    depth_base,
    gt_depth,
    hole_mask,
    confidence,
    noisy_iq,
    denoised_iq,
    ideal_iq=None,
):
    """Save per-sample tensors for later learned depth completion training."""
    if not args.save_depth_completion_cache:
        return None

    out_path = depth_completion_cache_path(args, scene, idx, sample_name)
    cache_dir = os.path.dirname(out_path)
    os.makedirs(cache_dir, exist_ok=True)

    valid_mask = (
        (gt_depth > 0.1)
        & (gt_depth < 9.9)
        & np.isfinite(gt_depth)
    ).astype(np.uint8)
    hole_mask_uint8 = (hole_mask > 0.5).astype(np.uint8)
    refine_mask = dilate_binary_mask(hole_mask_uint8, args.depth_cache_refine_dilation)

    noisy_amplitude, noisy_amplitude_mean = compute_iq_amplitude_features(noisy_iq)
    denoised_amplitude, denoised_amplitude_mean = compute_iq_amplitude_features(denoised_iq)

    cache = {
        "sample_name": np.array(f"{scene}/{idx}/{sample_name}"),
        "depth_noisy": depth_noisy.astype(np.float32),
        "depth_depthcad": depth_depthcad.astype(np.float32),
        "depth_base": depth_base.astype(np.float32),
        "gt_depth": gt_depth.astype(np.float32),
        "hole_mask": hole_mask_uint8,
        "refine_mask": refine_mask,
        "confidence": confidence.astype(np.float32),
        "valid_mask": valid_mask,
        "noisy_amplitude": noisy_amplitude,
        "noisy_amplitude_mean": noisy_amplitude_mean,
        "denoised_amplitude": denoised_amplitude,
        "denoised_amplitude_mean": denoised_amplitude_mean,
        "depth_fill_method": np.array(args.depth_fill_method),
        "depth_fill_radius": np.array(args.depth_fill_radius, dtype=np.int32),
        "plane_max_ring_radius": np.array(args.plane_max_ring_radius, dtype=np.int32),
        "plane_min_boundary_points": np.array(args.plane_min_boundary_points, dtype=np.int32),
        "iq_cache_only": np.array(bool(args.iq_cache_only), dtype=np.uint8),
    }

    if args.depth_cache_save_iq:
        cache.update({
            "noisy_iq": noisy_iq.astype(np.float32),
            "denoised_iq": denoised_iq.astype(np.float32),
        })
        if ideal_iq is not None:
            cache["ideal_iq"] = ideal_iq.astype(np.float32)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **cache)
    os.replace(tmp_path, out_path)
    print(f"    Saved depth completion cache to {out_path}")
    return out_path


def wrapped_phase_abs_error(pred_phase, target_phase):
    diff = np.angle(np.exp(1j * (pred_phase - target_phase)))
    return np.abs(diff)


def masked_mean_np(values, mask):
    valid = mask & np.isfinite(values)
    if valid.sum() == 0:
        return np.nan
    return float(values[valid].mean())


def compute_iq_physics_metrics(pred_iq, ideal_iq, hole_mask):
    """Measure whether generated IQ preserves per-channel and I/Q-pair physics."""
    hole = hole_mask > 0.5
    metrics = {}

    abs_err = np.abs(pred_iq - ideal_iq)
    metrics["iq_l1_hole_mean"] = masked_mean_np(abs_err.mean(axis=0), hole)
    for ch in range(min(pred_iq.shape[0], ideal_iq.shape[0])):
        metrics[f"iq_l1_hole_ch{ch}"] = masked_mean_np(abs_err[ch], hole)

    for pair_index, (i_idx, q_idx) in enumerate([(0, 1), (2, 3), (4, 5)]):
        pred_i = pred_iq[i_idx]
        pred_q = pred_iq[q_idx]
        ideal_i = ideal_iq[i_idx]
        ideal_q = ideal_iq[q_idx]

        pred_amp = np.sqrt(pred_i ** 2 + pred_q ** 2)
        ideal_amp = np.sqrt(ideal_i ** 2 + ideal_q ** 2)
        pred_phase = np.arctan2(pred_q, pred_i)
        ideal_phase = np.arctan2(ideal_q, ideal_i)
        phase_err = wrapped_phase_abs_error(pred_phase, ideal_phase)

        metrics[f"amp_l1_hole_pair{pair_index}"] = masked_mean_np(np.abs(pred_amp - ideal_amp), hole)
        metrics[f"phase_l1_hole_pair{pair_index}"] = masked_mean_np(phase_err, hole)

    return metrics


def save_sd_diagnostics(
    args,
    scene,
    idx,
    sample_name,
    ideal_iq,
    noisy_iq,
    depth_noisy,
    depth_depthcad,
    depth_sdinpaint,
    depth_full,
    depth_depthfill,
    gt_depth,
    hole_mask,
    confidence,
    pred_iq_denoised,
    filled_iq_sdinpaint,
    filled_iq_full,
):
    """Save tensors that explain why pseudo-RGB SD inpainting succeeds or fails."""
    if not args.save_sd_diagnostics:
        return None

    out_path = sd_diagnostics_path(args, scene, idx, sample_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    hole = hole_mask > 0.5
    valid = (gt_depth > 0.1) & (gt_depth < 9.9) & np.isfinite(gt_depth)
    hole_valid = hole & valid
    valid_nonhole = valid & (~hole)

    method_depths = {
        "noisy": depth_noisy,
        "depthcad": depth_depthcad,
        "sdinpaint": depth_sdinpaint,
        "full": depth_full,
        "depthfill": depth_depthfill,
    }
    metrics = {}
    for name, depth in method_depths.items():
        err = np.abs(depth - gt_depth)
        metrics[f"{name}_depth_l1_hole"] = masked_mean_np(err, hole_valid)
        metrics[f"{name}_depth_l1_valid"] = masked_mean_np(err, valid_nonhole)
        metrics[f"{name}_depth_l1_global"] = masked_mean_np(err, valid)

    for name, iq in [
        ("noisy", noisy_iq),
        ("depthcad", pred_iq_denoised),
        ("sdinpaint", filled_iq_sdinpaint),
        ("full", filled_iq_full),
    ]:
        iq_metrics = compute_iq_physics_metrics(iq, ideal_iq, hole)
        for key, value in iq_metrics.items():
            metrics[f"{name}_{key}"] = value

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            sample_name=np.array(f"{scene}/{idx}/{sample_name}"),
            ideal_iq=ideal_iq.astype(np.float32),
            noisy_iq=noisy_iq.astype(np.float32),
            pred_iq_denoised=pred_iq_denoised.astype(np.float32),
            filled_iq_sdinpaint=filled_iq_sdinpaint.astype(np.float32),
            filled_iq_full=filled_iq_full.astype(np.float32),
            depth_noisy=depth_noisy.astype(np.float32),
            depth_depthcad=depth_depthcad.astype(np.float32),
            depth_sdinpaint=depth_sdinpaint.astype(np.float32),
            depth_full=depth_full.astype(np.float32),
            depth_depthfill=depth_depthfill.astype(np.float32),
            gt_depth=gt_depth.astype(np.float32),
            hole_mask=(hole_mask > 0.5).astype(np.uint8),
            confidence=confidence.astype(np.float32),
            metrics_json=np.array(json.dumps(metrics, sort_keys=True)),
        )
    os.replace(tmp_path, out_path)

    metrics_path = out_path.replace(".npz", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"    Saved SD diagnostics to {out_path}")
    return out_path


# =============================================================================
# 评估指标
# =============================================================================

def compute_metrics(pred, gt, valid_mask=None):
    """计算深度估计指标"""
    # GT 深度范围假设是 0-10 米。不要用 pred 的有效范围过滤，
    # 否则空洞内预测为 0 的失败样本会被全图 MAE 静默排除。
    metric_mask = (gt > 0.1) & (gt < 9.9) & np.isfinite(gt) & np.isfinite(pred)
    if valid_mask is not None:
        metric_mask = metric_mask & valid_mask

    if metric_mask.sum() == 0:
        return np.nan

    pred_valid = pred[metric_mask]
    gt_valid = gt[metric_mask]

    mae = np.abs(pred_valid - gt_valid).mean()

    return mae


def compute_mae_and_count(pred, gt, valid_mask=None):
    """计算 MAE 和实际参与评估的像素数。"""
    metric_mask = (gt > 0.1) & (gt < 9.9) & np.isfinite(gt) & np.isfinite(pred)
    if valid_mask is not None:
        metric_mask = metric_mask & valid_mask

    count = int(metric_mask.sum())
    if count == 0:
        return np.nan, 0

    mae = np.abs(pred[metric_mask] - gt[metric_mask]).mean()
    return float(mae), count


def compute_region_metrics(pred, gt, hole_mask):
    """一次性计算全图、空洞区、有效区 MAE，并检查区域加权是否自洽。"""
    hole_mask = hole_mask.astype(bool)
    valid_mask = ~hole_mask

    global_mae, global_count = compute_mae_and_count(pred, gt)
    hole_mae, hole_count = compute_mae_and_count(pred, gt, hole_mask)
    valid_mae, valid_count = compute_mae_and_count(pred, gt, valid_mask)

    region_count = hole_count + valid_count
    if region_count > 0:
        weighted_sum = 0.0
        if hole_count > 0 and not np.isnan(hole_mae):
            weighted_sum += hole_mae * hole_count
        if valid_count > 0 and not np.isnan(valid_mae):
            weighted_sum += valid_mae * valid_count
        expected = weighted_sum / region_count
        delta = abs(global_mae - expected) if not np.isnan(global_mae) else np.nan
        evaluated_hole_ratio = hole_count / region_count
    else:
        expected = np.nan
        delta = np.nan
        evaluated_hole_ratio = np.nan

    return {
        "global": global_mae,
        "holes": hole_mae,
        "valid": valid_mae,
        "global_count": global_count,
        "hole_count": hole_count,
        "valid_count": valid_count,
        "evaluated_hole_ratio": evaluated_hole_ratio,
        "expected_from_regions": expected,
        "consistency_delta": delta,
    }


# =============================================================================
# 主流程
# =============================================================================

def process_single_sample(scene, idx, sample_name, args, pipe, sd_pipe, estimator, hole_ratio=0.15, save_visualization=False):
    """处理单个样本，返回结果字典

    支持两种模式：
    1. 预处理模式：直接加载预先生成的带空洞图像和空洞mask
    2. 实时生成模式：从ideal_IQ和noise_IQ实时生成空洞
    """
    t_start = time.time()

    target_h, target_w = args.target_size

    # 1. 加载数据
    print(f"    Loading: scene={scene}, idx={idx}, sample={sample_name}")
    result = load_pbrt_sample(
        args.ideal_iq_dir, args.noise_iq_dir,
        args.noise_iq_holes_dir, args.hole_mask_dir,
        scene, idx, sample_name
    )

    if result[0] is None:
        return None

    ideal_IQ, noise_IQ, precomputed_hole_mask = result

    print(f"    Loaded: ideal_IQ={ideal_IQ.shape}, noise_IQ={noise_IQ.shape}")

    # 确保所有数据都是 target_size
    actual_h, actual_w = ideal_IQ.shape[1:]
    if (actual_h, actual_w) != (target_h, target_w):
        print(f"    Resizing ideal_IQ from {(actual_h, actual_w)} to {(target_h, target_w)}")
        ideal_IQ_orig = ideal_IQ
        ideal_IQ = np.zeros((6, target_h, target_w), dtype=np.float32)
        for i in range(6):
            ideal_IQ[i] = cv2.resize(ideal_IQ_orig[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    actual_h, actual_w = noise_IQ.shape[1:]
    if (actual_h, actual_w) != (target_h, target_w):
        print(f"    Resizing noise_IQ from {(actual_h, actual_w)} to {(target_h, target_w)}")
        noise_IQ_orig = noise_IQ
        noise_IQ = np.zeros((6, target_h, target_w), dtype=np.float32)
        for i in range(6):
            noise_IQ[i] = cv2.resize(noise_IQ_orig[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 2. 确定空洞mask
    if precomputed_hole_mask is not None:
        # 预处理模式：直接使用预先生成的空洞mask
        hole_mask = precomputed_hole_mask
        if hole_mask.shape != (target_h, target_w):
            hole_mask = cv2.resize(hole_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        kinect_conf = 1.0 - hole_mask
        print(f"    Using precomputed hole mask, shape: {hole_mask.shape}")
    else:
        # 实时生成模式：根据ideal_IQ生成空洞
        t0 = time.time()
        depth_est = load_gt_depth(ideal_IQ)
        I = np.stack([ideal_IQ[0], ideal_IQ[2], ideal_IQ[4]], axis=0)
        Q = np.stack([ideal_IQ[1], ideal_IQ[3], ideal_IQ[5]], axis=0)
        amplitude = np.sqrt(I**2 + Q**2)
        amplitude_mean = amplitude.mean(axis=0)
        hole_mask, kinect_conf = generate_kinect_holes(
            depth_est, amplitude_mean, hole_ratio=hole_ratio,
            block_size=args.block_size,
            amp_threshold=args.amp_threshold,
            amp_percentile=args.amp_percentile,
            low_amp_ratio=args.low_amp_ratio
        )
        print(f"    [T] Kinect holes generated: {time.time()-t0:.1f}s")

    # 确保空洞mask尺寸一致
    if hole_mask.shape != (target_h, target_w):
        hole_mask = cv2.resize(hole_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if kinect_conf.shape != (target_h, target_w):
        kinect_conf = cv2.resize(kinect_conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 3. 带空洞的噪声图。预处理模式下通常已经置零，这里再应用一次保证语义一致。
    noisy_input = noise_IQ.copy()
    noisy_input[:, hole_mask > 0.5] = 0
    combined_conf = np.minimum(kinect_conf, 1.0 - hole_mask)

    print(f"    Hole ratio: {hole_mask.mean():.4f}, Combined confidence shape: {combined_conf.shape}")

    if args.iq_cache_only:
        if not (args.save_depth_completion_cache and args.depth_cache_save_iq):
            raise ValueError("--iq_cache_only requires --save_depth_completion_cache and --depth_cache_save_iq")

        t3 = time.time()
        gt_depth = load_gt_depth_from_file(scene, idx, sample_name, args.gt_depth_dir)
        if gt_depth is not None:
            if gt_depth.shape != (target_h, target_w):
                gt_depth = cv2.resize(gt_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            gt_depth_resized = gt_depth
            print(f"    GT depth loaded from file: range [{gt_depth.min():.4f}, {gt_depth.max():.4f}]")
        else:
            gt_depth = load_gt_depth(ideal_IQ)
            gt_depth_resized = cv2.resize(gt_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            print(f"    GT depth estimated from IQ: range [{gt_depth.min():.4f}, {gt_depth.max():.4f}]")

        depth_noisy = estimator.process(noisy_input)
        hole_mask_final = cv2.resize(hole_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        confidence_final = cv2.resize(combined_conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        low_conf_mask = hole_mask_final > 0.5
        noisy_metrics = compute_region_metrics(depth_noisy, gt_depth_resized, low_conf_mask)
        print(f"    [T] IQ-cache-only depth estimation: {time.time()-t3:.1f}s")

        cache_path = save_depth_completion_cache(
            args=args,
            scene=scene,
            idx=idx,
            sample_name=sample_name,
            depth_noisy=depth_noisy,
            # These fields are duplicated intentionally so downstream readers that
            # expect a depth cache do not crash. The iq_cache_only flag marks that
            # no DepthCAD/depth-fill baseline was actually computed.
            depth_depthcad=depth_noisy.copy(),
            depth_base=depth_noisy.copy(),
            gt_depth=gt_depth_resized,
            hole_mask=hole_mask_final,
            confidence=confidence_final,
            noisy_iq=noisy_input,
            denoised_iq=noisy_input.copy(),
            ideal_iq=ideal_IQ,
        )

        result = {
            'sample_name': f"{scene}/{idx}/{sample_name}",
            'mae_noisy': noisy_metrics['global'],
            'mae_depthcad': np.nan,
            'mae_sdinpaint': np.nan,
            'mae_full': np.nan,
            'mae_depthfill': np.nan,
            'depth_fill_method': args.depth_fill_method,
            'iq_cache_only': True,
            'hole_ratio': float(hole_mask.mean()),
            'evaluated_hole_ratio': float(noisy_metrics['evaluated_hole_ratio']),
            'hole_pixel_count': int(low_conf_mask.sum()),
            'eval_pixel_count': int(noisy_metrics['global_count']),
            'total_time': time.time() - t_start,
        }
        if cache_path is not None:
            result['depth_completion_cache_path'] = cache_path

        result.update({
            'mae_noisy_holes': noisy_metrics['holes'],
            'mae_noisy_valid': noisy_metrics['valid'],
            'mae_noisy_eval_pixel_count': noisy_metrics['global_count'],
            'mae_noisy_hole_eval_pixel_count': noisy_metrics['hole_count'],
            'mae_noisy_valid_eval_pixel_count': noisy_metrics['valid_count'],
            'mae_noisy_evaluated_hole_ratio': noisy_metrics['evaluated_hole_ratio'],
            'mae_noisy_expected_from_regions': noisy_metrics['expected_from_regions'],
            'mae_noisy_consistency_delta': noisy_metrics['consistency_delta'],
        })
        for name in ['depthcad', 'sdinpaint', 'full', 'depthfill']:
            result.update({
                f'mae_{name}_holes': np.nan,
                f'mae_{name}_valid': np.nan,
                f'mae_{name}_eval_pixel_count': 0,
                f'mae_{name}_hole_eval_pixel_count': 0,
                f'mae_{name}_valid_eval_pixel_count': 0,
                f'mae_{name}_evaluated_hole_ratio': np.nan,
                f'mae_{name}_expected_from_regions': np.nan,
                f'mae_{name}_consistency_delta': np.nan,
            })

        print(f"    [T] Total sample time: {time.time()-t_start:.1f}s")
        return result

    # 4. ControlNet 去噪 (基于填补后的数据和置信度)
    t1 = time.time()
    scale = max(noisy_input.max(), abs(noisy_input.min()), 1e-8)
    noise_norm = noisy_input / scale

    infer_size = 512
    hole_mask_512 = cv2.resize(hole_mask, (infer_size, infer_size), interpolation=cv2.INTER_NEAREST)
    pred_IQs_denoised = np.zeros((6, infer_size, infer_size), dtype=np.float32)
    conf_resized = cv2.resize(combined_conf, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)

    for i in range(6):
        noise_resized = cv2.resize(noise_norm[i], (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        guidance = np.stack([noise_resized, conf_resized], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0).to("cuda")

        generator = torch.manual_seed(args.seed)
        pred_IQ = pipe(
            prompt="",
            num_inference_steps=20,
            generator=generator,
            image=guidance,
            height=infer_size,
            width=infer_size
        ).images[0]

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0
        pred_IQ = 2 * pred_IQ - 1
        pred_IQs_denoised[i] = pred_IQ * scale

    print(f"    [T] ControlNet denoising (6 ch): {time.time()-t1:.1f}s")

    # Resize 回目标尺寸
    pred_IQs_denoised_resized = np.zeros((6, target_h, target_w), dtype=np.float32)
    for i in range(6):
        pred_IQs_denoised_resized[i] = cv2.resize(pred_IQs_denoised[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 5. SD Inpaint only (直接填补空洞，不去噪)
    print(f"    [Mode B] SD Inpaint only...")
    t_b = time.time()
    filled_IQs_sdinpaint = np.zeros((6, infer_size, infer_size), dtype=np.float32)

    for i in range(6):
        # noisy_input 是 target_size，需要 resize 到 infer_size
        ch_noise = cv2.resize(noisy_input[i], (infer_size, infer_size), interpolation=cv2.INTER_LINEAR) / scale
        ch_min, ch_max = ch_noise.min(), ch_noise.max()
        ch_norm = (ch_noise - ch_min) / (ch_max - ch_min + 1e-8)
        # 使用 resized 的 conf
        rgb = iq_to_rgb_triplet(ch_norm, None)  # 不使用 conf_mask，避免尺寸不匹配
        rgb_filled = sd_inpaint_channel(rgb, hole_mask_512, sd_pipe, seed=args.seed)
        filled_channel = rgb_filled[:, :, 0] * (ch_max - ch_min) + ch_min
        filled_IQs_sdinpaint[i] = blend_hole_regions(ch_noise, filled_channel, hole_mask_512)

    print(f"    [T] SD Inpaint only (6 ch): {time.time()-t_b:.1f}s")

    filled_IQs_sdinpaint_resized = np.zeros((6, target_h, target_w), dtype=np.float32)
    for i in range(6):
        filled_IQs_sdinpaint_resized[i] = cv2.resize(filled_IQs_sdinpaint[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 6. DepthCAD + SD Inpaint (完整流程)
    print(f"    [Mode C] DepthCAD + SD Inpaint...")
    t_c = time.time()
    filled_IQs_full = np.zeros((6, infer_size, infer_size), dtype=np.float32)

    for i in range(6):
        ch_denoised = pred_IQs_denoised[i]
        ch_min, ch_max = ch_denoised.min(), ch_denoised.max()
        ch_norm = (ch_denoised - ch_min) / (ch_max - ch_min + 1e-8)
        rgb = iq_to_rgb_triplet(ch_norm, conf_resized)
        rgb_filled = sd_inpaint_channel(rgb, hole_mask_512, sd_pipe, seed=args.seed)
        filled_channel = rgb_filled[:, :, 0] * (ch_max - ch_min) + ch_min
        filled_IQs_full[i] = blend_hole_regions(ch_denoised, filled_channel, hole_mask_512)

    print(f"    [T] DepthCAD + SD Inpaint (6 ch): {time.time()-t_c:.1f}s")

    filled_IQs_full_resized = np.zeros((6, target_h, target_w), dtype=np.float32)
    for i in range(6):
        filled_IQs_full_resized[i] = cv2.resize(filled_IQs_full[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 7. 计算深度和 MAE
    t3 = time.time()
    gt_depth = load_gt_depth_from_file(scene, idx, sample_name, args.gt_depth_dir)
    if gt_depth is not None:
        if gt_depth.shape != (target_h, target_w):
            gt_depth = cv2.resize(gt_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        gt_depth_resized = gt_depth
        print(f"    GT depth loaded from file: range [{gt_depth.min():.4f}, {gt_depth.max():.4f}]")
    else:
        gt_depth = load_gt_depth(ideal_IQ)
        gt_depth_resized = cv2.resize(gt_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"    GT depth estimated from IQ: range [{gt_depth.min():.4f}, {gt_depth.max():.4f}]")

    depth_noisy = estimator.process(noisy_input)
    depth_depthcad = estimator.process(pred_IQs_denoised_resized)
    depth_sdinpaint = estimator.process(filled_IQs_sdinpaint_resized)
    depth_full = estimator.process(filled_IQs_full_resized)
    hole_mask_final = cv2.resize(hole_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    confidence_final = cv2.resize(combined_conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    depth_depthfill = depth_domain_fill_holes(
        depth_depthcad,
        hole_mask_final,
        confidence=confidence_final,
        method=args.depth_fill_method,
        radius=args.depth_fill_radius,
        bilateral_radius=args.bilateral_radius,
        bilateral_sigma_depth=args.bilateral_sigma_depth,
        bilateral_sigma_conf=args.bilateral_sigma_conf,
        bilateral_iters=args.bilateral_iters,
        plane_max_ring_radius=args.plane_max_ring_radius,
        plane_min_boundary_points=args.plane_min_boundary_points,
    )

    # 调试：打印深度范围
    print(f"    [DEBUG] Noisy depth: range [{depth_noisy.min():.4f}, {depth_noisy.max():.4f}]")
    print(f"    [DEBUG] DepthCAD depth: range [{depth_depthcad.min():.4f}, {depth_depthcad.max():.4f}]")
    print(f"    [DEBUG] SDInpaint depth: range [{depth_sdinpaint.min():.4f}, {depth_sdinpaint.max():.4f}]")
    print(f"    [DEBUG] Full depth: range [{depth_full.min():.4f}, {depth_full.max():.4f}]")
    print(f"    [DEBUG] DepthFill ({args.depth_fill_method}) depth: range [{depth_depthfill.min():.4f}, {depth_depthfill.max():.4f}]")
    print(f"    [DEBUG] Hole ratio: {hole_mask.mean():.4f}")
    print(f"    [T] Depth estimation (4 runs): {time.time()-t3:.1f}s")

    low_conf_mask = hole_mask_final > 0.5

    # 计算指标：保留像素计数，避免全图/空洞区/有效区不自洽时被忽略。
    method_depths = {
        'noisy': depth_noisy,
        'depthcad': depth_depthcad,
        'sdinpaint': depth_sdinpaint,
        'full': depth_full,
        'depthfill': depth_depthfill,
    }
    region_metrics = {
        name: compute_region_metrics(depth, gt_depth_resized, low_conf_mask)
        for name, depth in method_depths.items()
    }

    mae_noisy = region_metrics['noisy']['global']
    mae_depthcad = region_metrics['depthcad']['global']
    mae_sdinpaint = region_metrics['sdinpaint']['global']
    mae_full = region_metrics['full']['global']
    mae_depthfill = region_metrics['depthfill']['global']

    result = {
        'sample_name': f"{scene}/{idx}/{sample_name}",
        'mae_noisy': mae_noisy,
        'mae_depthcad': mae_depthcad,
        'mae_sdinpaint': mae_sdinpaint,
        'mae_full': mae_full,
        'mae_depthfill': mae_depthfill,
        'depth_fill_method': args.depth_fill_method,
        'hole_ratio': float(hole_mask.mean()),
        'evaluated_hole_ratio': float(region_metrics['noisy']['evaluated_hole_ratio']),
        'hole_pixel_count': int(low_conf_mask.sum()),
        'eval_pixel_count': int(region_metrics['noisy']['global_count']),
        'total_time': time.time() - t_start,
    }

    cache_path = save_depth_completion_cache(
        args=args,
        scene=scene,
        idx=idx,
        sample_name=sample_name,
        depth_noisy=depth_noisy,
        depth_depthcad=depth_depthcad,
        depth_base=depth_depthfill,
        gt_depth=gt_depth_resized,
        hole_mask=hole_mask_final,
        confidence=confidence_final,
        noisy_iq=noisy_input,
        denoised_iq=pred_IQs_denoised_resized,
        ideal_iq=ideal_IQ,
    )
    if cache_path is not None:
        result['depth_completion_cache_path'] = cache_path

    sd_diag_path = save_sd_diagnostics(
        args=args,
        scene=scene,
        idx=idx,
        sample_name=sample_name,
        ideal_iq=ideal_IQ,
        noisy_iq=noisy_input,
        depth_noisy=depth_noisy,
        depth_depthcad=depth_depthcad,
        depth_sdinpaint=depth_sdinpaint,
        depth_full=depth_full,
        depth_depthfill=depth_depthfill,
        gt_depth=gt_depth_resized,
        hole_mask=hole_mask_final,
        confidence=confidence_final,
        pred_iq_denoised=pred_IQs_denoised_resized,
        filled_iq_sdinpaint=filled_IQs_sdinpaint_resized,
        filled_iq_full=filled_IQs_full_resized,
    )
    if sd_diag_path is not None:
        result['sd_diagnostics_path'] = sd_diag_path

    for name, metrics in region_metrics.items():
        result.update({
            f'mae_{name}_holes': metrics['holes'],
            f'mae_{name}_valid': metrics['valid'],
            f'mae_{name}_eval_pixel_count': metrics['global_count'],
            f'mae_{name}_hole_eval_pixel_count': metrics['hole_count'],
            f'mae_{name}_valid_eval_pixel_count': metrics['valid_count'],
            f'mae_{name}_evaluated_hole_ratio': metrics['evaluated_hole_ratio'],
            f'mae_{name}_expected_from_regions': metrics['expected_from_regions'],
            f'mae_{name}_consistency_delta': metrics['consistency_delta'],
        })

    # 保存可视化
    if save_visualization:
        import matplotlib.pyplot as plt

        sample_name_safe = sample_name.replace('/', '_')
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        gt_vis_mask = (gt_depth_resized > 0.1) & (gt_depth_resized < 9.9) & np.isfinite(gt_depth_resized)
        if gt_vis_mask.sum() > 0:
            gt_vmin = float(gt_depth_resized[gt_vis_mask].min())
            gt_vmax = float(gt_depth_resized[gt_vis_mask].max())
        else:
            gt_vmin = float(np.nanmin(gt_depth_resized))
            gt_vmax = float(np.nanmax(gt_depth_resized))

        depth_vis_kwargs = {'cmap': 'turbo'}
        if np.isfinite(gt_vmin) and np.isfinite(gt_vmax) and gt_vmax > gt_vmin:
            depth_vis_kwargs.update({'vmin': gt_vmin, 'vmax': gt_vmax})

        axes[0, 0].imshow(gt_depth_resized, **depth_vis_kwargs)
        axes[0, 0].set_title(f'GT Depth')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(depth_noisy, **depth_vis_kwargs)
        axes[0, 1].set_title(f'Noisy\nMAE={mae_noisy:.4f}')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(depth_depthcad, **depth_vis_kwargs)
        axes[0, 2].set_title(f'DepthCAD\nMAE={mae_depthcad:.4f}')
        axes[0, 2].axis('off')

        axes[0, 3].imshow(depth_sdinpaint, **depth_vis_kwargs)
        axes[0, 3].set_title(f'SD Inpaint\nMAE={mae_sdinpaint:.4f}')
        axes[0, 3].axis('off')

        axes[0, 4].imshow(depth_depthfill, **depth_vis_kwargs)
        axes[0, 4].set_title(f'Depth Fill\nMAE={mae_depthfill:.4f}')
        axes[0, 4].axis('off')

        axes[1, 0].imshow(hole_mask_final, cmap='gray')
        axes[1, 0].set_title('Hole Mask')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(np.abs(depth_noisy - gt_depth_resized), cmap='hot')
        axes[1, 1].set_title('|Noisy - GT|')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(np.abs(depth_depthcad - gt_depth_resized), cmap='hot')
        axes[1, 2].set_title('|DepthCAD - GT|')
        axes[1, 2].axis('off')

        axes[1, 3].imshow(np.abs(depth_sdinpaint - gt_depth_resized), cmap='hot')
        axes[1, 3].set_title('|SD Inpaint - GT|')
        axes[1, 3].axis('off')

        axes[1, 4].imshow(np.abs(depth_depthfill - gt_depth_resized), cmap='hot')
        axes[1, 4].set_title('|Depth Fill - GT|')
        axes[1, 4].axis('off')

        axes[2, 0].imshow(depth_full, **depth_vis_kwargs)
        axes[2, 0].set_title(f'Full\nMAE={mae_full:.4f}')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(np.abs(depth_full - gt_depth_resized), cmap='hot')
        axes[2, 1].set_title('|Full - GT|')
        axes[2, 1].axis('off')

        diff_depthcad_sdinpaint = np.abs(depth_depthcad - depth_sdinpaint)
        axes[2, 2].imshow(diff_depthcad_sdinpaint, cmap='hot')
        axes[2, 2].set_title('|DepthCAD - SDInpaint|')
        axes[2, 2].axis('off')

        diff_depthcad_full = np.abs(depth_depthcad - depth_full)
        axes[2, 3].imshow(diff_depthcad_full, cmap='hot')
        axes[2, 3].set_title('|DepthCAD - Full|')
        axes[2, 3].axis('off')

        diff_depthcad_depthfill = np.abs(depth_depthcad - depth_depthfill)
        axes[2, 4].imshow(diff_depthcad_depthfill, cmap='hot')
        axes[2, 4].set_title('|DepthCAD - DepthFill|')
        axes[2, 4].axis('off')

        plt.tight_layout()
        out_png = os.path.join(args.output_dir, f'vis_{scene}_{idx}_{sample_name_safe}.png')
        plt.savefig(out_png, dpi=150)
        print(f"    Saved visualization to {out_png}")
        plt.close()

    print(f"    [T] Total sample time: {time.time()-t_start:.1f}s")
    return result


def main():
    args = parse_args()

    print("=" * 60)
    if args.iq_cache_only:
        print("Kinect Hole Simulation: IQ Cache Only")
        print("  Saves noisy IQ, confidence, hole mask, GT depth, and noisy depth")
        print("  Skips DepthCAD, SD Inpaint, and depth-domain fill")
    else:
        print("Kinect Hole Simulation: 5-Mode Comparison")
        print("  Mode A: DepthCAD only (denoise, no hole filling)")
        print("  Mode B: SD Inpaint only (fill holes, no denoising)")
        print("  Mode C: DepthCAD + SD Inpaint (full pipeline)")
        print("  Mode D: DepthCAD + Depth-domain fill (hole-mask only)")
    print("=" * 60)

    if args.iq_cache_only and not (args.save_depth_completion_cache and args.depth_cache_save_iq):
        raise ValueError("--iq_cache_only requires --save_depth_completion_cache and --depth_cache_save_iq")

    # 收集样本
    samples = collect_samples(args)
    print(f"\nFound {len(samples)} samples to process")

    if len(samples) == 0:
        print("No samples found!")
        return

    prepare_output_dir(args)
    prepare_depth_cache_dir(args)
    prepare_sd_diagnostics_dir(args)
    save_depth_cache_manifest(args, samples)
    samples = filter_resume_depth_cache_samples(args, samples)
    if len(samples) == 0:
        print("All selected depth completion cache samples are already complete. Nothing to process.")
        return

    # 加载模型 (只加载一次)
    print("\n[Loading Models]")
    MODEL_DIR = "stabilityai/stable-diffusion-2-1"

    if args.iq_cache_only:
        print("  IQ-cache-only mode: skipping ControlNet and SD Inpaint.")
        pipe = None
        sd_pipe = None
    else:
        ensure_diffusion_dependencies()

        print("  Loading ControlNet...")
        depthcad = ControlNetModel.from_pretrained(args.checkpoint_path, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_DIR, controlnet=depthcad, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.to("cuda")

        print("  Loading SD Inpaint...")
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        )
        sd_pipe = sd_pipe.to("cuda")

    print("  Loading Depth Estimator...")
    estimator = DepthEstimator()

    # 处理每个样本
    all_results = []
    for i, (scene, idx, sample_name) in enumerate(tqdm(samples, desc="Processing samples")):
        print(f"\n[{i+1}/{len(samples)}] Processing {scene}/{idx}/{sample_name}")

        try:
            result = process_single_sample(scene, idx, sample_name, args, pipe, sd_pipe, estimator,
                                           hole_ratio=args.hole_ratio,
                                           save_visualization=args.visualize)
            if result:
                all_results.append(result)
                if args.iq_cache_only:
                    print(
                        f"  IQ cache: noisy_mae={result['mae_noisy']:.4f}, "
                        f"hole_mae={result.get('mae_noisy_holes', np.nan):.4f}, "
                        f"time={result['total_time']:.1f}s"
                    )
                else:
                    print(
                        f"  MAE: noisy={result['mae_noisy']:.4f}, "
                        f"depthcad={result['mae_depthcad']:.4f}, "
                        f"sdinpaint={result['mae_sdinpaint']:.4f}, "
                        f"full={result['mae_full']:.4f}, "
                        f"depthfill={result['mae_depthfill']:.4f} | "
                        f"time={result['total_time']:.1f}s"
                    )
                consistency_deltas = [
                    result.get(f'mae_{name}_consistency_delta', np.nan)
                    for name in ['noisy', 'depthcad', 'sdinpaint', 'full', 'depthfill']
                ]
                consistency_deltas = [d for d in consistency_deltas if not np.isnan(d)]
                if consistency_deltas and max(consistency_deltas) > 1e-4:
                    print(f"  WARNING: metric consistency delta is high: {max(consistency_deltas):.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # 汇总结果
    print("\n" + "=" * 60)
    if args.iq_cache_only:
        print("Aggregated Results (IQ cache only)")
    else:
        print("Aggregated Results (5 modes comparison)")
    print("=" * 60)

    if len(all_results) == 0:
        print("No samples processed successfully!")
        return

    # 转换 per_sample_results 中的 numpy 类型为 Python 原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    if args.iq_cache_only:
        avg_mae_noisy = np.mean([r['mae_noisy'] for r in all_results])
        avg_mae_noisy_holes = np.nanmean([r.get('mae_noisy_holes', np.nan) for r in all_results])
        avg_mae_noisy_valid = np.nanmean([r.get('mae_noisy_valid', np.nan) for r in all_results])

        print(f"\n[IQ Cache Only - Average over {len(all_results)} samples]")
        print(f"  Noisy (with holes) global MAE: {avg_mae_noisy:.4f} (+/- {np.std([r['mae_noisy'] for r in all_results]):.4f})")
        print(f"  Noisy hole-region MAE:        {avg_mae_noisy_holes:.4f}")
        print(f"  Noisy valid-region MAE:       {avg_mae_noisy_valid:.4f}")
        print("  DepthCAD / SD Inpaint / depth fill were intentionally skipped.")

        summary = {
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'output_dir': args.output_dir,
            'args': vars(args),
            'metric_note': 'IQ-cache-only run; legacy DepthCAD/SD/depth-fill baselines were not computed.',
            'depth_completion_cache_dir': args.depth_cache_dir,
            'num_samples': len(all_results),
            'avg_mae_noisy': float(avg_mae_noisy),
            'avg_mae_noisy_holes': float(avg_mae_noisy_holes),
            'avg_mae_noisy_valid': float(avg_mae_noisy_valid),
            'std_mae_noisy': float(np.std([r['mae_noisy'] for r in all_results])),
            'per_sample_results': convert_to_serializable(all_results),
        }
        with open(os.path.join(args.output_dir, 'mae_results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        for result in all_results:
            sample_name = result['sample_name'].replace('/', '_')
            result_serializable = convert_to_serializable(result)
            with open(os.path.join(args.output_dir, f'result_{sample_name}.json'), 'w') as f:
                json.dump(result_serializable, f, indent=2)

        print(f"\nResults saved to {args.output_dir}/mae_results_summary.json")
        print("\nDone!")
        return

    # 计算平均 MAE
    avg_mae_noisy = np.mean([r['mae_noisy'] for r in all_results])
    avg_mae_depthcad = np.mean([r['mae_depthcad'] for r in all_results])
    avg_mae_sdinpaint = np.mean([r['mae_sdinpaint'] for r in all_results])
    avg_mae_full = np.mean([r['mae_full'] for r in all_results])
    avg_mae_depthfill = np.mean([r['mae_depthfill'] for r in all_results])

    print(f"\n[Global MAE - Average over {len(all_results)} samples]")
    print(f"  Noisy (with holes):      {avg_mae_noisy:.4f} (+/- {np.std([r['mae_noisy'] for r in all_results]):.4f})")
    print(f"  DepthCAD only:            {avg_mae_depthcad:.4f} (+/- {np.std([r['mae_depthcad'] for r in all_results]):.4f})")
    print(f"  SD Inpaint only:         {avg_mae_sdinpaint:.4f} (+/- {np.std([r['mae_sdinpaint'] for r in all_results]):.4f})")
    print(f"  DepthCAD + SD Inpaint:   {avg_mae_full:.4f} (+/- {np.std([r['mae_full'] for r in all_results]):.4f})")
    print(f"  DepthCAD + Depth Fill:   {avg_mae_depthfill:.4f} (+/- {np.std([r['mae_depthfill'] for r in all_results]):.4f})")

    def is_finite_metric(value):
        return value is not None and np.isfinite(value)

    def mean_metric(results, key):
        values = [r.get(key, np.nan) for r in results]
        values = [v for v in values if is_finite_metric(v)]
        return float(np.mean(values)) if values else np.nan

    print(f"\n[Key Comparisons]")
    print(f"  DepthCAD 去噪效果 (Noisy -> DepthCAD): {(avg_mae_noisy - avg_mae_depthcad) / avg_mae_noisy * 100:.1f}% 改善")
    print(f"  SD Inpaint 去噪+填补效果 (Noisy -> SDInpaint): {(avg_mae_noisy - avg_mae_sdinpaint) / avg_mae_noisy * 100:.1f}% 改善")
    print(f"  Depth-domain 填补效果 (DepthCAD -> DepthFill): {(avg_mae_depthcad - avg_mae_depthfill) / max(avg_mae_depthcad, 1e-8) * 100:.1f}% 改善")
    print(f"  SD Inpaint 单独填补 (Noisy holes区域): ", end="")
    if 'mae_noisy_holes' in all_results[0]:
        valid_sdinpaint_hole_results = [
            r for r in all_results
            if is_finite_metric(r.get('mae_noisy_holes')) and is_finite_metric(r.get('mae_sdinpaint_holes'))
        ]
        if valid_sdinpaint_hole_results:
            print(
                f"空洞区域 MAE: {mean_metric(valid_sdinpaint_hole_results, 'mae_noisy_holes'):.4f} "
                f"-> {mean_metric(valid_sdinpaint_hole_results, 'mae_sdinpaint_holes'):.4f} "
                f"(valid holes: {len(valid_sdinpaint_hole_results)}/{len(all_results)})"
            )
        else:
            print("N/A")
    else:
        print("N/A")

    # 空洞区域
    hole_metric_keys = [
        'mae_noisy_holes',
        'mae_depthcad_holes',
        'mae_sdinpaint_holes',
        'mae_full_holes',
        'mae_depthfill_holes',
    ]
    hole_results = [
        r for r in all_results
        if all(is_finite_metric(r.get(key)) for key in hole_metric_keys)
    ]
    if hole_results:
        avg_mae_noisy_holes = mean_metric(hole_results, 'mae_noisy_holes')
        avg_mae_depthcad_holes = mean_metric(hole_results, 'mae_depthcad_holes')
        avg_mae_sdinpaint_holes = mean_metric(hole_results, 'mae_sdinpaint_holes')
        avg_mae_full_holes = mean_metric(hole_results, 'mae_full_holes')
        avg_mae_depthfill_holes = mean_metric(hole_results, 'mae_depthfill_holes')

        print(f"\n[Hole Region MAE - Average over {len(hole_results)}/{len(all_results)} samples with valid holes]")
        print(f"  Noisy (with holes):      {avg_mae_noisy_holes:.4f}")
        print(f"  DepthCAD only:           {avg_mae_depthcad_holes:.4f}")
        print(f"  SD Inpaint only:          {avg_mae_sdinpaint_holes:.4f}")
        print(f"  DepthCAD + SD Inpaint:    {avg_mae_full_holes:.4f}")
        print(f"  DepthCAD + Depth Fill:    {avg_mae_depthfill_holes:.4f}")

        consistency_keys = [k for k in all_results[0].keys() if k.endswith('_consistency_delta')]
        if consistency_keys:
            print(f"\n[Metric Consistency Check]")
            for key in consistency_keys:
                print(f"  {key}: {np.nanmean([r.get(key, np.nan) for r in all_results]):.6f}")

    # 保存汇总结果
    summary = {
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'output_dir': args.output_dir,
        'args': vars(args),
        'metric_note': 'MAE is computed on GT-valid finite pixels only; prediction value range is not used for filtering.',
        'depth_completion_cache_dir': args.depth_cache_dir if args.save_depth_completion_cache else None,
        'num_samples': len(all_results),
        'avg_mae_noisy': float(avg_mae_noisy),
        'avg_mae_depthcad': float(avg_mae_depthcad),
        'avg_mae_sdinpaint': float(avg_mae_sdinpaint),
        'avg_mae_full': float(avg_mae_full),
        'avg_mae_depthfill': float(avg_mae_depthfill),
        'std_mae_noisy': float(np.std([r['mae_noisy'] for r in all_results])),
        'std_mae_depthcad': float(np.std([r['mae_depthcad'] for r in all_results])),
        'std_mae_sdinpaint': float(np.std([r['mae_sdinpaint'] for r in all_results])),
        'std_mae_full': float(np.std([r['mae_full'] for r in all_results])),
        'std_mae_depthfill': float(np.std([r['mae_depthfill'] for r in all_results])),
    }
    if hole_results:
        summary.update({
            'num_samples_with_valid_hole_metrics': len(hole_results),
            'num_samples_without_valid_hole_metrics': len(all_results) - len(hole_results),
            'avg_mae_noisy_holes': float(avg_mae_noisy_holes),
            'avg_mae_depthcad_holes': float(avg_mae_depthcad_holes),
            'avg_mae_sdinpaint_holes': float(avg_mae_sdinpaint_holes),
            'avg_mae_full_holes': float(avg_mae_full_holes),
            'avg_mae_depthfill_holes': float(avg_mae_depthfill_holes),
        })

    summary['per_sample_results'] = convert_to_serializable(all_results)

    with open(os.path.join(args.output_dir, 'mae_results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 保存每个样本的单独结果
    for result in all_results:
        sample_name = result['sample_name'].replace('/', '_')
        result_serializable = convert_to_serializable(result)
        with open(os.path.join(args.output_dir, f'result_{sample_name}.json'), 'w') as f:
            json.dump(result_serializable, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/mae_results_summary.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
