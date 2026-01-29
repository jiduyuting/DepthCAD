import os
import argparse
import numpy as np
import csv
import cv2

# Try to import mask-enabled preprocessing
try:
    from pbrt_dataset.process_mask import load_raw as load_raw_with_mask, compute_gradient_confidence
    MASK_AVAILABLE = True
except ImportError:
    try:
        from DepthCAD.pbrt_dataset.preprocess import load_raw as load_raw_with_mask, compute_gradient_confidence
        MASK_AVAILABLE = True
    except ImportError:
        MASK_AVAILABLE = False

def data_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith('.npy'):
        data = np.load(path)
    elif path.endswith('.bin'):
        shape = (240, 320)
        data = np.fromfile(path, dtype=np.float32).reshape(shape)
    else:
        try:
            data = np.load(path)
        except:
            raise ValueError(f"Unknown file format: {path}")

    data = np.nan_to_num(data, 0)
    return data

def find_matching_gt(pred_rel_path, gt_dir):
    """
    根据预测文件的相对路径，去 GT 目录找对应的真值文件。
    """
    base_rel = os.path.splitext(pred_rel_path)[0] # "bathroom/1/115"
    
    # 1. 尝试绝对匹配
    gt_path_v1 = os.path.join(gt_dir, base_rel + ".npy")
    if os.path.exists(gt_path_v1):
        return gt_path_v1
    
    gt_path_v2 = os.path.join(gt_dir, base_rel + ".bin")
    if os.path.exists(gt_path_v2):
        return gt_path_v2

    # 2. 尝试去父目录找 (例如 bathroom/1/ 下的 depth.bin)
    parent_rel = os.path.dirname(pred_rel_path)
    gt_parent_dir = os.path.join(gt_dir, parent_rel)
    
    if os.path.isdir(gt_parent_dir):
        files = os.listdir(gt_parent_dir)
        if "depth.bin" in files:
            return os.path.join(gt_parent_dir, "depth.bin")
        
        bin_files = [f for f in files if f.endswith('.bin')]
        if bin_files:
            return os.path.join(gt_parent_dir, bin_files[0])
            
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            return os.path.join(gt_parent_dir, npy_files[0])

    return None

def scale_alignment(pred, ideal, method='median'):
    """
    对预测深度进行尺度对齐，使其与 GT 深度尺度匹配。

    Args:
        pred: 预测深度图 [H, W]
        ideal: GT 深度图 [H, W]
        method: 对齐方法
            - 'median': 中位数对齐 (推荐，速度快)
            - 'linear': 线性回归对齐 (更精确但稍慢)
            - 'none': 不进行对齐

    Returns:
        对齐后的预测深度图
    """
    if method == 'none':
        return pred

    # 创建有效掩码 (排除无效值)
    valid_mask = (ideal > 0.001) & (ideal < 9) & (pred > 0.001) & (pred < 9)

    if not valid_mask.any():
        return pred

    pred_valid = pred[valid_mask]
    ideal_valid = ideal[valid_mask]

    if method == 'median':
        # 中位数对齐: pred * (ideal_median / pred_median)
        pred_median = np.median(pred_valid)
        ideal_median = np.median(ideal_valid)

        if pred_median > 1e-8:
            scale = ideal_median / pred_median
            print(f"  [Scale Alignment] pred_median={pred_median:.4f}, ideal_median={ideal_median:.4f}, scale={scale:.4f}")
            return pred * scale
        else:
            return pred

    elif method == 'linear':
        # 线性回归对齐: pred * slope + intercept
        # 使用最小二乘法
        A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
        slope, intercept = np.linalg.lstsq(A, ideal_valid, rcond=None)[0]
        print(f"  [Scale Alignment] linear: slope={slope:.4f}, intercept={intercept:.4f}")
        return pred * slope + intercept

    else:
        print(f"  Warning: Unknown scale alignment method '{method}', skipping alignment")
        return pred


def loss(pred, ideal, amp_mask=None):
    t_valid = 0.001
    t_max = 9

    pred[pred >= t_max] = 0
    pred[pred < t_valid] = 0

    mask = (ideal > t_valid) & (ideal < t_max)

    # If amplitude mask is provided, exclude masked regions from evaluation
    if amp_mask is not None:
        mask = mask & (~amp_mask)

    num_valid = mask.sum()

    if num_valid == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    pred = pred[mask]
    ideal = ideal[mask]

    diff = pred - ideal
    diff_abs = np.abs(diff)
    mae = diff_abs.sum() / (num_valid + 1e-8)

    rel = diff_abs / (ideal + 1e-8)
    rel = rel.sum() / (num_valid + 1e-8)

    r1 = ideal / (pred + 1e-8)
    r2 = pred / (ideal + 1e-8)
    ratio = np.maximum(r1, r2)

    del_1 = ((ratio < 1.25).astype('float32').sum()) / (num_valid + 1e-8)
    del_2 = ((ratio < 1.25 ** 2).astype('float32').sum()) / (num_valid + 1e-8)
    del_3 = ((ratio < 1.25 ** 3).astype('float32').sum()) / (num_valid + 1e-8)

    return [mae, rel, del_1, del_2, del_3]


def compute_amplitude_mask(noise_iq_path, target_size=(240, 320)):
    """
    Compute amplitude mask from noise IQ file.
    Uses the same logic as training/inference.
    """
    if not MASK_AVAILABLE:
        print("Warning: Mask preprocessing not available, returning None")
        return None

    try:
        # Use same parameters as inference: adaptive 5% threshold + 99.5% upper percentile
        noise_result = load_raw_with_mask(noise_iq_path, target_size=target_size, sqrt_in=True,
                                          amplitude_threshold=None, upper_percentile=99.5)
        if isinstance(noise_result, tuple):
            _, amp_mask = noise_result
        else:
            amp_mask = None
        return amp_mask
    except Exception as e:
        print(f"Warning: Failed to compute amplitude mask for {noise_iq_path}: {e}")
        return None

def eval(pred_dir, out_dir, gt_dir, noise_iq_dir=None, use_mask=False, scale_method='median'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if use_mask:
        if noise_iq_dir is None:
            print("Warning: use_mask=True but noise_iq_dir not provided, falling back to no mask mode")
            use_mask = False
        elif not MASK_AVAILABLE:
            print("Warning: Mask preprocessing not available, falling back to no mask mode")
            use_mask = False

    if use_mask:
        print(f"Mask mode enabled: will exclude low-amplitude regions from evaluation")
        print(f"Noise IQ directory: {noise_iq_dir}")

    loss_mae, loss_rel = [], []
    loss_del_1, loss_del_2, loss_del_3 = [], [], []

    print(f"Scanning prediction directory: {pred_dir}")
    print("This may take a moment...")

    # --- 扫描所有 .npy 文件 ---
    pred_files = []
    for root, dirs, files in os.walk(pred_dir):
        for file in files:
            if file.endswith('.npy'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, pred_dir)
                pred_files.append((rel_path, full_path))

    pred_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(pred_files)} prediction files.")
    if len(pred_files) == 0:
        print("Error: No .npy files found in pred_dir. Please check the path.")
        return

    detail_csv_path = os.path.join(out_dir, "per_image_metrics.csv")
    
    with open(detail_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image_ID', 'MAE', 'Rel', 'Delta_1', 'Delta_2', 'Delta_3'])

        success_count = 0
        for i, (rel_path, pred_full_path) in enumerate(pred_files):
            try:
                # 1. 自动寻找匹配的 GT
                gt_full_path = find_matching_gt(rel_path, gt_dir)
                
                if gt_full_path is None:
                    continue

                # 2. 加载数据
                if success_count == 0:
                    print("-" * 40)
                    print(f"DEBUG Match Example:")
                    print(f"  Pred File: {rel_path}")
                    print(f"  GT File:   {gt_full_path}")
                    print("-" * 40)

                pred = data_loader(pred_full_path)
                ideal = data_loader(gt_full_path)

                # 2.5. 尺度对齐 (如果启用)
                if scale_method != 'none':
                    pred = scale_alignment(pred, ideal, method=scale_method)

                # 2.5. 计算掩码（如果启用）
                amp_mask = None
                if use_mask:
                    # Construct path to noise IQ file
                    # pred_rel_path format: bathroom/1/100.npy
                    # noise_iq format: bathroom/1/100.npy (9 channels)
                    base_rel = os.path.splitext(rel_path)[0]  # remove .npy
                    noise_iq_path = os.path.join(noise_iq_dir, base_rel + ".npy")

                    if os.path.exists(noise_iq_path):
                        amp_mask = compute_amplitude_mask(noise_iq_path, target_size=pred.shape)
                        if amp_mask is not None and success_count == 0:
                            masked_pct = 100.0 * np.count_nonzero(amp_mask) / amp_mask.size
                            print(f"  Noise IQ: {noise_iq_path}")
                            print(f"  Amplitude mask: {masked_pct:.2f}% pixels excluded")
                    else:
                        if success_count == 0:
                            print(f"  Warning: Noise IQ file not found: {noise_iq_path}")

                # 3. 计算 Loss
                loss_list = loss(pred, ideal, amp_mask=amp_mask)

                loss_mae.append(loss_list[0])
                loss_rel.append(loss_list[1])
                loss_del_1.append(loss_list[2])
                loss_del_2.append(loss_list[3])
                loss_del_3.append(loss_list[4])

                img_id = os.path.splitext(rel_path)[0]
                csv_writer.writerow([img_id] + ["{:.4f}".format(x) for x in loss_list])
                success_count += 1

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(pred_files)} files... (MAE: {loss_list[0]:.4f})")

            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
                continue

    if success_count > 0:
        mae_mean = sum(loss_mae) / len(loss_mae)
        rel_mean = sum(loss_rel) / len(loss_rel)
        del_1_mean = sum(loss_del_1) / len(loss_del_1)
        del_2_mean = sum(loss_del_2) / len(loss_del_2)
        del_3_mean = sum(loss_del_3) / len(loss_del_3)
        
        result_str = "Mean Metrics ({5} images) -> MAE: {0:.4f}, Rel: {1:.4f}, D1: {2:.4f}, D2: {3:.4f}, D3: {4:.4f}".format(
            mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean, success_count)
        
        print("\n" + "="*60)
        print(result_str)
        print("="*60)

        with open(f"{out_dir}/result_metrics.txt", "w") as text_file:
            text_file.write(result_str)
    else:
        print("\nFailed! No images were successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_list_path", type=str, default=None, help="Not used in this version (auto-scan)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default="/data/pre_student/hcy/pbrt/gt_depth")

    # Mask mode arguments
    parser.add_argument("--use_mask", action="store_true", help="Enable mask mode: exclude low-amplitude regions from evaluation")
    parser.add_argument("--noise_iq_dir", type=str, default="/data/pre_student/hcy/pbrt/noise", help="Directory containing noise IQ files (for mask computation)")

    # Scale alignment arguments
    parser.add_argument("--scale_method", type=str, default='median', choices=['none', 'median', 'linear'],
                        help="Method for scale alignment: 'none' (no alignment), 'median' (median alignment), 'linear' (linear regression)")

    args = parser.parse_args()

    eval(args.pred_dir, args.out_dir, args.gt_dir, noise_iq_dir=args.noise_iq_dir, use_mask=args.use_mask, scale_method=args.scale_method)