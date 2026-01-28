#!/usr/bin/env python3
"""
DepthCAD Evaluation Script for HYPIR-Enhanced Model

This script evaluates depth predictions against ground truth.
Simplified version without mask support, designed for models trained with train_hypir.py.

Key features:
- No amplitude masking (evaluates all pixels)
- Automatic GT file matching
- Supports both .npy and .bin file formats
- Outputs detailed metrics per image and aggregated results
"""

import os
import argparse
import numpy as np
import csv


def data_loader(path):
    """
    Load depth data from file.

    Supports:
    - .npy files (NumPy arrays)
    - .bin files (binary float32 arrays)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith('.npy'):
        data = np.load(path)
    elif path.endswith('.bin'):
        shape = (240, 320)
        data = np.fromfile(path, dtype=np.float32).reshape(shape)
    else:
        # Try to load as .npy by default
        try:
            data = np.load(path)
        except:
            raise ValueError(f"Unknown file format: {path}")

    data = np.nan_to_num(data, 0)
    return data


def find_matching_gt(pred_rel_path, gt_dir):
    """
    Find matching ground truth file for a prediction file.

    Args:
        pred_rel_path: Relative path of prediction file (e.g., "bathroom/1/115.npy")
        gt_dir: Root directory of ground truth files

    Returns:
        Full path to matching GT file, or None if not found
    """
    base_rel = os.path.splitext(pred_rel_path)[0]  # "bathroom/1/115"

    # 1. Try exact match
    gt_path_v1 = os.path.join(gt_dir, base_rel + ".npy")
    if os.path.exists(gt_path_v1):
        return gt_path_v1

    gt_path_v2 = os.path.join(gt_dir, base_rel + ".bin")
    if os.path.exists(gt_path_v2):
        return gt_path_v2

    # 2. Try to find in parent directory (e.g., depth.bin in bathroom/1/)
    parent_rel = os.path.dirname(pred_rel_path)
    gt_parent_dir = os.path.join(gt_dir, parent_rel)

    if os.path.isdir(gt_parent_dir):
        files = os.listdir(gt_parent_dir)
        if "depth.bin" in files:
            return os.path.join(gt_parent_dir, "depth.bin")

        # Try any .bin file
        bin_files = [f for f in files if f.endswith('.bin')]
        if bin_files:
            return os.path.join(gt_parent_dir, bin_files[0])

        # Try any .npy file
        npy_files = [f for f in files if f.endswith('.npy')]
        if npy_files:
            return os.path.join(gt_parent_dir, npy_files[0])

    return None


def compute_metrics(pred, ideal):
    """
    Compute depth estimation metrics.

    Metrics:
    - MAE: Mean Absolute Error
    - Rel: Mean Relative Error
    - Delta_1: Percentage of pixels with ratio < 1.25
    - Delta_2: Percentage of pixels with ratio < 1.25^2
    - Delta_3: Percentage of pixels with ratio < 1.25^3

    Args:
        pred: Predicted depth map
        ideal: Ground truth depth map

    Returns:
        List of [mae, rel, delta_1, delta_2, delta_3]
    """
    t_valid = 0.001
    t_max = 9

    # Filter invalid predictions
    pred[pred >= t_max] = 0
    pred[pred < t_valid] = 0

    # Create valid mask based on ground truth
    mask = (ideal > t_valid) & (ideal < t_max)
    num_valid = mask.sum()

    if num_valid == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # Apply mask
    pred = pred[mask]
    ideal = ideal[mask]

    # MAE (Mean Absolute Error)
    diff = pred - ideal
    diff_abs = np.abs(diff)
    mae = diff_abs.sum() / (num_valid + 1e-8)

    # Rel (Mean Relative Error)
    rel = diff_abs / (ideal + 1e-8)
    rel = rel.sum() / (num_valid + 1e-8)

    # Delta metrics (threshold accuracy)
    r1 = ideal / (pred + 1e-8)
    r2 = pred / (ideal + 1e-8)
    ratio = np.maximum(r1, r2)

    del_1 = ((ratio < 1.25).astype('float32').sum()) / (num_valid + 1e-8)
    del_2 = ((ratio < 1.25 ** 2).astype('float32').sum()) / (num_valid + 1e-8)
    del_3 = ((ratio < 1.25 ** 3).astype('float32').sum()) / (num_valid + 1e-8)

    return [mae, rel, del_1, del_2, del_3]


def evaluate(pred_dir, out_dir, gt_dir):
    """
    Evaluate depth predictions against ground truth.

    Args:
        pred_dir: Directory containing prediction files
        out_dir: Directory to save evaluation results
        gt_dir: Directory containing ground truth files
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize metric lists
    loss_mae, loss_rel = [], []
    loss_del_1, loss_del_2, loss_del_3 = [], [], []

    print(f"Scanning prediction directory: {pred_dir}")
    print("This may take a moment...")

    # Scan all .npy files
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

    # Create CSV file for per-image metrics
    detail_csv_path = os.path.join(out_dir, "per_image_metrics.csv")

    with open(detail_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image_ID', 'MAE', 'Rel', 'Delta_1', 'Delta_2', 'Delta_3'])

        success_count = 0
        no_gt_count = 0
        error_count = 0

        for i, (rel_path, pred_full_path) in enumerate(pred_files):
            try:
                # Find matching GT file
                gt_full_path = find_matching_gt(rel_path, gt_dir)

                if gt_full_path is None:
                    no_gt_count += 1
                    if no_gt_count <= 3:
                        print(f"[Warning] No GT found for: {rel_path}")
                    continue

                # Load data
                if success_count == 0:
                    print("-" * 40)
                    print(f"First Match Example:")
                    print(f"  Pred File: {rel_path}")
                    print(f"  GT File:   {gt_full_path}")
                    print("-" * 40)

                pred = data_loader(pred_full_path)
                ideal = data_loader(gt_full_path)

                # Compute metrics
                metrics = compute_metrics(pred, ideal)

                loss_mae.append(metrics[0])
                loss_rel.append(metrics[1])
                loss_del_1.append(metrics[2])
                loss_del_2.append(metrics[3])
                loss_del_3.append(metrics[4])

                # Write to CSV
                img_id = os.path.splitext(rel_path)[0]
                csv_writer.writerow([img_id] + ["{:.4f}".format(x) for x in metrics])
                success_count += 1

                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(pred_files)} files... "
                          f"(Matched: {success_count}, No GT: {no_gt_count}, MAE: {metrics[0]:.4f})")

            except Exception as e:
                error_count += 1
                print(f"[Error] Failed to process {rel_path}: {e}")
                continue

    # Compute and print aggregated results
    if success_count > 0:
        mae_mean = sum(loss_mae) / len(loss_mae)
        rel_mean = sum(loss_rel) / len(loss_rel)
        del_1_mean = sum(loss_del_1) / len(loss_del_1)
        del_2_mean = sum(loss_del_2) / len(loss_del_2)
        del_3_mean = sum(loss_del_3) / len(loss_del_3)

        result_str = (
            f"Mean Metrics ({success_count} images)\n"
            f"{'='*50}\n"
            f"  MAE:      {mae_mean:.4f}\n"
            f"  Rel:      {rel_mean:.4f}\n"
            f"  Delta_1:  {del_1_mean:.4f}\n"
            f"  Delta_2:  {del_2_mean:.4f}\n"
            f"  Delta_3:  {del_3_mean:.4f}\n"
            f"{'='*50}\n"
            f"Files processed: {success_count}\n"
            f"No GT found:     {no_gt_count}\n"
            f"Errors:          {error_count}"
        )

        print("\n" + result_str)

        # Save to file
        with open(os.path.join(out_dir, "result_metrics.txt"), "w") as f:
            f.write(result_str)

        print(f"\nResults saved to: {out_dir}")
        print(f"  - per_image_metrics.csv: Detailed metrics for each image")
        print(f"  - result_metrics.txt:     Aggregated results")

    else:
        print("\nFailed! No images were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DepthCAD predictions against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing prediction files (will scan recursively)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/data/pre_student/hcy/pbrt/gt_depth",
        help="Directory containing ground truth files"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DepthCAD Evaluation for HYPIR-Enhanced Model")
    print("=" * 60)
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 60)
    print()

    evaluate(args.pred_dir, args.out_dir, args.gt_dir)
