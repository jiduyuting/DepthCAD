import os
import cv2
import argparse
import numpy as np


def bin_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    shape = (240, 320)
    data = np.fromfile(path, dtype=np.float32).reshape(shape)
    data = np.nan_to_num(data, 0)

    return data


def depth_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    return data


def loss(pred, ideal):
    """Calculate depth estimation error metrics.

    Only considers pixels where both prediction and GT are valid (within [t_valid, t_max]).

    Args:
        pred: predicted depth map
        ideal: ground truth depth map

    Returns:
        list: [mae, rel, del_1, del_2, del_3]
    """
    t_valid = 0
    t_max = 9

    # Create mask for valid pixels: both pred AND ideal must be in valid range
    # This avoids counting clipped predictions (set to 0) as errors
    mask = (ideal > t_valid) & (ideal < t_max) & (pred > t_valid) & (pred < t_max)
    num_valid = mask.sum()

    # If no valid pixels, return zeros
    if num_valid == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    pred = pred[mask]
    ideal = ideal[mask]

    # MAE
    diff = pred - ideal
    diff_abs = np.abs(diff)
    mae = diff_abs.sum() / (num_valid + 1e-8)

    # Rel
    rel = diff_abs / (ideal + 1e-8)
    rel = rel.sum() / (num_valid + 1e-8)

    # delta
    r1 = ideal / (pred + 1e-8)
    r2 = pred / (ideal + 1e-8)
    ratio = np.maximum(r1, r2)

    del_1 = (ratio < 1.25).astype('float32')
    del_2 = (ratio < 1.25 ** 2).astype('float32')
    del_3 = (ratio < 1.25 ** 3).astype('float32')

    del_1 = del_1.sum() / (num_valid + 1e-8)
    del_2 = del_2.sum() / (num_valid + 1e-8)
    del_3 = del_3.sum() / (num_valid + 1e-8)

    result = [mae, rel, del_1, del_2, del_3]
    return result


def eval(test_list_path, pred_dir, out_dir):
    loss_mae = []
    loss_rel = []
    loss_del_1 = []
    loss_del_2 = []
    loss_del_3 = []

    with open(test_list_path, 'r') as f:
        idxs = f.readlines()
    idxs = [idx.strip() for idx in idxs if idx.strip()]

    ideal_root = "/data/pre_student/hcy/pbrt/gt_depth"

    for idx in idxs:
        # Parse idx: format is "category/version" e.g., "bathroom/1"
        parts = idx.split('/')
        if len(parts) != 2:
            print(f"Warning: Invalid index format '{idx}', skipping")
            continue

        category, version = parts[0], parts[1]

        # Get all prediction files in this directory
        pred_dir_version = os.path.join(pred_dir, category, version)
        if not os.path.exists(pred_dir_version):
            print(f"Warning: Prediction directory not found for {idx}")
            continue

        pred_files = [f for f in os.listdir(pred_dir_version) if f.endswith('.npy')]

        for pred_file in pred_files:
            # Remove _depth suffix if present (inference saves as {index}_depth.npy)
            index = pred_file.replace('.npy', '').replace('_depth', '')

            # Load ideal depth
            ideal_path = os.path.join(ideal_root, category, version, f"{index}.npy")
            if not os.path.exists(ideal_path):
                continue
            ideal = depth_loader(ideal_path)

            # Load prediction
            pred_path = os.path.join(pred_dir_version, pred_file)
            pred = np.load(pred_path)
            pred = np.nan_to_num(pred, 0)

            # Resize ideal to match prediction if needed
            if ideal.shape != pred.shape:
                ideal = cv2.resize(ideal, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)

            loss_list = loss(pred, ideal)

            loss_mae.append(loss_list[0])
            loss_rel.append(loss_list[1])
            loss_del_1.append(loss_list[2])
            loss_del_2.append(loss_list[3])
            loss_del_3.append(loss_list[4])

    # save results
    mae_mean = sum(loss_mae) / len(loss_mae)
    rel_mean = sum(loss_rel) / len(loss_rel)
    del_1_mean = sum(loss_del_1) / len(loss_del_1)
    del_2_mean = sum(loss_del_2) / len(loss_del_2)
    del_3_mean = sum(loss_del_3) / len(loss_del_3)

    print("mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}".format(mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean))

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/result_metrics.txt", "w") as text_file:
        text_file.write("mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}".format(mae_mean, rel_mean, del_1_mean, del_2_mean, del_3_mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_list_path",
        type=str,
        default="pbrt_dataset/test.txt",
    )
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)

    args = parser.parse_args()

    eval(args.test_list_path, args.pred_dir, args.out_dir)
