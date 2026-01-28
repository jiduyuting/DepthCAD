import os
import argparse
import numpy as np
import csv

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

def loss(pred, ideal):
    t_valid = 0.001
    t_max = 9
    
    pred[pred >= t_max] = 0
    pred[pred < t_valid] = 0

    mask = (ideal > t_valid) & (ideal < t_max) 
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

def eval(pred_dir, out_dir, gt_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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

                # 3. 计算 Loss
                loss_list = loss(pred, ideal)

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
    # 【修复】把参数加回来，设为可选（这样你的旧命令就不会报错了）
    parser.add_argument("--test_list_path", type=str, default=None, help="Not used in this version (auto-scan)")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default="/data/pre_student/hcy/pbrt/gt_depth")
    
    args = parser.parse_args()

    # 这里的参数里虽然有 test_list_path，但我们不传给 eval 函数，eval 自己会去扫描文件夹
    eval(args.pred_dir, args.out_dir, args.gt_dir)