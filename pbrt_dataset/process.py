import argparse
import os
import numpy as np
import cv2
from pathlib import Path


def load_raw(npy_file, target_size=(512, 512), sqrt_in=True,
             amplitude_threshold=None, upper_percentile=None):
    """
    Load PBRT data from .npy file and extract IQ channels.

    Parameters:
    -----------
    npy_file : str or Path
        Path to .npy file containing 9-channel data (shape: 9, H, W)
    target_size : tuple
        Target size (height, width) for resizing
    sqrt_in : bool
        Whether to apply sqrt_ldr transformation
    amplitude_threshold : float or None
        Unused parameter (for compatibility with inference.py)
    upper_percentile : float or None
        Unused parameter (for compatibility with inference.py)

    Returns:
    --------
    tof_IQs : numpy.ndarray
        6-channel IQ data (I30, Q30, I40, Q40, I58, Q58) with shape (6, h, w)
    """
    # Load .npy file
    data = np.load(npy_file)

    # Ensure data is in (9, H, W) format
    if data.shape[0] != 9:
        if data.ndim == 3:
            # Try to find the channel axis
            if data.shape[1] == 9:
                data = np.moveaxis(data, 1, 0)
            elif data.shape[2] == 9:
                data = np.moveaxis(data, 2, 0)

    # Extract IQ pairs: indices (0,1), (3,4), (6,7)
    pairs = [(0, 1), (3, 4), (6, 7)]
    iq_list = []

    for a, b in pairs:
        tof_iq = np.stack((data[a], data[b]), axis=0)
        if sqrt_in:
            tof_iq = sqrt_ldr(tof_iq)
        iq_list.extend([tof_iq[0], tof_iq[1]])

    # Stack to get (6, H, W)
    tof_IQs = np.stack(iq_list, axis=0).astype(np.float32)
    tof_IQs = np.nan_to_num(tof_IQs, nan=0, neginf=0, posinf=0)

    # Resize to target size if needed
    target_h, target_w = target_size
    if tof_IQs.shape[1:] != (target_h, target_w):
        resized = np.zeros((6, target_h, target_w), dtype=np.float32)
        for i in range(6):
            resized[i] = cv2.resize(tof_IQs[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        tof_IQs = resized

    return tof_IQs


def sqrt_ldr(correlations):
    # correlations: (2, H, W)
    tof_conf = np.abs(correlations[0]) + np.abs(correlations[1])
    tof_conf_l = 16 * np.sqrt(tof_conf + 36) - 96
    tof_conf[tof_conf == 0] = 1
    i_tmp = tof_conf_l * correlations[0] / tof_conf
    q_tmp = tof_conf_l * correlations[1] / tof_conf
    return np.stack((i_tmp, q_tmp), axis=0)


def detect_and_prepare(arr, expected_channels=9):
    # Ensure arr becomes shape (C, H, W) where C == expected_channels
    a = np.asarray(arr)
    if a.ndim == 3:
        # find axis with length == expected_channels
        if a.shape[0] == expected_channels:
            return a
        if a.shape[1] == expected_channels:
            return np.moveaxis(a, 1, 0)
        if a.shape[2] == expected_channels:
            return np.moveaxis(a, 2, 0)
        # fallback: if total size matches, try reshape
        if a.size % expected_channels == 0:
            hw = a.size // expected_channels
            # try to infer H,W: prefer square-ish
            h = int(np.sqrt(hw))
            if h * h == hw:
                return a.reshape((expected_channels, h, h))
    elif a.ndim == 1:
        # maybe raw flattened
        if a.size % expected_channels == 0:
            hw = a.size // expected_channels
            h = int(np.sqrt(hw))
            if h * h == hw:
                return a.reshape((expected_channels, h, h))

    raise ValueError(f"Unable to coerce array with shape {a.shape} to (C={expected_channels}, H, W)")


def resize_channels(channels, target_h, target_w):
    C, H, W = channels.shape
    out = np.zeros((C, target_h, target_w), dtype=np.float32)
    for i in range(C):
        out[i] = cv2.resize(channels[i].astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return out


def compute_gradient_confidence(depth_map):
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    confidence_map = 1 - grad_magnitude
    return confidence_map.astype(np.float32)


def process_pair(ideal_path: Path, noise_path: Path, noise_depth_path: Path, out_ideal_cat: Path, out_noise_cat: Path, out_conf_cat: Path, target_size=(512, 512), sqrt_in=True, overwrite=False):
    base = ideal_path.stem
    out_conf_file = out_conf_cat.joinpath(f"{base}.npy")
    if out_conf_file.exists() and not overwrite:
        return

    try:
        ideal_arr = np.load(str(ideal_path))
    except Exception as e:
        print(f"[ERROR] load ideal {ideal_path}: {e}")
        return
    try:
        noise_arr = np.load(str(noise_path))
    except Exception as e:
        print(f"[ERROR] load noise {noise_path}: {e}")
        return

    try:
        ideal_ch = detect_and_prepare(ideal_arr, expected_channels=9)
        noise_ch = detect_and_prepare(noise_arr, expected_channels=9)
    except Exception as e:
        print(f"[ERROR] detect shape failed for {base}: {e}")
        return

    target_h, target_w = target_size
    ideal_rs = resize_channels(ideal_ch, target_h, target_w)
    noise_rs = resize_channels(noise_ch, target_h, target_w)

    # Now compute IQ pairs. mapping indices: (0,1), (3,4), (6,7)
    pairs = [ (0,1), (3,4), (6,7) ]
    iq_list_ideal = []
    iq_list_noise = []
    for a,b in pairs:
        tof_iq_ideal = np.stack((ideal_rs[a], ideal_rs[b]), axis=0)
        tof_iq_noise = np.stack((noise_rs[a], noise_rs[b]), axis=0)
        if sqrt_in:
            tof_iq_ideal = sqrt_ldr(tof_iq_ideal)
            tof_iq_noise = sqrt_ldr(tof_iq_noise)
        iq_list_ideal.extend([tof_iq_ideal[0], tof_iq_ideal[1]])
        iq_list_noise.extend([tof_iq_noise[0], tof_iq_noise[1]])

    ideal_IQs = np.stack(iq_list_ideal, axis=0)
    noise_IQs = np.stack(iq_list_noise, axis=0)

    noise_max = max(np.abs(noise_IQs).max(), 1e-8)
    ideal_IQs = (ideal_IQs.astype(np.float32) / noise_max)
    noise_IQs = (noise_IQs.astype(np.float32) / noise_max)

    try:
        depth = np.load(str(noise_depth_path))
    except Exception as e:
        print(f"[WARN] load noise depth {noise_depth_path}: {e}")
        return

    try:
        depth_rs = cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"[WARN] resize depth failed for {noise_depth_path}: {e}")
        return

    confidence = compute_gradient_confidence(depth_rs)

    suffixes = ['A','B','C','D','E','F']
    for i in range(6):
        ideal_out = out_ideal_cat.joinpath(f"{base}_{suffixes[i]}.npy")
        noise_out = out_noise_cat.joinpath(f"{base}_{suffixes[i]}.npy")
        np.save(str(ideal_out), ideal_IQs[i])
        np.save(str(noise_out), noise_IQs[i])

    np.save(str(out_conf_file), confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ideal_root', type=str, default='/data/pre_student/hcy/pbrt/gt')
    parser.add_argument('--noise_root', type=str, default='/data/pre_student/hcy/pbrt/noise')
    parser.add_argument('--noise_depth_root', type=str, default='/data/pre_student/hcy/pbrt/noise_depth')
    parser.add_argument('--out_root', type=str, default='data')
    parser.add_argument('--target_h', type=int, default=512)
    parser.add_argument('--target_w', type=int, default=512)
    parser.add_argument('--sqrt_in', action='store_true', help='apply sqrt_ldr transformation')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    ideal_root = Path(args.ideal_root)
    noise_root = Path(args.noise_root)
    noise_depth_root = Path(args.noise_depth_root)

    ideal_norm_root = Path(args.out_root) / 'ideal_IQ'
    noise_norm_root = Path(args.out_root) / 'noise_IQ'
    conf_root = Path(args.out_root) / 'confidence'

    for d in (ideal_norm_root, noise_norm_root, conf_root):
        d.mkdir(parents=True, exist_ok=True)

    categories = sorted([c for c in os.listdir(ideal_root) if os.path.isdir(os.path.join(ideal_root, c))])
    print(f"Found categories: {categories}")

    for cat in categories:
        # Process all version subdirectories (0, 1, 2, 3, ...) instead of just '0'
        cat_dir = ideal_root / cat
        version_dirs = sorted([d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))],
                             key=lambda x: int(x) if x.isdigit() else x)

        print(f"Processing {cat} with versions: {version_dirs}")

        for version in version_dirs:
            print(f"  Processing version {version}...")
            ideal_version_dir = ideal_root / cat / version
            if not ideal_version_dir.is_dir():
                print(f"[WARN] missing {version} dir for category {cat}, skipping")
                continue

            files = sorted([f for f in ideal_version_dir.iterdir() if f.suffix == '.npy'])
            if len(files) == 0:
                print(f"[WARN] no .npy files in {ideal_version_dir}, skipping")
                continue

            out_ideal_cat = ideal_norm_root / cat / version
            out_noise_cat = noise_norm_root / cat / version
            out_conf_cat = conf_root / cat / version
            out_ideal_cat.mkdir(parents=True, exist_ok=True)
            out_noise_cat.mkdir(parents=True, exist_ok=True)
            out_conf_cat.mkdir(parents=True, exist_ok=True)

            for ideal_file in files:
                base = ideal_file.stem
                noise_file = noise_root / cat / version / ideal_file.name
                noise_depth_file = noise_depth_root / cat / version / ideal_file.name

                if not noise_file.exists():
                    print(f"[WARN] noise missing {noise_file}, skipping")
                    continue
                if not noise_depth_file.exists():
                    print(f"[WARN] noise depth missing {noise_depth_file}, skipping")
                    continue

                process_pair(ideal_file, noise_file, noise_depth_file, out_ideal_cat, out_noise_cat, out_conf_cat,
                             target_size=(args.target_h, args.target_w), sqrt_in=args.sqrt_in, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
