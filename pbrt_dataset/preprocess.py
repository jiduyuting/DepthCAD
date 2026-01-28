import os
import numpy as np
import cv2


def sqrt_ldr(correlations):
    tof_conf = np.abs(correlations[0, :, :]) + np.abs(correlations[1, :, :])
    tof_conf_l = 16 * np.sqrt(tof_conf + 36) - 96
    tof_conf[tof_conf == 0] = 1
    i_tmp = tof_conf_l * correlations[0, :, :] / tof_conf
    q_tmp = tof_conf_l * correlations[1, :, :] / tof_conf

    return np.stack((i_tmp, q_tmp), axis=0)


def load_split_channels(file_path, verbose=False):
    """
    Load multi-channel data from separate files (e.g., 100_A.npy, 100_B.npy, ...).

    Supports both 6-channel (A-F) and 9-channel (A-I) formats.

    The file_path should be like: /path/to/file.npy (with _A suffix)
    This function will load available channels: file_A.npy, file_B.npy, etc.

    Args:
        file_path: Path to one of the channel files (e.g., /path/to/100_A.npy)
        verbose: Print debug information

    Returns:
        numpy array with shape (N, H, W) where N is 6 or 9
    """
    # Extract base path and suffix
    # Handle both formats:
    # 1. /path/to/100_A.npy -> base=/path/to/100, ext=_A.npy
    # 2. /path/to/100 -> base=/path/to/100, will try _A through _F or _I

    # Remove extension if present
    if file_path.endswith('.npy'):
        file_path_no_ext = file_path[:-4]
    else:
        file_path_no_ext = file_path

    # Check if it ends with _suffix pattern
    if file_path_no_ext.endswith('_A') or file_path_no_ext.endswith('_B') or \
       file_path_no_ext.endswith('_C') or file_path_no_ext.endswith('_D') or \
       file_path_no_ext.endswith('_E') or file_path_no_ext.endswith('_F') or \
       file_path_no_ext.endswith('_G') or file_path_no_ext.endswith('_H') or \
       file_path_no_ext.endswith('_I'):
        # Extract base name by removing _suffix
        base_path = file_path_no_ext[:-2]
    else:
        # Use as-is
        base_path = file_path_no_ext

    # Try 9-channel format first, then 6-channel
    suffixes_9ch = ['_A', '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_I']
    suffixes_6ch = ['_A', '_B', '_C', '_D', '_E', '_F']

    # Load available channels
    channels = []
    loaded_format = None

    # Try 9-channel format
    for suffix in suffixes_9ch:
        ch_path = f"{base_path}{suffix}.npy"
        if os.path.exists(ch_path):
            try:
                ch_data = np.load(ch_path).astype(np.float32)
                channels.append(ch_data)
                if verbose and suffix == '_A':
                    print(f"  [load_split_channels] Trying 9-channel format...")
                    print(f"  [load_split_channels] Loaded {ch_path}, shape: {ch_data.shape}")
            except Exception as e:
                raise ValueError(f"Failed to load {ch_path}: {e}")

    if len(channels) == 9:
        loaded_format = "9-channel"
    elif len(channels) == 6:
        # Found all 6 channels (A-F)
        loaded_format = "6-channel"
    elif len(channels) == 0:
        # Try 6-channel format directly
        channels = []
        for suffix in suffixes_6ch:
            ch_path = f"{base_path}{suffix}.npy"
            if os.path.exists(ch_path):
                try:
                    ch_data = np.load(ch_path).astype(np.float32)
                    channels.append(ch_data)
                    if verbose and suffix == '_A':
                        print(f"  [load_split_channels] Trying 6-channel format...")
                        print(f"  [load_split_channels] Loaded {ch_path}, shape: {ch_data.shape}")
                except Exception as e:
                    raise ValueError(f"Failed to load {ch_path}: {e}")

        if len(channels) == 6:
            loaded_format = "6-channel"
        else:
            raise ValueError(f"Found {len(channels)} channels, expected 6 or 9")
    elif len(channels) < 9:
        # Partial 9-channel - treat as error
        raise ValueError(f"Found only {len(channels)}/9 channels. Check if files are missing.")
    else:
        loaded_format = "9-channel"

    # Stack into (N, H, W) format
    stacked = np.stack(channels, axis=0)

    if verbose:
        print(f"  [load_split_channels] Loaded {loaded_format} data: {stacked.shape}")
        print(f"  [load_split_channels] Value range: [{stacked.min():.4f}, {stacked.max():.4f}]")

    return stacked


def load_raw(scene_path, target_size=(512, 512), sqrt_in=True, verbose=False,
             amplitude_threshold=None, upper_percentile=None):
    """
    Load pbrt raw data from .npy file.

    Supports two formats:
    1. Combined format: Single .npy file with shape (9, H, W)
    2. Split format: 9 separate .npy files (e.g., 100_A.npy, 100_B.npy, ...)

    Input: Single file path or base path for split files
    Output: (6, H, W) - 6 IQ channels after processing

    Parameters:
    -----------
    scene_path : str
        Path to .npy file (combined or split format base)
    target_size : tuple
        Target size (height, width)
    sqrt_in : bool
        Whether to apply sqrt_ldr transformation
    verbose : bool
        Print debug information
    amplitude_threshold : float or None
        Unused (for compatibility)
    upper_percentile : float or None
        Unused (for compatibility)
    """
    # Try to detect data format
    if os.path.isfile(scene_path):
        # Single file - try to load it
        try:
            ori_correlations = np.load(scene_path).astype(np.float32)
            if verbose:
                print(f"  [load_raw] Loaded single file: {scene_path}")
                print(f"  [load_raw] Original shape: {ori_correlations.shape}")

            # Check if it's the split format indicator (single channel 2D array)
            if len(ori_correlations.shape) == 2:
                # This is a single channel file - try to load all 9 channels
                if verbose:
                    print(f"  [load_raw] Detected 2D array - trying split channel format...")
                ori_correlations = load_split_channels(scene_path, verbose=verbose)

        except Exception as e:
            raise ValueError(f"Failed to load {scene_path}: {e}")
    else:
        # Not a file - assume it's a base path for split format
        if verbose:
            print(f"  [load_raw] Path is not a file - trying split channel format...")
        ori_correlations = load_split_channels(scene_path, verbose=verbose)

    # Ensure shape is (9, H, W) or (6, H, W)
    num_channels = ori_correlations.shape[0]

    # Handle 6-channel format (already processed IQ data)
    if num_channels == 6:
        if verbose:
            print(f"  [load_raw] Detected 6-channel preprocessed IQ data")
            print(f"  [load_raw] Skipping IQ extraction and sqrt_ldr transform")

        # Just resize to target size and return
        target_h, target_w = target_size
        if ori_correlations.shape[1:] != (target_h, target_w):
            resized = np.zeros((6, target_h, target_w), dtype=np.float32)
            for i in range(6):
                resized[i] = cv2.resize(ori_correlations[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            ori_correlations = resized
            if verbose:
                print(f"  [load_raw] Resized to {target_size}")

        ori_correlations = np.nan_to_num(ori_correlations, nan=0, neginf=0, posinf=0)
        if verbose:
            print(f"  [load_raw] Output shape: {ori_correlations.shape}, range: [{ori_correlations.min():.4f}, {ori_correlations.max():.4f}]")
        return ori_correlations

    # Handle 9-channel format (raw correlation data)
    if num_channels != 9:
        # Try to transpose to get 9 channels first
        if len(ori_correlations.shape) == 3 and ori_correlations.shape[2] == 9:
            ori_correlations = np.transpose(ori_correlations, (2, 0, 1))
            num_channels = ori_correlations.shape[0]
            if verbose:
                print(f"  [load_raw] Transposed from (H, W, 9) to (9, H, W)")
        elif len(ori_correlations.shape) == 3 and ori_correlations.shape[1] == 9:
            ori_correlations = np.transpose(ori_correlations, (1, 0, 2))
            num_channels = ori_correlations.shape[0]
            if verbose:
                print(f"  [load_raw] Transposed from (H, 9, W) to (9, H, W)")
        else:
            raise ValueError(f"Unexpected shape: {ori_correlations.shape}, expected (6, H, W) or (9, H, W)")
    
    target_h, target_w = target_size
    correlations = np.zeros((9, target_h, target_w), dtype=np.float32)
    correlations = np.nan_to_num(correlations, nan=0, neginf=0, posinf=0)

    # Resize each channel to target size
    for i in range(9):
        correlations[i, :, :] = cv2.resize(ori_correlations[i, :, :], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Extract IQ pairs: (1,0), (4,3), (7,6) - matching flat_dataset order
    # Note: flat_dataset uses (1,0), (4,3), (7,6) for tof_IQ_40, tof_IQ_30, tof_IQ_58
    tof_IQ_40 = np.stack((correlations[1, :, :], correlations[0, :, :]), axis=0)
    tof_IQ_30 = np.stack((correlations[4, :, :], correlations[3, :, :]), axis=0)
    tof_IQ_58 = np.stack((correlations[7, :, :], correlations[6, :, :]), axis=0)

    if sqrt_in:
        tof_IQ_40 = sqrt_ldr(tof_IQ_40)
        tof_IQ_30 = sqrt_ldr(tof_IQ_30)
        tof_IQ_58 = sqrt_ldr(tof_IQ_58)

    tof_IQs = np.stack((
        tof_IQ_30[0], tof_IQ_30[1],
        tof_IQ_40[0], tof_IQ_40[1],
        tof_IQ_58[0], tof_IQ_58[1]
    ), axis=0)

    tof_IQs = np.nan_to_num(tof_IQs, nan=0, neginf=0, posinf=0)
    if verbose:
        print(f"  [load_raw] Output shape: {tof_IQs.shape}, value range: [{tof_IQs.min():.4f}, {tof_IQs.max():.4f}]")
    return tof_IQs


def compute_gradient_confidence(depth_map):
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    confidence_map = 1 - grad_magnitude
    return confidence_map


if __name__ == '__main__':
    print("=" * 60)
    print("PBRT Dataset Preprocessing")
    print("=" * 60)
    
    # PBRT dataset paths
    ideal_root = '/data/pre_student/hcy/pbrt/gt'
    noise_root = '/data/pre_student/hcy/pbrt/noise'
    noise_depth_root = '/data/pre_student/hcy/pbrt/noise_depth'
    
    # Output paths
    ideal_norm_root = 'data/ideal_IQ'
    noise_norm_root = 'data/noise_IQ'
    conf_root = 'data/confidence'
    
    print(f"\nInput paths:")
    print(f"  Ideal: {ideal_root}")
    print(f"  Noise: {noise_root}")
    print(f"  Noise Depth: {noise_depth_root}")
    print(f"\nOutput paths:")
    print(f"  Ideal IQ: {ideal_norm_root}")
    print(f"  Noise IQ: {noise_norm_root}")
    print(f"  Confidence: {conf_root}")
    
    # Create output directories
    roots_to_check = [ideal_norm_root, noise_norm_root, conf_root]
    for root in roots_to_check:
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
            print(f"  Created directory: {root}")
    
    # Process each category
    categories = [d for d in os.listdir(ideal_root) if os.path.isdir(os.path.join(ideal_root, d))]
    categories.sort()
    
    print(f"\nFound {len(categories)} categories: {categories}")
    
    suffixes = ['A', 'B', 'C', 'D', 'E', 'F']
    target_size = (512, 512)  # Match training resolution
    print(f"Target size: {target_size}")
    print("=" * 60)
    
    for category in categories:
        # Process all version subdirectories (0, 1, 2, 3, ...) instead of just '0'
        category_base_ideal = os.path.join(ideal_root, category)
        category_base_noise = os.path.join(noise_root, category)
        category_base_noise_depth = os.path.join(noise_depth_root, category)

        # Get all version subdirectories
        version_dirs = sorted([d for d in os.listdir(category_base_ideal) if os.path.isdir(os.path.join(category_base_ideal, d))],
                             key=lambda x: int(x) if x.isdigit() else x)

        print(f"\n=== Processing category: {category} ===")
        print(f"Found versions: {version_dirs}")

        for version in version_dirs:
            print(f"  Processing version {version}...")

            category_ideal_root = os.path.join(category_base_ideal, version)
            category_noise_root = os.path.join(category_base_noise, version)
            category_noise_depth_root = os.path.join(category_base_noise_depth, version)

            category_ideal_norm_root = os.path.join(ideal_norm_root, category, version)
            category_noise_norm_root = os.path.join(noise_norm_root, category, version)
            category_conf_root = os.path.join(conf_root, category, version)

            # Create category output directories
            for d in [category_ideal_norm_root, category_noise_norm_root, category_conf_root]:
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)

            if not os.path.exists(category_ideal_root):
                print(f"  Warning: {category_ideal_root} does not exist, skipping version {version}")
                continue

            # Get all .npy files in the category/version
            idxs = [f for f in os.listdir(category_ideal_root) if f.endswith('.npy')]
            idxs.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))

            print(f"  Found {len(idxs)} files to process")

            for file_idx, idx in enumerate(idxs):
                ideal_path = os.path.join(category_ideal_root, idx)
                noise_path = os.path.join(category_noise_root, idx)
                noise_depth_path = os.path.join(category_noise_depth_root, idx)

                if not os.path.exists(ideal_path):
                    print(f"  Warning: {ideal_path} does not exist, skipping")
                    continue
                if not os.path.exists(noise_path):
                    print(f"  Warning: {noise_path} does not exist, skipping")
                    continue
                if not os.path.exists(noise_depth_path):
                    print(f"  Warning: {noise_depth_path} does not exist, skipping")
                    continue

                print(f"\n  [{file_idx+1}/{len(idxs)}] Processing {category}/{version}/{idx}")

                # Load and process ideal and noise IQ data
                try:
                    ideal_IQs = load_raw(ideal_path, target_size=target_size, verbose=(file_idx == 0))
                    noise_IQs = load_raw(noise_path, target_size=target_size, verbose=(file_idx == 0))
                    print(f"    Loaded IQ data - ideal shape: {ideal_IQs.shape}, noise shape: {noise_IQs.shape}")
                    print(f"    Ideal value range: [{ideal_IQs.min():.4f}, {ideal_IQs.max():.4f}]")
                    print(f"    Noise value range: [{noise_IQs.min():.4f}, {noise_IQs.max():.4f}]")
                except Exception as e:
                    print(f"    ERROR loading IQ data: {e}")
                    continue

                # Normalize by noise max value
                noise_max = max(noise_IQs.max(), abs(noise_IQs.min()), 1e-8)
                ideal_IQs = ideal_IQs / noise_max
                noise_IQs = noise_IQs / noise_max
                print(f"    Normalized by noise_max: {noise_max:.4f}")
                print(f"    After normalization - ideal range: [{ideal_IQs.min():.4f}, {ideal_IQs.max():.4f}]")
                print(f"    After normalization - noise range: [{noise_IQs.min():.4f}, {noise_IQs.max():.4f}]")

                # Load and process depth for confidence map
                try:
                    noise_depth = np.load(noise_depth_path)
                    print(f"    Loaded depth - shape: {noise_depth.shape}, range: [{noise_depth.min():.4f}, {noise_depth.max():.4f}]")
                    noise_depth = cv2.resize(noise_depth.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
                    confidence = compute_gradient_confidence(noise_depth)
                    print(f"    Computed confidence - shape: {confidence.shape}, range: [{confidence.min():.4f}, {confidence.max():.4f}]")
                    conf_path = os.path.join(category_conf_root, idx)
                    np.save(conf_path, confidence)
                except Exception as e:
                    print(f"    ERROR processing depth/confidence: {e}")
                    continue

                # Save individual IQ channels
                try:
                    for i in range(6):
                        ideal_norm_path = os.path.join(category_ideal_norm_root, f"{idx.split('.')[0]}_{suffixes[i]}.npy")
                        noise_norm_path = os.path.join(category_noise_norm_root, f"{idx.split('.')[0]}_{suffixes[i]}.npy")
                        np.save(ideal_norm_path, ideal_IQs[i])
                        np.save(noise_norm_path, noise_IQs[i])
                    print(f"    Saved all IQ channels and confidence map")
                except Exception as e:
                    print(f"    ERROR saving files: {e}")
                    continue

                print(f"    ✓ Successfully processed {category}/{version}/{idx}")

        print(f"\n=== Finished category: {category} ===\n")


