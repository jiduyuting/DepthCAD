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


def load_raw(scene_path, target_size=(256, 256), sqrt_in=True, verbose=False, amplitude_threshold=None, upper_percentile=None):
    """
    Load pbrt raw data from .npy file.
    Input shape: (9, 240, 320) - 9 channels of 240x320 images
    Output: (6, H, W) - 6 IQ channels after processing
    """
    # Load the raw data: shape (9, 240, 320)
    ori_correlations = np.load(scene_path).astype(np.float32)
    if verbose:
        print(f"  [load_raw] Loaded {scene_path}, original shape: {ori_correlations.shape}")
    
    # Ensure shape is (9, 240, 320)
    if ori_correlations.shape[0] != 9:
        if len(ori_correlations.shape) == 3 and ori_correlations.shape[2] == 9:
            ori_correlations = np.transpose(ori_correlations, (2, 0, 1))
            if verbose:
                print(f"  [load_raw] Transposed from (H, W, 9) to (9, H, W)")
        elif len(ori_correlations.shape) == 3 and ori_correlations.shape[1] == 9:
            ori_correlations = np.transpose(ori_correlations, (1, 0, 2))
            if verbose:
                print(f"  [load_raw] Transposed from (H, 9, W) to (9, H, W)")
        else:
            raise ValueError(f"Unexpected shape: {ori_correlations.shape}, expected (9, 240, 320)")
    
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
    # Apply amplitude-based mask (adaptive percentile or fixed threshold), with optional upper cap.
    amp_mask = None
    # amplitude channels are correlations[2], correlations[5], correlations[8]
    amp_channels = correlations[[2, 5, 8], :, :]
    amp_mean = np.mean(np.abs(amp_channels), axis=0)

    # Determine lower threshold: given or adaptive percentile (default p=5%)
    if amplitude_threshold is None:
        p = 5.0
        amplitude_threshold = np.percentile(amp_mean, p)
        if verbose:
            masked_preview = (amp_mean < amplitude_threshold)
            masked_pct = 100.0 * np.count_nonzero(masked_preview) / masked_preview.size
            print(f"  [mask] adaptive percentile={p:.1f}% -> thr={amplitude_threshold:.6f}, preview masked={masked_pct:.2f}%")

    amp_mask_low = amp_mean < amplitude_threshold

    # Optional upper cap to remove extreme noisy amplitudes
    amp_mask_high = None
    if upper_percentile is not None:
        upper_thr = np.percentile(amp_mean, upper_percentile)
        amp_mask_high = amp_mean > upper_thr
        if verbose:
            low_pct = 100.0 * np.count_nonzero(amp_mask_low) / amp_mask_low.size
            high_pct = 100.0 * np.count_nonzero(amp_mask_high) / amp_mask_low.size
            overlap = np.logical_and(amp_mask_low, amp_mask_high)
            overlap_pct = 100.0 * np.count_nonzero(overlap) / amp_mask_low.size
            total_pct = 100.0 * np.count_nonzero(np.logical_or(amp_mask_low, amp_mask_high)) / amp_mask_low.size
            print(f"  [mask] lower={low_pct:.2f}% (thr={amplitude_threshold:.6f}), upper={high_pct:.2f}% (thr={upper_thr:.6f}), overlap={overlap_pct:.2f}%, total={total_pct:.2f}%")

    # Combine masks
    if amp_mask_high is not None:
        amp_mask = np.logical_or(amp_mask_low, amp_mask_high)
    else:
        amp_mask = amp_mask_low

    # zero out IQ outputs where amplitude mask is True
    if amp_mask is not None:
        for c in range(tof_IQs.shape[0]):
            tof_IQs[c][amp_mask] = 0.0

    return tof_IQs, amp_mask


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
    ideal_norm_root = 'data/ideal_IQ_masked'
    noise_norm_root = 'data/noise_IQ_masked'
    conf_root = 'data/confidence_masked'
    
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
    target_size = (512, 512)  # Match FLAT dataset resolution
    # Use adaptive thresholding: set to None to enable 5th-percentile auto threshold
    # You can also set a fixed value like 0.01 to use as absolute threshold
    amp_thresh = None  # None = adaptive 5th percentile, or set fixed value
    upper_percent = 99.5  # also mask extreme high amplitudes (optional), set to None to disable
    # Alternative thresholds to try:
    # amp_thresh = None  # adaptive 5%
    # amp_thresh = 0.01  # fixed threshold
    # upper_percent = 99.5  # mask top 0.5%
    # upper_percent = None  # disable upper masking
    print(f"Target size: {target_size}")
    print(f"Amplitude threshold: {amp_thresh if amp_thresh else 'adaptive 5%'}")
    print(f"Upper percentile: {upper_percent if upper_percent else 'disabled'}")
    print("=" * 60)
    
    for category in categories:
        category_ideal_root = os.path.join(ideal_root, category, '0')
        category_noise_root = os.path.join(noise_root, category, '0')
        category_noise_depth_root = os.path.join(noise_depth_root, category, '0')
        
        category_ideal_norm_root = os.path.join(ideal_norm_root, category, '0')
        category_noise_norm_root = os.path.join(noise_norm_root, category, '0')
        category_conf_root = os.path.join(conf_root, category, '0')
        
        # Create category output directories
        for d in [category_ideal_norm_root, category_noise_norm_root, category_conf_root]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
        
        if not os.path.exists(category_ideal_root):
            print(f"Warning: {category_ideal_root} does not exist, skipping category {category}")
            continue
        
        # Get all .npy files in the category
        idxs = [f for f in os.listdir(category_ideal_root) if f.endswith('.npy')]
        idxs.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))
        
        print(f"\n=== Processing category: {category} ===")
        print(f"Found {len(idxs)} files to process")
        
        for file_idx, idx in enumerate(idxs):
            ideal_path = os.path.join(category_ideal_root, idx)
            noise_path = os.path.join(category_noise_root, idx)
            noise_depth_path = os.path.join(category_noise_depth_root, idx)
            
            if not os.path.exists(ideal_path):
                print(f"Warning: {ideal_path} does not exist, skipping")
                continue
            if not os.path.exists(noise_path):
                print(f"Warning: {noise_path} does not exist, skipping")
                continue
            if not os.path.exists(noise_depth_path):
                print(f"Warning: {noise_depth_path} does not exist, skipping")
                continue
            
            print(f"\n[{file_idx+1}/{len(idxs)}] Processing {category}/0/{idx}")
            
            # Load and process ideal and noise IQ data
            try:
                ideal_IQs, ideal_mask = load_raw(
                    ideal_path,
                    target_size=target_size,
                    verbose=(file_idx == 0),
                    amplitude_threshold=amp_thresh,
                    upper_percentile=upper_percent
                )
                noise_IQs, noise_mask = load_raw(
                    noise_path,
                    target_size=target_size,
                    verbose=(file_idx == 0),
                    amplitude_threshold=amp_thresh,
                    upper_percentile=upper_percent
                )
                print(f"  Loaded IQ data - ideal shape: {ideal_IQs.shape}, noise shape: {noise_IQs.shape}")
                print(f"  Ideal value range: [{ideal_IQs.min():.4f}, {ideal_IQs.max():.4f}]")
                print(f"  Noise value range: [{noise_IQs.min():.4f}, {noise_IQs.max():.4f}]")
                if ideal_mask is not None:
                    im_pct = 100.0 * np.count_nonzero(ideal_mask) / ideal_mask.size
                    no_pct = 100.0 * np.count_nonzero(noise_mask) / noise_mask.size
                    print(f"  Applied amplitude mask (adaptive + upper {upper_percent}%) - ideal masked: {im_pct:.2f}%, noise masked: {no_pct:.2f}%")
            except Exception as e:
                print(f"  ERROR loading IQ data: {e}")
                continue
            
            # Normalize by noise max value
            noise_max = max(noise_IQs.max(), abs(noise_IQs.min()), 1e-8)
            ideal_IQs = ideal_IQs / noise_max
            noise_IQs = noise_IQs / noise_max
            print(f"  Normalized by noise_max: {noise_max:.4f}")
            print(f"  After normalization - ideal range: [{ideal_IQs.min():.4f}, {ideal_IQs.max():.4f}]")
            print(f"  After normalization - noise range: [{noise_IQs.min():.4f}, {noise_IQs.max():.4f}]")
            
            # Load and process depth for confidence map
            try:
                noise_depth = np.load(noise_depth_path)
                print(f"  Loaded depth - shape: {noise_depth.shape}, range: [{noise_depth.min():.4f}, {noise_depth.max():.4f}]")
                noise_depth = cv2.resize(noise_depth.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
                confidence = compute_gradient_confidence(noise_depth)
                # Apply amplitude mask from IQ processing to the confidence map
                if noise_mask is not None:
                    try:
                        # Ensure mask shape matches confidence; resize if necessary
                        if noise_mask.shape != confidence.shape:
                            mask_resized = cv2.resize(noise_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                        else:
                            mask_resized = noise_mask
                        confidence[mask_resized] = 0.0
                    except Exception as e:
                        print(f"  WARNING applying amp mask to confidence: {e}")

                print(f"  Computed confidence - shape: {confidence.shape}, range: [{confidence.min():.4f}, {confidence.max():.4f}]")
                conf_path = os.path.join(category_conf_root, idx)
                np.save(conf_path, confidence)
            except Exception as e:
                print(f"  ERROR processing depth/confidence: {e}")
                continue
            
            # Save individual IQ channels
            try:
                for i in range(6):
                    ideal_norm_path = os.path.join(category_ideal_norm_root, f"{idx.split('.')[0]}_{suffixes[i]}.npy")
                    noise_norm_path = os.path.join(category_noise_norm_root, f"{idx.split('.')[0]}_{suffixes[i]}.npy")
                    np.save(ideal_norm_path, ideal_IQs[i])
                    np.save(noise_norm_path, noise_IQs[i])
                print(f"  Saved all IQ channels and confidence map")
            except Exception as e:
                print(f"  ERROR saving files: {e}")
                continue
            
            print(f"  ✓ Successfully processed {category}/0/{idx}")
        
        print(f"\n=== Finished category: {category} ===\n")


