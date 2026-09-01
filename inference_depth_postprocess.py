"""
DepthCAD Inference with Confidence-based Post-processing

思路：低置信度区域的深度值几乎是常数（不可靠），
用周围高置信度像素的加权值来填充空洞区域。

方法：
1. 用原始 ControlNet 推理得到 pred_depth
2. 对每个低置信度像素，用周围高置信度像素的距离加权平均填充
"""

import cv2
import numpy as np


def _as_hole_mask(mask):
    """Return a boolean mask with True at hole pixels."""
    return np.asarray(mask) > 0.5


def _valid_depth_mask(depth, hole_mask):
    return (~hole_mask) & np.isfinite(depth)


def _normalize_from_valid(depth, valid_mask):
    valid_depth = depth[valid_mask]
    if valid_depth.size == 0:
        return None, None, None

    d_min = float(valid_depth.min())
    d_max = float(valid_depth.max())
    d_range = d_max - d_min
    if d_range < 1e-6:
        return None, None, None

    depth_norm = (depth - d_min) / d_range
    depth_norm = np.nan_to_num(depth_norm, nan=0.0, neginf=0.0, posinf=1.0)
    depth_norm = np.clip(depth_norm, 0.0, 1.0).astype(np.float32)
    return depth_norm, d_min, d_range


def _normalize_guidance(guidance):
    if guidance is None:
        return None

    guidance = np.asarray(guidance, dtype=np.float32)
    finite = np.isfinite(guidance)
    if finite.sum() == 0:
        return None

    lo, hi = np.percentile(guidance[finite], [1.0, 99.0])
    g_range = float(hi - lo)
    if g_range < 1e-6:
        return None

    guidance_norm = (guidance - lo) / g_range
    guidance_norm = np.nan_to_num(guidance_norm, nan=0.0, neginf=0.0, posinf=1.0)
    return np.clip(guidance_norm, 0.0, 1.0).astype(np.float32)


def opencv_depth_inpaint(depth, hole_mask, method="ns", radius=5):
    """
    Fill explicit depth holes with OpenCV Telea or Navier-Stokes inpainting.

    Only pixels where hole_mask is true are replaced; original valid depth pixels
    are copied back verbatim.
    """
    depth = np.asarray(depth, dtype=np.float32)
    hole = _as_hole_mask(hole_mask)
    if hole.sum() == 0:
        return depth.copy()

    valid = _valid_depth_mask(depth, hole)
    depth_norm, d_min, d_range = _normalize_from_valid(depth, valid)
    if depth_norm is None:
        return depth.copy()

    mask_uint8 = (hole.astype(np.uint8) * 255)
    if method == "telea":
        flags = cv2.INPAINT_TELEA
    elif method == "ns":
        flags = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unsupported OpenCV depth inpaint method: {method}")

    filled_norm = cv2.inpaint(depth_norm, mask_uint8, inpaintRadius=radius, flags=flags)
    filled_norm = np.clip(filled_norm, 0.0, 1.0)

    filled = depth.copy()
    filled[hole] = filled_norm[hole] * d_range + d_min
    filled[~hole] = depth[~hole]
    return filled.astype(np.float32)


def edge_aware_ns_bilateral_fill(
    depth,
    hole_mask,
    confidence=None,
    guidance=None,
    inpaint_radius=5,
    bilateral_radius=5,
    sigma_spatial=None,
    sigma_depth=0.05,
    sigma_guidance=0.25,
    iterations=1,
):
    """
    NS initialization followed by bilateral smoothing only inside holes.

    The bilateral range term is computed in normalized depth space, so strong
    depth jumps at the valid boundary are not averaged across aggressively.
    Valid pixels remain hard constraints and are copied back after each pass.
    confidence, when supplied, is used as a neighbor reliability term rather
    than a hard guide so binary hole confidence does not disconnect the fill
    from its boundary.
    """
    depth = np.asarray(depth, dtype=np.float32)
    hole = _as_hole_mask(hole_mask)
    if hole.sum() == 0:
        return depth.copy()

    valid = _valid_depth_mask(depth, hole)
    depth_norm, d_min, d_range = _normalize_from_valid(depth, valid)
    if depth_norm is None:
        return depth.copy()

    current = opencv_depth_inpaint(depth, hole, method="ns", radius=inpaint_radius)
    current[valid] = depth[valid]

    if sigma_spatial is None:
        sigma_spatial = max(float(bilateral_radius) * 0.5, 1.0)
    bilateral_radius = max(int(bilateral_radius), 1)
    iterations = max(int(iterations), 1)

    conf_norm = _normalize_guidance(confidence)
    if conf_norm is not None:
        reliability = 0.25 + 0.75 * conf_norm
    else:
        reliability = np.ones_like(depth, dtype=np.float32)

    guide_norm = _normalize_guidance(guidance)
    if guide_norm is not None and guide_norm.shape != depth.shape:
        guide_norm = None

    finite = np.isfinite(depth) | hole

    for _ in range(iterations):
        current[valid] = depth[valid]
        current_norm = np.clip((current - d_min) / d_range, 0.0, 1.0).astype(np.float32)
        current_norm = np.nan_to_num(current_norm, nan=0.0, neginf=0.0, posinf=1.0)
        if guide_norm is None:
            active_guide = current_norm
        else:
            active_guide = guide_norm

        pad = bilateral_radius
        padded_depth = cv2.copyMakeBorder(current, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        padded_depth_norm = cv2.copyMakeBorder(current_norm, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        padded_guide = cv2.copyMakeBorder(active_guide, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        padded_hole = cv2.copyMakeBorder(hole.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        padded_reliability = cv2.copyMakeBorder(reliability, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        padded_finite = cv2.copyMakeBorder(finite.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        acc = np.zeros_like(current, dtype=np.float32)
        wsum = np.zeros_like(current, dtype=np.float32)

        center_depth_norm = current_norm
        center_guide = active_guide

        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                spatial_dist2 = float(dx * dx + dy * dy)
                spatial_w = np.exp(-0.5 * spatial_dist2 / (sigma_spatial * sigma_spatial))

                ys = pad + dy
                xs = pad + dx
                neighbor_depth = padded_depth[ys:ys + depth.shape[0], xs:xs + depth.shape[1]]
                neighbor_depth = np.nan_to_num(neighbor_depth, nan=0.0, neginf=0.0, posinf=0.0)
                neighbor_depth_norm = padded_depth_norm[ys:ys + depth.shape[0], xs:xs + depth.shape[1]]
                neighbor_guide = padded_guide[ys:ys + depth.shape[0], xs:xs + depth.shape[1]]
                neighbor_hole = padded_hole[ys:ys + depth.shape[0], xs:xs + depth.shape[1]]
                neighbor_reliability = padded_reliability[ys:ys + depth.shape[0], xs:xs + depth.shape[1]]
                neighbor_finite = padded_finite[ys:ys + depth.shape[0], xs:xs + depth.shape[1]]

                depth_diff = neighbor_depth_norm - center_depth_norm
                depth_w = np.exp(-0.5 * (depth_diff * depth_diff) / (sigma_depth * sigma_depth))

                guide_diff = neighbor_guide - center_guide
                guide_w = np.exp(-0.5 * (guide_diff * guide_diff) / (sigma_guidance * sigma_guidance))

                # Filled hole pixels are useful for continuity, but boundary
                # valid pixels should dominate whenever they are nearby.
                trust_w = neighbor_reliability * np.where(neighbor_hole > 0.5, 0.65, 1.0)
                w = spatial_w * depth_w * guide_w * trust_w * neighbor_finite

                acc += w.astype(np.float32) * neighbor_depth
                wsum += w.astype(np.float32)

        update = hole & (wsum > 1e-8)
        current[update] = acc[update] / wsum[update]
        current[valid] = depth[valid]

    filled = depth.copy()
    filled[hole] = current[hole]
    filled[~hole] = depth[~hole]
    return filled.astype(np.float32)


def local_plane_fit_fill(
    depth,
    hole_mask,
    confidence=None,
    max_ring_radius=12,
    min_boundary_points=12,
    fallback_radius=5,
    clip_margin_ratio=0.10,
):
    """
    Fill each connected hole component by fitting z = ax + by + c to its valid boundary.

    The implementation intentionally stays small and deterministic: connected
    components are fit independently, valid pixels are never modified, and
    components with too few boundary samples fall back to NS inpainting.
    """
    depth = np.asarray(depth, dtype=np.float32)
    hole = _as_hole_mask(hole_mask)
    if hole.sum() == 0:
        return depth.copy()

    valid = _valid_depth_mask(depth, hole)
    if valid.sum() == 0:
        return depth.copy()

    fallback = opencv_depth_inpaint(depth, hole, method="ns", radius=fallback_radius)
    filled = fallback.copy()

    conf_norm = _normalize_guidance(confidence)
    labels_count, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    valid_depth = depth[valid]
    global_range = float(valid_depth.max() - valid_depth.min()) if valid_depth.size else 0.0

    for label in range(1, labels_count):
        component = labels == label
        area = int(component.sum())
        if area == 0:
            continue

        boundary = None
        for ring_radius in range(1, max_ring_radius + 1):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * ring_radius + 1, 2 * ring_radius + 1),
            )
            expanded = cv2.dilate(component.astype(np.uint8), kernel) > 0
            candidate = expanded & (~hole) & np.isfinite(depth)
            if candidate.sum() >= min_boundary_points:
                boundary = candidate
                break

        if boundary is None:
            continue

        by, bx = np.nonzero(boundary)
        bz = depth[boundary].astype(np.float32)
        if bz.size < min_boundary_points:
            continue

        cy, cx = np.mean(np.nonzero(component), axis=1)
        scale = max(np.sqrt(float(area)), 1.0)
        design = np.stack(
            [(bx.astype(np.float32) - cx) / scale,
             (by.astype(np.float32) - cy) / scale,
             np.ones_like(bz, dtype=np.float32)],
            axis=1,
        )

        distance_to_component = cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 3)
        weights = 1.0 / np.square(distance_to_component[boundary].astype(np.float32) + 1.0)
        if conf_norm is not None and conf_norm.shape == depth.shape:
            weights *= 0.25 + 0.75 * conf_norm[boundary]

        try:
            sqrt_w = np.sqrt(np.maximum(weights, 1e-6)).astype(np.float32)
            coeff, _, rank, _ = np.linalg.lstsq(design * sqrt_w[:, None], bz * sqrt_w, rcond=None)
            if rank < 3:
                continue

            residual = np.abs(design @ coeff - bz)
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            robust_thr = med + max(3.0 * 1.4826 * mad, 0.02 * max(global_range, 1e-6))
            inliers = residual <= robust_thr
            if inliers.sum() >= min_boundary_points and inliers.sum() < residual.size:
                coeff, _, rank, _ = np.linalg.lstsq(
                    design[inliers] * sqrt_w[inliers, None],
                    bz[inliers] * sqrt_w[inliers],
                    rcond=None,
                )
                if rank < 3:
                    continue
        except np.linalg.LinAlgError:
            continue

        hy, hx = np.nonzero(component)
        hole_design = np.stack(
            [(hx.astype(np.float32) - cx) / scale,
             (hy.astype(np.float32) - cy) / scale,
             np.ones_like(hx, dtype=np.float32)],
            axis=1,
        )
        pred = hole_design @ coeff

        local_min = float(bz.min())
        local_max = float(bz.max())
        local_range = local_max - local_min
        margin = max(local_range * clip_margin_ratio, global_range * 0.02)
        pred = np.clip(pred, local_min - margin, local_max + margin)
        filled[component] = pred.astype(np.float32)

    filled[~hole] = depth[~hole]
    return filled.astype(np.float32)


def confidence_fill_depth(depth, confidence, threshold=0.5, k=5):
    """
    对低置信度区域的深度值用高置信度邻近像素加权填充。

    Args:
        depth: (H, W) 原始预测深度图
        confidence: (H, W) 置信度图 (0-1)
        threshold: 置信度阈值，低于此值为空洞
        k: 搜索窗口半径 (k*2+1 为窗口大小)

    Returns:
        filled_depth: (H, W) 填充后的深度图
    """
    filled = depth.copy()
    h, w = depth.shape
    low_conf_mask = confidence < threshold

    # 对于每个低置信度像素
    for y in range(h):
        for x in range(w):
            if not low_conf_mask[y, x]:
                continue

            # 在周围窗口内找高置信度像素
            weights = []
            values = []

            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if confidence[ny, nx] >= threshold:
                            # 距离衰减权重
                            dist = np.sqrt(dx * dx + dy * dy)
                            if dist < 0.1:
                                dist = 0.1  # 防止除零
                            w_val = 1.0 / (dist ** 2)  # 距离平方反比
                            weights.append(w_val * confidence[ny, nx])  # 也考虑置信度
                            values.append(depth[ny, nx])

            if values:
                filled[y, x] = np.average(values, weights=weights)
            else:
                # 周围没有高置信度像素，用全局高置信度区域的均值
                high_conf_mean = depth[confidence >= threshold].mean()
                filled[y, x] = high_conf_mean if not np.isnan(high_conf_mean) else depth.mean()

    return filled


def confidence_fill_depth_fast(depth, confidence, threshold=0.5):
    """
    在归一化空间做置信度填充：
    1. 归一化到 0-1
    2. 在归一化空间用膨胀填充
    3. 反算回原始深度空间
    """
    filled = depth.copy()
    h, w = depth.shape

    low_conf_mask = confidence < threshold
    high_conf_mask = ~low_conf_mask

    if low_conf_mask.sum() == 0:
        return depth

    # 用高置信度区域计算归一化参数
    valid_depth = depth[high_conf_mask]
    d_min, d_max = valid_depth.min(), valid_depth.max()
    d_range = d_max - d_min
    if d_range < 1e-6:
        return depth  # 避免除零

    # 归一化到 0-1
    depth_norm = (depth - d_min) / d_range
    filled_norm = depth_norm.copy()

    # 在归一化空间做多尺度膨胀填充
    for kernel_size in [25, 15, 7]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_dilated = cv2.dilate(high_conf_mask.astype(np.uint8), kernel) > 0
        dilated_norm = cv2.dilate(depth_norm.astype(np.float32), kernel)
        # 只更新低置信度区域
        update_mask = low_conf_mask & mask_dilated
        filled_norm[update_mask] = dilated_norm[update_mask]

    # 对边界做羽化（平滑过渡）
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_erode = cv2.erode(high_conf_mask.astype(np.uint8), kernel_edge) > 0
    boundary_mask = low_conf_mask & ~mask_erode
    if boundary_mask.sum() > 0:
        filled_norm = cv2.GaussianBlur(filled_norm.astype(np.float32), (15, 15), 0)

    # 反算回原始深度空间
    filled = filled_norm * d_range + d_min
    return filled


if __name__ == '__main__':
    # 测试
    pred_depth_path = '/data/pre_student/GJ/DepthCAD/output/visualization/masked/bathroom/1/100/pred_depth.npy'
    conf_path = '/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/confidence_masked/bathroom/1/100.npy'

    pred_depth = np.load(pred_depth_path)
    confidence = np.load(conf_path)

    confidence_resized = cv2.resize(confidence, (pred_depth.shape[1], pred_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    low_conf_mask = confidence_resized < 0.5
    print("=== 原始 pred_depth ===")
    print("  range: [{:.4f}, {:.4f}], mean: {:.4f}".format(pred_depth.min(), pred_depth.max(), pred_depth.mean()))
    print("  低置信度区域 mean: {:.4f}, std: {:.4f}".format(
        pred_depth[low_conf_mask].mean(), pred_depth[low_conf_mask].std()))

    filled_fast = confidence_fill_depth_fast(pred_depth, confidence_resized)

    print("\n=== 后处理后 ===")
    print("  range: [{:.4f}, {:.4f}], mean: {:.4f}".format(filled_fast.min(), filled_fast.max(), filled_fast.mean()))
    print("  低置信度区域 mean: {:.4f}, std: {:.4f}".format(
        filled_fast[low_conf_mask].mean(), filled_fast[low_conf_mask].std()))

    print("\n=== 变化量 ===")
    diff = np.abs(filled_fast - pred_depth)
    print("  max diff: {:.4f}, mean diff: {:.4f}".format(diff.max(), diff.mean()))
    print("  变化像素数: {}".format((diff > 0.001).sum()))

    out_path = pred_depth_path.replace('.npy', '_filled.npy')
    np.save(out_path, filled_fast)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(pred_depth, cmap='turbo')
    axes[0].set_title('Original Depth')
    axes[0].axis('off')

    axes[1].imshow(confidence_resized, cmap='gray')
    axes[1].set_title('Confidence Map')
    axes[1].axis('off')

    axes[2].imshow(filled_fast, cmap='turbo')
    axes[2].set_title('Filled Depth')
    axes[2].axis('off')

    axes[3].imshow(np.abs(filled_fast - pred_depth), cmap='hot')
    axes[3].set_title('|diff|')
    axes[3].axis('off')

    plt.tight_layout()
    out_png = out_path.replace('.npy', '.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print("\nSaved to", out_png)
