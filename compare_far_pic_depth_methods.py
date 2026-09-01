import argparse
import json
import os
from glob import glob

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from inference_depth_postprocess import opencv_depth_inpaint


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def finite_percentiles(values, percentiles):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None for _ in percentiles]
    return [float(np.percentile(values, p)) for p in percentiles]


def load_source_mapping(case_dir):
    mapping_path = os.path.join(case_dir, "source_mapping.json")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping["frame_mapping"]


def boundary_jump(depth, reference_depth, hole):
    hole = np.asarray(hole, dtype=bool)
    valid = (~hole) & np.isfinite(reference_depth)
    if hole.sum() == 0 or valid.sum() == 0:
        return None

    jumps = []
    pairs = [
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ]
    for hole_slice, valid_slice in pairs:
        edge = hole[hole_slice] & valid[valid_slice]
        if edge.any():
            diff = np.abs(depth[hole_slice][edge] - reference_depth[valid_slice][edge])
            diff = diff[np.isfinite(diff)]
            if diff.size:
                jumps.append(diff)
    if not jumps:
        return None
    jumps = np.concatenate(jumps)
    return {
        "mean": float(np.mean(jumps)),
        "median": float(np.median(jumps)),
        "p95": float(np.percentile(jumps, 95.0)),
    }


def masked_tv(depth, mask):
    mask = np.asarray(mask, dtype=bool)
    vals = []
    dx_mask = mask[:, 1:] & mask[:, :-1]
    if dx_mask.any():
        vals.append(np.abs(depth[:, 1:][dx_mask] - depth[:, :-1][dx_mask]))
    dy_mask = mask[1:, :] & mask[:-1, :]
    if dy_mask.any():
        vals.append(np.abs(depth[1:, :][dy_mask] - depth[:-1, :][dy_mask]))
    if not vals:
        return None
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95.0)),
    }


def summarize_method(name, output, raw, hole):
    valid_outside = (~hole) & np.isfinite(raw)
    hole_values = output[hole & np.isfinite(output)]
    outside_diff = output[valid_outside] - raw[valid_outside]
    outside_diff = outside_diff[np.isfinite(outside_diff)]
    summary = {
        "name": name,
        "zero_ratio_in_hole": float(np.mean(output[hole] == 0.0)) if hole.any() else 0.0,
        "hole_min_p50_p95_p99_max": finite_percentiles(hole_values, [0, 50, 95, 99, 100]),
        "outside_mean_abs_change": float(np.mean(np.abs(outside_diff))) if outside_diff.size else None,
        "outside_max_abs_change": float(np.max(np.abs(outside_diff))) if outside_diff.size else None,
        "boundary_jump_m": boundary_jump(output, raw, hole),
        "hole_total_variation_m": masked_tv(output, hole),
    }
    return summary


def image_limits(arrays, mask=None):
    values = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if mask is not None:
            vals = arr[mask & np.isfinite(arr)]
        else:
            vals = arr[np.isfinite(arr)]
        if vals.size:
            values.append(vals)
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_visualization(path, stem, raw, hole, methods):
    valid = (~hole) & np.isfinite(raw)
    vmin, vmax = image_limits([raw] + list(methods.values()), valid)
    delta_ours_prop = np.abs(methods["ours_hole_only"] - methods["propainter"])
    delta_ours_ns = np.abs(methods["ours_hole_only"] - methods["opencv_ns_r15"])
    delta_prop_ns = np.abs(methods["propainter"] - methods["opencv_ns_r15"])
    deltas = [delta_ours_prop[hole], delta_ours_ns[hole], delta_prop_ns[hole]]
    dvals = np.concatenate([d[np.isfinite(d)] for d in deltas if d.size and np.isfinite(d).any()])
    dmax = float(np.percentile(dvals, 98.0)) if dvals.size else 1.0
    dmax = max(dmax, 1e-6)

    raw_vis = raw.copy()
    raw_vis[hole] = np.nan
    panels = [
        ("raw depth\n(mask hidden)", raw_vis, "viridis", vmin, vmax),
        ("zero mask", hole.astype(np.float32), "gray", 0.0, 1.0),
        ("OpenCV NS r15", methods["opencv_ns_r15"], "viridis", vmin, vmax),
        ("OpenCV Telea r15", methods["opencv_telea_r15"], "viridis", vmin, vmax),
        ("ours hole_only", methods["ours_hole_only"], "viridis", vmin, vmax),
        ("ours restored", methods["ours_restored"], "viridis", vmin, vmax),
        ("ProPainter", methods["propainter"], "viridis", vmin, vmax),
        ("|ours hole-ProPainter|\nin mask", np.where(hole, delta_ours_prop, np.nan), "magma", 0.0, dmax),
        ("|ours hole-NS|\nin mask", np.where(hole, delta_ours_ns, np.nan), "magma", 0.0, dmax),
        ("|ProPainter-NS|\nin mask", np.where(hole, delta_prop_ns, np.nan), "magma", 0.0, dmax),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    for ax, (title, image, cmap, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(stem)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def aggregate(rows):
    out = {"num_samples": len(rows)}
    method_names = sorted(rows[0]["methods"].keys()) if rows else []
    for name in method_names:
        method_rows = [row["methods"][name] for row in rows]
        for key in ["zero_ratio_in_hole", "outside_mean_abs_change", "outside_max_abs_change"]:
            vals = [r[key] for r in method_rows if r.get(key) is not None]
            if vals:
                out[f"{name}_{key}_mean"] = float(np.mean(vals))
                out[f"{name}_{key}_max"] = float(np.max(vals))
        for nested_key in ["boundary_jump_m", "hole_total_variation_m"]:
            for stat in ["mean", "median", "p95"]:
                vals = [
                    r[nested_key][stat]
                    for r in method_rows
                    if r.get(nested_key) is not None and r[nested_key].get(stat) is not None
                ]
                if vals:
                    out[f"{name}_{nested_key}_{stat}_mean"] = float(np.mean(vals))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", default="propainter_depth_test/far_pic_noise_depth_240x320_m_zero_mask")
    parser.add_argument("--flow_dir", default="output/far_pic_noise_depth_flow_endpoint")
    parser.add_argument("--output_dir", default="output/far_pic_method_comparison")
    parser.add_argument("--max_visualizations", type=int, default=1000000)
    parser.add_argument("--inpaint_radius", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = args.case_dir
    out_dir = args.output_dir
    vis_dir = os.path.join(out_dir, "visualizations")
    ensure_dir(vis_dir)

    rows = []
    mapping = load_source_mapping(case_dir)
    for idx, item in enumerate(mapping):
        stem = item["source_stem"]
        raw_path = item["source_path_240x320_m"]
        raw = np.load(raw_path).astype(np.float32)
        mask_path = os.path.join(case_dir, "mask_npy", f"{idx:04d}.npy")
        hole = np.load(mask_path).astype(bool)

        ns = opencv_depth_inpaint(raw, hole, method="ns", radius=args.inpaint_radius)
        telea = opencv_depth_inpaint(raw, hole, method="telea", radius=args.inpaint_radius)
        ours_hole = np.load(
            os.path.join(args.flow_dir, "hole_only", f"{stem}_hole_only.npy")
        ).astype(np.float32)
        ours_restored = np.load(
            os.path.join(args.flow_dir, "restored", f"{stem}_restored.npy")
        ).astype(np.float32)
        propainter = np.load(
            os.path.join(case_dir, "restored_by_stem", f"{stem}_propainter_restored.npy")
        ).astype(np.float32)

        methods = {
            "opencv_ns_r15": ns,
            "opencv_telea_r15": telea,
            "ours_hole_only": ours_hole,
            "ours_restored": ours_restored,
            "propainter": propainter,
        }
        method_summary = {
            name: summarize_method(name, output, raw, hole)
            for name, output in methods.items()
        }

        row = {
            "frame_index": idx,
            "source_stem": stem,
            "raw_path": raw_path,
            "mask_path": mask_path,
            "mask_ratio": float(hole.mean()),
            "raw_valid_min_p50_p95_p99_max": finite_percentiles(
                raw[(~hole) & np.isfinite(raw)], [0, 50, 95, 99, 100]
            ),
            "methods": method_summary,
        }
        rows.append(row)

        if idx < args.max_visualizations:
            save_visualization(
                os.path.join(vis_dir, f"{idx:02d}_{stem}.png"),
                stem,
                raw,
                hole,
                methods,
            )

    result = {
        "case_dir": os.path.abspath(case_dir),
        "flow_dir": os.path.abspath(args.flow_dir),
        "output_dir": os.path.abspath(out_dir),
        "notes": (
            "No ground truth is available. Metrics are no-reference diagnostics: "
            "lower boundary_jump can mean smoother mask boundaries, lower outside change "
            "means better preservation of observed valid pixels, and lower TV means smoother "
            "filled holes but may also indicate oversmoothing."
        ),
        "aggregate": aggregate(rows),
        "per_sample": rows,
    }
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved comparison to {out_dir}")


if __name__ == "__main__":
    main()
