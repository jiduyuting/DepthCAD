#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    mkdir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def finite_percentiles(values, percentiles):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None for _ in percentiles]
    return [float(np.percentile(values, p)) for p in percentiles]


def boundary_jump(depth, reference_depth, hole):
    hole = np.asarray(hole, dtype=bool)
    valid = (~hole) & np.isfinite(reference_depth) & (reference_depth > 0.0)
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
    dx = mask[:, 1:] & mask[:, :-1]
    if dx.any():
        vals.append(np.abs(depth[:, 1:][dx] - depth[:, :-1][dx]))
    dy = mask[1:, :] & mask[:-1, :]
    if dy.any():
        vals.append(np.abs(depth[1:, :][dy] - depth[:-1, :][dy]))
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


def summarize_method(output, raw, observed, repair_mask, has_gt):
    outside = (~repair_mask) & np.isfinite(observed) & (observed > 0.0)
    diff = output[outside] - observed[outside]
    diff = diff[np.isfinite(diff)]
    inside = repair_mask & np.isfinite(output)
    out = {
        "zero_ratio_in_repair_mask": float(np.mean(output[repair_mask] == 0.0)) if repair_mask.any() else 0.0,
        "finite_ratio_in_repair_mask": float(np.mean(np.isfinite(output[repair_mask]))) if repair_mask.any() else 1.0,
        "outside_mean_abs_change": float(np.mean(np.abs(diff))) if diff.size else None,
        "outside_max_abs_change": float(np.max(np.abs(diff))) if diff.size else None,
        "hole_min_p50_p95_p99_max": finite_percentiles(output[inside], [0, 50, 95, 99, 100]),
        "boundary_jump_m": boundary_jump(output, observed, repair_mask),
        "hole_total_variation_m": masked_tv(output, repair_mask),
    }
    if has_gt:
        gt = repair_mask & np.isfinite(raw) & (raw > 0.0) & np.isfinite(output)
        if gt.any():
            abs_err = np.abs(output[gt] - raw[gt])
            out["eval_mask_mae"] = float(np.mean(abs_err))
            out["eval_mask_rmse"] = float(np.sqrt(np.mean(abs_err ** 2)))
            out["eval_mask_p95"] = float(np.percentile(abs_err, 95.0))
            out["eval_mask_count"] = int(gt.sum())
        else:
            out["eval_mask_mae"] = None
            out["eval_mask_rmse"] = None
            out["eval_mask_p95"] = None
            out["eval_mask_count"] = 0
    return out


def aggregate_rows(rows):
    out = {"num_rows": len(rows)}
    methods = sorted({name for row in rows for name in row["methods"].keys()})
    for method in methods:
        method_rows = [row["methods"][method] for row in rows if method in row["methods"]]
        for key in [
            "zero_ratio_in_repair_mask",
            "finite_ratio_in_repair_mask",
            "outside_mean_abs_change",
            "outside_max_abs_change",
            "eval_mask_mae",
            "eval_mask_rmse",
            "eval_mask_p95",
        ]:
            vals = [float(r[key]) for r in method_rows if r.get(key) is not None]
            if vals:
                out[f"{method}_{key}_mean"] = float(np.mean(vals))
                out[f"{method}_{key}_max"] = float(np.max(vals))
        if any("eval_mask_count" in r for r in method_rows):
            weighted_total = 0.0
            weighted_count = 0
            for r in method_rows:
                if r.get("eval_mask_mae") is None:
                    continue
                count = int(r.get("eval_mask_count", 0))
                weighted_total += float(r["eval_mask_mae"]) * count
                weighted_count += count
            if weighted_count:
                out[f"{method}_eval_mask_mae_weighted"] = float(weighted_total / weighted_count)
                out[f"{method}_eval_mask_count"] = int(weighted_count)
        for nested in ["boundary_jump_m", "hole_total_variation_m"]:
            for stat in ["mean", "median", "p95"]:
                vals = [
                    r[nested][stat]
                    for r in method_rows
                    if r.get(nested) is not None and r[nested].get(stat) is not None
                ]
                if vals:
                    out[f"{method}_{nested}_{stat}_mean"] = float(np.mean(vals))
    return out


def discover_methods(case_dir, requested):
    outputs = case_dir / "outputs"
    if requested:
        return requested
    methods = []
    if not outputs.is_dir():
        return methods
    for path in sorted(outputs.iterdir()):
        if path.is_dir() and any(path.glob("*.npy")):
            methods.append(path.name)
    return methods


def load_method(case_dir, method, stem):
    method_dir = case_dir / "outputs" / method
    candidates = [method_dir / f"{stem}_{method}.npy", method_dir / f"{stem}.npy"]
    candidates.extend(sorted(method_dir.glob(f"{stem}_*.npy")))
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float32), path
    return None, None


def finite_values(arr, mask=None):
    arr = np.asarray(arr)
    if mask is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[mask & np.isfinite(arr)]
    return vals


def image_limits(arrays, valid_mask):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        v = finite_values(arr, valid_mask)
        v = v[v > 0.0]
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize_depth(depth, vmin, vmax, cmap_name="viridis"):
    depth = np.asarray(depth, dtype=np.float32)
    norm = (depth - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    invalid = (~np.isfinite(depth)) | (depth <= 0.0)
    rgb[invalid] = 0
    return rgb


def draw_mask_boundary(rgb, mask, color=(255, 0, 0)):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.max() == 0:
        return rgb
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = dilated != eroded
    out = rgb.copy()
    out[boundary] = color
    return out


def save_panel_grid(path, title, panels, cols=5, dpi=130):
    cols = min(cols, max(1, len(panels)))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.3 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax, panel in zip(axes, panels):
        label, image, cmap, vmin, vmax = panel
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
        if cmap is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_contact_sheet(path, rows, labels, thumb_w=176):
    if not rows:
        return
    label_h = 28
    row_gap = 10
    col_gap = 7
    text_w = 225
    h, w = rows[0]["images"][0].shape[:2]
    thumb_h = int(round(h * (thumb_w / float(w))))
    canvas_w = text_w + len(labels) * thumb_w + max(0, len(labels) - 1) * col_gap
    canvas_h = label_h + len(rows) * thumb_h + max(0, len(rows) - 1) * row_gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, label in enumerate(labels):
        x = text_w + i * (thumb_w + col_gap)
        cv2.putText(canvas, label[:22], (x + 3, 19), font, 0.43, (20, 20, 20), 1, cv2.LINE_AA)

    y = label_h
    for row in rows:
        cv2.putText(canvas, row["stem"][:32], (4, y + 17), font, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        for i, image in enumerate(row["images"]):
            thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            x = text_w + i * (thumb_w + col_gap)
            canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        y += thumb_h + row_gap
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", default="output/far_pic_benchmark/bad_depth_mask_v1")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--max_frames", type=int, default=1000000)
    parser.add_argument("--max_contact_rows", type=int, default=1000000)
    parser.add_argument("--thumb_w", type=int, default=176)
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    external_dir = case_dir / "external_inputs"
    mapping = load_json(external_dir / "source_mapping.json")["frame_mapping"]
    out_dir = Path(args.output_dir).resolve() if args.output_dir else case_dir / "visualizations_compare"
    detail_dir = out_dir / "per_frame"
    quick_dir = out_dir / "quick_png"
    mkdir(detail_dir)
    mkdir(quick_dir)

    case_summary_path = case_dir / "summary.json"
    case_summary = load_json(case_summary_path) if case_summary_path.exists() else {}
    case_kind = str(case_summary.get("kind", ""))
    has_gt = case_kind == "synthetic" or case_dir.name.startswith("synthetic")
    methods = discover_methods(case_dir, args.methods)
    if not methods:
        raise SystemExit(f"ERROR: no method outputs found under {case_dir / 'outputs'}")

    contact_labels = ["raw", "input", "mask"] + methods
    contact_rows = []
    rows = []
    missing = []

    for item in mapping[: int(args.max_frames)]:
        idx = int(item["frame_index"])
        stem = item["source_stem"]
        raw_path = item.get("source_path_240x320_m") or item.get("source_path_m")
        if raw_path is None:
            raise KeyError("source mapping item must contain source_path_240x320_m or source_path_m")
        raw = np.load(raw_path).astype(np.float32)
        observed = np.load(external_dir / "depth_npy" / f"{idx:04d}.npy").astype(np.float32)
        mask = np.load(external_dir / "mask_npy" / f"{idx:04d}.npy").astype(bool)

        loaded = {}
        paths = {}
        for method in methods:
            arr, path = load_method(case_dir, method, stem)
            if arr is None:
                missing.append({"frame_index": idx, "source_stem": stem, "method": method})
                continue
            if arr.shape != raw.shape:
                arr = cv2.resize(arr, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            loaded[method] = arr
            paths[method] = str(path)

        valid = (~mask) & np.isfinite(observed) & (observed > 0.0)
        vmin, vmax = image_limits([raw, observed] + list(loaded.values()), valid)
        raw_hidden = raw.copy()
        observed_hidden = observed.copy()
        raw_hidden[mask] = np.nan
        observed_hidden[mask] = np.nan

        panels = [
            ("raw source\n(mask hidden)", raw_hidden, "viridis", vmin, vmax),
            ("model input\n(mask hidden)", observed_hidden, "viridis", vmin, vmax),
            ("repair mask", mask.astype(np.float32), "gray", 0.0, 1.0),
        ]
        for method in methods:
            if method in loaded:
                panels.append((method, loaded[method], "viridis", vmin, vmax))
        save_panel_grid(
            detail_dir / f"{idx:02d}_{stem}.png",
            f"{idx:02d} {stem} | mask={mask.mean():.3f} | scale={vmin:.2f}-{vmax:.2f} m",
            panels,
            cols=5,
        )

        raw_rgb = draw_mask_boundary(colorize_depth(raw, vmin, vmax), mask)
        observed_rgb = draw_mask_boundary(colorize_depth(observed, vmin, vmax), mask)
        mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        image_row = [raw_rgb, observed_rgb, mask_rgb]
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_raw.png"), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_input.png"), cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_mask.png"), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
        for method in methods:
            if method not in loaded:
                image_row.append(np.zeros_like(raw_rgb))
                continue
            rgb = draw_mask_boundary(colorize_depth(loaded[method], vmin, vmax), mask)
            image_row.append(rgb)
            cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_{method}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if len(contact_rows) < int(args.max_contact_rows):
            contact_rows.append({"stem": f"{idx:02d}_{stem}", "images": image_row})

        method_summary = {
            method: summarize_method(arr, raw, observed, mask, has_gt)
            for method, arr in loaded.items()
        }
        rows.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "mask_ratio": float(mask.mean()),
                "raw_path": raw_path,
                "observed_path": str(external_dir / "depth_npy" / f"{idx:04d}.npy"),
                "mask_path": str(external_dir / "mask_npy" / f"{idx:04d}.npy"),
                "depth_vmin": float(vmin),
                "depth_vmax": float(vmax),
                "method_paths": paths,
                "methods": method_summary,
                "detail_png": str(detail_dir / f"{idx:02d}_{stem}.png"),
            }
        )

    save_contact_sheet(out_dir / "contact_sheet.png", contact_rows, contact_labels, thumb_w=int(args.thumb_w))
    result = {
        "case_dir": str(case_dir),
        "output_dir": str(out_dir),
        "methods": methods,
        "has_synthetic_gt": bool(has_gt),
        "notes": (
            "For real bad-depth masks there is no true ground truth inside the repair mask. "
            "Use outside-change, boundary-jump, hole-TV, and visual sheets as diagnostics. "
            "Synthetic cases additionally report eval_mask_* against the hidden raw depth."
        ),
        "aggregate": aggregate_rows(rows),
        "missing": missing,
        "per_sample": rows,
    }
    save_json(out_dir / "summary.json", result)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved comparison visualizations: {out_dir}")
    if missing:
        print(f"WARNING: missing {len(missing)} method outputs; see {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
