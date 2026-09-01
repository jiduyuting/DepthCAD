import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank, visualize, and classify failure cases from real raw9 masked self-test outputs."
    )
    parser.add_argument(
        "--selftest_dir",
        type=str,
        default=(
            "output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030/"
            "real_val_selftest"
        ),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--worse_margin", type=float, default=0.005)
    parser.add_argument(
        "--raw9_transform",
        type=str,
        default=None,
        choices=["none", "flip_lr", "flip_ud", "rot180"],
        help="Override raw9 transform for amplitude visualization. Defaults to self-test summary/per-sample metadata.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def depth_to_meters(depth, unit="auto"):
    depth = np.asarray(depth, dtype=np.float32)
    if unit == "m":
        return depth
    if unit == "mm":
        return depth / 1000.0
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return depth
    median = float(np.nanmedian(finite))
    p95 = float(np.nanpercentile(finite, 95.0))
    if median > 20.0 or p95 > 100.0:
        return depth / 1000.0
    return depth


def raw9_to_amplitude(raw9, mode):
    raw9 = np.asarray(raw9, dtype=np.float32)
    if mode == "raw_258":
        channels = raw9[[2, 5, 8]]
        return channels, channels.mean(axis=0)
    pairs = [(0, 1), (2, 3), (4, 5)]
    amp = np.stack([np.hypot(raw9[i], raw9[q]) for i, q in pairs], axis=0).astype(np.float32)
    return amp, amp.mean(axis=0)


def apply_spatial_transform(image, mode):
    mode = str(mode or "none")
    if mode == "none":
        return np.asarray(image)
    if mode == "flip_lr":
        return np.flip(image, axis=-1).copy()
    if mode == "flip_ud":
        return np.flip(image, axis=-2).copy()
    if mode == "rot180":
        return np.flip(np.flip(image, axis=-1), axis=-2).copy()
    raise ValueError(f"Unknown spatial transform: {mode}")


def raw9_transform_for_case(row, summary, args):
    if args is not None and args.raw9_transform is not None:
        return str(args.raw9_transform)
    mode = row.get("raw9_transform_estimated") or row.get("raw9_transform")
    if not mode:
        mode = summary.get("raw9_transform", "none")
    if mode == "auto":
        return "none"
    return str(mode or "none")


def dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def connected_component_stats(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
            "largest_bbox_h": 0,
            "largest_bbox_w": 0,
            "touches_border": False,
        }
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return {
            "component_count": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
            "largest_bbox_h": 0,
            "largest_bbox_w": 0,
            "touches_border": False,
        }
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    height, width = mask.shape
    touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
    area = int(stats[largest_label, cv2.CC_STAT_AREA])
    return {
        "component_count": int(num_labels - 1),
        "largest_component_area": area,
        "largest_component_ratio": float(area / max(int(mask.sum()), 1)),
        "largest_bbox_h": h,
        "largest_bbox_w": w,
        "touches_border": bool(touches_border),
    }


def masked_mae(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return None, 0
    return float(np.mean(np.abs(pred[valid] - target[valid]))), int(valid.sum())


def safe_percentile(values, percentile, default=0.0):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return default
    return float(np.percentile(values, percentile))


def load_case_arrays(selftest_dir, row, summary, args=None):
    name = row["name"]
    repeat = int(row.get("repeat", 0))
    case_name = f"{name}_r{repeat:02d}"
    base = Path(selftest_dir)
    clean = depth_to_meters(np.load(row["depth_path"]), summary.get("depth_unit", "auto"))
    raw9 = np.load(row["raw_path"]).astype(np.float32)
    raw9_transform = raw9_transform_for_case(row, summary, args)
    raw9 = apply_spatial_transform(raw9, raw9_transform).astype(np.float32, copy=False)
    arrays = {
        "case_name": case_name,
        "clean": clean.astype(np.float32),
        "raw9": raw9,
        "raw9_transform": raw9_transform,
        "anchor": np.load(base / "anchor" / f"{case_name}_anchor.npy").astype(np.float32),
        "model": np.load(base / "restored" / f"{case_name}_restored.npy").astype(np.float32),
        "hole_only": np.load(base / "hole_only" / f"{case_name}_hole_only.npy").astype(np.float32),
        "corrupted": np.load(base / "corrupted" / f"{case_name}_corrupted.npy").astype(np.float32),
        "mask": np.load(base / "mask" / f"{case_name}_mask.npy").astype(bool),
        "condition_mask": np.load(base / "condition_mask" / f"{case_name}_condition_mask.npy").astype(bool),
    }
    _, amp_mean = raw9_to_amplitude(raw9, summary.get("amplitude_mode", "iq6"))
    arrays["amplitude_mean"] = amp_mean.astype(np.float32)
    return arrays


def classify_case(row, metrics):
    labels = []
    improve = float(row.get("mask_improve_vs_anchor", 0.0))
    anchor_mae = float(row.get("anchor_mask_mae", 0.0))
    model_mae = float(row.get("model_mask_mae", 0.0))

    if improve < 0.0:
        labels.append("模型改坏/anchor更稳")
    elif improve < 0.08:
        labels.append("提升很小")
    if model_mae >= 0.12:
        labels.append("残余误差大")
    if metrics["boundary_excess_mae"] > 0.0015 or metrics["worse_pixel_ratio_boundary"] > 0.40:
        labels.append("边界融合差")
    if metrics["mean_abs_model_anchor_mask"] > 0.06 or metrics["p95_abs_model_anchor_mask"] > 0.16:
        labels.append("模型相对anchor偏移大")
    if metrics["largest_component_ratio"] > 0.75 and metrics["largest_component_area"] > 12000:
        labels.append("大连通洞主导")
    if metrics["touches_border"] and metrics["largest_component_ratio"] > 0.50:
        labels.append("靠边界/视野边缘")
    if metrics["observed_hole_near_repair_ratio"] > 0.70:
        labels.append("邻近真实无效洞")
    if metrics["mask_amp_ratio_to_valid"] < 0.12:
        labels.append("低幅值极端区域")
    if not labels:
        labels.append("可接受但需看细节")

    priority = "high"
    if improve >= 0.08 and model_mae < 0.12:
        priority = "medium"
    if improve >= 0.20 and model_mae < 0.08:
        priority = "low"
    if improve < 0.0 or model_mae >= 0.12:
        priority = "high"

    return labels, priority, model_mae - anchor_mae


def ascii_labels(labels):
    mapping = {
        "模型改坏/anchor更稳": "anchor_better",
        "提升很小": "low_gain",
        "残余误差大": "high_residual",
        "边界融合差": "boundary_artifact",
        "模型相对anchor偏移大": "large_model_anchor_shift",
        "大连通洞主导": "large_component",
        "靠边界/视野边缘": "border_case",
        "邻近真实无效洞": "near_observed_hole",
        "低幅值极端区域": "very_low_amplitude",
        "可接受但需看细节": "check_detail",
    }
    return [mapping.get(label, label) for label in labels]


def analyze_case(selftest_dir, row, summary, args):
    arrays = load_case_arrays(selftest_dir, row, summary, args)
    clean = arrays["clean"]
    anchor = arrays["anchor"]
    model = arrays["model"]
    mask = arrays["mask"]
    condition_mask = arrays["condition_mask"]
    valid = (
        np.isfinite(clean)
        & (clean > float(summary.get("hole_depth_threshold", 0.0)))
        & (clean >= float(summary.get("valid_min_depth", 0.5)))
        & (clean <= float(summary.get("valid_max_depth", 6.0)))
    )

    boundary = dilate_mask(mask, int(args.boundary_width)) & valid & (~mask)
    observed_hole = condition_mask & (~mask)
    observed_near = dilate_mask(observed_hole, int(args.boundary_width)) & mask

    anchor_err = np.abs(anchor - clean)
    model_err = np.abs(model - clean)
    model_anchor_abs = np.abs(model - anchor)
    worse_mask = mask & (model_err > anchor_err + float(args.worse_margin))
    worse_boundary = boundary & (model_err > anchor_err + float(args.worse_margin))

    boundary_model_mae, boundary_count = masked_mae(model, clean, boundary)
    boundary_anchor_mae, _ = masked_mae(anchor, clean, boundary)
    boundary_excess = (
        float(boundary_model_mae - boundary_anchor_mae)
        if boundary_model_mae is not None and boundary_anchor_mae is not None
        else 0.0
    )

    amp = arrays["amplitude_mean"]
    amp_valid = amp[valid & np.isfinite(amp)]
    amp_mask = amp[mask & np.isfinite(amp)]
    amp_valid_median = float(np.median(amp_valid)) if amp_valid.size else 0.0
    amp_mask_median = float(np.median(amp_mask)) if amp_mask.size else 0.0

    cc = connected_component_stats(mask)
    metrics = {
        "case_name": arrays["case_name"],
        "boundary_model_mae": boundary_model_mae,
        "boundary_anchor_mae": boundary_anchor_mae,
        "boundary_excess_mae": boundary_excess,
        "boundary_pixel_count": int(boundary_count),
        "worse_pixel_ratio_mask": float(worse_mask.sum() / max(int(mask.sum()), 1)),
        "worse_pixel_ratio_boundary": float(worse_boundary.sum() / max(int(boundary.sum()), 1)),
        "mean_abs_model_anchor_mask": float(np.mean(model_anchor_abs[mask])) if mask.any() else 0.0,
        "p95_abs_model_anchor_mask": safe_percentile(model_anchor_abs[mask], 95.0),
        "observed_hole_near_repair_ratio": float(observed_near.sum() / max(int(mask.sum()), 1)),
        "mask_amp_median": amp_mask_median,
        "valid_amp_median": amp_valid_median,
        "mask_amp_ratio_to_valid": float(amp_mask_median / max(amp_valid_median, 1e-6)),
    }
    metrics.update(cc)
    labels, priority, excess_mae = classify_case(row, metrics)

    out = dict(row)
    out.update(metrics)
    out["labels"] = labels
    out["priority"] = priority
    out["model_minus_anchor_mask_mae"] = float(excess_mae)
    return out, arrays, valid, boundary, worse_mask


def add_panel(ax, image, title, cmap="viridis", vmin=None, vmax=None, colorbar=False):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def save_case_visualization(out_path, row, arrays, valid, boundary, worse_mask):
    clean = arrays["clean"]
    anchor = arrays["anchor"]
    model = arrays["model"]
    hole_only = arrays["hole_only"]
    corrupted = arrays["corrupted"]
    mask = arrays["mask"]
    condition_mask = arrays["condition_mask"]
    amp = arrays["amplitude_mean"]
    raw9_transform = arrays.get("raw9_transform", "none")

    valid_values = clean[valid & np.isfinite(clean)]
    if valid_values.size:
        dmin, dmax = np.percentile(valid_values, [1.0, 99.0])
    else:
        dmin, dmax = 0.5, 6.0
    err_values = np.concatenate(
        [
            np.abs(anchor[valid] - clean[valid]).ravel(),
            np.abs(model[valid] - clean[valid]).ravel(),
        ]
    )
    err_vmax = max(0.05, safe_percentile(err_values, 99.0, 0.2))
    diff = model - anchor
    diff_abs = np.abs(diff)
    diff_lim = max(0.05, safe_percentile(np.abs(diff[valid]), 99.0, 0.1))
    amp_log = np.log1p(np.clip(amp, 0.0, None))

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    title = (
        f"{row['case_name']} | anchor={row['anchor_mask_mae']:.4f} "
        f"model={row['model_mask_mae']:.4f} improve={row['mask_improve_vs_anchor']:.1%} | "
        f"{', '.join(ascii_labels(row['labels']))}"
    )
    fig.suptitle(title, fontsize=12)

    add_panel(axes[0, 0], clean, "GT / pseudo-GT depth", vmin=dmin, vmax=dmax)
    add_panel(axes[0, 1], corrupted, "Corrupted input", vmin=dmin, vmax=dmax)
    add_panel(axes[0, 2], anchor, "Anchor (NS)", vmin=dmin, vmax=dmax)
    add_panel(axes[0, 3], model, "Model restored", vmin=dmin, vmax=dmax)

    add_panel(axes[1, 0], hole_only, "Hole-only final", vmin=dmin, vmax=dmax)
    add_panel(axes[1, 1], mask.astype(np.float32), "Repair mask", cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*mask.shape, 3), dtype=np.float32)
    overlay[..., 0] = condition_mask.astype(np.float32)
    overlay[..., 1] = mask.astype(np.float32)
    overlay[..., 2] = boundary.astype(np.float32)
    add_panel(axes[1, 2], overlay, "Condition/red + repair/green + boundary/blue")
    add_panel(axes[1, 3], amp_log, f"log amplitude mean ({raw9_transform})", cmap="magma")

    add_panel(axes[2, 0], np.abs(anchor - clean), "|anchor - GT|", cmap="hot", vmin=0, vmax=err_vmax)
    add_panel(axes[2, 1], np.abs(model - clean), "|model - GT|", cmap="hot", vmin=0, vmax=err_vmax)
    add_panel(axes[2, 2], diff, "model - anchor", cmap="coolwarm", vmin=-diff_lim, vmax=diff_lim)
    worse_image = np.zeros((*mask.shape, 3), dtype=np.float32)
    worse_image[..., 0] = worse_mask.astype(np.float32)
    worse_image[..., 1] = (mask & (~worse_mask)).astype(np.float32) * 0.6
    add_panel(axes[2, 3], worse_image, "red: model worse than anchor in mask")

    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    keys = [
        "case_name",
        "priority",
        "labels",
        "anchor_mask_mae",
        "model_mask_mae",
        "model_minus_anchor_mask_mae",
        "mask_improve_vs_anchor",
        "model_global_mae",
        "mask_ratio_actual",
        "condition_hole_ratio",
        "boundary_anchor_mae",
        "boundary_model_mae",
        "boundary_excess_mae",
        "worse_pixel_ratio_mask",
        "worse_pixel_ratio_boundary",
        "mean_abs_model_anchor_mask",
        "p95_abs_model_anchor_mask",
        "component_count",
        "largest_component_area",
        "largest_component_ratio",
        "touches_border",
        "observed_hole_near_repair_ratio",
        "mask_amp_ratio_to_valid",
        "raw_path",
        "depth_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            record = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, list):
                    value = ";".join(str(item) for item in value)
                record[key] = value
            writer.writerow(record)


def format_float(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_report(path, summary, rows, relative_worst, absolute_worst):
    aggregate = summary["aggregate"]
    label_counts = {}
    for row in rows:
        for label in row["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    def table(items):
        lines = [
            "| case | labels | anchor | model | improve | boundary_excess | worse_pixels | amp_ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in items:
            lines.append(
                "| {case} | {labels} | {anchor} | {model} | {improve:.1%} | {boundary} | {worse:.1%} | {amp:.2f} |".format(
                    case=row["case_name"],
                    labels=", ".join(row["labels"]),
                    anchor=format_float(row.get("anchor_mask_mae")),
                    model=format_float(row.get("model_mask_mae")),
                    improve=float(row.get("mask_improve_vs_anchor", 0.0)),
                    boundary=format_float(row.get("boundary_excess_mae")),
                    worse=float(row.get("worse_pixel_ratio_mask", 0.0)),
                    amp=float(row.get("mask_amp_ratio_to_valid", 0.0)),
                )
            )
        return "\n".join(lines)

    lines = [
        "# Real Val Failure Analysis",
        "",
        "## Aggregate",
        "",
        f"- num_cases: {aggregate['num_cases']}",
        f"- anchor_mask_mae: {aggregate['anchor_mask_mae']:.6f}",
        f"- model_mask_mae: {aggregate['model_mask_mae']:.6f}",
        f"- mask_improve_vs_anchor: {aggregate['mask_improve_vs_anchor']:.2%}",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {label}: {count}")
    lines.extend(
        [
            "",
            "## Worst By Relative Improvement",
            "",
            table(relative_worst),
            "",
            "## Worst By Absolute Model MAE",
            "",
            table(absolute_worst),
            "",
            "## Reading Notes",
            "",
            "- `模型改坏/anchor更稳`: model mask MAE is worse than the NS anchor.",
            "- `提升很小`: model improves less than 8% over anchor.",
            "- `残余误差大`: model mask MAE remains at or above 12 cm.",
            "- `边界融合差`: boundary ring got worse or many boundary pixels are worse than anchor.",
            "- `模型相对anchor偏移大`: model changed anchor strongly inside the repair mask.",
            "- `邻近真实无效洞`: repair mask sits near already-invalid depth pixels, so pseudo-GT is harder.",
            "- `低幅值极端区域`: masked amplitude is much lower than the valid-scene median.",
        ]
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    selftest_dir = args.selftest_dir
    output_dir = args.output_dir or os.path.join(selftest_dir, "analysis_worst_cases")
    vis_dir = os.path.join(output_dir, "visualizations")
    ensure_dir(output_dir)
    ensure_dir(vis_dir)

    summary_path = os.path.join(selftest_dir, "summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)

    analyzed = []
    arrays_by_case = {}
    valid_by_case = {}
    boundary_by_case = {}
    worse_by_case = {}
    for row in summary["per_sample"]:
        result, arrays, valid, boundary, worse_mask = analyze_case(selftest_dir, row, summary, args)
        analyzed.append(result)
        arrays_by_case[result["case_name"]] = arrays
        valid_by_case[result["case_name"]] = valid
        boundary_by_case[result["case_name"]] = boundary
        worse_by_case[result["case_name"]] = worse_mask

    relative_worst = sorted(analyzed, key=lambda row: row.get("mask_improve_vs_anchor", 0.0))[: args.top_k]
    absolute_worst = sorted(analyzed, key=lambda row: row.get("model_mask_mae", 0.0), reverse=True)[: args.top_k]
    excess_worst = sorted(analyzed, key=lambda row: row.get("model_minus_anchor_mask_mae", 0.0), reverse=True)[: args.top_k]

    selected = []
    seen = set()
    for group_name, rows in [
        ("relative", relative_worst),
        ("absolute", absolute_worst),
        ("excess", excess_worst),
    ]:
        for rank, row in enumerate(rows):
            key = row["case_name"]
            if key in seen:
                continue
            seen.add(key)
            selected.append((group_name, rank, row))

    for group_name, rank, row in selected:
        case_name = row["case_name"]
        out_png = os.path.join(vis_dir, f"{group_name}_{rank:02d}_{case_name}.png")
        save_case_visualization(
            out_png,
            row,
            arrays_by_case[case_name],
            valid_by_case[case_name],
            boundary_by_case[case_name],
            worse_by_case[case_name],
        )
        row["visualization"] = out_png

    with open(os.path.join(output_dir, "failure_cases.json"), "w") as f:
        json.dump(
            {
                "selftest_dir": selftest_dir,
                "aggregate": summary["aggregate"],
                "relative_worst": relative_worst,
                "absolute_worst": absolute_worst,
                "excess_worst": excess_worst,
                "all_cases": analyzed,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    write_csv(os.path.join(output_dir, "failure_cases.csv"), analyzed)
    write_report(os.path.join(output_dir, "ERROR_REPORT.md"), summary, analyzed, relative_worst, absolute_worst)

    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    print(f"Analyzed cases: {len(analyzed)}")
    print("Worst relative:")
    for row in relative_worst[: min(8, len(relative_worst))]:
        print(
            f"  {row['case_name']} model={row['model_mask_mae']:.4f} "
            f"anchor={row['anchor_mask_mae']:.4f} improve={row['mask_improve_vs_anchor']:.1%} "
            f"labels={','.join(row['labels'])}"
        )
    print("Worst absolute:")
    for row in absolute_worst[: min(8, len(absolute_worst))]:
        print(
            f"  {row['case_name']} model={row['model_mask_mae']:.4f} "
            f"anchor={row['anchor_mask_mae']:.4f} improve={row['mask_improve_vs_anchor']:.1%} "
            f"labels={','.join(row['labels'])}"
        )
    print(f"Saved failure analysis to {output_dir}")


if __name__ == "__main__":
    main()
