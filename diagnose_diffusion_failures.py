import argparse
import csv
import json
import math
import os
from glob import glob

import numpy as np


METHODS = ["noisy", "depthcad", "sdinpaint", "full", "depthfill"]
REGIONS = ["global", "holes", "valid"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose why SD/IQ diffusion inpainting underperforms depth-domain restoration."
    )
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Directory containing apply_kinect_holes_and_eval.py result_*.json files.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def load_results(eval_dir):
    paths = sorted(glob(os.path.join(eval_dir, "result_*.json")))
    rows = []
    for path in paths:
        with open(path, "r") as f:
            row = json.load(f)
        row["_path"] = path
        parts = row.get("sample_name", "").split("/")
        row["_scene"] = parts[0] if parts else "unknown"
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result_*.json files found in {eval_dir}")
    return rows


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def metric_key(method, region):
    if region == "global":
        return f"mae_{method}"
    if region == "holes":
        return f"mae_{method}_holes"
    if region == "valid":
        return f"mae_{method}_valid"
    raise ValueError(region)


def metric(row, method, region):
    return safe_float(row.get(metric_key(method, region)))


def weighted_mean(rows, method, region):
    total = 0.0
    count = 0
    mae_key = metric_key(method, region)
    if region == "global":
        count_key = f"mae_{method}_eval_pixel_count"
    elif region == "holes":
        count_key = f"mae_{method}_hole_eval_pixel_count"
    else:
        count_key = f"mae_{method}_valid_eval_pixel_count"

    for row in rows:
        mae = safe_float(row.get(mae_key))
        n = int(row.get(count_key, 0) or 0)
        if math.isnan(mae) or n <= 0:
            continue
        total += mae * n
        count += n
    return total / count if count > 0 else math.nan, count


def summarize_methods(rows):
    out = {}
    for method in METHODS:
        out[method] = {}
        for region in REGIONS:
            mean, count = weighted_mean(rows, method, region)
            out[method][region] = {"mae": mean, "count": count}
    return out


def compare(rows, method, baseline, region="holes"):
    deltas = []
    better = 0
    worse = 0
    ties = 0
    for row in rows:
        a = metric(row, method, region)
        b = metric(row, baseline, region)
        if math.isnan(a) or math.isnan(b):
            continue
        delta = a - b
        deltas.append(delta)
        if delta < 0:
            better += 1
        elif delta > 0:
            worse += 1
        else:
            ties += 1
    deltas_sorted = sorted(deltas)

    def percentile(pct):
        if not deltas_sorted:
            return math.nan
        index = int(round((pct / 100.0) * (len(deltas_sorted) - 1)))
        index = min(max(index, 0), len(deltas_sorted) - 1)
        return deltas_sorted[index]

    return {
        "method": method,
        "baseline": baseline,
        "region": region,
        "better": better,
        "worse": worse,
        "ties": ties,
        "mean_delta": float(np.mean(deltas)) if deltas else math.nan,
        "median_delta": percentile(50),
        "p95_delta": percentile(95),
        "worst_delta": max(deltas) if deltas else math.nan,
        "best_delta": min(deltas) if deltas else math.nan,
    }


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return math.nan
    xs = xs[mask]
    ys = ys[mask]
    if xs.std() < 1e-12 or ys.std() < 1e-12:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def correlation_report(rows):
    predictors = {
        "hole_ratio": [safe_float(r.get("hole_ratio")) for r in rows],
        "evaluated_hole_ratio": [safe_float(r.get("evaluated_hole_ratio")) for r in rows],
        "depthcad_hole_mae": [metric(r, "depthcad", "holes") for r in rows],
        "noisy_hole_mae": [metric(r, "noisy", "holes") for r in rows],
    }
    targets = {
        "sd_minus_depthfill_hole": [
            metric(r, "sdinpaint", "holes") - metric(r, "depthfill", "holes")
            for r in rows
        ],
        "full_minus_depthcad_hole": [
            metric(r, "full", "holes") - metric(r, "depthcad", "holes")
            for r in rows
        ],
        "sd_hole_mae": [metric(r, "sdinpaint", "holes") for r in rows],
    }
    out = []
    for pred_name, pred_values in predictors.items():
        for target_name, target_values in targets.items():
            out.append({
                "predictor": pred_name,
                "target": target_name,
                "pearson_r": pearson(pred_values, target_values),
            })
    return out


def group_by_scene(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["_scene"], []).append(row)
    scene_rows = []
    for scene, scene_items in sorted(groups.items()):
        summary = summarize_methods(scene_items)
        scene_rows.append({
            "scene": scene,
            "num_samples": len(scene_items),
            "sd_hole": summary["sdinpaint"]["holes"]["mae"],
            "depthfill_hole": summary["depthfill"]["holes"]["mae"],
            "full_hole": summary["full"]["holes"]["mae"],
            "depthcad_hole": summary["depthcad"]["holes"]["mae"],
            "sd_minus_depthfill_hole": summary["sdinpaint"]["holes"]["mae"] - summary["depthfill"]["holes"]["mae"],
            "full_minus_depthcad_hole": summary["full"]["holes"]["mae"] - summary["depthcad"]["holes"]["mae"],
        })
    return scene_rows


def top_cases(rows, method, baseline, region, top_k):
    scored = []
    for row in rows:
        a = metric(row, method, region)
        b = metric(row, baseline, region)
        if math.isnan(a) or math.isnan(b):
            continue
        scored.append((a - b, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "sample_name": row.get("sample_name", ""),
            "scene": row["_scene"],
            "delta": delta,
            f"{method}_{region}": metric(row, method, region),
            f"{baseline}_{region}": metric(row, baseline, region),
            "hole_ratio": safe_float(row.get("hole_ratio")),
            "json_path": row["_path"],
        }
        for delta, row in scored[:top_k]
    ]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def fmt(value, digits=6):
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(path, report):
    method_summary = report["method_summary"]
    comparisons = report["comparisons"]
    scene_rows = report["scene_rows"]
    top_sd_vs_depthfill = report["top_sd_vs_depthfill"]
    top_full_vs_depthcad = report["top_full_vs_depthcad"]

    lines = [
        "# Diffusion/IQ Inpainting Failure Diagnosis",
        "",
        f"Eval dir: `{report['eval_dir']}`",
        f"Samples: `{report['num_samples']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | Global MAE | Hole MAE | Valid MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        lines.append(
            "| "
            + " | ".join([
                method,
                fmt(method_summary[method]["global"]["mae"]),
                fmt(method_summary[method]["holes"]["mae"]),
                fmt(method_summary[method]["valid"]["mae"]),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Key Comparisons",
        "",
        "| Method | Baseline | Region | Better/Worse | Mean Delta | Median Delta | P95 Delta | Worst Delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            "| "
            + " | ".join([
                row["method"],
                row["baseline"],
                row["region"],
                f"{row['better']}/{row['worse']}",
                fmt(row["mean_delta"]),
                fmt(row["median_delta"]),
                fmt(row["p95_delta"]),
                fmt(row["worst_delta"]),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Scene Breakdown",
        "",
        "| Scene | N | SD Hole | DepthFill Hole | SD-DepthFill | Full Hole | DepthCAD Hole | Full-DepthCAD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in scene_rows:
        lines.append(
            "| "
            + " | ".join([
                row["scene"],
                str(row["num_samples"]),
                fmt(row["sd_hole"]),
                fmt(row["depthfill_hole"]),
                fmt(row["sd_minus_depthfill_hole"]),
                fmt(row["full_hole"]),
                fmt(row["depthcad_hole"]),
                fmt(row["full_minus_depthcad_hole"]),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Worst SD Inpaint vs DepthFill Cases",
        "",
        "| Sample | Delta | SD Hole | DepthFill Hole | Hole Ratio |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in top_sd_vs_depthfill:
        lines.append(
            f"| {row['sample_name']} | {fmt(row['delta'])} | "
            f"{fmt(row['sdinpaint_holes'])} | {fmt(row['depthfill_holes'])} | "
            f"{fmt(row['hole_ratio'])} |"
        )

    lines.extend([
        "",
        "## Worst DepthCAD + SD Inpaint vs DepthCAD Cases",
        "",
        "| Sample | Delta | Full Hole | DepthCAD Hole | Hole Ratio |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in top_full_vs_depthcad:
        lines.append(
            f"| {row['sample_name']} | {fmt(row['delta'])} | "
            f"{fmt(row['full_holes'])} | {fmt(row['depthcad_holes'])} | "
            f"{fmt(row['hole_ratio'])} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The SD/IQ inpainting path is weak for this task when:",
        "",
        "1. It inpaints each IQ channel independently as a grayscale pseudo-RGB image, so the six-channel ToF phase/amplitude relationship is not explicitly preserved.",
        "2. The SD inpainting prior is an RGB natural-image prior, while IQ channels are signed physical measurement fields, not natural images.",
        "3. The generated IQ values are later passed through a nonlinear IQ-to-depth estimator; small channel-inconsistent hallucinations can become large metric depth errors.",
        "4. The inpainting objective does not know the metric depth loss, plane geometry, or boundary continuity in depth space.",
        "5. `DepthCAD + SD Inpaint` can be worse than `DepthCAD only` in holes when SD changes the denoised IQ channels in a way that violates depth geometry.",
        "",
        "This does not mean diffusion is useless. It means the current pseudo-RGB IQ-to-SD interface does not expose the right conditioning or loss for metric depth recovery.",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = load_results(args.eval_dir)
    out_dir = args.output_dir or os.path.join(args.eval_dir, "diffusion_diagnosis")
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "eval_dir": args.eval_dir,
        "num_samples": len(rows),
        "method_summary": summarize_methods(rows),
        "comparisons": [
            compare(rows, "sdinpaint", "depthfill", "holes"),
            compare(rows, "sdinpaint", "noisy", "holes"),
            compare(rows, "sdinpaint", "depthcad", "holes"),
            compare(rows, "full", "depthcad", "holes"),
            compare(rows, "full", "depthfill", "holes"),
        ],
        "correlations": correlation_report(rows),
        "scene_rows": group_by_scene(rows),
        "top_sd_vs_depthfill": top_cases(rows, "sdinpaint", "depthfill", "holes", args.top_k),
        "top_full_vs_depthcad": top_cases(rows, "full", "depthcad", "holes", args.top_k),
    }

    with open(os.path.join(out_dir, "diagnosis.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    write_csv(
        os.path.join(out_dir, "scene_breakdown.csv"),
        report["scene_rows"],
        [
            "scene",
            "num_samples",
            "sd_hole",
            "depthfill_hole",
            "sd_minus_depthfill_hole",
            "full_hole",
            "depthcad_hole",
            "full_minus_depthcad_hole",
        ],
    )
    write_csv(
        os.path.join(out_dir, "worst_sd_vs_depthfill.csv"),
        report["top_sd_vs_depthfill"],
        ["sample_name", "scene", "delta", "sdinpaint_holes", "depthfill_holes", "hole_ratio", "json_path"],
    )
    write_csv(
        os.path.join(out_dir, "worst_full_vs_depthcad.csv"),
        report["top_full_vs_depthcad"],
        ["sample_name", "scene", "delta", "full_holes", "depthcad_holes", "hole_ratio", "json_path"],
    )
    write_csv(
        os.path.join(out_dir, "correlations.csv"),
        report["correlations"],
        ["predictor", "target", "pearson_r"],
    )
    write_markdown(os.path.join(out_dir, "diagnosis.md"), report)

    print(f"Saved diagnosis to {out_dir}")
    with open(os.path.join(out_dir, "diagnosis.md"), "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
