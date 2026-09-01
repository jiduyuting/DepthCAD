import argparse
import csv
import json
import math
import os
from glob import glob

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize --save_sd_diagnostics outputs.")
    parser.add_argument("--diagnostics_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_rows(diagnostics_dir):
    metric_paths = sorted(glob(os.path.join(diagnostics_dir, "**", "*_metrics.json"), recursive=True))
    rows = []
    for path in metric_paths:
        with open(path, "r") as f:
            metrics = json.load(f)
        rel = os.path.relpath(path, diagnostics_dir)
        sample_name = rel.replace("_metrics.json", "").replace(os.sep, "/")
        row = {"sample_name": sample_name, "metrics_path": path}
        row.update(metrics)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json files found under {diagnostics_dir}")
    return rows


def mean_metric(rows, key):
    values = [safe_float(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def aggregate(rows):
    keys = sorted({key for row in rows for key in row if key not in ["sample_name", "metrics_path"]})
    return {key: mean_metric(rows, key) for key in keys}


def top_rows(rows, key, top_k, reverse=True):
    scored = []
    for row in rows:
        value = safe_float(row.get(key))
        if math.isfinite(value):
            scored.append((value, row))
    scored.sort(key=lambda item: item[0], reverse=reverse)
    return [{"sample_name": row["sample_name"], "value": value, "metrics_path": row["metrics_path"]} for value, row in scored[:top_k]]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def fmt(value):
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def write_markdown(path, diagnostics_dir, rows, summary, top_tables):
    lines = [
        "# SD Diagnostics Summary",
        "",
        f"Diagnostics dir: `{diagnostics_dir}`",
        f"Samples: `{len(rows)}`",
        "",
        "## Mean Depth Metrics",
        "",
        "| Method | Global | Hole | Valid |",
        "|---|---:|---:|---:|",
    ]
    for method in ["noisy", "depthcad", "sdinpaint", "full", "depthfill"]:
        lines.append(
            f"| {method} | {fmt(summary.get(method + '_depth_l1_global'))} | "
            f"{fmt(summary.get(method + '_depth_l1_hole'))} | "
            f"{fmt(summary.get(method + '_depth_l1_valid'))} |"
        )

    lines.extend([
        "",
        "## Mean IQ Physics Metrics in Holes",
        "",
        "| Method | IQ L1 | Amp Pair0 | Amp Pair1 | Amp Pair2 | Phase Pair0 | Phase Pair1 | Phase Pair2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in ["noisy", "depthcad", "sdinpaint", "full"]:
        lines.append(
            f"| {method} | "
            f"{fmt(summary.get(method + '_iq_l1_hole_mean'))} | "
            f"{fmt(summary.get(method + '_amp_l1_hole_pair0'))} | "
            f"{fmt(summary.get(method + '_amp_l1_hole_pair1'))} | "
            f"{fmt(summary.get(method + '_amp_l1_hole_pair2'))} | "
            f"{fmt(summary.get(method + '_phase_l1_hole_pair0'))} | "
            f"{fmt(summary.get(method + '_phase_l1_hole_pair1'))} | "
            f"{fmt(summary.get(method + '_phase_l1_hole_pair2'))} |"
        )

    for title, rows_for_key in top_tables.items():
        lines.extend([
            "",
            f"## {title}",
            "",
            "| Sample | Value | Metrics JSON |",
            "|---|---:|---|",
        ])
        for row in rows_for_key:
            lines.append(f"| {row['sample_name']} | {fmt(row['value'])} | `{row['metrics_path']}` |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.join(args.diagnostics_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows(args.diagnostics_dir)
    summary = aggregate(rows)
    top_tables = {
        "Worst SDInpaint Depth Hole MAE": top_rows(rows, "sdinpaint_depth_l1_hole", args.top_k),
        "Worst Full Depth Hole MAE": top_rows(rows, "full_depth_l1_hole", args.top_k),
        "Worst SDInpaint IQ L1 in Holes": top_rows(rows, "sdinpaint_iq_l1_hole_mean", args.top_k),
        "Worst SDInpaint Phase Pair0 Error": top_rows(rows, "sdinpaint_phase_l1_hole_pair0", args.top_k),
        "Worst SDInpaint Phase Pair1 Error": top_rows(rows, "sdinpaint_phase_l1_hole_pair1", args.top_k),
        "Worst SDInpaint Phase Pair2 Error": top_rows(rows, "sdinpaint_phase_l1_hole_pair2", args.top_k),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"aggregate": summary, "top": top_tables}, f, indent=2, sort_keys=True)
    write_csv(
        os.path.join(out_dir, "per_sample_metrics.csv"),
        rows,
        sorted({key for row in rows for key in row}),
    )
    write_markdown(os.path.join(out_dir, "summary.md"), args.diagnostics_dir, rows, summary, top_tables)

    print(f"Saved SD diagnostics summary to {out_dir}")
    with open(os.path.join(out_dir, "summary.md"), "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
