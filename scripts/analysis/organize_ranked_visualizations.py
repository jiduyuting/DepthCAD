import argparse
import csv
import json
import math
import os
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ranked case tables and a Markdown visualization index from eval per-sample results."
    )
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Directory containing per_sample_results.json and optionally visualizations/*.png.")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="Output prefix, e.g. output/summary/ranked_cases_endpoint_w2.")
    parser.add_argument("--baseline", type=str, default="anchor",
                        choices=["anchor", "base", "noisy"])
    parser.add_argument("--region", type=str, default="hole",
                        choices=["global", "hole", "valid"])
    parser.add_argument("--top_k", type=int, default=12,
                        help="Number of best and worst rows to include in summary sections.")
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return math.nan
    value = float(value)
    return value if math.isfinite(value) else math.nan


def load_rows(eval_dir, baseline, region):
    path = os.path.join(eval_dir, "per_sample_results.json")
    with open(path, "r") as f:
        rows = json.load(f)

    out = []
    for row in rows:
        model = safe_float(row.get(f"model_{region}_mae"))
        base = safe_float(row.get(f"{baseline}_{region}_mae"))
        if not math.isfinite(model) or not math.isfinite(base):
            continue
        item = dict(row)
        item["delta"] = model - base
        item["improvement"] = base - model
        item["improvement_pct"] = (base - model) / max(base, 1e-8) * 100.0
        out.append(item)
    return out


def collect_visualizations(eval_dir):
    vis_dir = os.path.join(eval_dir, "visualizations")
    mapping = {}
    for path in sorted(glob(os.path.join(vis_dir, "*.png"))):
        name = os.path.basename(path)
        stem = name[:-4]
        parts = stem.split("_")
        if len(parts) < 7:
            continue
        sample_key = "_".join(parts[5:])
        mapping[sample_key] = path
    return mapping


def sample_key(sample_name):
    return sample_name.replace("/", "_")


def write_csv(path, rows, baseline, region):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "rank_by_delta",
        "sample_name",
        "delta",
        "improvement",
        "improvement_pct",
        f"model_{region}_mae",
        f"{baseline}_{region}_mae",
        "model_global_mae",
        "model_hole_mae",
        "model_valid_mae",
        f"{baseline}_global_mae",
        f"{baseline}_hole_mae",
        f"{baseline}_valid_mae",
        "path",
        "visualization",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({key: row.get(key, "") for key in fieldnames} | {"rank_by_delta": index})


def fmt(value):
    value = safe_float(value)
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


def md_table(rows, baseline, region):
    lines = [
        "| Rank | Sample | Model | Baseline | Delta | Improvement | Improvement % | Visualization |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(rows, start=1):
        vis = row.get("visualization", "")
        vis_link = f"[png]({vis})" if vis else ""
        lines.append(
            f"| {i} | `{row['sample_name']}` | "
            f"{fmt(row.get(f'model_{region}_mae'))} | "
            f"{fmt(row.get(f'{baseline}_{region}_mae'))} | "
            f"{fmt(row.get('delta'))} | "
            f"{fmt(row.get('improvement'))} | "
            f"{fmt(row.get('improvement_pct'))} | "
            f"{vis_link} |"
        )
    return "\n".join(lines)


def md_images(rows, title):
    lines = [f"## {title}", ""]
    for row in rows:
        vis = row.get("visualization", "")
        if not vis:
            continue
        lines.extend([
            f"### {row['sample_name']}",
            "",
            f"![{row['sample_name']}]({vis})",
            "",
        ])
    return "\n".join(lines)


def write_markdown(path, rows_sorted, best_rows, worst_rows, baseline, region, eval_dir):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    improved = sum(1 for row in rows_sorted if row["delta"] < 0)
    worsened = sum(1 for row in rows_sorted if row["delta"] > 0)
    tied = len(rows_sorted) - improved - worsened
    mean_delta = sum(row["delta"] for row in rows_sorted) / max(len(rows_sorted), 1)
    mean_improvement = -mean_delta

    lines = [
        "# Ranked Visualizations",
        "",
        f"Eval dir: `{eval_dir}`",
        "",
        f"Ranking: `model_{region}_mae - {baseline}_{region}_mae`",
        "",
        "Negative delta means the model is better than the baseline.",
        "",
        "## Summary",
        "",
        f"- Samples: `{len(rows_sorted)}`",
        f"- Improved / worsened / tied: `{improved} / {worsened} / {tied}`",
        f"- Mean delta: `{mean_delta:.6f}`",
        f"- Mean improvement: `{mean_improvement:.6f}`",
        "",
        "## Best Cases",
        "",
        md_table(best_rows, baseline, region),
        "",
        "## Worst Cases",
        "",
        md_table(worst_rows, baseline, region),
        "",
        md_images(best_rows, "Best Case Images"),
        "",
        md_images(worst_rows, "Worst Case Images"),
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    rows = load_rows(args.eval_dir, args.baseline, args.region)
    vis = collect_visualizations(args.eval_dir)
    for row in rows:
        key = sample_key(row["sample_name"])
        row["visualization"] = vis.get(key, "")

    rows_sorted = sorted(rows, key=lambda row: row["delta"])
    best_rows = rows_sorted[:args.top_k]
    worst_rows = list(reversed(rows_sorted[-args.top_k:]))

    write_csv(args.output_prefix + ".csv", rows_sorted, args.baseline, args.region)
    write_markdown(args.output_prefix + ".md", rows_sorted, best_rows, worst_rows, args.baseline, args.region, args.eval_dir)

    print(json.dumps({
        "eval_dir": args.eval_dir,
        "num_rows": len(rows_sorted),
        "csv": args.output_prefix + ".csv",
        "markdown": args.output_prefix + ".md",
        "best": [row["sample_name"] for row in best_rows[:6]],
        "worst": [row["sample_name"] for row in worst_rows[:6]],
    }, indent=2))


if __name__ == "__main__":
    main()
