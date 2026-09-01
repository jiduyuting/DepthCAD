import argparse
import csv
import json
import os


DEFAULT_RUNS = [
    ("n100", "output/depth_completion_unet_depth_amp_n100_validw1/eval_seed123_fixed_holemask"),
    ("n500", "output/depth_completion_unet_depth_amp_n500_hole_binary/eval_seed123"),
    ("n1000", "output/depth_completion_unet_depth_amp_n1000_hole_binary/eval_seed123"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize depth completion eval summaries and per-sample regression statistics."
    )
    parser.add_argument("--run", action="append", default=[],
                        help="Run spec as name:path_to_eval_dir. Can be repeated.")
    parser.add_argument("--output_dir", type=str, default="output/depth_completion_summary")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_run(name, eval_dir):
    summary = load_json(os.path.join(eval_dir, "summary.json"))
    per_sample = load_json(os.path.join(eval_dir, "per_sample_results.json"))
    aggregate = summary["aggregate"]

    deltas = []
    for row in per_sample:
        base = row.get("base_hole_mae")
        model = row.get("model_hole_mae")
        if base is None or model is None:
            continue
        deltas.append(float(model) - float(base))
    deltas.sort()

    def percentile(values, pct):
        if not values:
            return None
        index = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
        return values[index]

    base_global = aggregate["base_global_mae"]
    base_hole = aggregate["base_hole_mae"]
    base_valid = aggregate["base_valid_mae"]
    model_global = aggregate["model_global_mae"]
    model_hole = aggregate["model_hole_mae"]
    model_valid = aggregate["model_valid_mae"]

    return {
        "name": name,
        "eval_dir": eval_dir,
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "input_mode": summary.get("input_mode"),
        "residual_apply_mask": summary.get("residual_apply_mask"),
        "residual_gate": summary.get("residual_gate"),
        "num_samples": summary.get("num_samples"),
        "base_global_mae": base_global,
        "model_global_mae": model_global,
        "global_improvement_pct": (base_global - model_global) / base_global * 100.0,
        "base_hole_mae": base_hole,
        "model_hole_mae": model_hole,
        "hole_improvement_pct": (base_hole - model_hole) / base_hole * 100.0,
        "base_valid_mae": base_valid,
        "model_valid_mae": model_valid,
        "valid_delta": model_valid - base_valid,
        "worse_samples": sum(delta > 0 for delta in deltas),
        "better_samples": sum(delta < 0 for delta in deltas),
        "worst_hole_delta": max(deltas) if deltas else None,
        "p95_hole_delta": percentile(deltas, 95),
        "median_hole_delta": percentile(deltas, 50),
        "best_hole_delta": min(deltas) if deltas else None,
    }


def format_float(value, digits=6):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def write_csv(path, rows):
    fieldnames = [
        "name",
        "num_samples",
        "checkpoint_epoch",
        "input_mode",
        "residual_apply_mask",
        "residual_gate",
        "base_global_mae",
        "model_global_mae",
        "global_improvement_pct",
        "base_hole_mae",
        "model_hole_mae",
        "hole_improvement_pct",
        "base_valid_mae",
        "model_valid_mae",
        "valid_delta",
        "better_samples",
        "worse_samples",
        "worst_hole_delta",
        "p95_hole_delta",
        "median_hole_delta",
        "best_hole_delta",
        "eval_dir",
        "checkpoint",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path, rows):
    headers = [
        "Run",
        "Global",
        "Hole",
        "Hole Improve",
        "Valid",
        "Better/Worse",
        "Worst Delta",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["name"]),
                format_float(row["model_global_mae"]),
                format_float(row["model_hole_mae"]),
                f"{row['hole_improvement_pct']:.1f}%",
                format_float(row["model_valid_mae"]),
                f"{row['better_samples']}/{row['worse_samples']}",
                format_float(row["worst_hole_delta"]),
            ])
            + " |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    run_specs = args.run or [f"{name}:{path}" for name, path in DEFAULT_RUNS]
    rows = []
    for spec in run_specs:
        if ":" not in spec:
            raise ValueError(f"Run spec must be name:path, got {spec}")
        name, eval_dir = spec.split(":", 1)
        rows.append(summarize_run(name, eval_dir))

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "summary.json")
    csv_path = os.path.join(args.output_dir, "summary.csv")
    md_path = os.path.join(args.output_dir, "summary.md")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    print()
    with open(md_path, "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
