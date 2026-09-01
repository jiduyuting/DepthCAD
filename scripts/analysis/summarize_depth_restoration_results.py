import argparse
import csv
import json
import os


DEFAULT_MAIN_EVAL = "output/depth_restoration_unet_noisy_ns_n1000/eval_seed123"
DEFAULT_TWO_STAGE_EVAL = "output/depth_completion_unet_depth_n1000_hole_binary/eval_seed123_ranked"
DEFAULT_PILOT_EVAL = "output/depth_restoration_unet_noisy_ns_n1000_pilot/eval_seed123"
DEFAULT_REGULARIZED_EVAL = "output/depth_restoration_unet_noisy_ns_n1000_valid2_anchor02/eval_seed123"
DEFAULT_GATED_EVAL = "output/depth_restoration_unet_noisy_ns_n1000_gated/eval_seed123"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize single depth restoration results and ablations."
    )
    parser.add_argument("--main_eval", type=str, default=DEFAULT_MAIN_EVAL)
    parser.add_argument("--two_stage_eval", type=str, default=DEFAULT_TWO_STAGE_EVAL)
    parser.add_argument("--pilot_eval", type=str, default=DEFAULT_PILOT_EVAL)
    parser.add_argument("--regularized_eval", type=str, default=DEFAULT_REGULARIZED_EVAL)
    parser.add_argument("--gated_eval", type=str, default=DEFAULT_GATED_EVAL)
    parser.add_argument("--output_dir", type=str, default="output/depth_restoration_summary_final")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_eval(eval_dir):
    summary = load_json(os.path.join(eval_dir, "summary.json"))
    per_sample_path = os.path.join(eval_dir, "per_sample_results.json")
    per_sample = load_json(per_sample_path) if os.path.exists(per_sample_path) else []
    return summary, per_sample


def metric_row(name, aggregate, prefix, learned_models, uses_depthcad, note=""):
    return {
        "method": name,
        "learned_models": learned_models,
        "uses_depthcad": uses_depthcad,
        "global_mae": aggregate[f"{prefix}_global_mae"],
        "hole_mae": aggregate[f"{prefix}_hole_mae"],
        "valid_mae": aggregate[f"{prefix}_valid_mae"],
        "note": note,
    }


def improvement_pct(old, new):
    if old == 0:
        return None
    return (old - new) / old * 100.0


def delta_stats(per_sample, model_prefix="model", baseline_prefix="anchor", region="hole"):
    deltas = []
    for row in per_sample:
        model = row.get(f"{model_prefix}_{region}_mae")
        baseline = row.get(f"{baseline_prefix}_{region}_mae")
        if model is None or baseline is None:
            continue
        deltas.append(float(model) - float(baseline))
    deltas.sort()

    def percentile(pct):
        if not deltas:
            return None
        index = max(0, min(len(deltas) - 1, int(round((pct / 100.0) * (len(deltas) - 1)))))
        return deltas[index]

    return {
        "better_samples": sum(delta < 0 for delta in deltas),
        "worse_samples": sum(delta > 0 for delta in deltas),
        "best_delta": min(deltas) if deltas else None,
        "median_delta": percentile(50),
        "p95_delta": percentile(95),
        "worst_delta": max(deltas) if deltas else None,
    }


def format_float(value, digits=6):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def format_pct(value):
    if value is None:
        return ""
    return f"{float(value):.1f}%"


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_main_markdown(path, rows):
    lines = [
        "| Method | Learned Models | Uses DepthCAD | Global MAE | Hole MAE | Valid MAE |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["method"],
                str(row["learned_models"]),
                row["uses_depthcad"],
                format_float(row["global_mae"]),
                format_float(row["hole_mae"]),
                format_float(row["valid_mae"]),
            ])
            + " |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_ablation_markdown(path, rows):
    lines = [
        "| Variant | Global MAE | Hole MAE | Valid MAE | Hole Improve vs Anchor/Base | Better/Worse | Worst Delta | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["variant"],
                format_float(row["global_mae"]),
                format_float(row["hole_mae"]),
                format_float(row["valid_mae"]),
                format_pct(row["hole_improvement_pct"]),
                f"{row['better_samples']}/{row['worse_samples']}",
                format_float(row["worst_delta"]),
                row["note"],
            ])
            + " |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ranked_cases(per_sample, baseline_prefix="anchor", region="hole", count=6):
    scored = []
    for row in per_sample:
        model = row.get(f"model_{region}_mae")
        baseline = row.get(f"{baseline_prefix}_{region}_mae")
        if model is None or baseline is None:
            continue
        delta = float(model) - float(baseline)
        scored.append((delta, row))
    scored.sort(key=lambda item: item[0])

    rows = []
    for label, items in [
        ("best", scored[:count]),
        ("worst", list(reversed(scored[-count:]))),
    ]:
        for rank, (delta, row) in enumerate(items, start=1):
            rows.append({
                "type": label,
                "rank": rank,
                "sample_name": row["sample_name"],
                "baseline_hole_mae": row.get(f"{baseline_prefix}_hole_mae"),
                "model_hole_mae": row.get("model_hole_mae"),
                "hole_delta": delta,
                "baseline_global_mae": row.get(f"{baseline_prefix}_global_mae"),
                "model_global_mae": row.get("model_global_mae"),
                "global_delta": (
                    float(row["model_global_mae"]) - float(row[f"{baseline_prefix}_global_mae"])
                    if row.get("model_global_mae") is not None and row.get(f"{baseline_prefix}_global_mae") is not None
                    else None
                ),
                "path": row["path"],
            })
    return rows


def write_ranked_cases_markdown(path, rows, baseline_name):
    lines = [
        f"Ranked by model hole MAE minus {baseline_name} hole MAE.",
        "",
        "| Type | Rank | Sample | Baseline Hole | Model Hole | Hole Delta | Baseline Global | Model Global | Global Delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["type"],
                str(row["rank"]),
                row["sample_name"],
                format_float(row["baseline_hole_mae"]),
                format_float(row["model_hole_mae"]),
                format_float(row["hole_delta"]),
                format_float(row["baseline_global_mae"]),
                format_float(row["model_global_mae"]),
                format_float(row["global_delta"]),
            ])
            + " |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_ablation_row(name, summary, per_sample, baseline_prefix, note):
    aggregate = summary["aggregate"]
    model_hole = aggregate["model_hole_mae"]
    baseline_hole = aggregate[f"{baseline_prefix}_hole_mae"]
    stats = delta_stats(per_sample, baseline_prefix=baseline_prefix, region="hole")
    return {
        "variant": name,
        "global_mae": aggregate["model_global_mae"],
        "hole_mae": model_hole,
        "valid_mae": aggregate["model_valid_mae"],
        "hole_improvement_pct": improvement_pct(baseline_hole, model_hole),
        "better_samples": stats["better_samples"],
        "worse_samples": stats["worse_samples"],
        "worst_delta": stats["worst_delta"],
        "p95_delta": stats["p95_delta"],
        "median_delta": stats["median_delta"],
        "best_delta": stats["best_delta"],
        "baseline_prefix": baseline_prefix,
        "note": note,
    }


def main():
    args = parse_args()
    main_summary, main_per_sample = load_eval(args.main_eval)
    two_stage_summary, two_stage_per_sample = load_eval(args.two_stage_eval)
    pilot_summary, pilot_per_sample = load_eval(args.pilot_eval)
    regularized_summary, regularized_per_sample = load_eval(args.regularized_eval)
    gated_summary, gated_per_sample = load_eval(args.gated_eval)

    main_aggregate = main_summary["aggregate"]
    two_stage_aggregate = two_stage_summary["aggregate"]

    main_results = [
        metric_row("Noisy", main_aggregate, "noisy", 0, "No", "degraded input"),
        metric_row("NS Anchor", main_aggregate, "anchor", 0, "No", "deterministic depth inpaint"),
        metric_row("DepthCAD/Plane Base", main_aggregate, "base", 1, "Yes", "DepthCAD plus plane fill"),
        metric_row("Two-stage Completion", two_stage_aggregate, "model", 2, "Yes", "DepthCAD/plane plus learned completion"),
        metric_row("Ours Single Restoration", main_aggregate, "model", 1, "No", "final model"),
    ]

    ours = main_results[-1]
    for row in main_results[:-1]:
        row["ours_global_improvement_pct"] = improvement_pct(row["global_mae"], ours["global_mae"])
        row["ours_hole_improvement_pct"] = improvement_pct(row["hole_mae"], ours["hole_mae"])
        row["ours_valid_improvement_pct"] = improvement_pct(row["valid_mae"], ours["valid_mae"])
    ours["ours_global_improvement_pct"] = 0.0
    ours["ours_hole_improvement_pct"] = 0.0
    ours["ours_valid_improvement_pct"] = 0.0

    ablations = [
        build_ablation_row(
            "30-epoch pilot",
            pilot_summary,
            pilot_per_sample,
            "anchor",
            "undertrained single restoration",
        ),
        build_ablation_row(
            "Ours main",
            main_summary,
            main_per_sample,
            "anchor",
            "best single-model result",
        ),
        build_ablation_row(
            "valid2_anchor02",
            regularized_summary,
            regularized_per_sample,
            "anchor",
            "over-constrained valid/anchor regularization",
        ),
        build_ablation_row(
            "gated residual",
            gated_summary,
            gated_per_sample,
            "anchor",
            "hole-focused gated architecture variant",
        ),
        build_ablation_row(
            "Two-stage completion",
            two_stage_summary,
            two_stage_per_sample,
            "base",
            "strong two-stage baseline",
        ),
    ]
    cases = ranked_cases(main_per_sample, baseline_prefix="anchor", region="hole", count=6)

    output = {
        "main_eval": args.main_eval,
        "two_stage_eval": args.two_stage_eval,
        "pilot_eval": args.pilot_eval,
        "regularized_eval": args.regularized_eval,
        "gated_eval": args.gated_eval,
        "main_results": main_results,
        "ablations": ablations,
        "ranked_cases_vs_anchor": cases,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)

    main_fields = [
        "method",
        "learned_models",
        "uses_depthcad",
        "global_mae",
        "hole_mae",
        "valid_mae",
        "ours_global_improvement_pct",
        "ours_hole_improvement_pct",
        "ours_valid_improvement_pct",
        "note",
    ]
    ablation_fields = [
        "variant",
        "global_mae",
        "hole_mae",
        "valid_mae",
        "hole_improvement_pct",
        "better_samples",
        "worse_samples",
        "worst_delta",
        "p95_delta",
        "median_delta",
        "best_delta",
        "baseline_prefix",
        "note",
    ]
    write_csv(os.path.join(args.output_dir, "main_results.csv"), main_results, main_fields)
    write_csv(os.path.join(args.output_dir, "ablation_results.csv"), ablations, ablation_fields)
    write_csv(
        os.path.join(args.output_dir, "ranked_cases_vs_anchor.csv"),
        cases,
        [
            "type",
            "rank",
            "sample_name",
            "baseline_hole_mae",
            "model_hole_mae",
            "hole_delta",
            "baseline_global_mae",
            "model_global_mae",
            "global_delta",
            "path",
        ],
    )
    write_main_markdown(os.path.join(args.output_dir, "main_results.md"), main_results)
    write_ablation_markdown(os.path.join(args.output_dir, "ablation_results.md"), ablations)
    write_ranked_cases_markdown(
        os.path.join(args.output_dir, "ranked_cases_vs_anchor.md"),
        cases,
        "anchor",
    )

    print(f"Saved summaries to {args.output_dir}")
    print()
    with open(os.path.join(args.output_dir, "main_results.md"), "r") as f:
        print(f.read())
    with open(os.path.join(args.output_dir, "ablation_results.md"), "r") as f:
        print(f.read())
    with open(os.path.join(args.output_dir, "ranked_cases_vs_anchor.md"), "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
