import argparse
import csv
import json
import os


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def metric_block(model_dir):
    direct_summary_path = os.path.join(model_dir, "summary.json")
    nested_summary_path = os.path.join(model_dir, "real_val_selftest_auto", "summary.json")
    if os.path.exists(nested_summary_path):
        real_summary_path = nested_summary_path
        real_dir = os.path.join(model_dir, "real_val_selftest_auto")
    elif os.path.exists(direct_summary_path):
        direct = load_json(direct_summary_path)
        if "aggregate" in direct and "per_sample" in direct:
            real_summary_path = direct_summary_path
            real_dir = model_dir
        else:
            real_summary_path = nested_summary_path
            real_dir = os.path.join(model_dir, "real_val_selftest_auto")
    else:
        real_summary_path = nested_summary_path
        real_dir = os.path.join(model_dir, "real_val_selftest_auto")

    pbrt_summary_path = os.path.join(model_dir, "eval_pbrt_val97_endpoint", "summary.json")
    failure_csv_path = os.path.join(real_dir, "analysis_worst_cases", "failure_cases.csv")

    out = {"model_dir": model_dir}
    if os.path.exists(real_summary_path):
        summary = load_json(real_summary_path)
        aggregate = summary["aggregate"]
        out.update(
            {
                "real_model_mask_mae": aggregate.get("model_mask_mae"),
                "real_anchor_mask_mae": aggregate.get("anchor_mask_mae"),
                "real_mask_improve": aggregate.get("mask_improve_vs_anchor"),
                "real_model_global_mae": aggregate.get("model_global_mae"),
                "real_model_unmasked_mae": aggregate.get("model_unmasked_mae"),
            }
        )
    if os.path.exists(pbrt_summary_path):
        summary = load_json(pbrt_summary_path)
        aggregate = summary["aggregate"]
        out.update(
            {
                "pbrt_model_hole_mae": aggregate.get("model_hole_mae"),
                "pbrt_model_global_mae": aggregate.get("model_global_mae"),
                "pbrt_model_valid_mae": aggregate.get("model_valid_mae"),
            }
        )
    if os.path.exists(failure_csv_path):
        with open(failure_csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        negative = [row for row in rows if float(row.get("mask_improve_vs_anchor", 0.0)) < 0.0]
        high_residual = [row for row in rows if float(row.get("model_mask_mae", 0.0)) >= 0.12]
        out.update(
            {
                "negative_improve_count": len(negative),
                "negative_improve_cases": [row["case_name"] for row in negative],
                "high_residual_count": len(high_residual),
                "high_residual_cases": [row["case_name"] for row in high_residual[:12]],
            }
        )
    return out


def fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize Real/PBRT boundary experiment metrics.")
    parser.add_argument("model_dirs", nargs="+")
    args = parser.parse_args()

    rows = [metric_block(model_dir) for model_dir in args.model_dirs]
    keys = [
        "model_dir",
        "real_model_mask_mae",
        "real_mask_improve",
        "negative_improve_count",
        "high_residual_count",
        "pbrt_model_hole_mae",
        "pbrt_model_global_mae",
    ]
    print("\t".join(keys))
    for row in rows:
        print("\t".join(fmt(row.get(key)) for key in keys))
        if row.get("negative_improve_cases"):
            print("  negative:", ", ".join(row["negative_improve_cases"]))
        if row.get("high_residual_cases"):
            print("  high_residual:", ", ".join(row["high_residual_cases"]))


if __name__ == "__main__":
    main()
