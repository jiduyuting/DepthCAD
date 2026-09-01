#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DEFAULT_RUNS = [
    ("CompletionFormer", "output/pbrt100_depth_completion/completionformer/summary.json", "supervised or fine-tuned"),
    ("DMD3C", "output/pbrt100_depth_completion/dmd3c/summary.json", "zero-shot or PBRT-trained; record setting"),
    ("OMNI-DC", "output/pbrt100_depth_completion/omnidc_zero_shot/summary.json", "official zero-shot"),
    ("LDCM", "output/pbrt100_depth_completion/ldcm_zero_shot/summary.json", "official zero-shot"),
    ("LingBot-Depth", "output/pbrt100_depth_completion/lingbot_dc_zero_shot/summary.json", "official zero-shot"),
    ("DEPTHOR", "output/pbrt100_depth_completion/depthor/summary.json", "official or PBRT-trained; record setting"),
    ("RGBD-Imaging", "output/pbrt100_depth_completion/rgbd_lfrd2/summary.json", "adapted baseline"),
    ("LFRD2", "output/pbrt100_depth_completion/rgbd_lfrd2/summary.json", "adapted/proxy baseline"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize a selected model and depth-completion baselines on the seed123 100-sample PBRT holdout."
    )
    parser.add_argument("--selected", action="append", default=[], help="Selected model as name:path_to_summary.json.")
    parser.add_argument("--run", action="append", default=[], help="Baseline as name:path_to_summary.json[:note].")
    parser.add_argument("--output_dir", type=Path, default=Path("output/pbrt100_depth_completion/comparison"))
    parser.add_argument("--expected_samples", type=int, default=100)
    parser.add_argument("--include_defaults", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def split_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Run spec must be name:path[:note], got {spec}")
    name, path = parts[:2]
    note = parts[2] if len(parts) == 3 else ""
    return name, path, note


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_samples(summary):
    samples = summary.get("samples", summary.get("num_samples"))
    if isinstance(samples, dict):
        return samples.get("test", samples.get("val"))
    return samples


def metric(summary, region, kind):
    if "metrics" in summary:
        metrics = summary.get("metrics", {})
        nested_value = metrics.get(region, {}).get(f"{kind}_m")
        if nested_value is not None:
            return nested_value
        direct_aliases = {
            "global": ("model_global", "global"),
            "hole": ("model_hole", "hole", "model_mask", "mask"),
            "observed": ("model_observed", "model_valid", "observed", "valid"),
        }
        suffix = "mae" if kind == "mae" else "rmse"
        for prefix in direct_aliases[region]:
            value = metrics.get(f"{prefix}_{suffix}")
            if value is not None:
                return value
    aggregate = summary.get("aggregate", {})
    aliases = {
        "global": ("model_global", "global"),
        "hole": ("model_hole", "hole", "model_mask", "mask"),
        "observed": ("model_observed", "model_valid", "observed", "valid", "model_unmasked", "unmasked"),
    }
    suffix = "mae" if kind == "mae" else "rmse"
    for prefix in aliases[region]:
        key = f"{prefix}_{suffix}"
        if key in aggregate:
            return aggregate[key]
    return None


def parse_unified_baseline(summary, name):
    key = name.lower()
    if key.startswith("rgbd"):
        payload = summary.get("rgbd")
    elif key.startswith("lfrd2"):
        payload = summary.get("lfrd2")
    else:
        payload = None
    if payload is None:
        return None
    return {
        "method": name,
        "samples": summary.get("num_samples"),
        "hole_mae": payload.get("hole_mae_m"),
        "hole_rmse": payload.get("hole_rmse_m"),
        "global_mae": payload.get("global_mae_m"),
        "global_rmse": payload.get("global_rmse_m"),
        "observed_mae": payload.get("valid_mae_m"),
        "observed_rmse": payload.get("valid_rmse_m"),
    }


def row_from_summary(name, path, note, expected_samples):
    path = Path(path)
    if not path.exists():
        return {
            "method": name,
            "status": "missing",
            "samples": "",
            "hole_mae": None,
            "hole_rmse": None,
            "global_mae": None,
            "global_rmse": None,
            "observed_mae": None,
            "observed_rmse": None,
            "source": str(path),
            "note": note,
        }
    summary = load_json(path)
    if "num_samples" in summary and (name.lower().startswith("rgbd") or name.lower().startswith("lfrd2")):
        payload_key = "rgbd" if name.lower().startswith("rgbd") else "lfrd2"
        if payload_key not in summary:
            return {
                "method": name,
                "status": "missing",
                "samples": summary.get("num_samples"),
                "hole_mae": None,
                "hole_rmse": None,
                "global_mae": None,
                "global_rmse": None,
                "observed_mae": None,
                "observed_rmse": None,
                "source": str(path),
                "note": f"{note}; {payload_key} not present",
            }
    parsed = parse_unified_baseline(summary, name)
    if parsed is None:
        parsed = {
            "method": name,
            "samples": get_samples(summary),
            "hole_mae": metric(summary, "hole", "mae"),
            "hole_rmse": metric(summary, "hole", "rmse"),
            "global_mae": metric(summary, "global", "mae"),
            "global_rmse": metric(summary, "global", "rmse"),
            "observed_mae": metric(summary, "observed", "mae"),
            "observed_rmse": metric(summary, "observed", "rmse"),
        }
    samples = parsed["samples"]
    status = "ok" if samples == expected_samples else f"non-{expected_samples}"
    parsed.update({"status": status, "source": str(path), "note": note})
    return parsed


def fmt_float(value):
    return "" if value is None else f"{float(value):.6f}"


def write_csv(path, rows):
    fieldnames = [
        "method",
        "status",
        "samples",
        "hole_mae",
        "hole_rmse",
        "global_mae",
        "global_rmse",
        "observed_mae",
        "observed_rmse",
        "source",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path, rows):
    lines = [
        "# PBRT100 Depth Completion Comparison",
        "",
        "> Protocol: `gt_depth` is a ToF-derived reference map and may contain invalid (`<=0.1` or non-finite) pixels. All reported metrics use `valid_mask`; `hole` is `valid_mask & hole_mask`, `observed` is `valid_mask & ~hole_mask`, and `global` is `valid_mask`. GT visualizations show the raw array without invalid-pixel masking; `Hole mask` is a separate panel without contour overlays.",
        "",
        "| Method | Status | Samples | Hole MAE | Hole RMSE | Global MAE | Global RMSE | Observed MAE | Observed RMSE | Note | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["status"]),
                    str(row["samples"]),
                    fmt_float(row.get("hole_mae")),
                    fmt_float(row.get("hole_rmse")),
                    fmt_float(row.get("global_mae")),
                    fmt_float(row.get("global_rmse")),
                    fmt_float(row.get("observed_mae")),
                    fmt_float(row.get("observed_rmse")),
                    str(row.get("note", "")),
                    str(row.get("source", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    specs = []
    for selected in args.selected:
        name, path, note = split_spec(selected)
        specs.append((name, path, note or "selected model"))
    if args.include_defaults:
        specs.extend(DEFAULT_RUNS)
    for run in args.run:
        specs.append(split_spec(run))

    rows = [row_from_summary(name, path, note, args.expected_samples) for name, path, note in specs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "summary.md", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
