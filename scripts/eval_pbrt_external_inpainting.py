#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from inference_depth_postprocess import opencv_depth_inpaint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external inpainting outputs exported from a PBRT cache. "
            "This is intended for ProPainter/RAD-style depth-as-grayscale baselines "
            "and compares them against existing DepthCAD/ours eval summaries."
        )
    )
    parser.add_argument("--case_dir", default="output/pbrt_propainter_seed123")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--propainter_dir",
        default=None,
        help=(
            "Directory containing <source_stem>_propainter_restored.npy. "
            "Default: <case_dir>/propainter_run/restored_by_stem."
        ),
    )
    parser.add_argument("--propainter_name", default="propainter")
    parser.add_argument("--inpaint_radius", type=int, default=15)
    parser.add_argument(
        "--existing_eval",
        action="append",
        default=[],
        help=(
            "Existing eval directory or summary.json to include, formatted name:path. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--allow_missing_propainter", action="store_true", default=False)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def resolve_case(case_dir):
    case_dir = Path(case_dir)
    external_dir = case_dir / "external_inputs"
    mapping_path = external_dir / "source_mapping.json"
    meta_path = external_dir / "export" / "depth_meta.json"
    if not mapping_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing exported case files under {case_dir}")
    mapping = load_json(mapping_path)["frame_mapping"]
    meta = load_json(meta_path)
    return case_dir, mapping, meta


def safe_float(value):
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def metric_sum_count(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return 0.0, 0.0, 0
    err = pred[valid] - target[valid]
    return float(np.abs(err).sum()), float((err * err).sum()), count


def per_sample_metrics(depths, target, valid_mask, hole_mask):
    regions = {
        "global": valid_mask,
        "hole": valid_mask & hole_mask,
        "valid": valid_mask & (~hole_mask),
    }
    row = {}
    for method, pred in depths.items():
        for region, mask in regions.items():
            abs_sum, sq_sum, count = metric_sum_count(pred, target, mask)
            if count > 0:
                row[f"{method}_{region}_mae"] = safe_float(abs_sum / count)
                row[f"{method}_{region}_rmse"] = safe_float(math.sqrt(sq_sum / count))
            else:
                row[f"{method}_{region}_mae"] = None
                row[f"{method}_{region}_rmse"] = None
            row[f"{method}_{region}_count"] = count
    return row


def aggregate(rows, methods):
    totals = {}
    for method in methods:
        for region in ["global", "hole", "valid"]:
            totals[(method, region)] = {"abs": 0.0, "sq": 0.0, "count": 0}

    for row in rows:
        for method in methods:
            for region in ["global", "hole", "valid"]:
                count = int(row.get(f"{method}_{region}_count", 0))
                mae = row.get(f"{method}_{region}_mae")
                rmse = row.get(f"{method}_{region}_rmse")
                if count <= 0 or mae is None or rmse is None:
                    continue
                totals[(method, region)]["abs"] += float(mae) * count
                totals[(method, region)]["sq"] += float(rmse) ** 2 * count
                totals[(method, region)]["count"] += count

    out = {}
    for method in methods:
        for region in ["global", "hole", "valid"]:
            total = totals[(method, region)]
            count = total["count"]
            if count > 0:
                out[f"{method}_{region}_mae"] = total["abs"] / count
                out[f"{method}_{region}_rmse"] = math.sqrt(total["sq"] / count)
            else:
                out[f"{method}_{region}_mae"] = None
                out[f"{method}_{region}_rmse"] = None
            out[f"{method}_{region}_count"] = count
    return out


def load_existing_eval(spec):
    if ":" not in spec:
        raise ValueError("--existing_eval must be formatted name:path")
    name, path = spec.split(":", 1)
    path = Path(path)
    if path.is_dir():
        summary_path = path / "summary.json"
        per_sample_path = path / "per_sample_results.json"
    else:
        summary_path = path
        per_sample_path = path.parent / "per_sample_results.json"
    summary = load_json(summary_path)
    per_sample = load_json(per_sample_path) if per_sample_path.exists() else []
    by_sample = {
        row["sample_name"]: row
        for row in per_sample
        if isinstance(row, dict) and "sample_name" in row
    }
    return {
        "name": name,
        "summary_path": str(summary_path),
        "per_sample_path": str(per_sample_path) if per_sample_path.exists() else None,
        "summary": summary,
        "per_sample_by_name": by_sample,
    }


def weighted_metric(rows, method, region, metric):
    total = 0.0
    count = 0
    for row in rows:
        value = row.get(f"{method}_{region}_{metric}")
        n = int(row.get(f"{method}_{region}_count", 0))
        if value is None or n <= 0:
            continue
        total += float(value) * n
        count += n
    return (total / count) if count > 0 else None


def compare_existing(prop_rows, prop_method, existing):
    comparisons = {}
    prop_by_sample = {row["sample_name"]: row for row in prop_rows}
    for item in existing:
        name = item["name"]
        summary_agg = item["summary"].get("aggregate", {})
        existing_hole = summary_agg.get("model_hole_mae")
        existing_global = summary_agg.get("model_global_mae")
        existing_valid = summary_agg.get("model_valid_mae")
        prop_hole = weighted_metric(prop_rows, prop_method, "hole", "mae")
        if prop_hole is not None and existing_hole is not None:
            comparisons[f"{prop_method}_vs_{name}_hole_mae_delta"] = prop_hole - float(existing_hole)
            comparisons[f"{prop_method}_vs_{name}_hole_mae_ratio"] = prop_hole / max(float(existing_hole), 1e-12)
        if existing_global is not None:
            comparisons[f"{name}_model_global_mae"] = float(existing_global)
        if existing_hole is not None:
            comparisons[f"{name}_model_hole_mae"] = float(existing_hole)
        if existing_valid is not None:
            comparisons[f"{name}_model_valid_mae"] = float(existing_valid)

        paired = []
        for sample_name, ext_row in item["per_sample_by_name"].items():
            prop_row = prop_by_sample.get(sample_name)
            if prop_row is None:
                continue
            prop_mae = prop_row.get(f"{prop_method}_hole_mae")
            ours_mae = ext_row.get("model_hole_mae")
            if prop_mae is None or ours_mae is None:
                continue
            paired.append((float(prop_mae), float(ours_mae), sample_name))
        if paired:
            prop_better = sum(1 for prop_mae, ours_mae, _ in paired if prop_mae < ours_mae)
            ours_better = sum(1 for prop_mae, ours_mae, _ in paired if prop_mae > ours_mae)
            tied = len(paired) - prop_better - ours_better
            deltas = np.asarray([prop_mae - ours_mae for prop_mae, ours_mae, _ in paired], dtype=np.float64)
            comparisons[f"{prop_method}_vs_{name}_paired_count"] = len(paired)
            comparisons[f"{prop_method}_vs_{name}_propainter_better"] = int(prop_better)
            comparisons[f"{prop_method}_vs_{name}_ours_better"] = int(ours_better)
            comparisons[f"{prop_method}_vs_{name}_tied"] = int(tied)
            comparisons[f"{prop_method}_vs_{name}_paired_hole_delta_mean"] = float(deltas.mean())
            comparisons[f"{prop_method}_vs_{name}_paired_hole_delta_median"] = float(np.median(deltas))
            worst_for_prop = sorted(paired, key=lambda x: x[0] - x[1], reverse=True)[:5]
            best_for_prop = sorted(paired, key=lambda x: x[0] - x[1])[:5]
            comparisons[f"{prop_method}_vs_{name}_worst_for_propainter"] = [
                {"sample_name": s, "propainter_hole_mae": p, "ours_hole_mae": o, "delta": p - o}
                for p, o, s in worst_for_prop
            ]
            comparisons[f"{prop_method}_vs_{name}_best_for_propainter"] = [
                {"sample_name": s, "propainter_hole_mae": p, "ours_hole_mae": o, "delta": p - o}
                for p, o, s in best_for_prop
            ]
    return comparisons


def main():
    args = parse_args()
    case_dir, mapping, meta = resolve_case(args.case_dir)
    out_dir = Path(args.output_dir) if args.output_dir else case_dir / "evaluation"
    ensure_dir(out_dir)

    propainter_dir = (
        Path(args.propainter_dir)
        if args.propainter_dir
        else case_dir / "propainter_run" / "restored_by_stem"
    )

    depth_dir = Path(meta["source_depth_npy"])
    gt_dir = Path(meta["gt_source"])
    mask_dir = Path(meta["mask_source"])
    valid_dir = Path(meta["valid_source"])

    rows = []
    methods = ["noisy", f"opencv_ns_r{args.inpaint_radius}"]
    propainter_available = propainter_dir.is_dir()
    if propainter_available:
        methods.append(args.propainter_name)
    elif not args.allow_missing_propainter:
        raise FileNotFoundError(f"ProPainter restored dir not found: {propainter_dir}")

    for item in mapping:
        idx = int(item["frame_index"])
        stem = item["source_stem"]
        sample_name = item.get("sample_name", stem)
        noisy = np.load(depth_dir / f"{idx:04d}.npy").astype(np.float32)
        gt = np.load(gt_dir / f"{idx:04d}.npy").astype(np.float32)
        hole = np.load(mask_dir / f"{idx:04d}.npy").astype(bool)
        valid = np.load(valid_dir / f"{idx:04d}.npy").astype(bool)
        ns = opencv_depth_inpaint(noisy, hole, method="ns", radius=args.inpaint_radius)

        depths = {
            "noisy": noisy,
            f"opencv_ns_r{args.inpaint_radius}": ns,
        }
        prop_path = propainter_dir / f"{stem}_propainter_restored.npy"
        if propainter_available:
            if not prop_path.exists():
                if args.allow_missing_propainter:
                    propainter_available = False
                else:
                    raise FileNotFoundError(f"Missing ProPainter output: {prop_path}")
            else:
                depths[args.propainter_name] = np.load(prop_path).astype(np.float32)

        row = {
            "frame_index": idx,
            "source_stem": stem,
            "sample_name": sample_name,
            "cache_path": item.get("cache_path"),
            "hole_ratio": float(hole.mean()),
        }
        row.update(per_sample_metrics(depths, gt, valid, hole))
        rows.append(row)

    aggregate_metrics = aggregate(rows, methods)
    existing = [load_existing_eval(spec) for spec in args.existing_eval]
    comparisons = compare_existing(rows, args.propainter_name, existing) if propainter_available else {}

    summary = {
        "case_dir": str(case_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "num_samples": len(rows),
        "methods": methods,
        "propainter_dir": str(propainter_dir.resolve()),
        "propainter_available": propainter_available,
        "aggregate": aggregate_metrics,
        "existing_evals": [
            {
                "name": item["name"],
                "summary_path": item["summary_path"],
                "per_sample_path": item["per_sample_path"],
                "num_per_sample": len(item["per_sample_by_name"]),
            }
            for item in existing
        ],
        "comparisons": comparisons,
    }

    save_json(out_dir / "per_sample_results.json", rows)
    save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
