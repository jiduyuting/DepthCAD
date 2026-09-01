#!/usr/bin/env python3
import _bootstrap
import argparse
import json
import math
from pathlib import Path

import torch


TARGETS = {
    "hole_mae": 0.1029596836777437,
    "global_mae": 0.03868146848423947,
    "observed_mae": 0.0267,
}


def checkpoint_metrics(path):
    payload = torch.load(path, map_location="cpu")
    metrics = payload.get("metrics", {})
    hole = float(metrics.get("model_hole_mae", math.inf))
    global_mae = float(metrics.get("model_global_mae", math.inf))
    observed = float(metrics.get("model_valid_mae", math.inf))
    return {
        "checkpoint": str(path.resolve()),
        "epoch": int(payload.get("epoch", -1)),
        "hole_mae": hole,
        "global_mae": global_mae,
        "observed_mae": observed,
        "selection_score": hole + observed,
    }


def select_checkpoint(root, output):
    candidates = []
    for experiment in sorted((root / "stage1").glob("*")):
        if not experiment.is_dir():
            continue
        for name in ("best.pt", "best_hole.pt", "best_global.pt"):
            path = experiment / name
            if path.is_file():
                row = checkpoint_metrics(path)
                row["experiment"] = experiment.name
                row["checkpoint_kind"] = name
                candidates.append(row)
    if not candidates:
        raise FileNotFoundError(f"No stage-1 checkpoints found under {root / 'stage1'}")
    candidates.sort(key=lambda row: (row["selection_score"], row["global_mae"], row["hole_mae"]))
    result = {"selected": candidates[0], "candidates": candidates}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(candidates[0]["checkpoint"])


def metric_triplet(summary):
    aggregate = summary["aggregate"]
    return {
        "hole_mae": float(aggregate["model_hole_mae"]),
        "global_mae": float(aggregate["model_global_mae"]),
        "observed_mae": float(aggregate["model_valid_mae"]),
        "hole_rmse": aggregate.get("model_hole_rmse"),
        "global_rmse": aggregate.get("model_global_rmse"),
    }


def summarize(root, output):
    rows = []
    for path in sorted((root / "test_eval").glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        row = {"experiment": path.parent.name, "summary": str(path.resolve())}
        row.update(metric_triplet(summary))
        row["beats_rgbd_hole"] = row["hole_mae"] < TARGETS["hole_mae"]
        row["beats_completionformer_global"] = row["global_mae"] < TARGETS["global_mae"]
        row["meets_joint_target"] = row["beats_rgbd_hole"] and row["beats_completionformer_global"]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No test summaries found under {root / 'test_eval'}")
    rows.sort(key=lambda row: (not row["meets_joint_target"], row["hole_mae"], row["global_mae"]))
    payload = {"targets": TARGETS, "best": rows[0], "experiments": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Flow SOTA experiment summary",
        "",
        f"Targets: Hole MAE < {TARGETS['hole_mae']:.6f}; Global MAE < {TARGETS['global_mae']:.6f}.",
        "",
        "| Experiment | Hole MAE | Global MAE | Observed MAE | Joint target |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['hole_mae']:.6f} | {row['global_mae']:.6f} | "
            f"{row['observed_mae']:.6f} | {'YES' if row['meets_joint_target'] else 'NO'} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["best"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        select_checkpoint(args.root, args.output)
    else:
        summarize(args.root, args.output)


if __name__ == "__main__":
    main()
