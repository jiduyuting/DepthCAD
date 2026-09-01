#!/usr/bin/env python3
import _bootstrap
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def selected_paths(path, max_samples):
    rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if max_samples and len(rows) > max_samples:
        indices = np.linspace(0, len(rows) - 1, max_samples).round().astype(int)
        rows = [rows[index] for index in indices]
    return rows


def audit(path, max_samples):
    records = []
    for source in selected_paths(path, max_samples):
        with np.load(source, allow_pickle=False) as data:
            gt = data["gt_depth"]
            raw_hole = data["hole_mask"] > 0.5
            valid = (
                (data["valid_mask"] > 0.5)
                & np.isfinite(gt)
                & (gt > 0.1)
                & (gt < 9.9)
            )
        hole = (raw_hole & valid).astype(np.uint8)
        components, _, stats, _ = cv2.connectedComponentsWithStats(hole, 8)
        distance = cv2.distanceTransform(hole, cv2.DIST_L2, 3)
        hole_count = int(hole.sum())
        records.append({
            "raw_hole_fraction": float(raw_hole.mean()),
            "effective_hole_fraction": float(hole.mean()),
            "invalid_overlap_fraction": float((raw_hole & (~valid)).mean()),
            "invalid_gt_fraction": float((~valid).mean()),
            "components": int(max(components - 1, 0)),
            "largest_component": int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else 0,
            "edge_fraction_in_hole": float(((hole > 0) & (distance <= 2.0)).sum() / max(hole_count, 1)),
            "mean_hole_distance": float(distance[hole > 0].mean()) if hole_count else 0.0,
        })
    result = {"list": str(path.resolve()), "sampled": len(records)}
    for key in records[0] if records else []:
        values = np.asarray([row[key] for row in records], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit effective-hole and geometry statistics for Flow splits.")
    parser.add_argument("--train_list", type=Path, required=True)
    parser.add_argument("--val_list", type=Path, required=True)
    parser.add_argument("--test_list", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "hole_definition": "hole_mask & valid_mask",
        "splits": {
            "train": audit(args.train_list, args.max_samples),
            "val": audit(args.val_list, args.max_samples),
            "test": audit(args.test_list, args.max_samples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
