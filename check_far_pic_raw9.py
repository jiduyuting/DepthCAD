import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from infer_real_raw9_flow import raw9_to_amplitude


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether converted far_pic raw9 tensors look compatible with the "
            "existing real raw9 dataset by comparing channel/amplitude statistics "
            "and saving preview mosaics."
        )
    )
    parser.add_argument("--far_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, default="raw")
    parser.add_argument("--output_dir", type=str, default="output/far_pic_raw9_check")
    parser.add_argument("--amplitude_mode", type=str, default="iq6", choices=["iq6", "raw_258"])
    parser.add_argument("--max_far", type=int, default=12)
    parser.add_argument("--max_ref", type=int, default=12)
    parser.add_argument("--clip_percentile", type=float, default=99.0)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_for_vis(arr, clip_percentile):
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    lo = float(np.min(finite))
    hi = float(np.percentile(finite, clip_percentile))
    if hi <= lo:
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-6)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def load_paths(root, max_count, numeric_only=False):
    paths = sorted(Path(root).glob("*.npy"))
    if numeric_only:
        paths = [p for p in paths if p.stem.isdigit()]
        paths = sorted(paths, key=lambda p: int(p.stem))
    return paths[:max_count]


def summarize_one(path, amplitude_mode):
    raw9 = np.load(path).astype(np.float32)
    if raw9.shape != (9, 240, 320):
        raise ValueError(f"{path} has shape {raw9.shape}, expected (9, 240, 320)")
    amp, amp_mean = raw9_to_amplitude(raw9, amplitude_mode)
    return {
        "name": path.name,
        "path": str(path),
        "shape": list(raw9.shape),
        "raw_min": float(np.min(raw9)),
        "raw_max": float(np.max(raw9)),
        "raw_mean": float(np.mean(raw9)),
        "channel_mean": [float(np.mean(raw9[i])) for i in range(raw9.shape[0])],
        "channel_std": [float(np.std(raw9[i])) for i in range(raw9.shape[0])],
        "amp_mean_min": float(np.min(amp_mean)),
        "amp_mean_max": float(np.max(amp_mean)),
        "amp_mean_mean": float(np.mean(amp_mean)),
        "amp_plane_mean": [float(np.mean(amp[i])) for i in range(amp.shape[0])],
        "amp_plane_std": [float(np.std(amp[i])) for i in range(amp.shape[0])],
    }


def make_preview(path, amplitude_mode, clip_percentile):
    raw9 = np.load(path).astype(np.float32)
    amp, amp_mean = raw9_to_amplitude(raw9, amplitude_mode)

    tiles = []
    for idx in range(9):
        ch = normalize_for_vis(raw9[idx], clip_percentile)
        tiles.append(cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR))
    tiles.append(cv2.cvtColor(normalize_for_vis(amp_mean, clip_percentile), cv2.COLOR_GRAY2BGR))

    if amplitude_mode == "iq6":
        labels = ["I30", "Q30", "I40", "Q40", "I58", "Q58", "Aux30", "Aux40", "Aux58", "AmpMean"]
    else:
        labels = [f"Ch{i}" for i in range(9)] + ["AmpMean"]
    labeled = []
    for label, tile in zip(labels, tiles):
        tile = tile.copy()
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        labeled.append(tile)

    row1 = np.concatenate(labeled[:5], axis=1)
    row2 = np.concatenate(labeled[5:10], axis=1)
    mosaic = np.concatenate([row1, row2], axis=0)
    return mosaic


def aggregate_stats(items, prefix):
    if not items:
        return {}
    def mean_of(key):
        return float(np.mean([item[key] for item in items]))

    out = {
        f"{prefix}_count": len(items),
        f"{prefix}_raw_mean_mean": mean_of("raw_mean"),
        f"{prefix}_amp_mean_mean": mean_of("amp_mean_mean"),
        f"{prefix}_amp_mean_max_mean": float(np.mean([item["amp_mean_max"] for item in items])),
        f"{prefix}_channel_mean_mean": [
            float(np.mean([item["channel_mean"][i] for item in items])) for i in range(9)
        ],
        f"{prefix}_amp_plane_mean_mean": [
            float(np.mean([item["amp_plane_mean"][i] for item in items])) for i in range(3)
        ],
    }
    return out


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    preview_dir = Path(args.output_dir) / "previews"
    ensure_dir(preview_dir)

    far_paths = load_paths(args.far_dir, args.max_far, numeric_only=False)
    ref_paths = load_paths(args.ref_dir, args.max_ref, numeric_only=True)

    far_items = [summarize_one(path, args.amplitude_mode) for path in far_paths]
    ref_items = [summarize_one(path, args.amplitude_mode) for path in ref_paths]

    for group_name, paths in [("far", far_paths), ("ref", ref_paths)]:
        for path in paths:
            mosaic = make_preview(path, args.amplitude_mode, args.clip_percentile)
            out_path = preview_dir / f"{group_name}_{path.stem}.png"
            cv2.imwrite(str(out_path), mosaic)

    summary = {
        "far_dir": str(Path(args.far_dir).resolve()),
        "ref_dir": str(Path(args.ref_dir).resolve()),
        "amplitude_mode": args.amplitude_mode,
        "far": far_items,
        "ref": ref_items,
    }
    summary.update(aggregate_stats(far_items, "far"))
    summary.update(aggregate_stats(ref_items, "ref"))

    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list) and not isinstance(v, dict)}, indent=2))
    print(f"Saved check results to {args.output_dir}")


if __name__ == "__main__":
    main()
