import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare newly captured depth/IQ npy pairs for real-data inference."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--output_root", type=Path, default=Path("data/prepared_new_capture"))
    parser.add_argument("--good_valid_ratio", type=float, default=0.10)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--depth_vis_min", type=float, default=0.5)
    parser.add_argument("--depth_vis_max", type=float, default=4.5)
    return parser.parse_args()


def natural_key(path):
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def sample_id(path):
    stem = Path(path).stem
    return stem.split("_")[-1]


def to_uint8(values, valid=None, vmin=None, vmax=None):
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    out = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    if vmin is None:
        vmin = float(np.quantile(values[valid], 0.01))
    if vmax is None:
        vmax = float(np.quantile(values[valid], 0.99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    scaled = (values - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[valid] = (scaled[valid] * 255.0).astype(np.uint8)
    return out


def colorize_gray(gray, mask=None):
    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    color = cv2.applyColorMap(gray, cmap)
    if mask is not None:
        color[~mask] = 0
    return color


def put_label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (220, 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return out


def iq_to_chw(iq):
    if iq.ndim != 3:
        raise ValueError(f"Expected IQ/raw9 array with 3 dims, got {iq.shape}")
    if iq.shape[-1] == 9:
        return np.moveaxis(iq, -1, 0).astype(np.float32), "HWC_to_CHW"
    if iq.shape[0] == 9:
        return iq.astype(np.float32), "already_CHW"
    raise ValueError(f"Expected IQ/raw9 shape (H,W,9) or (9,H,W), got {iq.shape}")


def iq6_amplitude(raw9_chw):
    i_channels = np.stack([raw9_chw[0], raw9_chw[2], raw9_chw[4]], axis=0)
    q_channels = np.stack([raw9_chw[1], raw9_chw[3], raw9_chw[5]], axis=0)
    return np.sqrt(i_channels * i_channels + q_channels * q_channels)


def quantiles(values, qs):
    values = np.asarray(values)
    if values.size == 0:
        return {str(q): None for q in qs}
    out = {}
    for q in qs:
        out[str(q)] = float(np.quantile(values, q))
    return out


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        "sample",
        "status",
        "good_for_first_test",
        "depth_shape",
        "raw9_shape",
        "depth_unit",
        "valid_ratio",
        "depth_min_m",
        "depth_median_m",
        "depth_p99_m",
        "depth_max_m",
        "iq_min",
        "iq_median",
        "iq_p99",
        "iq_max",
        "sat_value_ratio_65535",
        "sat_pixel_ratio_65535",
        "layout_action",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    depth_dir = args.data_root / "depth"
    iq_dir = args.data_root / "iq"
    output_root = args.output_root

    all_depth_dir = output_root / "all" / "depth_m"
    all_raw_dir = output_root / "all" / "raw9_chw"
    good_depth_dir = output_root / "good" / "depth_m"
    good_raw_dir = output_root / "good" / "raw9_chw"
    qa_dir = output_root / "qa"
    for path in [all_depth_dir, all_raw_dir, good_depth_dir, good_raw_dir, qa_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    depth_paths = sorted(depth_dir.glob("depth_*.npy"), key=natural_key)
    if not depth_paths:
        raise FileNotFoundError(f"No depth_*.npy files found under {depth_dir}")

    for depth_path in depth_paths:
        sid = sample_id(depth_path)
        iq_path = iq_dir / f"iq_{sid}.npy"
        if not iq_path.exists():
            raise FileNotFoundError(f"Missing IQ pair for {depth_path.name}: {iq_path}")

        depth_mm = np.load(depth_path).astype(np.float32)
        iq = np.load(iq_path).astype(np.float32)
        raw9_chw, layout_action = iq_to_chw(iq)
        if raw9_chw.shape[1:] != depth_mm.shape:
            raise ValueError(f"Shape mismatch for {sid}: depth {depth_mm.shape}, raw9 {raw9_chw.shape}")

        depth_m = depth_mm / float(args.depth_scale)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        valid_values = depth_m[valid]
        iq_values = iq[np.isfinite(iq)]
        sat_values = iq >= 65535.0
        if iq.shape[-1] == 9:
            sat_pixels = np.any(sat_values, axis=-1)
        else:
            sat_pixels = np.any(sat_values, axis=0)
        valid_ratio = float(valid.mean())
        good = valid_ratio >= float(args.good_valid_ratio)
        status = "ok" if good else ("bad_depth" if valid_ratio < 0.01 else "low_depth")

        out_name = f"{sid}.npy"
        np.save(all_depth_dir / out_name, depth_m.astype(np.float32))
        np.save(all_raw_dir / out_name, raw9_chw.astype(np.float32))
        if good:
            np.save(good_depth_dir / out_name, depth_m.astype(np.float32))
            np.save(good_raw_dir / out_name, raw9_chw.astype(np.float32))

        depth_gray = to_uint8(
            depth_m,
            valid=valid,
            vmin=float(args.depth_vis_min),
            vmax=float(args.depth_vis_max),
        )
        depth_vis = colorize_gray(depth_gray, valid)
        hole_vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        hole_vis[~valid] = (255, 255, 255)
        amplitude = iq6_amplitude(raw9_chw).mean(axis=0)
        amp_gray = to_uint8(amplitude, valid=np.isfinite(amplitude))
        amp_vis = colorize_gray(amp_gray)
        sat_vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        sat_vis[sat_pixels] = (255, 255, 255)

        cv2.imwrite(str(qa_dir / f"{sid}_depth_m.png"), depth_vis)
        cv2.imwrite(str(qa_dir / f"{sid}_valid_holes.png"), hole_vis)
        cv2.imwrite(str(qa_dir / f"{sid}_iq6_amp.png"), amp_vis)
        cv2.imwrite(str(qa_dir / f"{sid}_sat65535.png"), sat_vis)
        preview = np.concatenate(
            [
                put_label(depth_vis, f"{sid} depth m"),
                put_label(hole_vis, "holes"),
                put_label(amp_vis, "iq6 amp"),
                put_label(sat_vis, "sat 65535"),
            ],
            axis=1,
        )
        cv2.imwrite(str(qa_dir / f"{sid}_preview.png"), preview)

        depth_q = quantiles(valid_values, [0.0, 0.5, 0.99, 1.0])
        iq_q = quantiles(iq_values, [0.0, 0.5, 0.99, 1.0])
        rows.append(
            {
                "sample": sid,
                "status": status,
                "good_for_first_test": bool(good),
                "depth_shape": list(depth_m.shape),
                "raw9_shape": list(raw9_chw.shape),
                "depth_unit": "m",
                "valid_ratio": valid_ratio,
                "depth_min_m": depth_q["0.0"],
                "depth_median_m": depth_q["0.5"],
                "depth_p99_m": depth_q["0.99"],
                "depth_max_m": depth_q["1.0"],
                "iq_min": iq_q["0.0"],
                "iq_median": iq_q["0.5"],
                "iq_p99": iq_q["0.99"],
                "iq_max": iq_q["1.0"],
                "sat_value_ratio_65535": float(sat_values.mean()),
                "sat_pixel_ratio_65535": float(sat_pixels.mean()),
                "layout_action": layout_action,
            }
        )

    summary = {
        "source_data_root": str(args.data_root),
        "output_root": str(output_root),
        "all_depth_dir": str(all_depth_dir),
        "all_raw9_dir": str(all_raw_dir),
        "good_depth_dir": str(good_depth_dir),
        "good_raw9_dir": str(good_raw_dir),
        "good_valid_ratio_threshold": float(args.good_valid_ratio),
        "num_samples": len(rows),
        "num_good_for_first_test": sum(1 for row in rows if row["good_for_first_test"]),
        "bad_or_low_depth_samples": [
            row["sample"] for row in rows if not row["good_for_first_test"]
        ],
        "rows": rows,
    }
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_csv(output_root / "summary.csv", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
