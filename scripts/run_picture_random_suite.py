import argparse
import csv
import json
import random
import subprocess
from pathlib import Path


DEFAULT_PYTHON = "/home/lab507/anaconda3/envs/control/bin/python"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Discover real-capture depth/IQ groups under picture/, randomly sample each "
            "group, and run run_real_capture_method_suite.py for every group."
        )
    )
    parser.add_argument("--picture_root", type=Path, default=Path("picture"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--python_bin", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--samples_per_group", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--methods", nargs="+", default=["core"])
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--valid_min_depth", type=float, default=0.5)
    parser.add_argument("--valid_max_depth", type=float, default=4.5)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--allow_depthcad_cpu", action="store_true")
    parser.add_argument("--propainter_neighbor_length", type=int, default=4)
    parser.add_argument("--propainter_ref_stride", type=int, default=4)
    parser.add_argument("--propainter_subvideo_length", type=int, default=4)
    parser.add_argument("--gated_fill", action="store_true", default=False)
    parser.add_argument("--gate_diff_soft", type=float, default=0.02)
    parser.add_argument("--gate_diff_hard", type=float, default=0.08)
    parser.add_argument("--gate_component_max_mean_abs_diff", type=float, default=0.0)
    parser.add_argument("--gate_component_max_p95_abs_diff", type=float, default=0.0)
    parser.add_argument("--gate_keep_border_anchor", action="store_true", default=False)
    parser.add_argument("--repair_mask_mode", choices=["all", "exclude_large_border"], default="all")
    parser.add_argument("--preserve_border_hole_min_area", type=int, default=1024)
    parser.add_argument("--preserve_large_hole_min_area", type=int, default=24000)
    parser.add_argument("--preserve_hole_max_bbox_side", type=int, default=220)
    parser.add_argument("--preserve_holes_as_nan", action="store_true", default=False)
    parser.add_argument("--plane_fill", action="store_true", default=False)
    parser.add_argument("--plane_ring_radius", type=int, default=15)
    parser.add_argument("--plane_min_points", type=int, default=48)
    parser.add_argument("--plane_max_component_area", type=int, default=200000)
    parser.add_argument("--plane_max_abs_residual", type=float, default=0.08)
    parser.add_argument("--plane_blend_model", type=float, default=0.0)
    return parser.parse_args()


def natural_key(path_or_text):
    stem = Path(path_or_text).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def sample_id(path):
    return Path(path).stem.split("_")[-1]


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def matching_iq_dir(dataset_dir, depth_dir, iq_dirs):
    suffix = depth_dir.name.replace("depth", "", 1)
    exact = dataset_dir / f"iq{suffix}"
    if exact.is_dir():
        return exact
    if len(iq_dirs) == 1:
        return iq_dirs[0]
    return None


def discover_groups(root):
    groups = []
    for dataset_dir in sorted(Path(root).iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "__MACOSX":
            continue
        depth_dirs = sorted(
            [p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("depth")]
        )
        iq_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("iq")])
        if not depth_dirs or not iq_dirs:
            continue
        for depth_dir in depth_dirs:
            iq_dir = matching_iq_dir(dataset_dir, depth_dir, iq_dirs)
            if iq_dir is None:
                continue
            depth_paths = sorted(depth_dir.glob("depth_*.npy"), key=natural_key)
            pairs = []
            for depth_path in depth_paths:
                sid = sample_id(depth_path)
                if (iq_dir / f"iq_{sid}.npy").exists():
                    pairs.append(sid)
            if pairs:
                suffix = depth_dir.name.replace("depth_", "").replace("depth", "")
                group_name = dataset_dir.name if not suffix else f"{dataset_dir.name}_{suffix}"
                groups.append(
                    {
                        "group": group_name,
                        "dataset_dir": dataset_dir,
                        "depth_dir": depth_dir,
                        "iq_dir": iq_dir,
                        "samples": pairs,
                    }
                )
    return groups


def choose_random(samples, count, rng):
    samples = list(samples)
    if count >= len(samples):
        return samples
    return sorted(rng.sample(samples, count), key=natural_key)


def run_group(args, group, selected):
    out_dir = Path(args.output_root) / group["group"]
    mkdir(out_dir)
    log_path = out_dir / "run.log"
    cmd = [
        args.python_bin,
        "scripts/run_real_capture_method_suite.py",
        "--data_root",
        group["dataset_dir"],
        "--depth_dir",
        group["depth_dir"],
        "--iq_dir",
        group["iq_dir"],
        "--output_root",
        out_dir,
        "--samples",
    ] + selected + [
        "--methods",
    ] + list(args.methods) + [
        "--depth_scale",
        str(args.depth_scale),
        "--hole_depth_threshold",
        str(args.hole_depth_threshold),
        "--valid_min_depth",
        str(args.valid_min_depth),
        "--valid_max_depth",
        str(args.valid_max_depth),
        "--propainter_neighbor_length",
        str(args.propainter_neighbor_length),
        "--propainter_ref_stride",
        str(args.propainter_ref_stride),
        "--propainter_subvideo_length",
        str(args.propainter_subvideo_length),
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.allow_depthcad_cpu:
        cmd.append("--allow_depthcad_cpu")
    if args.gated_fill:
        cmd += [
            "--gated_fill",
            "--gate_diff_soft",
            str(args.gate_diff_soft),
            "--gate_diff_hard",
            str(args.gate_diff_hard),
            "--gate_component_max_mean_abs_diff",
            str(args.gate_component_max_mean_abs_diff),
            "--gate_component_max_p95_abs_diff",
            str(args.gate_component_max_p95_abs_diff),
        ]
        if args.gate_keep_border_anchor:
            cmd.append("--gate_keep_border_anchor")
    cmd += [
        "--repair_mask_mode",
        str(args.repair_mask_mode),
        "--preserve_border_hole_min_area",
        str(args.preserve_border_hole_min_area),
        "--preserve_large_hole_min_area",
        str(args.preserve_large_hole_min_area),
        "--preserve_hole_max_bbox_side",
        str(args.preserve_hole_max_bbox_side),
    ]
    if args.preserve_holes_as_nan:
        cmd.append("--preserve_holes_as_nan")
    if args.plane_fill:
        cmd += [
            "--plane_fill",
            "--plane_ring_radius",
            str(args.plane_ring_radius),
            "--plane_min_points",
            str(args.plane_min_points),
            "--plane_max_component_area",
            str(args.plane_max_component_area),
            "--plane_max_abs_residual",
            str(args.plane_max_abs_residual),
            "--plane_blend_model",
            str(args.plane_blend_model),
        ]

    print(f"[run] {group['group']} samples={' '.join(selected)}")
    print(f"      log: {log_path}")
    with log_path.open("w") as log:
        proc = subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, out_dir, log_path, cmd


def main():
    args = parse_args()
    groups = discover_groups(args.picture_root)
    if not groups:
        raise FileNotFoundError(f"No depth/IQ groups found under {args.picture_root}")

    rng = random.Random(int(args.seed))
    mkdir(args.output_root)

    rows = []
    for group in groups:
        selected = choose_random(group["samples"], int(args.samples_per_group), rng)
        returncode, out_dir, log_path, cmd = run_group(args, group, selected)
        row = {
            "group": group["group"],
            "dataset_dir": str(group["dataset_dir"]),
            "depth_dir": str(group["depth_dir"]),
            "iq_dir": str(group["iq_dir"]),
            "num_available_samples": len(group["samples"]),
            "selected_samples": " ".join(selected),
            "returncode": returncode,
            "output_dir": str(out_dir),
            "figures_dir": str(out_dir / "comparison" / "figures"),
            "summary_json": str(out_dir / "comparison" / "summary.json"),
            "log": str(log_path),
            "command": " ".join(str(x) for x in cmd),
        }
        rows.append(row)
        if returncode != 0:
            print(f"[error] {group['group']} failed with code {returncode}. See {log_path}")
            if not args.continue_on_error:
                break

    summary = {
        "picture_root": str(args.picture_root),
        "output_root": str(args.output_root),
        "samples_per_group": int(args.samples_per_group),
        "seed": int(args.seed),
        "methods": args.methods,
        "groups": rows,
    }
    with (Path(args.output_root) / "batch_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (Path(args.output_root) / "batch_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved batch summary: {Path(args.output_root) / 'batch_summary.json'}")
    print(f"Saved batch CSV:     {Path(args.output_root) / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
