import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PYTHON = "/home/lab507/anaconda3/envs/control/bin/python"

METHOD_PRESETS = {
    "core": [
        "depth_only",
        "raw9_satclip",
        "raw9_realholes",
        "after_synth",
        "propagation",
        "propainter",
    ],
    "flows": [
        "depth_only",
        "raw9_satclip",
        "raw9_realholes",
        "after_synth",
        "propagation",
    ],
    "all": [
        "depth_only",
        "raw9_satclip",
        "raw9_realholes",
        "after_synth",
        "propagation",
        "propainter",
        "depthcad_depth_gray",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-click suite for real captured depth/raw9 completion: prepare data, run "
            "selected methods, and build unified comparison visualizations."
        )
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--depth_dir", type=Path, default=None)
    parser.add_argument("--iq_dir", type=Path, default=None)
    parser.add_argument("--python_bin", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--sample_mode", choices=["auto", "all"], default="auto")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--auto_count", type=int, default=4)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--valid_min_depth", type=float, default=0.5)
    parser.add_argument("--valid_max_depth", type=float, default=4.5)
    parser.add_argument("--depth_vis_min", type=float, default=0.5)
    parser.add_argument("--depth_vis_max", type=float, default=4.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["core"],
        help=(
            "Method names or presets. Presets: core, flows, all. "
            "Names: depth_only raw9_satclip raw9_realholes after_synth propagation "
            "propainter depthcad_depth_gray."
        ),
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_prepare", action="store_true")
    parser.add_argument("--no_compare", action="store_true")
    parser.add_argument("--allow_depthcad_cpu", action="store_true")
    parser.add_argument("--depthcad_steps", type=int, default=5)
    parser.add_argument("--depthcad_infer_size", type=int, default=256)
    parser.add_argument("--propainter_height", type=int, default=None)
    parser.add_argument("--propainter_width", type=int, default=None)
    parser.add_argument("--propainter_neighbor_length", type=int, default=10)
    parser.add_argument("--propainter_ref_stride", type=int, default=10)
    parser.add_argument("--propainter_subvideo_length", type=int, default=80)
    parser.add_argument(
        "--gated_fill",
        action="store_true",
        default=False,
        help="Ask raw9 flow methods to save anchor-gated hole-only outputs for comparison.",
    )
    parser.add_argument("--gate_diff_soft", type=float, default=0.02)
    parser.add_argument("--gate_diff_hard", type=float, default=0.08)
    parser.add_argument("--gate_component_max_mean_abs_diff", type=float, default=0.0)
    parser.add_argument("--gate_component_max_p95_abs_diff", type=float, default=0.0)
    parser.add_argument("--gate_keep_border_anchor", action="store_true", default=False)
    parser.add_argument(
        "--repair_mask_mode",
        choices=["all", "exclude_large_border"],
        default="all",
        help="Filter raw9 flow repair masks before writing final hole-only outputs.",
    )
    parser.add_argument("--preserve_border_hole_min_area", type=int, default=1024)
    parser.add_argument("--preserve_large_hole_min_area", type=int, default=24000)
    parser.add_argument("--preserve_hole_max_bbox_side", type=int, default=220)
    parser.add_argument("--preserve_holes_as_nan", action="store_true", default=False)
    parser.add_argument(
        "--plane_fill",
        action="store_true",
        default=False,
        help="Ask raw9 flow methods to save local plane/median hole-only fallback outputs.",
    )
    parser.add_argument("--plane_ring_radius", type=int, default=15)
    parser.add_argument("--plane_min_points", type=int, default=48)
    parser.add_argument("--plane_max_component_area", type=int, default=200000)
    parser.add_argument("--plane_max_abs_residual", type=float, default=0.08)
    parser.add_argument("--plane_blend_model", type=float, default=0.0)
    return parser.parse_args()


def natural_key(path_or_text):
    stem = Path(path_or_text).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (0, int(digits)) if digits else (1, stem)


def sample_id(path):
    stem = Path(path).stem
    return stem.split("_")[-1]


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_subdir(root, prefix, pattern):
    candidates = []
    for child in sorted(Path(root).iterdir()):
        if child.is_dir() and child.name.startswith(prefix) and list(child.glob(pattern)):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"Could not find {prefix} directory with {pattern} under {root}")
    return candidates[0]


def resolve_dirs(args):
    data_root = Path(args.data_root)
    depth_dir = Path(args.depth_dir) if args.depth_dir else find_subdir(data_root, "depth", "depth_*.npy")
    iq_dir = Path(args.iq_dir) if args.iq_dir else find_subdir(data_root, "iq", "iq_*.npy")
    return depth_dir, iq_dir


def load_pairs(depth_dir, iq_dir):
    depth_paths = sorted(Path(depth_dir).glob("depth_*.npy"), key=natural_key)
    pairs = []
    for depth_path in depth_paths:
        sid = sample_id(depth_path)
        iq_path = Path(iq_dir) / f"iq_{sid}.npy"
        if iq_path.exists():
            pairs.append((sid, depth_path, iq_path))
    if not pairs:
        raise FileNotFoundError(f"No paired depth_*.npy / iq_*.npy files in {depth_dir} and {iq_dir}")
    return pairs


def depth_stats(depth_path):
    depth = np.load(depth_path).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    vals = depth[valid]
    return {
        "shape": list(depth.shape),
        "valid_ratio": float(valid.mean()),
        "min": float(np.min(vals)) if vals.size else None,
        "median": float(np.median(vals)) if vals.size else None,
        "p99": float(np.percentile(vals, 99.0)) if vals.size else None,
        "max": float(np.max(vals)) if vals.size else None,
    }


def choose_samples(pairs, args):
    if args.samples:
        wanted = {str(s) for s in args.samples}
        chosen = [sid for sid, _, _ in pairs if sid in wanted]
        if not chosen:
            raise ValueError(f"--samples did not match any paired sample: {args.samples}")
        return chosen
    if args.sample_mode == "all":
        return [sid for sid, _, _ in pairs]

    rows = [(sid, depth_stats(depth_path)["valid_ratio"]) for sid, depth_path, _ in pairs]
    n = len(rows)
    target = []
    candidate_indices = [0, n // 2, n - 1]
    sorted_by_valid = sorted(range(n), key=lambda i: rows[i][1])
    candidate_indices += [sorted_by_valid[0], sorted_by_valid[len(sorted_by_valid) // 2], sorted_by_valid[-1]]
    if args.auto_count > len(candidate_indices):
        even = np.linspace(0, n - 1, min(args.auto_count, n)).round().astype(int).tolist()
        candidate_indices += even
    for idx in candidate_indices:
        sid = rows[idx][0]
        if sid not in target:
            target.append(sid)
        if len(target) >= int(args.auto_count):
            break
    return target


def expand_methods(methods):
    expanded = []
    for item in methods:
        if item in METHOD_PRESETS:
            for name in METHOD_PRESETS[item]:
                if name not in expanded:
                    expanded.append(name)
        else:
            if item not in expanded:
                expanded.append(item)
    valid = set(METHOD_PRESETS["all"])
    unknown = [m for m in expanded if m not in valid]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")
    return expanded


def run_cmd(cmd, cwd, skip_existing_path=None, skip_existing=False):
    if skip_existing and skip_existing_path is not None and Path(skip_existing_path).exists():
        print(f"[skip existing] {skip_existing_path}")
        return
    print()
    print("[run]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def stage_selected_inputs(pairs, selected, output_root):
    selected_set = set(selected)
    input_root = Path(output_root) / "input"
    depth_out = input_root / "depth"
    iq_out = input_root / "iq"
    mkdir(depth_out)
    mkdir(iq_out)
    copied = []
    for sid, depth_path, iq_path in pairs:
        if sid not in selected_set:
            continue
        shutil.copy2(depth_path, depth_out / f"depth_{sid}.npy")
        shutil.copy2(iq_path, iq_out / f"iq_{sid}.npy")
        copied.append({"sample": sid, "depth": str(depth_path), "iq": str(iq_path)})
    return input_root, copied


def write_selection_report(output_root, depth_dir, iq_dir, selected, copied, pairs):
    rows = []
    for sid, depth_path, iq_path in pairs:
        stats = depth_stats(depth_path)
        iq = np.load(iq_path).astype(np.float32)
        sat = iq >= 65535.0
        if iq.ndim == 3 and iq.shape[-1] == 9:
            sat_pixel = np.any(sat, axis=-1)
        elif iq.ndim == 3 and iq.shape[0] == 9:
            sat_pixel = np.any(sat, axis=0)
        else:
            sat_pixel = sat
        rows.append(
            {
                "sample": sid,
                "selected": sid in set(selected),
                "valid_ratio": stats["valid_ratio"],
                "depth_shape": stats["shape"],
                "depth_min": stats["min"],
                "depth_median": stats["median"],
                "depth_p99": stats["p99"],
                "depth_max": stats["max"],
                "iq_shape": list(iq.shape),
                "iq_min": float(np.nanmin(iq)),
                "iq_median": float(np.nanmedian(iq)),
                "iq_p99": float(np.nanpercentile(iq, 99.0)),
                "iq_max": float(np.nanmax(iq)),
                "sat_value_ratio_65535": float(sat.mean()),
                "sat_pixel_ratio_65535": float(sat_pixel.mean()),
            }
        )
    output_root = Path(output_root)
    mkdir(output_root)
    with (output_root / "selection_summary.json").open("w") as f:
        json.dump(
            {
                "source_depth_dir": str(depth_dir),
                "source_iq_dir": str(iq_dir),
                "selected_samples": selected,
                "copied": copied,
                "rows": rows,
            },
            f,
            indent=2,
        )
    with (output_root / "selection_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(args, input_root):
    prepared = Path(args.output_root) / "prepared"
    cmd = [
        args.python_bin,
        "scripts/prepare_new_capture_data.py",
        "--data_root",
        input_root,
        "--output_root",
        prepared,
        "--good_valid_ratio",
        "0.10",
        "--depth_scale",
        str(args.depth_scale),
        "--depth_vis_min",
        str(args.depth_vis_min),
        "--depth_vis_max",
        str(args.depth_vis_max),
    ]
    run_cmd(cmd, Path.cwd(), prepared / "summary.json", args.skip_existing)
    return prepared


def run_methods(args, methods, samples):
    root = Path(args.output_root)
    prepared = root / "prepared"
    depth_m = prepared / "all" / "depth_m"
    raw9 = prepared / "all" / "raw9_chw"
    method_root = root / "methods"
    sample_args = [str(s) for s in samples]

    common_depth = [
        "--hole_depth_threshold",
        str(args.hole_depth_threshold),
        "--valid_min_depth",
        str(args.valid_min_depth),
        "--valid_max_depth",
        str(args.valid_max_depth),
    ]
    common_raw = [
        "--hole_depth_threshold",
        str(args.hole_depth_threshold),
        "--valid_min_depth",
        str(args.valid_min_depth),
        "--valid_max_depth",
        str(args.valid_max_depth),
        "--amplitude_mode",
        "iq6",
        "--hole_amplitude_mode",
        "keep_all",
    ]
    gated_raw = []
    if args.gated_fill:
        gated_raw = [
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
            gated_raw.append("--gate_keep_border_anchor")
    repair_raw = [
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
        repair_raw.append("--preserve_holes_as_nan")
    plane_raw = []
    if args.plane_fill:
        plane_raw = [
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

    if "depth_only" in methods:
        out = method_root / "depth_only_flow"
        cmd = [
            args.python_bin,
            "scripts/flow/infer_real_depth_flow.py",
            "--input_dir",
            depth_m,
            "--checkpoint",
            "output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt",
            "--output_dir",
            out,
            "--sampling_mode",
            "endpoint",
        ] + common_depth
        run_cmd(cmd, Path.cwd(), out / "summary.json", args.skip_existing)

    if "raw9_satclip" in methods:
        out = method_root / "raw9_satclip"
        cmd = [
            args.python_bin,
            "scripts/flow/infer_real_raw9_flow.py",
            "--raw_dir",
            raw9,
            "--depth_dir",
            depth_m,
            "--checkpoint",
            "output/real_raw9_flow_finetune_overexposure_satclip_from_generalized_e20/best.pt",
            "--output_dir",
            out,
            "--sampling_mode",
            "endpoint",
        ] + common_raw + gated_raw + repair_raw + plane_raw
        run_cmd(cmd, Path.cwd(), out / "summary.json", args.skip_existing)

    if "raw9_realholes" in methods:
        out = method_root / "raw9_realholes"
        cmd = [
            args.python_bin,
            "scripts/flow/infer_real_raw9_flow.py",
            "--raw_dir",
            raw9,
            "--depth_dir",
            depth_m,
            "--checkpoint",
            "output/real_raw9_flow_finetune_iq6_realholes_e40_m8/best.pt",
            "--output_dir",
            out,
            "--sampling_mode",
            "endpoint",
        ] + common_raw + gated_raw + repair_raw + plane_raw
        run_cmd(cmd, Path.cwd(), out / "summary.json", args.skip_existing)

    if "after_synth" in methods:
        out = method_root / "after_synth"
        cmd = [
            args.python_bin,
            "scripts/flow/infer_real_raw9_flow.py",
            "--raw_dir",
            raw9,
            "--depth_dir",
            depth_m,
            "--checkpoint",
            "output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/best.pt",
            "--output_dir",
            out,
            "--sampling_mode",
            "endpoint",
        ] + common_raw + gated_raw + repair_raw + plane_raw + [
            "--hole_mask_mode",
            "amp_speckle_cleaned",
            "--clean_dilate",
            "1",
            "--speckle_link_radius",
            "2",
            "--split_added_fill",
            "--split_added_mode",
            "anchor_ns",
            "--split_added_inpaint_radius",
            "5",
        ]
        run_cmd(cmd, Path.cwd(), out / "summary.json", args.skip_existing)

    if "propagation" in methods:
        out = method_root / "propagation"
        cmd = [
            args.python_bin,
            "scripts/flow/infer_real_raw9_propagation_refine.py",
            "--raw_dir",
            raw9,
            "--depth_dir",
            depth_m,
            "--checkpoint",
            "output/real_raw9_propagation_refine_iq6_e40/best.pt",
            "--output_dir",
            out,
            "--amplitude_mode",
            "iq6",
            "--hole_depth_threshold",
            str(args.hole_depth_threshold),
            "--valid_min_depth",
            str(args.valid_min_depth),
            "--valid_max_depth",
            str(args.valid_max_depth),
            "--hole_mask_mode",
            "amp_speckle_cleaned",
            "--clean_dilate",
            "1",
            "--speckle_link_radius",
            "2",
            "--split_added_fill",
            "--split_added_mode",
            "anchor_ns",
            "--split_added_inpaint_radius",
            "5",
        ]
        run_cmd(cmd, Path.cwd(), out / "summary.json", args.skip_existing)

    if "propainter" in methods:
        external = root / "external_inpaint" / "external_inputs"
        cmd = [
            args.python_bin,
            "scripts/export_new_capture_external_inpainting.py",
            "--depth_dir",
            depth_m,
            "--output_dir",
            external,
            "--hole_depth_threshold",
            str(args.hole_depth_threshold),
            "--valid_min_depth",
            str(args.valid_min_depth),
            "--valid_max_depth",
            str(args.valid_max_depth),
            "--percentile_min",
            "1",
            "--percentile_max",
            "99",
        ]
        if samples:
            cmd += ["--samples"] + sample_args
        run_cmd(cmd, Path.cwd(), external / "export_summary.json", args.skip_existing)

        first_depth = np.load(next(depth_m.glob("*.npy"))).astype(np.float32)
        frame_count = max(1, len(sample_args) if sample_args else len(list(depth_m.glob("*.npy"))))
        neighbor_length = min(int(args.propainter_neighbor_length), frame_count)
        ref_stride = min(int(args.propainter_ref_stride), frame_count)
        subvideo_length = min(int(args.propainter_subvideo_length), frame_count)
        height = int(args.propainter_height or first_depth.shape[0])
        width = int(args.propainter_width or first_depth.shape[1])
        propainter_out = root / "external_inpaint" / "propainter_run"
        cmd = [
            args.python_bin,
            "run_external_inpainting_far_pic.py",
            "run-propainter",
            "--case",
            root / "external_inpaint",
            "--output_dir",
            propainter_out,
            "--height",
            str(height),
            "--width",
            str(width),
            "--mask_dilation",
            "0",
            "--neighbor_length",
            str(neighbor_length),
            "--ref_stride",
            str(ref_stride),
            "--subvideo_length",
            str(subvideo_length),
            "--decode",
        ]
        run_cmd(cmd, Path.cwd(), propainter_out / "decode_summary.json", args.skip_existing)

    if "depthcad_depth_gray" in methods:
        out = method_root / "depthcad_depth_gray"
        cmd = [
            args.python_bin,
            "scripts/run_depthcad_depth_hole_baseline.py",
            "--depth_dir",
            depth_m,
            "--output_dir",
            out,
            "--checkpoint",
            "output/depthcad",
            "--pretrained_model_name_or_path",
            "stabilityai/stable-diffusion-2-1",
            "--num_inference_steps",
            str(args.depthcad_steps),
            "--infer_size",
            str(args.depthcad_infer_size),
            "--depth_min",
            str(args.valid_min_depth),
            "--depth_max",
            str(args.valid_max_depth),
            "--hole_depth_threshold",
            str(args.hole_depth_threshold),
            "--local_files_only",
        ]
        if samples:
            cmd += ["--samples"] + sample_args
        if args.allow_depthcad_cpu:
            cmd += ["--allow_cpu"]
        run_cmd(cmd, Path.cwd(), out / "summary.json", args.skip_existing)


def method_specs(output_root):
    root = Path(output_root)
    return [
        ("ns_anchor", "NS anchor", root / "methods/raw9_satclip/anchor/{s}_anchor.npy"),
        ("depth_only_flow", "depth-only flow", root / "methods/depth_only_flow/hole_only/{s}_hole_only.npy"),
        ("raw9_satclip", "raw9 satclip", root / "methods/raw9_satclip/hole_only/{s}_hole_only.npy"),
        (
            "raw9_satclip_gated",
            "raw9 satclip gated",
            root / "methods/raw9_satclip/gated_hole_only/{s}_gated_hole_only.npy",
        ),
        (
            "raw9_satclip_plane",
            "raw9 satclip plane",
            root / "methods/raw9_satclip/plane_hole_only/{s}_plane_hole_only.npy",
        ),
        ("raw9_realholes", "raw9 realholes", root / "methods/raw9_realholes/hole_only/{s}_hole_only.npy"),
        (
            "raw9_realholes_gated",
            "raw9 realholes gated",
            root / "methods/raw9_realholes/gated_hole_only/{s}_gated_hole_only.npy",
        ),
        (
            "raw9_realholes_plane",
            "raw9 realholes plane",
            root / "methods/raw9_realholes/plane_hole_only/{s}_plane_hole_only.npy",
        ),
        ("after_synth_split", "after-synth split", root / "methods/after_synth/split_hole_only/{s}_split_hole_only.npy"),
        (
            "after_synth_gated",
            "after-synth gated",
            root / "methods/after_synth/gated_hole_only/{s}_gated_hole_only.npy",
        ),
        (
            "after_synth_plane",
            "after-synth plane",
            root / "methods/after_synth/plane_hole_only/{s}_plane_hole_only.npy",
        ),
        ("propagation_split", "propagation split", root / "methods/propagation/split_hole_only/{s}_split_hole_only.npy"),
        ("propainter", "ProPainter", root / "external_inpaint/propainter_run/restored_by_stem/{s}_propainter_restored.npy"),
        ("depthcad_depth_gray", "DepthCAD depth-gray", root / "methods/depthcad_depth_gray/hole_only/{s}_depthcad_depth_hole_only.npy"),
    ]


def load_array(path):
    path = Path(str(path))
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def finite_valid(arr, vmin, vmax):
    return np.isfinite(arr) & (arr >= float(vmin)) & (arr <= float(vmax))


def add_panel(ax, title, image, cmap, vmin, vmax):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def build_comparison(args, selected):
    root = Path(args.output_root)
    depth_dir = root / "prepared" / "all" / "depth_m"
    out_dir = root / "comparison"
    fig_dir = out_dir / "figures"
    mkdir(fig_dir)

    specs = method_specs(root)
    rows = []
    figure_paths = []
    for sample in selected:
        raw = load_array(depth_dir / f"{sample}.npy")
        if raw is None:
            continue
        threshold = load_array(root / "methods/raw9_satclip/hole_mask" / f"{sample}_hole_mask.npy")
        if threshold is None:
            threshold = (~np.isfinite(raw)) | (raw <= float(args.hole_depth_threshold))
        else:
            threshold = threshold.astype(bool)
        cleaned = load_array(root / "methods/after_synth/hole_mask" / f"{sample}_hole_mask.npy")
        cleaned = threshold if cleaned is None else cleaned.astype(bool)

        outputs = []
        for key, title, template in specs:
            arr = load_array(Path(str(template).format(s=sample)))
            if arr is not None:
                outputs.append((key, title, arr))
        anchor = next((arr for key, _, arr in outputs if key == "ns_anchor"), None)
        if anchor is None:
            anchor = raw

        panels = [
            ("raw depth", raw, "viridis", args.depth_vis_min, args.depth_vis_max),
            ("threshold hole", threshold.astype(np.float32), "gray", 0.0, 1.0),
            ("cleaned hole", cleaned.astype(np.float32), "gray", 0.0, 1.0),
        ]
        panels += [
            (title, arr, "viridis", args.depth_vis_min, args.depth_vis_max)
            for _key, title, arr in outputs
        ]
        cols = 4
        rows_fig = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows_fig, cols, figsize=(4.6 * cols, 3.6 * rows_fig), constrained_layout=True)
        axes = np.asarray(axes).ravel()
        last_depth_im = None
        for ax, panel in zip(axes, panels):
            im = add_panel(ax, *panel)
            if panel[2] != "gray":
                last_depth_im = im
        for ax in axes[len(panels):]:
            ax.axis("off")
        if last_depth_im is not None:
            fig.colorbar(last_depth_im, ax=axes.tolist(), fraction=0.025, pad=0.01, label="depth (m)")
        fig.suptitle(f"real capture method comparison: {sample}", fontsize=14)
        fig_path = fig_dir / f"{sample}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figure_paths.append(str(fig_path))

        valid_raw = finite_valid(raw, args.valid_min_depth, args.valid_max_depth)
        for key, title, arr in outputs:
            valid_out = finite_valid(arr, args.valid_min_depth, args.valid_max_depth)
            valid_change = np.abs(arr[valid_raw] - raw[valid_raw])
            valid_change = valid_change[np.isfinite(valid_change)]
            diff_anchor = np.abs(arr - anchor)
            diff_anchor_threshold = diff_anchor[threshold & np.isfinite(diff_anchor)]
            threshold_vals = arr[threshold & valid_out]
            rows.append(
                {
                    "sample": sample,
                    "method": key,
                    "title": title,
                    "raw_valid_ratio": float(valid_raw.mean()),
                    "threshold_hole_ratio": float(threshold.mean()),
                    "cleaned_hole_ratio": float(cleaned.mean()),
                    "threshold_fill_ratio": float(valid_out[threshold].mean()) if threshold.any() else None,
                    "cleaned_fill_ratio": float(valid_out[cleaned].mean()) if cleaned.any() else None,
                    "mean_abs_change_on_raw_valid_m": float(valid_change.mean()) if valid_change.size else None,
                    "mean_abs_vs_ns_anchor_threshold_hole_m": (
                        float(diff_anchor_threshold.mean()) if diff_anchor_threshold.size else None
                    ),
                    "filled_threshold_median_m": (
                        float(np.median(threshold_vals)) if threshold_vals.size else None
                    ),
                }
            )

    summary = {}
    for key, title, _template in specs:
        subset = [row for row in rows if row["method"] == key]
        if not subset:
            continue
        def finite_mean(key):
            values = [
                r[key]
                for r in subset
                if r.get(key) is not None and np.isfinite(float(r[key]))
            ]
            return float(np.mean(values)) if values else None

        summary[key] = {
            "title": title,
            "num_samples": len(subset),
            "mean_threshold_fill_ratio": finite_mean("threshold_fill_ratio"),
            "mean_cleaned_fill_ratio": finite_mean("cleaned_fill_ratio"),
            "mean_abs_change_on_raw_valid_m": finite_mean("mean_abs_change_on_raw_valid_m"),
            "mean_abs_vs_ns_anchor_threshold_hole_m": finite_mean("mean_abs_vs_ns_anchor_threshold_hole_m"),
            "mean_filled_threshold_median_m": finite_mean("filled_threshold_median_m"),
        }

    with (out_dir / "per_sample_metrics.csv").open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "samples": selected,
                "figures_dir": str(fig_dir),
                "figures": figure_paths,
                "method_summary": summary,
            },
            f,
            indent=2,
        )
    with (out_dir / "figures.txt").open("w") as f:
        for path in figure_paths:
            f.write(path + "\n")
    print(json.dumps({"figures_dir": str(fig_dir), "method_summary": summary}, indent=2))


def main():
    args = parse_args()
    methods = expand_methods(args.methods)
    depth_dir, iq_dir = resolve_dirs(args)
    pairs = load_pairs(depth_dir, iq_dir)
    selected = choose_samples(pairs, args)
    input_root, copied = stage_selected_inputs(pairs, selected, args.output_root)
    write_selection_report(args.output_root, depth_dir, iq_dir, selected, copied, pairs)

    print(f"Source depth: {depth_dir}")
    print(f"Source IQ:    {iq_dir}")
    print(f"Selected:     {' '.join(selected)}")
    print(f"Methods:      {' '.join(methods)}")

    if not args.no_prepare:
        prepare_data(args, input_root)
    run_methods(args, methods, selected)
    if not args.no_compare:
        build_comparison(args, selected)


if __name__ == "__main__":
    main()
