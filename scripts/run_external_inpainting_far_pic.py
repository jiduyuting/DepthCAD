#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# Local ProPainter checkout on this machine.
DEFAULT_PROPAINTER_REPO = "/data/pre_student/cyx/GIGA/ProPainter"
DEFAULT_PROPAINTER_PYTHON = "/home/lab507/anaconda3/envs/SVDC/bin/python"
DEFAULT_RAD_REPO = "/data/pre_student/GJ/RAD"
DEFAULT_RAD_PYTHON = "/home/lab507/anaconda3/envs/depthcad_zimage/bin/python"


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    mkdir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def resolve_external_inputs(path):
    root = Path(path).resolve()
    if (root / "external_inputs" / "export" / "frames").is_dir():
        case_dir = root
        external_dir = root / "external_inputs"
    elif (root / "export" / "frames").is_dir():
        external_dir = root
        case_dir = root.parent if root.name == "external_inputs" else root
    else:
        fail(
            "expected a benchmark case dir containing external_inputs/export/frames, "
            "or an external_inputs dir containing export/frames"
        )

    export_dir = external_dir / "export"
    frames_dir = export_dir / "frames"
    masks_dir = export_dir / "masks"
    mapping_path = external_dir / "source_mapping.json"
    meta_path = export_dir / "depth_meta.json"
    for p in [frames_dir, masks_dir, mapping_path, meta_path]:
        if not p.exists():
            fail(f"missing required external input: {p}")

    return {
        "case_dir": case_dir,
        "external_dir": external_dir,
        "export_dir": export_dir,
        "frames_dir": frames_dir,
        "masks_dir": masks_dir,
        "mapping_path": mapping_path,
        "meta_path": meta_path,
    }


def quote_cmd(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def build_propainter_cmd(args, paths):
    repo = Path(args.propainter_repo).resolve()
    script = repo / "inference_propainter.py"
    if not script.exists():
        fail(f"ProPainter entry script not found: {script}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else paths["case_dir"] / "propainter_run"
    cmd = [
        args.python,
        str(script),
        "--video",
        str(paths["frames_dir"]),
        "--mask",
        str(paths["masks_dir"]),
        "--output",
        str(output_dir),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--mask_dilation",
        str(args.mask_dilation),
        "--save_frames",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.neighbor_length is not None:
        cmd += ["--neighbor_length", str(args.neighbor_length)]
    if args.ref_stride is not None:
        cmd += ["--ref_stride", str(args.ref_stride)]
    if args.subvideo_length is not None:
        cmd += ["--subvideo_length", str(args.subvideo_length)]
    return repo, output_dir, cmd


def run_propainter(args):
    paths = resolve_external_inputs(args.case)
    repo, output_dir, cmd = build_propainter_cmd(args, paths)
    mkdir(output_dir)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = args.mplconfigdir
    mkdir(args.mplconfigdir)

    print("Running ProPainter:")
    print(f"  cd {repo}")
    print(f"  MPLCONFIGDIR={args.mplconfigdir} {quote_cmd(cmd)}")
    result = subprocess.run(cmd, cwd=str(repo), env=env, check=False)
    if result.returncode != 0:
        mapping = load_json(paths["mapping_path"])["frame_mapping"]
        try:
            frames_dir = find_propainter_frames(output_dir, paths["frames_dir"], len(mapping))
            print(
                "WARNING: ProPainter returned a non-zero exit code after writing frames. "
                "This commonly happens when imageio cannot write mp4. "
                f"Found {count_pngs(frames_dir)} output PNG frames at {frames_dir}."
            )
        except SystemExit:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        if not args.decode:
            raise subprocess.CalledProcessError(result.returncode, cmd)

    if args.decode:
        decode_args = argparse.Namespace(
            case=args.case,
            output_dir=str(output_dir),
            frames_dir=None,
            decode_name=args.decode_name,
            clip=not args.no_clip,
        )
        decode_propainter(decode_args)


def print_propainter_command(args):
    paths = resolve_external_inputs(args.case)
    repo, _output_dir, cmd = build_propainter_cmd(args, paths)
    print(f"cd {shlex.quote(str(repo))}")
    print(f"MPLCONFIGDIR={shlex.quote(args.mplconfigdir)} {quote_cmd(cmd)}")


def count_pngs(path):
    return len(sorted(Path(path).glob("*.png")))


def find_propainter_frames(output_dir, input_frames_dir, expected_count):
    output_dir = Path(output_dir).resolve()
    input_name = Path(input_frames_dir).name
    candidates = [
        output_dir / input_name / "frames",
        output_dir / "frames" / "frames",
        output_dir / "frames",
        output_dir,
    ]
    for candidate in candidates:
        if (candidate / "0000.png").exists() and count_pngs(candidate) >= expected_count:
            return candidate

    matches = []
    for p in output_dir.rglob("0000.png"):
        parent = p.parent
        if count_pngs(parent) >= expected_count:
            matches.append(parent)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches = sorted(matches, key=lambda x: len(str(x)))
        return matches[0]
    fail(f"could not find ProPainter output frames under {output_dir}")


def read_gray_png(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        fail(f"failed to read image: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img.astype(np.float32)


def resize_like(arr, shape, is_mask=False):
    if arr.shape == shape:
        return arr
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=interp)


def decode_propainter(args):
    paths = resolve_external_inputs(args.case)
    mapping = load_json(paths["mapping_path"])["frame_mapping"]
    meta = load_json(paths["meta_path"])
    expected_count = len(mapping)
    if args.frames_dir:
        frames_dir = Path(args.frames_dir).resolve()
    else:
        if not args.output_dir:
            fail("--output_dir is required unless --frames_dir is given")
        frames_dir = find_propainter_frames(args.output_dir, paths["frames_dir"], expected_count)

    lo = float(meta["depth_min"])
    hi = float(meta["depth_max"])
    if hi <= lo:
        fail(f"invalid depth range in {paths['meta_path']}: {lo}..{hi}")

    depth_dir = Path(meta["source_depth_npy"])
    mask_dir = Path(meta["mask_source"])
    out_root = Path(args.output_dir).resolve() if args.output_dir else frames_dir.parent
    out_by_index = out_root / "restored_by_index"
    out_by_stem = out_root / "restored_by_stem"
    mkdir(out_by_index)
    mkdir(out_by_stem)
    clip_decoded = getattr(args, "clip", not getattr(args, "no_clip", False))

    stack = []
    rows = []
    for idx, item in enumerate(mapping):
        frame_path = frames_dir / f"{idx:04d}.png"
        corrupted_path = depth_dir / f"{idx:04d}.npy"
        mask_path = mask_dir / f"{idx:04d}.npy"
        for p in [frame_path, corrupted_path, mask_path]:
            if not p.exists():
                fail(f"missing decode input: {p}")

        gray = read_gray_png(frame_path)
        decoded = gray / 255.0 * (hi - lo) + lo
        corrupted = np.load(corrupted_path).astype(np.float32)
        mask = np.load(mask_path).astype(bool)
        decoded = resize_like(decoded, corrupted.shape, is_mask=False).astype(np.float32)
        mask = resize_like(mask.astype(np.uint8), corrupted.shape, is_mask=True).astype(bool)
        if clip_decoded:
            decoded = np.clip(decoded, lo, hi)

        restored = corrupted.copy()
        restored[mask] = decoded[mask]
        restored = restored.astype(np.float32)

        stem = item.get("source_stem", f"{idx:04d}")
        index_path = out_by_index / f"{idx:04d}.npy"
        stem_path = out_by_stem / f"{stem}_{args.decode_name}.npy"
        np.save(index_path, restored)
        np.save(stem_path, restored)
        stack.append(restored)

        hole_vals = restored[mask & np.isfinite(restored)]
        rows.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "frame_png": str(frame_path),
                "mask_ratio": float(mask.mean()),
                "output_by_index": str(index_path),
                "output_by_stem": str(stem_path),
                "hole_min": float(np.min(hole_vals)) if hole_vals.size else None,
                "hole_median": float(np.median(hole_vals)) if hole_vals.size else None,
                "hole_max": float(np.max(hole_vals)) if hole_vals.size else None,
            }
        )

    stack_path = out_root / "restored_depth.npy"
    np.save(stack_path, np.stack(stack, axis=0).astype(np.float32))
    summary = {
        "case": str(paths["case_dir"]),
        "external_inputs": str(paths["external_dir"]),
        "propainter_frames": str(frames_dir),
        "depth_meta": str(paths["meta_path"]),
        "depth_min": lo,
        "depth_max": hi,
        "decode_rule": "gray/255*(depth_max-depth_min)+depth_min, merged only inside mask",
        "outside_mask_policy": "kept exactly from external_inputs/depth_npy",
        "stack_path": str(stack_path),
        "restored_by_index": str(out_by_index),
        "restored_by_stem": str(out_by_stem),
        "frames": rows,
    }
    save_json(out_root / "decode_summary.json", summary)
    print(f"Decoded {len(rows)} ProPainter frames")
    print(f"  frames: {frames_dir}")
    print(f"  npy by stem: {out_by_stem}")
    print(f"  stack: {stack_path}")


def standard_output_path(case_dir, method, stem):
    return Path(case_dir) / "outputs" / method / f"{stem}_{method}.npy"


def standardize_method(args):
    paths = resolve_external_inputs(args.case)
    mapping = load_json(paths["mapping_path"])["frame_mapping"]
    source_dir = Path(args.source_dir).resolve()
    out_dir = paths["case_dir"] / "outputs" / args.method
    mkdir(out_dir)

    copied = []
    missing = []
    for item in mapping:
        idx = int(item["frame_index"])
        stem = item.get("source_stem", f"{idx:04d}")
        candidates = [
            source_dir / f"{stem}_{args.source_suffix}.npy",
            source_dir / f"{stem}_{args.method}.npy",
            source_dir / f"{stem}.npy",
            source_dir / f"{idx:04d}.npy",
        ]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            missing.append({"frame_index": idx, "source_stem": stem, "tried": [str(p) for p in candidates]})
            continue
        arr = np.load(src).astype(np.float32)
        dst = standard_output_path(paths["case_dir"], args.method, stem)
        np.save(dst, arr)
        copied.append({"frame_index": idx, "source_stem": stem, "source": str(src), "output": str(dst)})

    summary = {
        "case": str(paths["case_dir"]),
        "method": args.method,
        "source_dir": str(source_dir),
        "output_dir": str(out_dir),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied": copied,
        "missing": missing,
    }
    save_json(paths["case_dir"] / "outputs" / args.method / "standardize_summary.json", summary)
    print(f"Standardized {len(copied)} frames for {args.method}: {out_dir}")
    if missing:
        print(f"WARNING: missing {len(missing)} frames; see standardize_summary.json")


def default_rad_result_dir(args):
    dataset = args.dataset_name
    exp_name = args.exp_name + f"Local-{dataset.split('/')[-1]}-{args.ddpm_num_steps}-{args.ddpm_mask_num_steps}"
    root = (
        Path(args.rad_repo).resolve()
        / f"ddpm-model-{dataset}-{args.resolution}"
        / f"{exp_name}_lora_rank_{args.rank}"
    )
    step = Path(args.resume_from_checkpoint).name.split("-")[-1]
    return root / f"Inpainting images_FID_{step}"


def decode_rad(args):
    paths = resolve_external_inputs(args.case)
    mapping = load_json(paths["mapping_path"])["frame_mapping"]
    meta = load_json(paths["meta_path"])
    result_dir = Path(args.rad_output_dir).resolve() if args.rad_output_dir else default_rad_result_dir(args)
    frames_dir = result_dir / args.mask_type / "inpainted"
    if not frames_dir.is_dir():
        fail(f"RAD inpainted dir not found: {frames_dir}")

    lo = float(meta["depth_min"])
    hi = float(meta["depth_max"])
    if hi <= lo:
        fail(f"invalid depth range in {paths['meta_path']}: {lo}..{hi}")

    depth_dir = Path(meta["source_depth_npy"])
    mask_dir = Path(meta["mask_source"])
    out_dir = paths["case_dir"] / "outputs" / args.method
    mkdir(out_dir)

    rows = []
    stack = []
    for item in mapping:
        idx = int(item["frame_index"])
        stem = item.get("source_stem", f"{idx:04d}")
        frame_path = frames_dir / f"{idx:04d}.png"
        corrupted_path = depth_dir / f"{idx:04d}.npy"
        mask_path = mask_dir / f"{idx:04d}.npy"
        for p in [frame_path, corrupted_path, mask_path]:
            if not p.exists():
                fail(f"missing RAD decode input: {p}")

        gray = read_gray_png(frame_path)
        decoded = gray / 255.0 * (hi - lo) + lo
        corrupted = np.load(corrupted_path).astype(np.float32)
        mask = np.load(mask_path).astype(bool)
        decoded = resize_like(decoded, corrupted.shape, is_mask=False).astype(np.float32)
        mask = resize_like(mask.astype(np.uint8), corrupted.shape, is_mask=True).astype(bool)
        if args.clip:
            decoded = np.clip(decoded, lo, hi)

        restored = corrupted.copy()
        restored[mask] = decoded[mask]
        restored = restored.astype(np.float32)
        out_path = standard_output_path(paths["case_dir"], args.method, stem)
        np.save(out_path, restored)
        stack.append(restored)

        hole_vals = restored[mask & np.isfinite(restored)]
        rows.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "rad_png": str(frame_path),
                "output": str(out_path),
                "mask_ratio": float(mask.mean()),
                "hole_min": float(np.min(hole_vals)) if hole_vals.size else None,
                "hole_median": float(np.median(hole_vals)) if hole_vals.size else None,
                "hole_max": float(np.max(hole_vals)) if hole_vals.size else None,
            }
        )

    stack_path = out_dir / f"{args.method}_stack.npy"
    np.save(stack_path, np.stack(stack, axis=0).astype(np.float32))
    summary = {
        "case": str(paths["case_dir"]),
        "method": args.method,
        "rad_result_dir": str(result_dir),
        "rad_frames_dir": str(frames_dir),
        "mask_type": args.mask_type,
        "depth_meta": str(paths["meta_path"]),
        "depth_min": lo,
        "depth_max": hi,
        "decode_rule": "gray/255*(depth_max-depth_min)+depth_min, resized to depth shape, merged only inside mask",
        "outside_mask_policy": "kept exactly from external_inputs/depth_npy",
        "output_dir": str(out_dir),
        "stack_path": str(stack_path),
        "frames": rows,
    }
    save_json(out_dir / "decode_summary.json", summary)
    print(f"Decoded {len(rows)} RAD frames for {args.method}")
    print(f"  RAD frames: {frames_dir}")
    print(f"  npy output: {out_dir}")
    print(f"  stack: {stack_path}")


def copy_or_resize_png(src, dst, resolution, is_mask):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        fail(f"failed to read image: {src}")
    if resolution is not None:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        img = cv2.resize(img, (resolution, resolution), interpolation=interp)
    if is_mask:
        if img.ndim == 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        img = (img > 127).astype(np.uint8) * 255
    mkdir(Path(dst).parent)
    if not cv2.imwrite(str(dst), img):
        fail(f"failed to write image: {dst}")


def prepare_rad(args):
    paths = resolve_external_inputs(args.case)
    mapping = load_json(paths["mapping_path"])["frame_mapping"]
    out_dir = Path(args.output_dir).resolve() if args.output_dir else paths["case_dir"] / "rad_val"
    mask_types = args.mask_types
    resolution = None if args.keep_size else int(args.resolution)

    for mask_type in mask_types:
        original_dir = out_dir / mask_type / "original"
        mask_dir = out_dir / mask_type / "mask"
        mkdir(original_dir)
        mkdir(mask_dir)
        for idx, _item in enumerate(mapping):
            frame_src = paths["frames_dir"] / f"{idx:04d}.png"
            mask_src = paths["masks_dir"] / f"{idx:04d}.png"
            copy_or_resize_png(frame_src, original_dir / f"{idx:04d}.png", resolution, is_mask=False)
            copy_or_resize_png(mask_src, mask_dir / f"{idx:04d}.png", resolution, is_mask=True)

    save_json(
        out_dir / "rad_val_meta.json",
        {
            "source_external_inputs": str(paths["external_dir"]),
            "mask_types": mask_types,
            "frame_count": len(mapping),
            "resolution": "kept original" if resolution is None else [resolution, resolution],
            "note": "RAD inpaint.py loops over thick/box/extreme, so the same masks are duplicated by default.",
        },
    )
    print(f"Prepared RAD validation data: {out_dir}")
    print(f"  mask_types: {', '.join(mask_types)}")


def print_rad_command(args):
    paths = resolve_external_inputs(args.case)
    val_dir = Path(args.val_data_path).resolve() if args.val_data_path else paths["case_dir"] / "rad_val"
    repo = Path(args.rad_repo).resolve()
    script = repo / "examples" / "unconditional_image_generation" / "inpaint.py"
    if not script.exists():
        fail(f"RAD entry script not found: {script}")

    cmd = [
        args.python,
        str(script),
        "--val_data_path",
        str(val_dir),
        "--dataset_name",
        args.dataset_name,
        "--pretrained_model_name_or_path",
        args.pretrained_model_name_or_path,
        "--resume_from_checkpoint",
        args.resume_from_checkpoint,
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        str(args.train_batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--num_samples",
        str(args.num_samples),
        "--ddpm_num_inference_steps",
        str(args.ddpm_num_inference_steps),
        "--rank",
        str(args.rank),
        "--exp_name",
        args.exp_name,
    ]
    print(f"cd {shlex.quote(str(repo))}")
    print(f"PYTHONPATH={shlex.quote(str(repo / 'src'))} {quote_cmd(cmd)}")
    exp_name = args.exp_name + f"Local-{args.dataset_name.split('/')[-1]}-2000-1000"
    computed_output = f"ddpm-model-{args.dataset_name}-{args.resolution}/{exp_name}_lora_rank_{args.rank}"
    print()
    print("RAD checkpoint lookup note:")
    print(f"  inpaint.py ignores --output_dir and loads from: {computed_output}/{Path(args.resume_from_checkpoint).name}")
    print("  Put the official RAD checkpoint there, or patch RAD inpaint.py before running.")


def add_common_case(parser):
    parser.add_argument(
        "--case",
        default="output/far_pic_benchmark/bad_depth_mask_v1",
        help="Benchmark case dir or external_inputs dir.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Adapters for running image/video inpainting baselines on far_pic depth PNG exports."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("propainter-command")
    add_common_case(p)
    p.add_argument("--propainter_repo", default=DEFAULT_PROPAINTER_REPO)
    p.add_argument("--python", default=DEFAULT_PROPAINTER_PYTHON)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--mask_dilation", type=int, default=0)
    p.add_argument("--neighbor_length", type=int, default=None)
    p.add_argument("--ref_stride", type=int, default=None)
    p.add_argument("--subvideo_length", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--mplconfigdir", default="/tmp/mpl_propainter")
    p.set_defaults(func=print_propainter_command)

    p = sub.add_parser("run-propainter")
    add_common_case(p)
    p.add_argument("--propainter_repo", default=DEFAULT_PROPAINTER_REPO)
    p.add_argument("--python", default=DEFAULT_PROPAINTER_PYTHON)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--mask_dilation", type=int, default=0)
    p.add_argument("--neighbor_length", type=int, default=None)
    p.add_argument("--ref_stride", type=int, default=None)
    p.add_argument("--subvideo_length", type=int, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--mplconfigdir", default="/tmp/mpl_propainter")
    p.add_argument("--decode", action="store_true")
    p.add_argument("--decode_name", default="propainter_restored")
    p.add_argument("--no_clip", action="store_true")
    p.set_defaults(func=run_propainter)

    p = sub.add_parser("decode-propainter")
    add_common_case(p)
    p.add_argument("--output_dir", required=True, help="ProPainter output dir, e.g. case/propainter_run.")
    p.add_argument("--frames_dir", default=None, help="Optional direct path to ProPainter output PNG frames.")
    p.add_argument("--decode_name", default="propainter_restored")
    p.add_argument("--no_clip", action="store_true")
    p.set_defaults(func=decode_propainter)

    p = sub.add_parser("standardize-method")
    add_common_case(p)
    p.add_argument("--method", required=True, help="Standard method name under outputs/<method>.")
    p.add_argument("--source_dir", required=True, help="Directory containing source .npy files.")
    p.add_argument(
        "--source_suffix",
        default="",
        help="Suffix in source files, e.g. propainter_restored for <stem>_propainter_restored.npy.",
    )
    p.set_defaults(func=standardize_method)

    p = sub.add_parser("prepare-rad")
    add_common_case(p)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--keep_size", action="store_true")
    p.add_argument("--mask_types", nargs="+", default=["thick", "box", "extreme"])
    p.set_defaults(func=prepare_rad)

    p = sub.add_parser("rad-command")
    add_common_case(p)
    p.add_argument("--rad_repo", default=DEFAULT_RAD_REPO)
    p.add_argument("--python", default=DEFAULT_RAD_PYTHON)
    p.add_argument("--val_data_path", default=None)
    p.add_argument("--dataset_name", default="merkol/ffhq-256")
    p.add_argument("--pretrained_model_name_or_path", default="xutongda/adm_ffhq_256x256")
    p.add_argument("--resume_from_checkpoint", default="checkpoint-300000")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=23)
    p.add_argument("--ddpm_num_inference_steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--exp_name", default="")
    p.set_defaults(func=print_rad_command)

    p = sub.add_parser("decode-rad")
    add_common_case(p)
    p.add_argument("--rad_repo", default=DEFAULT_RAD_REPO)
    p.add_argument("--rad_output_dir", default=None, help="RAD 'Inpainting images_FID_*' dir. Guessed if omitted.")
    p.add_argument("--dataset_name", default="pcuenq/lsun-bedrooms")
    p.add_argument("--resume_from_checkpoint", default="checkpoint-300000")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--exp_name", default="")
    p.add_argument("--ddpm_num_steps", type=int, default=2000)
    p.add_argument("--ddpm_mask_num_steps", type=int, default=1000)
    p.add_argument("--mask_type", default="thick", choices=["thick", "box", "extreme"])
    p.add_argument("--method", default="rad_lsun_bedroom")
    p.add_argument("--no_clip", dest="clip", action="store_false")
    p.set_defaults(func=decode_rad, clip=True)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
