"""
Approximate BayesToF baseline for DepthCAD experiments.

The paper "BayesToF: multiresolution denoising of indirect time-of-flight
distance maps" derives a full Bayesian MMSE estimator from 4-tap IToF photon
counts. DepthCAD data in this repo usually stores 6-channel multi-frequency
correlation/IQ tensors instead of tap counts J0..J3, so this script implements
the paper's two-measurement spirit as an engineering baseline:

  * split IQ as [I30,Q30,I40,Q40,I58,Q58]
  * normalize each I/Q pair to its phase unit vector
  * run multiresolution wavelet coring with thresholds increased at low amplitude
  * restore IQ direction and use DepthEstimator for multi-frequency depth

This is intentionally named "approx" because it is not a faithful reproduction
of Eq. 28 full BayesToF. A faithful reproduction needs raw tap counts and the
paper's GSM/EM numerical marginalization.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pywt

from depth_estimator import DepthEstimator
from inference_depth_postprocess import opencv_depth_inpaint


CHANNELS = ["A", "B", "C", "D", "E", "F"]
PAIR_INDICES = [(0, 1), (2, 3), (4, 5)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an approximate BayesToF-style wavelet denoising baseline on "
            "DepthCAD 6-channel IQ data."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cache_dir",
        type=Path,
        help="Depth completion cache containing .npz files with noisy_iq.",
    )
    source.add_argument(
        "--iq_dir",
        type=Path,
        help="PBRT-style IQ directory with scene/idx/*_A..F.npy files.",
    )
    parser.add_argument(
        "--sample_list",
        type=Path,
        default=None,
        help="Optional text file of cache .npz paths or scene/idx/sample ids.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional sample ids, e.g. pavilion/1/15 or 15.",
    )
    parser.add_argument(
        "--cache_iq_key",
        type=str,
        default="noisy_iq",
        help="IQ key to read when using --cache_dir.",
    )
    parser.add_argument(
        "--gt_depth_dir",
        type=Path,
        default=Path("/data/pre_student/hcy/pbrt/gt_depth"),
        help="GT depth root for --iq_dir mode. Ignored if cache has gt_depth.",
    )
    parser.add_argument(
        "--hole_mask_dir",
        type=Path,
        default=None,
        help="Optional hole mask root for --iq_dir mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/approx_bayestof_baseline"),
        help="Directory for outputs and metrics.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--maxd", type=float, default=10.0)
    parser.add_argument("--nt", type=int, default=5000)

    parser.add_argument(
        "--wavelet",
        type=str,
        default="bior1.3",
        help="Wavelet name. The paper reports bior1.3/bior1.5 and Haar as strong options.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Wavelet decomposition level. Default: min(3, pywt max level).",
    )
    parser.add_argument(
        "--threshold_scale",
        type=float,
        default=0.65,
        help="Multiplier for the universal wavelet threshold.",
    )
    parser.add_argument(
        "--amplitude_percentile",
        type=float,
        default=95.0,
        help="Robust high percentile used to normalize I/Q amplitude reliability.",
    )
    parser.add_argument(
        "--min_reliability",
        type=float,
        default=0.08,
        help="Lower bound for amplitude-derived reliability.",
    )
    parser.add_argument(
        "--hole_policy",
        type=str,
        default="prefill",
        choices=["preserve", "prefill", "ignore"],
        help=(
            "How to handle explicit holes before denoising. preserve keeps zero holes, "
            "prefill inpaints IQ channels first, ignore treats the sample as hole-free."
        ),
    )
    parser.add_argument(
        "--prefill_method",
        type=str,
        default="telea",
        choices=["telea", "ns"],
        help="OpenCV method for IQ prefill when --hole_policy=prefill.",
    )
    parser.add_argument("--prefill_radius", type=int, default=3)
    parser.add_argument(
        "--fill_depth_holes",
        action="store_true",
        default=False,
        help="Also save/evaluate BayesToF plus depth-domain inpainting inside holes.",
    )
    parser.add_argument(
        "--depth_fill_method",
        type=str,
        default="telea",
        choices=["telea", "ns"],
    )
    parser.add_argument("--depth_fill_radius", type=int, default=15)
    parser.add_argument("--save_iq", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--vis_max_samples", type=int, default=20)
    return parser.parse_args()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    mkdir(Path(path).parent)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(path, record):
    mkdir(Path(path).parent)
    with Path(path).open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def read_list(path):
    with Path(path).open("r") as f:
        return [line.strip() for line in f if line.strip()]


def natural_key(path):
    path = Path(path)
    parts = []
    for part in path.with_suffix("").parts:
        if part.isdigit():
            parts.append(int(part))
        else:
            digits = "".join(ch for ch in part if ch.isdigit())
            parts.append(int(digits) if digits else part)
    return parts


def sample_matches(sample_name, path, wanted):
    if not wanted:
        return True
    stem = Path(path).stem
    suffix3 = "/".join(Path(path).with_suffix("").parts[-3:])
    return sample_name in wanted or suffix3 in wanted or stem in wanted


def collect_cache_paths(args):
    if args.sample_list:
        paths = [resolve_cache_list_item(args, item) for item in read_list(args.sample_list)]
    elif args.samples:
        paths = [resolve_cache_list_item(args, item) for item in args.samples]
    else:
        paths = sorted(args.cache_dir.rglob("*.npz"), key=natural_key)
    if args.max_samples is not None:
        paths = paths[: int(args.max_samples)]
    if not paths:
        raise FileNotFoundError(f"No .npz files found for {args.cache_dir}")
    return paths


def resolve_cache_list_item(args, item):
    p = Path(item)
    if p.suffix == ".npz" or p.exists():
        return p if p.is_absolute() else Path.cwd() / p

    parts = str(item).split("/")
    if len(parts) == 3:
        candidate = args.cache_dir / parts[0] / parts[1] / f"{parts[2]}.npz"
        if candidate.exists():
            return candidate

    matches = sorted(args.cache_dir.rglob(f"{Path(item).name}.npz"), key=natural_key)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous cache sample id {item!r}; use scene/idx/sample or full path.")
    raise FileNotFoundError(f"Could not resolve cache sample {item!r} under {args.cache_dir}")


def collect_iq_samples(args):
    if args.sample_list:
        ids = read_list(args.sample_list)
    elif args.samples:
        ids = list(args.samples)
    else:
        ids = []
        for a_path in sorted(args.iq_dir.rglob("*_A.npy"), key=natural_key):
            rel = a_path.relative_to(args.iq_dir)
            if len(rel.parts) < 3:
                continue
            scene, idx = rel.parts[-3], rel.parts[-2]
            sample = a_path.stem[:-2]
            ids.append(f"{scene}/{idx}/{sample}")
    if args.max_samples is not None:
        ids = ids[: int(args.max_samples)]
    if not ids:
        raise FileNotFoundError(f"No IQ samples found for {args.iq_dir}")
    return ids


def parse_cache_sample_name(data, path):
    if "sample_name" in data.files:
        value = data["sample_name"]
        return str(value.item() if value.shape == () else value)
    return "/".join(Path(path).with_suffix("").parts[-3:])


def parse_sample_id(sample_id):
    parts = str(sample_id).split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected scene/idx/sample id, got {sample_id!r}")
    return parts[0], parts[1], parts[2]


def load_iq_stack(root, scene, idx, sample):
    channels = []
    for ch in CHANNELS:
        path = Path(root) / scene / str(idx) / f"{sample}_{ch}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing IQ channel: {path}")
        channels.append(np.load(path).astype(np.float32))
    return np.stack(channels, axis=0).astype(np.float32)


def load_optional_array(path):
    if path is None or not Path(path).exists():
        return None
    return np.load(path).astype(np.float32)


def resize_float_to_shape(array, shape, interpolation=cv2.INTER_LINEAR):
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float32)
    if array.shape == shape:
        return array
    return cv2.resize(array, (shape[1], shape[0]), interpolation=interpolation).astype(np.float32)


def resize_bool_to_shape(mask, shape):
    if mask is None:
        return None
    mask = np.asarray(mask) > 0.5
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def load_cache_sample(path, args):
    with np.load(path, allow_pickle=False) as data:
        sample_name = parse_cache_sample_name(data, path)
        if not sample_matches(sample_name, path, set(args.samples or [])):
            return None
        if args.cache_iq_key not in data.files:
            raise KeyError(
                f"{path} does not contain {args.cache_iq_key!r}. "
                "Regenerate cache with --depth_cache_save_iq."
            )
        iq = data[args.cache_iq_key].astype(np.float32)
        shape = tuple(iq.shape[-2:])
        gt_depth = data["gt_depth"].astype(np.float32) if "gt_depth" in data.files else None
        gt_depth = resize_float_to_shape(gt_depth, shape)
        hole_mask = data["hole_mask"].astype(np.float32) > 0.5 if "hole_mask" in data.files else None
        hole_mask = resize_bool_to_shape(hole_mask, shape)
        confidence = data["confidence"].astype(np.float32) if "confidence" in data.files else None
        confidence = resize_float_to_shape(confidence, shape)
        valid_mask = data["valid_mask"].astype(np.float32) > 0.5 if "valid_mask" in data.files else None
        valid_mask = resize_bool_to_shape(valid_mask, shape)

        cached_depths = {}
        for name, key in [
            ("noisy", "depth_noisy"),
            ("depthcad", "depth_depthcad"),
            ("base", "depth_base"),
        ]:
            if key in data.files:
                cached_depths[name] = resize_float_to_shape(data[key].astype(np.float32), shape)

    return {
        "sample_name": sample_name,
        "source_path": str(Path(path).resolve()),
        "iq": iq,
        "gt_depth": gt_depth,
        "hole_mask": hole_mask,
        "confidence": confidence,
        "valid_mask": valid_mask,
        "cached_depths": cached_depths,
    }


def load_iq_dir_sample(sample_id, args):
    scene, idx, sample = parse_sample_id(sample_id)
    iq = load_iq_stack(args.iq_dir, scene, idx, sample)
    shape = tuple(iq.shape[-2:])
    gt_depth = load_optional_array(Path(args.gt_depth_dir) / scene / idx / f"{sample}.npy")
    gt_depth = resize_float_to_shape(gt_depth, shape)
    hole_mask = load_optional_array(Path(args.hole_mask_dir) / scene / idx / f"{sample}.npy") if args.hole_mask_dir else None
    hole_mask = resize_bool_to_shape(hole_mask, shape)
    return {
        "sample_name": f"{scene}/{idx}/{sample}",
        "source_path": str((Path(args.iq_dir) / scene / idx).resolve()),
        "iq": iq,
        "gt_depth": gt_depth,
        "hole_mask": hole_mask,
        "confidence": None,
        "valid_mask": None,
        "cached_depths": {},
    }


def robust_percentile(values, percentile, fallback=1.0):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    out = float(np.percentile(values, percentile))
    if not np.isfinite(out) or abs(out) < 1e-8:
        return float(fallback)
    return out


def robust_sigma(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    med = np.median(values)
    sigma = np.median(np.abs(values - med)) / 0.6745
    if not np.isfinite(sigma):
        return 0.0
    return float(sigma)


def crop_or_pad_like(image, shape):
    h, w = shape
    out = np.asarray(image, dtype=np.float32)
    out = out[:h, :w]
    if out.shape == (h, w):
        return out
    padded = np.zeros((h, w), dtype=np.float32)
    padded[: out.shape[0], : out.shape[1]] = out
    if out.shape[0] < h and out.shape[0] > 0:
        padded[out.shape[0] :, : out.shape[1]] = out[-1:, :]
    if out.shape[1] < w and out.shape[1] > 0:
        padded[:, out.shape[1] :] = padded[:, out.shape[1] - 1 : out.shape[1]]
    return padded


def soft_threshold(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def resize_reliability(reliability, shape):
    resized = cv2.resize(
        reliability.astype(np.float32),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_AREA,
    )
    return np.clip(resized, 1e-6, 1.0).astype(np.float32)


def denoise_wavelet_component(component, reliability, wavelet, level, threshold_scale):
    coeffs = pywt.wavedec2(component.astype(np.float32), wavelet=wavelet, level=level, mode="symmetric")
    finest = coeffs[-1]
    sigma = robust_sigma(np.concatenate([band.reshape(-1) for band in finest]))
    if sigma <= 1e-8:
        return component.astype(np.float32), sigma

    threshold_base = float(threshold_scale) * sigma * math.sqrt(2.0 * math.log(component.size))
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_detail = []
        for band in detail:
            rel = resize_reliability(reliability, band.shape)
            threshold = threshold_base / np.sqrt(rel)
            denoised_detail.append(soft_threshold(band, threshold).astype(np.float32))
        denoised_coeffs.append(tuple(denoised_detail))

    denoised = pywt.waverec2(denoised_coeffs, wavelet=wavelet, mode="symmetric")
    return crop_or_pad_like(denoised, component.shape), sigma


def inpaint_float_channel(channel, hole_mask, method="telea", radius=3):
    hole = np.asarray(hole_mask) > 0.5
    if hole.sum() == 0:
        return channel.astype(np.float32).copy()

    channel = np.asarray(channel, dtype=np.float32)
    valid = (~hole) & np.isfinite(channel)
    if valid.sum() == 0:
        return np.nan_to_num(channel, nan=0.0).astype(np.float32)

    lo, hi = np.percentile(channel[valid], [1.0, 99.0])
    scale = float(hi - lo)
    if scale < 1e-8:
        out = channel.copy()
        out[hole] = float(np.median(channel[valid]))
        return out.astype(np.float32)

    normalized = np.clip((channel - lo) / scale, 0.0, 1.0)
    normalized = np.nan_to_num(normalized, nan=0.0, neginf=0.0, posinf=1.0)
    image_uint8 = (normalized * 255.0).astype(np.uint8)
    mask_uint8 = (hole.astype(np.uint8) * 255)
    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    filled = cv2.inpaint(image_uint8, mask_uint8, inpaintRadius=int(radius), flags=flags)
    filled = filled.astype(np.float32) / 255.0 * scale + lo

    out = channel.copy()
    out[hole] = filled[hole]
    return out.astype(np.float32)


def prefill_iq_holes(iq, hole_mask, method, radius):
    if hole_mask is None or np.asarray(hole_mask).sum() == 0:
        return iq.astype(np.float32).copy()
    out = np.empty_like(iq, dtype=np.float32)
    for channel in range(iq.shape[0]):
        out[channel] = inpaint_float_channel(iq[channel], hole_mask, method=method, radius=radius)
    return out


def denoise_iq_approx_bayestof(iq, hole_mask, args):
    iq = np.asarray(iq, dtype=np.float32)
    if iq.ndim != 3 or iq.shape[0] < 6:
        raise ValueError(f"Expected IQ shape (6,H,W), got {iq.shape}")

    if args.hole_policy == "ignore":
        active_hole = None
    else:
        active_hole = hole_mask

    work_iq = iq
    if args.hole_policy == "prefill" and active_hole is not None:
        work_iq = prefill_iq_holes(
            iq,
            active_hole,
            method=args.prefill_method,
            radius=args.prefill_radius,
        )

    wavelet = pywt.Wavelet(args.wavelet)
    max_level = pywt.dwtn_max_level(work_iq.shape[-2:], wavelet)
    if max_level < 1:
        return work_iq.astype(np.float32), {"level": 0, "sigmas": []}
    level = int(args.level) if args.level is not None else min(3, max_level)
    level = max(1, min(level, max_level))

    out = work_iq.copy().astype(np.float32)
    sigmas = []
    for i_idx, q_idx in PAIR_INDICES:
        i_ch = work_iq[i_idx].astype(np.float32)
        q_ch = work_iq[q_idx].astype(np.float32)
        amplitude = np.sqrt(i_ch * i_ch + q_ch * q_ch)

        if active_hole is not None:
            valid_amp = amplitude[(~active_hole) & np.isfinite(amplitude)]
        else:
            valid_amp = amplitude[np.isfinite(amplitude)]
        amp_ref = robust_percentile(valid_amp, args.amplitude_percentile, fallback=1.0)
        reliability = np.clip(amplitude / amp_ref, args.min_reliability, 1.0)
        if active_hole is not None and args.hole_policy == "preserve":
            reliability = np.where(active_hole, args.min_reliability, reliability)

        denom = np.maximum(amplitude, 1e-8)
        unit_i = np.nan_to_num(i_ch / denom, nan=0.0, neginf=0.0, posinf=0.0)
        unit_q = np.nan_to_num(q_ch / denom, nan=0.0, neginf=0.0, posinf=0.0)

        den_i, sigma_i = denoise_wavelet_component(
            unit_i,
            reliability,
            wavelet=args.wavelet,
            level=level,
            threshold_scale=args.threshold_scale,
        )
        den_q, sigma_q = denoise_wavelet_component(
            unit_q,
            reliability,
            wavelet=args.wavelet,
            level=level,
            threshold_scale=args.threshold_scale,
        )
        norm = np.sqrt(den_i * den_i + den_q * den_q)
        den_i = np.where(norm > 1e-8, den_i / norm, unit_i)
        den_q = np.where(norm > 1e-8, den_q / norm, unit_q)

        out[i_idx] = (den_i * amplitude).astype(np.float32)
        out[q_idx] = (den_q * amplitude).astype(np.float32)
        if active_hole is not None and args.hole_policy == "preserve":
            out[i_idx][active_hole] = iq[i_idx][active_hole]
            out[q_idx][active_hole] = iq[q_idx][active_hole]
        sigmas.append({"pair": [i_idx, q_idx], "sigma_i": sigma_i, "sigma_q": sigma_q})

    return out.astype(np.float32), {"level": level, "sigmas": sigmas}


def valid_depth_mask(gt_depth, valid_mask=None):
    if gt_depth is None:
        return None
    valid = np.isfinite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 9.9)
    if valid_mask is not None:
        valid = valid & (valid_mask > 0.5)
    return valid


def mae_and_count(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return None, 0
    return float(np.abs(pred[valid] - target[valid]).mean()), count


def sample_metrics(depths, target, valid_mask, hole_mask):
    if target is None:
        return {}
    valid = valid_depth_mask(target, valid_mask)
    if valid is None:
        return {}
    if hole_mask is None:
        hole = np.zeros_like(valid, dtype=bool)
    else:
        hole = np.asarray(hole_mask) > 0.5
        if hole.shape != valid.shape:
            hole = cv2.resize(
                hole.astype(np.uint8),
                (valid.shape[1], valid.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

    metrics = {}
    for name, depth in depths.items():
        if depth.shape != target.shape:
            continue
        for region_name, region_mask in [
            ("global", valid),
            ("hole", valid & hole),
            ("valid", valid & (~hole)),
        ]:
            mae, count = mae_and_count(depth, target, region_mask)
            metrics[f"{name}_{region_name}_mae"] = mae
            metrics[f"{name}_{region_name}_count"] = count
    return metrics


def aggregate_metrics(rows):
    totals = {}
    counts = {}
    for row in rows:
        metrics = row.get("metrics", {})
        for key, value in metrics.items():
            if not key.endswith("_mae") or value is None:
                continue
            count_key = key[:-4] + "_count"
            count = int(metrics.get(count_key, 0) or 0)
            if count <= 0:
                continue
            prefix = key[:-4]
            totals[prefix] = totals.get(prefix, 0.0) + float(value) * count
            counts[prefix] = counts.get(prefix, 0) + count

    out = {}
    for key in sorted(totals):
        out[f"{key}_mae"] = totals[key] / max(counts[key], 1)
        out[f"{key}_count"] = counts[key]
    return out


def output_path(output_dir, subdir, sample_name, suffix):
    parts = str(sample_name).split("/")
    path = Path(output_dir) / subdir
    for part in parts[:-1]:
        path = path / part
    mkdir(path)
    return path / f"{parts[-1]}{suffix}"


def save_visualization(path, sample_name, depths, target, hole_mask):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if target is not None:
        panels.append(("GT", target))
    for name in ["noisy", "depthcad", "base", "bayestof", "bayestof_filled"]:
        if name in depths:
            panels.append((name, depths[name]))
    if hole_mask is not None:
        panels.append(("hole", hole_mask.astype(np.float32)))

    if not panels:
        return

    depth_values = []
    for title, image in panels:
        if title == "hole":
            continue
        finite = image[np.isfinite(image)]
        if finite.size:
            depth_values.append(finite)
    if depth_values:
        all_values = np.concatenate(depth_values)
        vmin, vmax = np.percentile(all_values, [1.0, 99.0])
        if float(vmax - vmin) < 1e-6:
            vmin, vmax = float(all_values.min()), float(all_values.max())
    else:
        vmin, vmax = 0.0, 1.0

    cols = min(len(panels), 4)
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, (title, image) in zip(axes.reshape(-1), panels):
        if title == "hole":
            im = ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(image, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(sample_name)
    fig.tight_layout()
    mkdir(Path(path).parent)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def process_sample(sample, args, estimator):
    iq = sample["iq"]
    hole_mask = sample["hole_mask"]
    gt_depth = sample["gt_depth"]
    valid_mask = sample["valid_mask"]
    sample_name = sample["sample_name"]

    denoised_iq, denoise_meta = denoise_iq_approx_bayestof(iq, hole_mask, args)
    depth_bayestof = estimator.process(denoised_iq)
    depth_bayestof = np.nan_to_num(depth_bayestof, nan=0.0, neginf=0.0, posinf=0.0).astype(np.float32)

    depths = dict(sample["cached_depths"])
    if "noisy" not in depths:
        depths["noisy"] = estimator.process(iq).astype(np.float32)
    depths["bayestof"] = depth_bayestof

    filled_path = None
    if args.fill_depth_holes and hole_mask is not None:
        depth_filled = opencv_depth_inpaint(
            depth_bayestof,
            hole_mask,
            method=args.depth_fill_method,
            radius=args.depth_fill_radius,
        )
        depths["bayestof_filled"] = depth_filled.astype(np.float32)
        filled_path = output_path(args.output_dir, "depth_filled", sample_name, "_approx_bayestof_filled.npy")
        np.save(filled_path, depths["bayestof_filled"])

    depth_path = output_path(args.output_dir, "depth", sample_name, "_approx_bayestof.npy")
    np.save(depth_path, depth_bayestof)

    iq_path = None
    if args.save_iq:
        iq_path = output_path(args.output_dir, "iq", sample_name, "_approx_bayestof_iq.npy")
        np.save(iq_path, denoised_iq)

    metrics = sample_metrics(depths, gt_depth, valid_mask, hole_mask)
    row = {
        "sample_name": sample_name,
        "source_path": sample["source_path"],
        "depth_path": str(depth_path.resolve()),
        "filled_depth_path": str(filled_path.resolve()) if filled_path else None,
        "iq_path": str(iq_path.resolve()) if iq_path else None,
        "hole_ratio": float(np.mean(hole_mask > 0.5)) if hole_mask is not None else 0.0,
        "denoise": denoise_meta,
        "metrics": metrics,
    }
    return row, depths


def main():
    args = parse_args()
    mkdir(args.output_dir)
    for subdir in ["depth", "depth_filled", "iq", "visualizations"]:
        mkdir(args.output_dir / subdir)

    estimator = DepthEstimator(maxd=args.maxd, nt=args.nt)
    rows = []
    skipped = []

    if args.cache_dir:
        items = collect_cache_paths(args)
        loader = lambda item: load_cache_sample(item, args)
    else:
        items = collect_iq_samples(args)
        loader = lambda item: load_iq_dir_sample(item, args)

    metrics_path = args.output_dir / "per_sample.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for index, item in enumerate(items, start=1):
        try:
            sample = loader(item)
            if sample is None:
                continue
            row, depths = process_sample(sample, args, estimator)
            rows.append(row)
            append_jsonl(metrics_path, row)
            if args.visualize and len(rows) <= args.vis_max_samples:
                vis_path = output_path(args.output_dir, "visualizations", sample["sample_name"], ".png")
                save_visualization(vis_path, sample["sample_name"], depths, sample["gt_depth"], sample["hole_mask"])

            metric = row["metrics"].get("bayestof_global_mae")
            metric_text = "NA" if metric is None else f"{metric:.6f}"
            print(
                f"[{index}/{len(items)}] {sample['sample_name']} "
                f"hole={row['hole_ratio']:.3f} bayestof_global_mae={metric_text}"
            )
        except Exception as exc:
            skipped.append({"item": str(item), "error": str(exc)})
            print(f"[skip] {item}: {exc}")

    summary = {
        "note": (
            "Approximate BayesToF-style baseline. Uses 6-channel IQ only and "
            "does not reproduce full tap-count GSM MMSE BayesToF."
        ),
        "paper": str((Path("paper") / "oe-34-11-20044.pdf").resolve()),
        "source": "cache_dir" if args.cache_dir else "iq_dir",
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "iq_dir": str(args.iq_dir.resolve()) if args.iq_dir else None,
        "output_dir": str(args.output_dir.resolve()),
        "num_samples": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped[:100],
        "args": {
            "cache_iq_key": args.cache_iq_key,
            "wavelet": args.wavelet,
            "level": args.level,
            "threshold_scale": args.threshold_scale,
            "amplitude_percentile": args.amplitude_percentile,
            "min_reliability": args.min_reliability,
            "hole_policy": args.hole_policy,
            "prefill_method": args.prefill_method,
            "prefill_radius": args.prefill_radius,
            "fill_depth_holes": bool(args.fill_depth_holes),
            "depth_fill_method": args.depth_fill_method,
            "depth_fill_radius": args.depth_fill_radius,
            "maxd": args.maxd,
            "nt": args.nt,
        },
        "aggregate": aggregate_metrics(rows),
        "rows": rows,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(f"Saved Approx-BayesToF baseline to {args.output_dir}")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
