import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert far_pic raw byte-stream .npy files into raw9-style tensors "
            "with shape (9, H, W) for the existing real-raw9 pipeline."
        )
    )
    parser.add_argument("--input_dir", type=str, default="far_pic/raw")
    parser.add_argument("--output_dir", type=str, default="far_pic/raw9_240x320")
    parser.add_argument("--src_h", type=int, default=352)
    parser.add_argument("--src_w", type=int, default=424)
    parser.add_argument("--src_c", type=int, default=10)
    parser.add_argument(
        "--channel_indices",
        type=int,
        nargs=9,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        help="Which 9 channels to keep from the decoded source tensor.",
    )
    parser.add_argument("--target_h", type=int, default=240)
    parser.add_argument("--target_w", type=int, default=320)
    parser.add_argument(
        "--layout",
        type=str,
        default="hwc",
        choices=["hwc", "chw"],
        help="Decoded layout before channel selection.",
    )
    parser.add_argument(
        "--aspect_mode",
        type=str,
        default="crop_resize",
        choices=["crop_resize", "direct_resize", "pad_resize"],
        help=(
            "crop_resize preserves aspect ratio by center-cropping before resize. "
            "direct_resize may distort the image. pad_resize letterboxes before resize."
        ),
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="tof10_iq6layout",
        choices=[
            "tof10_iq6layout",
            "tof10_triplet_layout",
            "tof10_signed_triplet",
            "first9_uint16",
            "first9_int16",
        ],
        help=(
            "tof10_iq6layout interprets the source as uint16 HxWx10 ToF data and writes "
            "the first 6 channels in the same IQ order expected by the current real-raw9 "
            "flow pipeline: [I30,Q30,I40,Q40,I58,Q58], followed by three auxiliary channels. "
            "tof10_triplet_layout keeps the older [I,Q,Aux]x3 triplet layout for debugging. "
            "tof10_signed_triplet keeps the real-raw9 restoration layout [corr0,corr1,amp]x3: "
            "correlation channels are interpreted as int16, amplitude channels as uint16."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_flat_uint8(path, src_h, src_w, src_c, layout):
    flat = np.load(path)
    if flat.dtype != np.uint8 or flat.ndim != 1:
        raise ValueError(f"{path} is expected to be flat uint8, got shape={flat.shape}, dtype={flat.dtype}")

    expected = int(src_h) * int(src_w) * int(src_c) * 2
    if flat.size != expected:
        raise ValueError(f"{path} has {flat.size} bytes, expected {expected} for uint16[{src_h},{src_w},{src_c}]")

    data = flat.view(np.uint16)
    if layout == "hwc":
        return data.reshape(src_h, src_w, src_c)
    return data.reshape(src_c, src_h, src_w)


def sqrt_ldr(correlations):
    tof_conf = np.abs(correlations[0]) + np.abs(correlations[1])
    tof_conf_l = 16 * np.sqrt(tof_conf + 36.0) - 96.0
    tof_conf = tof_conf.copy()
    tof_conf[tof_conf == 0] = 1.0
    i_tmp = tof_conf_l * correlations[0] / tof_conf
    q_tmp = tof_conf_l * correlations[1] / tof_conf
    return np.stack((i_tmp, q_tmp), axis=0)


def compress_aux(channel):
    return 16.0 * np.sqrt(np.maximum(channel, 0.0) + 36.0) - 96.0


def select_channels(decoded, channel_indices, layout):
    indices = list(channel_indices)
    if layout == "hwc":
        chosen = decoded[:, :, indices]
        return np.moveaxis(chosen, -1, 0).astype(np.float32)
    return decoded[indices].astype(np.float32)


def decode_tof10_iq6layout(decoded, layout):
    if layout == "hwc":
        raw_u16 = np.moveaxis(decoded, -1, 0)
    else:
        raw_u16 = decoded
    u = raw_u16.astype(np.float32)
    s = raw_u16.view(np.int16).astype(np.float32)

    out = np.zeros((9, u.shape[1], u.shape[2]), dtype=np.float32)
    # Historical pairing from the old preprocessing code:
    # (4,3)->30MHz, (1,0)->40MHz, (7,6)->58MHz, with 2/5/8 as per-frequency auxiliaries.
    iq30 = sqrt_ldr(np.stack([s[4], s[3]], axis=0))
    iq40 = sqrt_ldr(np.stack([s[1], s[0]], axis=0))
    iq58 = sqrt_ldr(np.stack([s[7], s[6]], axis=0))

    out[0:2] = iq30
    out[2:4] = iq40
    out[4:6] = iq58
    out[6] = compress_aux(u[2])
    out[7] = compress_aux(u[5])
    out[8] = compress_aux(u[8])
    return out


def decode_tof10_triplet_layout(decoded, layout):
    if layout == "hwc":
        raw_u16 = np.moveaxis(decoded, -1, 0)
    else:
        raw_u16 = decoded
    u = raw_u16.astype(np.float32)
    s = raw_u16.view(np.int16).astype(np.float32)

    out = np.zeros((9, u.shape[1], u.shape[2]), dtype=np.float32)
    out[0:2] = sqrt_ldr(np.stack([s[1], s[0]], axis=0))
    out[2] = compress_aux(u[2])
    out[3:5] = sqrt_ldr(np.stack([s[4], s[3]], axis=0))
    out[5] = compress_aux(u[5])
    out[6:8] = sqrt_ldr(np.stack([s[7], s[6]], axis=0))
    out[8] = compress_aux(u[8])
    return out


def decode_tof10_signed_triplet(decoded, layout):
    if layout == "hwc":
        raw_u16 = np.moveaxis(decoded, -1, 0)
    else:
        raw_u16 = decoded

    u = raw_u16.astype(np.float32)
    s = raw_u16.view(np.int16).astype(np.float32)

    out = np.zeros((9, u.shape[1], u.shape[2]), dtype=np.float32)
    for base in (0, 3, 6):
        out[base] = s[base]
        out[base + 1] = s[base + 1]
        out[base + 2] = u[base + 2]
    return out


def center_crop_to_aspect(chw, target_h, target_w):
    _, h, w = chw.shape
    target_aspect = float(target_w) / float(target_h)
    src_aspect = float(w) / float(h)

    if abs(src_aspect - target_aspect) < 1e-6:
        return chw

    if src_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        x0 = max(0, (w - new_w) // 2)
        return chw[:, :, x0 : x0 + new_w]

    new_h = int(round(w / target_aspect))
    y0 = max(0, (h - new_h) // 2)
    return chw[:, y0 : y0 + new_h, :]


def pad_to_aspect(chw, target_h, target_w):
    _, h, w = chw.shape
    target_aspect = float(target_w) / float(target_h)
    src_aspect = float(w) / float(h)

    if abs(src_aspect - target_aspect) < 1e-6:
        return chw

    if src_aspect > target_aspect:
        new_h = int(round(w / target_aspect))
        pad = max(0, new_h - h)
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        return np.pad(chw, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant")

    new_w = int(round(h * target_aspect))
    pad = max(0, new_w - w)
    pad_left = pad // 2
    pad_right = pad - pad_left
    return np.pad(chw, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")


def resize_chw(chw, target_h, target_w):
    out = np.zeros((chw.shape[0], target_h, target_w), dtype=np.float32)
    for i in range(chw.shape[0]):
        out[i] = cv2.resize(chw[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return out


def convert_one(path, args):
    decoded = decode_flat_uint8(path, args.src_h, args.src_w, args.src_c, args.layout)
    if args.decode_mode == "tof10_iq6layout":
        raw9 = decode_tof10_iq6layout(decoded, args.layout)
    elif args.decode_mode == "tof10_triplet_layout":
        raw9 = decode_tof10_triplet_layout(decoded, args.layout)
    elif args.decode_mode == "tof10_signed_triplet":
        raw9 = decode_tof10_signed_triplet(decoded, args.layout)
    elif args.decode_mode == "first9_int16":
        signed = decoded.view(np.int16)
        raw9 = select_channels(signed, args.channel_indices, args.layout)
    else:
        raw9 = select_channels(decoded, args.channel_indices, args.layout)

    if args.aspect_mode == "crop_resize":
        raw9 = center_crop_to_aspect(raw9, args.target_h, args.target_w)
    elif args.aspect_mode == "pad_resize":
        raw9 = pad_to_aspect(raw9, args.target_h, args.target_w)

    if raw9.shape[1:] != (args.target_h, args.target_w):
        raw9 = resize_chw(raw9, args.target_h, args.target_w)

    return raw9.astype(np.float32)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    for path in paths:
        out_path = output_dir / path.name.replace("_raw.npy", ".npy")
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue
        raw9 = convert_one(path, args)
        np.save(out_path, raw9)
        print(f"[ok] {path.name} -> {out_path.name} shape={raw9.shape} dtype={raw9.dtype}")


if __name__ == "__main__":
    main()
