import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Decode far_pic Kinect-v2-style raw byte-stream .npy files. "
            "The stream layout matches 10 packed 11-bit 424x512 subimages."
        )
    )
    parser.add_argument("--input_dir", type=str, default="far_pic/raw")
    parser.add_argument("--subimage_dir", type=str, default="far_pic/kinect_raw11_10x424x512")
    parser.add_argument("--raw9_dir", type=str, default="far_pic/kinect_raw9_424x512_sign_extend")
    parser.add_argument("--preview_dir", type=str, default="output/far_pic_kinect_raw_previews_sign_extend")
    parser.add_argument("--target_h", type=int, default=240)
    parser.add_argument("--target_w", type=int, default=320)
    parser.add_argument(
        "--lut_mode",
        type=str,
        default="sign_extend",
        choices=["unsigned", "sign_extend", "sign_extend_shift5"],
        help=(
            "Approximation for the missing Kinect 11-to-16 bit lookup table. "
            "sign_extend best matches the existing raw/ channel scale in this repo."
        ),
    )
    parser.add_argument("--save_resized_raw9", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_raw11_subimages(flat, subimages=10, height=424, width=512):
    flat = np.asarray(flat)
    bytes_per_subimage = height * width * 11 // 8
    expected = subimages * bytes_per_subimage
    if flat.dtype != np.uint8 or flat.ndim != 1 or flat.size != expected:
        raise ValueError(
            f"Expected flat uint8 Kinect raw stream with {expected} bytes, "
            f"got shape={flat.shape}, dtype={flat.dtype}, size={flat.size}"
        )

    out = np.zeros((subimages, height, width), dtype=np.uint16)
    x = np.arange(width, dtype=np.int64)
    valid_x = (x >= 1) & (x <= 510)
    r1zi = ((x >> 2) + ((x & 0x3) << 7)) * 11
    word_index = r1zi >> 4
    bit_offset = r1zi & 15

    for sub in range(subimages):
        start = sub * bytes_per_subimage
        words = flat[start : start + bytes_per_subimage].view("<u2")
        words = words.reshape(height, 352)
        for y in range(height):
            src_y = y + 212 if y < 212 else 423 - y
            row = words[src_y]
            i1 = row[word_index].astype(np.uint32)
            i2 = row[np.minimum(word_index + 1, row.shape[0] - 1)].astype(np.uint32)
            values = ((i1 >> bit_offset) | (i2 << (16 - bit_offset))) & 2047
            values[~valid_x] = 0
            out[sub, y] = values.astype(np.uint16)
    return out


def apply_lut_approx(raw11, mode):
    raw11 = np.asarray(raw11)
    if mode == "unsigned":
        return raw11.astype(np.float32)
    signed = raw11.astype(np.int32)
    signed = np.where(signed >= 1024, signed - 2048, signed)
    if mode == "sign_extend_shift5":
        signed = signed << 5
    return signed.astype(np.float32)


def process_measurement_triple(measurements, trig_table, ab_multiplier_per_frq, z_table, ab_multiplier):
    # measurements: (3,H,W), trig_table: (H,W,6)
    saturated = np.any(measurements == 32767.0, axis=0)
    valid = z_table > 0

    ir_a = (
        trig_table[:, :, 0] * measurements[0]
        + trig_table[:, :, 1] * measurements[1]
        + trig_table[:, :, 2] * measurements[2]
    )
    ir_b = (
        trig_table[:, :, 3] * measurements[0]
        + trig_table[:, :, 4] * measurements[1]
        + trig_table[:, :, 5] * measurements[2]
    )
    ir_a = ir_a * float(ab_multiplier_per_frq)
    ir_b = ir_b * float(ab_multiplier_per_frq)
    amp = np.sqrt(ir_a * ir_a + ir_b * ir_b) * float(ab_multiplier)

    ir_a = np.where(valid & ~saturated, ir_a, 0.0)
    ir_b = np.where(valid & ~saturated, ir_b, 0.0)
    amp = np.where(valid & ~saturated, amp, np.where(saturated, 65535.0, 0.0))
    return np.stack([ir_a, ir_b, amp], axis=0).astype(np.float32)


def raw11_to_raw9(raw11, lut_mode):
    z_table = np.fromfile("params/kinect/z_table", dtype=np.float32).reshape(424, 512)
    trig0 = np.fromfile("params/kinect/trig_table0", dtype=np.float32).reshape(424, 512, 6)
    trig1 = np.fromfile("params/kinect/trig_table1", dtype=np.float32).reshape(424, 512, 6)
    trig2 = np.fromfile("params/kinect/trig_table2", dtype=np.float32).reshape(424, 512, 6)

    ab_multiplier = 0.66666687
    ab_multiplier_per_frq = [1.32258105, 1.0, 1.61290300]
    m = apply_lut_approx(raw11[:9], lut_mode)

    out0 = process_measurement_triple(m[0:3], trig0, ab_multiplier_per_frq[0], z_table, ab_multiplier)
    out1 = process_measurement_triple(m[3:6], trig1, ab_multiplier_per_frq[1], z_table, ab_multiplier)
    out2 = process_measurement_triple(m[6:9], trig2, ab_multiplier_per_frq[2], z_table, ab_multiplier)
    return np.concatenate([out0, out1, out2], axis=0).astype(np.float32)


def resize_chw(chw, target_h, target_w):
    out = np.zeros((chw.shape[0], target_h, target_w), dtype=np.float32)
    for idx in range(chw.shape[0]):
        out[idx] = cv2.resize(chw[idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return out


def normalize_u8(image):
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def save_contact_sheet(chw, out_path, labels):
    tiles = []
    for idx, frame in enumerate(chw):
        tile = cv2.cvtColor(normalize_u8(frame), cv2.COLOR_GRAY2BGR)
        label = labels[idx] if idx < len(labels) else f"ch{idx}"
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 5:
        tiles.append(blank.copy())
    rows = [np.concatenate(tiles[i : i + 5], axis=1) for i in range(0, len(tiles), 5)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    subimage_dir = Path(args.subimage_dir)
    raw9_dir = Path(args.raw9_dir)
    raw9_240_dir = raw9_dir.with_name(raw9_dir.name + "_240x320")
    preview_dir = Path(args.preview_dir)
    for directory in [subimage_dir, raw9_dir, raw9_240_dir, preview_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    for path in paths:
        stem = path.name.replace("_raw.npy", "")
        sub_path = subimage_dir / f"{stem}.npy"
        raw9_path = raw9_dir / f"{stem}.npy"
        raw9_240_path = raw9_240_dir / f"{stem}.npy"
        sub_preview = preview_dir / f"{stem}_raw11.png"
        raw9_preview = preview_dir / f"{stem}_raw9.png"

        if sub_path.exists() and raw9_path.exists() and raw9_240_path.exists() and not args.overwrite:
            raw11 = np.load(sub_path)
            raw9 = np.load(raw9_path)
        else:
            raw11 = decode_raw11_subimages(np.load(path))
            raw9 = raw11_to_raw9(raw11, args.lut_mode)
            raw9_240 = resize_chw(raw9, args.target_h, args.target_w)
            np.save(sub_path, raw11)
            np.save(raw9_path, raw9)
            np.save(raw9_240_path, raw9_240)

        save_contact_sheet(raw11.astype(np.float32), sub_preview, [f"raw11_{i}" for i in range(10)])
        save_contact_sheet(raw9, raw9_preview, ["a0", "b0", "amp0", "a1", "b1", "amp1", "a2", "b2", "amp2"])
        print(f"[ok] {path.name} -> raw11={raw11.shape} raw9={raw9.shape}")

    print(f"Saved raw11 subimages to {subimage_dir}")
    print(f"Saved raw9 full-res to {raw9_dir}")
    print(f"Saved raw9 resized to {raw9_240_dir}")
    print(f"Saved previews to {preview_dir}")


if __name__ == "__main__":
    main()
