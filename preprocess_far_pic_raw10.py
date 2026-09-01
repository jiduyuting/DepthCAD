import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unpack far_pic raw byte-stream .npy files as packed 10-bit RAW10 frames. "
            "The observed file size matches 11 frames of 424x512 RAW10 data."
        )
    )
    parser.add_argument("--input_dir", type=str, default="far_pic/raw")
    parser.add_argument("--output_dir", type=str, default="far_pic/raw10_11x424x512")
    parser.add_argument("--preview_dir", type=str, default="output/far_pic_raw10_previews")
    parser.add_argument("--height", type=int, default=424)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=11)
    parser.add_argument(
        "--bit_order",
        type=str,
        default="high8",
        choices=["high8", "low8"],
        help=(
            "RAW10 packing convention. high8 means the first four bytes contain the "
            "8 MSBs and the fifth byte contains packed 2-bit LSBs."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def unpack_raw10(flat, frames=11, height=424, width=512, bit_order="high8"):
    flat = np.asarray(flat)
    expected = int(frames) * int(height) * int(width) * 10 // 8
    if flat.dtype != np.uint8 or flat.ndim != 1 or flat.size != expected:
        raise ValueError(
            f"Expected flat uint8 RAW10 stream with {expected} bytes, "
            f"got shape={flat.shape}, dtype={flat.dtype}, size={flat.size}"
        )

    b = flat.reshape(-1, 5).astype(np.uint16)
    if bit_order == "high8":
        p0 = (b[:, 0] << 2) | (b[:, 4] & 0x03)
        p1 = (b[:, 1] << 2) | ((b[:, 4] >> 2) & 0x03)
        p2 = (b[:, 2] << 2) | ((b[:, 4] >> 4) & 0x03)
        p3 = (b[:, 3] << 2) | ((b[:, 4] >> 6) & 0x03)
    else:
        p0 = b[:, 0] | ((b[:, 4] & 0x03) << 8)
        p1 = b[:, 1] | (((b[:, 4] >> 2) & 0x03) << 8)
        p2 = b[:, 2] | (((b[:, 4] >> 4) & 0x03) << 8)
        p3 = b[:, 3] | (((b[:, 4] >> 6) & 0x03) << 8)

    pixels = np.stack([p0, p1, p2, p3], axis=1).reshape(-1).astype(np.uint16)
    return pixels.reshape(int(frames), int(height), int(width))


def normalize_u8(image):
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def save_contact_sheet(frames, out_path, label_prefix):
    tiles = []
    for idx, frame in enumerate(frames):
        tile = cv2.cvtColor(normalize_u8(frame), cv2.COLOR_GRAY2BGR)
        cv2.putText(
            tile,
            f"{label_prefix}_{idx}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    blank = np.zeros_like(tiles[0])
    while len(tiles) % 4:
        tiles.append(blank.copy())
    rows = [np.concatenate(tiles[i : i + 4], axis=1) for i in range(0, len(tiles), 4)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    preview_dir = Path(args.preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    temporal_sum = None
    converted = 0
    for path in paths:
        out_path = output_dir / path.name.replace("_raw.npy", ".npy")
        preview_path = preview_dir / path.name.replace("_raw.npy", ".png")
        if out_path.exists() and preview_path.exists() and not args.overwrite:
            frames = np.load(out_path)
        else:
            flat = np.load(path)
            frames = unpack_raw10(
                flat,
                frames=args.frames,
                height=args.height,
                width=args.width,
                bit_order=args.bit_order,
            )
            np.save(out_path, frames)
            save_contact_sheet(frames, preview_path, args.bit_order)
        temporal_sum = frames.astype(np.float64) if temporal_sum is None else temporal_sum + frames
        converted += 1
        print(f"[ok] {path.name} -> {out_path.name} shape={frames.shape} dtype={frames.dtype}")

    temporal_mean = (temporal_sum / max(converted, 1)).astype(np.float32)
    np.save(output_dir / "temporal_mean.npy", temporal_mean)
    save_contact_sheet(temporal_mean, preview_dir / "temporal_mean.png", f"{args.bit_order}_mean")
    print(f"Saved {converted} unpacked RAW10 files to {output_dir}")
    print(f"Saved previews to {preview_dir}")


if __name__ == "__main__":
    main()
