#!/usr/bin/env python3
"""Build leakage-free Flow train/val splits for model selection.

The original ``full_pbrt_flow_lists_iq`` split is a random frame-level split
over the *same* 5 scenes and the *same* 40 camera views, so 98.6% of val frames
have a train frame at +/-1 in the identical view. That val measures memorisation
of near-duplicate frames, not restoration, which is why selecting on it picks
checkpoints that degrade on every held-out set.

This script rebuilds train/val with a real distribution shift:

  --mode scene : hold out one whole scene for val (strongest shift, rotate over
                 all scenes and average -- this is the protocol for reporting).
  --mode view  : hold out N camera views per scene (weaker shift, more val
                 samples, useful as an intermediate rung of the ladder).

Both modes guarantee zero shared views and therefore zero adjacent-frame
leakage across the boundary; the emitted summary asserts this.

Test sets stay external and are not written here: PBRT100
(``depth_cache_0514_n100_plane_r12_seed123_iq``, a *different cache pipeline* --
its amplitude median is ~24% above the training cache, so it measures pipeline
shift on top of frame novelty) and FLAT (cross-domain).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True,
                        help="Existing list dir; train.txt+val.txt define the sample universe.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("scene", "view"), default="scene")
    parser.add_argument("--holdout_scene", help="Required for --mode scene.")
    parser.add_argument("--val_views", type=int, default=2,
                        help="Views held out per scene for --mode view.")
    return parser.parse_args()


def read(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key_for(sample, cache_dir):
    """Return (scene, view, frame) for a cached sample path.

    Anchored on the cache directory *name* so relative and absolute list entries
    both parse.
    """
    parts = Path(sample).parts
    marker = Path(cache_dir).name
    try:
        anchor = len(parts) - 1 - parts[::-1].index(marker)
        scene, view, frame = parts[anchor + 1], parts[anchor + 2], parts[anchor + 3]
    except (ValueError, IndexError):
        raise SystemExit(f"Cannot parse scene/view/frame from {sample!r} against cache {cache_dir}")
    return scene, view, int(Path(frame).stem)


def adjacency_leak(train_keys, val_keys):
    """Fraction of val frames with a train frame at +/-1 in the same view."""
    train_index = defaultdict(set)
    for scene, view, frame in train_keys:
        train_index[(scene, view)].add(frame)
    if not val_keys:
        return 0.0
    leaked = sum(
        1 for scene, view, frame in val_keys
        if {frame - 1, frame + 1} & train_index[(scene, view)]
    )
    return leaked / len(val_keys)


def main():
    args = parse_args()
    universe = read(args.source_dir / "train.txt") + read(args.source_dir / "val.txt")
    if not universe:
        raise SystemExit(f"Empty sample universe under {args.source_dir}")
    keyed = [(sample, key_for(sample, args.cache_dir)) for sample in universe]

    scenes = sorted({key[0] for _, key in keyed})
    if args.mode == "scene":
        if not args.holdout_scene:
            raise SystemExit("--mode scene requires --holdout_scene")
        if args.holdout_scene not in scenes:
            raise SystemExit(f"Unknown scene {args.holdout_scene!r}; available: {scenes}")
        is_val = lambda key: key[0] == args.holdout_scene  # noqa: E731
        holdout_desc = args.holdout_scene
    else:
        views_by_scene = defaultdict(set)
        for _, (scene, view, _frame) in keyed:
            views_by_scene[scene].add(view)
        held = set()
        for scene in scenes:
            ordered = sorted(views_by_scene[scene], key=lambda view: (len(view), view))
            if args.val_views >= len(ordered):
                raise SystemExit(f"--val_views {args.val_views} leaves no train views in {scene}")
            held.update((scene, view) for view in ordered[-args.val_views:])
        is_val = lambda key: (key[0], key[1]) in held  # noqa: E731
        holdout_desc = f"{args.val_views}views/scene"

    train = [sample for sample, key in keyed if not is_val(key)]
    val = [sample for sample, key in keyed if is_val(key)]
    train_keys = [key for _, key in keyed if not is_val(key)]
    val_keys = [key for _, key in keyed if is_val(key)]
    if not train or not val:
        raise SystemExit(f"Degenerate split: train={len(train)} val={len(val)}")

    shared_views = {(s, v) for s, v, _ in train_keys} & {(s, v) for s, v, _ in val_keys}
    leak = adjacency_leak(train_keys, val_keys)
    if shared_views or leak:
        raise SystemExit(f"Split is leaky: {len(shared_views)} shared views, adjacency={leak:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("train", train), ("val", val)):
        (args.output_dir / f"{name}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")

    summary = {
        "mode": args.mode,
        "holdout": holdout_desc,
        "cache_dir": str(args.cache_dir),
        "source_dir": str(args.source_dir),
        "train_samples": len(train),
        "val_samples": len(val),
        "train_scenes": sorted({s for s, _, _ in train_keys}),
        "val_scenes": sorted({s for s, _, _ in val_keys}),
        "shared_views": len(shared_views),
        "val_adjacent_frame_leak": leak,
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"mode={args.mode} holdout={holdout_desc} train={len(train)} val={len(val)} "
          f"shared_views={len(shared_views)} adjacency_leak={leak:.4f} -> {args.output_dir}")


if __name__ == "__main__":
    main()
