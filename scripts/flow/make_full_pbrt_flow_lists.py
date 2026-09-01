import argparse
import json
from pathlib import Path


def cache_path(root, sample):
    scene, view, frame = sample.split("/")
    return root / scene / view / f"{frame}.npz"


def write_split(root, samples, output):
    paths = [cache_path(root, sample) for sample in samples]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cache files; first: {missing[0]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(path) for path in paths) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Create canonical Flow train/val lists from a PBRT manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("output/full_pbrt_manifest_seed123.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/full_pbrt_flow_lists"))
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    train_root = Path(payload["train_cache"])
    train_paths = write_split(train_root, payload["samples"]["train"], args.output_dir / "train.txt")
    val_paths = write_split(train_root, payload["samples"]["val"], args.output_dir / "val.txt")

    test_root = Path(payload["holdout_cache"])
    test_paths = [cache_path(test_root, sample) for sample in payload["samples"]["test"]]
    missing_test = [str(path) for path in test_paths if not path.is_file()]
    if missing_test:
        raise FileNotFoundError(f"Missing {len(missing_test)} test cache files; first: {missing_test[0]}")
    (args.output_dir / "test.txt").write_text("\n".join(str(path) for path in test_paths) + "\n")

    print(json.dumps({
        "manifest": str(args.manifest),
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
