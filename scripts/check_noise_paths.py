#!/usr/bin/env python3
import os

TEST_LIST_PATH="/data/pre_student/hcy/pbrt/list/test.txt"
NOISE_IQ_DIR="/data/pre_student/hcy/pbrt/noise"
NOISE_DEPTH_DIR="/data/pre_student/hcy/pbrt/noise_depth"


def resolve_file(path):
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + '.npy'):
        return path + '.npy'
    if os.path.isdir(path):
        candidates = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.npy')]
        if candidates:
            return candidates[0]
        for d in sorted(os.listdir(path)):
            sub = os.path.join(path, d)
            if os.path.isdir(sub):
                subc = [os.path.join(sub, f) for f in sorted(os.listdir(sub)) if f.endswith('.npy')]
                if subc:
                    return subc[0]
    nested1 = os.path.join(path, '1.npy')
    if os.path.isfile(nested1):
        return nested1
    return None


if not os.path.isfile(TEST_LIST_PATH):
    print(f"Test list not found: {TEST_LIST_PATH}")
    raise SystemExit(1)

with open(TEST_LIST_PATH, 'r') as f:
    samples = [l.strip() for l in f if l.strip()]

missing = []
for s in samples:
    iq = os.path.join(NOISE_IQ_DIR, s)
    depth = os.path.join(NOISE_DEPTH_DIR, s + '.npy')
    iq_res = resolve_file(iq)
    depth_res = resolve_file(depth[:-4]) if depth.endswith('.npy') else resolve_file(depth)
    if iq_res is None or depth_res is None:
        missing.append((s, iq_res, depth_res))
        print(f"MISSING: {s} -> IQ={iq_res}, DEPTH={depth_res}")
    else:
        print(f"OK: {s} -> IQ={iq_res}, DEPTH={depth_res}")

print('\nSummary:')
print(f"Total samples: {len(samples)}")
print(f"Missing entries: {len(missing)}")

for s, iq_res, depth_res in missing[:50]:
    print(f" - {s}: IQ={iq_res}, DEPTH={depth_res}")
