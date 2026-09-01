#!/usr/bin/env python3
import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth_restoration_backbones import build_depth_backbone
from eval_depth_flow_restoration import predict_batch
from train_depth_completion import move_batch_to_device
from train_depth_restoration import DepthRestorationCacheDataset


def sample_ids(manifest):
    return json.loads(Path(manifest).read_text())['samples']['test']


def load_npy(root, sample):
    return np.load(Path(root) / f'{sample}.npy').astype(np.float32)


def load_indexed(root, index_json, sample):
    rows = json.loads(Path(index_json).read_text())
    file_name = next(row['file'] for row in rows if row['sample_id'] == sample)
    path = Path(root) / file_name
    png_path = Path(root) / f'{Path(file_name).stem.zfill(10)}.png'
    if png_path.exists():
        path = png_path
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True).astype(np.float32)
    return np.asarray(Image.open(path), dtype=np.float32) * 0.00390625


@torch.no_grad()
def flow_predictions(args, samples, device):
    ckpt = torch.load(args.flow_checkpoint, map_location=device)
    saved = ckpt['args']
    paths = [str(Path(args.cache_root) / f'{sample}.npz') for sample in samples]
    kwargs = {key: saved.get(key, default) for key, default in {
        'input_mode': 'noisy', 'include_hole_distance': False, 'anchor_mode': 'noisy_ns',
        'anchor_inpaint_radius': 15, 'norm_percentiles': [5.0, 95.0],
        'min_depth_scale': 0.25, 'clip_norm_depth': 8.0, 'feature_percentile': 99.0,
        'feature_clip': 3.0, 'iq_clip': 3.0}.items()}
    dataset = DepthRestorationCacheDataset(paths, **kwargs)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model = build_depth_backbone(saved.get('backbone', 'resunet'),
        in_channels=dataset.input_channels + 1 + int(saved.get('time_channels', 16)),
        base_channels=int(saved.get('base_channels', 32)), out_channels=1,
        res_blocks=int(saved.get('res_blocks', 2)), transformer_layers=int(saved.get('transformer_layers', 2)),
        transformer_heads=int(saved.get('transformer_heads', 8)), transformer_mlp_ratio=float(saved.get('transformer_mlp_ratio', 4.0)),
        transformer_pool=int(saved.get('transformer_pool', 2))).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    out = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = predict_batch(model, batch, saved, int(saved.get('sample_steps', 8)), saved.get('eval_sampling_mode', 'endpoint'))
        for i, name in enumerate(batch['sample_name']):
            out[name] = pred[i, 0].cpu().numpy()
    return out


def main(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    samples = sample_ids(args.manifest)
    cache = Path(args.cache_root)
    predictions = {'CompletionFormer': {s: load_npy(args.completion_root, s) for s in samples},
                   'DEPTHOR': {s: load_npy(args.depthor_root, s) for s in samples},
                   'LDCM': {s: load_npy(args.ldcm_root, s) for s in samples},
                   'LingBot-Depth': {s: load_npy(args.lingbot_root, s) for s in samples},
                   'DMD3C': {s: load_indexed(args.dmd_root, args.dmd_index, s) for s in samples},
                   'OMNI-DC': {s: load_indexed(args.omni_root, args.omni_index, s) for s in samples}}
    device = torch.device(args.device)
    predictions['Ours-Flow'] = flow_predictions(args, samples, device)
    from unified_pbrt_dataset import UnifiedPbrtDataset
    class ModelArgs: pass
    model_args = ModelArgs(); model_args.rgbd_checkpoint = args.rgbd_checkpoint
    dataset = UnifiedPbrtDataset(args.manifest, 'test'); loader = DataLoader(dataset, 4, False, num_workers=0)
    sys.path.insert(0, '/data/pre_student/hcy/RGBD_imaging')
    from srresnet_unet3 import _NetG
    rgbd = _NetG().to(device); rgbd.load_state_dict(torch.load(args.rgbd_checkpoint, map_location=device)); rgbd.eval()
    predictions['RGBD-Imaging'] = {}
    with torch.no_grad():
        for batch in loader:
            iq, depth, amp = batch['iq'][:, :4].to(device), batch['depth'].to(device), batch['amplitude'].to(device)
            pred = rgbd(torch.cat((iq, depth / 10.0, amp), dim=1)) * 10.0
            for i, sid in enumerate(batch['sample_name']): predictions['RGBD-Imaging'][sid] = pred[i, 0].cpu().numpy()
    from run_lfrd2_raw9_masked_self_test import import_lfrd2
    FracDiff = import_lfrd2()
    lfrd2_args = SimpleNamespace(**json.loads(Path(args.lfrd2_args).read_text()))
    lfrd2 = FracDiff(lfrd2_args).to(device)
    lfrd2.load_state_dict(torch.load(args.lfrd2_checkpoint, map_location=device)); lfrd2.eval()
    predictions['LFRD2'] = {}
    with torch.no_grad():
        for batch in loader:
            depth = batch['depth'].to(device) / 10.0; amp = batch['amplitude'].to(device); conf = batch['confidence'].to(device)
            pred = lfrd2(depth, amp, conf)['y_pred'][0] * 10.0
            for i, sid in enumerate(batch['sample_name']): predictions['LFRD2'][sid] = pred[i, 0].cpu().numpy()
    for name, root in [('CompletionFormer', args.completion_root), ('DEPTHOR', args.depthor_root), ('LDCM', args.ldcm_root), ('LingBot-Depth', args.lingbot_root)]:
        for sid, pred in predictions[name].items():
            path = out / 'predictions' / name / f'{sid}.npy'; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, pred)
    for name in ['DMD3C', 'OMNI-DC', 'Ours-Flow', 'RGBD-Imaging', 'LFRD2']:
        for sid, pred in predictions[name].items():
            path = out / 'predictions' / name / f'{sid}.npy'; path.parent.mkdir(parents=True, exist_ok=True); np.save(path, pred)
    panels = [
        ('Input', 'Input'), ('GT (raw)', 'GT'),
        ('RGBD-Imaging', 'RGBD-Imaging'), ('CompletionFormer', 'CompletionFormer'),
        ('Ours-Flow', 'Ours-Flow'), ('LFRD2', 'LFRD2'), ('DMD3C', 'DMD3C'),
        ('OMNI-DC', 'OMNI-DC'), ('LingBot-Depth', 'LingBot-Depth'),
        ('LDCM', 'LDCM'), ('DEPTHOR', 'DEPTHOR'), ('Hole mask', 'Hole'),
    ]
    records = []
    for sid in samples:
        with np.load(cache / f'{sid}.npz') as data:
            noisy, gt = data['depth_noisy'], data['gt_depth']
            hole = data['hole_mask'] > .5
            valid_gt = data['valid_mask'] > .5
        records.append((float(hole.mean()), sid, noisy, gt, hole, valid_gt))
    records.sort(reverse=True)
    for rank, (_, sid, noisy, gt, hole, valid_gt) in enumerate(records[:args.max_samples]):
        arrays = {'Input': noisy, 'GT': gt, 'Hole': hole.astype(np.float32)}
        arrays.update({key: predictions[key][sid] for _, key in panels[2:]})
        valid_values = gt[valid_gt & np.isfinite(gt)]
        vmin, vmax = np.nanpercentile(valid_values, [2, 98])
        cmap = plt.get_cmap('turbo')
        fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True); axes = axes.ravel()
        color_image = None
        for ax, (label, key) in zip(axes, panels):
            image = arrays[key]
            if image.shape != gt.shape: image = cv2.resize(image, (gt.shape[1], gt.shape[0]))
            if key == 'Hole':
                ax.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
            else:
                color_image = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(label); ax.axis('off')
        for ax in axes[len(panels):]:
            ax.axis('off')
        fig.colorbar(color_image, ax=axes[:len(panels) - 1], shrink=0.72, pad=0.02, label='Depth (m)')
        fig.suptitle(f'{sid} | hole={hole.mean():.3f} | gt_invalid={(~valid_gt).mean():.3f} | color scale={vmin:.2f}–{vmax:.2f} m')
        figure_path = out / 'figures' / f'{rank:02d}_{sid.replace("/", "_")}.png'; figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', required=True); p.add_argument('--cache_root', required=True); p.add_argument('--output_dir', required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--max_samples', type=int, default=12)
    p.add_argument('--flow_checkpoint', required=True); p.add_argument('--rgbd_checkpoint', required=True); p.add_argument('--lfrd2_checkpoint', required=True); p.add_argument('--lfrd2_args', required=True); p.add_argument('--completion_root', required=True); p.add_argument('--depthor_root', required=True); p.add_argument('--ldcm_root', required=True); p.add_argument('--lingbot_root', required=True); p.add_argument('--dmd_root', required=True); p.add_argument('--dmd_index', required=True); p.add_argument('--omni_root', required=True); p.add_argument('--omni_index', required=True)
    main(p.parse_args())
