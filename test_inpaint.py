"""
验证 Inpaint 模块的空洞填补能力

测试 inpaint_net 能否将带空洞的 noise IQ 填补成接近 ideal IQ。
只验证填补能力，不涉及 ControlNet 去噪效果。

Usage:
    python test_inpaint.py --inpaint_net_path output/.../checkpoint-XXX/inpaint_net/pytorch_model.bin
"""

import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# Inpaint Network (与 train_pbrt_rad.py 完全一致，1 通道版本)
# =============================================================================

class RegionAwareAttention(nn.Module):
    def __init__(self, channels, num_heads=8, chunk_size=256):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.chunk_size = chunk_size
        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask):
        B, C, H, W = x.shape
        hole_mask = (mask >= 0.5).float()
        valid_mask = 1 - hole_mask
        if hole_mask.sum() == 0 or valid_mask.sum() == 0:
            return x
        hole_flat = hole_mask.reshape(B, -1)
        valid_flat = valid_mask.reshape(B, -1)
        hole_idx = [torch.where(hole_flat[b] > 0.5)[0] for b in range(B)]
        valid_idx = [torch.where(valid_flat[b] > 0.5)[0] for b in range(B)]
        if all(len(v) == 0 for v in valid_idx) or all(len(h) == 0 for h in hole_idx):
            return x
        q = self.query_conv(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        k = self.key_conv(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        v = self.value_conv(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.zeros_like(q)
        for b in range(B):
            h_idx, v_idx = hole_idx[b], valid_idx[b]
            if len(h_idx) == 0 or len(v_idx) == 0:
                continue
            q_b, k_b, v_b = q[b, :, h_idx, :], k[b, :, v_idx, :], v[b, :, v_idx, :]
            for start in range(0, len(h_idx), self.chunk_size):
                end = min(start + self.chunk_size, len(h_idx))
                q_chunk = q_b[:, start:end, :]
                attn_chunk = torch.einsum('nhd,nvd->nhv', q_chunk, k_b) * self.scale
                attn_chunk = F.softmax(attn_chunk, dim=-1)
                out_chunk = torch.einsum('nhv,nvd->nhd', attn_chunk, v_b)
                out[b, :, h_idx[start:end], :] = out_chunk
        out = out.transpose(1, 2).reshape(B, H * W, C).permute(0, 2, 1).reshape(B, C, H, W)
        out = self.out_conv(out)
        out = x + out * hole_mask
        return out


class RegionAwareInpaintBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.region_emb = nn.Embedding(3, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = RegionAwareAttention(out_channels, num_heads)
        self.region_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x, region_map):
        region_feat = self.region_emb(region_map.long())
        region_feat = region_feat.squeeze(1).permute(0, 3, 1, 2)
        region_cond = self.region_scale(region_feat)
        h = self.conv1(x)
        h = self.bn1(h)
        h = h + region_cond
        h = F.relu(h, inplace=True)
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h, inplace=True)
        h = self.attention(h, (region_map > 0).float())
        return h


class RegionAwareInpaintNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, num_regions=3):
        super().__init__()
        self.num_regions = num_regions
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 7, padding=3),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        self.middle1 = RegionAwareInpaintBlock(base_filters * 4, base_filters * 4, num_heads=8)
        self.middle2 = RegionAwareInpaintBlock(base_filters * 4, base_filters * 4, num_heads=8)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 4, 4, stride=2, padding=1)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(base_filters * 4 + base_filters * 2, base_filters * 2, 3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters * 2, 4, stride=2, padding=1)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_filters * 2 + base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, out_channels, 3, padding=1)
        )

    def compute_region_map(self, confidence):
        # confidence: (B, 1, H, W)
        hole_mask = (confidence < 0.5).float()
        kernel = torch.ones(3, 3, device=hole_mask.device)
        hole_boundary = F.conv2d(hole_mask, kernel.unsqueeze(0).unsqueeze(0), padding=1)
        hole_boundary = (hole_boundary > 0) & (hole_boundary < 9)
        hole_center = torch.clamp(hole_mask - hole_boundary.float(), 0, 1)
        return hole_boundary.float() + hole_center * 2

    def forward(self, x, confidence):
        region_map = self.compute_region_map(confidence)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        region_map_e3 = F.interpolate(region_map, size=e3.shape[2:], mode='nearest')
        m = self.middle1(e3, region_map_e3)
        m = self.middle2(m, region_map_e3)
        d = self.decoder2(torch.cat([self.up2(m), e2], dim=1))
        d = self.decoder1(torch.cat([self.up1(d), e1], dim=1))
        return x + self.out_conv(d)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpaint_net_path", type=str, required=True,
                        help="Path to inpaint_net.pth checkpoint")
    parser.add_argument("--ideal_path", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/ideal_IQ_masked/bathroom/1/100_A.npy")
    parser.add_argument("--noise_path", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/noise_IQ_masked/bathroom/1/100_A.npy")
    parser.add_argument("--conf_path", type=str,
                        default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data/confidence_masked/bathroom/1/100.npy")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    ideal = np.load(args.ideal_path)
    noise = np.load(args.noise_path)
    conf = np.load(args.conf_path)

    # Resize if needed
    if ideal.shape[:2] != (args.resolution, args.resolution):
        ideal = cv2.resize(ideal, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
    if noise.shape[:2] != (args.resolution, args.resolution):
        noise = cv2.resize(noise, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
    if conf.shape[:2] != (args.resolution, args.resolution):
        conf = cv2.resize(conf, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)

    # 归一化（与训练一致）
    scale = max(noise.max(), abs(noise.min()), 1e-8)
    noise_norm = noise / scale
    ideal_norm = ideal / scale

    # 加载 inpaint_net
    inpaint_net = RegionAwareInpaintNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    state_dict = torch.load(args.inpaint_net_path, map_location=device)
    inpaint_net.load_state_dict(state_dict)
    inpaint_net.eval()
    print(f"Loaded inpaint_net from {args.inpaint_net_path}")

    # 推理
    noise_tensor = torch.from_numpy(noise_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    conf_tensor = torch.from_numpy(conf).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        filled = inpaint_net(noise_tensor, conf_tensor).squeeze().cpu().numpy()

    # 分析
    low_conf = conf < 0.5
    high_conf = conf >= 0.5

    print("\n" + "=" * 60)
    print("Inpaint 填补效果验证")
    print("=" * 60)
    print(f"\n【全局】")
    print(f"  noise vs ideal RMSE: {np.sqrt(((noise_norm - ideal_norm)**2).mean()):.6f}")
    print(f"  filled vs ideal RMSE: {np.sqrt(((filled - ideal_norm)**2).mean()):.6f}")

    if low_conf.sum() > 0:
        print(f"\n【低置信度区域（空洞）】({low_conf.sum()} pixels)")
        print(f"  noise vs ideal RMSE: {np.sqrt(((noise_norm[low_conf] - ideal_norm[low_conf])**2).mean()):.6f}")
        print(f"  filled vs ideal RMSE: {np.sqrt(((filled[low_conf] - ideal_norm[low_conf])**2).mean()):.6f}")

    if high_conf.sum() > 0:
        print(f"\n【高置信度区域（有效）】({high_conf.sum()} pixels)")
        print(f"  noise vs ideal RMSE: {np.sqrt(((noise_norm[high_conf] - ideal_norm[high_conf])**2).mean()):.6f}")
        print(f"  filled vs ideal RMSE: {np.sqrt(((filled[high_conf] - ideal_norm[high_conf])**2).mean()):.6f}")

    improvement = np.sqrt(((noise_norm[low_conf] - ideal_norm[low_conf])**2).mean()) - \
                   np.sqrt(((filled[low_conf] - ideal_norm[low_conf])**2).mean())
    print(f"\n【空洞区域 RMSE 改善量】: {improvement:.6f}")
    if improvement > 0:
        print("  → inpaint 确实有填补效果！")
    else:
        print("  → inpaint 没有改善，甚至更差")

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(ideal_norm, cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('Ideal IQ')
    axes[0].axis('off')

    axes[1].imshow(noise_norm, cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title('Noisy IQ')
    axes[1].axis('off')

    axes[2].imshow(filled, cmap='gray', vmin=-1, vmax=1)
    axes[2].set_title('Filled IQ')
    axes[2].axis('off')

    axes[3].imshow(conf, cmap='gray')
    axes[3].set_title('Confidence')
    axes[3].axis('off')

    diff_map = np.abs(filled - ideal_norm)
    im = axes[4].imshow(diff_map, cmap='hot')
    axes[4].set_title('|filled - ideal|')
    axes[4].axis('off')
    plt.colorbar(im, ax=axes[4])

    plt.tight_layout()
    out_png = args.inpaint_net_path.replace('.bin', '_inpaint_eval.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n可视化保存到: {out_png}")


if __name__ == '__main__':
    main()
