import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

try:
    from pbrt_dataset.preprocess import load_raw as load_raw_pbrt, compute_gradient_confidence
    PBRT_AVAILABLE = True
except ImportError:
    PBRT_AVAILABLE = False
try:
    from flat_dataset.preprocess import load_raw as load_raw_flat, compute_gradient_confidence as compute_gradient_confidence_flat
    FLAT_AVAILABLE = True
except ImportError:
    FLAT_AVAILABLE = False

from depth_estimator import DepthEstimatorTorch


# ---- RegionAwareInpaintBlock (from train_pbrt_rad.py) ----
class RegionAwareAttention(nn.Module):
    def __init__(self, channels, num_heads=8, chunk_size=256):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        self.chunk_size = chunk_size
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask):
        B, C, H, W = x.shape
        valid_mask = (mask < 0.5).float()
        hole_mask = (mask >= 0.5).float()
        if valid_mask.sum() == 0 or hole_mask.sum() == 0:
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
            h_idx = hole_idx[b]
            v_idx = valid_idx[b]
            if len(h_idx) == 0 or len(v_idx) == 0:
                continue
            q_b = q[b, :, h_idx, :]
            k_b = k[b, :, v_idx, :]
            v_b = v[b, :, v_idx, :]
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
        hole_mask = (region_map > 0).float()
        h = self.attention(h, hole_mask)
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
        hole_mask = (confidence < 0.5).float()
        kernel = torch.ones(1, 1, 3, 3, device=confidence.device)
        hole_boundary = F.conv2d(hole_mask, kernel, padding=1)
        hole_boundary = (hole_boundary > 0) & (hole_boundary < 9)
        hole_boundary = hole_boundary.float()
        hole_center = hole_mask - hole_boundary
        hole_center = torch.clamp(hole_center, 0, 1)
        region_map = hole_boundary + hole_center * 2
        return region_map

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
        residual = self.out_conv(d)
        filled = x + residual
        return filled


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1"
    )
    parser.add_argument(
        "--depthcad_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--inpaint_net_path",
        type=str,
        default=None,
        help="Path to inpaint_net.pth (inside checkpoint directory)"
    )
    parser.add_argument(
        "--noise_IQ_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--noise_depth_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="pbrt",
        choices=["pbrt", "flat"],
        help="Dataset type: 'pbrt' for .npy files (9 channels) or 'flat' for binary files"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[240, 320],
        help="Target size for resizing (height, width)"
    )

    args = parser.parse_args()
    return args


def inference(pipe, inpaint_net, noise, conf, scale, target_size=(240, 320), device="cuda"):
    """
    Run inference to predict IQ channels.
    内部强制使用 512 进行推理，最后还原回 target_size

    pipeline expects 2-channel conditioning (filled_iq 1ch + confidence 1ch).
    inpaint_net fills the masked noise first.
    """
    infer_h, infer_w = 512, 512

    # 1. Prepare inpaint_net input: 1ch noise
    noise_1ch = noise[0]  # (H, W)
    noise_1ch_resized = cv2.resize(noise_1ch, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)  # (512, 512)

    # 2. Resize confidence map
    conf_resized = cv2.resize(conf, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

    # 3. inpaint_net forward: masked_noise (1ch) + confidence -> filled_iq (1ch)
    with torch.no_grad():
        masked_noise_tensor = torch.from_numpy(noise_1ch_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)
        conf_tensor = torch.from_numpy(conf_resized).unsqueeze(0).unsqueeze(0)  # (1, 1, infer_h, infer_w)
        filled_iq = inpaint_net(masked_noise_tensor.to(device), conf_tensor.to(device))  # (1, 3, infer_h, infer_w)
        print(f"[DEBUG] filled_iq stats: mean={filled_iq.mean().item():.6f}, std={filled_iq.std().item():.6f}, min={filled_iq.min().item():.6f}, max={filled_iq.max().item():.6f}")
        filled_iq = filled_iq.squeeze(0).cpu().numpy()  # (1, infer_h, infer_w)

    # 4. Concatenate: filled_iq (1ch) + confidence (1ch) -> 2ch ControlNet condition
    depthcad_cond = np.concatenate([filled_iq, np.expand_dims(conf_resized, 0)], axis=0)  # (2, infer_h, infer_w)
    depthcad_cond = torch.from_numpy(depthcad_cond).unsqueeze(0)  # (1, 2, infer_h, infer_w)

    # 5. Initialize result array
    pred_IQs_infer = np.zeros((6, infer_h, infer_w))

    for i in range(6):
        noise_resized = cv2.resize(noise[i], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

        prompt = ""
        generator = torch.manual_seed(42)

        pred_IQ = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=depthcad_cond,
            height=infer_h,
            width=infer_w
        ).images[0]

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0
        pred_IQ = 2 * pred_IQ - 1

        pred_IQs_infer[i] = pred_IQ * scale

    # 6. Resize back to target size
    target_h, target_w = target_size
    reshaped_IQs = np.zeros((6, target_h, target_w), dtype=np.float32)

    for i in range(6):
        reshaped_IQs[i, :, :] = cv2.resize(
            pred_IQs_infer[i, :, :],
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    return reshaped_IQs


if __name__ == '__main__':
    args = parse_args()
    base_model_path = args.pretrained_model_name_or_path
    depthcad_path = args.depthcad_path
    inpaint_net_path = args.inpaint_net_path
    noise_file = args.noise_IQ_file
    noise_depth_file = args.noise_depth_file
    out_file = args.out_file
    dataset_type = args.dataset_type
    target_size = tuple(args.target_size)

    print("=" * 60)
    print("DepthCAD Inference v2 (with InpaintNet)")
    print("=" * 60)
    print(f"Dataset type: {dataset_type}")
    print(f"Target size: {target_size}")
    print(f"Noise IQ file: {noise_file}")
    print(f"Noise depth file: {noise_depth_file}")
    print(f"Output file: {out_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data based on dataset type
    if dataset_type == "pbrt":
        if not PBRT_AVAILABLE:
            raise ImportError("pbrt_dataset.preprocess not available. Please check your imports.")

        print("\nLoading PBRT data...")
        noise_result = load_raw_pbrt(noise_file, target_size=target_size, sqrt_in=True)
        if isinstance(noise_result, tuple) or isinstance(noise_result, list):
            noise, amp_mask = noise_result
        else:
            noise = noise_result
            amp_mask = None
        print(f"Loaded noise IQ shape: {noise.shape}")

        noise_depth = np.load(noise_depth_file)
        noise_depth = cv2.resize(noise_depth.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        confidence = compute_gradient_confidence(noise_depth)
        print(f"Computed confidence map shape: {confidence.shape}")

        scale = max(noise.max(), abs(noise.min()), 1e-8)
        print(f"Scale factor: {scale:.4f}")
        noise /= scale

    elif dataset_type == "flat":
        if not FLAT_AVAILABLE:
            raise ImportError("flat_dataset.preprocess not available. Please check your imports.")

        print("\nLoading FLAT data...")
        noise_result = load_raw_flat(noise_file)
        if isinstance(noise_result, tuple) or isinstance(noise_result, list):
            noise, amp_mask = noise_result
        else:
            noise = noise_result
            amp_mask = None
        print(f"Loaded noise IQ shape: {noise.shape}")

        if noise.shape[1:] != target_size:
            reshaped_noise = np.zeros((6, target_size[0], target_size[1]), dtype=np.float32)
            for i in range(6):
                reshaped_noise[i] = cv2.resize(noise[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            noise = reshaped_noise
            print(f"Resized noise IQ to: {noise.shape}")

        noise_depth = np.load(noise_depth_file)
        noise_depth = cv2.resize(noise_depth.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        confidence = compute_gradient_confidence_flat(noise_depth)
        print(f"Computed confidence map shape: {confidence.shape}")

        scale = max(noise.max(), abs(noise.min()), 1e-8)
        print(f"Scale factor: {scale:.4f}")
        noise /= scale
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load pipeline
    print("\nLoading DepthCAD ControlNet model...")
    depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=depthcad, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    print("ControlNet loaded successfully!")

    # Load inpaint_net
    print("\nLoading InpaintNet model...")
    inpaint_net = RegionAwareInpaintNet(in_channels=1, out_channels=1, base_filters=64).to(device)
    state_dict = torch.load(inpaint_net_path, map_location=device)
    inpaint_net.load_state_dict(state_dict)
    inpaint_net.eval()
    print("InpaintNet loaded successfully!")

    # Run inference
    print("\nRunning inference...")
    pred_IQs = inference(pipe, inpaint_net, noise, confidence, scale, target_size=target_size, device=device)

    # Convert IQ to depth
    print("\nConverting IQ to depth...")
    estimator = DepthEstimatorTorch(device=device)
    pred_IQs_tensor = torch.from_numpy(pred_IQs).unsqueeze(0).to(device)
    depth_tensor = estimator.process(pred_IQs_tensor)
    depth = depth_tensor.squeeze(0).cpu().numpy()
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}] meters")

    # Save output
    print(f"\nSaving depth to {out_file}...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.save(out_file, depth)
    print("Done!")
