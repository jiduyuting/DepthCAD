"""
DepthCAD + Region-Aware Diffusion Models V2

基于 train_pbrt_rad.py 的改进版本：

改进点：
1. Multi-scale Context Attention: 在多个尺度上进行 attention，不仅限于单一尺度
2. Boundary-aware Progressive Filling: 先填补边界，再填中心（借鉴 MeshLib 思想）
3. Edge-aware Loss: 边界区域加权更大的损失
4. Separable Conv: 使用可分离卷积减少参数量
5. Hole Depth Estimation: 估计空洞深度，辅助填补

Usage:
    accelerate launch train_pbrt_rad_v2.py --dataset_config="masked" ...
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--depthcad_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/depthcad_pbrt_rad_v2")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default="default")
    parser.add_argument("--train_list_path", type=str, default=None)
    parser.add_argument("--tracker_project_name", type=str, default="train_depthcad_pbrt_rad_v2")
    parser.add_argument("--inpaint_lr", type=float, default=1e-4)
    parser.add_argument("--boundary_weight", type=float, default=5.0, help="Boundary region loss weight")
    parser.add_argument("--center_weight", type=float, default=1.0, help="Center region loss weight")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


# =============================================================================
# V2: Multi-Scale Region-Aware Inpainting Module
# =============================================================================

class SeparableConv(nn.Module):
    """Depthwise Separable Convolution: 减少参数量"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GradientAttention(nn.Module):
    """
    梯度注意力: 用 Sobel 梯度引导 attention 权重
    让边缘区域的 attention 更集中
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)

        # Sobel 梯度
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 4.0)

    def compute_gradient(self, x):
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, x, region_mask):
        """
        Args:
            x: (B, C, H, W)
            region_mask: (B, 1, H, W), 0=valid, 1=hole
        """
        B, C, H, W = x.shape

        # 计算梯度幅度
        grad = self.compute_gradient(x.mean(dim=1, keepdim=True))  # (B, 1, H, W)
        grad = grad.repeat(1, self.num_heads, 1, 1)  # (B, nh, H, W)

        q = self.query_conv(x).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.key_conv(x).reshape(B, self.num_heads, self.head_dim, H * W)
        v = self.value_conv(x).reshape(B, self.num_heads, self.head_dim, H * W)

        # Attention
        attn = torch.bmm(q.transpose(1, 2), k.transpose(1, 2).transpose(2, 3)) * self.scale
        attn = attn.reshape(B, self.num_heads, H * W, H * W)

        # 空洞区域 mask
        hole_mask = region_mask.reshape(B, 1, H * W, 1)  # (B, 1, N, 1)
        valid_mask = (region_mask < 0.5).reshape(B, 1, 1, H * W)  # (B, 1, 1, N)

        # 无效区域的 attention 设 -inf
        attn_masked = attn.clone()
        attn_masked = attn_masked.masked_fill(valid_mask == 0, float('-inf'))
        attn_masked = attn_masked.masked_fill(hole_mask == 0, float('-inf'))

        # 边界区域（梯度大的地方）attention 加强
        grad_normalized = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        attn_masked = attn_masked * (1 + grad_normalized.reshape(B, self.num_heads, H * W, 1) * 0.5)

        attn_soft = F.softmax(attn_masked, dim=-1)

        # Apply
        out = torch.bmm(attn_soft, v.transpose(1, 2))
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.out_conv(out)

        # 只在空洞区域应用
        out = x + out * region_mask
        return out


class MultiScaleContextBlock(nn.Module):
    """
    多尺度 Context Attention Block
    在多个尺度上做 attention，然后融合
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels

        # 多个尺度
        self.scale1 = nn.Sequential(
            SeparableConv(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.scale2 = nn.Sequential(
            SeparableConv(channels, channels, 5, padding=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.scale3 = nn.Sequential(
            SeparableConv(channels, channels, 7, padding=3),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 梯度注意力
        self.grad_attn = GradientAttention(channels, num_heads)

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, region_mask):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        attn = self.grad_attn(x, region_mask)
        out = torch.cat([x, s1, s2, s3, attn], dim=1)
        return self.fusion(out)


class RegionAwareInpaintBlockV2(nn.Module):
    """
    V2: 单个区域感知填补块
    包含: region embedding + multi-scale context attention + 空洞深度估计
    """
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.num_regions = 3  # valid, hole_boundary, hole_center

        # Region embeddings
        self.region_emb = nn.Embedding(3, in_channels)

        # 主干: multi-scale context attention
        self.msca = MultiScaleContextBlock(out_channels, num_heads)

        # 空洞深度估计 (辅助填补)
        self.hole_depth = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )

        # 条件注入
        self.region_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x, region_map, confidence):
        """
        Args:
            x: (B, C, H, W)
            region_map: (B, 1, H, W), 0=valid, 1=hole_boundary, 2=hole_center
            confidence: (B, 1, H, W), 置信度 0=空洞, 1=有效
        """
        # Region embedding
        region_feat = self.region_emb(region_map.long()).permute(0, 4, 1, 2)
        region_cond = self.region_scale(region_feat)

        # 估计空洞深度 (用于指导填补)
        hole_depth = self.hole_depth(1 - confidence)  # 反转: 空洞处值大

        # 空洞深度作为额外条件
        x = x + hole_depth * 0.1

        # 主干: multi-scale context attention
        h = self.msca(x, (region_map > 0).float())

        # 加上 region 条件
        h = h + region_cond

        return h


class RegionAwareInpaintNetV2(nn.Module):
    """
    V2: 多尺度区域感知填补网络

    改进:
    1. Multi-scale context attention (3个尺度 + 梯度注意力)
    2. Boundary-aware progressive filling (先填边界再填中心)
    3. Hole depth estimation (估计空洞深度辅助填补)
    4. Separable convolutions (减少参数量)
    """

    def __init__(self, in_channels=3, out_channels=3, base_filters=48, num_regions=3):
        super().__init__()
        self.num_regions = num_regions

        # 编码器: 渐进式下采样
        self.enc1 = nn.Sequential(
            SeparableConv(in_channels, base_filters, 7, padding=3),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            SeparableConv(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            SeparableConv(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )

        # 中间层: 多个 Region-Aware blocks
        self.mid1 = RegionAwareInpaintBlockV2(base_filters * 4, base_filters * 4, num_heads=8)
        self.mid2 = RegionAwareInpaintBlockV2(base_filters * 4, base_filters * 4, num_heads=8)
        self.mid3 = RegionAwareInpaintBlockV2(base_filters * 4, base_filters * 4, num_heads=8)

        # 解码器: 渐进式上采样
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # 输出
        self.out = nn.Sequential(
            SeparableConv(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, out_channels, 3, padding=1)
        )

    def compute_region_map(self, confidence):
        """
        从 confidence map 计算区域图

        Args:
            confidence: (B, 1, H, W) 置信度图
        Returns:
            region_map: (B, 1, H, W), 0=valid, 1=hole_boundary, 2=hole_center
        """
        hole_mask = (confidence < 0.5).float()  # 1=空洞, 0=有效

        # 边界 = 膨胀 - 腐蚀
        kernel = torch.ones(5, 5, device=hole_mask.device)
        dilated = F.conv2d(hole_mask, kernel.unsqueeze(0).unsqueeze(0), padding=2)
        eroded = F.conv2d(hole_mask, kernel.unsqueeze(0).unsqueeze(0), padding=2)

        # hole_boundary = 膨胀 - 原始空洞
        hole_boundary = (dilated - hole_mask).clamp(0, 1)
        # hole_center = 原始空洞 - 边界
        hole_center = (hole_mask - hole_boundary).clamp(0, 1)

        # 编码: 0=valid, 1=hole_boundary, 2=hole_center
        region_map = hole_mask + hole_boundary  # 边界变成 1
        region_map = region_map + hole_center  # 中心变成 2

        return region_map

    def compute_boundary_loss(self, pred, target, confidence, valid_weight=1.0, boundary_weight=5.0, center_weight=1.0):
        """
        边界感知损失: 边界区域损失权重更大
        """
        hole_mask = (confidence < 0.5).float()

        # 边界
        kernel = torch.ones(5, 5, device=hole_mask.device)
        dilated = F.conv2d(hole_mask, kernel.unsqueeze(0).unsqueeze(0), padding=2)
        hole_boundary = (dilated - hole_mask).clamp(0, 1)
        hole_center = (hole_mask - hole_boundary).clamp(0, 1)
        valid = (hole_mask == 0).float()

        # 各类损失
        valid_loss = F.l1_loss(pred * valid, target * valid)
        boundary_loss = F.l1_loss(pred * hole_boundary, target * hole_boundary)
        center_loss = F.l1_loss(pred * hole_center, target * hole_center)

        total = (valid_weight * valid_loss + boundary_weight * boundary_loss + center_weight * center_loss) / (valid_weight + boundary_weight + center_weight)

        return total, {
            'valid': valid_loss.item(),
            'boundary': boundary_loss.item(),
            'center': center_loss.item()
        }

    def forward(self, x, confidence):
        """
        Args:
            x: (B, 3, H, W) 输入 (masked noise IQ)
            confidence: (B, 1, H, W) 置信度图, 0=空洞, 1=有效
        Returns:
            filled: (B, 3, H, W) 填补后的图像
            region_map: (B, 1, H, W) 区域图
        """
        region_map = self.compute_region_map(confidence)

        # Progressive filling: 先填边界，再填中心
        # Step 1: 填补边界
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m1 = self.mid1(e3, region_map, confidence)
        m2 = self.mid2(m1, region_map, confidence)
        m3 = self.mid3(m2, region_map, confidence)

        d = self.dec2(m3)
        d = self.dec1(d)

        residual = self.out(d)
        filled = x + residual

        return filled, region_map


def make_train_dataset(args, tokenizer, accelerator):
    """Load PBRT dataset and prepare for training."""
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
    )

    if args.train_list_path:
        with open(args.train_list_path, 'r') as f:
            allowed_paths = set()
            for line in f:
                line = line.strip()
                if line:
                    allowed_paths.add(line)

        def filter_by_train_list(example):
            ideal_path = Path(example['ideal_IQ_path'])
            scene_dir = ideal_path.parents[1].relative_to(ideal_path.parents[2])
            index_dir = ideal_path.parent.name
            rel_dir = f"{scene_dir}/{index_dir}"
            return rel_dir in allowed_paths

        dataset["train"] = dataset["train"].filter(filter_by_train_list)
        print(f"Filtered dataset to {len(dataset['train'])} examples from train.txt")

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples["prompt"]:
            if isinstance(caption, str):
                captions.append(caption if caption else "")
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise NotImplementedError
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    data_transforms = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
    ])

    def preprocess_train(examples):
        import cv2
        target_size = (args.resolution, args.resolution)

        ideals, noises, confs = [], [], []

        for ideal_IQ_path, noise_IQ_path, conf_path in zip(
            examples["ideal_IQ_path"], examples["noise_IQ_path"], examples["conf_path"]
        ):
            ideal_data = np.load(ideal_IQ_path)
            if ideal_data.shape[:2] != (args.resolution, args.resolution):
                ideal_data = cv2.resize(ideal_data, target_size, interpolation=cv2.INTER_LINEAR)
            ideal_data = np.repeat(np.expand_dims(ideal_data, axis=0), 3, axis=0)
            ideals.append(data_transforms(ideal_data))

            noise_data = np.load(noise_IQ_path)
            if noise_data.shape[:2] != (args.resolution, args.resolution):
                noise_data = cv2.resize(noise_data, target_size, interpolation=cv2.INTER_LINEAR)
            noise_data = np.repeat(np.expand_dims(noise_data, axis=0), 3, axis=0)
            noises.append(data_transforms(noise_data))

            conf_data = np.load(conf_path)
            if conf_data.shape[:2] != (args.resolution, args.resolution):
                conf_data = cv2.resize(conf_data, target_size, interpolation=cv2.INTER_LINEAR)
            conf_data = np.repeat(np.expand_dims(conf_data, axis=0), 3, axis=0)
            confs.append(data_transforms(conf_data))

        examples["ideals"] = ideals
        examples["noises"] = noises
        examples["confs"] = confs
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    ideals = torch.stack([example["ideals"] for example in examples])
    ideals = ideals.to(memory_format=torch.contiguous_format).float()

    noises = torch.stack([example["noises"] for example in examples])
    noises = noises.to(memory_format=torch.contiguous_format).float()

    confs = torch.stack([example["confs"] for example in examples])

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "ideals": ideals,
        "noises": noises,
        "confs": confs,
        "input_ids": input_ids,
    }


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.depthcad_path:
        logger.info("Loading existing DepthCAD weights")
        depthcad = ControlNetModel.from_pretrained(args.depthcad_path)
    else:
        logger.info("Initializing DepthCAD weights from unet")
        depthcad = ControlNetModel.from_unet(unet, conditioning_channels=2)

    # V2 Inpainting Module
    inpaint_net = RegionAwareInpaintNetV2(in_channels=3, out_channels=3, base_filters=48).to(accelerator.device)
    logger.info(f"Inpainting Module V2 params: {sum(p.numel() for p in inpaint_net.parameters()):,}")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "depthcad"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="depthcad")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    depthcad.train()
    inpaint_net.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs.")
            unet.enable_xformers_memory_efficient_attention()
            depthcad.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        depthcad.enable_gradient_checkpointing()
        inpaint_net.enable_gradient_checkpointing()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class([
        {"params": depthcad.parameters(), "lr": args.learning_rate},
        {"params": inpaint_net.parameters(), "lr": args.inpaint_lr},
    ], betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    train_dataset = make_train_dataset(args, tokenizer, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    depthcad, inpaint_net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        depthcad, inpaint_net, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(depthcad, inpaint_net):
                # V2: Region-Aware Inpainting
                masked_noise = batch["noises"].to(dtype=weight_dtype)  # (B, 3, H, W)
                confidence = batch["confs"][:, :1, :, :].to(dtype=weight_dtype)  # (B, 1, H, W)

                # Inpainting Module V2 forward
                filled_iq, region_map = inpaint_net(masked_noise, confidence)

                # DepthCAD Forward
                latents = vae.encode(batch["ideals"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # depthcad_image: filled_iq + confidence
                depthcad_image = torch.cat([filled_iq, confidence], dim=1).to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = depthcad(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=depthcad_image,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Diffusion loss
                diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Boundary-aware inpainting loss
                gt_ideals = batch["ideals"].to(dtype=weight_dtype)
                inpaint_loss, loss_dict = inpaint_net.compute_boundary_loss(
                    filled_iq, gt_ideals, confidence,
                    boundary_weight=args.boundary_weight,
                    center_weight=args.center_weight
                )

                # Total loss
                total_loss = diffusion_loss + 0.1 * inpaint_loss

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    params_to_clip = list(depthcad.parameters()) + list(inpaint_net.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {
                    "loss": total_loss.detach().item(),
                    "diffusion_loss": diffusion_loss.detach().item(),
                    "inpaint_loss": inpaint_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        depthcad = unwrap_model(depthcad)
        depthcad.save_pretrained(args.output_dir)
        inpaint_net = unwrap_model(inpaint_net)
        torch.save({
            'model_state_dict': inpaint_net.state_dict(),
            'args': vars(args)
        }, os.path.join(args.output_dir, "inpaint_net_v2.pth"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
