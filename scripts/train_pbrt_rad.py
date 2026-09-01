"""
DepthCAD + Region-Aware Inpainting Training Script

基于 train_pbrt.py，加入 RAD-inspired 的区域感知填补模块。

改进:
1. Region-Aware Inpainting Module: 将图像分为 hole_center, hole_boundary, valid 三个区域
2. Multi-scale region embeddings 注入
3. Context Attention: 从有效区域复制特征到空洞区域
4. 填补后的 IQ 与 confidence 拼接后送入 ControlNet

Usage:
    accelerate launch train_pbrt_rad.py --dataset_config="masked" ...
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
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--depthcad_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/depthcad_pbrt_rad",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='Scheduler type: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to. Supported platforms are "tensorboard" (default), "wandb" and "comet_ml".',
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Save more memory by using setting grads to None instead of zero.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="default",
        help="Config name of the Dataset. 'default' (without mask), 'masked' (with amplitude mask).",
    )
    parser.add_argument(
        "--train_list_path",
        type=str,
        default=None,
        help="Path to a .txt file containing list of allowed paths.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_depthcad_pbrt_rad",
        help="The project_name argument passed to Accelerator.init_trackers.",
    )
    # 新增: Inpainting Module 相关参数
    parser.add_argument(
        "--inpaint_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the inpainting module.",
    )
    parser.add_argument(
        "--freeze_depthcad",
        action="store_true",
        help="Freeze DepthCAD weights, only train inpainting module.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


# =============================================================================
# RAD-Inspired Region-Aware Inpainting Module
# =============================================================================

class RegionAwareAttention(nn.Module):
    """
    Context Attention: 从有效区域复制特征到空洞区域
    类似于 RAD 的 context reasoning 机制

    使用 chunked sparse attention，只计算 hole->valid 的注意力，
    避免 O(N^2) 显存开销。
    """

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
        """
        Args:
            x: (B, C, H, W) 输入特征
            mask: (B, 1, H, W) 掩码, 1=空洞, 0=有效
        Returns:
            attended: (B, C, H, W) 注意后的特征
        """
        B, C, H, W = x.shape

        valid_mask = (mask < 0.5).float()  # (B, 1, H, W), 1=有效
        hole_mask = (mask >= 0.5).float()   # (B, 1, H, W), 1=空洞

        if valid_mask.sum() == 0 or hole_mask.sum() == 0:
            return x

        # 将 mask 展平为 (B, N)
        hole_flat = hole_mask.reshape(B, -1)  # (B, N)
        valid_flat = valid_mask.reshape(B, -1)  # (B, N)

        # 找出每个 batch 中的空洞位置和有效位置的索引
        hole_idx = [torch.where(hole_flat[b] > 0.5)[0] for b in range(B)]
        valid_idx = [torch.where(valid_flat[b] > 0.5)[0] for b in range(B)]

        # 如果有效区域太少，直接返回
        if all(len(v) == 0 for v in valid_idx) or all(len(h) == 0 for h in hole_idx):
            return x

        # 提取 q, k, v 并重排为 (B, N, C)
        q = self.query_conv(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        k = self.key_conv(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        v = self.value_conv(x).permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 分头: (B, N, C) -> (B, N, nh, hd) -> (B, nh, N, hd)
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        # 初始化输出 (只在空洞位置写入)
        out = torch.zeros_like(q)  # (B, nh, N, hd)

        # Chunked attention: 每次处理一批 hole query
        for b in range(B):
            h_idx = hole_idx[b]   # 所有空洞位置的索引
            v_idx = valid_idx[b]  # 所有有效位置的索引
            if len(h_idx) == 0 or len(v_idx) == 0:
                continue

            q_b = q[b, :, h_idx, :]   # (nh, num_holes, hd)
            k_b = k[b, :, v_idx, :]   # (nh, num_valid, hd)
            v_b = v[b, :, v_idx, :]   # (nh, num_valid, hd)

            # 分块处理空洞 query，避免显存爆炸
            for start in range(0, len(h_idx), self.chunk_size):
                end = min(start + self.chunk_size, len(h_idx))
                q_chunk = q_b[:, start:end, :]  # (nh, chunk_size, hd)

                # 注意力分数: (nh, chunk, hd) @ (nh, valid, hd)^T -> (nh, chunk, valid)
                attn_chunk = torch.einsum('nhd,nvd->nhv', q_chunk, k_b) * self.scale
                attn_chunk = F.softmax(attn_chunk, dim=-1)

                # 输出: (nh, chunk, valid) @ (nh, valid, hd) -> (nh, chunk, hd)
                out_chunk = torch.einsum('nhv,nvd->nhd', attn_chunk, v_b)
                out[b, :, h_idx[start:end], :] = out_chunk

        # 重组为 (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, H * W, C).permute(0, 2, 1).reshape(B, C, H, W)
        out = self.out_conv(out)
        out = x + out * hole_mask

        return out


class RegionAwareInpaintBlock(nn.Module):
    """
    单个 RAD-inspired 区域感知填补块
    包含: region embedding + conv + context attention
    """

    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()

        # Region embeddings: hole_center, hole_boundary, valid
        self.region_emb = nn.Embedding(3, in_channels)

        # 主干卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Region-aware attention
        self.attention = RegionAwareAttention(out_channels, num_heads)

        # 条件注入 (region embedding)
        self.region_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x, region_map):
        """
        Args:
            x: (B, C, H, W) 输入特征
            region_map: (B, 1, H, W) 区域图, 0=valid, 1=hole_boundary, 2=hole_center
        """
        # Region embedding: (B, 1, H, W) -> Embedding -> (B, 1, H, W, C) -> squeeze -> permute -> (B, C, H, W)
        region_feat = self.region_emb(region_map.long())  # (B, 1, H, W, C)
        region_feat = region_feat.squeeze(1).permute(0, 3, 1, 2)  # (B, C, H, W)
        region_cond = self.region_scale(region_feat)

        # Conv + BN + ReLU
        h = self.conv1(x)
        h = self.bn1(h)
        h = h + region_cond  # 区域条件注入
        h = F.relu(h, inplace=True)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h, inplace=True)

        # Context attention
        hole_mask = (region_map > 0).float()  # 空洞区域
        h = self.attention(h, hole_mask)

        return h


class RegionAwareInpaintNet(nn.Module):
    """
    RAD-Inspired 区域感知填补网络

    将图像分为三个区域:
    - Region 0: Valid (有效区域)
    - Region 1: Hole Boundary (空洞边界)
    - Region 2: Hole Center (空洞内部)

    特点:
    1. Multi-scale 区域特征提取
    2. Region-aware attention 从有效区域复制特征到空洞
    3. 空洞区域和有效区域的特征分别处理后融合
    """

    def __init__(self, in_channels=3, out_channels=3, base_filters=64, num_regions=3):
        super().__init__()
        self.num_regions = num_regions

        # 编码器: 逐步下采样提取多尺度特征
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

        # 中间层: Region-Aware blocks
        self.middle1 = RegionAwareInpaintBlock(base_filters * 4, base_filters * 4, num_heads=8)
        self.middle2 = RegionAwareInpaintBlock(base_filters * 4, base_filters * 4, num_heads=8)

        # 解码器: 逐步上采样
        # 上采样 + skip concat + conv（正确 U-Net 顺序）
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

        # 输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, out_channels, 3, padding=1)
        )

    def compute_region_map(self, confidence):
        """
        从 confidence map 计算区域图

        Args:
            confidence: (B, 1, H, W) 置信度图, 0=空洞, 1=有效
        Returns:
            region_map: (B, 1, H, W), 0=valid, 1=hole_boundary, 2=hole_center
        """
        hole_mask = (confidence < 0.5).float()  # 1=空洞, 0=有效

        # 膨胀得到边界 (hole_boundary)
        kernel = torch.ones(3, 3, device=hole_mask.device)
        hole_boundary = F.conv2d(hole_mask, kernel.unsqueeze(0).unsqueeze(0), padding=1)
        hole_boundary = (hole_boundary > 0) & (hole_boundary < 9)  # 边界 pixel
        hole_boundary = hole_boundary.float()

        # 空洞中心 = 空洞 - 边界
        hole_center = hole_mask - hole_boundary
        hole_center = torch.clamp(hole_center, 0, 1)

        # 0: valid, 1: hole_boundary, 2: hole_center
        region_map = hole_boundary + hole_center * 2

        return region_map

    def forward(self, x, confidence):
        """
        Args:
            x: (B, 3, H, W) 输入 (masked noise IQ)
            confidence: (B, 1, H, W) 置信度图, 0=空洞, 1=有效
        Returns:
            filled: (B, 3, H, W) 填补后的图像
        """
        region_map = self.compute_region_map(confidence)

        # 编码
        e1 = self.encoder1(x)      # (B, 64, H, W)
        e2 = self.encoder2(e1)     # (B, 128, H/2, W/2)
        e3 = self.encoder3(e2)     # (B, 256, H/4, W/4)

        # 中间处理 - 需要将 region_map 下采样到匹配各层特征图的尺寸
        region_map_e3 = F.interpolate(region_map, size=e3.shape[2:], mode='nearest')
        m = self.middle1(e3, region_map_e3)
        m = self.middle2(m, region_map_e3)

        # 解码：先上采样，再 cat encoder feature，再 conv
        d = self.decoder2(torch.cat([self.up2(m), e2], dim=1))   # (B, 128, H/2, W/2)
        d = self.decoder1(torch.cat([self.up1(d), e1], dim=1))   # (B, 64, H, W)

        # 输出
        residual = self.out_conv(d)
        filled = x + residual

        return filled


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
            # Load ideal IQ
            ideal_data = np.load(ideal_IQ_path)
            if ideal_data.shape[:2] != (args.resolution, args.resolution):
                ideal_data = cv2.resize(ideal_data, target_size, interpolation=cv2.INTER_LINEAR)

            # Load noise IQ
            noise_data = np.load(noise_IQ_path)
            if noise_data.shape[:2] != (args.resolution, args.resolution):
                noise_data = cv2.resize(noise_data, target_size, interpolation=cv2.INTER_LINEAR)

            # Scale normalization (与推理脚本一致)
            scale = max(noise_data.max(), abs(noise_data.min()), 1e-8)
            noise_data = noise_data / scale
            ideal_data = ideal_data / scale

            # ideal 复制 3 通道 (for VAE)，noise 保持 1 通道
            ideal_data = np.repeat(np.expand_dims(ideal_data, axis=0), 3, axis=0)
            noise_data = np.expand_dims(noise_data, axis=0)  # 保持 1 通道

            ideals.append(data_transforms(ideal_data))
            noises.append(data_transforms(noise_data))

            # Load confidence map
            conf_data = np.load(conf_path)
            if conf_data.shape[:2] != (args.resolution, args.resolution):
                conf_data = cv2.resize(conf_data, target_size, interpolation=cv2.INTER_LINEAR)
            conf_data = np.expand_dims(conf_data, axis=0)  # 保持 1 通道
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
    """Collate function for PBRT dataset."""
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

    # Load tokenizer
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
        logger.info("Initializing DepthCAD weights from unet (2ch)")
        depthcad = ControlNetModel.from_unet(unet, conditioning_channels=2)

    # =================================================================
    # 新增: Region-Aware Inpainting Module
    # =================================================================
    inpaint_net = RegionAwareInpaintNet(in_channels=1, out_channels=1, base_filters=64).to(accelerator.device)
    logger.info(f"Inpainting Module params: {sum(p.numel() for p in inpaint_net.parameters()):,}")

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
                    if hasattr(model, 'save_pretrained'):
                        sub_dir = "depthcad"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                    else:
                        sub_dir = "inpaint_net"
                        os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(output_dir, sub_dir, "pytorch_model.bin"))
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                if hasattr(model, 'save_pretrained'):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="depthcad")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    inpaint_path = os.path.join(input_dir, "inpaint_net", "pytorch_model.bin")
                    if os.path.exists(inpaint_path):
                        state_dict = torch.load(inpaint_path, map_location=next(model.parameters()).device)
                        model.load_state_dict(state_dict)
                        del state_dict

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if args.freeze_depthcad:
        depthcad.requires_grad_(False)
        depthcad.eval()
        logger.info("DepthCAD is FROZEN (--freeze_depthcad), only inpaint_net will be trained.")
    else:
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
        # inpaint_net 是自定义网络，不支持 gradient_checkpointing，跳过

    low_precision_error_string = "Please make sure to always have all model weights in full float32 precision when starting training."
    if unwrap_model(depthcad).dtype != torch.float32:
        raise ValueError(f"DepthCAD loaded as datatype {unwrap_model(depthcad).dtype}. {low_precision_error_string}")

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 分别优化 DepthCAD 和 Inpainting Module
    inpaint_params = inpaint_net.parameters()

    if args.freeze_depthcad:
        optimizer = optimizer_class(
            [{"params": inpaint_params, "lr": args.inpaint_lr}],
            betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon
        )
    else:
        depthcad_params = depthcad.parameters()
        optimizer = optimizer_class([
            {"params": depthcad_params, "lr": args.learning_rate},
            {"params": inpaint_params, "lr": args.inpaint_lr},
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
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
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
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
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
                # =================================================================
                # 1. Region-Aware Inpainting Module 填补空洞
                # =================================================================
                masked_noise = batch["noises"].to(dtype=weight_dtype)  # (B, 1, H, W)
                confidence = batch["confs"][:, :1, :, :].to(dtype=weight_dtype)  # (B, 1, H, W)

                # Inpainting Module forward
                filled_iq = inpaint_net(masked_noise, confidence)  # (B, 1, H, W)

                # =================================================================
                # 2. DepthCAD Forward
                # =================================================================
                latents = vae.encode(batch["ideals"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # 拼接: filled_iq (1ch) + confidence (1ch) -> 2ch ControlNet 条件
                depthcad_image = torch.cat([filled_iq, confidence], dim=1).to(dtype=weight_dtype)  # (B, 2, H, W)

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

                diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                inpaint_loss = F.l1_loss(
                    filled_iq.float(),
                    batch["ideals"][:, :1].to(dtype=weight_dtype).float()
                )
                loss = diffusion_loss + inpaint_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = list(inpaint_net.parameters())
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

                logs = {"loss": loss.detach().item(), "diffusion_loss": diffusion_loss.detach().item(), "inpaint_loss": inpaint_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        depthcad = unwrap_model(depthcad)
        depthcad.save_pretrained(args.output_dir)
        inpaint_net = unwrap_model(inpaint_net)
        torch.save(inpaint_net.state_dict(), os.path.join(args.output_dir, "inpaint_net.pth"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
