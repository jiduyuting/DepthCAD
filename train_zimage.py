#!/usr/bin/env python3
"""
DepthCAD Training Script for Z-Image-Turbo Model

This script is adapted from train.py to work with Tongyi-MAI/Z-Image-Turbo.

Key differences from original train.py:
1. Uses Qwen3Model instead of CLIP for text encoding
2. Uses ZImageTransformer2DModel instead of UNet2DConditionModel
3. Uses FlowMatchEulerDiscreteScheduler instead of DDPMScheduler
4. Potentially uses QwenImageControlNetModel for ControlNet

⚠️ WARNING: This is EXPERIMENTAL and may not work without further modifications.
The ZImagePipeline may not support ControlNet natively.
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import accelerate
import numpy as np
import gc
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import cv2
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

# Fix torch compatibility issues with newer diffusers
import torch
if not hasattr(torch, 'xpu'):
    class XPUDummy:
        def empty_cache(self):
            pass
        @property
        def device_count(self):
            return 0
        def manual_seed(self, seed):
            pass
        def is_available(self):
            return False
    torch.xpu = XPUDummy()
if not hasattr(torch, 'mps'):
    class MPSDummy:
        def empty_cache(self):
            pass
        @property
        def device_count(self):
            return 0
        def manual_seed(self, seed):
            pass
        def is_available(self):
            return False
    torch.mps = MPSDummy()

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,  # Changed from DDPMScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Try to import Z-Image-Turbo specific components
# These may fail if diffusers version is not up to date
TRANSFORMER_AVAILABLE = False
CONTROLNET_AVAILABLE = False

try:
    from diffusers.models.transformers import ZImageTransformer2DModel
    TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

# Always import ControlNetModel as fallback
from diffusers import ControlNetModel

try:
    from diffusers.models.controlnets import QwenImageControlNetModel
    CONTROLNET_AVAILABLE = True
except ImportError:
    pass


def scale_invariant_gradient_loss(pred, target, reduction='mean'):
    """
    Scale-Invariant Gradient Matching Loss

    计算预测和目标之间的梯度差异，使模型更好地学习边缘和结构信息。

    Args:
        pred: 预测的张量 [B, C, H, W]
        target: 目标张量 [B, C, H, W]
        reduction: 'mean', 'sum', 或 'none'

    Returns:
        梯度损失值
    """
    # 计算x方向的梯度 (右 - 左)
    pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]

    # 计算y方向的梯度 (下 - 上)
    pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
    target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]

    # 计算L1损失 (对异常值更鲁棒)
    loss_dx = torch.abs(pred_dx - target_dx)
    loss_dy = torch.abs(pred_dy - target_dy)

    if reduction == 'mean':
        return loss_dx.mean() + loss_dy.mean()
    elif reduction == 'sum':
        return loss_dx.sum() + loss_dy.sum()
    else:
        return loss_dx + loss_dy


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """
    Import the appropriate text encoder class based on the model config.

    Now supports both CLIP and Qwen3.
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0] if text_encoder_config.architectures else "Unknown"

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        logger.info("Using CLIPTextModel")
        return CLIPTextModel
    elif model_class in ["Qwen3ForCausalLM", "Qwen3Model"]:
        # Z-Image-Turbo uses Qwen3
        from transformers import Qwen3Model
        logger.info(f"Using Qwen3Model ({model_class})")
        return Qwen3Model
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",  # Changed default
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
        default="model",
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
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        "--precomputed_embeddings_path",
        type=str,
        default=None,
        help="Path to precomputed text embeddings (.npy file). If provided, text_encoder will not be loaded.",
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
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
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
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
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
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
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
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report to results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml". Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="default",
        help=(
            "The configuration of the dataset to use (e.g., 'default' for no mask, 'masked' for with mask)."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_depthcad_zimage",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerator.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def make_train_dataset(args, tokenizer, accelerator, precomputed_embeddings=None):
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    # If precomputed embeddings are provided, load them
    if precomputed_embeddings is not None:
        logger.info(f"Loading precomputed embeddings from {args.precomputed_embeddings_path}")
        embeddings_array = np.load(args.precomputed_embeddings_path)
        logger.info(f"Loaded embeddings shape: {embeddings_array.shape}")

        # Add embeddings index to dataset
        dataset["train"] = dataset["train"].add_column(
            "embedding_idx", range(len(dataset["train"]))
        )

        return dataset["train"], embeddings_array

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples["prompt"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise NotImplementedError
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    data_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
        ]
    )

    def preprocess_train(examples):
        target_size = (args.resolution, args.resolution)

        # Process ideals with resize
        ideals = []
        for ideal_IQ_path in examples["ideal_IQ_path"]:
            data = np.load(ideal_IQ_path)
            # Resize to target resolution if needed
            if data.shape[:2] != (args.resolution, args.resolution):
                data = cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
            # Repeat to 3 channels
            data = np.repeat(np.expand_dims(data, axis=0), 3, axis=0)
            ideals.append(data_transforms(data))

        # Process noises with resize
        noises = []
        for noise_IQ_path in examples["noise_IQ_path"]:
            data = np.load(noise_IQ_path)
            # Resize to target resolution if needed
            if data.shape[:2] != (args.resolution, args.resolution):
                data = cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
            data = np.expand_dims(data, axis=0)
            noises.append(data_transforms(data))

        # Process confs with resize
        confs = []
        for conf_path in examples["conf_path"]:
            data = np.load(conf_path)
            # Resize to target resolution if needed
            if data.shape[:2] != (args.resolution, args.resolution):
                data = cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
            data = np.expand_dims(data, axis=0)
            confs.append(data_transforms(data))

        examples["ideals"] = ideals
        examples["noises"] = noises
        examples["confs"] = confs
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples, use_precomputed_embeddings=False, embeddings_array=None):
    ideals = torch.stack([example["ideals"] for example in examples])
    ideals = ideals.to(memory_format=torch.contiguous_format).float()

    noises = torch.stack([example["noises"] for example in examples])
    noises = noises.to(memory_format=torch.contiguous_format).float()

    confs = torch.stack([example["confs"] for example in examples])

    if use_precomputed_embeddings:
        # Get embedding indices for this batch
        embedding_indices = [example["embedding_idx"] for example in examples]
        # Load precomputed embeddings
        encoder_hidden_states = torch.from_numpy(embeddings_array[embedding_indices]).float()
        return {
            "ideals": ideals,
            "noises": noises,
            "confs": confs,
            "encoder_hidden_states": encoder_hidden_states,
        }
    else:
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

    # Try to free python-side refs and GPU cache before allocating large models
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Log basic CUDA memory stats for debugging OOMs
        try:
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            total_gb = props.total_memory / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(dev) / (1024 ** 3)
            allocated_gb = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            logger.info(
                f"CUDA device {dev} memory (GB) - total: {total_gb:.2f}, reserved: {reserved_gb:.2f}, allocated: {allocated_gb:.2f}"
            )
        except Exception as e:
            logger.warning(f"Unable to query CUDA memory stats: {e}")

    # Warn if requested effective total batch size is very large (common OOM cause)
    try:
        effective_total_batch = args.train_batch_size * getattr(accelerator, "num_processes", 1) * args.gradient_accumulation_steps
        if effective_total_batch >= 32:
            logger.warning(
                "Effective total batch size (%d) is large and may cause CUDA OOM. "
                "Consider lowering --train_batch_size or --gradient_accumulation_steps.",
                effective_total_batch,
            )
    except Exception:
        pass

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Check component availability
    if not TRANSFORMER_AVAILABLE:
        logger.warning("ZImageTransformer2DModel not available. Update diffusers to >=0.36.0")
        logger.warning("Training will fail without this component!")

    if not CONTROLNET_AVAILABLE:
        logger.warning("QwenImageControlNetModel not available. Falling back to ControlNetModel")
        logger.warning("This may not work correctly with Z-Image-Turbo!")
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Check if using precomputed embeddings
    use_precomputed_embeddings = args.precomputed_embeddings_path is not None
    embeddings_array = None
    text_encoder = None

    if use_precomputed_embeddings:
        logger.info(f"Using precomputed embeddings from {args.precomputed_embeddings_path}")
        logger.info("Text encoder will NOT be loaded to save GPU memory")
    else:
        # Load scheduler and models
        # Use FlowMatchEulerDiscreteScheduler for Z-Image-Turbo
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler"
        )
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant
        ).to("cpu")  # Keep text_encoder on CPU to save GPU memory

        # Enable gradient checkpointing for text encoder to save memory
        # Qwen3Model is very large and causes OOM even with batch_size=1
        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for text_encoder to reduce memory usage")

        logger.info("Text encoder is kept on CPU to save GPU memory. It will be moved to GPU only when needed.")

    # Load noise scheduler (needed for training even with precomputed embeddings)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant
    )

    # Load ZImageTransformer2DModel instead of UNet
    if not TRANSFORMER_AVAILABLE:
        raise RuntimeError(
            "ZImageTransformer2DModel is not available. "
            "Please update diffusers: pip install diffusers>=0.36.0"
        )

    transformer = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant
    )

    # Load or create ControlNet
    # Try to use QwenImageControlNetModel if available, otherwise fall back to ControlNetModel
    if args.depthcad_path:
        logger.info("Loading existing DepthCAD weights")
        if CONTROLNET_AVAILABLE:
            depthcad = QwenImageControlNetModel.from_pretrained(args.depthcad_path)
        else:
            depthcad = ControlNetModel.from_pretrained(args.depthcad_path)
    else:
        logger.info("Initializing DepthCAD weights from transformer")
        # For Transformer-based models like Z-Image-Turbo
        # Try multiple approaches to create the ControlNet

        depthcad = None
        last_error = None

        # Method 1: Try using from_transformer if available
        if CONTROLNET_AVAILABLE:
            try:
                logger.info("Trying QwenImageControlNetModel.from_transformer()...")
                depthcad = QwenImageControlNetModel.from_transformer(
                    transformer,
                    conditioning_channels=2
                )
                logger.info("Successfully created QwenImageControlNetModel using from_transformer()")
            except Exception as e:
                logger.warning(f"from_transformer() failed: {e}")
                last_error = e

        # Method 2: Try direct initialization with minimal config
        if depthcad is None and CONTROLNET_AVAILABLE:
            try:
                logger.info("Trying QwenImageControlNetModel direct initialization with config...")

                # Get transformer config
                config = transformer.config

                # Create config dict with correct parameter names for QwenImageControlNetModel
                # Based on error message, it expects: out_channels, patch_size, axes_dims_rope,
                # extra_condition_channels, joint_attention_dim
                controlnet_config_dict = {
                    "num_layers": config.n_layers if hasattr(config, 'n_layers') else 30,
                    "attention_head_dim": config.axes_dims if hasattr(config, 'axes_dims') else [32, 48, 48],
                    "num_attention_heads": config.n_heads if hasattr(config, 'n_heads') else 30,
                    "in_channels": config.in_channels if hasattr(config, 'in_channels') else 16,
                    "out_channels": config.out_channels if hasattr(config, 'out_channels') else 16,
                    "patch_size": config.patch_size if hasattr(config, 'patch_size') else 2,
                    "axes_dims_rope": config.axes_dims_rope if hasattr(config, 'axes_dims_rope') else [32, 48, 48],
                    "extra_condition_channels": 2,  # depth + conf (not conditioning_channels)
                    "joint_attention_dim": config.cap_feat_dim if hasattr(config, 'cap_feat_dim') else 2560,
                    "cross_attention_dim": config.cap_feat_dim if hasattr(config, 'cap_feat_dim') else 2560,
                }

                logger.info(f"ControlNet config: {controlnet_config_dict}")

                # Use from_config if available
                if hasattr(QwenImageControlNetModel, 'from_config'):
                    depthcad = QwenImageControlNetModel.from_config(controlnet_config_dict)
                    logger.info("Created QwenImageControlNetModel using from_config()")
                else:
                    # Create with dict directly
                    depthcad = QwenImageControlNetModel(**controlnet_config_dict)
                    logger.info("Successfully created QwenImageControlNetModel with direct config")

            except Exception as e:
                logger.warning(f"Direct initialization failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                last_error = e

        # Method 3: Fallback to standard ControlNetModel
        if depthcad is None:
            logger.warning("All QwenImageControlNetModel methods failed, falling back to ControlNetModel")
            try:
                logger.info("Creating ControlNetModel with default SD21 config...")

                # Try loading from stable-diffusion-2-1 as fallback
                # This won't be perfect but allows training to proceed
                fallback_config = {
                    "in_channels": 4,  # SD21 VAE has 4 channels
                    "conditioning_channels": 2,  # depth + conf
                    "flip_sin_to_cos": True,
                    "freq_shift": 0,
                    "down_block_types": [
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "DownBlock2D",
                    ],
                    "mid_block_type": "UNetMidBlock2DCrossAttn",
                    "up_block_types": [
                        "UpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                    ],
                    "only_cross_attention": False,
                    "block_out_channels": [320, 640, 1280, 1280],
                    "layers_per_block": 2,
                    "attention_head_dim": [5, 10, 20, 20],
                    "cross_attention_dim": 1024,
                    "norm_num_groups": 32,
                    "sample_size": 32,
                }

                depthcad = ControlNetModel.from_config(fallback_config)
                logger.info("Successfully created standard ControlNetModel with SD21 fallback config")

            except Exception as e:
                logger.error(f"ControlNetModel fallback also failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                last_error = e

        # If all methods failed, raise informative error
        if depthcad is None:
            raise NotImplementedError(
                f"Failed to create ControlNet for Z-Image-Turbo. Last error: {last_error}\n\n"
                "Consider:\n"
                "1. Using standard train.sh with stable-diffusion-2-1 instead\n"
                "2. Checking if diffusers version supports QwenImageControlNetModel\n"
                "3. Implementing a custom ControlNet adapter for Z-Image-Turbo"
            )

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
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
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if CONTROLNET_AVAILABLE:
                    load_model = QwenImageControlNetModel.from_pretrained(input_dir, subfolder="depthcad")
                else:
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="depthcad")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    depthcad.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            # Try to enable xformers for transformer
            if hasattr(transformer, 'enable_xformers_memory_efficient_attention'):
                transformer.enable_xformers_memory_efficient_attention()
            if hasattr(depthcad, 'enable_xformers_memory_efficient_attention'):
                depthcad.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        if hasattr(depthcad, 'enable_gradient_checkpointing'):
            depthcad.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(depthcad).dtype != torch.float32:
        raise ValueError(
            f"DepthCAD loaded as datatype {unwrap_model(depthcad).dtype}. {low_precision_error_string}"
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = depthcad.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Load dataset (with or without precomputed embeddings)
    if use_precomputed_embeddings:
        train_dataset, embeddings_array = make_train_dataset(
            args, tokenizer, accelerator, precomputed_embeddings=True
        )
    else:
        train_dataset = make_train_dataset(args, tokenizer, accelerator)

    # Create collate function with appropriate parameters
    from functools import partial

    collate_fn_with_params = partial(
        collate_fn,
        use_precomputed_embeddings=use_precomputed_embeddings,
        embeddings_array=embeddings_array,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn_with_params,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
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

    # Prepare everything with our `accelerator`.
    depthcad, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        depthcad, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
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
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(depthcad):
                # Clean GPU cache before each step to reduce memory fragmentation
                if step % 10 == 0:
                    torch.cuda.empty_cache()

                # Convert images to latent space
                # NOTE: Z-Image-Turbo VAE has 16 channels instead of 4
                with torch.no_grad():
                    latents = vae.encode(batch["ideals"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Free up memory from VAE encode
                del batch["ideals"]
                torch.cuda.empty_cache()

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                # Flow-matching uses different timestep sampling than DDPM
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # For Flow-matching: interpolate between latents and noise based on timestep
                # FlowMatchEulerDiscreteScheduler doesn't have add_noise(), so we use the flow-matching formula
                # Get sigma values for the timesteps
                if hasattr(noise_scheduler, 'marginal_prob'):
                    # Use the scheduler's marginal_prob if available
                    sigma = noise_scheduler.marginal_prob(timesteps.view(-1), latents.view(bsz, -1))[0]
                    sigma = sigma.view(bsz, 1, 1, 1)
                    noisy_latents = (1 - sigma) * latents + sigma * noise
                else:
                    # Fallback: simple interpolation for flow-matching
                    # Normalize timesteps to [0, 1]
                    timesteps_norm = timesteps.float() / noise_scheduler.config.num_train_timesteps
                    timesteps_norm = timesteps_norm.view(bsz, 1, 1, 1)
                    # Flow-matching: linear interpolation between clean and noise
                    noisy_latents = (1 - timesteps_norm) * latents + timesteps_norm * noise

                # Get the text embedding for conditioning
                if use_precomputed_embeddings:
                    # Use precomputed embeddings directly from batch
                    encoder_hidden_states = batch["encoder_hidden_states"].to(dtype=weight_dtype)
                else:
                    # Move text_encoder to GPU temporarily for forward pass
                    with torch.cuda.device(accelerator.device):
                        text_encoder_device = next(text_encoder.parameters()).device
                        if text_encoder_device.type == "cpu":
                            text_encoder.to(accelerator.device)
                            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device), return_dict=False)[0]
                            # Move back to CPU immediately to free GPU memory
                            text_encoder.to("cpu")
                            torch.cuda.empty_cache()
                        else:
                            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device), return_dict=False)[0]

                depthcad_image = torch.concatenate([batch["noises"], batch["confs"]], dim=1).to(dtype=weight_dtype)

                # ⚠️ WARNING: This part may need significant changes for ZImageTransformer2DModel
                # The ControlNet integration may be different from UNet2DConditionModel
                try:
                    down_block_res_samples, mid_block_res_sample = depthcad(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=depthcad_image,
                        return_dict=False,
                    )
                except Exception as e:
                    logger.error(f"ControlNet forward failed: {e}")
                    logger.error("This may indicate that ZImageTransformer2DModel is not compatible with the current ControlNet implementation")
                    raise

                # Predict the noise residual using transformer instead of unet
                # ⚠️ WARNING: Transformer's forward signature may be different from UNet
                try:
                    model_pred = transformer(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ] if down_block_res_samples else None,
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype) if mid_block_res_sample is not None else None,
                        return_dict=False,
                    )[0]
                except Exception as e:
                    logger.error(f"Transformer forward failed: {e}")
                    logger.error("This may indicate that ZImageTransformer2DModel has a different API than UNet2DConditionModel")
                    raise

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = depthcad.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        depthcad = unwrap_model(depthcad)
        depthcad.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
