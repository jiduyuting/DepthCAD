import cv2
import os
import sys
import torch
import argparse
import numpy as np

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Support both flat_dataset and pbrt_dataset formats
# Import from preprocess.py (mask-enabled version) instead of preprocess.py
try:
    from pbrt_dataset.preprocess import load_raw as load_raw_pbrt, compute_gradient_confidence
    PBRT_AVAILABLE = True
    print("Successfully imported from pbrt_dataset.preprocess")
except ImportError as e:
    print(f"Failed to import from pbrt_dataset.preprocess: {e}")
    PBRT_AVAILABLE = False
try:
    from flat_dataset.preprocess import load_raw as load_raw_flat, compute_gradient_confidence as compute_gradient_confidence_flat
    FLAT_AVAILABLE = True
except ImportError:
    FLAT_AVAILABLE = False

from IQToDepth import IQ_to_depth


def compute_enhanced_features(noise, conf):
    """
    Marigold-style enhanced feature computation for ControlNet conditioning.

    Args:
        noise: 噪声图 [H, W]
        conf: 置信度图 [H, W]

    Returns:
        Enhanced features [6, H, W] including:
        - noise (原始噪声)
        - conf (原始置信度)
        - noise_dx (噪声x方向梯度)
        - noise_dy (噪声y方向梯度)
        - conf_dx (置信度x方向梯度)
        - conf_dy (置信度y方向梯度)
    """
    import torch.nn.functional as F

    # 转换为 torch tensor 并确保是 2D [H, W]
    if isinstance(noise, np.ndarray):
        noise = torch.from_numpy(noise).float()
    if isinstance(conf, np.ndarray):
        conf = torch.from_numpy(conf).float()

    # 确保是 2D 张量 [H, W]
    if noise.dim() == 3:
        noise = noise.squeeze(0)
    if conf.dim() == 3:
        conf = conf.squeeze(0)

    # 计算噪声的梯度，使用 numpy 计算避免 torch.pad 的问题
    noise_np = noise.cpu().numpy()
    conf_np = conf.cpu().numpy()

    # x 方向梯度: 右 - 左
    noise_dx_np = noise_np[:, :-1] - noise_np[:, 1:]
    noise_dx_np = np.pad(noise_dx_np, ((0, 0), (0, 1)), mode='edge')

    # y 方向梯度: 下 - 上
    noise_dy_np = noise_np[:-1, :] - noise_np[1:, :]
    noise_dy_np = np.pad(noise_dy_np, ((0, 1), (0, 0)), mode='edge')

    # 置信度梯度
    conf_dx_np = conf_np[:, :-1] - conf_np[:, 1:]
    conf_dx_np = np.pad(conf_dx_np, ((0, 0), (0, 1)), mode='edge')

    conf_dy_np = conf_np[:-1, :] - conf_np[1:, :]
    conf_dy_np = np.pad(conf_dy_np, ((0, 1), (0, 0)), mode='edge')

    # 转回 torch tensor 并拼接
    noise_dx = torch.from_numpy(noise_dx_np).float()
    noise_dy = torch.from_numpy(noise_dy_np).float()
    conf_dx = torch.from_numpy(conf_dx_np).float()
    conf_dy = torch.from_numpy(conf_dy_np).float()

    # 拼接所有特征 [6, H, W]
    enhanced_features = torch.stack([noise, conf, noise_dx, noise_dy, conf_dx, conf_dy], dim=0)

    return enhanced_features


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
    parser.add_argument(
        "--num_channels",
        type=int,
        default=None,
        help="Number of ControlNet input channels (2 or 6). If None, auto-detect from model."
    )

    args = parser.parse_args()
    return args


# def inference(pipe, noise, conf, scale, target_size=(256, 256)):
#     """
#     Run inference to predict IQ channels from noisy IQ and confidence map.
    
#     Parameters:
#     -----------
#     pipe : StableDiffusionControlNetPipeline
#         The diffusion pipeline
#     noise : numpy.ndarray
#         Noisy IQ data with shape (6, h, w)
#     conf : numpy.ndarray
#         Confidence map with shape (h, w)
#     scale : float
#         Scaling factor for denormalization
#     target_size : tuple
#         Target size (height, width) for output
    
#     Returns:
#     --------
#     pred_IQs : numpy.ndarray
#         Predicted IQ data with shape (6, h, w)
#     """
#     pred_IQs = np.zeros(noise.shape)

#     for i in range(6):
#         guidance = np.stack([noise[i], conf], axis=0)        
#         guidance = torch.from_numpy(guidance).unsqueeze(0)

#         prompt = ""

#         # generate image
#         generator = torch.manual_seed(42)
#         pred_IQ = pipe(
#             prompt, 
#             num_inference_steps=20, 
#             generator=generator, 
#             image=guidance
#         ).images[0]

#         pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
#         pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0    # convert to (0, 1)
#         pred_IQ = 2 * pred_IQ - 1   # (-1, 1)
#         pred_IQs[i] = pred_IQ * scale

#     # Resize to target size
#     target_h, target_w = target_size
#     reshaped_IQs = np.zeros((6, target_h, target_w), dtype=np.float32)
#     for i in range(6):
#         reshaped_IQs[i, :, :] = cv2.resize(
#             pred_IQs[i, :, :], 
#             (target_w, target_h), 
#             interpolation=cv2.INTER_LINEAR
#         )
    
#     return reshaped_IQs
def inference(pipe, noise, conf, scale, target_size=(240, 320), num_channels=2):
    """
    Run inference to predict IQ channels.
    内部强制使用 512 进行推理，最后还原回 target_size

    Args:
        pipe: StableDiffusionControlNetPipeline
        noise: numpy array of shape (6, H, W) - 6 IQ channels
        conf: numpy array of shape (H, W) - confidence map
        scale: float - scaling factor
        target_size: tuple - target (height, width)
        num_channels: int - number of input channels for ControlNet (2 or 6)
    """
    # 1. 定义模型推理需要的尺寸
    # 使用训练时的分辨率（512），确保与训练一致

    infer_h, infer_w = 512, 512

    # 2. 预处理：将置信度图 resize 到推理尺寸
    # cv2.resize 参数顺序是 (width, height)
    conf_resized = cv2.resize(conf, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

    # 初始化用于存储推理结果的数组 (使用推理尺寸 512x512)
    pred_IQs_infer = np.zeros((6, infer_h, infer_w))

    for i in range(6):
        # 3. 预处理：将当前通道的 noise resize 到推理尺寸
        noise_resized = cv2.resize(noise[i], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)

        # 构建 guidance 输入
        if num_channels == 6:
            # 使用增强特征 (6 通道): noise, conf, noise_dx, noise_dy, conf_dx, conf_dy
            guidance = compute_enhanced_features(noise_resized, conf_resized)
            # compute_enhanced_features 返回 torch.Tensor [6, H, W]
            guidance = guidance.unsqueeze(0)  # [1, 6, H, W]
        else:
            # 使用原始 2 通道输入: noise, conf
            guidance = np.stack([noise_resized, conf_resized], axis=0)
            # guidance 是 numpy array [2, H, W]
            guidance = torch.from_numpy(guidance).unsqueeze(0)  # [1, 2, H, W]

        prompt = ""

        # generate image
        generator = torch.manual_seed(42)

        # 模型输出的尺寸将是 (infer_h, infer_w)
        pred_IQ = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            image=guidance,
            height=infer_h, # 显式指定推理高度
            width=infer_w   # 显式指定推理宽度
        ).images[0]

        pred_IQ = np.nan_to_num(pred_IQ, nan=0, neginf=0, posinf=0)
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0    # convert to (0, 1)
        pred_IQ = 2 * pred_IQ - 1   # (-1, 1)

        # 将结果存入推理尺寸的数组中
        pred_IQs_infer[i] = pred_IQ * scale

    # 4. 后处理：将结果 Resize 回用户指定的目标尺寸 (240, 320)
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
    noise_file = args.noise_IQ_file
    noise_depth_file = args.noise_depth_file
    out_file = args.out_file
    dataset_type = args.dataset_type
    target_size = tuple(args.target_size)
    num_channels = args.num_channels
    
    print("=" * 60)
    print("DepthCAD Inference")
    print("=" * 60)
    print(f"Dataset type: {dataset_type}")
    print(f"Target size: {target_size}")
    print(f"Noise IQ file: {noise_file}")
    print(f"Noise depth file: {noise_depth_file}")
    print(f"Output file: {out_file}")
    
    # Load data based on dataset type
    if dataset_type == "pbrt":
        if not PBRT_AVAILABLE:
            raise ImportError("pbrt_dataset.preprocess not available. Please check your imports.")
        
        print("\nLoading PBRT data...")
        # Load IQ data from .npy file (shape: 9, 240, 320)
        # For masked model training, use amplitude thresholding
        noise_result = load_raw_pbrt(noise_file, target_size=target_size, sqrt_in=True,
                                      amplitude_threshold=None, upper_percentile=99.5)
        # load_raw_pbrt returns (tof_IQs, amp_mask)
        if isinstance(noise_result, tuple) or isinstance(noise_result, list):
            noise, amp_mask = noise_result
        else:
            noise = noise_result
            amp_mask = None
        print(f"Loaded noise IQ shape: {noise.shape}")

        # Apply amplitude mask to noise (zero out low amplitude regions)
        if amp_mask is not None:
            masked_pct = 100.0 * np.count_nonzero(amp_mask) / amp_mask.size
            print(f"Applying amplitude mask: {masked_pct:.2f}% pixels masked")
            for c in range(noise.shape[0]):
                noise[c][amp_mask] = 0.0

        # Load depth for confidence computation
        noise_depth = np.load(noise_depth_file)
        # cv2.resize expects (width, height), but target_size is (height, width)
        noise_depth = cv2.resize(noise_depth.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        confidence = compute_gradient_confidence(noise_depth)

        # Apply amplitude mask to confidence map (zero out low amplitude regions)
        if amp_mask is not None:
            confidence[amp_mask] = 0.0
            print(f"Applied mask to confidence map")

        print(f"Computed confidence map shape: {confidence.shape}")
        
        # Scale normalization
        scale = max(noise.max(), abs(noise.min()), 1e-8)
        print(f"Scale factor: {scale:.4f}")
        noise /= scale
        
    elif dataset_type == "flat":
        if not FLAT_AVAILABLE:
            raise ImportError("flat_dataset.preprocess not available. Please check your imports.")
        
        print("\nLoading FLAT data...")
        # Load IQ data from binary file
        noise_result = load_raw_flat(noise_file)
        if isinstance(noise_result, tuple) or isinstance(noise_result, list):
            noise, amp_mask = noise_result
        else:
            noise = noise_result
            amp_mask = None
        print(f"Loaded noise IQ shape: {noise.shape}")
        
        # Resize to target size if needed
        if noise.shape[1:] != target_size:
            reshaped_noise = np.zeros((6, target_size[0], target_size[1]), dtype=np.float32)
            for i in range(6):
                reshaped_noise[i] = cv2.resize(noise[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            noise = reshaped_noise
            print(f"Resized noise IQ to: {noise.shape}")
        
        # Load depth for confidence computation
        noise_depth = np.load(noise_depth_file)
        noise_depth = cv2.resize(noise_depth.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        confidence = compute_gradient_confidence_flat(noise_depth)
        print(f"Computed confidence map shape: {confidence.shape}")
        
        # Scale normalization
        scale = max(noise.max(), abs(noise.min()), 1e-8)
        print(f"Scale factor: {scale:.4f}")
        noise /= scale
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load pipeline
    print("\nLoading DepthCAD model...")
    depthcad = ControlNetModel.from_pretrained(depthcad_path, torch_dtype=torch.float16)

    # Determine number of input channels for ControlNet
    if num_channels is not None:
        # User specified, use it
        print(f"Using specified ControlNet input channels: {num_channels}")
    else:
        # Auto-detect from model
        num_channels = depthcad.controlnet_cond_embedding.conv_in.in_channels
        print(f"Detected ControlNet input channels: {num_channels}")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=depthcad, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    print("Model loaded successfully!")

    # Run inference
    print("\nRunning inference...")
    pred_IQs = inference(pipe, noise, confidence, scale, target_size=target_size, num_channels=num_channels)
    # print(f"Predicted IQ shape: {pred_IQs.shape}")
    # print(f"Predicted IQ range: [{pred_IQs.min():.4f}, {pred_IQs.max():.4f}]")
    # np.save('pred_iq_104.npy', pred_IQs)
    # print("Predicted IQs saved to pred_IQs.npy")
    # exit()

    # Convert IQ to depth
    print("\nConverting IQ to depth...")
    # Ensure pred_IQs is in the correct format: (6, h, w) with order I30 Q30 I40 Q40 I58 Q58
    # The IQ_to_depth function expects this order
    depth = IQ_to_depth(pred_IQs, corr_save_path=None, depth_save_path=None)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}] meters")

    # Save output
    print(f"\nSaving depth to {out_file}...")
    np.save(out_file, depth)
    print("Done!")