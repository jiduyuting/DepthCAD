import os
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import cv2
from depth_estimator import DepthEstimator
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# 加载模型
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=ControlNetModel.from_pretrained(
        "./output/depthcad_pbrt_1_4/checkpoint-50000/depthcad",
        torch_dtype=torch.float16
    ),
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
pipe.to("cuda")

# 加载数据
channels = []
for ch in ['A', 'B', 'C', 'D', 'E', 'F']:
    channels.append(np.load(f'./pbrt_dataset/data/noise_IQ_masked/bathroom/1/100_{ch}.npy'))
noise_IQ = np.stack(channels, axis=0)  # (6, 512, 512)

conf = np.load('./pbrt_dataset/data/confidence_masked/bathroom/1/100.npy')
conf_512 = cv2.resize(conf, (512, 512))

# GT
gt = np.load('/data/pre_student/hcy/pbrt/gt_depth/bathroom/1/100.npy')
gt_512 = cv2.resize(gt, (512, 512))

estimator = DepthEstimator()

def run_controlnet(iq_input, conf_input):
    scale = max(iq_input.max(), abs(iq_input.min()), 1e-8)
    iq_norm = iq_input / scale
    
    pred_IQs = np.zeros((6, 512, 512))
    for i in range(6):
        noise_resized = cv2.resize(iq_norm[i], (512, 512))
        guidance = np.stack([noise_resized, conf_input], axis=0)
        guidance = torch.from_numpy(guidance).unsqueeze(0).to("cuda")
        
        pred_IQ = pipe(prompt="", num_inference_steps=20, 
                       generator=torch.manual_seed(42), 
                       image=guidance, height=512, width=512).images[0]
        pred_IQ = np.mean(np.array(pred_IQ), axis=2) / 255.0
        pred_IQ = 2 * pred_IQ - 1
        pred_IQs[i] = pred_IQ * scale
    
    depth = estimator.process(pred_IQs)
    mae = np.abs(depth - gt_512).mean()
    return mae, depth

print("=" * 60)
print("对比测试：不同空洞处理方式的效果")
print("=" * 60)

# Test 1: 原始 noise（不做任何空洞处理）
mae1, _ = run_controlnet(noise_IQ, conf_512)
print(f"\n1. 原始 noise (无空洞处理): MAE = {mae1:.4f}")

# Test 2: 空洞区域置零
noisy_zeros = noise_IQ.copy()
hole_mask = np.random.random((512, 512)) < 0.15  # 随机15%空洞
hole_3d = np.stack([hole_mask] * 6, axis=0)
noisy_zeros[hole_3d] = 0
mae2, _ = run_controlnet(noisy_zeros, conf_512)
print(f"2. 空洞置零: MAE = {mae2:.4f}")

# Test 3: NS Inpaint 填补
noisy_ns = np.zeros((6, 512, 512), dtype=np.float32)
for i in range(6):
    noisy_ns[i] = cv2.inpaint((noise_IQ[i] * 255).astype(np.uint8), 
                                (hole_mask * 255).astype(np.uint8), 
                                inpaintRadius=3, flags=cv2.INPAINT_NS).astype(np.float32) / 255.0
mae3, _ = run_controlnet(noisy_ns, conf_512)
print(f"3. NS Inpaint 填补: MAE = {mae3:.4f}")

# Test 4: 不用 ControlNet，直接 NS Inpaint → Depth
depth_ns = estimator.process(noisy_ns)
mae4 = np.abs(depth_ns - gt_512).mean()
print(f"4. NS Inpaint 直接→Depth (无 ControlNet): MAE = {mae4:.4f}")

print("\n" + "=" * 60)
