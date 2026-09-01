"""
调试脚本：检查训练和推理的数据流是否一致
"""
import numpy as np
import glob
import os

# ============================================================================
# 1. 检查训练数据 (从 dataset 加载)
# ============================================================================
print("=" * 60)
print("1. 检查训练数据格式")
print("=" * 60)

# 加载一个样本
ideal_files = sorted(glob.glob('pbrt_dataset/data/ideal_IQ_masked/bathroom/1/*_A.npy'))
noise_files = sorted(glob.glob('pbrt_dataset/data/noise_IQ_masked/bathroom/1/*_A.npy'))
conf_files = sorted(glob.glob('pbrt_dataset/data/confidence_masked/bathroom/1/*.npy'))

idx = 2  # 用 100 这个样本
ideal = np.load(ideal_files[idx])
noise = np.load(noise_files[idx])
conf = np.load(conf_files[idx])

print(f"\n[原始数据] bathroom/1/100")
print(f"  ideal_IQ: shape={ideal.shape}, range=[{ideal.min():.4f}, {ideal.max():.4f}], mean={ideal.mean():.4f}")
print(f"  noise_IQ: shape={noise.shape}, range=[{noise.min():.4f}, {noise.max():.4f}], mean={noise.mean():.4f}")
print(f"  confidence: shape={conf.shape}, range=[{conf.min():.4f}, {conf.max():.4f}], mean={conf.mean():.4f}")

# ============================================================================
# 2. 模拟训练时的预处理 (修改后的 train_pbrt_rad.py)
# ============================================================================
print("\n" + "=" * 60)
print("2. 模拟训练预处理 (train_pbrt_rad.py 修改版)")
print("=" * 60)

# Scale 归一化
scale = max(noise.max(), abs(noise.min()), 1e-8)
print(f"\n  scale = {scale:.4f}")

noise_train = noise / scale
ideal_train = ideal / scale

print(f"  [训练归一化后]")
print(f"  ideal: range=[{ideal_train.min():.4f}, {ideal_train.max():.4f}], mean={ideal_train.mean():.4f}")
print(f"  noise: range=[{noise_train.min():.4f}, {noise_train.max():.4f}], mean={noise_train.mean():.4f}")

# 扩展为 3 通道 (训练时的格式)
noise_train_3ch = np.repeat(np.expand_dims(noise_train, axis=0), 3, axis=0)
conf_train_1ch = np.expand_dims(conf, axis=0)

print(f"\n  [训练输入格式]")
print(f"  noise (3ch): shape={noise_train_3ch.shape}, range=[{noise_train_3ch.min():.4f}, {noise_train_3ch.max():.4f}]")
print(f"  conf (1ch): shape={conf_train_1ch.shape}, range=[{conf_train_1ch.min():.4f}, {conf_train_1ch.max():.4f}]")

# ControlNet 条件: [noise, conf] = 2 通道
controlnet_cond = np.concatenate([noise_train_3ch, conf_train_1ch], axis=0)
print(f"  controlnet_cond (2ch): shape={controlnet_cond.shape}")

# ============================================================================
# 3. 模拟推理时的预处理 (inference_pbrt_depth.py)
# ============================================================================
print("\n" + "=" * 60)
print("3. 模拟推理预处理 (inference_pbrt_depth.py)")
print("=" * 60)

# 推理时用 load_raw_pbrt(sqrt_in=True)，但这里是 6 通道已处理的数据
# 所以直接用 np.load，然后 scale 归一化

# 推理也会做 scale
scale_inf = max(noise.max(), abs(noise.min()), 1e-8)
noise_inf = noise / scale_inf

print(f"\n  scale = {scale_inf:.4f}")
print(f"  [推理归一化后]")
print(f"  noise: range=[{noise_inf.min():.4f}, {noise_inf.max():.4f}], mean={noise_inf.mean():.4f}")

# 推理时 guidance 是 [noise[i], conf] 两个通道
# 对每个 IQ 通道分别处理
guidance_ch0 = np.stack([noise_inf, conf], axis=0)
print(f"\n  guidance (单通道): shape={guidance_ch0.shape}, range=[{guidance_ch0.min():.4f}, {guidance_ch0.max():.4f}]")

# ============================================================================
# 4. 对比训练和推理的输入
# ============================================================================
print("\n" + "=" * 60)
print("4. 训练 vs 推理 输入对比")
print("=" * 60)

print(f"\n  训练时 ControlNet 输入:")
print(f"    noise 通道: range=[{noise_train_3ch.min():.4f}, {noise_train_3ch.max():.4f}]")
print(f"    conf 通道: range=[{conf_train_1ch.min():.4f}, {conf_train_1ch.max():.4f}]")

print(f"\n  推理时 guidance:")
print(f"    noise 通道: range=[{noise_inf.min():.4f}, {noise_inf.max():.4f}]")
print(f"    conf 通道: range=[{conf.min():.4f}, {conf.max():.4f}]")

# 检查是否一致
noise_diff = np.abs(noise_train_3ch[0] - noise_inf).max()
print(f"\n  ⚠️ noise 通道最大差异: {noise_diff:.6f}")

if noise_diff < 1e-5:
    print("  ✅ 训练和推理的 noise 输入一致")
else:
    print("  ❌ 训练和推理的 noise 输入不一致!")

# ============================================================================
# 5. 检查推理模型的输出 (如果存在)
# ============================================================================
print("\n" + "=" * 60)
print("5. 检查推理输出")
print("=" * 60)

pred_path = 'output/visualization/10000/bathroom/1/100/pred_depth.npy'
if os.path.exists(pred_path):
    pred = np.load(pred_path)
    gt_path = 'output/visualization/bathroom/1/100/gt_depth.npy'
    gt = np.load(gt_path)
    print(f"\n  GT depth: range=[{gt.min():.4f}, {gt.max():.4f}], mean={gt.mean():.4f}")
    print(f"  Pred depth: range=[{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")
    print(f"  RMSE: {np.sqrt(np.mean((gt - pred)**2)):.4f}")
else:
    print(f"\n  推理输出不存在: {pred_path}")