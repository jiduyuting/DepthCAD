#!/usr/bin/env python3
"""
测试修改后的数据加载代码
验证训练脚本是否能正常使用新的数据加载功能
"""

import sys
sys.path.insert(0, '/data/pre_student/GJ/DepthCAD')

from datasets import load_dataset
from pbrt_dataset.preprocess import load_raw

print("=" * 60)
print("测试训练数据加载流程")
print("=" * 60)

# 加载数据集
dataset = load_dataset("pbrt_dataset", "default", cache_dir=None)

print(f"\n数据集大小: {len(dataset['train'])}")

# 测试加载前3个样本
for i in range(min(3, len(dataset['train']))):
    sample = dataset['train'][i]

    print(f"\n{'='*60}")
    print(f"样本 {i+1}")
    print(f"{'='*60}")
    print(f"Ideal IQ path: {sample['ideal_IQ_path']}")
    print(f"Noise IQ path: {sample['noise_IQ_path']}")
    print(f"Conf path: {sample['conf_path']}")

    try:
        # 加载ideal IQ
        ideal_IQ = load_raw(sample['ideal_IQ_path'], target_size=(512, 512), verbose=(i==0))
        print(f"✓ Ideal IQ 加载成功: {ideal_IQ.shape}, 范围 [{ideal_IQ.min():.4f}, {ideal_IQ.max():.4f}]")

        # 加载noise IQ
        noise_IQ = load_raw(sample['noise_IQ_path'], target_size=(512, 512), verbose=False)
        print(f"✓ Noise IQ 加载成功: {noise_IQ.shape}, 范围 [{noise_IQ.min():.4f}, {noise_IQ.max():.4f}]")

        # 加载confidence
        import numpy as np
        conf = np.load(sample['conf_path'])
        print(f"✓ Confidence 加载成功: {conf.shape}, 范围 [{conf.min():.4f}, {conf.max():.4f}]")

        # 验证形状（confidence可能是256x256或512x512）
        assert ideal_IQ.shape == (6, 512, 512), f"Ideal IQ shape错误: {ideal_IQ.shape}"
        assert noise_IQ.shape == (6, 512, 512), f"Noise IQ shape错误: {noise_IQ.shape}"
        # Confidence可以是任意尺寸，训练时会resize
        assert len(conf.shape) == 2, f"Confidence应该是2D数组: {conf.shape}"

        print(f"✓ 所有数据验证通过!")

    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n{'='*60}")
print("测试完成！")
print(f"{'='*60}")
