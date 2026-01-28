import numpy as np
from PIL import Image

# 1. 加载数据
arr = np.load('/data/pre_student/GJ/DepthCAD/flat_dataset/data/ideal_IQ/1499392477071669_A.npy')

# --- 预处理步骤 (根据你的数据情况调整) ---
# 情况 A: 如果数据是 (Channel, Height, Width) -> 转为 (Height, Width, Channel)
if arr.ndim == 3 and arr.shape[0] in [1, 3]: 
    arr = arr.transpose(1, 2, 0)

# 情况 B: 如果数据是 float 类型 (0.0 - 1.0) -> 转为 0-255 的整数
if arr.dtype == np.float32 or arr.dtype == np.float64:
    # 甚至如果是 -1 到 1 的归一化数据，需要先 (arr + 1) / 2
    arr = (arr * 255).astype(np.uint8)

# 情况 C: 如果是单通道 (H, W, 1) -> 降维成 (H, W)
if arr.ndim == 3 and arr.shape[2] == 1:
    arr = arr.squeeze()
# ---------------------------------------

# 2. 转为图像对象
# 如果是灰度图/深度图
if arr.ndim == 2:
    img = Image.fromarray(arr, mode='L') # 'L' 代表灰度
# 如果是 RGB
else:
    img = Image.fromarray(arr, mode='RGB')

# 3. 保存
img.save('output.png')
print("保存成功！")