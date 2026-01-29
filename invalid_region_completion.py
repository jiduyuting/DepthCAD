"""
增强的无效区域检测与补全模块
DepthCAD Enhancement: Invalid Region Detection and Completion

核心功能：
1. 无效区域精准检测（幅值 + 有效性双判定）
2. 三层置信度建模（高/低/无效）
3. 几何先验注入（平面拟合、边缘检测）
4. 上下文感知补全引导
5. 分层扩散引导策略
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# ==================== 配置参数 ====================

@dataclass
class InvalidRegionConfig:
    """无效区域检测配置"""
    amplitude_threshold: float = 0.01  # 幅值阈值 τ
    depth_invalid_value: float = 0.0   # 深度无效值标记
    confidence_high_low: float = 0.7   # 高/低置信度分界
    confidence_low_invalid: float = 0.1  # 低/无效置信度分界

    # 几何先验配置
    plane_ransac_threshold: float = 0.05  # RANSAC 平面拟合阈值
    edge_kernel_size: int = 3  # 边缘检测核大小

    # 补全引导配置
    context_window_size: int = 5  # 上下文窗口大小
    fill_weight: float = 0.8  # 无效区域补全损失权重
    consistency_weight: float = 0.5  # 边界一致性损失权重


# ==================== 无效区域检测 ====================

def detect_invalid_regions(
    iq_data: np.ndarray,
    depth_data: Optional[np.ndarray] = None,
    config: Optional[InvalidRegionConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    检测无效区域

    Args:
        iq_data: IQ数据，shape为 (6, H, W) 或 (H, W) 如果是幅值图
        depth_data: 深度数据，shape为 (H, W)，包含无效值标记
        config: 配置参数

    Returns:
        invalid_mask: 无效区域掩码，shape为 (H, W)，bool类型
        amplitude_map: 幅值图，shape为 (H, W)
        confidence_map: 三层置信度图，shape为 (H, W)
    """
    if config is None:
        config = InvalidRegionConfig()

    # 1. 计算幅值特征 R = |x_i| + |x_q|
    if iq_data.ndim == 3:
        # IQ数据 shape: (6, H, W) - 3个IQ对，每对2个通道
        # 计算每对的幅值: (I30, Q30), (I40, Q40), (I58, Q58)
        amplitude_map = np.zeros(iq_data.shape[1:], dtype=np.float32)
        for i in range(0, 6, 2):
            amplitude_map += np.abs(iq_data[i, :, :]) + np.abs(iq_data[i+1, :, :])
        amplitude_map /= 3  # 归一化
    else:
        # 已经是幅值图
        amplitude_map = np.abs(iq_data).astype(np.float32)

    # 2. 基于幅值的无效判定
    amplitude_invalid = amplitude_map < config.amplitude_threshold

    # 3. 基于深度有效性的无效判定
    depth_invalid = np.zeros_like(amplitude_map, dtype=bool)
    if depth_data is not None:
        depth_invalid = depth_data <= config.depth_invalid_value

    # 4. 组合无效区域掩码
    invalid_mask = amplitude_invalid | depth_invalid

    # 5. 计算三层置信度
    confidence_map = compute_three_layer_confidence(
        amplitude_map, invalid_mask, config
    )

    return invalid_mask, amplitude_map, confidence_map


def compute_three_layer_confidence(
    amplitude_map: np.ndarray,
    invalid_mask: np.ndarray,
    config: InvalidRegionConfig
) -> np.ndarray:
    """
    计算三层置信度

    高置信区域 (c ∈ [0.7, 1.0]): SNR高，优先保真
    低置信区域 (c ∈ [0.1, 0.7]): SNR较低但存在有效信号
    无效区域 (c = 0): 无有效信号
    """
    H, W = amplitude_map.shape

    # 归一化幅值到 [0, 1]
    amp_min, amp_max = amplitude_map.min(), amplitude_map.max()
    if amp_max > amp_min:
        amp_normalized = (amplitude_map - amp_min) / (amp_max - amp_min)
    else:
        amp_normalized = np.zeros_like(amplitude_map)

    # 初始化置信度
    confidence = np.zeros((H, W), dtype=np.float32)

    # 有效区域置信度
    valid_mask = ~invalid_mask

    # 高置信区域: 归一化幅值 >= confidence_high_low
    high_conf_mask = valid_mask & (amp_normalized >= config.confidence_high_low)
    confidence[high_conf_mask] = amp_normalized[high_conf_mask]
    confidence[high_conf_mask] = np.clip(confidence[high_conf_mask], 0.7, 1.0)

    # 低置信区域: 归一化幅值在 [confidence_low_invalid, confidence_high_low) 之间
    low_conf_mask = valid_mask & (
        (amp_normalized >= config.confidence_low_invalid) &
        (amp_normalized < config.confidence_high_low)
    )
    confidence[low_conf_mask] = amp_normalized[low_conf_mask] * (
        config.confidence_high_low - config.confidence_low_invalid
    ) + config.confidence_low_invalid
    confidence[low_conf_mask] = np.clip(confidence[low_conf_mask], 0.1, 0.7)

    # 无效区域: 置信度为 0
    confidence[invalid_mask] = 0.0

    return confidence


# ==================== 几何先验注入 ====================

class GeometryPriorExtractor(nn.Module):
    """
    几何先验提取器
    功能：
    1. 场景平面拟合 (RANSAC)
    2. 边缘检测提取轮廓
    3. 全局几何参数编码
    """

    def __init__(self, config: Optional[InvalidRegionConfig] = None):
        super().__init__()
        self.config = config or InvalidRegionConfig()

        # 轻量级CNN用于提取上下文特征
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 几何参数编码器
        self.geometry_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 平面参数 + 边缘特征
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        depth_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取几何先验

        Args:
            depth_map: 深度图 [B, 1, H, W]
            valid_mask: 有效区域掩码 [B, 1, H, W]

        Returns:
            geometry_features: 几何特征字典
        """
        B, _, H, W = depth_map.shape

        # 1. 平面拟合
        plane_params = self.fit_plane_ransac(depth_map, valid_mask)

        # 2. 边缘检测
        edge_map = self.detect_edges(depth_map)

        # 3. 上下文特征提取
        context_features = self.context_encoder(depth_map)

        # 4. 全局几何特征
        geometry_features = {
            'plane_params': plane_params,  # [B, 4] - 法向量 + 点
            'edge_map': edge_map,          # [B, 1, H, W]
            'context_features': context_features,  # [B, 64, H, W]
        }

        return geometry_features

    def fit_plane_ransac(
        self,
        depth_map: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        RANSAC 平面拟合

        Returns:
            plane_params: [B, 4] - (nx, ny, nz, d) 平面方程 n·x + d = 0
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device

        plane_params = torch.zeros(B, 4, device=device)

        for b in range(B):
            depth = depth_map[b, 0].cpu().numpy()

            if valid_mask is not None:
                valid = valid_mask[b, 0].cpu().numpy() > 0.5
            else:
                valid = np.ones_like(depth, dtype=bool)

            # 获取有效点
            y_coords, x_coords = np.where(valid)
            if len(y_coords) < 10:
                # 点太少，使用默认值
                plane_params[b] = torch.tensor([0., 0., 1., 0.])
                continue

            z_values = depth[y_coords, x_coords]

            # 归一化坐标
            x_norm = (x_coords / W - 0.5) * 2
            y_norm = (y_coords / H - 0.5) * 2
            z_norm = z_values / (z_values.max() + 1e-8)

            # 简单平面拟合 (最小二乘)
            A = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, z_norm, rcond=None)

                # 法向量
                normal = np.array([coeffs[0], coeffs[1], -1.0])
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                d = coeffs[2]

                plane_params[b] = torch.tensor([
                    normal[0], normal[1], normal[2], d
                ], device=device)
            except:
                plane_params[b] = torch.tensor([0., 0., 1., 0.])

        return plane_params

    def detect_edges(self, depth_map: torch.Tensor) -> torch.Tensor:
        """
        使用Sobel算子检测深度边缘

        Returns:
            edge_map: 边缘概率图 [B, 1, H, W]
        """
        B, _, H, W = depth_map.shape

        # Sobel梯度
        depth_np = depth_map[:, 0].cpu().numpy()

        grad_x = np.zeros_like(depth_np)
        grad_y = np.zeros_like(depth_np)

        for b in range(B):
            d = depth_np[b]

            # Sobel算子
            gx = cv2.Sobel(d, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(d, cv2.CV_64F, 0, 1, ksize=3)

            grad_x[b] = gx
            grad_y[b] = gy

        # 梯度幅值
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 归一化到 [0, 1]
        grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (
            grad_magnitude.max() - grad_magnitude.min() + 1e-8
        )

        return torch.from_numpy(grad_magnitude).unsqueeze(1).float().to(depth_map.device)


def compute_context_guidance(
    depth_map: torch.Tensor,
    invalid_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算上下文感知补全引导

    对无效区域的边界像素，计算周围有效像素的深度梯度方向和幅值

    Args:
        depth_map: 深度图 [B, 1, H, W]
        invalid_mask: 无效区域掩码 [B, 1, H, W], bool

    Returns:
        guidance_map: 补全引导向量 [B, 2, H, W] - (梯度方向, 梯度幅值)
    """
    B, _, H, W = depth_map.shape
    device = depth_map.device

    # 计算深度梯度
    depth_float = depth_map.float()

    # x方向梯度
    grad_x = F.pad(depth_float[:, :, :, :-1] - depth_float[:, :, :, 1:],
                   (1, 0, 0, 0), mode='replicate')

    # y方向梯度
    grad_y = F.pad(depth_float[:, :, :-1, :] - depth_float[:, :, 1:, :],
                   (0, 0, 1, 0), mode='replicate')

    # 梯度幅值和方向
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    grad_dir = torch.atan2(grad_y, grad_x)

    # 有效区域掩码
    valid_mask = ~invalid_mask

    # 对无效区域进行引导传播
    # 使用距离加权平均
    guidance = torch.zeros(B, 2, H, W, device=device)

    # 找到无效区域的边界
    invalid_uint8 = invalid_mask[:, 0].float()
    kernel = torch.ones(3, 3, device=device)
    boundary = F.conv2d(invalid_uint8.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0),
                        padding=1) > 0
    boundary = boundary & invalid_uint8.unsqueeze(1)

    # 对边界像素进行插值
    for b in range(B):
        for h in range(H):
            for w in range(W):
                if invalid_mask[b, 0, h, w]:
                    # 找最近的n个有效像素
                    h_range = max(0, h-3), min(H, h+4)
                    w_range = max(0, w-3), min(W, w+4)

                    valid_region = valid_mask[b, 0, h_range[0]:h_range[1],
                                               w_range[0]:w_range[1]]
                    if valid_region.sum() > 0:
                        grad_mag_valid = grad_mag[b, 0, h_range[0]:h_range[1],
                                                   w_range[0]:w_range[1]][valid_region]
                        grad_dir_valid = grad_dir[b, 0, h_range[0]:h_range[1],
                                                   w_range[0]:w_range[1]][valid_region]

                        # 距离加权平均
                        weights = 1.0 / (torch.arange(len(grad_mag_valid)) + 1)
                        guidance[b, 0, h, w] = (grad_mag_valid * weights).sum() / weights.sum()
                        guidance[b, 1, h, w] = (grad_dir_valid * weights).sum() / weights.sum()

    return guidance


# ==================== 分层引导策略 ====================

class LayeredGuidanceModule(nn.Module):
    """
    分层引导模块

    针对无效区域在扩散过程的不同阶段采用不同策略：
    - 高噪声阶段 (t ∈ [800, 1000]): 增强生成权重
    - 低噪声阶段 (t ∈ [0, 800]): 引入几何约束
    """

    def __init__(self, config: Optional[InvalidRegionConfig] = None):
        super().__init__()
        self.config = config or InvalidRegionConfig()

        # 无效区域专用引导分支
        self.fill_guidance_branch = nn.Sequential(
            nn.Conv2d(64 + 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )

        # 门控融合模块
        self.gate_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        conf_guidance: torch.Tensor,      # 原有置信度引导 [B, C, H, W]
        fill_guidance: torch.Tensor,       # 补全引导 [B, 2, H, W]
        invalid_mask: torch.Tensor,        # 无效区域掩码 [B, 1, H, W]
        timestep: int,                     # 当前时间步
        max_timestep: int = 1000           # 最大时间步
    ) -> torch.Tensor:
        """
        分层引导融合

        Args:
            conf_guidance: 原有置信度引导
            fill_guidance: 无效区域补全引导
            invalid_mask: 无效区域掩码
            timestep: 当前时间步

        Returns:
            fused_guidance: 融合后的引导
        """
        B, C, H, W = conf_guidance.shape
        device = conf_guidance.device

        # 计算自适应权重 α
        # 无效区域 α=0，有效区域 α=1
        alpha = (invalid_mask[:, 0] < 0.5).float()  # [B, H, W]

        # 时间步相关权重
        # t > max_timestep/2 时增强无效区域生成
        t_ratio = timestep / max_timestep
        if t_ratio > 0.5:
            fill_weight = t_ratio * 2 - 1  # 0 ~ 1
        else:
            fill_weight = 0.0

        # 无效区域掩码 (无效区域为 1，其余为 0)
        M_invalid = invalid_mask[:, 0].float()  # [B, H, W]

        # 扩展 fill_guidance 到与 conf_guidance 相同通道数
        if fill_guidance.shape[1] != C:
            fill_guidance_expanded = F.interpolate(
                fill_guidance, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            fill_guidance_expanded = fill_guidance

        # 门控融合: G = α * G_conf + (1 - α) * G_fill * M_invalid
        fused = alpha.unsqueeze(1) * conf_guidance + \
                (1 - alpha.unsqueeze(1)) * fill_weight * fill_guidance_expanded * M_invalid.unsqueeze(1)

        return fused

    def get_timestep_weights(self, timestep: int, max_timestep: int = 1000) -> Dict[str, float]:
        """
        获取时间步相关的权重

        Returns:
            weights: 包含各阶段权重的字典
        """
        t_ratio = timestep / max_timestep

        return {
            'generation_weight': max(0, t_ratio - 0.5) * 2,  # 高噪声阶段增强
            'refinement_weight': max(0, 0.5 - t_ratio) * 2 + 0.5,  # 低噪声阶段
            'geometry_weight': 1.0 if t_ratio < 0.5 else (1.0 - t_ratio) * 2,  # 几何约束
        }


# ==================== 损失函数 ====================

class CompletionLoss(nn.Module):
    """
    无效区域补全损失

    包含：
    1. 无效区域补全损失 L_fill
    2. 边界一致性损失 L_consist
    """

    def __init__(self, config: Optional[InvalidRegionConfig] = None):
        super().__init__()
        self.config = config or InvalidRegionConfig()

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        invalid_mask: torch.Tensor,
        original_depth: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算补全损失

        Args:
            pred_depth: 预测深度 [B, 1, H, W]
            target_depth: 目标深度 [B, 1, H, W]
            invalid_mask: 无效区域掩码 [B, 1, H, W], bool
            original_depth: 原始噪声深度 [B, 1, H, W]

        Returns:
            losses: 损失字典
        """
        losses = {}

        # 1. 无效区域补全损失 L_fill (MAE)
        if invalid_mask.sum() > 0:
            fill_loss = F.l1_loss(
                pred_depth[invalid_mask],
                target_depth[invalid_mask]
            )
            losses['fill_loss'] = fill_loss * self.config.fill_weight
        else:
            losses['fill_loss'] = torch.tensor(0.0, device=pred_depth.device)

        # 2. 边界一致性损失 L_consist
        consist_loss = self._compute_boundary_consistency(
            pred_depth, target_depth, invalid_mask
        )
        losses['consist_loss'] = consist_loss * self.config.consistency_weight

        # 3. 总损失
        losses['total_fill_loss'] = losses['fill_loss'] + losses['consist_loss']

        return losses

    def _compute_boundary_consistency(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        invalid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算边界连续性损失
        """
        B, _, H, W = pred_depth.shape

        # 找到无效区域边界
        invalid_float = invalid_mask[:, 0].float()

        # 使用 Sobel 检测边界
        kernel_x = torch.tensor([[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]], dtype=torch.float32, device=pred_depth.device)
        kernel_y = torch.tensor([[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]], dtype=torch.float32, device=pred_depth.device)

        # 边界检测
        boundary_pred = F.conv2d(invalid_float.unsqueeze(1), kernel_x.unsqueeze(0), padding=1).abs() + \
                        F.conv2d(invalid_float.unsqueeze(1), kernel_y.unsqueeze(0), padding=1).abs()
        boundary_mask = (boundary_pred > 0).float()

        # 计算边界处的深度梯度差异
        pred_grad_x = pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:]
        pred_grad_y = pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :]
        target_grad_x = target_depth[:, :, :, :-1] - target_depth[:, :, :, 1:]
        target_grad_y = target_depth[:, :, :-1, :] - target_depth[:, :, 1:, :]

        # 扩展到原始尺寸
        pred_grad_x = F.pad(pred_grad_x, (1, 0, 0, 0), mode='replicate')
        pred_grad_y = F.pad(pred_grad_y, (0, 0, 1, 0), mode='replicate')
        target_grad_x = F.pad(target_grad_x, (1, 0, 0, 0), mode='replicate')
        target_grad_y = F.pad(target_grad_y, (0, 0, 1, 0), mode='replicate')

        # 边界处的梯度损失
        boundary_pred_grad = torch.cat([pred_grad_x, pred_grad_y], dim=1) * boundary_mask.unsqueeze(1)
        boundary_target_grad = torch.cat([target_grad_x, target_grad_y], dim=1) * boundary_mask.unsqueeze(1)

        consist_loss = F.mse_loss(boundary_pred_grad, boundary_target_grad)

        return consist_loss


# ==================== 数据增强工具 ====================

def create_invalid_region_dataset(
    ideal_depth: np.ndarray,
    noise_depth: np.ndarray,
    amp_mask: Optional[np.ndarray] = None,
    invalid_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    创建带无效区域的数据集

    在原始数据基础上人工添加无效区域掩码

    Args:
        ideal_depth: 理想深度图 [H, W]
        noise_depth: 噪声深度图 [H, W]
        amp_mask: 幅值掩码 [H, W]
        invalid_ratio: 无效区域比例
        random_seed: 随机种子

    Returns:
        enhanced_invalid_mask: 增强的无效区域掩码 [H, W]
        enhanced_ideal: 增强后的理想数据
        fill_ground_truth: 用于监督补全的真值
    """
    np.random.seed(random_seed)
    H, W = ideal_depth.shape

    # 创建随机无效区域掩码
    random_invalid = np.random.random((H, W)) < invalid_ratio

    # 避免选择边界区域（选择背景、远距离）
    far_mask = noise_depth > np.percentile(noise_depth, 70)
    background_mask = noise_depth < np.percentile(noise_depth, 10)

    # 组合掩码
    enhanced_invalid = random_invalid & (far_mask | background_mask)

    # 合并原始幅值掩码
    if amp_mask is not None:
        enhanced_invalid = enhanced_invalid | amp_mask

    return enhanced_invalid


# ==================== 评估指标 ====================

class CompletionMetrics:
    """
    无效区域补全评估指标
    """

    @staticmethod
    def fill_accuracy(
        pred_depth: np.ndarray,
        target_depth: np.ndarray,
        invalid_mask: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        计算补全准确率

        Args:
            pred_depth: 预测深度 [H, W]
            target_depth: 目标深度 [H, W]
            invalid_mask: 无效区域掩码 [H, W]
            threshold: 有效补全阈值 (米)

        Returns:
            metrics: 指标字典
        """
        # 无效区域误差
        pred_invalid = pred_depth[invalid_mask]
        target_invalid = target_depth[invalid_mask]

        fill_mae = np.mean(np.abs(pred_invalid - target_invalid))
        fill_rmse = np.sqrt(np.mean((pred_invalid - target_invalid)**2))

        # 有效补全率
        errors = np.abs(pred_invalid - target_invalid)
        valid_fill_ratio = np.mean(errors < threshold)

        return {
            'fill_mae': fill_mae,
            'fill_rmse': fill_rmse,
            'valid_fill_ratio': valid_fill_ratio,
        }

    @staticmethod
    def boundary_consistency(
        pred_depth: np.ndarray,
        target_depth: np.ndarray,
        invalid_mask: np.ndarray,
        boundary_width: int = 5
    ) -> float:
        """
        计算边界连续性误差

        Args:
            pred_depth: 预测深度 [H, W]
            target_depth: 目标深度 [H, W]
            invalid_mask: 无效区域掩码 [H, W]
            boundary_width: 边界宽度

        Returns:
            boundary_error: 边界误差
        """
        # 创建边界掩码
        kernel = np.ones((boundary_width, boundary_width))
        boundary = cv2.dilate(invalid_mask.astype(np.uint8), kernel) - invalid_mask.astype(np.uint8)
        boundary = boundary > 0

        # 边界处的深度梯度差异
        pred_dx = np.abs(pred_depth[:, 1:] - pred_depth[:, :-1])
        pred_dy = np.abs(pred_depth[1:, :] - pred_depth[:-1, :])
        target_dx = np.abs(target_depth[:, 1:] - target_depth[:, :-1])
        target_dy = np.abs(target_depth[1:, :] - target_depth[:-1, :])

        # 计算边界区域的误差
        boundary_dx = np.mean(np.abs(pred_dx - target_dx))
        boundary_dy = np.mean(np.abs(pred_dy - target_dy))

        return (boundary_dx + boundary_dy) / 2

    @staticmethod
    def full_fill_rate(
        pred_depth: np.ndarray,
        target_depth: np.ndarray,
        invalid_mask: np.ndarray,
        error_threshold: float = 0.1
    ) -> float:
        """
        完全补全率

        Args:
            pred_depth: 预测深度 [H, W]
            target_depth: 目标深度 [H, W]
            invalid_mask: 无效区域掩码 [H, W]
            error_threshold: 误差阈值 (米)

        Returns:
            full_fill_rate: 完全补全率
        """
        errors = np.abs(pred_depth[invalid_mask] - target_depth[invalid_mask])
        valid_pixels = np.sum(errors < error_threshold)
        total_invalid = np.sum(invalid_mask)

        if total_invalid == 0:
            return 1.0

        return valid_pixels / total_invalid


# ==================== 集成工具函数 ====================

def enhanced_preprocess(
    iq_data: np.ndarray,
    depth_data: np.ndarray,
    amp_mask: Optional[np.ndarray] = None,
    config: Optional[InvalidRegionConfig] = None
) -> Dict[str, np.ndarray]:
    """
    增强的预处理函数

    返回无效区域掩码、三层置信度、几何先验等

    Returns:
        enhanced_data: 增强数据字典
    """
    if config is None:
        config = InvalidRegionConfig()

    # 1. 检测无效区域
    invalid_mask, amplitude_map, confidence_map = detect_invalid_regions(
        iq_data, depth_data, config
    )

    # 2. 合并幅值掩码
    if amp_mask is not None:
        invalid_mask = invalid_mask | amp_mask
        confidence_map[amp_mask] = 0.0

    # 3. 计算几何先验 (使用深度图)
    geometry_prior = compute_geometry_prior(depth_data, invalid_mask)

    enhanced_data = {
        'invalid_mask': invalid_mask,
        'amplitude_map': amplitude_map,
        'confidence_map': confidence_map,  # 三层置信度
        'geometry_prior': geometry_prior,
        'valid_mask': ~invalid_mask,
    }

    return enhanced_data


def compute_geometry_prior(
    depth_map: np.ndarray,
    invalid_mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    计算几何先验参数
    """
    # 使用RANSAC拟合平面 (简化版本)
    valid_depth = depth_map[~invalid_mask]

    if len(valid_depth) < 100:
        # 点太少，返回默认先验
        return {
            'plane_normal': np.array([0., 0., 1.]),
            'plane_d': 0.,
            'edge_map': np.zeros_like(depth_map),
        }

    # 简单平面拟合
    H, W = depth_map.shape
    y_coords, x_coords = np.where(~invalid_mask)

    x_norm = (x_coords / W - 0.5) * 2
    y_norm = (y_coords / H - 0.5) * 2
    z_norm = valid_depth / (valid_depth.max() + 1e-8)

    A = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, z_norm, rcond=None)

    normal = np.array([coeffs[0], coeffs[1], -1.0])
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    # 边缘检测
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(grad_x**2 + grad_y**2)
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)

    return {
        'plane_normal': normal,
        'plane_d': coeffs[2],
        'edge_map': edge_map,
    }


# ==================== 导出配置 ====================

__all__ = [
    'InvalidRegionConfig',
    'detect_invalid_regions',
    'compute_three_layer_confidence',
    'GeometryPriorExtractor',
    'compute_context_guidance',
    'LayeredGuidanceModule',
    'CompletionLoss',
    'create_invalid_region_dataset',
    'CompletionMetrics',
    'enhanced_preprocess',
]
