"""
可视化 DepthCAD 推理结果
将 IQ、mask、GT depth、pred depth 转为 PNG 图像进行对比
"""
import cv2
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_split_iq(base_path, target_size=(240, 320)):
    """
    加载分通道存储的 IQ 数据 (100_A.npy ~ 100_F.npy)
    返回 shape (6, H, W)
    IQ order: [I30, Q30, I40, Q40, I58, Q58]
    """
    suffixes = ['A', 'B', 'C', 'D', 'E', 'F']
    channels = []
    for suffix in suffixes:
        ch_path = f"{base_path}_{suffix}.npy"
        data = np.load(ch_path).astype(np.float32)
        channels.append(data)

    iq = np.stack(channels, axis=0)  # (6, H, W)

    # Resize if needed
    target_h, target_w = target_size
    if iq.shape[1:] != (target_h, target_w):
        iq_resized = np.zeros((6, target_h, target_w), dtype=np.float32)
        for i in range(6):
            iq_resized[i] = cv2.resize(iq[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        iq = iq_resized

    return iq

def compute_mask_from_iq(iq):
    """从 IQ 数据计算 mask（基于 amplitude）"""
    # IQ order: [I30, Q30, I40, Q40, I58, Q58]
    amp_30 = np.sqrt(iq[0]**2 + iq[1]**2)
    amp_40 = np.sqrt(iq[2]**2 + iq[3]**2)
    amp_58 = np.sqrt(iq[4]**2 + iq[5]**2)
    amplitude = (amp_30 + amp_40 + amp_58) / 3.0

    # Mask: amplitude 低于阈值的区域为 0
    threshold = np.percentile(amplitude, 5)
    mask = (amplitude > threshold).astype(np.uint8)
    return mask

def load_confidence(conf_path, target_size=(240, 320)):
    """加载置信度图"""
    conf = np.load(conf_path).astype(np.float32)
    conf_resized = cv2.resize(conf, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return conf_resized

def load_ideal_iq(ideal_iq_path, target_size=(240, 320)):
    """加载 GT IQ 数据"""
    iq = np.load(ideal_iq_path)  # shape: (6, H, W)
    iq_resized = np.zeros((6, target_size[0], target_size[1]), dtype=np.float32)
    for i in range(6):
        iq_resized[i] = cv2.resize(iq[i], (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return iq_resized

def iq_to_depth(iq):
    """
    将 IQ 转为深度图（与 depth_estimator.py 中的逻辑一致）
    IQ order: [I30, Q30, I40, Q40, I58, Q58]
    """
    maxd = 10.0
    nt = 5000
    freqVec = np.array([40, 1e2 / 3.3, 1e2 / 1.7], dtype=np.float32) * 1e6
    nf = len(freqVec)

    delayVec = np.linspace(0, 2 * maxd, nt, dtype=np.float32)
    DepthRange = delayVec / 2.0

    c = 3e8
    CandidatePhases = np.empty((nt, nf), dtype=np.float32)
    for fi in range(nf):
        CandidatePhases[:, fi] = 2.0 * np.pi * 2.0 * DepthRange / (c / freqVec[fi])

    C_I = np.cos(CandidatePhases)
    C_Q = np.sin(CandidatePhases)
    C_concat = np.hstack([C_I, C_Q])
    C_T = C_concat.T  # (2*nf, nt)

    # IQ -> [I40, I30, I58] 和 [Q40, Q30, Q58]
    h = np.stack([
        iq[2], iq[0], iq[4],  # [I40, I30, I58] - Real
        iq[3], iq[1], iq[5]   # [Q40, Q30, Q58] - Imag
    ], axis=0)

    P_I = h[:nf].astype(np.float32)
    P_Q = h[nf:].astype(np.float32)

    # 归一化
    amp = np.sqrt(P_I**2 + P_Q**2) + 1e-12
    P_I /= amp
    P_Q /= amp

    H, W = P_I.shape[1:]
    N = H * W

    P_I_flat = P_I.transpose(1, 2, 0).reshape(N, nf)
    P_Q_flat = P_Q.transpose(1, 2, 0).reshape(N, nf)
    P_concat = np.hstack([P_I_flat, P_Q_flat])

    # 矩阵乘法
    score = np.dot(P_concat, C_T)
    idx = np.argmax(score, axis=1)
    depths = DepthRange[idx].reshape(H, W)

    return depths

def depth_to_color(depth, mask=None, vmin=None, vmax=None, add_colorbar=True, save_colorbar_path=None):
    """将深度图转为彩色以便观察（带颜色条版本）

    Args:
        depth: 深度图
        mask: 可选的 mask
        vmin, vmax: 归一化范围，如果为 None 则使用数据的 min/max
        add_colorbar: 是否添加颜色条
        save_colorbar_path: 保存颜色条的路径
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    d_min = vmin if vmin is not None else depth.min()
    d_max = vmax if vmax is not None else depth.max()

    # 应用 mask
    depth_vis = depth.copy()
    if mask is not None:
        depth_vis = depth_vis * mask

    # 使用 matplotlib 绘制（更美观）
    fig, ax = plt.subplots(figsize=(8, 6))

    # JET colormap
    im = ax.imshow(depth_vis, cmap='jet', vmin=d_min, vmax=d_max)

    # 添加 colorbar
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Depth (m)', fontsize=12)

    ax.set_title(f'Depth Map\nRange: [{d_min:.2f}, {d_max:.2f}] m', fontsize=12)
    ax.axis('off')

    # 转为 numpy 数组
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    depth_color = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    plt.close(fig)

    # 保存单独的颜色条
    if save_colorbar_path:
        fig_cb, ax_cb = plt.subplots(figsize=(2, 6))
        norm = mcolors.Normalize(vmin=d_min, vmax=d_max)
        cb = fig_cb.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax_cb)
        cb.set_label('Depth (m)', fontsize=12)
        ax_cb.axis('off')
        fig_cb.canvas.draw()
        w_cb, h_cb = fig_cb.canvas.get_width_height()
        colorbar_img = np.frombuffer(fig_cb.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_cb, w_cb, 3)
        colorbar_img = cv2.cvtColor(colorbar_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_colorbar_path, colorbar_img)
        plt.close(fig_cb)

    return depth_color

def save_iq_channel(iq, channel_idx, mask, out_path):
    """保存单个 IQ 通道为灰度图"""
    ch = iq[channel_idx]
    ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8) * 255
    ch_img = ch_norm.astype(np.uint8)
    if mask is not None:
        ch_img = ch_img * mask
    cv2.imwrite(out_path, ch_img)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="bathroom/0")
    parser.add_argument("--idx", type=str, default="100")
    parser.add_argument("--data_dir", type=str, default="/data/pre_student/GJ/DepthCAD/pbrt_dataset/data")
    parser.add_argument("--pred_depth", type=str, default=None)  # 预测的深度 npy
    parser.add_argument("--out_dir", type=str, default="/data/pre_student/GJ/DepthCAD/output/visualization")
    parser.add_argument("--exp_name", type=str, default=None)   # 实验名（如 checkpoint 名称），会作为子文件夹
    parser.add_argument("--target_size", type=int, nargs=2, default=[240, 320])
    args = parser.parse_args()

    target_size = tuple(args.target_size)

    # 创建输出目录：out_dir / [exp_name/] scene / idx
    if args.exp_name:
        out_dir = os.path.join(args.out_dir, args.exp_name, args.scene, f"{args.idx}")
    else:
        out_dir = os.path.join(args.out_dir, args.scene, f"{args.idx}")
    os.makedirs(out_dir, exist_ok=True)

    # 加载 GT depth（用于 pred 的归一化，始终需要）
    print("Computing GT depth for normalization...")
    ideal_iq_base = os.path.join(args.data_dir, "ideal_IQ_masked", args.scene, f"{args.idx}")
    ideal_iq = load_split_iq(ideal_iq_base, target_size)
    gt_depth = iq_to_depth(ideal_iq)
    vmin, vmax = gt_depth.min(), gt_depth.max()

    # 保存预测深度（只有指定 --pred_depth 时才运行）
    if args.pred_depth and os.path.exists(args.pred_depth):
        print(f"Loading pred depth: {args.pred_depth}")
        pred_depth = np.load(args.pred_depth)
        pred_depth_color = depth_to_color(pred_depth, mask=None, vmin=vmin, vmax=vmax)
        cv2.imwrite(os.path.join(out_dir, "pred_depth.png"), pred_depth_color)
        np.save(os.path.join(out_dir, "pred_depth.npy"), pred_depth)
        pred_depth_color_raw = depth_to_color(pred_depth, mask=None)
        cv2.imwrite(os.path.join(out_dir, "pred_depth_raw.png"), pred_depth_color_raw)

    # 完整模式（IQ / mask / confidence / GT depth / noisy depth）
    if not args.pred_depth:
        noise_iq_base = os.path.join(args.data_dir, "noise_IQ_masked", args.scene, f"{args.idx}")
        conf_path = os.path.join(args.data_dir, "confidence_masked", args.scene, f"{args.idx}.npy")

        print(f"Loading: {noise_iq_base}_A~F.npy")
        noise_iq = load_split_iq(noise_iq_base, target_size)
        mask = compute_mask_from_iq(noise_iq)

        print(f"Loading: {conf_path}")
        confidence = load_confidence(conf_path, target_size)

        print(f"Loading: {ideal_iq_base}_A~F.npy")
        channel_names = ["I30", "Q30", "I40", "Q40", "I58", "Q58"]
        for i, name in enumerate(channel_names):
            save_iq_channel(noise_iq, i, mask, os.path.join(out_dir, f"noise_IQ_{name}.png"))
            save_iq_channel(ideal_iq, i, None, os.path.join(out_dir, f"ideal_IQ_{name}.png"))

        amp_30 = np.sqrt(noise_iq[0]**2 + noise_iq[1]**2)
        amp_40 = np.sqrt(noise_iq[2]**2 + noise_iq[3]**2)
        amp_58 = np.sqrt(noise_iq[4]**2 + noise_iq[5]**2)
        amp_avg = (amp_30 + amp_40 + amp_58) / 3.0
        amp_norm = (amp_avg / amp_avg.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "noise_amplitude.png"), amp_norm)

        amp_30_gt = np.sqrt(ideal_iq[0]**2 + ideal_iq[1]**2)
        amp_40_gt = np.sqrt(ideal_iq[2]**2 + ideal_iq[3]**2)
        amp_58_gt = np.sqrt(ideal_iq[4]**2 + ideal_iq[5]**2)
        amp_avg_gt = (amp_30_gt + amp_40_gt + amp_58_gt) / 3.0
        amp_norm_gt = (amp_avg_gt / amp_avg_gt.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "ideal_amplitude.png"), amp_norm_gt)

        cv2.imwrite(os.path.join(out_dir, "mask.png"), mask * 255)

        conf_norm = (confidence / confidence.max() * 255).astype(np.uint8)
        conf_color = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, "confidence.png"), conf_color)

        print("Computing GT depth...")
        gt_depth_color = depth_to_color(gt_depth, mask=None)
        cv2.imwrite(os.path.join(out_dir, "gt_depth.png"), gt_depth_color)
        np.save(os.path.join(out_dir, "gt_depth.npy"), gt_depth)

        print("Computing noisy depth...")
        noisy_depth = iq_to_depth(noise_iq)
        noisy_depth_color = depth_to_color(noisy_depth, mask=None, vmin=vmin, vmax=vmax)
        cv2.imwrite(os.path.join(out_dir, "noisy_depth.png"), noisy_depth_color)
        np.save(os.path.join(out_dir, "noisy_depth.npy"), noisy_depth)

    print(f"\nVisualization saved to: {out_dir}")

if __name__ == "__main__":
    main()