import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from dataset import load_ideal
from iq2depth_np import iq2depth
from post_process import fill_holes_with_distance_transform


def zoom_in_fig(ax, img, h, w, xylim, zoom, locs, cmap, vmin, vmax, edge_color='lime', img_type='depth', depsp=None):
    axsin = zoomed_inset_axes(ax, zoom=zoom, loc=locs[0])
    x1, x2, y1, y2 = xylim
    imc = None

    if img_type == 'scatter':
        x_sp_tmp, y_sp_tmp = np.where(img > 0)
        d_sp_tmp = depsp[x_sp_tmp, y_sp_tmp]
        axsin.scatter(y_sp_tmp, h - x_sp_tmp, np.ones_like(x_sp_tmp) * 3, c=d_sp_tmp, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        imc = axsin.imshow(img, extent=(0, w, 0, h), origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)

    axsin.set_xlim([x1, x2])
    axsin.set_ylim([y1, y2])
    mark_inset(ax, axsin, loc1=locs[1], loc2=locs[2], ec=edge_color, linewidth=0.5, fc='None')

    for edge in ['top', 'left', 'bottom', 'right']:
        axsin.spines[edge].set_color(edge_color)
        axsin.spines[edge].set_linewidth(0.5)
    axsin.set_xticks([])
    axsin.set_yticks([])
    return imc


def plot(save_dir, idx, data, vmin=-1, vmax=-1):
    gt = data["Ideal"]

    if vmin == -1 and vmax == -1:
        v_min, v_max = gt.min(), gt.max()
    else:
        v_min, v_max = vmin, vmax

    c_map = 'jet_r'
    diff_map = "bwr"
    font_size = 16

    cols = len(data.keys()) 
    titles = tuple(data.keys())
    depths = tuple(data.values())

    # configs
    v_min, v_max = gt.min(), gt.max()
    diff_min, diff_max = -0.1, 0.1
    zoom_min, zoom_max = v_min, v_max
    h, w, xylim, zoom, locs, ec = gt.shape[0], gt.shape[1], [110, 200, 230, 380], 1.5, [4, 1, 3], 'k'     
    
    # plot images
    fig, axs = plt.subplots(2, cols, figsize=(3 * cols, 5), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    axs = axs.ravel()
    
    # depths
    img_gt = axs[0].imshow(gt, extent=(0, w, 0, h), origin='upper', cmap=c_map, vmin=v_min, vmax=v_max)
    # zoom_in_fig(axs[0], gt, h, w, xylim, zoom, locs, 'jet_r', vmin=zoom_min, vmax=zoom_max, edge_color=ec)

    for i in range(1, cols):
        axs[i].imshow(depths[i], extent=(0, w, 0, h), origin='upper', cmap=c_map, vmin=v_min, vmax=v_max)
        # img_zoom = zoom_in_fig(axs[i], depths[i], h, w, xylim, zoom, locs, 'jet_r', vmin=zoom_min, vmax=zoom_max, edge_color=ec)

    # diff
    for i in range(1, cols):
        img_diff = axs[i + cols].imshow(depths[i] - gt, extent=(0, w, 0, h), origin='upper', cmap=diff_map, vmin=diff_min, vmax=diff_max)
    # img_giga = axs[-1].imshow(data["GIGA-ToF"] - gt, extent=(0, w, 0, h), origin='upper', cmap=diff_map, vmin=diff_min, vmax=diff_max)

    # Add titles
    for i in range(cols):
        axs[i].set_title(f"({chr(97 + i)}) {titles[i]}", fontsize=font_size)
        axs[i + cols].set_title(f"({chr(97 + i + cols)}) {titles[i]} Error", fontsize=font_size)

    # colorbar of depth 
    left, bottom, width, height = axs[cols - 1].get_position().bounds
    cax = fig.add_axes([left + width + 0.01, bottom, 0.008, height])
    cbar = plt.colorbar(img_gt, cax=cax, ticks=np.linspace(v_min, v_max, 5), format='%.1f')
    cbar.ax.tick_params(labelsize=font_size-2)
    cbar.set_label('Depth (m)', rotation=90, fontsize=font_size-2)

    # # colorbar of depth
    # left, bottom, width, height = axs[cols - 1].get_position().bounds
    # cax = fig.add_axes([left + width + 0.08, bottom, 0.008, height])
    # cbar = plt.colorbar(img_zoom, cax=cax, ticks=np.linspace(zoom_min, zoom_max, 5), format='%.1f')
    # cbar.ax.tick_params(labelsize=font_size-2)
    # cbar.set_label('Depth in Zoomed Region\n(m)', rotation=90, fontsize=font_size-6)

    # colorbar of diff
    left, bottom, width, height = axs[-1].get_position().bounds
    cax = fig.add_axes([left + width + 0.01, bottom, 0.008, height])
    cbar = plt.colorbar(img_diff, cax=cax, ticks=np.linspace(diff_min, diff_max, 5), format='%.2f')
    cbar.ax.tick_params(labelsize=font_size-2)
    cbar.set_label('Error (m)', rotation=90, fontsize=font_size-2)

    # 关闭坐标轴的值    
    for i in range(2 * cols):
        axs[i].set_xticks([])  # 去掉x轴刻度
        axs[i].set_yticks([])  # 去掉y轴刻度

    # plt.show()
    plt.savefig(f'{save_dir}/{idx}.png', bbox_inches='tight', dpi=400)
    plt.close()


if __name__ == '__main__':
    save_root = "./plots"
    os.makedirs(save_root, exist_ok=True)

    version = "cg_W_0.2"
    idx = "1516056249759530"
    # idx = "1519695966159931"

    gt = load_ideal(f"/data/pre_student/hcy/pbrt/gt/{idx}")[0]
    
    # noise_raw = np.fromfile(f"/home/lab507/Documents/JishenLin/GLRUN/FLAT/noise/{idx}", dtype=np.float32).reshape((424, 512, 9))
    # concat_IQ = np.stack([
    #     noise_raw[:, :, 4], noise_raw[:, :, 3], 
    #     noise_raw[:, :, 1], noise_raw[:, :, 0], 
    #     noise_raw[:, :, 7], noise_raw[:, :, 6], 
    # ])
    # noise = iq2depth(concat_IQ, noise_raw)
    # noise = np.nan_to_num(noise, nan=0)
    noise = np.load(f"drafts/noise_depth/{idx}.npy") / 1e3
    noise = np.nan_to_num(noise, nan=0)
    
    # urncgtv = np.load(f"results/debug_cg_36/depth/{idx}.npy")
    # urncgtv_ori = np.load(f"/data/pre_student/hcy/Unrolling_NCGTV/results/cropped36_ori/depth/{idx}.npy")
    # # urncgtv = np.load("noise_loadraw.npy")
    # urncgtv = np.nan_to_num(urncgtv, nan=0)
    # urncgtv_ori = np.nan_to_num(urncgtv_ori, nan=0)
    # cg_w = np.load(f"/data/pre_student/hcy/Unrolling_NCGTV/results/cg_W/depth/{idx}.npy")
    # cg_w = np.nan_to_num(cg_w, nan=0)
    # cg_w_2 = np.load(f"/data/pre_student/hcy/Unrolling_NCGTV/results/cg_W_0.2/depth/{idx}.npy")
    # cg_w_2 = np.nan_to_num(cg_w_2, nan=0)
    # filled = np.load(f"results/{version}/depth_filled/{idx}.npy")

    # glrun = np.fromfile(f"drafts/GLRUN_results/depth/{idx}", dtype=np.float32).reshape((424, 512)) / 1e3
    # glrun = np.nan_to_num(glrun, nan=0)
    # mask = ((urncgtv <= 0) | (urncgtv > 6))
    # filled= fill_holes_with_distance_transform(urncgtv, mask)

    # top, down, left, right = 0, 408, 0, 501
    top, down, left, right = 0, 256, 0, 256
    data = {
        "Ideal": gt[top:down, left:right],
        "noise": noise[top:down, left:right],
        "GLRUN": glrun[top:down, left:right],
        "GLRUN_0.8": urncgtv_ori[top:down, left:right],
        "CG": urncgtv[top:down, left:right],
        "CG_W": cg_w[top:down, left:right],
        "CG_a": cg_w_2[top:down, left:right],
    }
    plot(save_root, f"{version}_{idx}", data)
    # plot("./", "load_raw", data)
