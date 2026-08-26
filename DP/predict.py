import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import ot  # Python Optimal Transport
from tqdm import tqdm

# 导入之前定义好的 PyTorch 版 MLP
# from networks_pytorch import MLP
from KS_data_3d import KSDataGenerator
from networks import MLP,ResMLP
from set_seed import set_seed
from gaussian import GMMDataGenerator


# ==========================================
# 1. 距离计算工具 (保持 NumPy 逻辑)
# ==========================================

def calculate_swd(x1, x2, n_projections=10000, seed=42):
    """ 使用 POT 库计算 SWD """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return ot.sliced_wasserstein_distance(x1, x2, n_projections=n_projections, p=2, seed=seed)


def dist_approx_w2(x1, x2, bins=20, dims=2):
    """
    基于网格的 W2 近似计算
    dims=2: 强制只使用前两个维度 (x, y) 进行计算
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    # === [修改] 强制截取前 dims 个维度 ===
    if dims is not None and x1.shape[1] >= dims:
        x1 = x1[:, :dims]
        x2 = x2[:, :dims]
    # ===================================

    # 计算 2D 直方图 (density=True 保证归一化)
    Href, edges = np.histogramdd(x2, bins=bins, density=True)
    H, _ = np.histogramdd(x1, bins=edges, density=True)

    # 计算每个 bin 的体积 (如果是 2D，则是面积 dx * dy)
    vol = np.prod([e[1] - e[0] for e in edges])

    # 获取网格中心点
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    mesh = np.meshgrid(*centers, indexing='ij')

    # 将网格展平为坐标点列表
    pos = np.vstack([m.flatten() for m in mesh]).T

    # 计算距离矩阵 (默认是平方欧氏距离)
    W = ot.dist(pos)

    # 计算 W2 距离 (注意：ot.emd2 返回的是 Cost，即 W2^2)
    # 如果想返回 W2，建议最后开根号: return np.sqrt(ot.emd2(...))
    return ot.emd2(Href.flatten() * vol, H.flatten() * vol, W)


# ==========================================
# 2. 绘图辅助函数 (与原代码基本一致)
# ==========================================

def _to_numpy(a):
    return np.asarray(a).ravel()


def square_limits(x, y, pad=0.1):
    x, y = _to_numpy(x), _to_numpy(y)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    span = max(xmax - xmin, ymax - ymin)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad)
    return cx - half, cx + half, cy - half, cy + half


def hist2d_square(x, y, ax, labels=('x', 'y'), bins=60, cmap='viridis', xylims=None, vmax=None):
    if xylims is None:
        xylims = square_limits(x, y)
    counts, xedges, yedges, im = ax.hist2d(
        _to_numpy(x), _to_numpy(y), bins=bins,
        range=[[xylims[0], xylims[1]], [xylims[2], xylims[3]]],
        cmap=cmap, vmax=vmax
    )
    ax.set_facecolor(plt.get_cmap(cmap)(0))
    ax.set_xlim(xylims[0], xylims[1]);
    ax.set_ylim(xylims[2], xylims[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return xylims, im, np.nanmax(counts)


def plot_comparison(X_sim, X_gen, sigma, w2_val=None, swd_score=None, save_dir='./plots/'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    def add_cbar(im, ax):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Reference
    l_xy, im00, v_xy = hist2d_square(X_sim[:, 0], X_sim[:, 1], axes[0, 0], labels=('x', 'y'))
    l_xz, im01, v_xz = hist2d_square(X_sim[:, 0], X_sim[:, 2], axes[0, 1], labels=('x', 'z'))
    l_yz, im02, v_yz = hist2d_square(X_sim[:, 1], X_sim[:, 2], axes[0, 2], labels=('y', 'z'))
    for im, ax in zip([im00, im01, im02], axes[0, :]): add_cbar(im, ax)
    axes[0, 1].set_title('Reference (Simulation)', fontsize=12)

    # Prediction
    _, im10, _ = hist2d_square(X_gen[:, 0], X_gen[:, 1], axes[1, 0], labels=('x', 'y'), xylims=l_xy, vmax=v_xy)
    _, im11, _ = hist2d_square(X_gen[:, 0], X_gen[:, 2], axes[1, 1], labels=('x', 'z'), xylims=l_xz, vmax=v_xz)
    _, im12, _ = hist2d_square(X_gen[:, 1], X_gen[:, 2], axes[1, 2], labels=('y', 'z'), xylims=l_yz, vmax=v_yz)
    for im, ax in zip([im10, im11, im12], axes[1, :]): add_cbar(im, ax)
    axes[1, 1].set_title(f'DPNet Prediction\nW2={w2_val:.4f}, SWD={swd_score:.4f}', fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'comparison_sigma_{int(sigma)}.png'), dpi=200)
    plt.close()


# ==========================================
# 3. 主推理逻辑
# ==========================================

import os
import json
import torch
import numpy as np


# 假设你的模型类和函数在这里定义
# from your_module import MLP, KSDataGenerator, calculate_swd, dist_approx_w2, plot_comparison, set_seed

def run_predict(exp_dir, test_sigma, model_specified):
    """
    exp_dir: 训练时创建的实验文件夹路径
    test_sigma: list, 例如 [0.1, 0.5, 1.0]
    model_specified: 模型文件名 (不含 .pth)
    """
    # --- 1. 加载配置 (保持不变) ---
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    dim = config["dim"]
    layer_sizes = config["layers"]
    seed = config.get("seed", 88)
    data_filepath = config.get("data_filepath", f'./KS_{dim}d_data/')
    model_path = os.path.join(exp_dir, f'{model_specified}.pth')

    # --- 2. 环境准备 (保持不变) ---
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading experiment from: {exp_dir}")

    # --- 3. 模型加载 (保持不变) ---
    model = MLP(layer_sizes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # --- 4. 数据准备 (保持不变) ---
    # test_gen = KSDataGenerator(mode='test', sigma=test_sigma, N=20000,dim=dim, filepath=data_filepath)
    test_gen=GMMDataGenerator(mode="test", filepath=data_filepath, dim=dim,weights=test_sigma)
    test_inputs, test_targets = test_gen.get_all()


    # --- 5. 推理 (保持不变) ---
    with torch.no_grad():
        inputs_torch = torch.as_tensor(test_inputs, dtype=torch.float32).to(device)
        predictions_torch = model(inputs_torch)
        predictions = predictions_torch.cpu().numpy()

    # --- 6. 循环计算指标并保存结果 (已修改) ---
    eval_results = {}
    plot_dir = os.path.join(exp_dir, "test_plots")
    os.makedirs(plot_dir, exist_ok=True)
    pred_save_path = os.path.join(exp_dir, "all_predictions.npy")


    np.save(pred_save_path, predictions)

    # [新增] 初始化列表用于存储汇总数据
    dist_log = []  # 存储每个 sigma 的 W2 距离
    f_x_mean_list = []  # 存储 network 预测数据的均值 (第0维)
    ref_x_mean_list = []  # 存储 reference 真实数据的均值 (第0维)

    print(f"{'Sigma':<10} | {'W2 Score':<12} | {'SWD Score':<12}")
    print("-" * 40)

    for i, s in enumerate(test_sigma):
        # 兼容处理：确保数据是 numpy
        sim_data = test_targets[i]
        if torch.is_tensor(sim_data):
            sim_data = sim_data.cpu().numpy()

        gen_data = predictions[i]

        # 计算指标
        swd_score = calculate_swd(gen_data, sim_data, n_projections=500)
        w2_score = dist_approx_w2(gen_data, sim_data, bins=20)

        # [新增] 收集汇总数据 (对应 f[:,:,0] 的逻辑)
        # 假设数据形状为 [Samples, Dims]，我们取第 0 维度的均值
        current_gen_mean = np.mean(gen_data[:, 0])
        current_ref_mean = np.mean(sim_data[:, 0])

        f_x_mean_list.append(current_gen_mean)
        ref_x_mean_list.append(current_ref_mean)
        dist_log.append(w2_score)

        print(f"{s:<10.2f} | {w2_score:.6f}     | {swd_score:.6f}")

        # 绘制单个 sigma 的对比图
        plot_comparison(sim_data, gen_data, s, w2_val=w2_score, swd_score=swd_score, save_dir=plot_dir)

        eval_results[float(s)] = {
            "w2": float(w2_score),
            "swd": float(swd_score),
            "mean_gen": float(current_gen_mean),  # 顺便保存到 json
            "mean_ref": float(current_ref_mean)
        }

    # --- [新增] 绘制汇总图表 (你要求的两个图) ---

    # 转换为 numpy 数组方便运算
    f_x_mean = np.array(f_x_mean_list)
    ref_x_mean = np.array(ref_x_mean_list)
    dist_log = np.array(dist_log)
    sigma_arr = np.array(test_sigma)

    # 图 1: Network vs Reference 均值对比
    fig1 = plt.figure(figsize=(3, 3))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(sigma_arr, f_x_mean, '+', label='network')
    ax1.plot(sigma_arr, ref_x_mean, 'x', label='reference')
    ax1.legend()
    ax1.set_xlabel('A')  # 保持了原本代码中的 'A'，如需改为 'Sigma' 可在此修改
    ax1.set_title("Mean Comparison")
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "summary_mean_comparison.png"), dpi=300)
    plt.close(fig1)

    # 图 2: W2 距离 vs 均值误差
    fig2 = plt.figure(figsize=(3, 3))
    ax2 = plt.subplot(1, 1, 1)
    ax2.plot(sigma_arr, dist_log, label='W2 (approx.)')
    ax2.plot(sigma_arr, np.abs(f_x_mean - ref_x_mean), '--', label='err of Mean')
    ax2.legend()
    ax2.set_xlabel('A')
    ax2.set_title("W2 Dist vs Mean Error")
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "summary_w2_vs_mean_err.png"), dpi=300)
    plt.close(fig2)

    # --- 7. 保存结果到文件 (保持不变) ---
    with open(os.path.join(exp_dir, "test_results.json"), "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"\nAnalysis complete. Results and summary plots saved in: {plot_dir}")
def run_predictions_for_all_experiments(base_dir, test_sigma, model_specified):
    # 遍历 base_dir 目录下的所有文件夹
    for experiment_folder in os.listdir(base_dir):
        # 获取文件夹的完整路径
        experiment_path = os.path.join(base_dir, experiment_folder)

        # 检查是否是文件夹
        if os.path.isdir(experiment_path):
            # 在这里你可以添加一些条件来过滤特定格式的文件夹（例如根据文件夹名字判断）
            # 假设你只关心以 "dim" 作为部分名称的文件夹
            if "dim" in experiment_folder:
                print(f"Running prediction for experiment: {experiment_folder}")
                # 调用 run_predict 函数来对每个实验文件夹进行预测
                run_predict(experiment_path, test_sigma, model_specified)
            else:
                print(f"Skipping folder: {experiment_folder} (doesn't match expected format)")
        else:
            print(f"Skipping non-directory: {experiment_folder}")



if __name__ == "__main__":
    # 你只需要在这里指定实验文件夹的路径即可
    # 示例调用
    base_dir = "./experiments/observe"
    test_sigma = [0.5]
    model_specified = "best_model"

    # 运行所有符合条件的实验文件夹
    run_predictions_for_all_experiments(base_dir, test_sigma, model_specified)
    # target_experiment = base_dir + "20260110_035541_dim32"
    # run_predict(target_experiment,test_sigma,model_specified)