import torch
import numpy as np
import ot
import torch.optim as optim


# ==========================================
# 1. 标准 Wasserstein Solver (Stateful)
#    需要维护 gamma 矩阵和迭代计数器
# ==========================================
class StandardWassersteinSolver:
    def __init__(self, n_batch_renew, n_gamma_renew):
        self.n_batch_renew = n_batch_renew
        self.n_gamma_renew = n_gamma_renew
        self.gamma = None
        self.counter = 0

    def __call__(self, x, y, device, **kwargs):
        """
        x, y shape: [Batch, N, Dim]
        Return: [Batch] (每个样本的距离)
        """
        batch_size, n, dim = x.shape

        # --- 1. Renew Batch: 重置 Gamma ---
        if self.counter % self.n_batch_renew == 0 or self.gamma is None:
            self.gamma = torch.ones(batch_size, n, n, device=device) / n

        # --- 2. Renew Gamma: 求解 OT (使用 CPU POT 库) ---
        if self.counter % self.n_gamma_renew == 0:
            x_cpu = x.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()

            # 【核心修改】预分配一个空的 numpy 数组 [B, N, N]
            # 类似 JAX 中的 gamma 初始化，这里用 zeros 或 empty 都可以
            gamma_new_cpu = np.zeros((batch_size, n, n))

            for i in range(batch_size):
                # 计算单个样本的 OT 矩阵
                T = self.solver_EMD(x_cpu[i], y_cpu[i])

                # 【核心修改】原地赋值 (In-place assignment)
                # 对应 JAX 的 gamma.at[ib].set(newval)
                gamma_new_cpu[i] = T

            # 循环结束后，一次性转为 Tensor 并移动到 GPU
            self.gamma = torch.from_numpy(gamma_new_cpu).to(device).type_as(x)

            # --- 3. 计算 Loss ---
        C = torch.cdist(x, y, p=2) ** 2
        loss_batch = torch.mean(C * self.gamma, dim=(0,1))

        self.counter += 1
        return loss_batch
    def solver_EMD(self,x,y):
        C = ot.dist(x, y)
        n = x.shape[0]
        a, b = np.ones((n,)) / n, np.ones((n,)) / n
        return ot.emd(a, b, C) * n



# ==========================================
# 2. Sliced Wasserstein 系列 (Stateless)
#    包括 Sliced, Max-Sliced, Spherical
# ==========================================
import torch
import torch.nn.functional as F


class SlicedWassersteinSolver:
    def __init__(self, n_projections=10, p=2, type="sliced",
                 lr=0.001, max_iter=50):
        """
        Args:
            n_projections: 投影数量。
                           - 对于 'sliced'，越大越准（推荐 100-1000）。
                           - 对于 'max_optimized'，这代表寻找几个最大方向，通常设为 1 即可。
            p: 范数 (通常为 1 或 2)。
            type: 'sliced', 'spherical', 'max', 'max_optimized'。
            lr: (仅用于 max_optimized) 梯度上升的学习率。
            max_iter: (仅用于 max_optimized) 寻找最大方向的迭代次数。
        """
        self.n_projections = n_projections
        self.p = p
        self.type = type
        self.lr = lr
        self.max_iter = max_iter

    def _get_projections(self, dim, device):
        """生成初始投影方向"""
        theta = torch.randn(dim, self.n_projections, device=device)
        theta = theta / (torch.norm(theta, dim=0, keepdim=True) + 1e-8)
        return theta

    def _compute_sort_and_dist(self, x, y, theta):
        """核心计算逻辑：投影 -> 排序 -> 计算距离"""
        # 1. 投影
        # x: [Batch, N, Dim], theta: [Dim, L] -> x_proj: [Batch, N, L]
        x_proj = torch.matmul(x, theta)
        y_proj = torch.matmul(y, theta)

        # 特殊处理 Spherical 类型
        if self.type == "spherical":
            x_proj = x_proj / (torch.abs(x_proj) + 1e-8)
            y_proj = y_proj / (torch.abs(y_proj) + 1e-8)

        # 2. 排序
        # PyTorch 的 sort 在值上是可导的
        x_sorted, _ = torch.sort(x_proj, dim=1)
        y_sorted, _ = torch.sort(y_proj, dim=1)

        # 3. 计算距离
        # [Batch, n_projections]
        per_proj_dist = torch.pow(torch.abs(x_sorted - y_sorted), self.p).mean(dim=1)
        return per_proj_dist

    def __call__(self, x, y, device, **kwargs):
        batch_size, n_points, dim = x.shape

        # --- 情况 A: 优化算法 (Max-Optimized) ---
        if self.type == "max_optimized":
            # 1. 初始化 Theta (需开启梯度)
            theta = self._get_projections(dim, device)
            theta.requires_grad_(True)

            # 2. 定义优化器 (按照你的逻辑使用 Adam)
            optimizer = optim.Adam([theta], lr=self.lr)
            x_detached = x.detach()
            y_detached = y.detach()

            # 3. 迭代优化
            for _ in range(self.max_iter):
                optimizer.zero_grad()

                # 前向计算距离
                per_proj_dist = self._compute_sort_and_dist(x_detached, y_detached, theta)

                # 定义 Loss: 我们要最大化距离，所以最小化负距离
                # 对 Batch 取平均，以便优化出一个通用的 theta
                loss = -per_proj_dist.mean()

                # 反向传播
                # 注意：通常在循环内部不需要 retain_graph=True，除非你在外层还有梯度依赖。
                # 如果是单纯计算距离，loss.backward() 即可。
                loss.backward()

                # 更新 theta
                optimizer.step()

                # 4. 投影回单位球 (Projected Gradient Ascent)
                # 必须在 no_grad 下操作 data，防止破坏计算图
                with torch.no_grad():
                    theta.data = theta.data / (torch.norm(theta.data, dim=0, keepdim=True) + 1e-8)

            # 5. 计算最终距离 (不带梯度，仅作为度量值返回)
            # 使用最终优化好的 theta 再算一次
            theta_final = theta.detach()
            final_dists = self._compute_sort_and_dist(x, y, theta_final)

            # Max-SW 定义通常是取那个最大的距离
            # 如果 n_projections > 1，我们取这些方向中最大的那个
            return final_dists.max(dim=1)[0].mean()* dim

        # --- 情况 B: 随机采样 (Sliced, Max, Spherical) ---
        else:
            theta = self._get_projections(dim, device)
            per_proj_dist = self._compute_sort_and_dist(x, y, theta)* dim

            if self.type == "max":
                # 随机采样取最大 (Naive approach)
                dist = per_proj_dist.max(dim=1)[0]
            else:
                # Sliced or Spherical (取平均)
                dist = per_proj_dist.mean(dim=1)

        return dist * dim  # [Batch]


# ==========================================
# 3. 统一工厂函数 (Factory)
# ==========================================
def get_metric_solver(params):
    """
    根据 params["type"] 返回对应的 Solver 实例 (Callable)
    """
    dist_type = params["type"]

    # 1. 标准 Wasserstein (需要 n_batch_renew, n_gamma_renew)
    if dist_type == "w2":
        print(
            f"Initializing Standard Wasserstein Solver (Batch Renew: {params['n_batch_renew']}, Gamma Renew: {params['n_gamma_renew']})")
        return StandardWassersteinSolver(
            n_batch_renew=params.get("n_batch_renew", 1000),
            n_gamma_renew=params.get("n_gamma_renew", 500)
        )

    # 2. Sliced 系列 (需要 n_projections)
    elif dist_type in ["sliced", "max", "spherical", "sphere","max_optimized"]:
        # 兼容一下 typo "shpere" 或 "sphere"
        clean_type = "spherical" if dist_type in ["sphere", "shpere", "spherical"] else dist_type

        print(f"Initializing {clean_type.capitalize()} Wasserstein Solver (Projections: {params['n_projections']})")
        return SlicedWassersteinSolver(
            n_projections=params.get("n_projections", 1000),
            p=2,
            type=clean_type
        )

    else:
        raise ValueError(f"Unknown distance type: {dist_type}")