"""Mini-batch EMD solver with stateful coupling reuse for Stage II."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .data_io import TWO_PI


def periodic_l2_sq_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared shortest-path distance on the torus.

    Args:
        a: (N, d)
        b: (M, d)
    Returns:
        cost: (N, M)
    """
    diff = b.unsqueeze(0) - a.unsqueeze(1)
    diff = diff - TWO_PI * torch.round(diff / TWO_PI)
    return (diff * diff).sum(dim=-1)


def soft_hist2d(
    xy: torch.Tensor,
    bins: int = 72,
    blur: float = 0.8,
    dims: tuple = (0, 1),
) -> torch.Tensor:
    """Differentiable periodic 2D histogram, rows/cols sum to one."""
    xs = torch.remainder(xy[:, dims[0]], TWO_PI) / TWO_PI * bins
    ys = torch.remainder(xy[:, dims[1]], TWO_PI) / TWO_PI * bins
    grid = torch.arange(bins, device=xy.device, dtype=xy.dtype) + 0.5

    def weights(coord: torch.Tensor) -> torch.Tensor:
        delta = coord.unsqueeze(1) - grid.unsqueeze(0)
        delta = delta - bins * torch.round(delta / bins)
        w = torch.exp(-0.5 * (delta / blur) ** 2)
        return w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

    hist = torch.einsum("ni,nj->ij", weights(xs), weights(ys))
    return hist / hist.sum().clamp_min(1e-8)


def hard_hist2d(xy: torch.Tensor, bins: int = 72, dims: tuple = (0, 1)) -> torch.Tensor:
    """Detached count histogram on the torus, normalized to a probability."""
    xs = torch.remainder(xy[:, dims[0]], TWO_PI) / TWO_PI * bins
    ys = torch.remainder(xy[:, dims[1]], TWO_PI) / TWO_PI * bins
    idx = torch.stack([xs.long().clamp(0, bins - 1), ys.long().clamp(0, bins - 1)], dim=1)
    flat = idx[:, 0] * bins + idx[:, 1]
    hist = torch.bincount(flat, minlength=bins * bins).to(dtype=xy.dtype).reshape(bins, bins)
    return hist / hist.sum().clamp_min(1e-8)


def torus_grid_cost(bins: int, device: torch.device) -> torch.Tensor:
    """Pairwise squared torus distance between centers of a bins x bins grid."""
    centers = (torch.arange(bins, device=device, dtype=torch.float32) + 0.5) / bins * TWO_PI
    xx, yy = torch.meshgrid(centers, centers, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    return periodic_l2_sq_torch(pts, pts)


def sinkhorn_w2(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 30,
) -> torch.Tensor:
    """Differentiable entropic W2^2 between two histograms on a fixed cost."""
    a = a.reshape(-1).clamp_min(1e-12)
    b = b.reshape(-1).clamp_min(1e-12)
    a = a / a.sum()
    b = b / b.sum()
    kernel = torch.exp(-cost / eps)
    u = torch.ones_like(a)
    for _ in range(n_iter):
        u = a / (kernel @ (b / (kernel.t() @ u).clamp_min(1e-12))).clamp_min(1e-12)
    v = b / (kernel.t() @ u).clamp_min(1e-12)
    plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    return (plan * cost).sum()


def periodic_rbf_mmd(
    x: torch.Tensor, y: torch.Tensor, lengthscale: float = 0.40
) -> torch.Tensor:
    """Biased RBF MMD on the torus. ``y`` should be detached."""
    scale = 2.0 * lengthscale * lengthscale

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = b.unsqueeze(0) - a.unsqueeze(1)
        diff = diff - TWO_PI * torch.round(diff / TWO_PI)
        return torch.exp(-(diff * diff).sum(dim=-1) / scale)

    return rbf(x, x).mean() + rbf(y, y).mean() - 2.0 * rbf(x, y).mean()


def periodic_l2_sq_vector(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise squared periodic distance for aligned point clouds."""
    diff = b - a
    diff = diff - TWO_PI * torch.round(diff / TWO_PI)
    return (diff * diff).sum(dim=-1)


class EMDSolver:
    """Stateful exact EMD coupling for a single particle mini-batch.

    The coupling is held fixed between refreshes so gradients flow only through
    the cost matrix, matching Algorithm 2 in the paper.
    """

    def __init__(self, gamma_refresh: int = 50):
        self.gamma_refresh = gamma_refresh
        self.gamma: Optional[torch.Tensor] = None
        self.counter = 0

    def reset(self) -> None:
        """Drop the cached coupling so it is recomputed on the next call."""
        self.gamma = None
        self.counter = 0

    def _solve_coupling(self, x_cpu: np.ndarray, y_cpu: np.ndarray) -> np.ndarray:
        import ot  # imported lazily so the package can be imported without POT

        n = x_cpu.shape[0]
        cost = periodic_l2_sq_torch(
            torch.from_numpy(x_cpu), torch.from_numpy(y_cpu)
        ).numpy()
        a = np.ones(n, dtype=np.float64) / n
        b = np.ones(n, dtype=np.float64) / n
        return ot.emd(a, b, cost, numItermax=1_000_000)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the differentiable OT loss for a single pair of point clouds."""
        n = x.shape[0]
        if n != y.shape[0]:
            raise ValueError("EMDSolver requires x and y to have the same batch size")

        if self.gamma is None or self.counter % self.gamma_refresh == 0:
            x_np = x.detach().cpu().numpy().astype(np.float64)
            y_np = y.detach().cpu().numpy().astype(np.float64)
            gamma_np = self._solve_coupling(x_np, y_np)
            self.gamma = torch.from_numpy(gamma_np.astype(np.float32)).to(x.device)

        cost = periodic_l2_sq_torch(x, y)
        self.counter += 1
        return torch.sum(cost * self.gamma)


class MiniBatchEMDSolver:
    """Frequent mini-batch EMD used by the original DeepParticle algorithm.

    Each call samples a small subset of source/target particles, solves the
    corresponding small exact EMD, and returns its transport cost.  This gives
    a fresh, local OT signal at every DP update instead of holding a stale full
    coupling for many steps.
    """

    def __init__(self, subsample: int = 200, seed: int = 0):
        self.subsample = subsample
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        import ot

        n = x.shape[0]
        m = min(self.subsample, n)
        idx = torch.randperm(n, generator=self.generator)[:m].to(x.device)
        cost_torch = periodic_l2_sq_torch(x[idx], y[idx].detach())
        cost_np = cost_torch.detach().cpu().numpy().astype(np.float64)
        a = np.ones(m, dtype=np.float64) / m
        b = np.ones(m, dtype=np.float64) / m
        gamma = ot.emd(a, b, cost_np, numItermax=1_000_000)
        gamma_torch = torch.from_numpy(gamma.astype(np.float32)).to(x.device)
        return torch.sum(cost_torch * gamma_torch)
