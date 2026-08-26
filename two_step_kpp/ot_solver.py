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
