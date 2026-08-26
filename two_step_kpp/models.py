"""Particle-cloud models for Stage I (MeanFlow) and Stage II (DP corrector)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for scalar conditioning values."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=x.device)
            / half
        )
        args = x.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class ResidualLinear(nn.Module):
    """A small SELU residual block with a light skip connection."""

    def __init__(self, width: int):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.act = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear1(x))
        h = self.linear2(h)
        return x + h


class MFMLP(nn.Module):
    """MeanFlow transport predictor u_theta(z, r, t, sigma).

    Architecture follows the paper: width 64, depth 5, SELU, light skip
    connections, sinusoidal embeddings for (r, t, sigma), and a small-gain
    linear head.
    """

    def __init__(self, dim: int, width: int = 64, depth: int = 5, embed_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.width = width
        self.embed_r = SinusoidalEmbedding(embed_dim)
        self.embed_t = SinusoidalEmbedding(embed_dim)
        self.embed_sigma = SinusoidalEmbedding(embed_dim)
        self.cond = nn.Sequential(
            nn.Linear(embed_dim * 3, width),
            nn.SELU(),
            nn.Linear(width, width),
        )
        self.in_proj = nn.Linear(dim, width)
        self.blocks = nn.ModuleList([ResidualLinear(width) for _ in range(depth)])
        self.head = nn.Linear(width, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small-gain output head so early predictions stay close to zero.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _forward_scalars(
        self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        c = torch.cat(
            [self.embed_r(r), self.embed_t(t), self.embed_sigma(sigma)], dim=-1
        )
        h = self.in_proj(z) + self.cond(c)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def forward(
        self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_scalars(z, r, t, sigma)

    def jvp_u(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        sigma: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (u, u_dot) where u_dot = d_t u + (grad_z u) v.

        The tangent vector is (v, 0 for r, 1 for t, 0 for sigma).
        """

        def fn(zz, rr, tt, ss):
            return self._forward_scalars(zz, rr, tt, ss)

        inputs = (z, r, t, sigma)
        tangents = (
            v,
            torch.zeros_like(r),
            torch.ones_like(t),
            torch.zeros_like(sigma),
        )
        u, u_dot = torch.autograd.functional.jvp(
            fn, inputs, tangents, create_graph=True
        )
        return u, u_dot


class DPResidualBlock(nn.Module):
    """SiLU + LayerNorm residual block used by the near-identity DP corrector."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.linear1(self.norm(x)))
        h = self.linear2(h)
        return x + h


class DPResNet(nn.Module):
    """Near-identity corrector f(x, sigma) = x + alpha * h([x, sigma]).

    Depth 4, width 64, SiLU activations and LayerNorm, with a zero-initialized
    last layer so the initial map is exactly the identity.
    """

    def __init__(self, dim: int, width: int = 64, depth: int = 4, alpha: float = 0.1):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.embed_sigma = SinusoidalEmbedding(width)
        self.cond = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.in_proj = nn.Linear(dim + width, width)
        self.blocks = nn.ModuleList([DPResidualBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Zero the last layer: at initialization f(x, sigma) = x exactly.
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = self.cond(self.embed_sigma(sigma))
        h = self.in_proj(torch.cat([x, s], dim=-1))
        for block in self.blocks:
            h = block(h)
        return x + self.alpha * self.head(h)
