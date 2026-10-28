"""Particle-cloud models for Stage I (MeanFlow) and Stage II (DP corrector)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CompactSinusoidalEmbedding(nn.Module):
    """Paper-style scalar embedding: the value plus a few Fourier features.

    The previous 64-dim DDPM embedding (max_period=10000) is almost constant
    on log2(sigma) in [-4.5, -2] and on (r, t) in [0, 1]. The KS example and
    the paper's compact MLP use a short feature: [x, sin(2^k π x), cos(...)].
    """

    def __init__(self, n_frequencies: int = 4):
        super().__init__()
        self.register_buffer(
            "frequencies",
            (2.0 ** torch.arange(n_frequencies, dtype=torch.float32)) * math.pi,
        )

    @property
    def output_dim(self) -> int:
        return 1 + 2 * int(self.frequencies.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().unsqueeze(-1)
        phases = x * self.frequencies
        return torch.cat([x, phases.sin(), phases.cos()], dim=-1)


class LightSkipSELU(nn.Module):
    """Light skip used by the paper: SELU(Wx) + skip(x)."""

    def __init__(self, in_dim: int, width: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, width)
        self.skip = nn.Identity() if in_dim == width else nn.Linear(in_dim, width, bias=False)
        self.act = nn.SELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x)) + self.skip(x)


def torus_features(z: torch.Tensor) -> torch.Tensor:
    """Lift torus coordinates to [z, sin z, cos z] so the MLP sees periodicity."""
    return torch.cat([z, torch.sin(z), torch.cos(z)], dim=-1)


class MFMLP(nn.Module):
    """MeanFlow transport predictor u_theta(z, r, t, sigma).

    Width 64, depth 5, SELU, light skips, compact sinusoidal embeddings of
    (r, t, sigma), small-gain head. 2D KPP also feeds torus Fourier features
    of z; the head still predicts a Euclidean velocity in R^d.
    """

    def __init__(
        self,
        dim: int,
        width: int = 64,
        depth: int = 5,
        n_frequencies: int = 4,
        torus_features_on: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.width = width
        self.torus_features_on = torus_features_on
        self.embed_r = CompactSinusoidalEmbedding(n_frequencies)
        self.embed_t = CompactSinusoidalEmbedding(n_frequencies)
        self.embed_sigma = CompactSinusoidalEmbedding(n_frequencies)
        z_dim = 3 * dim if torus_features_on else dim
        in_dim = z_dim + 3 * self.embed_r.output_dim
        self.blocks = nn.ModuleList(
            [
                LightSkipSELU(in_dim if i == 0 else width, width)
                for i in range(depth)
            ]
        )
        self.head = nn.Linear(width, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _forward_scalars(
        self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        z_in = torus_features(z) if self.torus_features_on else z
        c = torch.cat([self.embed_r(r), self.embed_t(t), self.embed_sigma(sigma)], dim=-1)
        h = torch.cat([z_in, c], dim=-1)
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

        The tangent is (v, 0 for r, 1 for t, 0 for sigma), matching
        u_dot = partial_t u + (grad_z u) v in the paper.
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
    """Single-linear SiLU + LayerNorm residual block (paper / KS example)."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.linear(x)))


class DPResNet(nn.Module):
    """Near-identity corrector f(x, sigma) = x + alpha * h([x, sigma]).

    KPP §4.3 uses a 6-block residual corrector on the torus. The last layer
    is zero-initialized so the map starts as the identity. Output wrapping
    is off during training: the torus metric already sits in the OT cost.
    """

    def __init__(
        self,
        dim: int,
        width: int = 64,
        depth: int = 6,
        alpha: float = 0.1,
        wrap_output: bool = False,
        n_frequencies: int = 4,
        torus_features_on: bool = True,
        displacement_cap: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.wrap_output = wrap_output
        self.torus_features_on = torus_features_on
        self.displacement_cap = displacement_cap
        self.embed_sigma = CompactSinusoidalEmbedding(n_frequencies)
        self.cond = nn.Sequential(
            nn.Linear(self.embed_sigma.output_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        x_dim = 3 * dim if torus_features_on else dim
        self.in_proj = nn.Sequential(
            nn.Linear(x_dim + width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([DPResidualBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_in = torus_features(x) if self.torus_features_on else x
        s = self.cond(self.embed_sigma(sigma))
        h = self.in_proj(torch.cat([x_in, s], dim=-1))
        for block in self.blocks:
            h = block(h)
        out = x + self._displacement(h)
        if self.wrap_output:
            out = torch.remainder(out, 2.0 * math.pi)
        return out

    def _displacement(self, h: torch.Tensor) -> torch.Tensor:
        """Turn the head output into a bounded correction.

        ``alpha`` alone does not make the map near-identity: it multiplies the
        output but the linear head is free to grow, and in practice it does
        (measured mean displacement 1.02 rad against a required ~0.05). The
        corrector then searches globally instead of locally, which is why DP
        could not beat the identity map.

        ``displacement_cap`` bounds the correction magnitude with a smooth
        tanh, so ``|f(x) - x| < cap`` for every input while gradients stay
        non-zero everywhere. ``None`` reproduces the historical behaviour.
        """
        raw = self.alpha * self.head(h)
        cap = self.displacement_cap
        if cap is None:
            return raw
        norm = raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.tanh(norm / cap) * cap * (raw / norm)
