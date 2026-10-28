"""Stage II: near-identity DeepParticle corrector."""

import torch
from torch import Tensor, nn


class DPResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.activation(self.norm(self.linear(x)))


class NearIdentityDP(nn.Module):
    """f_phi(x, sigma) = x + alpha h_phi([x, sigma]).

    ``depth`` is the number of residual blocks after the input projection.
    The final layer is zero-initialized, so the complete map starts as identity.
    """
    def __init__(self, state_dim: int, condition_dim: int, width: int = 64, depth: int = 4, alpha: float = 0.1):
        super().__init__()
        if state_dim <= 0 or condition_dim <= 0 or width <= 0 or depth <= 0 or alpha <= 0:
            raise ValueError("state_dim, condition_dim, width, depth, and alpha must be positive.")
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.alpha = alpha
        self.input = nn.Sequential(nn.Linear(state_dim + condition_dim, width), nn.LayerNorm(width), nn.SiLU())
        self.blocks = nn.ModuleList(DPResidualBlock(width) for _ in range(depth))
        self.head = nn.Linear(width, state_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        if x.shape[:-1] != condition.shape[:-1]:
            raise ValueError("x and condition must have identical batch and particle dimensions.")
        if x.shape[-1] != self.state_dim or condition.shape[-1] != self.condition_dim:
            raise ValueError("x or condition has an unexpected feature dimension.")
        residual = self.input(torch.cat((x, condition), dim=-1))
        for block in self.blocks:
            residual = block(residual)
        return x + self.alpha * self.head(residual)
