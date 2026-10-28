"""Stage-I MeanFlow objectives and one-NFE sampling."""

import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, n_frequencies: int = 4):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(n_frequencies) * torch.pi)

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.frequencies.numel()

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim == 0 or value.shape[-1] != 1:
            raise ValueError(f"Expected a final singleton feature dimension, got {tuple(value.shape)}.")
        phases = value * self.frequencies
        return torch.cat((value, phases.sin(), phases.cos()), dim=-1)


class MeanflowBlock(nn.Module):
    """Light skip connection used by the paper's compact Meanflow MLP."""
    def __init__(self, in_dim: int, width: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, width)
        self.skip = nn.Identity() if in_dim == width else nn.Linear(in_dim, width, bias=False)
        self.activation = nn.SELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x)) + self.skip(x)


class MeanflowNet(nn.Module):
    """Average-velocity predictor u_theta(z, r, t, sigma)."""
    def __init__(self, state_dim: int, condition_dim: int, width: int = 64, depth: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.embedding = SinusoidalEmbedding()
        width_in = state_dim + self.embedding.output_dim * (2 + condition_dim)
        self.blocks = nn.ModuleList(MeanflowBlock(width_in if index == 0 else width, width) for index in range(depth))
        self.head = nn.Linear(width, state_dim)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, z: Tensor, r: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        if r.shape != z.shape[:-1] + (1,) or t.shape != r.shape:
            raise ValueError("r and t must have shape z.shape[:-1] + (1,).")
        if condition.shape[:-1] != z.shape[:-1] or condition.shape[-1] != self.condition_dim:
            raise ValueError("condition must match z's batch dimensions and configured condition_dim.")
        features = [z, self.embedding(r), self.embedding(t)]
        features.extend(self.embedding(condition[..., index:index + 1]) for index in range(condition.shape[-1]))
        output = torch.cat(features, dim=-1)
        for block in self.blocks:
            output = block(output)
        return self.head(output)


def meanflow_jvp(model: MeanflowNet, z: Tensor, r: Tensor, t: Tensor, condition: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
    """Return u and its total derivative along dz/dt=velocity (Algorithm 1)."""
    return torch.autograd.functional.jvp(
        lambda z_, r_, t_, c_: model(z_, r_, t_, c_), (z, r, t, condition),
        (velocity, torch.zeros_like(r), torch.ones_like(t), torch.zeros_like(condition)), create_graph=True,
    )


def adaptive_mean_squared_error(
    error: Tensor,
    power: float = 0.25,
    epsilon: float = 1e-3,
    weight_min: float = 0.25,
    weight_max: float = 4.0,
    sample_mask: Tensor | None = None,
) -> Tensor:
    """Normalized, clipped adaptive per-particle MSE.

    Detaching the weights prevents the model from reducing the loss by changing
    the weighting rule. Normalizing within the selected group keeps the gradient
    scale comparable to ordinary MSE, while clipping prevents either hard or
    easy particles from dominating. ``power=0`` recovers ordinary MSE.
    """
    if not 0.0 <= power <= 1.0:
        raise ValueError("Adaptive-loss power must lie in [0, 1].")
    if epsilon <= 0:
        raise ValueError("Adaptive-loss epsilon must be positive.")
    if weight_min <= 0 or weight_max < weight_min:
        raise ValueError("Adaptive-loss weight bounds must satisfy 0 < min <= max.")
    per_particle_mse = error.square().mean(dim=-1)
    if sample_mask is not None:
        if sample_mask.shape == per_particle_mse.shape + (1,):
            sample_mask = sample_mask.squeeze(-1)
        if sample_mask.shape != per_particle_mse.shape:
            raise ValueError("sample_mask must match error.shape[:-1].")
        per_particle_mse = per_particle_mse[sample_mask.bool()]
    if per_particle_mse.numel() == 0:
        raise ValueError("Adaptive loss received an empty sample group.")
    weight = (per_particle_mse.detach() + epsilon).pow(-power)
    weight = (weight / weight.mean().clamp_min(torch.finfo(weight.dtype).eps)).clamp(weight_min, weight_max)
    return (weight * per_particle_mse).sum() / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)


def _meanflow_loss_details(
    error: Tensor,
    same_time: Tensor | None,
    adaptive_power: float,
    adaptive_epsilon: float,
    adaptive_weight_min: float,
    adaptive_weight_max: float,
) -> dict[str, Tensor]:
    """Return balanced adaptive loss and raw MSE diagnostics."""
    per_particle_mse = error.square().mean(dim=-1)
    adaptive_all = adaptive_mean_squared_error(
        error,
        adaptive_power,
        adaptive_epsilon,
        adaptive_weight_min,
        adaptive_weight_max,
    )
    details = {
        "adaptive_loss": adaptive_all,
        "raw_mse": per_particle_mse.mean(),
        "raw_mse_same_time": per_particle_mse.new_tensor(float("nan")),
        "raw_mse_different_time": per_particle_mse.new_tensor(float("nan")),
    }
    if same_time is None:
        return details
    if same_time.shape == per_particle_mse.shape + (1,):
        same_time = same_time.squeeze(-1)
    if same_time.shape != per_particle_mse.shape:
        raise ValueError("same_time must match error.shape[:-1].")
    diagonal = same_time.bool()
    off_diagonal = ~diagonal
    if diagonal.any():
        details["raw_mse_same_time"] = per_particle_mse[diagonal].mean()
    if off_diagonal.any():
        details["raw_mse_different_time"] = per_particle_mse[off_diagonal].mean()
    if diagonal.any() and off_diagonal.any():
        diagonal_loss = adaptive_mean_squared_error(
            error,
            adaptive_power,
            adaptive_epsilon,
            adaptive_weight_min,
            adaptive_weight_max,
            diagonal,
        )
        off_diagonal_loss = adaptive_mean_squared_error(
            error,
            adaptive_power,
            adaptive_epsilon,
            adaptive_weight_min,
            adaptive_weight_max,
            off_diagonal,
        )
        details["adaptive_loss"] = 0.5 * (diagonal_loss + off_diagonal_loss)
    return details


def ordinary_meanflow_loss(
    model: MeanflowNet,
    z: Tensor,
    r: Tensor,
    t: Tensor,
    condition: Tensor,
    velocity: Tensor,
    adaptive_power: float = 0.25,
    adaptive_epsilon: float = 1e-3,
    same_time: Tensor | None = None,
    adaptive_weight_min: float = 0.25,
    adaptive_weight_max: float = 4.0,
    return_details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Algorithm-1 MeanFlow identity loss with a stopped JVP target."""
    average_velocity, derivative = meanflow_jvp(model, z, r, t, condition, velocity)
    target = (velocity - (t - r) * derivative).detach()
    details = _meanflow_loss_details(
        average_velocity - target,
        same_time,
        adaptive_power,
        adaptive_epsilon,
        adaptive_weight_min,
        adaptive_weight_max,
    )
    return (details["adaptive_loss"], details) if return_details else details["adaptive_loss"]


def improved_meanflow_loss(
    model: MeanflowNet,
    z: Tensor,
    r: Tensor,
    t: Tensor,
    condition: Tensor,
    velocity: Tensor,
    adaptive_power: float = 0.25,
    adaptive_epsilon: float = 1e-3,
    same_time: Tensor | None = None,
    adaptive_weight_min: float = 0.25,
    adaptive_weight_max: float = 4.0,
    return_details: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Improved MeanFlow's v-loss using the boundary-condition v predictor.

    The marginal instantaneous velocity is approximated without an auxiliary
    head as ``v_theta(z,t) = u_theta(z,t,t)``.  Its detached value is used as
    the JVP tangent.  The compound prediction

        V_theta = u_theta + (t-r) stopgrad(d u_theta / dt)

    is regressed onto the paired conditional velocity.
    """
    predicted_velocity = model(z, t, t, condition).detach()
    average_velocity, derivative = meanflow_jvp(model, z, r, t, condition, predicted_velocity)
    compound_velocity = average_velocity + (t - r) * derivative.detach()
    details = _meanflow_loss_details(
        compound_velocity - velocity,
        same_time,
        adaptive_power,
        adaptive_epsilon,
        adaptive_weight_min,
        adaptive_weight_max,
    )
    return (details["adaptive_loss"], details) if return_details else details["adaptive_loss"]


def meanflow_endpoint(model: MeanflowNet, source: Tensor, condition: Tensor) -> Tensor:
    """One-NFE endpoint map x_target = x_source - u(x_source, 0, 1)."""
    zeros = torch.zeros_like(source[..., :1])
    return source - model(source, zeros, torch.ones_like(zeros), condition)


@torch.no_grad()
def sample_meanflow(model: MeanflowNet, source: Tensor, condition: Tensor) -> Tensor:
    """Inference-only one-NFE endpoint map."""
    return meanflow_endpoint(model, source, condition)
