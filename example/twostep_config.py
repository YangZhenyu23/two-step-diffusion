"""Configuration for the Keller--Segel Two-Step Diffusion model."""

from dataclasses import dataclass

import torch


@dataclass
class TwoStepConfig:
    """Standard or Improved MeanFlow followed by frozen-MF DP refinement.

    Network widths/depths and ``gamma_refresh=50`` follow the Two-Step paper.
    ``batch_refresh=1000`` is inherited from DeepParticle 2. Learning rates,
    step counts, alpha, and lambda_res are explicit implementation choices
    because the paper does not specify all of them. Stage I uses normalized,
    clipped adaptive weights and a cosine decay with a conservative learning-
    rate floor. Stage II keeps its learning rate constant because each batch
    refresh presents a new empirical OT problem.
    """

    state_dim: int
    condition_dim: int = 1
    mf_width: int = 64
    mf_depth: int = 5
    dp_width: int = 64
    dp_depth: int = 4
    near_identity_scale: float = 0.1
    mf_lr: float = 1e-3
    dp_lr: float = 1e-3
    mf_steps: int = 20_000
    dp_steps: int = 20_000
    batch_refresh: int = 1_000
    gamma_refresh: int = 50
    residual_weight: float = 1e-3
    time_logit_mean: float = -0.4
    time_logit_std: float = 1.0
    improved_meanflow: bool = False
    mf_adaptive_power: float = 0.25
    mf_adaptive_epsilon: float = 1e-3
    mf_adaptive_weight_min: float = 0.25
    mf_adaptive_weight_max: float = 4.0
    mf_min_lr_ratio: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        positive_integers = {
            "state_dim": self.state_dim,
            "condition_dim": self.condition_dim,
            "mf_width": self.mf_width,
            "mf_depth": self.mf_depth,
            "dp_width": self.dp_width,
            "dp_depth": self.dp_depth,
            "mf_steps": self.mf_steps,
            "dp_steps": self.dp_steps,
            "batch_refresh": self.batch_refresh,
            "gamma_refresh": self.gamma_refresh,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        for name, value in {
            "near_identity_scale": self.near_identity_scale,
            "mf_lr": self.mf_lr,
            "dp_lr": self.dp_lr,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.residual_weight < 0:
            raise ValueError("residual_weight must be non-negative.")
        if not isinstance(self.improved_meanflow, bool):
            raise ValueError("improved_meanflow must be boolean.")
        if not 0.0 <= self.mf_adaptive_power <= 1.0:
            raise ValueError("mf_adaptive_power must lie in [0, 1].")
        if self.mf_adaptive_epsilon <= 0:
            raise ValueError("mf_adaptive_epsilon must be positive.")
        if self.mf_adaptive_weight_min <= 0:
            raise ValueError("mf_adaptive_weight_min must be positive.")
        if self.mf_adaptive_weight_max < self.mf_adaptive_weight_min:
            raise ValueError("mf_adaptive_weight_max must be at least mf_adaptive_weight_min.")
        if not 0.0 < self.mf_min_lr_ratio <= 1.0:
            raise ValueError("mf_min_lr_ratio must lie in (0, 1].")
