"""Central configuration for the 2D KPP package.

2D-specific choices (do not copy blindly to 3D):

- Stage I uses the paper's compact light-skip SELU MLP, torus Fourier
  features of z, OT pairing on the torus, geodesic interpolation, cosine
  decay from 1e-3, and a fresh OT pairing every ``pair_refresh`` steps.
- Adaptive MSE stayed off: it raised W2 when |v| was large (independent
  pairing). Revisit only after the new backbone is measured.
- Stage I is frozen after ``full_2d_v2``. Do not change Stage I defaults
  to chase DP.
- Stage II: OT is a probability coupling so ⟨C,γ⟩ already equals W2².
  A sum residual with λ=1e-3 then weighs 1.5·mean(||d||²) and freezes
  the map at the identity. Use a hinge residual that is free inside a
  local ball and only penalizes collapse-scale jumps. Sample FK
  references independently of the Meanflow particles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Stage1Config:
    batch_size: int = 4096
    steps: int = 20000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    ckpt_every: int = 2000
    pair_refresh: int = 5000
    prior_std: float = 3.141592653589793
    seed: int = 0
    time_mu: float = -0.4
    time_sigma: float = 1.0
    r_equals_t_ratio: float = 0.5
    pair_mode: str = "ot_periodic"
    geodesic_path: bool = True
    device: str = "cuda"
    # Weight on the explicit endpoint term ||u(eps,0,1,sigma) - (eps - x0)||^2.
    # 0.0 reproduces the historical runs exactly.
    endpoint_weight: float = 0.0


@dataclass
class Stage2Config:
    batch_size: int = 1500
    steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 0.0
    gamma_refresh: int = 50
    batch_refresh: int = 400
    residual_weight: float = 0.25
    residual_reduction: str = "mean"
    residual_free_radius: float = 0.12
    ot_weight: float = 0.15
    hist_weight: float = 1.0
    hist_bins: int = 36
    hist_blur: float = 0.6
    hist_dims: tuple = (0, 1)
    sinkhorn_eps: float = 0.05
    sinkhorn_iters: int = 30
    mmd_weight: float = 0.0
    mmd_samples: int = 1024
    mmd_lengthscale: float = 0.35
    alpha: float = 0.2
    depth: int = 6
    wrap_output: bool = False
    # Hard bound on |f(x)-x|. None keeps the historical unbounded head.
    displacement_cap: Optional[float] = None
    ckpt_every: int = 2000
    probe_every: int = 2000
    probe_samples: int = 4000
    seed: int = 0
    device: str = "cuda"


@dataclass
class EvalConfig:
    bins: int = 72
    n_samples: int = 20000
    seed: int = 666
    device: str = "cuda"
    cT_sigma: float = -4.5
    cT_generations: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )


@dataclass
class Config:
    exp: str = "kpp2d"
    train_levels: List[float] = field(
        default_factory=lambda: [-2.0 - 0.25 * k for k in range(8)]
    )
    eval_levels: List[float] = field(
        default_factory=lambda: [-2.0 - 0.25 * k for k in range(11)]
    )
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    evalt: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "run"

    def to_dict(self) -> Dict[str, object]:
        return {
            "exp": self.exp,
            "train_levels": self.train_levels,
            "eval_levels": self.eval_levels,
            "stage1": self.stage1.__dict__,
            "stage2": self.stage2.__dict__,
            "eval": self.evalt.__dict__,
            "run_name": self.run_name,
        }


def smoke_config(exp: str) -> Config:
    """Small config used by smoke tests and end-to-end sanity checks."""
    cfg = Config(exp=exp)
    cfg.stage1.batch_size = 128
    cfg.stage1.steps = 20
    cfg.stage1.ckpt_every = 10
    cfg.stage1.pair_refresh = 0
    cfg.stage1.device = "cpu"
    cfg.stage2.batch_size = 64
    cfg.stage2.steps = 20
    cfg.stage2.gamma_refresh = 5
    cfg.stage2.batch_refresh = 10
    cfg.stage2.ckpt_every = 10
    cfg.stage2.probe_every = 0
    cfg.stage2.device = "cpu"
    cfg.evalt.n_samples = 256
    cfg.evalt.device = "cpu"
    cfg.evalt.cT_generations = [1, 2, 4]
    cfg.run_name = "smoke"
    return cfg
