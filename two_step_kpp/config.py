"""Central configuration for two_step_kpp runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Stage1Config:
    batch_size: int = 4096
    steps: int = 20000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    ckpt_every: int = 2000
    prior_std: float = 3.141592653589793
    seed: int = 0
    time_mu: float = -0.4
    time_sigma: float = 1.0
    r_equals_t_ratio: float = 0.5
    device: str = "cuda"


@dataclass
class Stage2Config:
    batch_size: int = 1500
    steps: int = 10000
    lr: float = 1e-3
    weight_decay: float = 0.0
    gamma_refresh: int = 50
    batch_refresh: int = 100
    lambda_res: float = 0.01
    alpha: float = 0.1
    ckpt_every: int = 2000
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
    cfg.stage1.device = "cpu"
    cfg.stage2.batch_size = 64
    cfg.stage2.steps = 20
    cfg.stage2.gamma_refresh = 5
    cfg.stage2.batch_refresh = 10
    cfg.stage2.ckpt_every = 10
    cfg.stage2.device = "cpu"
    cfg.evalt.n_samples = 256
    cfg.evalt.device = "cpu"
    cfg.evalt.cT_generations = [1, 2, 4]
    cfg.run_name = "smoke"
    return cfg
