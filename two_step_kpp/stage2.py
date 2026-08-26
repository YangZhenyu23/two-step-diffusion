"""Stage II: DP refinement with mini-batch EMD coupling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .config import Config, Stage2Config
from .data_io import TWO_PI, build_stage1_arrays, wrap_torus
from .models import DPResNet, MFMLP
from .ot_solver import EMDSolver


def torch_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, TWO_PI)


def train_stage2(
    cfg: Config,
    stage1_model: MFMLP,
    dp_model: DPResNet,
    run_dir: Path,
    levels: Optional[Sequence[float]] = None,
) -> DPResNet:
    """Train the near-identity DP corrector using Algorithm 2."""
    s2: Stage2Config = cfg.stage2
    device = torch.device(s2.device)
    if levels is None:
        levels = cfg.train_levels

    eps, x_ref, sigma = build_stage1_arrays(
        exp=cfg.exp,
        levels=levels,
        n_samples=20000,
        prior_std=cfg.stage1.prior_std,
        seed=s2.seed,
    )
    eps_t = torch.from_numpy(eps)
    ref_t = torch.from_numpy(x_ref)
    sigma_t = torch.from_numpy(sigma)

    stage1_model = stage1_model.to(device).eval()
    dp_model = dp_model.to(device)
    optimizer = torch.optim.AdamW(
        dp_model.parameters(), lr=s2.lr, weight_decay=s2.weight_decay
    )
    solver = EMDSolver(gamma_refresh=s2.gamma_refresh)

    total = eps_t.shape[0]
    generator = torch.Generator().manual_seed(s2.seed)
    current_idx = None

    for step in range(s2.steps):
        if current_idx is None or step % s2.batch_refresh == 0:
            if current_idx is not None:
                solver.reset()
            current_idx = torch.randperm(total, generator=generator)[: s2.batch_size]

        eps_b = eps_t[current_idx].to(device)
        ref_b = ref_t[current_idx].to(device)
        sigma_b = sigma_t[current_idx].to(device)

        with torch.no_grad():
            r0 = torch.zeros_like(sigma_b)
            t1 = torch.ones_like(sigma_b)
            u = stage1_model(eps_b, r0, t1, sigma_b)
            x_mf = torch_wrap(eps_b - u)

        x_dp = dp_model(x_mf, sigma_b)
        ot_loss = solver(x_dp, ref_b)
        residual = torch.mean((x_dp - x_mf) ** 2)
        loss = ot_loss + s2.lambda_res * residual

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % s2.ckpt_every == 0 or step + 1 == s2.steps:
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dp_model.state_dict(), run_dir / "stage2_last.pt")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dp_model.state_dict(), run_dir / "stage2_final.pt")
    return dp_model


def generate_dp(
    stage1_model: MFMLP,
    dp_model: DPResNet,
    eps: np.ndarray,
    sigma: float,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """One-step deterministic sampling followed by the DP correction."""
    stage1_model.eval()
    dp_model.eval()
    eps_arr = np.asarray(eps, dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, eps_arr.shape[0], batch_size):
            e = torch.from_numpy(eps_arr[i : i + batch_size]).to(device)
            s = torch.full((e.shape[0],), sigma, dtype=torch.float32, device=device)
            r0 = torch.zeros_like(s)
            t1 = torch.ones_like(s)
            x_mf = torch_wrap(e - stage1_model(e, r0, t1, s))
            x_dp = dp_model(x_mf, s)
            out.append(torch_wrap(x_dp).cpu().numpy())
    return wrap_torus(np.concatenate(out, axis=0))
