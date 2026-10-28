"""Stage I: MeanFlow-style training for the KPP particle cloud."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config, Stage1Config
from .data_io import TWO_PI, build_stage1_arrays, wrap_torus
from .models import MFMLP
from .plot_kpp import plot_loss_curve


class Stage1Dataset(Dataset):
    """Torch dataset over (epsilon, x_ref, sigma) triplets."""

    def __init__(self, eps: np.ndarray, x_ref: np.ndarray, sigma: np.ndarray):
        self.eps = torch.from_numpy(np.asarray(eps, dtype=np.float32))
        self.x_ref = torch.from_numpy(np.asarray(x_ref, dtype=np.float32))
        self.sigma = torch.from_numpy(np.asarray(sigma, dtype=np.float32))

    def __len__(self) -> int:
        return self.eps.shape[0]

    def __getitem__(self, idx: int):
        return self.eps[idx], self.x_ref[idx], self.sigma[idx]


def logit_normal_t_r(
    batch_size: int,
    device: torch.device,
    mu: float = -0.4,
    sigma: float = 1.0,
    r_equals_t_ratio: float = 0.5,
):
    """Sample 0 <= r <= t <= 1 from a logit-normal distribution."""
    normal = torch.randn(batch_size, 2, device=device) * sigma + mu
    samples = torch.sigmoid(normal)
    t = samples.max(dim=1).values
    r = samples.min(dim=1).values
    if r_equals_t_ratio > 0:
        mask = torch.rand(batch_size, device=device) < r_equals_t_ratio
        r = torch.where(mask, t, r)
    return t, r


def adaptive_mean_squared_error(
    error: torch.Tensor,
    power: float = 0.25,
    epsilon: float = 1e-3,
    weight_min: float = 0.25,
    weight_max: float = 4.0,
) -> torch.Tensor:
    """Normalized, clipped adaptive per-particle MSE from the KS example.

    Leftover from a failed Stage I attempt: adaptive MSE + cosine scheduler
    raised 2D MeanFlow W2 and was rolled back. `train_stage1` uses ordinary
    `mean((u - u_tgt)**2)` and does not call this function. Do not wire it
    back in without a new ablation.
    """
    per_particle_mse = error.square().mean(dim=-1)
    weight = (per_particle_mse.detach() + epsilon).pow(-power)
    weight = (weight / weight.mean().clamp_min(torch.finfo(weight.dtype).eps)).clamp(
        weight_min, weight_max
    )
    return (weight * per_particle_mse).sum() / weight.sum().clamp_min(
        torch.finfo(weight.dtype).eps
    )


def save_model(model: MFMLP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: MFMLP, path: Path, device: torch.device) -> MFMLP:
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def train_stage1(
    cfg: Config,
    model: MFMLP,
    run_dir: Path,
    levels: Optional[Sequence[float]] = None,
    return_history: bool = False,
) -> MFMLP:
    """Train Stage I using Algorithm 1 from the paper."""
    s1: Stage1Config = cfg.stage1
    device = torch.device(s1.device)
    if levels is None:
        levels = cfg.train_levels

    def _make_loader(seed: int) -> DataLoader:
        eps, x_ref, sigma = build_stage1_arrays(
            exp=cfg.exp,
            levels=levels,
            n_samples=s1.batch_size * max(1, s1.steps // 10),
            prior_std=s1.prior_std,
            seed=seed,
            pair_mode=getattr(s1, "pair_mode", "ot_periodic"),
        )
        dataset = Stage1Dataset(eps, x_ref, sigma)
        return DataLoader(
            dataset,
            batch_size=s1.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=s1.lr, weight_decay=s1.weight_decay
    )
    min_lr_ratio = float(getattr(s1, "min_lr_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(s1.steps, 1),
        eta_min=s1.lr * min_lr_ratio,
    )
    grad_clip = float(getattr(s1, "grad_clip", 0.0))
    pair_refresh = int(getattr(s1, "pair_refresh", 0) or 0)
    history: List[float] = []
    step = 0

    model.train()
    loader = _make_loader(s1.seed)
    loader_iter = iter(loader)
    while step < s1.steps:
        try:
            eps_b, x0_b, sigma_b = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            eps_b, x0_b, sigma_b = next(loader_iter)
        eps_b = eps_b.to(device)
        x0_b = x0_b.to(device)
        sigma_b = sigma_b.to(device)
        bsz = eps_b.shape[0]

        t, r = logit_normal_t_r(
            bsz,
            device,
            mu=s1.time_mu,
            sigma=s1.time_sigma,
            r_equals_t_ratio=s1.r_equals_t_ratio,
        )
        t_ = t[:, None]
        r_ = r[:, None]
        if getattr(s1, "geodesic_path", True):
            x0_b = x0_b + TWO_PI * torch.round((eps_b - x0_b) / TWO_PI)
        z = (1.0 - t_) * x0_b + t_ * eps_b
        v = eps_b - x0_b

        u, u_dot = model.jvp_u(z, r, t, sigma_b, v)
        u_tgt = (v - (t_ - r_) * u_dot).detach()
        loss = torch.mean((u - u_tgt) ** 2)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        history.append(float(loss.detach().cpu()))
        step += 1
        if step % 500 == 0 or step == s1.steps:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"stage1 step {step}/{s1.steps} loss={history[-1]:.4f} lr={lr_now:.2e}",
                flush=True,
            )

        if step % s1.ckpt_every == 0 or step == s1.steps:
            save_model(model, run_dir / "stage1_last.pt")
        if pair_refresh > 0 and step < s1.steps and step % pair_refresh == 0:
            print(f"stage1 refresh OT pairs at step {step}", flush=True)
            loader = _make_loader(s1.seed + step)
            loader_iter = iter(loader)

    save_model(model, run_dir / "stage1_final.pt")
    (run_dir / "stage1_loss.json").write_text(json.dumps(history), encoding="utf-8")
    if history:
        plot_loss_curve(
            history,
            run_dir / "stage1_loss.png",
            title="Stage I MeanFlow training loss",
        )
    if return_history:
        return model, history
    return model


def generate_mf(
    model: MFMLP,
    eps: np.ndarray,
    sigma: float,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Deterministic one-step MF sampling with torus wrapping."""
    model.eval()
    eps_arr = np.asarray(eps, dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, eps_arr.shape[0], batch_size):
            e = torch.from_numpy(eps_arr[i : i + batch_size]).to(device)
            s = torch.full((e.shape[0],), sigma, dtype=torch.float32, device=device)
            r = torch.zeros_like(s)
            t = torch.ones_like(s)
            u = model(e, r, t, s)
            out.append((e - u).cpu().numpy())
    return wrap_torus(np.concatenate(out, axis=0))
