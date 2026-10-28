"""Stage II: DP refinement with mini-batch EMD coupling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config, Stage2Config
from .data_io import TWO_PI, build_stage1_arrays, load_reference, wrap_torus
from .models import DPResNet, MFMLP
from .ot_solver import (
    EMDSolver,
    hard_hist2d,
    periodic_l2_sq_vector,
    periodic_rbf_mmd,
    sinkhorn_w2,
    soft_hist2d,
    torus_grid_cost,
)
from .plot_kpp import plot_stage2_loss


def torch_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, TWO_PI)


def _residual_terms(
    x_dp: torch.Tensor, x_mf: torch.Tensor, s2: Stage2Config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (penalty, mean squared displacement) on the torus.

    ``hinge`` leaves displacements inside ``residual_free_radius`` unpenalized
    so OT can sharpen locally. Only jumps past that ball pay a mean cost.
    ``sum`` is the previous Eq.(11) scaling and freezes the map when OT is
    already a probability coupling.
    """
    disp2 = periodic_l2_sq_vector(x_dp, x_mf)
    mean_disp2 = torch.mean(disp2)
    reduction = getattr(s2, "residual_reduction", "hinge")
    if reduction == "hinge":
        tau = float(getattr(s2, "residual_free_radius", 0.30))
        return torch.mean(torch.relu(disp2 - tau * tau)), mean_disp2
    if reduction == "sum":
        return torch.sum(disp2), mean_disp2
    return mean_disp2, mean_disp2


def _sample_indices(
    pool: torch.Tensor, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    perm = torch.randperm(pool.shape[0], generator=generator)[:batch_size]
    return pool[perm]


def _probe_w2(
    cfg: Config,
    stage1_model: MFMLP,
    dp_model: DPResNet,
    device: torch.device,
    levels: Sequence[float],
    n_samples: int,
) -> Tuple[float, List[float]]:
    """Histogram W2 of MF+DP vs FK on a cheap subset. Lazy-import eval."""
    from .eval_kpp import w2_2d_hist, w2_3d_proj

    metric = w2_2d_hist if cfg.exp == "kpp2d" else w2_3d_proj
    rng = np.random.default_rng(cfg.evalt.seed)
    scores: List[float] = []
    was_training = dp_model.training
    dp_model.eval()
    for l2s in levels:
        ref = load_reference(cfg.exp, float(l2s))
        n = min(n_samples, ref.shape[0])
        idx = rng.choice(ref.shape[0], size=n, replace=False)
        ref = ref[idx]
        eps = rng.normal(0.0, cfg.stage1.prior_std, size=ref.shape).astype(np.float32)
        x_dp = generate_dp(stage1_model, dp_model, eps, float(l2s), device)
        scores.append(metric(x_dp, ref, cfg.evalt.bins))
    if was_training:
        dp_model.train()
    return float(np.mean(scores)), scores


def train_stage2(
    cfg: Config,
    stage1_model: MFMLP,
    dp_model: DPResNet,
    run_dir: Path,
    levels: Optional[Sequence[float]] = None,
) -> DPResNet:
    """Train the near-identity DP corrector using Algorithm 2.

    Stage I stays frozen. Priors and FK references are drawn independently,
    and FK references are redrawn together at batch_refresh so each
    mini-batch OT problem is held long enough to fit. The final weights
    are the probe-W2 best checkpoint, not the last step.
    """
    s2: Stage2Config = cfg.stage2
    device = torch.device(s2.device)
    if levels is None:
        levels = cfg.train_levels

    # Algorithm 2 draws priors and references independently; do not reuse
    # the Stage I OT pairing table.
    eps, x_ref, sigma = build_stage1_arrays(
        exp=cfg.exp,
        levels=levels,
        n_samples=20000,
        prior_std=cfg.stage1.prior_std,
        seed=s2.seed,
        pair_mode="independent",
    )
    eps_t = torch.from_numpy(eps)
    ref_t = torch.from_numpy(x_ref)
    sigma_t = torch.from_numpy(sigma)

    level_list = list(levels)
    index_map = {}
    for l2s in level_list:
        mask = torch.isclose(sigma_t, torch.tensor(l2s, dtype=torch.float32))
        index_map[l2s] = torch.nonzero(mask, as_tuple=False).squeeze(1)

    hist_dims = tuple(getattr(s2, "hist_dims", (0, 2)))
    hist_bins = int(getattr(s2, "hist_bins", 36))
    hist_cost = torus_grid_cost(hist_bins, device)
    fk_hist = {
        l2s: hard_hist2d(ref_t[index_map[l2s]], bins=hist_bins, dims=hist_dims).to(device)
        for l2s in level_list
    }

    stage1_model = stage1_model.to(device).eval()
    for parameter in stage1_model.parameters():
        parameter.requires_grad_(False)
    dp_model = dp_model.to(device)
    optimizer = torch.optim.AdamW(
        dp_model.parameters(), lr=s2.lr, weight_decay=s2.weight_decay
    )
    solver = EMDSolver(gamma_refresh=s2.gamma_refresh)

    generator = torch.Generator().manual_seed(s2.seed)
    src_idx = None
    ref_idx = None
    ref_big_idx = None
    pool = None
    level_index = 0
    current_level = level_list[0]
    ot_log: List[float] = []
    residual_log: List[float] = []
    mmd_log: List[float] = []
    hist_log: List[float] = []
    total_log: List[float] = []
    probe_log: List[dict] = []
    best_score = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in dp_model.state_dict().items()}

    reduction = getattr(s2, "residual_reduction", "hinge")
    probe_every = int(getattr(s2, "probe_every", 0) or 0)
    probe_n = int(getattr(s2, "probe_samples", 4000))
    probe_levels = list(level_list)

    from tqdm import tqdm

    def maybe_probe(step_num: int) -> None:
        nonlocal best_score, best_state
        if probe_every <= 0:
            return
        mean_w2, scores = _probe_w2(
            cfg, stage1_model, dp_model, device, probe_levels, probe_n
        )
        row = {"step": step_num, "mean_w2": mean_w2, "levels": probe_levels, "w2": scores}
        probe_log.append(row)
        print(
            f"stage2 probe step {step_num} mean_w2={mean_w2:.4f} "
            + " ".join(f"{lv:.2f}:{s:.4f}" for lv, s in zip(probe_levels, scores)),
            flush=True,
        )
        if mean_w2 < best_score:
            best_score = mean_w2
            best_state = {k: v.detach().cpu().clone() for k, v in dp_model.state_dict().items()}
            torch.save(best_state, run_dir / "stage2_best.pt")

    run_dir.mkdir(parents=True, exist_ok=True)
    maybe_probe(0)

    for step in tqdm(range(s2.steps), desc="stage2-dp"):
        if src_idx is None or step % s2.batch_refresh == 0:
            current_level = level_list[level_index % len(level_list)]
            level_index += 1
            pool = index_map[current_level]
            src_idx = _sample_indices(pool, s2.batch_size, generator)
            ref_idx = _sample_indices(pool, s2.batch_size, generator)
            big_n = min(int(pool.shape[0]), max(s2.batch_size, int(getattr(s2, "mmd_samples", 1024)) * 4))
            ref_big_idx = _sample_indices(pool, big_n, generator)
            solver.reset()

        eps_b = eps_t[src_idx].to(device)
        ref_b = ref_t[ref_idx].to(device)
        sigma_b = sigma_t[src_idx].to(device)

        with torch.no_grad():
            r0 = torch.zeros_like(sigma_b)
            t1 = torch.ones_like(sigma_b)
            u = stage1_model(eps_b, r0, t1, sigma_b)
            x_mf = torch_wrap(eps_b - u)

        x_dp = dp_model(x_mf, sigma_b)
        ot_loss = solver(x_dp, ref_b)
        residual_term, residual_mean = _residual_terms(x_dp, x_mf, s2)
        mmd_weight = float(getattr(s2, "mmd_weight", 0.0))
        mmd_loss = x_dp.new_zeros(())
        if mmd_weight > 0.0:
            mmd_n = min(
                int(getattr(s2, "mmd_samples", 1024)),
                x_dp.shape[0],
                int(ref_big_idx.shape[0]),
            )
            x_idx = torch.randperm(x_dp.shape[0], generator=generator)[:mmd_n]
            y_idx = torch.randperm(ref_big_idx.shape[0], generator=generator)[:mmd_n]
            ref_mmd = ref_t[ref_big_idx[y_idx]].to(device)
            mmd_loss = periodic_rbf_mmd(
                x_dp[x_idx],
                ref_mmd,
                lengthscale=float(getattr(s2, "mmd_lengthscale", 0.40)),
            )
        hist_loss = sinkhorn_w2(
            soft_hist2d(
                x_dp,
                bins=hist_bins,
                blur=float(getattr(s2, "hist_blur", 0.6)),
                dims=hist_dims,
            ),
            fk_hist[current_level],
            hist_cost,
            eps=float(getattr(s2, "sinkhorn_eps", 0.05)),
            n_iter=int(getattr(s2, "sinkhorn_iters", 30)),
        )
        ot_weight = float(getattr(s2, "ot_weight", 1.0))
        hist_weight = float(getattr(s2, "hist_weight", 0.0))
        loss = (
            ot_weight * ot_loss
            + hist_weight * hist_loss
            + s2.residual_weight * residual_term
            + mmd_weight * mmd_loss
        )

        ot_log.append(float(ot_loss.detach().cpu()))
        residual_log.append(float(residual_mean.detach().cpu()))
        mmd_log.append(float(mmd_loss.detach().cpu()))
        hist_log.append(float(hist_loss.detach().cpu()))
        total_log.append(float(loss.detach().cpu()))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dp_model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 500 == 0 or step + 1 == s2.steps:
            print(
                f"stage2 step {step + 1}/{s2.steps} "
                f"ot={ot_log[-1]:.4f} hist={hist_log[-1]:.5f} "
                f"res_mean={residual_log[-1]:.4f} disp={residual_log[-1] ** 0.5:.4f} "
                f"total={total_log[-1]:.4f}",
                flush=True,
            )

        if (step + 1) % s2.ckpt_every == 0 or step + 1 == s2.steps:
            torch.save(dp_model.state_dict(), run_dir / "stage2_last.pt")
        if probe_every > 0 and ((step + 1) % probe_every == 0 or step + 1 == s2.steps):
            maybe_probe(step + 1)

    dp_model.load_state_dict(best_state)
    torch.save(dp_model.state_dict(), run_dir / "stage2_final.pt")
    (run_dir / "stage2_loss.json").write_text(
        json.dumps(
            {
                "ot": ot_log,
                "residual": residual_log,
                "mmd": mmd_log,
                "hist": hist_log,
                "total": total_log,
                "probe": probe_log,
                "best_probe_w2": best_score,
                "residual_reduction": reduction,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if total_log:
        plot_stage2_loss(
            ot_log,
            residual_log,
            total_log,
            run_dir / "stage2_loss.png",
        )
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
