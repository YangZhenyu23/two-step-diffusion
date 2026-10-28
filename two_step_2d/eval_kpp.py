"""Evaluation metrics for the KPP two-step pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config
from .data_io import (
    TWO_PI,
    load_cold_start,
    load_meta,
    load_reference,
    periodic_diff,
)
from .models import DPResNet, MFMLP
from .stage1 import generate_mf
from .stage2 import generate_dp


def _hist_mass(samples: np.ndarray, dims: Sequence[int], bins: int):
    """Return normalized histogram mass and bin centers for selected dimensions."""
    sel = np.asarray(samples, dtype=np.float64)[:, dims]
    hist, edges = np.histogramdd(
        sel,
        bins=bins,
        range=[[0.0, TWO_PI], [0.0, TWO_PI]],
        density=False,
    )
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        raise ValueError("empty histogram")
    hist /= total
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    grid = np.meshgrid(*centers, indexing="ij")
    points = np.stack([g.ravel() for g in grid], axis=-1)
    return hist.ravel(), points


def _hist_w2(samples: np.ndarray, reference: np.ndarray, dims: Sequence[int], bins: int):
    import ot  # lazy import; requires POT

    p, points_p = _hist_mass(samples, dims, bins)
    q, points_q = _hist_mass(reference, dims, bins)
    # Zero-mass bins carry no transport; dropping them is exact and reduces the
    # network-simplex problem size without changing the optimal cost.
    keep = (p > 0) | (q > 0)
    p = p[keep]
    q = q[keep]
    points_p = points_p[keep]
    points_q = points_q[keep]
    diff = points_q[None, :, :] - points_p[:, None, :]
    diff = diff - TWO_PI * np.round(diff / TWO_PI)
    cost = (diff * diff).sum(axis=-1)
    # Paper metric: exact EMD on the 72x72 torus histogram with periodic cost.
    emd2 = ot.emd2(p, q, cost, numItermax=10_000_000)
    return float(math.sqrt(max(0.0, emd2)))


def w2_2d_hist(x: np.ndarray, ref: np.ndarray, bins: int = 72) -> float:
    return _hist_w2(x, ref, dims=(0, 1), bins=bins)


def w2_3d_proj(x: np.ndarray, ref: np.ndarray, bins: int = 72) -> float:
    return _hist_w2(x, ref, dims=(0, 2), bins=bins)


def w2_noise_floor(
    ref: np.ndarray,
    bins: int = 72,
    dims: Sequence[int] = (0, 1),
    n_gen: Optional[int] = None,
    seed: int = 0,
) -> float:
    """Sampling floor of the histogram W2 metric at the given cloud sizes.

    Two disjoint subsamples of the *same* reference cloud differ by finite-M
    discretization only, so their W2 is pure noise. It is reported here so that
    every W2 number can be read against the smallest difference this metric can
    actually resolve.

    The floor scales like 1/sqrt(M). Two disjoint halves each have M/2 points,
    which is equivalent to one M-point cloud compared with itself up to a
    factor sqrt(2); the raw half-vs-half value is therefore rescaled so the
    returned number is directly comparable to a `n_gen`-vs-`len(ref)`
    evaluation.
    """
    n_ref = ref.shape[0]
    n_half = n_ref // 2
    if n_half < 2:
        return float("nan")
    target = n_ref if n_gen is None else min(int(n_gen), n_ref)
    # Bootstrap two independent clouds of the *evaluation* size directly from
    # the reference. Drawing at the target size avoids extrapolating a
    # size-dependent rate: measured floors do not follow 1/sqrt(n) exactly, and
    # extrapolating from small splits biases the number low.
    rng = np.random.default_rng(seed)
    reps = 3
    vals = []
    for _ in range(reps):
        a = ref[rng.integers(0, n_ref, size=target)]
        b = ref[rng.integers(0, n_ref, size=target)]
        vals.append(_hist_w2(a, b, dims=dims, bins=bins))
    return float(np.mean(vals))


def _w2_halfsplit_floor(
    ref: np.ndarray,
    bins: int = 72,
    dims: Sequence[int] = (0, 1),
    seed: int = 0,
) -> float:
    """Disjoint-halves floor (no resampling). Kept for cross-checking."""
    n_ref = ref.shape[0]
    n_half = n_ref // 2
    if n_half < 2:
        return float("nan")
    perm = np.random.default_rng(seed).permutation(n_ref)
    return _hist_w2(
        ref[perm[:n_half]], ref[perm[n_half : 2 * n_half]], dims=dims, bins=bins
    )


def debias_w2(w2: float, floor: float) -> float:
    """Remove the known metric-floor variance: sqrt(W2^2 - floor^2).

    W2 measured against finite-M references is upward biased: even a perfect
    generator scores roughly the floor. Clipping at zero makes the result a
    conservative lower bound on the true distributional error.
    """
    return float(math.sqrt(max(0.0, w2 * w2 - floor * floor)))


def _resample(logw: np.ndarray, rng: np.random.Generator) -> tuple:
    mx = logw.max()
    w = np.exp(logw - mx)
    sw = w.sum()
    p = w / sw
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, rng.random(len(p)), side="right")
    np.clip(idx, 0, len(p) - 1, out=idx)
    return idx, mx + math.log(sw / len(p))


def eval_warm_cT(
    exp: str,
    x_init: np.ndarray,
    log2sigma: float,
    generations: Sequence[int],
    seed: int = 666,
    device: str = "cuda",
) -> Dict[int, float]:
    """Run the warm-start FK recursion and record c_T at requested generations.

    This is a standalone evaluator that mirrors Eq.(20)-(24) for 2D and the
    analogous 3D recursion, without modifying the original KPP scripts.
    """
    import torch

    dev = torch.device(device)
    rng = torch.Generator(device=dev).manual_seed(seed)
    lam = 2.0
    dt = 2.0 ** -8
    rn = 256
    kappa = 2.0 ** log2sigma
    sd = math.sqrt(2.0 * kappa * dt)
    q = torch.from_numpy(np.asarray(x_init, dtype=np.float64)).to(dev)
    n = q.shape[0]
    record = set(int(g) for g in generations if int(g) >= 1)
    max_g = max(record) if record else 0
    log_w_sum = 0.0
    out: Dict[int, float] = {}

    if exp == "kpp2d":
        theta = 1.0
        drift_q1 = 2.0 * kappa * lam
        v_const = kappa * lam * lam
        thetacos = theta * torch.cos(
            torch.tensor(TWO_PI, device=dev) * torch.arange(rn, device=dev) * dt
        )
        logw = torch.zeros(n, device=dev)
        for g in range(max_g):
            for j in range(rn):
                tc = thetacos[j]
                c1 = torch.cos(q[:, 0])
                c2 = torch.cos(q[:, 1])
                s1 = torch.sin(q[:, 0])
                s2 = torch.sin(q[:, 1])
                q[:, 0] += (drift_q1 - c2 - s1 * tc) * dt
                q[:, 1] += (c1 + s2 * tc) * dt
                q += sd * torch.randn((n, 2), device=dev, generator=rng)
                q %= TWO_PI
                logw += (
                    v_const
                    - lam * (torch.cos(q[:, 1]) + tc * torch.sin(q[:, 0]))
                ) * dt
                if (j + 1) % rn == 0:
                    idx, inc = _resample_torch(logw, rng, n)
                    log_w_sum += inc
                    q = q[idx]
                    logw.zero_()
            gen = g + 1
            if gen in record:
                mu = kappa + log_w_sum / gen
                out[gen] = mu / lam
        return out

    if exp == "kpp3d":
        drift_x1 = 2.0 * kappa * lam
        v_const = kappa * lam * lam
        phases = torch.sin(
            torch.tensor(TWO_PI, device=dev) * torch.arange(rn, device=dev) * dt
        )
        logw = torch.zeros(n, device=dev)
        for g in range(max_g):
            for j in range(rn):
                ph = phases[j]
                sx = torch.sin(q + ph)
                q[:, 0] += (drift_x1 - sx[:, 2]) * dt
                q[:, 1] += sx[:, 0] * dt
                q[:, 2] += sx[:, 1] * dt
                q += sd * torch.randn((n, 3), device=dev, generator=rng)
                q %= TWO_PI
                logw += (v_const - lam * torch.sin(q[:, 2] + ph)) * dt
                if (j + 1) % rn == 0:
                    idx, inc = _resample_torch(logw, rng, n)
                    log_w_sum += inc
                    q = q[idx]
                    logw.zero_()
            gen = g + 1
            if gen in record:
                mu = kappa + log_w_sum / gen
                out[gen] = mu / lam
        return out

    raise ValueError(f"exp must be 'kpp2d' or 'kpp3d', got {exp!r}")


def _resample_torch(logw: "torch.Tensor", rng: "torch.Generator", n: int):
    import torch

    mx = logw.max()
    w = torch.exp(logw - mx)
    sw = w.sum()
    p = w / sw
    cdf = torch.cumsum(p, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, torch.rand(n, device=logw.device, generator=rng))
    idx.clamp_(0, n - 1)
    return idx, float((mx + torch.log(sw / n)).detach().cpu())


def evaluate(
    cfg: Config,
    stage1_model: MFMLP,
    dp_model: DPResNet,
    out_dir: Path,
    levels: Optional[Sequence[float]] = None,
    compute_cT: bool = True,
) -> Dict[str, object]:
    """Evaluate MF-only and MF+DP against the FK reference and write metrics."""
    import torch

    device = torch.device(cfg.evalt.device)
    levels = list(cfg.eval_levels if levels is None else levels)
    prior_std = cfg.stage1.prior_std
    rng = np.random.default_rng(cfg.evalt.seed)
    results: Dict[str, object] = {"exp": cfg.exp, "levels": []}

    for l2s in levels:
        ref = load_reference(cfg.exp, l2s)
        cold = load_cold_start(cfg.exp, l2s)
        n = min(cfg.evalt.n_samples, ref.shape[0])
        idx = np.sort(rng.choice(ref.shape[0], size=n, replace=False))
        ref = ref[idx]
        cold = cold[idx]
        d = ref.shape[1]
        eps = rng.normal(0.0, prior_std, size=(n, d)).astype(np.float32)

        x_mf = generate_mf(stage1_model, eps, l2s, device)
        x_dp = generate_dp(stage1_model, dp_model, eps, l2s, device)

        if cfg.exp == "kpp2d":
            metric = w2_2d_hist
            dims = (0, 1)
        else:
            metric = w2_3d_proj
            dims = (0, 2)

        floor = w2_noise_floor(
            ref, bins=cfg.evalt.bins, dims=dims, n_gen=n, seed=cfg.evalt.seed
        )

        row: Dict[str, object] = {
            "log2sigma": l2s,
            "w2_cold": metric(cold, ref, cfg.evalt.bins),
            "w2_mf": metric(x_mf, ref, cfg.evalt.bins),
            "w2_dp": metric(x_dp, ref, cfg.evalt.bins),
            "w2_floor": floor,
        }
        # Debiased values make the "is this gain real?" question quantitative:
        # anything below the floor is indistinguishable from a perfect cloud.
        for key in ("w2_cold", "w2_mf", "w2_dp"):
            row[key + "_debiased"] = debias_w2(float(row[key]), floor)
        if compute_cT and abs(l2s - cfg.evalt.cT_sigma) < 1e-9:
            row["c_hat"] = float(load_meta(cfg.exp, l2s).get("c_hat", float("nan")))
            row["cT_mf"] = eval_warm_cT(
                cfg.exp,
                x_mf,
                l2s,
                cfg.evalt.cT_generations,
                seed=cfg.evalt.seed,
                device=cfg.evalt.device,
            )
            row["cT_dp"] = eval_warm_cT(
                cfg.exp,
                x_dp,
                l2s,
                cfg.evalt.cT_generations,
                seed=cfg.evalt.seed,
                device=cfg.evalt.device,
            )
        results["levels"].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results
