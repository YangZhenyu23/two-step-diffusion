"""Figure generation for KPP evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .data_io import TWO_PI


def plot_loss_curve(
    values: Sequence[float],
    out_path: Path,
    title: str = "Training loss",
    ylabel: str = "loss",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0), layout="constrained")
    ax.plot(values, lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(ls="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stage2_loss(
    ot_values: Sequence[float],
    residual_values: Sequence[float],
    total_values: Sequence[float],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0), layout="constrained")
    ax.plot(total_values, lw=0.7, label="total")
    ax.plot(ot_values, lw=0.7, alpha=0.7, label="OT")
    ax.plot(residual_values, lw=0.7, alpha=0.7, label="residual")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Stage II DP training loss")
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_w2_curve(results: Dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["levels"]
    xs = [-r["log2sigma"] for r in rows]
    mf = [r["w2_mf"] for r in rows]
    dp = [r["w2_dp"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    ax.plot(xs, mf, "o-", color="#e8a33d", label="MeanFlow only")
    ax.plot(xs, dp, "s-", color="#6cafe6", label="MF + DP")
    ax.axvline(3.75, ls="--", color="green", label="train range end")
    ax.set_xlabel(r"$-\log_2(\sigma)$")
    ax.set_ylabel(r"$W_2$")
    ax.grid(ls="--", alpha=0.4)
    ax.legend()
    ax.set_title(results.get("exp", "KPP"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cT_curve(results: Dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["levels"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    for r in rows:
        if "cT_mf" not in r or "cT_dp" not in r:
            continue
        gens = sorted(r["cT_mf"])
        mf = [r["cT_mf"][g] for g in gens]
        dp = [r["cT_dp"][g] for g in gens]
        label = f"sigma={r['log2sigma']:.2f}"
        ax.plot(gens, mf, "o-", alpha=0.5, label=f"{label} MF")
        ax.plot(gens, dp, "s--", alpha=0.5, label=f"{label} DP")
        if "c_hat" in r and r["c_hat"] == r["c_hat"]:
            ax.axhline(r["c_hat"], ls=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("generations")
    ax.set_ylabel(r"$\hat{c}_T(\lambda)$")
    ax.grid(ls="--", alpha=0.4)
    ax.set_title(f"{results.get('exp', 'KPP')} eigenvalue convergence")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_panels(
    samples: Sequence[np.ndarray],
    titles: Sequence[str],
    out_path: Path,
    dims: Sequence[int] = (0, 1),
    bins: int = 72,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(
        1, n, figsize=(3.5 * n, 3.2), squeeze=False, layout="constrained"
    )
    vmax = max(
        np.histogram2d(
            s[:, dims[0]], s[:, dims[1]], bins=bins, range=[[0, TWO_PI], [0, TWO_PI]]
        )[0].max()
        for s in samples
    )
    for ax, s, title in zip(axes.ravel(), samples, titles):
        ax.hist2d(
            s[:, dims[0]],
            s[:, dims[1]],
            bins=bins,
            range=[[0, TWO_PI], [0, TWO_PI]],
            cmap="viridis",
            vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_distributions(
    exp: str,
    stage1_model,
    dp_model,
    out_dir: Path,
    levels: Sequence[float],
    device,
    bins: int = 72,
    n_samples: int = 2000,
    seed: int = 666,
) -> None:
    """Plot reference / MeanFlow / DP columns for selected sigma levels."""
    import numpy as np
    import torch

    from .data_io import load_reference
    from .stage1 import generate_mf
    from .stage2 import generate_dp

    dim = 2 if exp == "kpp2d" else 3
    dims = (0, 1) if exp == "kpp2d" else (0, 2)
    rng = np.random.default_rng(seed)
    stage1_model = stage1_model.to(device).eval()
    dp_model = dp_model.to(device).eval()

    for l2s in levels:
        ref = load_reference(exp, l2s)
        idx = rng.choice(ref.shape[0], size=n_samples, replace=False)
        ref = ref[idx]
        eps = rng.normal(0.0, np.pi, size=(n_samples, dim)).astype(np.float32)
        mf = generate_mf(stage1_model, eps, l2s, device)
        dp = generate_dp(stage1_model, dp_model, eps, l2s, device)
        plot_panels(
            [ref, mf, dp],
            ["Reference", "MeanFlow", "MF+DP"],
            out_dir / f"dist_l2s{str(l2s).replace('.','p').replace('-','m')}.png",
            dims=dims,
            bins=bins,
        )
