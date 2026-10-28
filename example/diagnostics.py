"""Required training and evaluation plots for Keller--Segel Two-Step Diffusion."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import ot
except ImportError as error:
    raise ImportError("Diagnostics require POT: install it with `pip install POT`.") from error


def _sample_indices(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0:
        raise ValueError("Particle evaluation limit must be positive.")
    return np.arange(count) if count <= limit else rng.choice(count, limit, replace=False)


def _histogram_w2(prediction: np.ndarray, reference: np.ndarray, edges: list[np.ndarray]) -> float:
    """Approximate 3-D W2 by transporting occupied histogram-cell masses."""
    predicted_histogram, _ = np.histogramdd(prediction, bins=edges)
    reference_histogram, _ = np.histogramdd(reference, bins=edges)

    def occupied(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.argwhere(histogram > 0)
        if len(indices) == 0:
            raise ValueError("Histogram contains no occupied cells.")
        masses = histogram[tuple(indices.T)]
        centers = np.stack(
            [
                (edges[axis][indices[:, axis]] + edges[axis][indices[:, axis] + 1]) / 2
                for axis in range(histogram.ndim)
            ],
            axis=1,
        )
        return centers, masses / masses.sum()

    predicted_centers, predicted_mass = occupied(predicted_histogram)
    reference_centers, reference_mass = occupied(reference_histogram)
    cost = ((predicted_centers[:, None, :] - reference_centers[None, :, :]) ** 2).sum(axis=-1)
    return float(np.sqrt(max(ot.emd2(predicted_mass, reference_mass, cost), 0.0)))


def _histogram_edges(*clouds: np.ndarray, bins: int) -> list[np.ndarray]:
    if bins <= 0:
        raise ValueError("Histogram bin count must be positive.")
    combined = np.concatenate(clouds, axis=0)
    lower, upper = combined.min(axis=0), combined.max(axis=0)
    padding = np.maximum((upper - lower) * 0.02, 1e-6)
    return [
        np.linspace(lower[axis] - padding[axis], upper[axis] + padding[axis], bins + 1)
        for axis in range(combined.shape[-1])
    ]


def save_loss_plot(
    meanflow_loss: np.ndarray,
    dp_loss: np.ndarray,
    output_dir: str | Path,
    meanflow_history: dict[str, list[float]] | None = None,
    dp_history: dict[str, list[float]] | None = None,
) -> None:
    """Plot only the requested raw Stage-I groups and total Stage-II loss."""
    del dp_history  # Retained in the signature for compatibility with older callers.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meanflow_loss = np.asarray(meanflow_loss)
    dp_loss = np.asarray(dp_loss)
    if meanflow_history is None:
        raise ValueError("meanflow_history is required to plot the raw r<t and r=t metrics.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps_mf = np.arange(1, len(meanflow_loss) + 1)
    for key, label in (
        ("raw_mse_different_time", "raw MSE, r<t"),
        ("raw_mse_same_time", "raw MSE, r=t"),
    ):
        values = np.asarray(meanflow_history.get(key, []), dtype=float)
        if len(values) != len(meanflow_loss):
            raise ValueError(f"{key} must contain one value per Stage-I training step.")
        axes[0].plot(steps_mf, values, linewidth=0.9, label=label)
    axes[0].set(title="Stage I: raw MeanFlow metrics", xlabel="Training step", ylabel="Raw MSE")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.arange(1, len(dp_loss) + 1), dp_loss, linewidth=0.9, color="tab:blue")
    axes[1].set(title="Stage II: total DP loss", xlabel="Training step", ylabel="Total loss")

    for axis in axes:
        line_values = np.concatenate([line.get_ydata() for line in axis.lines]) if axis.lines else np.array([])
        finite = line_values[np.isfinite(line_values)]
        if len(finite) and np.all(finite > 0):
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_evaluation_diagnostics(
    model,
    inputs: np.ndarray,
    references: np.ndarray,
    output_dir: str | Path,
    split_name: str,
    eval_particles: int = 15_000,
    histogram_bins: int = 24,
    seed: int = 88,
    sigma_values: np.ndarray | None = None,
    regime_values: list[str] | np.ndarray | None = None,
) -> list[dict]:
    """Compute per-sigma raw W2 values and save one particle-distribution plot."""
    if split_name not in {"train", "test"}:
        raise ValueError("split_name must be 'train' or 'test'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if inputs.ndim != 3 or references.ndim != 3 or inputs.shape[:2] != references.shape[:2]:
        raise ValueError("inputs and references must be matching rank-3 particle arrays.")
    if not np.isfinite(inputs).all() or not np.isfinite(references).all():
        raise ValueError("Evaluation arrays contain NaN or infinity.")
    state_dim = references.shape[-1]
    if state_dim != 3:
        raise ValueError("The paper-matched histogram evaluation requires 3-D particle clouds.")

    source = np.array(inputs[..., :state_dim], copy=True)
    condition = np.array(inputs[..., state_dim:], copy=True)
    references = np.array(references, copy=True)
    predictions_mf = model.meanflow_sample(source, condition).cpu().numpy()
    predictions_dp = model.sample(source, condition).cpu().numpy()
    condition_count, particle_count, _ = references.shape
    if sigma_values is None:
        sigma_values = 10.0 ** np.asarray(condition[:, 0, 0], dtype=float)
    else:
        sigma_values = np.asarray(sigma_values, dtype=float)
    if sigma_values.shape != (condition_count,):
        raise ValueError("sigma_values must have one entry per conditioned cloud.")
    if regime_values is not None:
        regime_values = np.asarray(regime_values, dtype=str)
        if regime_values.shape != (condition_count,):
            raise ValueError("regime_values must have one entry per conditioned cloud.")

    metrics = []
    for index in range(condition_count):
        prediction_indices = _sample_indices(particle_count, eval_particles, rng)
        reference_indices = _sample_indices(particle_count, eval_particles, rng)
        target = references[index, reference_indices]
        meanflow = predictions_mf[index, prediction_indices]
        refined = predictions_dp[index, prediction_indices]
        edges = _histogram_edges(meanflow, refined, target, bins=histogram_bins)
        row = {
            "sigma": float(sigma_values[index]),
            "stage_one_w2": _histogram_w2(meanflow, target, edges),
            "stage_two_w2": _histogram_w2(refined, target, edges),
            "particles": int(len(target)),
        }
        if regime_values is not None:
            row["regime"] = str(regime_values[index])
        metrics.append(row)

    if regime_values is None:
        chosen = sorted(set((0, condition_count // 2, condition_count - 1)))
    else:
        interpolation = np.flatnonzero(regime_values == "interpolation")
        extrapolation = np.flatnonzero(regime_values == "extrapolation")
        chosen = []
        if len(interpolation):
            chosen.append(int(interpolation[len(interpolation) // 2]))
        if len(extrapolation):
            chosen.extend((int(extrapolation[0]), int(extrapolation[-1])))
        chosen = sorted(set(chosen))
        if not chosen:
            chosen = sorted(set((0, condition_count // 2, condition_count - 1)))

    coordinate_pairs = ((0, 1), (0, 2), (1, 2))
    coordinate_names = ("x", "y", "z")
    fig, axes = plt.subplots(
        len(chosen) * len(coordinate_pairs),
        4,
        figsize=(12, 3 * len(chosen) * len(coordinate_pairs)),
        squeeze=False,
    )
    labels = ("Source", "Reference", "MeanFlow", "Two-Step")
    plot_row = 0
    for index in chosen:
        prediction_indices = _sample_indices(particle_count, min(eval_particles, 1500), rng)
        reference_indices = _sample_indices(particle_count, min(eval_particles, 1500), rng)
        clouds = (
            source[index, prediction_indices],
            references[index, reference_indices],
            predictions_mf[index, prediction_indices],
            predictions_dp[index, prediction_indices],
        )
        sigma_label = f"sigma={sigma_values[index]:.4g}"
        if regime_values is not None:
            sigma_label += f"; {regime_values[index]}"
        for first, second in coordinate_pairs:
            projected = np.concatenate([cloud[:, (first, second)] for cloud in clouds])
            lower, upper = projected.min(axis=0), projected.max(axis=0)
            padding = np.maximum((upper - lower) * 0.05, 1e-6)
            for column, (label, cloud) in enumerate(zip(labels, clouds)):
                axis = axes[plot_row, column]
                axis.scatter(cloud[:, first], cloud[:, second], s=2, alpha=0.35)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel(coordinate_names[first])
                axis.set_ylabel(coordinate_names[second])
                axis.set_title(f"{label}\n{sigma_label}", fontsize=9)
                axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
                axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            plot_row += 1
    fig.tight_layout()
    fig.savefig(output_dir / f"particle_distributions_{split_name}.png", dpi=180)
    plt.close(fig)
    return metrics


def save_loss_vs_sigma_plot(
    training_metrics: list[dict],
    test_metrics: list[dict],
    output_dir: str | Path,
    histogram_bins: int = 24,
    training_boundary_sigma: float = 150.0,
) -> None:
    """Save the paper-style train/test W2-vs-sigma figure and its plotting data."""
    if not training_metrics or not test_metrics:
        raise ValueError("Training and test metrics must both be non-empty.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_metrics = sorted(training_metrics, key=lambda row: row["sigma"])
    test_metrics = sorted(test_metrics, key=lambda row: row["sigma"])
    particle_counts = sorted({row["particles"] for row in training_metrics + test_metrics})
    payload = {
        "metric": "histogram-based W2",
        "histogram_bins_per_axis": int(histogram_bins),
        "particle_counts": particle_counts,
        "training_boundary_sigma": float(training_boundary_sigma),
        "train": training_metrics,
        "test": test_metrics,
    }
    (output_dir / "loss_vs_sigma.json").write_text(json.dumps(payload, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, title, rows in zip(axes, ("Training data", "Test data"), (training_metrics, test_metrics)):
        sigma = np.asarray([row["sigma"] for row in rows], dtype=float)
        stage_one = np.asarray([row["stage_one_w2"] for row in rows], dtype=float)
        stage_two = np.asarray([row["stage_two_w2"] for row in rows], dtype=float)
        axis.plot(sigma, stage_one, "o-", color="#f39c12", linewidth=2.2, markersize=6, label="Meanflow")
        axis.plot(
            sigma,
            stage_two,
            "o-",
            color="#72a7f7",
            linewidth=2.2,
            markersize=6,
            label="Meanflow + DP refinement",
        )
        axis.axvline(training_boundary_sigma, color="#2ecc71", linestyle="--", linewidth=1.8)
        axis.set(title=title, xlabel=r"$\sigma$", ylabel=r"$W_2$")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.28)
        axis.legend(fontsize=9)
        axis.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_sigma.png", dpi=180)
    plt.close(fig)
