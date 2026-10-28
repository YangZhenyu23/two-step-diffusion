"""Train standard/Improved MeanFlow and a frozen-MF DP Keller--Segel model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics import save_evaluation_diagnostics, save_loss_plot, save_loss_vs_sigma_plot
from twostep import TwoStepDiffusion
from twostep_config import TwoStepConfig


class ParticleBatchDataset:
    """Condition-wise particle batches with paired or independent endpoints."""

    def __init__(
        self,
        inputs_path: Path,
        references_path: Path,
        particle_batch: int,
        seed: int,
        independent_reference: bool,
    ):
        self.inputs = np.load(inputs_path, mmap_mode="r")
        self.references = np.load(references_path, mmap_mode="r")
        if self.inputs.ndim != 3 or self.references.ndim != 3 or self.inputs.shape[:2] != self.references.shape[:2]:
            raise ValueError("inputs/references must be matching [n_conditions, N, features] arrays.")
        if not 0 < particle_batch <= self.inputs.shape[1]:
            raise ValueError(f"particle_batch must be in [1, {self.inputs.shape[1]}].")
        self.particle_batch = particle_batch
        self.independent_reference = independent_reference
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        condition_count, particles = self.inputs.shape[:2]
        rows = np.arange(condition_count)[:, None]
        while True:
            source_indices = np.stack(
                [self.rng.choice(particles, self.particle_batch, replace=False) for _ in range(condition_count)]
            )
            if self.independent_reference:
                reference_indices = np.stack(
                    [self.rng.choice(particles, self.particle_batch, replace=False) for _ in range(condition_count)]
                )
            else:
                reference_indices = source_indices
            yield self.inputs[rows, source_indices].copy(), self.references[rows, reference_indices].copy()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return str(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default="ks_data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="auto, cpu, cuda, or a device such as cuda:0")

    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument(
        "--imf",
        action="store_true",
        default=False,
        help="Use Improved Mean Flow for Stage I (default: standard MeanFlow).",
    )
    parser.add_argument("--particle-batch", type=int, default=1500, help="Paper Stage-II mini-batch size.")
    parser.add_argument("--mf-steps", type=int, default=20_000)
    parser.add_argument("--dp-steps", type=int, default=20_000)
    parser.add_argument("--batch-refresh", type=int, default=1000, help="Fresh Stage-II particle batch period.")
    parser.add_argument("--gamma-refresh", type=int, default=50, help="Exact EMD coupling refresh period.")
    parser.add_argument("--eval-particles", type=int, default=15_000, help="Particles per sigma used for paper-matched W2 evaluation.")
    parser.add_argument("--ks-w2-bins", type=int, default=24, help="Histogram cells per 3-D axis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs_path = args.data_dir / "inputs.npy"
    references_path = args.data_dir / "references.npy"
    test_inputs_path = args.data_dir / "test_inputs.npy"
    test_references_path = args.data_dir / "test_references.npy"
    required_paths = (inputs_path, references_path, test_inputs_path, test_references_path)
    if not all(path.exists() for path in required_paths):
        raise FileNotFoundError("Run generate_ks_twostep_data.py first.")
    inputs = np.load(inputs_path, mmap_mode="r")
    references = np.load(references_path, mmap_mode="r")
    test_inputs = np.load(test_inputs_path, mmap_mode="r")
    test_references = np.load(test_references_path, mmap_mode="r")
    metadata_path = args.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    state_dim = references.shape[-1]
    condition_dim = inputs.shape[-1] - state_dim
    if state_dim != 3 or condition_dim < 1:
        raise ValueError("This driver expects 3-D Keller--Segel inputs with at least one condition feature.")
    if test_references.shape[-1] != state_dim or test_inputs.shape[-1] != state_dim + condition_dim:
        raise ValueError("Training and test feature dimensions do not match.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    method_name = "imf" if args.imf else "original"
    
    print(f"\nUsing device: {device}\nUsing algorithm: {method_name}\n")
    
    output_dir = args.output_dir or args.data_dir / f"twostep_{method_name}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TwoStepConfig(
        state_dim=state_dim,
        condition_dim=condition_dim,
        mf_steps=args.mf_steps,
        dp_steps=args.dp_steps,
        batch_refresh=args.batch_refresh,
        gamma_refresh=args.gamma_refresh,
        improved_meanflow=args.imf,
        device=device,
    )
    paired_dataset = ParticleBatchDataset(
        inputs_path, references_path, args.particle_batch, args.seed, independent_reference=False
    )
    independent_dataset = ParticleBatchDataset(
        inputs_path, references_path, args.particle_batch, args.seed + 1, independent_reference=True
    )
    model = TwoStepDiffusion(config)
    print(
        f"Training data: inputs={tuple(inputs.shape)}, references={tuple(references.shape)}\n"
        f"Test data: inputs={tuple(test_inputs.shape)}, references={tuple(test_references.shape)}"
    )
    stage_one_name = "Improved Mean Flow" if config.improved_meanflow else "standard MeanFlow"
    print(f"Stage I: {stage_one_name}, {config.mf_steps} steps, paired endpoints")
    mf_history = model.train_meanflow(paired_dataset)
    model.save(output_dir / "meanflow.pt")
    print(
        f"Stage II: frozen MeanFlow + DP, {config.dp_steps} steps, "
        f"batch={args.particle_batch}, exact POT EMD, gamma refresh={config.gamma_refresh}"
    )
    dp_history = model.train_dp(independent_dataset)
    model.save(output_dir / "twostep.pt")
    sigma_values = (
        np.asarray(metadata["sigmas"], dtype=float)
        if "sigmas" in metadata
        else 10.0 ** np.asarray(inputs[:, 0, state_dim], dtype=float)
    )
    test_sigma_values = (
        np.asarray(metadata["test_sigmas"], dtype=float)
        if "test_sigmas" in metadata
        else 10.0 ** np.asarray(test_inputs[:, 0, state_dim], dtype=float)
    )
    test_regimes = metadata.get(
        "test_regimes",
        ["interpolation" if sigma <= float(sigma_values.max()) else "extrapolation" for sigma in test_sigma_values],
    )
    save_loss_plot(
        np.asarray(mf_history),
        np.asarray(dp_history),
        output_dir,
        model.meanflow_history,
        model.dp_history,
    )
    training_metrics = save_evaluation_diagnostics(
        model,
        inputs,
        references,
        output_dir,
        "train",
        eval_particles=args.eval_particles,
        histogram_bins=args.ks_w2_bins,
        seed=args.seed,
        sigma_values=sigma_values,
    )
    test_metrics = save_evaluation_diagnostics(
        model,
        test_inputs,
        test_references,
        output_dir,
        "test",
        eval_particles=args.eval_particles,
        histogram_bins=args.ks_w2_bins,
        seed=args.seed + 10_000,
        sigma_values=test_sigma_values,
        regime_values=test_regimes,
    )
    save_loss_vs_sigma_plot(
        training_metrics,
        test_metrics,
        output_dir,
        histogram_bins=args.ks_w2_bins,
        training_boundary_sigma=float(sigma_values.max()),
    )
    print("Finished. Saved the last Stage-I and Stage-II model states.")
    print("Saved four diagnostic images and loss_vs_sigma.json.")


if __name__ == "__main__":
    main()
