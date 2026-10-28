"""Generate 3-D Keller--Segel laminar-flow training and held-out test data.

The Two-Step paper specifies eight log-spaced flow amplitudes in [15, 150],
15,000 particles per amplitude, final time 0.02, and dt=0.005. Parameters not
fully restated there use the preceding DeepParticle KS setup: uniform unit-ball
initial data, mu=chi=1, total mass 16*pi, and delta^2=1e-3. Its Figure 2 test
sweep uses sigma=20,40,...,200, with 20--140 labeled interpolation and
160--200 labeled extrapolation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

N_SIGMA = 8
N_PARTICLES = 15_000
SIGMAS = 15.0 * 10.0 ** (np.arange(N_SIGMA) / (N_SIGMA - 1))
TEST_SIGMAS = np.arange(20.0, 201.0, 20.0)
FINAL_TIME = 2e-2
DT = 5e-3
MU = 1.0
CHI = 1.0
TOTAL_MASS = 16.0 * np.pi
KERNEL_DELTA = np.sqrt(1e-3)
SEED = 42
TEST_SEED = 43


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu.")
    return device


def sample_unit_ball(generator: torch.Generator, count: int, device: torch.device) -> Tensor:
    direction = torch.randn((count, 3), generator=generator, device=device)
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(torch.finfo(direction.dtype).eps)
    radius = torch.rand(count, generator=generator, device=device).pow(1.0 / 3.0)
    return direction * radius[:, None]


def chemotaxis_gradient(x: Tensor, particle_mass: float, delta: float, block_size: int) -> Tensor:
    """Compute grad(K_delta * rho) for K(r)=-1/(4*pi*r).

    K_delta(r) = -r / (4*pi*(r^2+delta^2)). With the outer ``-chi`` in the
    particle SDE, well-separated pairs move toward one another.
    """
    count = len(x)
    gradient = torch.zeros_like(x)
    delta_squared = delta * delta
    epsilon = torch.finfo(x.dtype).eps
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        difference = x[start:stop, None, :] - x[None, :, :]
        radius_squared = difference.square().sum(dim=-1)
        radius = radius_squared.sqrt()
        safe_radius = radius.clamp_min(epsilon)
        coefficient = (radius_squared - delta_squared) / (
            4.0 * torch.pi * safe_radius * (radius_squared + delta_squared).square()
        )
        coefficient.masked_fill_(radius_squared == 0, 0.0)
        gradient[start:stop] = (difference * coefficient[..., None]).sum(dim=1) * particle_mass
    return gradient


def laminar_velocity(x: Tensor, sigma: float) -> Tensor:
    velocity = torch.zeros_like(x)
    velocity[:, 0] = sigma * torch.exp(-(x[:, 1].square() + x[:, 2].square()))
    return velocity


def simulate_ks(
    initial: Tensor,
    sigma: float,
    generator: torch.Generator,
    block_size: int,
) -> Tensor:
    x = initial.clone()
    particle_mass = TOTAL_MASS / len(x)
    noise_scale = np.sqrt(2.0 * MU * DT)
    for _ in range(round(FINAL_TIME / DT)):
        interaction = -CHI * chemotaxis_gradient(x, particle_mass, KERNEL_DELTA, block_size)
        velocity = interaction + laminar_velocity(x, sigma)
        # Sequential coordinate update: x first, then y using new x, then z using new y.
        x[:, 0] += DT * velocity[:, 0]
        x[:, 1] += DT * velocity[:, 1]
        x[:, 2] += DT * velocity[:, 2]
        x += noise_scale * torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return x


def check_attractive_sign(device: torch.device) -> None:
    """Catch a reversed Green-kernel sign before generating an expensive data set."""
    pair = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
    gradient = chemotaxis_gradient(pair, particle_mass=0.5, delta=KERNEL_DELTA, block_size=2)
    velocity = -CHI * gradient
    separation = pair[0] - pair[1]
    separation_velocity = velocity[0] - velocity[1]
    if torch.dot(separation, separation_velocity) >= 0:
        raise RuntimeError("The chemotaxis interaction is not attractive; check the kernel sign.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("ks_data"))
    parser.add_argument("--device", default="cuda", help="auto, cpu, cuda, or a device such as cuda:0")
    parser.add_argument("--block-size", type=int, default=1024, help="Interaction rows evaluated at once.")
    return parser.parse_args()


def _simulate_conditions(
    initial: Tensor,
    sigmas: np.ndarray,
    generator: torch.Generator,
    block_size: int,
    device: torch.device,
    split_name: str,
) -> np.ndarray:
    references = []
    for index, sigma in tqdm(enumerate(sigmas, start=1), total=len(sigmas), desc=split_name):
        print(f"Simulating {split_name} sigma={sigma:.6g} ({index}/{len(sigmas)}) on {device}...")
        references.append(simulate_ks(initial, float(sigma), generator, block_size).cpu().numpy())
    return np.stack(references).astype(np.float32, copy=False)


def _conditioned_inputs(initial: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    source = np.repeat(initial[None], len(sigmas), axis=0)
    condition = np.log10(sigmas).astype(np.float32)[:, None, None]
    return np.concatenate((source, np.broadcast_to(condition, (len(sigmas), len(initial), 1))), axis=-1)


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")
    device = resolve_device(args.device)
    print(f"Using block_size {args.block_size}, device {device}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_generator = torch.Generator(device=device).manual_seed(SEED)
    test_generator = torch.Generator(device=device).manual_seed(TEST_SEED)
    check_attractive_sign(device)
    initial = sample_unit_ball(training_generator, N_PARTICLES, device)
    references_np = _simulate_conditions(
        initial, SIGMAS, training_generator, args.block_size, device, "training"
    )
    test_references_np = _simulate_conditions(
        initial, TEST_SIGMAS, test_generator, args.block_size, device, "test"
    )
    initial_np = initial.cpu().numpy().astype(np.float32, copy=False)
    inputs = _conditioned_inputs(initial_np, SIGMAS)
    test_inputs = _conditioned_inputs(initial_np, TEST_SIGMAS)
    np.save(args.output_dir / "inputs.npy", inputs)
    np.save(args.output_dir / "references.npy", references_np)
    np.save(args.output_dir / "test_inputs.npy", test_inputs)
    np.save(args.output_dir / "test_references.npy", test_references_np)
    test_regimes = ["interpolation" if sigma <= float(SIGMAS.max()) else "extrapolation" for sigma in TEST_SIGMAS]
    metadata = {
        "problem": "3D Keller-Segel laminar flow",
        "sigmas": SIGMAS.tolist(),
        "test_sigmas": TEST_SIGMAS.tolist(),
        "test_regimes": test_regimes,
        "test_protocol": "Two-Step Diffusion Figure 2 sweep; 20-140 interpolation, 160-200 extrapolation",
        "condition": "log10(sigma)",
        "particles_per_sigma": N_PARTICLES,
        "final_time": FINAL_TIME,
        "dt": DT,
        "mu": MU,
        "chi": CHI,
        "total_mass": TOTAL_MASS,
        "kernel": "K_delta(r)=-r/(4*pi*(r^2+delta^2))",
        "kernel_delta": KERNEL_DELTA,
        "parameter_source": "Two-Step protocol with missing KS constants inherited from DeepParticle 2",
        "seed": SEED,
        "test_seed": TEST_SEED,
        "shared_initial_cloud_across_splits": True,
        "device": str(device),
        "block_size": args.block_size,
        "inputs_shape": list(inputs.shape),
        "references_shape": list(references_np.shape),
        "test_inputs_shape": list(test_inputs.shape),
        "test_references_shape": list(test_references_np.shape),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Saved training inputs/references {inputs.shape}/{references_np.shape} and "
        f"test inputs/references {test_inputs.shape}/{test_references_np.shape} to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
