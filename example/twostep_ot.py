"""Exact mini-batch EMD routines for Keller--Segel Stage II."""

import numpy as np
import torch
from torch import Tensor

try:
    import ot
except ImportError:  # pragma: no cover - exercised only when POT is absent
    ot = None


def _validate_clouds(generated: Tensor, reference: Tensor) -> None:
    if generated.ndim != 3 or reference.ndim != 3:
        raise ValueError("generated and reference must have shape [B, N, d] and [B, M, d].")
    if generated.shape[0] != reference.shape[0] or generated.shape[-1] != reference.shape[-1]:
        raise ValueError("generated and reference must have matching batch and state dimensions.")
    if generated.shape[1] == 0 or reference.shape[1] == 0:
        raise ValueError("Particle clouds must be non-empty.")
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("Particle clouds contain NaN or infinity.")


def pairwise_squared_cost(generated: Tensor, reference: Tensor) -> Tensor:
    """Differentiable squared Euclidean cost without a [B,N,M,d] tensor."""
    _validate_clouds(generated, reference)
    generated_norm = generated.square().sum(dim=-1, keepdim=True)
    reference_norm = reference.square().sum(dim=-1).unsqueeze(-2)
    cross = generated @ reference.transpose(-1, -2)
    return (generated_norm + reference_norm - 2.0 * cross).clamp_min(0.0)


def exact_emd_coupling(generated: Tensor, reference: Tensor) -> Tensor:
    """Return the exact POT transport plan for each conditioned mini-batch."""
    if ot is None:
        raise ImportError("Stage-II exact EMD requires POT: install it with `pip install POT`.")
    _validate_clouds(generated, reference)
    generated_np = generated.detach().cpu().double().numpy()
    reference_np = reference.detach().cpu().double().numpy()
    batch_size, generated_count, _ = generated_np.shape
    reference_count = reference_np.shape[1]
    generated_weights = np.full(generated_count, 1.0 / generated_count, dtype=np.float64)
    reference_weights = np.full(reference_count, 1.0 / reference_count, dtype=np.float64)
    gamma = np.empty((batch_size, generated_count, reference_count), dtype=np.float64)
    for index in range(batch_size):
        difference = generated_np[index, :, None, :] - reference_np[index, None, :, :]
        cost = np.einsum("ijk,ijk->ij", difference, difference)
        gamma[index] = ot.emd(generated_weights, reference_weights, cost)
    return torch.from_numpy(gamma).to(device=generated.device, dtype=generated.dtype)


emd_coupling = exact_emd_coupling
