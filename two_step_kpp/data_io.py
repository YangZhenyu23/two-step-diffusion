"""Data loading and torus helpers for the KPP reproduction package."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


TWO_PI = 2.0 * math.pi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KPP_DATA_DIR = PROJECT_ROOT / "KPP data"
KPP_DATA_ROOT = KPP_DATA_DIR / "data"


def sigma_levels(n_train: int = 8, n_eval: int = 11) -> Tuple[List[float], List[float]]:
    """Return the dyadic log2(sigma) grids used by the paper (Eq. 25 and 26)."""
    train = [-2.0 - 0.25 * k for k in range(n_train)]
    evalt = [-2.0 - 0.25 * k for k in range(n_eval)]
    return train, evalt


TRAIN_LEVELS, EVAL_LEVELS = sigma_levels()


def log2sigma_tag(log2sigma: float) -> str:
    """Match the filename convention used by kpp_common/kpp3d_kolmogorov."""
    return str(log2sigma).replace(".", "p").replace("-", "m")


def resolve_reference_path(exp: str, log2sigma: float) -> Path:
    """Find an existing reference .npz for a given experiment and sigma."""
    if exp not in ("kpp2d", "kpp3d"):
        raise ValueError(f"exp must be 'kpp2d' or 'kpp3d', got {exp!r}")
    tag = log2sigma_tag(log2sigma)
    folder = KPP_DATA_ROOT / exp
    matches = sorted(folder.glob(f"*_l2s{tag}_*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No reference .npz found for exp={exp}, log2sigma={log2sigma} in {folder}"
        )
    return matches[0]


def load_reference(exp: str, log2sigma: float) -> np.ndarray:
    """Load the long-horizon FK reference point cloud as float32 (N, d)."""
    path = resolve_reference_path(exp, log2sigma)
    data = np.load(path, allow_pickle=False)
    key = "qT" if exp == "kpp2d" else "xT"
    if key not in data:
        raise KeyError(f"{path.name} does not contain {key!r}")
    return np.asarray(data[key], dtype=np.float32)


def load_cold_start(exp: str, log2sigma: float) -> np.ndarray:
    """Load the cold-start (uniform) point cloud stored in the reference file."""
    path = resolve_reference_path(exp, log2sigma)
    data = np.load(path, allow_pickle=False)
    key = "q0" if exp == "kpp2d" else "x0"
    if key not in data:
        raise KeyError(f"{path.name} does not contain {key!r}")
    return np.asarray(data[key], dtype=np.float32)


def load_meta(exp: str, log2sigma: float) -> Dict[str, object]:
    """Load and parse the metadata JSON stored inside the reference .npz."""
    path = resolve_reference_path(exp, log2sigma)
    data = np.load(path, allow_pickle=False)
    raw = str(data["meta"])
    return json.loads(raw)


def wrap_torus(x: np.ndarray) -> np.ndarray:
    """Fold coordinates into [0, 2*pi)."""
    return np.mod(np.asarray(x), TWO_PI)


def periodic_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed shortest displacement from a to b on the torus, per coordinate."""
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return diff - TWO_PI * np.round(diff / TWO_PI)


def periodic_l2_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared periodic Euclidean distance between matching or broadcastable points."""
    d = periodic_diff(a, b)
    return np.sum(d * d, axis=-1)


def build_stage1_arrays(
    exp: str,
    levels: Sequence[float],
    n_samples: int = 20000,
    prior_std: float = math.pi,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic Stage I pairs: Gaussian prior eps, target x_ref, sigma.

    Returns:
        eps:  float32 (n_levels * n_samples, d)
        x_ref: float32 (n_levels * n_samples, d)
        sigma: float32 (n_levels * n_samples,)
    """
    rng = np.random.default_rng(seed)
    eps_parts = []
    ref_parts = []
    sigma_parts = []

    for l2s in levels:
        x_ref = load_reference(exp, l2s)
        n, d = x_ref.shape
        count = min(n_samples, n)
        idx = np.sort(rng.choice(n, size=count, replace=False))
        x_ref = x_ref[idx]
        eps = rng.normal(0.0, prior_std, size=(count, d)).astype(np.float32)
        eps_parts.append(eps)
        ref_parts.append(x_ref)
        sigma_parts.append(np.full(count, l2s, dtype=np.float32))

    return (
        np.concatenate(eps_parts, axis=0),
        np.concatenate(ref_parts, axis=0),
        np.concatenate(sigma_parts, axis=0),
    )


def build_stage1_dataset(
    exp: str,
    levels: Sequence[float],
    n_samples: int = 20000,
    prior_std: float = math.pi,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility alias for build_stage1_arrays."""
    return build_stage1_arrays(
        exp=exp,
        levels=levels,
        n_samples=n_samples,
        prior_std=prior_std,
        seed=seed,
    )
