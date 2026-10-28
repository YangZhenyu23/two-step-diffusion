"""Quick hyperparameter search for the Stage II DP corrector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import Config
from .data_io import build_stage1_arrays, load_reference
from .eval_kpp import w2_2d_hist, w2_3d_proj
from .models import DPResNet, MFMLP
from .ot_solver import EMDSolver, periodic_l2_sq_vector
from .stage2 import torch_wrap


RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def _train_combo(
    cfg: Config,
    stage1_model: MFMLP,
    dp_model: DPResNet,
    eps_t: torch.Tensor,
    ref_t: torch.Tensor,
    sigma_t: torch.Tensor,
    index_map: Dict[float, torch.Tensor],
    level_list: List[float],
    steps: int,
    batch_size: int,
    lr: float,
    lambda_res: float,
    device: torch.device,
) -> DPResNet:
    stage1_model = stage1_model.to(device).eval()
    dp_model = dp_model.to(device)
    optimizer = torch.optim.AdamW(dp_model.parameters(), lr=lr)
    solver = EMDSolver(gamma_refresh=cfg.stage2.gamma_refresh)
    generator = torch.Generator().manual_seed(cfg.stage2.seed)
    current_idx = None
    level_index = 0

    for step in range(steps):
        if current_idx is None or step % cfg.stage2.batch_refresh == 0:
            if current_idx is not None:
                solver.reset()
            level = level_list[level_index % len(level_list)]
            level_index += 1
            pool = index_map[level]
            perm = torch.randperm(pool.shape[0], generator=generator)[:batch_size]
            current_idx = pool[perm]

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
        residual = torch.mean(periodic_l2_sq_vector(x_dp, x_mf))
        loss = ot_loss + lambda_res * residual
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return dp_model


def _eval_combo(
    exp: str,
    stage1_model: MFMLP,
    dp_model: DPResNet,
    eval_levels: List[float],
    device: torch.device,
    n_samples: int = 1000,
    bins: int = 36,
    seed: int = 123,
) -> Tuple[float, Dict[float, float]]:
    stage1_model = stage1_model.to(device).eval()
    dp_model = dp_model.to(device).eval()
    rng = np.random.default_rng(seed)
    dim = 2 if exp == "kpp2d" else 3
    values = []
    per_level = {}
    metric = w2_2d_hist if exp == "kpp2d" else w2_3d_proj

    for l2s in eval_levels:
        ref = load_reference(exp, l2s)
        idx = rng.choice(ref.shape[0], size=n_samples, replace=False)
        ref = ref[idx]
        eps = rng.normal(0.0, math.pi, size=(n_samples, dim)).astype(np.float32)
        eps_t = torch.from_numpy(eps).to(device)
        s = torch.full((n_samples,), l2s, dtype=torch.float32, device=device)
        with torch.no_grad():
            r0 = torch.zeros_like(s)
            t1 = torch.ones_like(s)
            x_mf = torch_wrap(eps_t - stage1_model(eps_t, r0, t1, s))
            x_dp = dp_model(x_mf, s)
        w = metric(x_dp.cpu().numpy(), ref, bins)
        values.append(w)
        per_level[l2s] = w
    return float(np.mean(values)), per_level


def run_search(exp: str, stage1_run: str, out_path: Path) -> None:
    cfg = Config(exp=exp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 2 if exp == "kpp2d" else 3

    stage1_path = RUNS_ROOT / exp / stage1_run / "stage1_final.pt"
    stage1_model = MFMLP(dim=dim)
    stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))

    levels = cfg.train_levels
    eps, ref, sigma = build_stage1_arrays(
        exp=exp,
        levels=levels,
        n_samples=20000,
        prior_std=cfg.stage1.prior_std,
        seed=cfg.stage2.seed,
    )
    eps_t = torch.from_numpy(eps)
    ref_t = torch.from_numpy(ref)
    sigma_t = torch.from_numpy(sigma)
    index_map = {}
    for l2s in levels:
        mask = torch.isclose(sigma_t, torch.tensor(l2s, dtype=torch.float32))
        index_map[l2s] = torch.nonzero(mask, as_tuple=False).squeeze(1)

    eval_levels = [-2.0, -3.0, -3.75, -4.25, -4.5]
    results = []
    for alpha in [0.05, 0.1]:
        for lambda_res in [0.0005, 0.001, 0.005]:
            dp_model = DPResNet(
                dim=dim, depth=cfg.stage2.depth, alpha=alpha, wrap_output=True
            )
            dp_model = _train_combo(
                cfg,
                stage1_model,
                dp_model,
                eps_t,
                ref_t,
                sigma_t,
                index_map,
                levels,
                steps=1500,
                batch_size=300,
                lr=cfg.stage2.lr,
                lambda_res=lambda_res,
                device=device,
            )
            w2, per_level = _eval_combo(
                exp, stage1_model, dp_model, eval_levels, device=device
            )
            results.append(
                {
                    "exp": exp,
                    "alpha": alpha,
                    "lambda_res": lambda_res,
                    "val_w2": w2,
                    "per_level": per_level,
                }
            )
            print(exp, "alpha", alpha, "lambda_res", lambda_res, "val_w2", round(w2, 5), "per_level", {k: round(v, 4) for k, v in per_level.items()})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = min(
        results,
        key=lambda r: max(r["per_level"].values()) + 0.5 * r["val_w2"],
    )
    print("BEST", best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["kpp2d", "kpp3d"], required=True)
    parser.add_argument("--stage1-run", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    stage1_run = args.stage1_run or ("full_2d" if args.exp == "kpp2d" else "full_3d")
    out = Path(args.out or f"search_stage2_{args.exp}.json")
    run_search(args.exp, stage1_run, out)


if __name__ == "__main__":
    main()
