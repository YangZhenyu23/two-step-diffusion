"""Single-sigma DP objective probe.

Compares, on one frozen MeanFlow and one sigma, whether a few thousand steps of
  (a) Algorithm-2 mini-batch EMD, or
  (b) a 72-bin periodic histogram Sinkhorn (the evaluation geometry)
actually lower the 72-bin W2 that we report.

Run:
    python -m two_step_2d.probe_dp_objective --objective emd
    python -m two_step_2d.probe_dp_objective --objective hist
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .data_io import TWO_PI, load_reference
from .eval_kpp import w2_2d_hist
from .models import DPResNet, MFMLP
from .ot_solver import EMDSolver, hard_hist2d, periodic_l2_sq_vector, sinkhorn_w2, soft_hist2d, torus_grid_cost
from .stage1 import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["emd", "hist"], required=True)
    ap.add_argument("--level", type=float, default=-3.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ckpt", default="two_step_2d/runs/kpp2d/full_2d_v3_ep/stage1_final.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mf = load_model(MFMLP(dim=2), args.ckpt, device).eval()
    for p in mf.parameters():
        p.requires_grad_(False)
    dp = DPResNet(dim=2, depth=6, alpha=0.2, wrap_output=False).to(device)
    opt = torch.optim.Adam(dp.parameters(), lr=args.lr)

    ref = load_reference("kpp2d", args.level)
    ref_t = torch.from_numpy(ref)
    bins = 72
    cost = torus_grid_cost(bins, device)
    target = hard_hist2d(ref_t, bins=bins, dims=(0, 1)).to(device)
    solver = EMDSolver(gamma_refresh=1)
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(0)

    def sample_mf(n: int, seed: int) -> torch.Tensor:
        g = np.random.default_rng(seed)
        eps = g.normal(0.0, np.pi, size=(n, 2)).astype(np.float32)
        e = torch.from_numpy(eps).to(device)
        s = torch.full((n,), args.level, device=device)
        with torch.no_grad():
            x = e - mf(e, torch.zeros_like(s), torch.ones_like(s), s)
            x = torch.remainder(x, TWO_PI)
        return x

    def w2(n: int = 8000, seed: int = 123) -> float:
        x = sample_mf(n, seed)
        s = torch.full((n,), args.level, device=device)
        with torch.no_grad():
            y = torch.remainder(dp(x, s), TWO_PI).cpu().numpy()
        idx = rng.choice(ref.shape[0], size=n, replace=False)
        return w2_2d_hist(y, ref[idx], bins)

    print(f"objective={args.objective} level={args.level} steps={args.steps}", flush=True)
    print(f"step 0 W2={w2():.4f}", flush=True)
    for step in range(args.steps):
        x = sample_mf(args.batch, 10_000 + step).detach()
        s = torch.full((args.batch,), args.level, device=device)
        y = dp(x, s)
        if args.objective == "emd":
            j = torch.randint(0, ref_t.shape[0], (args.batch,), generator=gen)
            loss = solver(y, ref_t[j].to(device))
            # tiny tether so the map stays a corrector, as in Eq.(11)
            loss = loss + 1e-3 * periodic_l2_sq_vector(y, x).mean()
        else:
            hist = soft_hist2d(y, bins=bins, blur=0.45, dims=(0, 1))
            loss = sinkhorn_w2(hist, target, cost, eps=0.02, n_iter=40)
            loss = loss + 1e-3 * periodic_l2_sq_vector(y, x).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dp.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0 or step + 1 == args.steps:
            disp = periodic_l2_sq_vector(y, x).mean().sqrt().item()
            print(
                f"step {step+1} loss={float(loss.detach()):.4f} "
                f"rms_disp={disp:.4f} W2={w2():.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
