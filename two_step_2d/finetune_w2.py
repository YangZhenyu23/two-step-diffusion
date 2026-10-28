"""Finetune Stage I or Stage II against the 72-bin torus W2, not the training loss.

The reported number is exact EMD on a 72x72 periodic histogram. This script
adds a differentiable Sinkhorn distance on that same grid and keeps the
checkpoint with the lowest exact W2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data_io import TWO_PI, load_reference
from .eval_kpp import w2_2d_hist
from .models import DPResNet, MFMLP
from .ot_solver import hard_hist2d, sinkhorn_w2, soft_hist2d, torus_grid_cost
from .stage1 import load_model


def exact_w2(mf, dp, level: float, device, n: int = 8000, seed: int = 7) -> float:
    ref = load_reference("kpp2d", level)
    rng = np.random.default_rng(seed)
    idx = rng.choice(ref.shape[0], size=n, replace=False)
    eps = rng.normal(0.0, np.pi, size=(n, 2)).astype(np.float32)
    e = torch.from_numpy(eps).to(device)
    s = torch.full((n,), level, device=device)
    with torch.no_grad():
        x = torch.remainder(e - mf(e, torch.zeros_like(s), torch.ones_like(s), s), TWO_PI)
        if dp is not None:
            x = torch.remainder(dp(x, s), TWO_PI)
        pred = x.cpu().numpy()
    return w2_2d_hist(pred, ref[idx], 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mf", "dp"], required=True)
    ap.add_argument("--level", type=float, default=-3.0)
    ap.add_argument(
        "--levels",
        default=None,
        help="comma-separated log2 sigma; cycles through them when set",
    )
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w2-weight", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=1.2)
    ap.add_argument("--ckpt", default="two_step_2d/runs/kpp2d/full_2d_v3_ep/stage1_final.pt")
    ap.add_argument("--out", default="two_step_2d/runs/kpp2d/w2_finetune")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mf = load_model(MFMLP(dim=2), args.ckpt, device).eval()
    for p in mf.parameters():
        p.requires_grad_(False)
    dp = None
    if args.mode == "mf":
        mf.train()
        for p in mf.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(mf.parameters(), lr=args.lr)
        trainable = mf
    else:
        dp = DPResNet(dim=2, depth=6, alpha=0.2, wrap_output=False, displacement_cap=0.25).to(device)
        opt = torch.optim.AdamW(dp.parameters(), lr=args.lr)
        trainable = dp

    if args.levels:
        levels = [float(x) for x in args.levels.split(",") if x.strip()]
    else:
        levels = [args.level]
    targets = {
        lv: hard_hist2d(torch.from_numpy(load_reference("kpp2d", lv)), bins=72, dims=(0, 1)).to(device)
        for lv in levels
    }
    cost = torus_grid_cost(72, device)

    def mean_exact() -> float:
        return float(np.mean([exact_w2(mf, dp, lv, device, n=6000, seed=7) for lv in levels]))

    best = mean_exact()
    print(
        f"mode={args.mode} levels={levels} exact W2 mean at step 0: {best:.4f}",
        flush=True,
    )
    best_state = {k: v.detach().cpu().clone() for k, v in trainable.state_dict().items()}

    for step in range(args.steps):
        level = levels[step % len(levels)]
        eps = torch.randn(args.batch, 2, device=device) * np.pi
        s = torch.full((args.batch,), level, device=device)
        zeros = torch.zeros_like(s)
        ones = torch.ones_like(s)
        if args.mode == "mf":
            x = torch.remainder(eps - mf(eps, zeros, ones, s), TWO_PI)
        else:
            with torch.no_grad():
                base = torch.remainder(eps - mf(eps, zeros, ones, s), TWO_PI)
            x = dp(base, s)
        hist = soft_hist2d(x, bins=72, blur=args.blur, dims=(0, 1))
        loss = args.w2_weight * sinkhorn_w2(hist, targets[level], cost, eps=0.05, n_iter=20)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        opt.step()
        if (step + 1) % 400 == 0 or step + 1 == args.steps:
            score = mean_exact()
            flag = ""
            if score < best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in trainable.state_dict().items()}
                flag = "  saved"
            print(
                f"step {step+1} sinkhorn={float(loss.detach()):.4f} exactW2={score:.4f}{flag}",
                flush=True,
            )

    torch.save(best_state, out / f"{args.mode}_l{args.level}_best.pt")
    print(f"best exact W2 {best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
