"""Finetune MeanFlow or DeepParticle against a high-resolution FK cloud.

The 20k-particle reference used by the 72-bin metric has a self-distance near
0.06, so a training histogram built from it cannot supervise a 0.02 correction.
This script builds the target histogram from 160k particles of a 200k cloud
and scores exact EMD on the held-out 40k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data_io import TWO_PI
from .eval_kpp import w2_2d_hist
from .models import DPResNet, MFMLP
from .ot_solver import hard_hist2d, soft_hist2d
from .stage1 import load_model


def log_sinkhorn(a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor, eps: float, n_iter: int) -> torch.Tensor:
    """Entropic-OT dual. Stable at the small epsilon the 72-bin metric needs."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a / a.sum()
    b = b / b.sum()
    log_a = torch.log(a.clamp_min(1e-12))
    log_b = torch.log(b.detach().clamp_min(1e-12))
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    # Potentials are solved to convergence, then detached. The envelope theorem
    # says the derivative of the dual in the histogram is the potential itself,
    # so backpropagating through all 40 kernel iterations is unnecessary and
    # materializes a 72^2 by 72^2 graph.
    with torch.no_grad():
        for _ in range(n_iter):
            f = eps * (log_a - torch.logsumexp((-cost + g.unsqueeze(0)) / eps, dim=1))
            g = eps * (log_b - torch.logsumexp((-cost + f.unsqueeze(1)) / eps, dim=0))
    return (f.detach() * a).sum()


def predict(mf, dp, eps: torch.Tensor, level: float) -> torch.Tensor:
    s = torch.full((eps.shape[0],), level, device=eps.device)
    x = torch.remainder(eps - mf(eps, torch.zeros_like(s), torch.ones_like(s), s), TWO_PI)
    if dp is not None:
        x = torch.remainder(dp(x, s), TWO_PI)
    return x


def exact(mf, dp, level: float, ref_score: np.ndarray, device, n: int, seed: int, bins: int) -> float:
    rng = np.random.default_rng(seed)
    eps = torch.from_numpy(rng.normal(0.0, np.pi, size=(n, 2)).astype(np.float32)).to(device)
    with torch.no_grad():
        pred = predict(mf, dp, eps, level).cpu().numpy()
    return w2_2d_hist(pred, ref_score, bins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mf", "dp"], required=True)
    ap.add_argument("--level", type=float, default=-3.0)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--mf-ckpt", required=True)
    ap.add_argument("--dp-ckpt", default=None)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--blur", type=float, default=0.7)
    ap.add_argument("--eps", type=float, default=0.015)
    ap.add_argument("--sinkhorn-iter", type=int, default=40)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--holdout", type=int, default=40000)
    ap.add_argument("--probe-n", type=int, default=20000)
    ap.add_argument("--probe-every", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = np.load(args.ref)["qT"]
    hold = ref[-args.holdout :]
    train_ref = ref[: ref.shape[0] - args.holdout]
    target = hard_hist2d(torch.from_numpy(train_ref), bins=args.bins, dims=(0, 1)).to(device)
    centers = (torch.arange(args.bins, device=device) + 0.5) / args.bins * TWO_PI
    xx, yy = torch.meshgrid(centers, centers, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    diff = pts.unsqueeze(0) - pts.unsqueeze(1)
    diff = diff - TWO_PI * torch.round(diff / TWO_PI)
    cost = (diff * diff).sum(dim=-1)

    floor = w2_2d_hist(train_ref[: args.holdout], hold, args.bins)
    print(f"ref {len(ref)} train {len(train_ref)} holdout {len(hold)} floor {floor:.4f}", flush=True)

    mf = load_model(MFMLP(dim=2), args.mf_ckpt, device)
    dp = None
    if args.mode == "dp":
        mf.eval()
        for p in mf.parameters():
            p.requires_grad_(False)
        dp = DPResNet(dim=2, depth=6, alpha=0.2, wrap_output=False, displacement_cap=0.4).to(device)
        if args.dp_ckpt:
            dp.load_state_dict(torch.load(args.dp_ckpt, map_location=device))
        opt = torch.optim.AdamW(dp.parameters(), lr=args.lr)
        trainable = dp
    else:
        for p in mf.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(mf.parameters(), lr=args.lr)
        trainable = mf

    def score() -> float:
        was = trainable.training
        trainable.eval()
        value = exact(mf, dp, args.level, hold, device, args.probe_n, 666, args.bins)
        trainable.train(was)
        return value

    best = score()
    print(f"mode={args.mode} held-out W2 at step 0: {best:.4f}", flush=True)
    best_state = {k: v.detach().cpu().clone() for k, v in trainable.state_dict().items()}
    history = [{"step": 0, "heldout_w2": best, "floor": floor}]

    for step in range(args.steps):
        eps = torch.randn(args.batch, 2, device=device) * np.pi
        if args.mode == "mf":
            x = predict(mf, None, eps, args.level)
        else:
            with torch.no_grad():
                base = predict(mf, None, eps, args.level)
            x = dp(base, torch.full((args.batch,), args.level, device=device))
        hist = soft_hist2d(x, bins=args.bins, blur=args.blur, dims=(0, 1))
        loss = log_sinkhorn(hist, target, cost, eps=args.eps, n_iter=args.sinkhorn_iter)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"step {step+1} sinkhorn={float(loss.detach()):.5f}", flush=True)
        if (step + 1) % args.probe_every == 0 or step + 1 == args.steps:
            held = score()
            flag = ""
            if held < best:
                best = held
                best_state = {k: v.detach().cpu().clone() for k, v in trainable.state_dict().items()}
                flag = "  saved"
            history.append({"step": step + 1, "sinkhorn": float(loss.detach()), "heldout_w2": held})
            print(
                f"step {step+1} sinkhorn={float(loss.detach()):.5f} heldout={held:.4f}{flag}",
                flush=True,
            )

    name = "mf_best.pt" if args.mode == "mf" else "dp_best.pt"
    torch.save(best_state, out / name)
    (out / "history.json").write_text(json.dumps({"best": best, "history": history}, indent=2))
    print(f"best held-out W2 {best:.4f} saved {out / name}", flush=True)


if __name__ == "__main__":
    main()
