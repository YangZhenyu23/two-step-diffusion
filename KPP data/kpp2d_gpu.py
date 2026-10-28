"""GPU Feynman-Kac reference clouds at higher particle count.

Same update as kpp_common.kpp2d_fk. Numpy's generator takes about 80 minutes
for M=2e4; this is about 10 minutes for M=2e5 on one GPU. The random stream
is not identical, so c_hat matches only statistically.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

TWO_PI = 2.0 * math.pi
OUT = Path(__file__).resolve().parent / "data" / "kpp2d_hires"


def run(log2sigma: float, M: int = 200_000, T: int = 4096, seed: int = 666) -> Path:
    dev = torch.device("cuda")
    lam, theta = 2.0, 1.0
    dt = 2.0 ** -8
    rn = 256
    kappa = 2.0 ** log2sigma
    sd = math.sqrt(2.0 * kappa * dt)
    drift = 2.0 * kappa * lam
    v_const = kappa * lam * lam
    thetacos = theta * torch.cos(torch.arange(rn, device=dev, dtype=torch.float32) * dt * TWO_PI)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    q = torch.rand(M, 2, device=dev, generator=g) * TWO_PI
    logw = torch.zeros(M, device=dev)
    log_w_sum = 0.0
    t0 = time.perf_counter()
    trace = []
    for gen in range(1, T + 1):
        for j in range(rn):
            tc = thetacos[j]
            c1 = torch.cos(q[:, 0])
            c2 = torch.cos(q[:, 1])
            s1 = torch.sin(q[:, 0])
            s2 = torch.sin(q[:, 1])
            nse = sd * torch.randn(M, 2, device=dev, generator=g)
            q = torch.stack(
                (q[:, 0] + (drift - c2 - s1 * tc) * dt, q[:, 1] + (c1 + s2 * tc) * dt),
                dim=1,
            )
            q = torch.remainder(q + nse, TWO_PI)
            logw = logw + (v_const - lam * (torch.cos(q[:, 1]) + tc * torch.sin(q[:, 0]))) * dt
        mx = torch.max(logw)
        w = torch.exp(logw - mx)
        sw = torch.sum(w)
        log_w_sum += float(mx + torch.log(sw / M))
        q = q[torch.multinomial(w, M, replacement=True, generator=g)]
        logw = torch.zeros(M, device=dev)
        if gen in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096) or gen % 256 == 0:
            c = (kappa + log_w_sum / gen) / lam
            el = time.perf_counter() - t0
            print(
                f"l2s={log2sigma:.2f} gen {gen}/{T} c={c:.4f} "
                f"{el / gen * 1000:.0f} ms/gen ETA {(T - gen) * el / gen / 60:.1f} min",
                flush=True,
            )
            trace.append([gen, c])
    qT = q.detach().float().cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = str(log2sigma).replace(".", "p").replace("-", "m")
    path = OUT / f"ref_l2s{tag}_M{M}_T{T}_seed{seed}.npz"
    meta = {
        "log2sigma": log2sigma,
        "M": M,
        "T": T,
        "seed": seed,
        "c_hat": trace[-1][1],
        "trace": trace,
        "kappa": kappa,
    }
    np.savez_compressed(path, qT=qT.astype(np.float32), meta=json.dumps(meta))
    print(f"wrote {path} c_hat={meta['c_hat']:.4f}", flush=True)
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="-3.0,-4.5")
    p.add_argument("--particles", type=int, default=200_000)
    p.add_argument("--generations", type=int, default=4096)
    args = p.parse_args()
    for lv in [float(x) for x in args.levels.split(",") if x.strip()]:
        run(lv, M=args.particles, T=args.generations)
