"""Temporary timing benchmark for Stage I/II. Not part of the package."""
from __future__ import annotations

import time
import torch

from .config import Config
from .data_io import build_stage1_arrays
from .models import DPResNet, MFMLP
from .ot_solver import EMDSolver, periodic_l2_sq_vector
from .stage1 import logit_normal_t_r
from .stage2 import torch_wrap


def bench() -> None:
    cfg = Config(exp="kpp2d")
    device = torch.device("cuda")
    dim = 2

    # Stage I micro-benchmark
    model = MFMLP(dim=dim).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    eps = torch.randn(4096, dim, device=device)
    x0 = torch.randn(4096, dim, device=device)
    sigma = torch.full((4096,), -2.0, device=device)
    t, r = logit_normal_t_r(4096, device)
    t_, r_ = t[:, None], r[:, None]
    z = (1 - t_) * x0 + t_ * eps
    v = eps - x0
    torch.cuda.synchronize()
    t0 = time.time()
    n1 = 5
    for _ in range(n1):
        u, u_dot = model.jvp_u(z, r, t, sigma, v)
        u_tgt = (v - (t_ - r_) * u_dot).detach()
        loss = torch.mean((u - u_tgt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    s1 = (time.time() - t0) / n1

    # Stage II micro-benchmark (batch 1500, full EMD each call here)
    dp = DPResNet(dim=dim, depth=6, alpha=0.1, wrap_output=True).to(device)
    dp.train()
    opt2 = torch.optim.AdamW(dp.parameters(), lr=1e-3)
    solver = EMDSolver(gamma_refresh=50)
    x = torch.randn(1500, dim, device=device)
    y = torch.randn(1500, dim, device=device)
    s = torch.full((1500,), -2.0, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    n2 = 3
    for _ in range(n2):
        x_dp = dp(x, s)
        ot_loss = solver(x_dp, y)
        residual = torch.mean(periodic_l2_sq_vector(x_dp, x))
        loss = ot_loss + 1e-3 * residual
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    torch.cuda.synchronize()
    s2 = (time.time() - t0) / n2

    print(f"stage1 {s1*1000:.1f} ms/step  (batch 4096)")
    print(f"stage2 {s2*1000:.1f} ms/step  (batch 1500)")


if __name__ == "__main__":
    bench()
