"""Compare a Stage-I run against the committed baseline on the same metric.

Usage:
    python -m two_step_2d.compare_stage1 --run probe_global_l3 --baseline full_2d_v2

Reports raw and floor-debiased W2 per level, plus how much of the gap to the
paper's MeanFlow column remains.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data_io import load_reference
from .eval_kpp import debias_w2, w2_2d_hist, w2_noise_floor
from .models import MFMLP
from .stage1 import generate_mf, load_model

RUNS = Path(__file__).resolve().parent / "runs" / "kpp2d"

PAPER_MF = {
    -2.0: 0.0477, -2.25: 0.0476, -2.5: 0.0532, -2.75: 0.0592,
    -3.0: 0.0639, -3.25: 0.0717, -3.5: 0.0870, -3.75: 0.0773,
    -4.0: 0.1012, -4.25: 0.1195, -4.5: 0.1176,
}


def evaluate_run(run: str, levels, device, n=20000, seed=666):
    run_dir = RUNS / run
    ckpt = run_dir / "stage1_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model = load_model(MFMLP(dim=2), ckpt, device)
    model.eval()
    out = []
    for l2s in levels:
        ref = load_reference("kpp2d", l2s)
        rng = np.random.default_rng(seed)
        k = min(n, ref.shape[0])
        idx = np.sort(rng.choice(ref.shape[0], size=k, replace=False))
        ref_sub = ref[idx]
        eps = rng.normal(0.0, np.pi, size=(k, 2)).astype(np.float32)
        x = generate_mf(model, eps, l2s, device)
        raw = w2_2d_hist(x, ref_sub, 72)
        floor = w2_noise_floor(ref_sub, bins=72, dims=(0, 1), n_gen=k, seed=seed)
        out.append(
            {
                "level": l2s,
                "w2": raw,
                "floor": floor,
                "debiased": debias_w2(raw, floor),
                "paper": PAPER_MF.get(round(l2s, 4)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument(
        "--levels",
        default="-2.0,-3.0,-3.75,-4.5",
        help="comma separated",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [float(x) for x in args.levels.split(",")]
    device = torch.device(args.device)

    new = evaluate_run(args.run, levels, device)
    base = evaluate_run(args.baseline, levels, device) if args.baseline else None

    hdr = f"{'level':>6} {'paper':>7} {'base':>7} {'base_deb':>9} {'new':>7} {'new_deb':>8} {'delta':>7} {'floor':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i, row in enumerate(new):
        b = base[i] if base else None
        print(
            f"{row['level']:6.2f} "
            f"{(row['paper'] if row['paper'] is not None else float('nan')):7.4f} "
            f"{(b['w2'] if b else float('nan')):7.4f} "
            f"{(b['debiased'] if b else float('nan')):9.4f} "
            f"{row['w2']:7.4f} {row['debiased']:8.4f} "
            f"{(row['debiased'] - b['debiased']) if b else float('nan'):7.4f} "
            f"{row['floor']:7.4f}"
        )
    if base:
        mb = float(np.mean([r["debiased"] for r in base]))
        mn = float(np.mean([r["debiased"] for r in new]))
        mp = float(np.mean([r["paper"] for r in new if r["paper"] is not None]))
        print(f"\ndebiased mean: baseline {mb:.4f} -> new {mn:.4f} (paper {mp:.4f})")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"run": args.run, "baseline": args.baseline, "new": new, "base": base}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
