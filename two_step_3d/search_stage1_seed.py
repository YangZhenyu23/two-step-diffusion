"""Short Stage-I seed screening for 3D KPP."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data_io import load_reference
from .eval_kpp import w2_3d_proj
from .models import MFMLP
from .stage1 import generate_mf, train_stage1


def main() -> None:
    seeds = [0, 1, 2, 3, 4, 5]
    eval_levels = [-2.0, -3.0, -3.75, -4.5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cfg = Config(exp="kpp3d")
        cfg.stage1.steps = 3000
        cfg.stage1.ckpt_every = 10000
        cfg.stage1.lr = 3e-4
        cfg.stage1.seed = seed
        cfg.stage1.device = str(device)

        model = MFMLP(dim=3)
        run_dir = Path(tempfile.mkdtemp(prefix=f"stage1_seed_{seed}_"))
        train_stage1(cfg, model, run_dir, levels=cfg.train_levels)
        model = model.cpu()

        values = []
        rng = np.random.default_rng(123)
        for l2s in eval_levels:
            ref = load_reference("kpp3d", l2s)
            idx = rng.choice(ref.shape[0], size=1000, replace=False)
            ref = ref[idx]
            eps = rng.normal(0.0, math.pi, size=(1000, 3)).astype(np.float32)
            x_mf = generate_mf(model, eps, l2s, torch.device("cpu"))
            values.append(w2_3d_proj(x_mf, ref, bins=36))

        mean_w2 = float(np.mean(values))
        max_w2 = float(np.max(values))
        results.append(
            {
                "seed": seed,
                "mean_val_w2": mean_w2,
                "max_val_w2": max_w2,
                "per_level": dict(zip(eval_levels, values)),
            }
        )
        print(seed, "mean", round(mean_w2, 5), "max", round(max_w2, 5), "per_level", [round(v, 5) for v in values])

    out = Path(__file__).with_name("stage1_seed_search_3d.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = min(results, key=lambda r: r["mean_val_w2"])
    print("BEST", best)


if __name__ == "__main__":
    main()
