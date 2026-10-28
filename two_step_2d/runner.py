"""Shared command-line runner used by run_2d.py and run_3d.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from .config import Config, smoke_config
from .data_io import EVAL_LEVELS, TRAIN_LEVELS
from .eval_kpp import evaluate
from .models import DPResNet, MFMLP
from .plot_kpp import plot_cT_curve, plot_final_distributions, plot_w2_curve
from .stage1 import generate_mf, load_model, train_stage1
from .stage2 import generate_dp, train_stage2


RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def _parse_levels(text: Optional[str], default: List[float]) -> List[float]:
    if not text:
        return list(default)
    return [float(x) for x in text.split(",") if x.strip()]


def _resolve_device(cfg: Config, requested: Optional[str]) -> None:
    if requested:
        cfg.stage1.device = requested
        cfg.stage2.device = requested
        cfg.evalt.device = requested
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.stage1.device = device
    cfg.stage2.device = device
    cfg.evalt.device = device


def run(exp: str, mode: str, args: argparse.Namespace) -> Path:
    exp = "kpp2d"
    if mode == "smoke":
        cfg = smoke_config(exp)
        cfg.train_levels = _parse_levels(args.levels, [-2.0])
        cfg.eval_levels = _parse_levels(args.eval_levels, [-2.0, -4.5])
        cfg.run_name = "smoke"
    else:
        cfg = Config(exp=exp, run_name=args.run_name or "run")
        cfg.train_levels = _parse_levels(args.levels, TRAIN_LEVELS)
        cfg.eval_levels = _parse_levels(args.eval_levels, EVAL_LEVELS)

    _resolve_device(cfg, args.device)
    torch.manual_seed(cfg.stage1.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.stage1.seed)
    # 2D defaults live in Stage2Config (6-block, no wrap, shorter refresh).

    run_dir = RUNS_ROOT / exp / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dim = 2 if exp == "kpp2d" else 3
    stage1_model = MFMLP(dim=dim)
    dp_model = DPResNet(
        dim=dim,
        depth=cfg.stage2.depth,
        alpha=cfg.stage2.alpha,
        wrap_output=cfg.stage2.wrap_output,
    )

    if mode in ("train-stage1", "all", "smoke"):
        train_stage1(cfg, stage1_model, run_dir, levels=cfg.train_levels)

    if mode in ("train-stage2", "all", "smoke"):
        device = torch.device(cfg.stage1.device)
        stage1_model = load_model(stage1_model, run_dir / "stage1_final.pt", device)
        train_stage2(cfg, stage1_model, dp_model, run_dir, levels=cfg.train_levels)

    if mode in ("evaluate", "all", "smoke"):
        device = torch.device(cfg.evalt.device)
        if not (run_dir / "stage1_final.pt").exists():
            raise FileNotFoundError(
                "stage1_final.pt is missing; run train-stage1 before evaluate"
            )
        if not (run_dir / "stage2_final.pt").exists():
            raise FileNotFoundError(
                "stage2_final.pt is missing; run train-stage2 before evaluate"
            )
        stage1_model = load_model(stage1_model, run_dir / "stage1_final.pt", device)
        dp_model.load_state_dict(torch.load(run_dir / "stage2_final.pt", map_location=device))
        dp_model = dp_model.to(device)
        results = evaluate(
            cfg,
            stage1_model,
            dp_model,
            run_dir,
            levels=cfg.eval_levels,
            compute_cT=(mode != "smoke"),
        )
        try:
            plot_w2_curve(results, run_dir / "w2_curve.png")
            if mode != "smoke":
                plot_cT_curve(results, run_dir / "cT_curve.png")
            plot_final_distributions(
                cfg.exp,
                stage1_model,
                dp_model,
                run_dir,
                levels=[cfg.evalt.cT_sigma],
                device=device,
                bins=cfg.evalt.bins,
            )
        except Exception as exc:  # plotting should never block metric output
            print(f"plot skipped: {exc}")

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-Step Diffusion KPP runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "train-stage1", "train-stage2", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--levels", default=None, help="comma-separated train sigma")
    parser.add_argument("--eval-levels", default=None, help="comma-separated eval sigma")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    return parser
