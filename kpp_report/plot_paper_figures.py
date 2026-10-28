import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from two_step_2d.data_io import load_reference as load_ref_2d
from two_step_2d.models import DPResNet as DP2, MFMLP as MF2
from two_step_2d.stage1 import generate_mf as gen_mf_2d
from two_step_2d.stage2 import generate_dp as gen_dp_2d
from two_step_3d.data_io import load_reference as load_ref_3d
from two_step_3d.models import DPResNet as DP3, MFMLP as MF3
from two_step_3d.stage1 import generate_mf as gen_mf_3d
from two_step_3d.stage2 import generate_dp as gen_dp_3d


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
})


def wrap(x):
    return np.mod(np.asarray(x, dtype=np.float64), 2.0 * math.pi)


def hist2d(x, y, bins=72):
    H, _, _ = np.histogram2d(
        wrap(x), wrap(y), bins=bins, range=[[0, 2*math.pi], [0, 2*math.pi]]
    )
    return H.T


def show_hist(ax, H, vmax=None, colorbar=False):
    im = ax.imshow(
        H, origin="lower", extent=[0, 2*math.pi, 0, 2*math.pi],
        cmap="viridis", interpolation="nearest", aspect="equal", vmax=vmax,
    )
    ticks = [0, 1, 2, 3, 4, 5, 6]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(0, 2*math.pi)
    ax.set_ylim(0, 2*math.pi)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([int(v) for v in np.linspace(0, H.max(), 5)])
    return im


def get_models_2d(device):
    run = ROOT / "two_step_2d/runs/kpp2d/full_2d_v2_dpfix2"
    mf = MF2(dim=2)
    mf.load_state_dict(torch.load(run / "stage1_final.pt", map_location=device))
    dp = DP2(dim=2, depth=6, alpha=0.2, wrap_output=False)
    dp.load_state_dict(torch.load(run / "stage2_final.pt", map_location=device))
    return mf.to(device), dp.to(device)


def get_models_3d(device):
    run = ROOT / "two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2"
    mf = MF3(dim=3)
    mf.load_state_dict(torch.load(run / "stage1_final.pt", map_location=device))
    dp = DP3(dim=3, depth=6, alpha=0.2, wrap_output=False)
    dp.load_state_dict(torch.load(run / "stage2_final.pt", map_location=device))
    return mf.to(device), dp.to(device)


def sample_2d(mf, dp, sigma, n=20000, device="cpu"):
    eps = np.random.default_rng(666).normal(0.0, math.pi, size=(n, 2)).astype(np.float32)
    mf_out = gen_mf_2d(mf, eps, sigma, torch.device(device))
    dp_out = gen_dp_2d(mf, dp, eps, sigma, torch.device(device))
    return mf_out, dp_out


def sample_3d(mf, dp, sigma, n=20000, device="cpu"):
    eps = np.random.default_rng(666).normal(0.0, math.pi, size=(n, 3)).astype(np.float32)
    mf_out = gen_mf_3d(mf, eps, sigma, torch.device(device))
    dp_out = gen_dp_3d(mf, dp, eps, sigma, torch.device(device))
    return mf_out, dp_out


def fig5(device):
    mf, dp = get_models_2d(device)
    sigmas = [-3.75, -4.0]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), layout="constrained")
    for r, sigma in enumerate(sigmas):
        ref = load_ref_2d("kpp2d", sigma)
        mf_out, dp_out = sample_2d(mf, dp, sigma, device=device)
        data = [mf_out, dp_out, ref]
        titles = ["Meanflow", "DP refined", "Reference"]
        for c, (d, t) in enumerate(zip(data, titles)):
            ax = axes[r, c]
            show_hist(ax, hist2d(d[:, 0], d[:, 1]), colorbar=True)
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("x")
            if c == 0:
                ax.set_ylabel("y")
    fig.savefig(OUT / "fig5_invariant_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig6(device):
    mf, dp = get_models_2d(device)
    met = json.loads((ROOT / "two_step_2d/runs/kpp2d/full_2d_v2_dpfix2/metrics.json").read_text())
    rows = met["levels"]
    xs = [-r["log2sigma"] for r in rows]
    mf_w2 = [r["w2_mf"] for r in rows]
    dp_w2 = [r["w2_dp"] for r in rows]
    fig, (ax_tab, ax_plt) = plt.subplots(1, 2, figsize=(10.2, 3.6), width_ratios=[1, 1.4], layout="constrained")
    table_data = [["$-\log_2\sigma$", "Meanflow", "DP refinement"]]
    table_data += [[f"{x:.2f}", f"{a:.4f}", f"{b:.4f}"] for x, a, b in zip(xs, mf_w2, dp_w2)]
    tab = ax_tab.table(cellText=table_data, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    ax_tab.axis("off")
    ax_plt.plot(xs, mf_w2, marker="o", color="#e8a33d", label="Meanflow")
    ax_plt.plot(xs, dp_w2, marker="o", color="#6cafe6", label="DP refinement")
    ax_plt.axvline(3.75, color="green", linestyle="--")
    ax_plt.set_xlabel(r"$-\log_2\sigma$")
    ax_plt.set_ylabel(r"$W_2$")
    ax_plt.set_ylim(0, None)
    ax_plt.legend(loc="upper left")
    fig.savefig(OUT / "fig6_w2_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig7(device):
    mf, dp = get_models_2d(device)
    met = json.loads((ROOT / "two_step_2d/runs/kpp2d/full_2d_v2_dpfix2/metrics.json").read_text())
    row = next(r for r in met["levels"] if abs(r["log2sigma"] + 4.5) < 1e-9)
    gens = sorted(int(k) for k in row["cT_mf"])
    mf_c = [row["cT_mf"][str(g)] for g in gens]
    dp_c = [row["cT_dp"][str(g)] for g in gens]
    fig, (ax_tab, ax_plt) = plt.subplots(1, 2, figsize=(10.2, 3.6), width_ratios=[1, 1.4], layout="constrained")
    table_data = [["iteration", "MF-only init", "MF+DP init"]]
    table_data += [[str(g), f"{row['cT_mf'][str(g)]:.6f}", f"{row['cT_dp'][str(g)]:.6f}"] for g in gens]
    tab = ax_tab.table(cellText=table_data, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    ax_tab.axis("off")
    ax_plt.plot(gens, mf_c, marker="o", color="#e8a33d", label="Meanflow warm")
    ax_plt.plot(gens, dp_c, marker="*", color="#6cafe6", label="MF+DP warm")
    # cold trace from FK npz trace
    npz = np.load(next((ROOT / "KPP data/data/kpp2d").glob("*l2sm4p5*.npz")), allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    cold = [(t, c) for t, _, c in meta["trace"]]
    ax_plt.plot([t for t, _ in cold], [c for _, c in cold], marker="s", color="#159e7a", label="cold start")
    ax_plt.axhline(row["c_hat"], color="0.35", linestyle="--", label="long-run FK")
    ax_plt.set_xscale("log")
    ax_plt.set_xlabel("generations")
    ax_plt.set_ylabel(r"$\hat c_T(\lambda)$")
    ax_plt.set_ylim(0.50, 0.83)
    ax_plt.legend(loc="lower right")
    fig.savefig(OUT / "fig7_eigen_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig8(device):
    mf, dp = get_models_3d(device)
    ref = load_ref_3d("kpp3d", -4.5)
    mf_out, dp_out = sample_3d(mf, dp, -4.5, device=device)
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0), layout="constrained")
    panels = [
        ("Reference", ref, [(0,1),(0,2),(1,2)]),
        ("Meanflow", mf_out, [(0,1),(0,2),(1,2)]),
        ("DP refined", dp_out, [(0,1),(0,2),(1,2)]),
    ]
    for r, (row_title, data, planes) in enumerate(panels):
        Hs = [hist2d(data[:, i], data[:, j]) for i, j in planes]
        vmax = max(H.max() for H in Hs)
        for c, ((i, j), H) in enumerate(zip(planes, Hs)):
            ax = axes[r, c]
            show_hist(ax, H, vmax=vmax)
            ax.set_xlabel(["x", "x", "y"][c])
            ax.set_ylabel(["y", "z", "z"][c])
            if c == 0:
                ax.set_title(row_title, loc="left", fontsize=10)
            if c == 2:
                plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
    fig.savefig(OUT / "fig8_projections_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig9(device):
    mf, dp = get_models_3d(device)
    met = json.loads((ROOT / "two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2/metrics.json").read_text())
    rows = met["levels"]
    xs = [-r["log2sigma"] for r in rows]
    mf_w2 = [r["w2_mf"] for r in rows]
    dp_w2 = [r["w2_dp"] for r in rows]
    fig, (ax_tab, ax_plt) = plt.subplots(1, 2, figsize=(10.2, 3.6), width_ratios=[1, 1.4], layout="constrained")
    table_data = [["$-\log_2\sigma$", "Meanflow", "DP refinement"]]
    table_data += [[f"{x:.2f}", f"{a:.4f}", f"{b:.4f}"] for x, a, b in zip(xs, mf_w2, dp_w2)]
    tab = ax_tab.table(cellText=table_data, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    ax_tab.axis("off")
    ax_plt.plot(xs, mf_w2, marker="o", color="#e8a33d", label="Meanflow")
    ax_plt.plot(xs, dp_w2, marker="o", color="#6cafe6", label="DP refinement")
    ax_plt.axvline(3.75, color="green", linestyle="--")
    ax_plt.set_xlabel(r"$-\log_2\sigma$")
    ax_plt.set_ylabel(r"$W_2$")
    ax_plt.set_ylim(0, None)
    ax_plt.legend(loc="upper left")
    fig.savefig(OUT / "fig9_w2_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig10(device):
    mf, dp = get_models_3d(device)
    met = json.loads((ROOT / "two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2/metrics.json").read_text())
    row = next(r for r in met["levels"] if abs(r["log2sigma"] + 4.5) < 1e-9)
    gens = sorted(int(k) for k in row["cT_mf"])
    mf_c = [row["cT_mf"][str(g)] for g in gens]
    dp_c = [row["cT_dp"][str(g)] for g in gens]
    fig, (ax_tab, ax_plt) = plt.subplots(1, 2, figsize=(10.2, 3.6), width_ratios=[1, 1.4], layout="constrained")
    table_data = [["iteration", "MF-only init", "MF+DP init"]]
    table_data += [[str(g), f"{row['cT_mf'][str(g)]:.6f}", f"{row['cT_dp'][str(g)]:.6f}"] for g in gens]
    tab = ax_tab.table(cellText=table_data, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    ax_tab.axis("off")
    ax_plt.plot(gens, mf_c, marker="o", color="#e8a33d", label="Meanflow warm")
    ax_plt.plot(gens, dp_c, marker="*", color="#6cafe6", label="MF+DP warm")
    npz = np.load(next((ROOT / "KPP data/data/kpp3d").glob("*l2sm4p5*.npz")), allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    cold = [(t, c) for t, _, c in meta["trace"]]
    ax_plt.plot([t for t, _ in cold], [c for _, c in cold], marker="s", color="#159e7a", label="cold start")
    ax_plt.axhline(row["c_hat"], color="0.35", linestyle="--", label="long-run FK")
    ax_plt.set_xscale("log")
    ax_plt.set_xlabel("generations")
    ax_plt.set_ylabel(r"$\hat c_T(\lambda)$")
    ax_plt.set_ylim(0.35, 0.65)
    ax_plt.legend(loc="lower right")
    fig.savefig(OUT / "fig10_eigen_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    fig5(device)
    fig6(device)
    fig7(device)
    fig8(device)
    fig9(device)
    fig10(device)
    print("saved", OUT)


if __name__ == "__main__":
    main()
