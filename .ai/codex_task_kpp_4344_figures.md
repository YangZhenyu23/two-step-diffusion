# Task: Generate paper-style figures for KPP Sec. 4.3 and 4.4

## Context

- Repo: `C:\Users\90592\Desktop\HKU\7600\project`
- Source paper figures live on PDF pages 18-22 (rendered copies in `.ai/paper_fig_pages/page_18.png` ... `page_22.png`). Inspect those PNGs before drawing.
- Existing drafts that are **not** paper-faithful and must not be copied as final:
  - `KPP data/viz.py` and `remote_results/figs/` (FK reference only, or paper-warm overlay)
  - `two_step_2d/plot_kpp.py` `plot_final_distributions` (1x3, ~2000 samples, no colorbar, no axis names)
  - `two_step_2d/runs/kpp2d/full_2d/dist_l2sm4p5.png`
  - `two_step_3d/runs/kpp3d/full_3d/dist_l2sm4p5.png`
  - loss curves (`stage1_loss.png`, `stage2_loss.png`) are **not** Sec. 4.3/4.4 figures

Read first:

- `.ai/paper_fig_pages/page_18.png` (Fig. 5)
- `.ai/paper_fig_pages/page_19.png` (Fig. 6)
- `.ai/paper_fig_pages/page_20.png` (Fig. 7)
- `.ai/paper_fig_pages/page_21.png` (Fig. 8)
- `.ai/paper_fig_pages/page_22.png` (Fig. 9 and Fig. 10)
- `two_step_2d/runs/kpp2d/full_2d/metrics.json`
- `two_step_3d/runs/kpp3d/full_3d/metrics.json`
- `two_step_2d/stage1.py` (`generate_mf`)
- `two_step_2d/stage2.py` (`generate_dp`)
- `two_step_2d/models.py`
- `two_step_2d/data_io.py`
- `two_step_3d/` counterparts for 3D
- `KPP data/data/kpp2d/*.npz` and `KPP data/data/kpp3d/*.npz`

## Only modify / create

- `kpp_report/plot_paper_figures.py` (new)
- `kpp_report/figs/` (new pngs listed below)
- `.ai/codex_result_kpp_4344_figures.md`

Do not edit training code, metrics.json, npz files, `midterm_report.tex`, or `KPP data/viz.py`.

## Goal

<task>
Write one script that regenerates the six paper figures of Shen et al. Sec. 4.3 and 4.4, using OUR FK clouds and OUR trained maps, with the same panel geometry, axes, colormap, and table+plot composites as the PDF.

Done looks like these files exist and match the layouts below:

4.3
- `kpp_report/figs/fig5_invariant_2d.png`
- `kpp_report/figs/fig6_w2_2d.png`
- `kpp_report/figs/fig7_eigen_2d.png`

4.4
- `kpp_report/figs/fig8_projections_3d.png`
- `kpp_report/figs/fig9_w2_3d.png`
- `kpp_report/figs/fig10_eigen_3d.png`

Do not emit sweep grids, speed-vs-sigma, or training-loss plots.
</task>

<completeness_contract>
Sample 20000 particles per panel (paper M), not 2000. Generate MF and DP clouds from the listed checkpoints. If CUDA is missing, use CPU. If a checkpoint fails to load, stop and report; do not silently drop a column.
</completeness_contract>

<verification_loop>
Open each saved PNG (or at least check width/height and that files are > 50 kB). Confirm Fig. 5 is 2x3, Fig. 8 is 3x3, Fig. 6/9 have a table on the left, Fig. 7/10 have log-x eigenvalue curves plus a left table. If layout is 1x3, redo.
</verification_loop>

<grounding_rules>
W2 and c_T numbers in tables must come from the two metrics.json files above, rounded to four decimals. Do not reuse the source paper's W2 table as our table. Cold-start c_T traces come from the FK npz `trace` field at log2sigma=-4.5. Long-run dashed line is that file's `c_hat`.
</grounding_rules>

<action_safety>
Do not retrain. Do not overwrite npz. Do not change model code except importing generate_mf / generate_dp.
</action_safety>

<default_follow_through_policy>
If matplotlib serif Computer Modern is unavailable, use DejaVu Serif. Keep going.
</default_follow_through_policy>

## Data sources (do not mix backup runs)

| Role | Path |
|---|---|
| 2D FK clouds | `KPP data/data/kpp2d/ref_l2s*_re256.npz` keys `qT`, `q0`, `meta` |
| 3D FK clouds | `KPP data/data/kpp3d/ref_l2s*_re256_s1m1.npz` keys `xT`, `x0`, `meta` |
| 2D metrics + cT | `two_step_2d/runs/kpp2d/full_2d/metrics.json` |
| 3D metrics + cT | `two_step_3d/runs/kpp3d/full_3d/metrics.json` |
| 2D weights | `two_step_2d/runs/kpp2d/full_2d/stage1_final.pt`, `stage2_final.pt` |
| 3D weights | `two_step_3d/runs/kpp3d/full_3d/stage1_final.pt`, `stage2_final.pt` |
| 2D/3D configs | sibling `config.json` (prior_std, wrap, alpha, depth) |

Sampling: Gaussian prior `N(0, prior_std I)` with `prior_std` from config (pi). Same seed 666 for all generated clouds. Wrap outputs onto the torus the same way `generate_mf` / `generate_dp` already do.

## Global style

```
matplotlib.rcParams.update({
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
```

Colors (match existing project, close to the paper):

| series | color | marker |
|---|---|---|
| Meanflow | `#e8a33d` | `o` |
| MF+DP | `#6cafe6` | `o` or `*` on eigenvalue plots |
| cold start | `#159e7a` | `s` |
| long-run FK | `0.35` | dashed hline, no marker |
| train-range end | green | dashed vline at x=3.75 |

Save `dpi=200`, `bbox_inches="tight"`. White background. No seaborn themes.

Histograms:

- `numpy.histogram2d` **counts** (not density), `bins=72`, `range=[[0, 2pi], [0, 2pi]]`
- display with `imshow(..., origin="lower", extent=[0, 2pi, 0, 2pi], cmap="viridis", interpolation="nearest", aspect="equal")`
- ticks `[0, 1, 2, 3, 4, 5, 6]` on both axes (paper style), xlim/ylim `[0, 2pi]`
- axis labels: `x`, `y`, or `z` as appropriate
- never use hexbin

---

## Figure 5 — 2D invariant measure (Sec. 4.3)

Paper layout (page 18): **2 rows x 3 columns**.

| | col 0 Meanflow | col 1 DP refined | col 2 Reference |
|---|---|---|---|
| row 0 | σ = 2^{-3.75} (train) | same | FK `qT` |
| row 1 | σ = 2^{-4} (extrap) | same | FK `qT` |

- Each panel has **its own** vertical colorbar.
- Colorbar integer ticks. Paper tops: train row ~40, extrap row ~35. Ours will differ; do not force vmax to the paper numbers. Use each panel's own max (per-panel colorbar).
- Column titles: `Meanflow`, `DP refined`, `Reference` (or `FK reference`).
- Row labels either as y-axis titles or as `$\sigma=2^{-3.75}$` / `$\sigma=2^{-4}$` above the row.
- Domain $\mathbb{T}^2=[0,2\pi)^2$. Content is a wavy ridge; do not rotate axes.
- figsize about `(10.5, 6.4)`.

## Figure 6 — 2D W2 (Sec. 4.3)

Paper layout (page 19): **left table + right line plot**, one figure.

Left table (booktabs look, matplotlib `table` is OK):

| $-\log_2\sigma$ | Meanflow | DP refinement |
| 2.00 ... 4.50 step 0.25 | w2_mf 4dp | w2_dp 4dp |

Do **not** bold a column by default (paper bolded DP because it always won; our DP does not always win). No extra styling that implies uniform improvement.

Right plot:

- x: $-\log_2\sigma$ from 2.0 to 4.5
- y: $W_2$, linear, start at 0. Include cold start as a third optional series only if it still fits; paper does **not** plot cold. Default: **two series**, Meanflow and DP, like the paper.
- green dashed vertical line at `x=3.75`
- legend inside, upper left or where it does not cover curves
- light dashed grid
- figsize about `(10.2, 3.6)` with width ratio table:plot ≈ 1:1.4

## Figure 7 — 2D eigenvalue (Sec. 4.3)

Paper layout (page 20): **left table + right log-x plot**. All at **σ = 2^{-4.5}**.

Left table:

| iteration | MF-only init | MF+DP init |
| 1,2,4,...,128 | cT_mf | cT_dp |

from `metrics.json` (six decimals like the paper, or four if space is tight).

Right plot four series:

1. warm start (MF-only) — our `cT_mf`
2. warm start (MF+DP) — our `cT_dp`
3. cold start (uniform init) — FK npz `trace` at log2sigma=-4.5, the `c_T` column
4. long-run FK reference — horizontal dashed at `c_hat` (0.7842)

- x: `generations`, **log scale**, ticks 10^0, 10^1, 10^2
- y: `eigval` (paper ylabel) or $\hat c_T(\lambda)$. Paper 2D y-range ≈ 0.50 to 0.80. Set ylim from min(cold T=1, 0.50) to 0.83 so our MF/DP (near 0.78-0.82) remain visible.
- legend lower right
- cold start uses the full dyadic trace in the npz (includes 256, 512, ... 4096 if present). Warm starts only go to 128 in metrics.json. That is OK: plot each series over the T it has.

## Figure 8 — 3D projections (Sec. 4.4)

Paper layout (page 21): **3 rows x 3 columns**, σ = 2^{-4.5} only.

| | x-y | x-z | y-z |
|---|---|---|---|
| row 0 Reference | (x,y) | (x,z) | (y,z) |
| row 1 Meanflow | | | |
| row 2 DP refined | | | |

- **Shared colorbar per row** (one bar on the right of each row), vmax = that row's max count. Paper bars go 0-35; ours follow data.
- Axis labels: bottom `x`/`y`, left `y`/`z` matching the plane.
- Ticks 0,1,2,3,4,5,6. Square panels.
- Row titles on the left: `Reference`, `Meanflow`, `DP refined`.
- 3D cloud: FK `xT`; MF/DP from 3D checkpoints, 20000 samples.
- figsize about `(10.2, 9.0)`.

## Figure 9 — 3D W2 on x-z (Sec. 4.4)

Same composite as Fig. 6. Numbers from 3D `metrics.json` (`w2_mf`, `w2_dp` are already the x-z projected EMD).

- y-axis label: $W_2$
- green dashed line at 3.75
- paper 3D y-range was 0-0.16; **do not clip**. Our values are ~0.11-0.23, so autoscale with ymin=0.

## Figure 10 — 3D eigenvalue (Sec. 4.4)

Same composite as Fig. 7, σ = 2^{-4.5}.

- long-run line: 3D npz `c_hat` (~0.6147)
- paper y-range ≈ 0.35-0.65. Use that band if cold T=1 sits near 0.36; otherwise expand slightly so all series fit.
- cold start from 3D npz trace; warm from 3D metrics `cT_mf` / `cT_dp`.

## What NOT to draw

- 11-level sweep montages
- front-speed vs σ (`speed.png`)
- training loss
- 1x3 "Reference / MeanFlow / MF+DP" drafts
- source-paper W2 numbers as our curves

## Implementation notes

- Reuse `generate_mf` / `generate_dp`; do not reimplement sampling.
- Load models with the same architecture as `runner.py` / `models.py` (2D vs 3D dim).
- `eval()` + `no_grad` for both stages. Stage I frozen when generating DP.
- Histogram helper must wrap samples into [0, 2pi) before binning.
- Script argparse: `--device cuda|cpu`, default cuda if available.
- Running the script from repo root must write the six pngs and print each path.

## Acceptance

1. `python kpp_report/plot_paper_figures.py` exits 0.
2. Six pngs listed above exist, each > 50 kB.
3. Fig. 5 image is 2 rows x 3 cols (not 1x3).
4. Fig. 8 image is 3 rows x 3 cols.
5. Fig. 6/9 contain a numeric table whose Meanflow/DP entries match metrics.json to 4 decimals.
6. Fig. 7/10 x-axis is logarithmic.
7. `.ai/codex_result_kpp_4344_figures.md` lists files, sample count, device, and any checkpoint load warnings.

<structured_output_contract>
Write `.ai/codex_result_kpp_4344_figures.md` with: files, pixel sizes, vmax of Fig.5/Fig.8 panels, and whether CUDA was used.
</structured_output_contract>
