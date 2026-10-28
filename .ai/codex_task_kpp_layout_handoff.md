# Task: Fine-tune KPP 4.3/4.4 draft — subsections, figure layout, replacement handoff

给操作者（中文，读完即可转交 Codex）：

把本文件整份交给 Codex，提示词用：

```
Read .ai/codex_task_kpp_layout_handoff.md and execute the task. Do not start a new manuscript. Edit the existing draft and write the handoff doc.
```

正确路径（上一轮不要再用截断的 PDF 文件名去猜原文）：

| 角色 | 路径 |
|---|---|
| **要微调的初稿** | `kpp_report/kpp_sections_4_3_4_4.tex` |
| **已生成、要插入的六张图** | `kpp_report/figs/fig5_invariant_2d.png` 等，见下文锁定表 |
| **原文 4.3/4.4 结构与图版式** | `.ai/paper_fig_pages/page_18.png` … `page_22.png`，文字 `.ai/paper_fig_pages_clean.txt` |
| **我们的数字** | `two_step_2d/runs/kpp2d/full_2d/metrics.json`，`two_step_3d/runs/kpp3d/full_3d/metrics.json` |

不要改 `midterm_report.tex`。不要把 `DP/Wang 等 - 2022 - DeepParticle....pdf` 当成 Two-Step 原文。原文 PDF 若要核对，用 PowerShell：

```powershell
Get-ChildItem -LiteralPath 'C:\Users\90592\Desktop\HKU\7600\project\two-step diffusion KPP'
```

文件夹名含空格，必须 `-LiteralPath`。找不到 PDF 就用 `.ai/paper_fig_pages/`，不要停。

---

## Context

- Repo: `C:\Users\90592\Desktop\HKU\7600\project`
- Judgment-sensitive layout + writing task. One Codex run. Three deliverables, not a rewrite of the science.

Read first, in this order:

1. This brief.
2. `kpp_report/kpp_sections_4_3_4_4.tex` (the preliminary draft to fine-tune).
3. `.ai/paper_fig_pages/page_18.png` (source Fig. 5, 2×3).
4. `.ai/paper_fig_pages/page_19.png` (source Fig. 6, table + \(W_2\) curve).
5. `.ai/paper_fig_pages/page_20.png` (source Fig. 7, table + eigenvalue).
6. `.ai/paper_fig_pages/page_21.png` (source Fig. 8, 3×3 projections).
7. `.ai/paper_fig_pages/page_22.png` (source Fig. 9 and Fig. 10).
8. `.paper/style_overrides.md`
9. `two_step_2d/runs/kpp2d/full_2d/metrics.json`
10. `two_step_3d/runs/kpp3d/full_3d/metrics.json`

Optional: `.ai/paper_fig_pages_clean.txt` for source-section order. Do not copy its OCR errors into the TeX.

The six paper-style figures already exist. Do **not** rerun training. Do **not** redraw unless a listed PNG is missing.

## Only modify / create

Allowed:

- `kpp_report/kpp_sections_4_3_4_4.tex`
- `kpp_report/FIGURE_HANDOFF.md` (required; Chinese; replacement guide)
- `kpp_report/README.md` (short compile + replace instructions)
- `kpp_report/extract_paper_numbers.py` (optional helper that prints 4-decimal tokens from the two `metrics.json` files)
- compile byproducts next to the tex (`pdf`, `aux`, `log`, …)
- `.ai/codex_result_kpp_layout_handoff.md`

Do **not** edit:

- `midterm_report.tex`
- `kpp_report/plot_paper_figures.py`
- `two_step_2d/`, `two_step_3d/`, `KPP data/`
- any `metrics.json` or `.npz`
- `.paper/` except you may leave it alone
- the six locked PNG filenames (overwrite only if a file is missing and you must copy; do not rename)

## Goal

<task>
Fine-tune the existing English LaTeX excerpt `kpp_report/kpp_sections_4_3_4_4.tex` so that:

1. 2D and 3D are split into subsections, matching the source paper’s 4.3 / 4.4 skeleton.
2. The six already-generated PNGs are inserted at the matching scientific location, with floats that stay with their subsection (no 2D figure in 4.4, no Fig. 9 after the 4.4.4 heading).
3. A Chinese handoff document exists so a later, better run can replace the PNGs without hunting through the TeX.

Done looks like:

- `kpp_report/kpp_sections_4_3_4_4.pdf` compiles with exit 0.
- Section 4.3 has four subsubsections: Setting and protocol; Invariant measure; Torus Wasserstein Error; Eigenvalue estimator.
- Section 4.4 has four subsubsections: Setting and protocol; Projected invariant measure; Projected Wasserstein Error; Eigenvalue estimator.
- PDF figure numbers are 5–10 (`\setcounter{figure}{4}` before the first figure), matching the source paper and the file names.
- Each of the six locked files is included exactly once, via the current `\includegraphics` basename (do not change those basenames).
- `kpp_report/FIGURE_HANDOFF.md` lists locked names, which decimals in the tex must be updated after a new run, which source-paper decimals must not be overwritten, and the exact regenerate/compile commands.
</task>

<completeness_contract>
Do not stop after editing the tex. Compile, fix errors, compile again for refs. Open or render the PDF and check figure placement. Write the handoff doc. Write `.ai/codex_result_kpp_layout_handoff.md`.
</completeness_contract>

<verification_loop>
After compile:

1. Confirm the PDF has two `\subsection`s equivalent to 4.3 then 4.4, each with four `\subsubsection`s.
2. Confirm include order: fig5 → fig6 → fig7, then `\FloatBarrier` (and `\clearpage` if needed), then fig8 → fig9 → fig10.
3. Confirm no include of `fig5_ref.png`, `fig8_ref.png`, `fig7_cT_2d_fk.png`, `fig10_cT_3d_fk.png`, loss curves, or `dist_l2sm4p5.png`.
4. Confirm in-text 4-decimal \(W_2\) / \(\hat c\) tokens match the two current `metrics.json` files, not `.paper/claim_evidence_ledger.md` (that ledger is stale).
5. If Fig. 9 sits after the 4.4.4 heading, switch those figures to `[H]` + `\FloatBarrier` and recompile. A half-page gap before the tall 3×3 Fig. 8 is acceptable; swapping 4.4.3 text in front of Fig. 8 is not.
</verification_loop>

<grounding_rules>
Our numbers come only from:

- `two_step_2d/runs/kpp2d/full_2d/metrics.json`
- `two_step_3d/runs/kpp3d/full_3d/metrics.json`

Round \(W_2\) and \(\hat c\) in body prose to four decimals. Do not use `runs_backup_20260829/`. Do not use `midterm_report.tex` tables.

Source-paper comparison constants (do not overwrite with our run):

- 2D Meanflow mean `0.0769`; 2D DP at \(-\log_2\sigma=4.50\) `0.0456`; 2D DP warm start \(T=1\) `0.7616`
- 3D Meanflow mean `0.1312`; 3D DP mean `0.0294`; 3D DP warm start \(T=1\) `0.6135`

If the current tex already has subsections and the six includes, keep the scientific prose. Fine-tune structure, floats, captions, and the handoff. Do not rewrite 4.3/4.4 as a technical report.
</grounding_rules>

<missing_context_gating>
If a locked PNG is missing, stop and name the path; do not substitute a `*_ref.png`. If `metrics.json` is missing, stop. If the source PDF filename is unknown, use `.ai/paper_fig_pages/` and continue.
</missing_context_gating>

<action_safety>
Do not retrain. Do not rename the six PNG files. Do not change `\label{fig:2d-inv}` … `\label{fig:3d-eig}`. Do not add KS sections, abstract, or a global conclusion. Do not invent a new tex filename.
</action_safety>

<default_follow_through_policy>
Default to `[H]` + `float` + `placeins` `\FloatBarrier` after each figure. Hide hyperref boxes with `\hypersetup{hidelinks}`. Keep going if the source PDF is missing. English body; Chinese handoff.
</default_follow_through_policy>

<structured_output_contract>
Write `.ai/codex_result_kpp_layout_handoff.md` with: files changed, compile command and exit code, PDF page count, figure-to-subsection map, and any leftover layout gap (e.g. whitespace before Fig. 8).
</structured_output_contract>

## Locked figure files (do not rename)

| PDF number | Subsection | `\label` | `\includegraphics{...}` |
|---|---|---|---|
| Fig. 5 | 4.3 Invariant measure | `fig:2d-inv` | `fig5_invariant_2d.png` |
| Fig. 6 | 4.3 Torus Wasserstein Error | `fig:2d-w2` | `fig6_w2_2d.png` |
| Fig. 7 | 4.3 Eigenvalue estimator | `fig:2d-eig` | `fig7_eigen_2d.png` |
| Fig. 8 | 4.4 Projected invariant measure | `fig:3d-inv` | `fig8_projections_3d.png` |
| Fig. 9 | 4.4 Projected Wasserstein Error | `fig:3d-w2` | `fig9_w2_3d.png` |
| Fig. 10 | 4.4 Eigenvalue estimator | `fig:3d-eig` | `fig10_eigen_3d.png` |

`\graphicspath{{figs/}}` stays. Width: Fig. 5/6/7/9/10 `\textwidth`; Fig. 8 may be `0.85`–`0.92\textwidth` because it is a tall 3×3.

## Required subsection skeleton

Keep `\setcounter{section}{4}` and `\setcounter{subsection}{2}` so the first heading is 4.3.

```tex
\subsection{KPP Front Speed Experiment on the Two-Torus}
  \subsubsection{Setting and protocol}
  \subsubsection{Invariant measure}              % Fig. 5 immediately after the finding paragraph
  \subsubsection{Torus Wasserstein Error}       % Fig. 6
  \subsubsection{Eigenvalue estimator}          % Fig. 7
\FloatBarrier
\clearpage
\subsection{3D KPP Front Speed Experiment with Time-Dependent Kolmogorov Flow}
  \subsubsection{Setting and protocol}
  \subsubsection{Projected invariant measure}   % Fig. 8
  \subsubsection{Projected Wasserstein Error}   % Fig. 9
  \subsubsection{Eigenvalue estimator}          % Fig. 10
```

Finding-first: the result paragraph comes first, then the figure. Do not open a result paragraph with “Figure X shows”.

Subsection titles: no math tokens (hyperref bookmarks). Use “Torus Wasserstein Error” and “Projected Wasserstein Error”, not a title that is only \(W_2\).

## Layout rules

- Package `float` and `placeins`. Every figure `[H]`. `\FloatBarrier` after each figure.
- `\clearpage` after the last 2D figure, before 4.4.
- Fig. 8 is allowed to start a new page. Do not pull 4.4.3 text onto that empty space.
- Captions: self-contained; name Meanflow / DP / Feynman–Kac; name \(\sigma\) or \(-\log_2\sigma\); do not claim a panel is reference-only if it is a three-way comparison.

## Current metrics (check against json; update tex if it drifted)

Use 4 decimals.

**2D** (`full_2d/metrics.json`):

- cold \(W_2\) range `1.0538`–`1.1718`
- Meanflow grid mean `0.2027` (source paper `0.0769`)
- \(-\log_2\sigma=2.00\): Meanflow `0.2322`, DP `0.1388`
- \(-\log_2\sigma=4.50\): Meanflow `0.2228`, DP `0.2644` (source paper DP `0.0456`)
- \(\sigma=2^{-4.5}\), \(\hat c=0.7842\)
- Meanflow \(T=1/128\): `0.7968` / `0.7843`
- DP \(T=1/128\): `0.8202` / `0.7845`

**3D** (`full_3d/metrics.json`):

- cold `0.8895`–`0.9600`
- Meanflow mean `0.1902` (source `0.1312`); DP mean `0.1578` (source `0.0294`)
- DP minimum `0.1079` at \(-\log_2\sigma=2.75\)
- \(-\log_2\sigma=4.50\): Meanflow `0.2220`, DP `0.2254`
- \(\hat c=0.6147\)
- Meanflow \(T=1/128\): `0.6029` / `0.6144`
- DP \(T=1/128\): `0.5888` / `0.6142`

Cold-start \(\hat c_T\) at \(T=1\) (~`0.51` in 2D, ~`0.36` in 3D) comes from the FK npz trace, not from `metrics.json`. Keep those as approximate if you do not re-read the npz.

Qualitative claims that must stay consistent with the current (not-yet-good) figures:

- 2D Meanflow ridge is thinner than the FK band; 2D DP is scattered clusters, not a continuous ridge.
- 3D Meanflow projections are box-like; 3D DP recovers topology but remains noisier than FK.
- Eigenvalue warm starts sit near the long-run FK value; that is **not** a claim that the invariant measure has been reproduced. \(W_2\) is still \(O(10^{-1})\).

If you change qualitative sentences, they must still match the six PNGs on disk.

## Handoff document (`kpp_report/FIGURE_HANDOFF.md`)

Write in Simplified Chinese. Future agents will replace PNGs after a better training run. The doc must contain:

1. **Locked filename table** (the six rows above). Instruction: replace by overwriting the same basename; never change `\includegraphics` or `\label`.
2. **Regenerate command** (project root): `python kpp_report/plot_paper_figures.py`
3. **Number-extract command** (optional if you add the helper): `python kpp_report/extract_paper_numbers.py`
4. **Compile command**: from `kpp_report/`, `latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex`
5. **Which tex decimals to update** after a new run (the lists in “Current metrics”), and which source-paper decimals must not change.
6. **Warning**: Fig. 6/7/9/10 embed their left-hand tables inside the PNG. Updating the tex without redrawing those PNGs makes table and prose disagree.
7. **Do not insert**: `fig5_ref.png`, `fig8_ref.png`, `fig7_cT_2d_fk.png`, `fig10_cT_3d_fk.png`.
8. **Do not mix** `runs_backup_20260829/`.
9. A short checklist: overwrite six PNGs → update decimals → rewrite 4.3.2 / 4.4.2 geometry sentences if the heatmaps change → compile → confirm Figs. 5–7 stay in 4.3 and 8–10 in 4.4.

`kpp_report/README.md` should point to that file and give the compile command. Keep it short.

## Genre (do not regress)

English Results excerpt, not a lab report. Forbidden in body prose: file paths, `metrics.json`, “we implemented”, “update interface”, “Furthermore”, “Moreover”, “utilize”, “leverage”. No training-loss discussion.

## Compile

```powershell
cd C:\Users\90592\Desktop\HKU\7600\project\kpp_report
latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex
```

## Acceptance

1. `kpp_sections_4_3_4_4.tex` still UTF-8, still one excerpt file.
2. Compile exits 0. PDF exists, typically 5–8 pages.
3. Body has 4.3 and 4.4 as subsections, each with four subsubsections.
4. Exactly the six locked `\includegraphics` names; none of the `*_ref` / `*_fk` leftovers.
5. Figure counter yields Figure 5 through Figure 10.
6. `FIGURE_HANDOFF.md` exists and names all six files plus the two `metrics.json` paths.
7. Result summary at `.ai/codex_result_kpp_layout_handoff.md`.
