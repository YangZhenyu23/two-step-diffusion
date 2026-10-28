# Task: Write and compile KPP Sections 4.3–4.4

## Context

- Repo: `C:\Users\90592\Desktop\HKU\7600\project`
- This is a judgment-sensitive writing task. Read these files first, in this order:
  1. `.paper/context.md`
  2. `.paper/style_overrides.md`
  3. `.paper/claim_evidence_ledger.md`
  4. `.paper/figure_inventory.md`
  5. `.paper/journal_format.md`
  6. This brief.
- Optional reference (do not copy its prose): `midterm_report.tex` is a technical-report draft. Treat it as a negative example.
- Source paper: Shen, Wang, Xin, Zhang, "Two-Step Diffusion...", arXiv:2601.20024v1, Sections 4.3–4.4. Local PDF:
  `two-step diffusion KPP/Shen 等 - 2026 - Two-Step Diffusion Fast Sampling and Reliable Prediction for 3D Keller--Segel and KPP Equations in.pdf`
- Do not reread the whole PDF if the brief and `.paper/` already supply the structure and numbers. If a symbol is missing, read only those two sections.

## Only modify / create

Allowed write paths:

- `kpp_report/` (new directory)
- `kpp_report/kpp_sections_4_3_4_4.tex`
- `kpp_report/refs.bib` (optional, small)
- `kpp_report/figs/` (copies of existing pngs listed in `.paper/figure_inventory.md`)
- `kpp_report/README.md` (one short paragraph: how to compile)
- `.ai/codex_result_kpp_4344_report.md` (required result summary)

Do not edit: `midterm_report.tex`, any Python, any `metrics.json`, `.paper/` (already written), `two_step_2d/`, `two_step_3d/`, `KPP data/`.

## Goal

<task>
Write a standalone English LaTeX excerpt that contains ONLY the scientific content of the source paper's Sections 4.3 and 4.4, rewritten as a group-paper drop-in for a KPP reproduction. Compile it to PDF. The prose must read as academic Results, not as a technical report.

Done looks like:
1. `kpp_report/kpp_sections_4_3_4_4.tex` exists and compiles with zero errors.
2. `kpp_report/kpp_sections_4_3_4_4.pdf` exists.
3. The PDF contains two numbered subsections equivalent to 4.3 and 4.4, with equations, protocol, figures, tables, and findings-first result paragraphs.
4. Every numeric token in the body is in `.paper/claim_evidence_ledger.md` or in a printed table/figure in the same excerpt.
5. `.ai/codex_result_kpp_4344_report.md` lists files written, compile command, page count, and residual risks.
</task>

<structured_output_contract>
After finishing, write `.ai/codex_result_kpp_4344_report.md` with:
1. files created
2. compile command and exit code
3. page count
4. which figures were included
5. any number you wanted to use but could not trace (must be none)
6. residual risks
</structured_output_contract>

<completeness_contract>
Do not stop after drafting TeX. Copy figures, compile, fix LaTeX errors, recompile. If a figure path breaks, copy the png into `kpp_report/figs/` and fix the path. If pdflatex/latexmk is missing, install is not required: state the exact missing binary and still leave a compilable TeX file.
</completeness_contract>

<verification_loop>
Before stopping:
- grep the .tex for banned words listed in Constraints
- grep every numeric token in body prose against the ledger
- confirm no 3D DP W2 is presented as a finding
- compile twice if using labels
- open-ended check: the first sentence of each result paragraph is a finding, not "Figure X shows"
</verification_loop>

<grounding_rules>
Ground every empirical sentence in the ledger or a figure listed in the inventory. Label comparisons with the source paper as comparisons, not as our measurements. If a newer `two_step_3d/runs/kpp3d/full_3d/metrics.json` exists, you MAY replace the 3D table with it; otherwise do not invent 3D DP numbers.
</grounding_rules>

<missing_context_gating>
Do not guess missing W2 values, seeds, or histogram peaks. If a figure is reference-only, say so in the caption. If 3D Stage II metrics are absent, omit the DP column.
</missing_context_gating>

<action_safety>
Do not modify existing experiment code or metrics. Do not overwrite `midterm_report.tex`. Do not add KS sections, abstract, introduction, or a global conclusion.
</action_safety>

<default_follow_through_policy>
If a minor LaTeX package is missing, switch to a package already available (amsmath, booktabs, graphicx, geometry, hyperref). Do not stall asking whether to use English; English is required.
</default_follow_through_policy>

## Constraints

### Genre (non-negotiable)

This is NOT a technical report. Forbidden patterns:

- "In this section we implement / we code / we run"
- file paths, class names, `metrics.json`, seed numbers in body prose
- "update interface", "preliminary results", "replace these rows"
- bullet lists of hyperparameters
- Methods restated as a lab log
- AI wrap-up: "In conclusion", "Taken together", "It is worth noting"

Write the way the source paper writes 4.3/4.4: PDE, variational formula, particle system, protocol in running prose, then figures/tables, then findings-first interpretation.

### Academic writing rules

- Findings first, then one mechanism sentence, then a handoff to the next paragraph.
- Open each subsection with one sentence that states what the subsection does and why it is needed.
- Average sentence length near 20–30 words. Vary rhythm. No mechanical rule-of-three.
- At most one colon and one semicolon per body paragraph.
- No invented hyphenated compounds.
- Hedging for inferred simulation findings. Direct language for protocol and table values.
- Panel-specific figure citations.
- Do not start result paragraphs with "Figure X shows".
- Banned words/phrases: utilize, leverage, robust, novel, comprehensive, significant (unless a statistical test), furthermore, moreover, additionally, notable/notably, crucial/crucially, demonstrates, confirms, proves, establishes, sheds light, in this regard, as mentioned above, in conclusion, to summarize, it is worth noting, plays a crucial role.
- "this/these/that" need clear antecedents.
- Do not praise the method in the last sentence. Last sentence of 4.4 may bound the claim (2D DP did not uniformly improve W2; 3D DP not reported) and point to the scientific quantity (front speed / invariant measure).

### Scope

- Only 4.3 and 4.4. A 4–6 line unnumbered preamble is allowed so the excerpt can stand alone: name the source paper, state that the group reproduced the KPP experiments, and that Stage I/II were defined earlier in the group manuscript. Do not re-derive MeanFlow identity or DP OT algorithms.
- Keep Stage I/II to one sentence each when they first appear in 4.3 protocol ("a one-step Meanflow map ... a near-identity Deep Particle corrector ...").

### Numbering

Use:

```tex
\setcounter{section}{4}
\setcounter{subsection}{2}
\subsection{KPP Front Speed Experiment}
...
\subsection{3D KPP Front Speed Experiment with Time-Dependent Kolmogorov Flow}
```

Equation numbers may restart or continue; prefer continuing from (17) style by using `\numberwithin{equation}{section}` and setting the first KPP equation to appear as (17) only if easy. If not easy, number locally and do not fake source-paper numbers. In prose, refer to "the twisted diffusion" not "Equation (20) of the source paper" except when comparing.

### Data (do not mix sources)

Use `.paper/claim_evidence_ledger.md` as the only number source.

- 2D table columns: $-\log_2\sigma$, cold $W_2$, Meanflow $W_2$, DP $W_2$, and optionally the source-paper MF/DP as extra columns or a second table clearly labeled "source paper".
- 3D table columns: $-\log_2\sigma$, cold $W_2$, Meanflow $W_2$, and source-paper MF/DP as a comparison table. Do not print our 3D DP $W_2$ as a result.
- Eigenvalue: report our long-run $\hat c=0.7842$ (2D) and $0.6148$ (3D) at $\sigma=2^{-4.5}$. Cold-start curves: use `fig7_2d.png` / `fig10_3d.png` and say the warm-start markers in those figures are the source paper's tabulated values. Our own MF warm-start $\hat c_T$ may be tabulated from the ledger (2D only). For 3D, report Meanflow $c_T$ from the ledger; omit DP $c_T$.

### Figures

Copy pngs into `kpp_report/figs/` with short names. Captions must be self-contained and must not claim a panel shows Meanflow or DP if it is a FK reference.

Recommended main figures:

4.3:
- FK reference invariant measure (fig5_2d.png)
- Our 2D $W_2$ vs $-\log_2\sigma$ (w2_curve.png from kpp2d) plus a table
- Eigenvalue convergence (prefer fig7_2d.png, because it includes cold start; mention paper warm starts in the caption)

4.4:
- FK reference projections (fig8_3d.png)
- 3D $W_2$ table (our MF + cold; paper MF/DP comparison)
- Eigenvalue convergence (fig10_3d.png)

Optional: `speed.png` once, with a caption that these are FK long-horizon estimates, not learned-map estimates.

Do not include training-loss plots. Those are technical-report figures.

### Paragraph-by-paragraph outline (follow this order)

**Preamble (unnumbered, ≤6 lines)**
Group reproduction of Shen et al. KPP experiments. Two-stage map defined earlier. This excerpt reports 2D cellular flow and 3D time-dependent Kolmogorov flow.

**4.3 paragraph A — why this experiment**
Front speed in periodic flows. KPP reaction-diffusion-advection. Introduce (17) with $\kappa$, $v$, $r$, $u$.

**4.3 paragraph B — variational formula**
If $v$ is periodic in space and time, $c^*(e)=\inf_{\lambda>0}\mu(\lambda)/\lambda$. Principal eigenvalue of $\partial_t-\mathcal{A}$. Display the twisted operator.

**4.3 paragraph C — particle representation**
Twisted SDE on $\mathbb{T}^2$. Cellular velocity field with $\theta$. Potential $V_\lambda$. Euler–Maruyama, multiplicative weights, resampling, running estimators $\hat\mu_T$, $\hat c_T$. Long-horizon FK cloud approximates $\eta_\lambda^\star$.

**4.3 paragraph D — protocol**
$\mathbb{T}^2=[0,2\pi)^2$, $(\lambda,\theta)=(2.0,1.0)$, $r_n=256$, $M=2\times10^4$, $T=4096$, cold start uniform. $\sigma=\kappa$. Training grid (25), test grid (26). Warm starts: one-step Meanflow from a Gaussian source with standard deviation $\approx\pi$; DP is a 6-block residual corrector on the torus, both conditioned on $\sigma$. Two metrics: eigenvalue vs $T$, and torus $W_2$ on a $72\times72$ histogram with periodic cost.

**4.3 paragraph E — qualitative finding (Fig. 5 analog)**
Finding first: the FK invariant measure concentrates on an anisotropic ridge aligned with the cellular flow; the ridge tightens as $\sigma$ decreases. Cite the reference figure. Mechanism: smaller $\kappa$ reduces Brownian spreading in the twisted diffusion, so mass tracks the advective skeleton more closely. Handoff: whether a learned one-step map recovers that ridge is measured by $W_2$ next. Do not claim our Meanflow histograms look like the paper's Fig. 5 unless such images exist (they do not in the inventory).

**4.3 paragraph F — $W_2$ finding (Fig. 6 analog)**
Finding first: uniform initialization remains at $W_2\approx 1.05$–$1.17$; Meanflow lowers the error to $0.20$–$0.28$ across the eleven levels. Relative to the source paper's Meanflow mean $0.077$, the reproduced Meanflow mean is $0.235$ (about threefold). On this run Deep Particle does not uniformly improve on Meanflow; the two grid means are $0.2354$ and $0.2359$. At the extrapolation point $-\log_2\sigma=4.50$, Meanflow is $0.2786$ and DP is $0.2647$, still far from the source paper's $0.0456$. Mechanism (one sentence): the one-step map moves mass onto the correct torus scale, but residual ridge misalignment remains large enough that a near-identity corrector cannot close the gap to the FK cloud. Print the table.

**4.3 paragraph G — eigenvalue finding (Fig. 7 analog)**
Finding first: at $\sigma=2^{-4.5}$ the long-horizon FK estimator is $\hat c=0.7842$. A uniform cold start begins well below that value and approaches it over $O(10^2)$ generations (use fig7_2d.png). The source paper's MF+DP warm start is already $0.7616$ at $T=1$. Our Meanflow warm start at $T=1$ is $0.8289$ and is within $0.001$ of the long-run value by $T=128$. Mechanism: $\hat c_T$ averages log-weights and is less sensitive to ridge shape than $W_2$, so eigenvalue agreement can coexist with a large histogram error. Close 4.3 by saying the 3D Kolmogorov flow tests the same estimators under a divergence-free, fully three-dimensional advection.

**4.4 paragraph A — PDE and flow**
Same KPP equation on $\mathbb{T}^3$. Display the time-dependent Kolmogorov field. Note that it is divergence-free for all $t$. Setup parallels 4.3: $\Delta t=2^{-8}$, $M=2\times10^4$, $\lambda=2$, training to $-\log_2\sigma=3.75$, evaluation to $4.5$.

**4.4 paragraph B — metrics**
Same two metrics. Invariant-measure error is $W_2$ of $x$-$z$ projections (display the projection formula). Cold start is uniform on $\mathbb{T}^3$.

**4.4 paragraph C — qualitative finding (Fig. 8 analog)**
Finding first: the FK cloud at $\sigma=2^{-4.5}$ is anisotropic across the three coordinate planes, with mass concentrated at larger $z$ in the $x$-$z$ and $y$-$z$ projections and a wavy ridge in $y$-$z$. Cite fig8_3d.png as the reference. Do not attribute those panels to Meanflow or DP. Mechanism: the potential $V_\lambda$ depends on $x_3$ through the first velocity component, so the invariant measure is not uniform in $z$. Handoff: projected $W_2$ quantifies whether Meanflow recovers this geometry.

**4.4 paragraph D — $W_2$ finding (Fig. 9 analog)**
Finding first: 3D Meanflow $W_2$ ranges from $0.121$ to $0.198$, with grid mean $0.146$, close to the source paper's Meanflow mean $0.131$. Cold start remains near $0.90$–$0.97$. The source paper's DP column falls to $0.016$–$0.061$; that refinement is not reported here because a protocol-compliant 3D Stage II evaluation is not in the ledger. Mechanism: the one-step map again supplies the correct spatial scale, so the remaining error is geometric rather than a failure to leave the uniform measure. Print tables (ours; source paper).

**4.4 paragraph E — eigenvalue finding (Fig. 10 analog)**
Finding first: at $\sigma=2^{-4.5}$ the long-horizon FK value is $0.6148$. Cold start begins near $0.36$ in the FK figure and reaches the reference by about $10^2$ generations. Our Meanflow $\hat c_T$ is already $0.6040$ at $T=1$ and $0.6138$ at $T=128$. The source paper's MF+DP start is $0.6135$ at $T=1$. Mechanism: as in two dimensions, the eigenvalue functional can sit near the FK limit while projected $W_2$ remains $O(10^{-1})$. End on the scientific bound: the reproduced FK estimators and the 3D Meanflow $W_2$ are consistent with the source protocol; the 2D corrector and an unevaluated 3D corrector leave the $W_2$ claim of the source paper open.

### LaTeX mechanics

```tex
\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption} % omit if unavailable
```

- `\graphicspath{{figs/}}`
- Use `booktabs` tables.
- Float figures/tables with informative captions.
- Title: `KPP Front-Speed Experiments (Sections 4.3 and 4.4)`
- Author: leave as a placeholder `KPP reproduction group` or omit.
- No `TODO`, no `\hl`, no comments that narrate "Codex wrote this".

### Compile

From `kpp_report/`:

```
latexmk -pdf -interaction=nonstopmode kpp_sections_4_3_4_4.tex
```

If `latexmk` is absent, run `pdflatex` twice. On Windows PowerShell, `cd` to the directory first. Do not use a path with unescaped spaces as `\includegraphics` targets; that is why figures are copied into `kpp_report/figs/`.

## Acceptance

- Required tests:
  1. TeX file exists and is UTF-8.
  2. Compile command exits 0, or residual warnings only (undefined citations OK if unused).
  3. PDF exists and has at least 5 pages.
  4. Body contains both subsection titles.
  5. Body does not contain: `metrics.json`, `TODO`, `update interface`, `we implemented`, `Furthermore`, `Moreover`, `utilize`, `leverage`.
  6. No 3D `w2_dp` number from the backup run appears as our result.
  7. Result summary written to `.ai/codex_result_kpp_4344_report.md`.
- Required files_changed expectation: new files under `kpp_report/` plus the result summary. Nothing else.
- Required result summary: `.ai/codex_result_kpp_4344_report.md`
