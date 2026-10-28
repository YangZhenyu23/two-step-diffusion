# Paper Context

## Working Title

Two-Step Diffusion for KPP Front-Speed Estimation: Reproduction of Sections 4.3 and 4.4

## One-Sentence Contribution

This excerpt reports a reproduction of the 2D cellular-flow and 3D Kolmogorov-flow
KPP experiments in Shen et al. (arXiv:2601.20024v1), using a Feynman-Kac particle
system as the reference and a MeanFlow plus DeepParticle pipeline as the learned
transport.

## Research Questions

1. Does the reproduced Feynman-Kac particle system recover the invariant measure
   geometry and the long-horizon front-speed estimator of the source paper?
2. Does a one-step MeanFlow map, trained on a subset of diffusion levels,
   reduce the torus (or projected) $W_2$ distance to that reference relative to
   a uniform cold start?
3. Does a frozen-MeanFlow DeepParticle corrector further reduce $W_2$ and
   shorten the transient of $\hat c_T(\lambda)$, including at extrapolated
   $\sigma$?

## Hypotheses Or Directional Expectations

The source paper's directional claim is that MeanFlow captures coarse support
and DeepParticle removes residual mass misalignment, with the largest $W_2$
gains in the small-$\sigma$ extrapolation regime, and that MF+DP warm starts
make $\hat c_T$ nearly unbiased at $T=1$. The reproduction tests that claim
on the author's own runs. Where the numbers diverge, the excerpt reports the
observed gap; it does not import the source paper's success narrative.

## Method Summary

Stage I: MeanFlow one-step map $x=\varepsilon-u_\theta(\varepsilon,0,1,\sigma)$.
Stage II: near-identity DeepParticle corrector trained with mini-batch $W_2$.
Reference: Feynman-Kac interacting particle system on $\mathbb{T}^2$ or
$\mathbb{T}^3$, $M=2\times10^4$, $T=4096$, $\lambda=2$, $\Delta t=2^{-8}$.

## Main Findings (from citable artifacts only)

- FK reference clouds recover the anisotropic ridge on $\mathbb{T}^2$ and the
  3D projection geometry; long-run $\hat c_T$ is $0.7842$ (2D) and $0.6147$ (3D)
  at $\sigma=2^{-4.5}$.
- Cold-start $W_2$ on the $72\times72$ periodic histogram is $O(1)$; MeanFlow
  lowers it to $O(10^{-1})$.
- On `full_2d_v2_dpfix2`, DeepParticle mean $W_2$ is 0.1345 vs MeanFlow 0.1434.
  DP is lower on 8 of 11 levels, including all three extrapolations.
  At $\sigma=2^{-4.5}$ DP is 0.2326 against the source-paper value 0.0456.
- On `full_3d_otgeo_dpfix2`, MeanFlow mean $W_2$ is 0.1293 (paper 0.1312).
  DP mean is 0.1065 and is below MeanFlow on 10 of 11 levels, including
  4.50 (0.1366 vs 0.1582). The source-paper DP mean is 0.0294.

## Key Terms

- $\sigma=\kappa$: diffusion constant in the KPP / twisted SDE.
- $\hat c_T(\lambda)=\hat\mu_T(\lambda)/\lambda$: running front-speed estimator.
- Interpolation: $-\log_2\sigma\le 3.75$. Extrapolation: $4.00,4.25,4.50$.
- $W_2$: exact EMD on a $72\times72$ torus histogram (2D) or $x$-$z$ projection (3D).

## Current Manuscript State

No academic 4.3/4.4 prose exists yet. `midterm_report.tex` is a technical-report
draft and must not be copied. Codex writes a new file under `kpp_report/`.

## Known Risks

- Two 2D $W_2$ tables exist (backup `metrics.json` vs incomplete `midterm_report.tex`).
  Canonical source is `runs_backup_20260829/**/metrics.json`.
- 3D DP from the backup run is not a valid test of Algorithm 2.
- FK histograms in `remote_results/figs/` are reference-only, not MF/DP triptychs.
  Do not caption them as MeanFlow or DP predictions.
