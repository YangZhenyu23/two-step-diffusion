# Claim-Evidence Ledger

Canonical numeric source for our two-step $W_2$ and $\hat c_T$ (current draft):
`two_step_2d/runs/kpp2d/full_2d_v2_dpfix2/metrics.json` and
`two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2/metrics.json`.
Stage I weights are frozen copies of `full_2d_v2` and `full_3d_otgeo`.

Do not mix with `midterm_report.tex` or `runs_backup_20260829/`.

Round table entries to four decimals. Keep six decimals only for the source-paper
eigenvalue table if copied as a comparison.

## Source paper (Shen et al., 4.3 Table / Fig. 6)

| -log2 σ | MF W2 | DP W2 |
|---:|---:|---:|
| 2.00 | 0.0477 | 0.0138 |
| 2.25 | 0.0476 | 0.0139 |
| 2.50 | 0.0532 | 0.0124 |
| 2.75 | 0.0592 | 0.0123 |
| 3.00 | 0.0639 | 0.0106 |
| 3.25 | 0.0717 | 0.0148 |
| 3.50 | 0.0870 | 0.0200 |
| 3.75 | 0.0773 | 0.0154 |
| 4.00 | 0.1012 | 0.0216 |
| 4.25 | 0.1195 | 0.0154 |
| 4.50 | 0.1176 | 0.0456 |

Paper 2D mean MF ≈ 0.0769, mean DP ≈ 0.0178.

## Our 2D (full_2d_v2_dpfix2/metrics.json)

| -log2 σ | cold W2 | MF W2 | DP W2 |
|---:|---:|---:|---:|
| 2.00 | 1.0538 | 0.0962 | 0.1061 |
| 2.25 | 1.0731 | 0.0962 | 0.1023 |
| 2.50 | 1.0901 | 0.0965 | 0.1019 |
| 2.75 | 1.1071 | 0.1203 | 0.1138 |
| 3.00 | 1.1137 | 0.1075 | 0.0983 |
| 3.25 | 1.1301 | 0.1207 | 0.1074 |
| 3.50 | 1.1436 | 0.1259 | 0.1123 |
| 3.75 | 1.1484 | 0.1424 | 0.1368 |
| 4.00 | 1.1495 | 0.1869 | 0.1701 |
| 4.25 | 1.1615 | 0.2220 | 0.1978 |
| 4.50 | 1.1718 | 0.2629 | 0.2326 |

Our 2D mean MF = 0.1434, mean DP = 0.1345.
At σ=2^{-4.5}: ĉ = 0.7842; cT_mf(1)=0.8180, cT_mf(128)=0.7841; cT_dp(1)=0.8167, cT_dp(128)=0.7848.

## Source paper (Shen et al., 4.4 Table / Fig. 9)

| -log2 σ | MF W2 | DP W2 |
|---:|---:|---:|
| 2.00 | 0.1591 | 0.0160 |
| 2.25 | 0.1257 | 0.0230 |
| 2.50 | 0.1442 | 0.0193 |
| 2.75 | 0.1646 | 0.0280 |
| 3.00 | 0.1250 | 0.0365 |
| 3.25 | 0.1175 | 0.0314 |
| 3.50 | 0.1342 | 0.0340 |
| 3.75 | 0.1261 | 0.0253 |
| 4.00 | 0.1049 | 0.0606 |
| 4.25 | 0.1325 | 0.0282 |
| 4.50 | 0.1099 | 0.0210 |

Paper 3D mean MF ≈ 0.1312, mean DP ≈ 0.0294.

## Our 3D (full_3d_otgeo_dpfix2/metrics.json)

| -log2 σ | cold W2 | MF W2 | DP W2 |
|---:|---:|---:|---:|
| 2.00 | 0.8895 | 0.0894 | 0.1047 |
| 2.25 | 0.9057 | 0.1052 | 0.0997 |
| 2.50 | 0.9329 | 0.1201 | 0.0954 |
| 2.75 | 0.9193 | 0.1272 | 0.0920 |
| 3.00 | 0.9224 | 0.1102 | 0.0988 |
| 3.25 | 0.9355 | 0.1266 | 0.1016 |
| 3.50 | 0.9368 | 0.1263 | 0.1126 |
| 3.75 | 0.9600 | 0.1417 | 0.0945 |
| 4.00 | 0.9569 | 0.1470 | 0.1168 |
| 4.25 | 0.9569 | 0.1701 | 0.1191 |
| 4.50 | 0.9564 | 0.1582 | 0.1366 |

Our 3D mean MF = 0.1293, mean DP = 0.1065.
At σ=2^{-4.5}: ĉ = 0.6147; cT_mf(1)=0.6148, cT_mf(128)=0.6131; cT_dp(1)=0.6171, cT_dp(128)=0.6142.

## Allowed claims

- FK reference geometry on T^2 is a concentrated anisotropic ridge; it tightens as σ decreases (fig5_ref, sweep_2d).
- FK long-run front speed decreases with -log2 σ (speed.png).
- Uniform cold start has torus W2 ≈ 1.05-1.17 (2D) and ≈ 0.90-0.97 (3D).
- MeanFlow reduces 2D W2 from O(1) to a grid mean of 0.1434, about 1.9× the paper MeanFlow mean (0.0769).
- On this 2D run, DP is below MeanFlow on 8 of 11 levels (2.75 through 4.50). Grid means are 0.1434 (MF) and 0.1345 (DP). At 4.50, DP is 0.2326 against the paper value 0.0456.
- 3D MeanFlow W2 (mean 0.1293) is next to the paper MeanFlow mean (0.1312). DP mean is 0.1065 and is below MeanFlow on 10 of 11 levels, including 4.50 (0.1366 vs 0.1582). The paper DP mean is 0.0294.
- Cold-start ĉ_T approaches the long-run FK value; warm-start transients sit near that limit by T=128. Source-paper DP T=1 values are 0.7616 (2D) and 0.6135 (3D).

## Forbidden claims

- DP uniformly reduces W2 (false at the three coarsest 2D levels and at 3D -log2 σ = 2.00).
- The reproduction matches the paper's W2.
- 3D DP restores ridges / reduces W2 to 0.02.
- Eigenvalue proximity implies that the invariant measure has been recovered.
- Any number not in this ledger, a listed figure, or a table printed in the excerpt.
- Overclaim verbs: demonstrates, confirms, proves, establishes, shows (for inferred findings).
