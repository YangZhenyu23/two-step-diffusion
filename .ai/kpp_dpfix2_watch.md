# Frozen Stage I + DP-only retrain (dpfix2)

Stage I weights are copies. Do not run `--mode all` or `train-stage1`.

| Job | Frozen Stage I | Run dir |
|---|---|---|
| 2D Stage II | `two_step_2d/runs/kpp2d/full_2d_v2/stage1_final.pt` | `two_step_2d/runs/kpp2d/full_2d_v2_dpfix2` |
| 3D Stage II | `two_step_3d/runs/kpp3d/full_3d_otgeo/stage1_final.pt` | `two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2` |

Stage II changes vs dpfix:

- Primary signal: differentiable Sinkhorn W2 on a 36x36 torus histogram (eval geometry)
- Down-weighted mini-batch EMD (ot_weight=0.25) so 1500-point matching cannot dominate
- Mean residual λ=0.25
- alpha 0.2, batch_refresh=400
- Probe averages all training σ
- 3D DP gets torus features (Stage II only)

Done when both `metrics.json` exist.
