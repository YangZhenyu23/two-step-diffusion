# Frozen Stage I + DP-only retrain

Stage I weights are copies. Do not run `--mode all` or `train-stage1`.

| Job | Frozen Stage I | Run dir | Log |
|---|---|---|---|
| 2D Stage II | `two_step_2d/runs/kpp2d/full_2d_v2/stage1_final.pt` | `two_step_2d/runs/kpp2d/full_2d_v2_dpfix` | `.ai/train_2d_dpfix.log` |
| 3D Stage II | `two_step_3d/runs/kpp3d/full_3d_otgeo/stage1_final.pt` | `two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix` | `.ai/train_3d_dpfix.log` |

Done when both `metrics.json` exist.
