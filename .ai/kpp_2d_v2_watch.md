# 2D v2 training

Only 2D. Do not start a 3D job.

| Job | Command | Run dir | Log |
|---|---|---|---|
| 2D | `python -u -m two_step_2d.run_2d --mode all --run-name full_2d_v2 --device cuda` | `two_step_2d/runs/kpp2d/full_2d_v2` | `.ai/train_2d_v2.log` |

Done when `metrics.json` exists and the train process has exited.

Then:

1. Point `kpp_report/plot_paper_figures.py` at `full_2d_v2` (keep 3D on `full_3d_otgeo`).
2. Redraw the three 2D figures and latexmk `kpp_report/kpp_sections_4_3_4_4.tex`.
3. Patch 4.3 decimals and the 4.3.2 geometry sentences.
