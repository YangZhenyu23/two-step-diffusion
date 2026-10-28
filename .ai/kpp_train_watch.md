# KPP OT+geodesic training watch

Policy: parallel is allowed. 3D Stage I uses ~1.3 GB of 8 GB, so 2D may run at the same time. If a later tick sees CUDA OOM, stop 2D and finish 3D first.

| Job | Command | Run dir | Log |
|---|---|---|---|
| 3D | `python -u -m two_step_3d.run_3d --mode all --run-name full_3d_otgeo --device cuda` | `two_step_3d/runs/kpp3d/full_3d_otgeo` | `.ai/train_3d_otgeo.log` |
| 2D | `python -u -m two_step_2d.run_2d --mode all --run-name full_2d_otgeo --device cuda` | `two_step_2d/runs/kpp2d/full_2d_otgeo` | `.ai/train_2d_otgeo.log` |

Done when both `metrics.json` exist.

Then, from repo root:

1. Point `kpp_report/plot_paper_figures.py` and `kpp_report/extract_paper_numbers.py` at the two `*_otgeo` dirs.
2. `python kpp_report/plot_paper_figures.py`
3. `python kpp_report/extract_paper_numbers.py`
4. Update decimals and 4.3.2 / 4.4.2 geometry sentences in `kpp_report/kpp_sections_4_3_4_4.tex`
5. `latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex` in `kpp_report/`
6. Stop the 30-minute watch loop.
