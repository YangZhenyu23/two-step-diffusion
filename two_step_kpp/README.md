# Two-Step Diffusion (KPP) Reproduction

Local reproduction of the KPP experiments from
*Two-Step Diffusion: Fast Sampling and Reliable Prediction for 3D Keller-Segel
and KPP Equations in Fluid Flows* (arXiv:2601.20024v1).

This package reuses the long-horizon Feynman-Kac reference point clouds in
`../KPP data/data/` as ground truth. It does **not** modify the original
`KPP data`, `meanflow`, or `DP` directories.

## Layout

- `data_io.py`: reference loading, torus helpers, dataset construction
- `models.py`: `MFMLP` (Stage I) and `DPResNet` (Stage II)
- `stage1.py`: MeanFlow-style JVP regression
- `stage2.py`: mini-batch EMD refinement
- `ot_solver.py`: stateful POT EMD coupling
- `eval_kpp.py`: W2 and warm-start eigenvalue metrics
- `run_2d.py` / `run_3d.py`: experiment entrypoints

## Quick smoke test

```powershell
$py = "C:\path\to\python.exe"
& $py -m two_step_kpp.run_2d --mode smoke
& $py -m two_step_kpp.run_3d --mode smoke
```

Run the full pipeline with:

```powershell
& $py -m two_step_kpp.run_2d --mode all --run-name full
& $py -m two_step_kpp.run_3d --mode all --run-name full
```

Results are written under `runs/<exp>/<run_name>/`, including model
checkpoints, `metrics.json`, and plots.
