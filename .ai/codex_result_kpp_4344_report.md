# Result: KPP Sections 4.3–4.4

## Files created

- `kpp_report/kpp_sections_4_3_4_4.tex`
- `kpp_report/kpp_sections_4_3_4_4.pdf`
- `kpp_report/figs/fig5_ref.png`
- `kpp_report/figs/fig6_w2_2d.png`
- `kpp_report/figs/fig7_cT_2d_fk.png`
- `kpp_report/figs/fig8_ref.png`
- `kpp_report/figs/fig10_cT_3d_fk.png`

## Compile command and exit code

From `kpp_report/`:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex
```

Exit code: `0`.

## Page count

5 pages.

## Figures included

- `fig5_ref`: FK reference histograms for 2D.
- `fig7_cT_2d_fk`: 2D eigenvalue convergence.
- `fig8_ref`: FK reference projections for 3D.
- `fig10_cT_3d_fk`: 3D eigenvalue convergence.

`fig6_w2_2d` was copied as a local asset but the quantitative \(W_2\) result
is presented as a table, so the figure is not included in the PDF.

## Untraceable numbers

None. All numeric values in the body are drawn from
`.paper/claim_evidence_ledger.md` or printed in the excerpt tables.

## Residual risks

- The 3D corrected \(W_2\) column is intentionally omitted because the citable
  run did not freeze MeanFlow before Stage II.
- The 2D corrected column is reported but does not uniformly improve on
  MeanFlow; this is stated directly.
- Figure captions for `fig5_ref` and `fig8_ref` explicitly mark the panels as
  reference clouds only.
- Cross-references required two LaTeX passes; the final compile is clean except
  for harmless hyperref warnings.
