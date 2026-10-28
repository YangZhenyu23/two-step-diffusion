# Result: KPP Layout Handoff

## Files changed

- `kpp_report/kpp_sections_4_3_4_4.tex`
- `kpp_report/FIGURE_HANDOFF.md`
- `kpp_report/README.md`

## Compile

Command:

```powershell
cd kpp_report
latexmk -pdf -interaction=nonstopmode -halt-on-error kpp_sections_4_3_4_4.tex
```

Exit code: `0`.

## PDF page count

6 pages.

## Figure-to-subsection map

- Figure 5: 4.3 Invariant measure
- Figure 6: 4.3 Torus Wasserstein Error
- Figure 7: 4.3 Eigenvalue estimator
- Figure 8: 4.4 Projected invariant measure
- Figure 9: 4.4 Projected Wasserstein Error
- Figure 10: 4.4 Eigenvalue estimator

## Layout gap

Figure 8 starts on its own page because it is a tall 3x3 panel. This is acceptable.
