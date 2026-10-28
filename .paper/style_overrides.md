# Style Overrides

## Register

Write as a scientific Results excerpt, not a lab notebook, README, or
technical report. Forbidden: "implementation details", "code structure",
"how to run", "update interface", "we implemented", "replace these rows",
bullet hyperparameter dumps, file-path narration in body prose.

Allowed: equations, experimental protocol in running prose, findings-first
result paragraphs, figure/table captions, hedged comparison with the source
paper.

## Domain Vocabulary

Use the source paper's terms: Meanflow (or MeanFlow after first definition),
Deep Particle / DP, Feynman-Kac, invariant measure, front speed, twisted
diffusion, cellular flow, Kolmogorov flow, interpolation / extrapolation,
one-step / 1-NFE, near-identity corrector, $W_2$.

Do not import software-repo jargon: pipeline, stack, checkpoint, seed search,
runner, smoke test, ablation (unless reporting a true ablation), ground truth
(prefer "long-horizon Feynman-Kac reference").

## Domain Vocabulary Swaps

| Avoid | Prefer |
|---|---|
| ground truth | long-horizon Feynman-Kac reference |
| pipeline | two-stage procedure / two-step method |
| checkpoint | trained map / trained corrector |
| ablation study | comparison across initializations |
| robust | (quantify, or delete) |
| significant | only with a statistical test |
| demonstrates / confirms / proves | indicates / is consistent with / remains |

## Comparison With The Source Paper

Every comparison sentence must name both numbers and the source. Do not write
"we reproduce the paper" as a blanket claim. Reproduce the *protocol*; report
the *measured* $W_2$ and $\hat c_T$.

## Hedging

Simulation output and inferred mechanisms take hedges: "is consistent with",
"suggests", "appears to". Direct language is allowed for protocol settings
the authors controlled and for numbers printed in a table.

## Language

English body. Chinese is not used in the LaTeX body, captions, or titles.

## Punctuation

At most one colon and one semicolon per body paragraph. No invented
hyphenated compounds. ASCII hyphen only in PDF-facing text.
