# sprc2026

Working notes for a talk on photonic computing in the AI era. The core of the repo
is an analysis of the mathematics in the Supplementary Materials of

> S. Pai et al., *Experimentally realized in situ backpropagation for deep learning
> in photonic neural networks*, Science 380, 398 (2023).
> DOI [10.1126/science.ade8450](https://doi.org/10.1126/science.ade8450)

and a comparison of that analysis against the photonic inference processor in

> S. R. Ahmed et al., *Universal photonic artificial intelligence acceleration*,
> Nature 640, 368 (2025). DOI [10.1038/s41586-025-08854-x](https://doi.org/10.1038/s41586-025-08854-x)

## Layout

| path | what |
|---|---|
| `docs/insitu-backprop-math.md` | Section-by-section breakdown of the Science SM: MZI algebra, vector units, the gradient identity and its derivation, the three readout schemes, pseudocode, analog update, energy model, noise model, and a list of errata found while checking. |
| `docs/lightmatter-comparison.md` | The same ideas placed against Lightmatter's 2025 processor: how its ABFP multiply-accumulate works, a side-by-side table, where the two analyses agree, why in situ backprop is a mesh-specific idea, and what the 2023 energy model got right and wrong. |
| `scripts/verify_gradient.py` | Numerical check of eqs. S3 to S8 and S12 on a random triangular mesh against finite differences. Finds one conjugation error in the printed VJP (S5) and a sign typo in S12. |
| `scripts/energy_model.py` | Rebuilds Tables S1 to S4 and the photonic-advantage contours of fig. S8, reconciles the "2× at N = 64, M ≥ 16" claim, and puts the Lightmatter chip on the same per-op axes. |
| `scripts/energy_breakdown.py` | Stacked per-component energy bars for inference and training: the SM model with segmented phase shifters for inputs and weights (no DACs) at 8 bits, projected to 4 bits for inference and to 12-bit readout for training, against Envise measured and digital lines, plus the contact count a segmented weight array implies. Writes `figs/energy_breakdown.tex` (pgfplots) which the deck inputs. |
| `talk.tex`, `talk.pdf`, `figs/` | The beamer deck (metropolis, 16:9) the notes feed into. |
| `refs/` | Local copies of the source PDFs. Git-ignored; see below. |

## Running the checks

Only NumPy is needed.

```bash
python3 scripts/verify_gradient.py
```

```bash
python3 scripts/energy_model.py
```

## Sources

`refs/` is git-ignored because the PDFs are publisher-copyrighted. To rebuild it:

- `refs/pai2023_science_sm.pdf`: the Science SM, from the article's supplementary
  materials tab.
- `refs/ahmed2025_nature_si.pdf`: the Nature Supplementary Information, free at
  the article page under "Supplementary information".
- `refs/basumallik2022_abfp_arxiv.pdf`: [arXiv:2205.06287](https://arxiv.org/abs/2205.06287).

The Nature main text is paywalled. The comparison note says which statements rest
on the SI, the Lightmatter blog, or secondary coverage.

## Building the deck

```bash
latexmk -xelatex talk.tex
```

Speaker notes are in `\note{}` blocks; add
`\setbeameroption{show notes on second screen}` to see them. Rendered slide PNGs
for review go in `qa/` (git-ignored):

```bash
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r110 -sOutputFile=qa/slide-%02d.png talk.pdf
```
