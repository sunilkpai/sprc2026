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
| `docs/rack-design.md` | Discussion note: what a 50 to 100 W accelerator package does to rack design and footprint. Where the heat is, stability versus capacity cooling, pooled lasers as the only liquid loop, the 30 kW air-rack arithmetic against an NVL72, and what memory, network and PUE leave unchanged. |
| `docs/footprint-tdm.md` | Discussion note: silicon area per weight for meshes, Envise and a GPU; why one device per parameter is the mesh's structural cost; what Opticore-style temporal mapping (photoelectric multiplication, weights as pulses) buys and what it gives up; how a mesh can be time-multiplexed at block level and what that does to the energy accounting. |
| `docs/transformers.md` | Discussion note: MACs per token for MLP, projections and attention versus context length; what GQA, MQA and the long-context tricks change; the batch per weight block by regime; why attention wants a streaming (temporal-mapping) circuit; tiling a wide matrix into SVD tiles of one mesh; why weights do not fit in silicon and decode stays memory-bound. |
| `scripts/verify_gradient.py` | Numerical check of eqs. S3 to S8 and S12 on a random triangular mesh against finite differences. Finds one conjugation error in the printed VJP (S5) and a sign typo in S12. |
| `scripts/energy_model.py` | Rebuilds Tables S1 to S4 and the photonic-advantage contours of fig. S8, reconciles the "2× at N = 64, M ≥ 16" claim, and puts the Lightmatter chip on the same per-op axes. |
| `scripts/energy_breakdown.py` | Stacked per-component energy bars for inference and training: the SM model with segmented phase shifters for inputs and weights (no DACs) at 8 bits, projected to 4 bits for inference and to batch-integrated 12-bit gradient readout for training, against Envise measured and digital lines, plus the contact count a segmented weight array implies. Writes `figs/energy_breakdown.tex` (pgfplots) which the deck inputs. |
| `slides/` | The talk as a reveal.js deck: `index.html`, theme, vendored reveal.js, KaTeX and Fira Sans (no CDN, works offline), the two Science training movies, and the generated chart SVG. Published to GitHub Pages at https://sunilkpai.github.io/sprc2026/ by `.github/workflows/pages.yml` (via the `gh-pages` branch); `.gitlab-ci.yml` does the same on GitLab Pages. |
| `talk.tex`, `talk.pdf`, `figs/` | The earlier beamer version of the deck (metropolis, 16:9). Kept for the PDF; the reveal.js deck is the one being maintained. |
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

## The deck

Serve `slides/` from any static server and open it; press `s` for the speaker
view with notes, `f` for fullscreen, `?` for the key map.

```bash
python3 -m http.server 8765 --directory slides
```

Every push to `main` that touches `slides/` republishes
https://sunilkpai.github.io/sprc2026/: the workflow in
`.github/workflows/pages.yml` copies `slides/` to the `gh-pages` branch, which
GitHub Pages serves. The QR code on the last slide points there. To serve from GitLab Pages instead, push the repo to
GitLab (the included `.gitlab-ci.yml` publishes `slides/`) and regenerate the QR
with the GitLab URL:

```bash
npx --yes qrcode -t svg -o slides/media/qr.svg "https://<namespace>.gitlab.io/sprc2026/"
```

The chart on the energy slide is `slides/media/energy_breakdown.svg`, written by
`scripts/energy_breakdown.py` together with the pgfplots version. Vendored
libraries: reveal.js 5 (MIT), KaTeX (MIT), Fira Sans (OFL); licenses are next
to the files under `slides/vendor/`.

The beamer version still builds with `latexmk -xelatex talk.tex`.
