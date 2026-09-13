# Revisiting the prospects of photonic computing in the AI era

Beamer deck (metropolis theme, 16:9). Build:

    latexmk -xelatex talk.tex

Render pages for review (no LibreOffice or poppler needed):

    gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r110 -sOutputFile=qa/slide-%02d.png talk.pdf

## Numbers to sanity-check before presenting

- Slide 6 chart. Digital bars are datasheet peak dense throughput divided by
  board power: H100 INT8 (~2 POPS / 700 W ≈ 0.35 pJ), B200 FP4 (~9 PFLOPS / 1000 W
  ≈ 0.11 pJ), and the 100 fJ digital MAC assumed in the Science paper. The Envise
  bar is 65.5 TOPS at 78 W electrical + 1.6 W optical from Harris's April 2025
  blog post on the Nature paper (Ahmed et al., Nature 640, 368, 2025). Formats
  differ, so the chart is labeled as order-of-magnitude.
- Slide 3 stats: Passage M1000 (114 Tbps, 4,000 mm^2, 256 fibers), $850M raised,
  $4.4B valuation (Oct 2024), all from Lightmatter press releases and site.
- Slide 8 quantum column cites only published work (Carolan 2015; Xanadu Aurora,
  Nature 2025; silicon platform, Nature 2025). Vet against employer disclosure rules.

## Figures

- `figs/bp.png`: Science perspective figure on the in situ backprop paper.
- `figs/parallelnull_a.png`: panel (a) of the parallel programming figure
  (JSTQE 2020), cropped from `figs/parallelnull.png`.

Speaker notes are in `\note{}` blocks; compile with
`\setbeameroption{show notes on second screen}` to see them.
