"""Stacked energy-breakdown bars: Science-SM model vs Lightmatter measured, 8-bit and
4-bit projections.  Writes figs/energy_breakdown.tex (a pgfplots tikzpicture for
\\input in the deck) and prints the numbers behind it.

Per-op basis:  an N x N MVM is 2 N^2 real ops; an in situ VJP/grad step (forward +
backward MVM equivalent) is 4 N^2 real ops per example.  Batch cost is divided by M.

All photonic bars use segmented phase shifters for inputs and weights: no DACs.

4-bit projection rules (all stated in the figure caption):
  * ADC energy          ~ 2^b   (Walden-type scaling; the Nature SI uses the thermal-limited
                                  2^(2 db) for its ADC, which would be 16x more optimistic)
  * segmented PS        ~ b     (b binary-weighted segments; the SM charges 16 E_mod at 8 bits)
  * digital op energy   /3      (B200 FP4 vs H100 INT8 at the wall is ~3.2x; Horowitz b^2 for
                                  the multiplier gives ~4x)
  * optical power       ~ 4^b   (shot-noise-limited amplitude SNR), i.e. /256 for inference.
  * TIA, switches: bit-independent.
Training is NOT projected to 4 bits: gradients need precision.  The training panel shows
the SM's 8-bit per-example readout at M=16 next to batch-integrated gradient readout: the
tap photocurrent is integrated over the batch and digitised once per phase shifter per batch
with a slow 12-16 bit ADC, so the gradient ADC runs at f/M and the integrated signal gains
sqrt(M) in SNR.  At M >= 256 that is 4 extra gradient bits with no extra light.

Run:  python3 scripts/energy_breakdown.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from energy_model import components, DEFAULT, fJ, pJ  # noqa: E402

N, M_TRAIN = 128, 16
OPS_INF, OPS_TRAIN = 2 * N**2, 4 * N**2
DB = 4  # bits removed: 8 -> 4

# ------------------------------------------------------------- Lightmatter measured
LM_TOPS = 65.5e12
LM_P_PTC, LM_P_OPT, LM_P_TOTAL = 0.25, 1.6, 78.0 + 1.6         # W  (SI VI)
LM_ENC = LM_P_PTC / LM_TOPS       # J/op: vector + weight encode incl. weight DACs
LM_OPT = LM_P_OPT / LM_TOPS
LM_REST = (LM_P_TOTAL - LM_P_PTC - LM_P_OPT) / LM_TOPS  # DCI: ADCs, digital pipeline, SRAM, PCIe

# --------------------------------------------------------------- scenarios
# All photonic scenarios use segmented ("digital control") phase shifters for both the
# input vectors and the mesh weights (SM sec. 2.7.3): a b-bit value is written as b binary
# weighted phase segments driven directly by logic, so no DAC appears anywhere.  The SM
# charges this as E_mod -> 16 E_mod at 8 bits; we scale the segment count with b.
SEG = dict(digital_control_ps=True)
FOUR_BIT = dict(E_ADC=DEFAULT["E_ADC"] / 2**DB, E_OP=DEFAULT["E_OP"] / 3,
                E_mod=DEFAULT["E_mod"] * (8 - DB) / 8)          # 4 segments instead of 8
FOUR_BIT_INF = dict(FOUR_BIT, E_mode=DEFAULT["E_mode"] / 4**DB)

KEYS = ["digital I/O prep", "modulator", "DAC", "ADC", "TIA", "gradient readout", "optical power",
        "switches", "rest of system"]


def per_op(comp, ops, M=1):
    return {k: comp.get(k, 0.0) / ops / M for k in KEYS}


M_BIG = 256
M_LLM = 4096          # vectors per weight block per update: a per-device transformer micro-batch
# Batch-integrated gradient readout (SM sec. 2.6.2, Alg. 4 lines 9-10): the tap photocurrent is
# integrated over the M examples of a batch and read ONCE per phase shifter per batch.  The
# gradient ADC therefore runs at f_clk / M, where 12+ bit converters are cheap and slow, and the
# integrated signal gains sqrt(M) in SNR (shot noise) over a single-example readout.
E_ADC_GRAD = 20 * pJ  # 12-16 bit SAR at <= 10 MS/s, per conversion (Murmann survey class)
E_TIA_GRAD = 5 * pJ   # precision TIA + gated integrator + sample/hold, per gradient sample
UP = 4                # extra gradient bits from batch integration at M >= 256 (sqrt(M) = 16 = 4 bits)


def batch_grad(M, **kw):
    """8-bit per-example forward/backward/sum passes, gradient read once per batch per phase."""
    d = per_op(components(N, M, **SEG, **kw)[1], OPS_TRAIN, M)
    # replace the per-batch analog updater term 4 N^2 E_TIA by explicit readout electronics
    d["TIA"] = 8 * M * N * DEFAULT["E_TIA"] / OPS_TRAIN / M
    d["gradient readout"] = N * (N - 1) * (E_ADC_GRAD + E_TIA_GRAD) / OPS_TRAIN / M
    return d


inf = {
    "Envise meas.": {**{k: 0.0 for k in KEYS}, "modulator": LM_ENC, "optical power": LM_OPT,
                     "rest of system": LM_REST},
    "SM 8-bit": per_op(components(N, 1, **SEG)[0], OPS_INF),
    "SM 4-bit": per_op(components(N, 1, **SEG, **FOUR_BIT_INF)[0], OPS_INF),
}
train = {
    f"SM 8-bit M{M_TRAIN}": per_op(components(N, M_TRAIN, **SEG)[1], OPS_TRAIN, M_TRAIN),
    f"batch-int. M{M_BIG}": batch_grad(M_BIG),
    f"batch-int. M{M_LLM}": batch_grad(M_LLM),
}
digital = {"model 8-bit": 6 * DEFAULT["E_OP"] / 2, "model 4-bit": 6 * FOUR_BIT["E_OP"] / 2,
           "H100 INT8": 0.35 * pJ, "B200 FP4": 0.11 * pJ,
           "H100 FP16": 700 / 990e12,              # dense FP16 peak over board power
           # OpenAI/Broadcom Jalapeno (Hot Chips 2026): 13.4 PFLOP/s MXFP4 at a 700 W package
           "Jalapeno MXFP4": 700 / 13.4e15,
           # Cerebras WSE-3 / CS-3: 125 PFLOPS FP16 vendor peak at ~23 kW system power
           "Cerebras FP16": 23e3 / 125e15}
NONZERO = {k for d in (inf, train) for c in d.values() for k, v in c.items() if v > 0}

# ------------------------------------------------------------------- report
def report(title, d, ops_label):
    print(f"\n{title}  [fJ per op, N={N}]")
    print("  " + "scenario".ljust(16) + "".join(k[:9].rjust(10) for k in KEYS) + "     total")
    for name, comp in d.items():
        tot = sum(comp.values())
        print("  " + name.ljust(16) + "".join(f"{comp[k] / fJ:10.1f}" for k in KEYS) + f"{tot / fJ:10.1f}")


report("INFERENCE (in situ MVM)", inf, "2N^2")
report("TRAINING (in situ VJP/grad)", train, "4N^2")
print("\nDigital baselines, fJ per op: " + ", ".join(f"{k} {v / fJ:.0f}" for k, v in digital.items()))
print(f"Envise rest-of-system (DCI) = {LM_REST / fJ:.0f} fJ/op, off the axis in the figure.")

# ------------------------------------------------------------ is the ADC fast enough?
print("\nADC budget: per-example activation readout at f_clk vs per-batch gradient readout at f_clk/M")
for M in (16, 256, M_LLM):
    print(f"  M={M:5d}: gradient ADC rate {1e9 / M / 1e6:8.2f} MS/s per tap; gradient readout "
          f"{N * (N - 1) * (E_ADC_GRAD + E_TIA_GRAD) / OPS_TRAIN / M / fJ:6.1f} fJ/op; "
          f"SNR gain from integration sqrt(M) = {M ** 0.5:5.1f} = {0.5 * __import__('math').log2(M):.1f} bits")

# ---------------------------------------------- what segmented weights cost in contacts
# A b-bit segmented phase shifter needs b digital lines from the control die, so an N x N
# mesh (N(N-1) phases) needs ~ N^2 b vertical interconnects: a bump / hybrid-bond array
# per weight cell.  Lightmatter's PTC has 6,000 bumps (SI III) and pushes weights serially
# through on-die 7-bit DACs instead.
print("\nSegmented-PS weight control: vertical contacts per N x N mesh and array area")
print("        bits   contacts   area @40um microbump   area @10um hybrid bond   (PTC die: 349 mm2)")
for b in (8, 4):
    n_contacts = N * (N - 1) * b
    a40 = n_contacts * (40e-6) ** 2 * 1e6
    a10 = n_contacts * (10e-6) ** 2 * 1e6
    print(f"        {b:4d}   {n_contacts:8d}   {a40:14.0f} mm2         {a10:12.1f} mm2")

# ------------------------------------------------------------------- pgfplots
SERIES = [("digital I/O prep", "slate", "digital I/O prep"),
          ("modulator", "moss", "encode: segmented PS (Envise: mod.\\ + weight DAC)"),
          ("DAC", "amber", "input DAC"),
          ("ADC", "sky", "ADC"),
          ("TIA", "ink2", "TIA (per example)"),
          ("gradient readout", "sky!50", "gradient readout, once per batch (12-bit ADC + integrator)"),
          ("optical power", "amber!40", "optical power"),
          ("switches", "mist", "switches"),
          ("rest of system", "ink!25", f"Envise DCI, rest of system: {LM_REST / fJ:.0f}, off scale")]
YMAX = 400.0


def axis(name, data, title, at=None, legend=False, ylabel=True, ymax=YMAX, baselines=None,
         offscale=()):
    cats = list(data)
    lines = []
    opts = [f"name={name}", "width=0.5\\textwidth", "height=3.9cm", "ybar stacked",
            "bar width=13pt", f"ymin=0, ymax={ymax:.0f}", "ylabel near ticks",
            "symbolic x coords={" + ",".join(cats) + "}", "xtick=data",
            "x tick label style={font=\\scriptsize, rotate=25, anchor=north east}",
            "y tick label style={font=\\scriptsize}", "ymajorgrids", "grid style={color=mist}",
            "axis line style={color=slate}", "enlarge x limits=0.18",
            f"title={{\\footnotesize {title}}}", "title style={yshift=-4pt}",
            "clip=true"]
    if ylabel:
        opts.append("ylabel={\\footnotesize fJ per op}")
    if at:
        opts.append(f"at={{({at}.south east)}}, anchor=south west, xshift=0.9cm")
    if legend:
        opts += ["legend style={font=\\scriptsize, draw=none, fill=none, at={(0.5,-0.42)}, "
                 "anchor=north, legend columns=4, /tikz/every even column/.append style={column sep=6pt}}",
                 "legend cell align=left"]
    lines.append("\\begin{axis}[\n  " + ",\n  ".join(opts) + "\n]")
    for key, color, label in SERIES:
        vals = [data[c][key] / fJ for c in cats]
        if key not in NONZERO or (max(vals) == 0 and not legend):
            continue
        coords = " ".join(f"({c},{v:.2f})" for c, v in zip(cats, vals))
        lines.append(f"\\addplot[fill={color},draw=none] coordinates {{{coords}}};")
        if legend:
            lines.append(f"\\addlegendentry{{{label}}}")
    # digital baselines: constant plots across the symbolic categories
    c0, c1 = cats[0], cats[-1]
    for label, val, style, inline in baselines:
        y = val / fJ
        if y > ymax:
            continue
        fp = "" if legend and not inline else "forget plot, "
        lines.append(f"\\addplot[{fp}sharp plot, stack plots=false, {style}, thick, no markers, "
                     f"line legend] coordinates {{({c0},{y:.0f}) ({c1},{y:.0f})}}"
                     + (f" node[pos=0, anchor=south west, font=\\tiny, color={style.split('color=')[1]}] "
                        f"{{{label} {y:.0f}}};" if inline else ";"))
        if legend and not inline:
            lines.append(f"\\addlegendentry{{{label} {y:.0f}}}")
    # off-scale bars: label beside the bar (nodes inside the axis are drawn under the bars)
    for c in offscale:
        tot = sum(data[c].values()) / fJ
        side = "anchor=east, xshift=-7pt, align=right" if c == cats[-1] else "anchor=west, xshift=7pt, align=left"
        lines.append(f"\\node[font=\\tiny, color=ink, {side}] at "
                     f"(axis cs:{c},{ymax * 0.55:.0f}) {{{tot:.0f} fJ/op,\\\\ off scale}};")
    # totals above each stack
    for c in cats:
        if c in offscale:
            continue
        tot = sum(data[c].values()) / fJ
        lines.append(f"\\node[font=\\tiny, color=ink, anchor=south, inner sep=1pt] at "
                     f"(axis cs:{c},{tot:.1f}) {{{tot:.0f}}};")
    lines.append("\\end{axis}")
    return "\n".join(lines)


INF_BASE = [("model 8-bit digital", digital["model 8-bit"], "dashed, color=slate", False),
            ("H100 INT8 wall", digital["H100 INT8"], "dotted, color=slate", False),
            ("B200 FP4 wall", digital["B200 FP4"], "dashdotted, color=slate", False),
            ("Cerebras WSE-3 FP16", digital["Cerebras FP16"], "dashdotted, color=amber", False),
            ("Jalapeno MXFP4 wall, inference only", digital["Jalapeno MXFP4"], "dashed, color=moss", False)]
TRAIN_BASE = [("model 8-bit digital", digital["model 8-bit"], "dashed, color=slate", False),
              ("H100 INT8 wall", digital["H100 INT8"], "dotted, color=slate", False),
              ("Cerebras WSE-3 FP16", digital["Cerebras FP16"], "dashdotted, color=amber", False)]
TRAIN_YMAX = 400
tex = "\n".join([
    "% Generated by scripts/energy_breakdown.py -- do not edit by hand.",
    "\\begin{tikzpicture}",
    axis("inf", inf, f"Inference: $N={N}$ MVM", legend=True, baselines=INF_BASE,
         offscale=("Envise meas.",)),
    axis("train", train, f"Training: in situ VJP/grad, $N={N}$", at="inf", ylabel=False,
         ymax=TRAIN_YMAX, baselines=TRAIN_BASE, offscale=()),
    "\\end{tikzpicture}",
])
out = os.path.join(os.path.dirname(__file__), "..", "figs", "energy_breakdown.tex")
with open(out, "w") as f:
    f.write(tex + "\n")
print(f"\nwrote {os.path.relpath(out)}")


# ------------------------------------------------------------------- SVG (reveal.js deck)
PALETTE = {"slate": "#5B6B7A", "moss": "#6B8F3D", "amber": "#D9821E", "sky": "#3A8FB7", "sky!50": "#9CC7DB",
           "ink2": "#2E4A6B", "amber!40": "#F0CDA5", "mist": "#EEF2F6", "ink!25": "#C5C9CE",
           "ink": "#16263A", "paper": "#FFFFFF"}


def svg_panel(x0, y0, w, h, data, title, ymax, baselines, offscale, ylabel):
    """One stacked-bar panel; returns SVG fragment.  Plot area inset for axes."""
    cats = list(data)
    L, R, T, B = 46, 8, 26, 58
    px, py, pw, ph = x0 + L, y0 + T, w - L - R, h - T - B
    sy = lambda v: py + ph * (1 - v / ymax)
    out = [f'<text x="{x0 + w / 2:.0f}" y="{y0 + 14}" class="ttl">{title}</text>']
    # grid + y axis
    step = 100 if ymax <= 500 else 200
    for v in range(0, int(ymax) + 1, step):
        out.append(f'<line x1="{px}" y1="{sy(v):.1f}" x2="{px + pw}" y2="{sy(v):.1f}" class="grid"/>')
        out.append(f'<text x="{px - 5}" y="{sy(v) + 3.5:.1f}" class="tick" text-anchor="end">{v}</text>')
    out.append(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" class="axis"/>')
    out.append(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" class="axis"/>')
    if ylabel:
        out.append(f'<text transform="translate({x0 + 11},{py + ph / 2:.0f}) rotate(-90)" class="lab" '
                   f'text-anchor="middle">fJ per op</text>')
    # bars
    n = len(cats); slot = pw / n; bw = min(34, slot * 0.5)
    for i, c in enumerate(cats):
        cx = px + slot * (i + 0.5); base = 0.0
        for key, color, _ in SERIES:
            v = data[c][key] / fJ
            if v <= 0:
                continue
            top = min(base + v, ymax)
            if base < ymax:
                out.append(f'<rect x="{cx - bw / 2:.1f}" y="{sy(top):.1f}" width="{bw:.1f}" '
                           f'height="{sy(base) - sy(top):.1f}" fill="{PALETTE[color]}"/>')
            base += v
        tot = sum(data[c].values()) / fJ
        if c in offscale:
            side = -1 if c == cats[-1] else 1
            anc = "end" if side < 0 else "start"
            out.append(f'<text x="{cx + side * (bw / 2 + 6):.1f}" y="{sy(ymax * 0.55):.1f}" class="tick" '
                       f'text-anchor="{anc}">{tot:.0f} fJ/op,</text>')
            out.append(f'<text x="{cx + side * (bw / 2 + 6):.1f}" y="{sy(ymax * 0.55) + 12:.1f}" class="tick" '
                       f'text-anchor="{anc}">off scale</text>')
        else:
            out.append(f'<text x="{cx:.1f}" y="{sy(tot) - 4:.1f}" class="tick" text-anchor="middle">{tot:.0f}</text>')
        out.append(f'<text transform="translate({cx:.1f},{py + ph + 10}) rotate(25)" class="tick" '
                   f'text-anchor="start">{c}</text>')
    # baselines
    dash = {"dashed": "8,5", "dotted": "2,4", "dashdotted": "8,4,2,4", "densely dotted": "2,2"}
    for label, val, style, inline in baselines:
        if val / fJ > ymax:
            continue
        y = sy(val / fJ)
        kind = style.split(",")[0]; col = PALETTE[style.split("color=")[1].strip()]
        out.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px + pw}" y2="{y:.1f}" stroke="{col}" '
                   f'stroke-width="2" stroke-dasharray="{dash[kind]}"/>')
        if inline:
            out.append(f'<text x="{px + 6}" y="{y - 4:.1f}" class="tick" fill="{col}">{label} {val / fJ:.0f}</text>')
    return "\n".join(out)


def write_svg(path):
    W, H = 1100, 430
    pw = 470
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'font-family="Fira Sans, Helvetica Neue, Arial, sans-serif">',
             '<style>.ttl{font-size:14px;fill:#16263A;text-anchor:middle}.tick{font-size:10.5px;fill:#5B6B7A}'
             '.lab{font-size:12px;fill:#5B6B7A}.grid{stroke:#EEF2F6;stroke-width:1}.axis{stroke:#5B6B7A;stroke-width:1}'
             '.leg{font-size:11.5px;fill:#16263A}</style>',
             f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
             svg_panel(30, 6, pw, 330, inf, f"Inference: N = {N} MVM", YMAX, INF_BASE, ("Envise meas.",), True),
             svg_panel(30 + pw + 60, 6, pw, 330, train, f"Training: in situ VJP/grad, N = {N}", TRAIN_YMAX,
                       TRAIN_BASE, (), False)]
    # legend
    items = [(PALETTE[c], lab) for k, c, lab in SERIES if k in NONZERO]
    items = [(c, lab.replace("\\\\ ", " ")) for c, lab in items]
    lines_ = [(PALETTE[st.split("color=")[1].strip()], f"{lab} {v / fJ:.0f}", st.split(",")[0])
              for lab, v, st, inl in INF_BASE]
    x, y = 40, 358
    for col, lab in items:
        parts.append(f'<rect x="{x}" y="{y - 9}" width="11" height="11" fill="{col}"/>')
        parts.append(f'<text x="{x + 16}" y="{y}" class="leg">{lab}</text>')
        x += 16 + 6.3 * len(lab) + 22
        if x > W - 260:
            x, y = 40, y + 20
    x, y = 40, y + 20
    dash = {"dashed": "8,5", "dotted": "2,4", "dashdotted": "8,4,2,4", "densely dotted": "2,2"}
    for col, lab, kind in lines_:
        parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 26}" y2="{y - 4}" stroke="{col}" stroke-width="2" '
                     f'stroke-dasharray="{dash[kind]}"/>')
        parts.append(f'<text x="{x + 32}" y="{y}" class="leg">{lab}</text>')
        x += 32 + 6.3 * len(lab) + 22
        if x > W - 260:
            x, y = 40, y + 20
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"wrote {os.path.relpath(path)}")


write_svg(os.path.join(os.path.dirname(__file__), "..", "slides", "media", "energy_breakdown.svg"))
