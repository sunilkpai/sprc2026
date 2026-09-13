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
  * optical power       ~ 4^b   for inference (shot-noise-limited amplitude SNR), i.e. /256;
                         unchanged for training, where sec. 2.7.10 sets the tap power by
                         gradient SNR (>= 0.5 uW per tap), not by output bit depth.
  * TIA, modulators, switches: bit-independent.

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

KEYS = ["digital I/O prep", "modulator", "DAC", "ADC", "TIA", "optical power", "switches",
        "rest of system"]


def per_op(comp, ops, M=1):
    return {k: comp.get(k, 0.0) / ops / M for k in KEYS}


M_BIG = 64
inf = {
    "Envise meas.": {**{k: 0.0 for k in KEYS}, "modulator": LM_ENC, "optical power": LM_OPT,
                     "rest of system": LM_REST},
    "SM 8-bit": per_op(components(N, 1, **SEG)[0], OPS_INF),
    "SM 4-bit": per_op(components(N, 1, **SEG, **FOUR_BIT_INF)[0], OPS_INF),
}
train = {
    f"SM 8-bit M{M_TRAIN}": per_op(components(N, M_TRAIN, **SEG)[1], OPS_TRAIN, M_TRAIN),
    f"SM 4-bit M{M_TRAIN}": per_op(components(N, M_TRAIN, **SEG, **FOUR_BIT)[1], OPS_TRAIN, M_TRAIN),
    f"SM 4-bit M{M_BIG}": per_op(components(N, M_BIG, **SEG, **FOUR_BIT)[1], OPS_TRAIN, M_BIG),
}
digital = {"model 8-bit": 6 * DEFAULT["E_OP"] / 2, "model 4-bit": 6 * FOUR_BIT["E_OP"] / 2,
           "H100 INT8": 0.35 * pJ, "B200 FP4": 0.11 * pJ}
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
          ("TIA", "ink2", "TIA + updater"),
          ("optical power", "amber!40", "optical power"),
          ("switches", "mist", "switches"),
          ("rest of system", "ink!25", f"Envise DCI, rest of system: {LM_REST / fJ:.0f}, off scale")]
YMAX = 400.0


def axis(name, data, title, at=None, legend=False, ylabel=True):
    cats = list(data)
    lines = []
    opts = [f"name={name}", "width=0.5\\textwidth", "height=3.5cm", "ybar stacked",
            "bar width=13pt", f"ymin=0, ymax={YMAX:.0f}", "ylabel near ticks",
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
    # digital baselines: constant plots across the symbolic categories, labelled in the legend
    c0, c1 = cats[0], cats[-1]
    for label, val, style in (("model 8-bit digital", digital["model 8-bit"], "dashed, color=slate"),
                              ("H100 INT8 wall", digital["H100 INT8"], "dotted, color=slate"),
                              ("B200 FP4 wall", digital["B200 FP4"], "dashdotted, color=slate"),
                              ("model 4-bit digital", digital["model 4-bit"], "dashed, color=amber")):
        y = val / fJ
        fp = "" if legend else "forget plot, "
        lines.append(f"\\addplot[{fp}sharp plot, stack plots=false, {style}, thick, no markers, "
                     f"line legend] "
                     f"coordinates {{({c0},{y:.0f}) ({c1},{y:.0f})}};")
        if legend:
            lines.append(f"\\addlegendentry{{{label} {y:.0f}}}")
    # totals above each stack
    for c in cats:
        if c == "Envise meas.":
            continue
        tot = sum(data[c].values()) / fJ
        lines.append(f"\\node[font=\\tiny, color=ink, anchor=south, inner sep=1pt] at "
                     f"(axis cs:{c},{tot:.1f}) {{{tot:.0f}}};")
    lines.append("\\end{axis}")
    return "\n".join(lines)


tex = "\n".join([
    "% Generated by scripts/energy_breakdown.py -- do not edit by hand.",
    "\\begin{tikzpicture}",
    axis("inf", inf, f"Inference: $N={N}$ MVM", legend=True),
    axis("train", train, f"Training: in situ VJP/grad, $N={N}$", at="inf", ylabel=False),
    "\\end{tikzpicture}",
])
out = os.path.join(os.path.dirname(__file__), "..", "figs", "energy_breakdown.tex")
with open(out, "w") as f:
    f.write(tex + "\n")
print(f"\nwrote {os.path.relpath(out)}")
