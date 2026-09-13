"""Reproduce the energy-scaling model of Pai et al. (2023) SM sec. 2.7, Tables S1-S4
and the photonic-advantage contours of fig. S8C-D, and put the Lightmatter
processor (Ahmed et al., Nature 640, 368, 2025) on the same axes.

Run:  python3 scripts/energy_model.py
"""
import numpy as np

# ----------------------------------------------------------------- Table S1 (1 GHz)
fJ, pJ = 1e-15, 1e-12
E_mod = 1 * fJ        # modulator (LiNbO3 / BTO)
E_sw = 1 * fJ         # switch
E_TIA = 700 * fJ      # transimpedance amp, 7 mW @ 10 GS/s rescaled to 1 GHz
E_DAC = 5 * pJ        # 8-bit DAC, 26 mW @ 5 GS/s rescaled
E_ADC = 1.38 * pJ     # 8-bit ADC, 1.73 mW @ 1.25 GS/s rescaled
E_OP = 100 * fJ       # digital 8-bit op (Horowitz 45 nm, "conservative")
E_mode = 1 * pJ       # 1 mW per optical mode at 1 GHz


def energies(N, M, digital_control_ps=False):
    """Return (E_MVM_dig, E_MVM_alg, E_grad_dig, E_grad_alg) in joules for one
    N x N layer at batch size M, following Tables S2-S4.

    digital_control_ps: use the 'digital control phase shifter' input scheme of
    sec. 2.7.3 (E_DAC -> 0, E_mod -> 16 E_mod)."""
    e_mod, e_dac = (16 * E_mod, 0.0) if digital_control_ps else (E_mod, E_DAC)
    # Table S2 subtasks
    E_in = (2 * e_mod + 2 * e_dac) * M * N
    E_out = (4 * E_TIA + 4 * E_ADC) * M * N
    E_grad_upd = (2 * E_TIA + 10 * E_sw) * 2 * N**2
    E_MVM_dig = 6 * E_OP * M * N**2
    E_ioprep = (6 * E_OP + E_mode) * M * N
    # Table S3 tasks
    E_MVM_alg = 2 * E_ioprep + E_in + E_out
    E_grad_dig = 2 * E_MVM_dig
    # Table S3 prints  E_grad,alg = 2 E_MVM,alg + 2 E_ioprep + 2 E_in + 2 E_out + E_grad,
    # but the per-component counts in Table S4 (8MN E_ADC, 8MN E_TIA, 6MN E_mod, ...)
    # correspond to  2 E_MVM,alg + 2 E_ioprep + E_in + E_grad : forward + backward pass
    # with full I/O, plus a "sum" pass that needs input preparation but no output
    # readout (the gradient is read at the taps by the analog updater).  We take the
    # Table S4 counts as canonical and flag the Table S3 formula as a typo.
    E_grad_alg = 2 * E_MVM_alg + 2 * E_ioprep + E_in + E_grad_upd
    return E_MVM_dig, E_MVM_alg, E_grad_dig, E_grad_alg


def check_table_s4():
    """Table S4 coefficients: verify the per-component counts we reproduce."""
    N, M = 7, 3  # arbitrary
    _, E_MVM_alg, _, E_grad_alg = energies(N, M)
    # Expected from Table S4: in situ MVM = 12MN E_OP + 2MN E_mod + 2MN E_DAC
    #                          + 4MN E_ADC + 4MN E_TIA + 2MN E_mode
    exp_mvm = M * N * (12 * E_OP + 2 * E_mod + 2 * E_DAC + 4 * E_ADC + 4 * E_TIA + 2 * E_mode)
    # in situ VJP/grad = 36MN E_OP + 6MN E_mod + 6MN E_DAC + 8MN E_ADC
    #                    + (8MN + 4N^2) E_TIA + 6MN E_mode  (+ 20 N^2 E_sw, dropped in S4)
    exp_grad = (M * N * (36 * E_OP + 6 * E_mod + 6 * E_DAC + 8 * E_ADC + 8 * E_TIA + 6 * E_mode)
                + 4 * N**2 * E_TIA + 20 * N**2 * E_sw)
    # NB: np.isclose's default atol=1e-8 J would make any pJ comparison pass trivially.
    assert np.isclose(E_MVM_alg, exp_mvm, rtol=1e-12, atol=0), (E_MVM_alg, exp_mvm)
    assert np.isclose(E_grad_alg, exp_grad, rtol=1e-12, atol=0), (E_grad_alg, exp_grad)
    print("Table S4 per-component counts reproduced  OK")
    # Show how far the Table S3 formula as printed deviates from Table S4
    E_in = (2 * E_mod + 2 * E_DAC) * M * N
    E_out = (4 * E_TIA + 4 * E_ADC) * M * N
    s3 = E_grad_alg + E_in + 2 * E_out
    print(f"  Table S3 formula as printed would be {s3 / E_grad_alg:.2f}x larger than Table S4 "
          f"(extra E_in + 2 E_out); Table S4 used below.")


check_table_s4()

# ----------------------------------------------------------- per-element costs
_, e_mvm, _, _ = energies(1, 1)
print(f"\nIn situ MVM cost per (M*N) input element from Table S1: {e_mvm / pJ:.2f} pJ "
      f"(sec. 2.7.4 prose says 'roughly MN x 1.54 pJ'; not derivable from Table S1)")
conv_only = 4 * E_ADC + 4 * E_TIA + 2 * E_DAC + 2 * E_mod
print(f"  of which ADC+TIA+DAC+mod = {conv_only / pJ:.2f} pJ, digital I/O prep = "
      f"{2 * 6 * E_OP / pJ:.2f} pJ, optical power = {2 * E_mode / pJ:.2f} pJ")

# ----------------------------------------------------------- advantage contours
print("\nPhotonic advantage  E_digital / E_in-situ  (Table S1 numbers, 100 fJ/OP digital)")
print("  Inference (MVM) is independent of batch size M:")
for N in (16, 32, 64, 128, 256):
    d, a, _, _ = energies(N, 1)
    d2, a2, _, _ = energies(N, 1, digital_control_ps=True)
    print(f"    N={N:4d}: {d / a:5.2f}x   (with digital-control phase shifters: {d2 / a2:5.2f}x)")


def n_for_advantage(ratio, M, grad=False, **kw):
    for N in range(2, 1025):
        d, a, gd, ga = energies(N, M, **kw)
        r = gd / ga if grad else d / a
        if r >= ratio:
            return N
    return None


print("\n  Training (in situ VJP/grad): smallest N reaching 2x / 4x advantage vs batch M")
print("      M    N(2x)  N(4x)")
for M in (1, 4, 16, 64, 256):
    print(f"    {M:4d}   {n_for_advantage(2, M, grad=True)!s:>5}  {n_for_advantage(4, M, grad=True)!s:>5}")

print("\n  Same, with digital-control phase shifters (no input DAC):")
print("      M    N(2x)  N(4x)")
for M in (1, 4, 16, 64, 256):
    print(f"    {M:4d}   {n_for_advantage(2, M, grad=True, digital_control_ps=True)!s:>5}"
          f"  {n_for_advantage(4, M, grad=True, digital_control_ps=True)!s:>5}")

# ------------------------------------------------ Lightmatter (Nature 2025) on same axes
print("\nLightmatter photonic processor (Ahmed et al. 2025) on the same per-op axes:")
N_lm, cores, f_clk = 128, 4, 500e6
ops_per_s = cores * N_lm**2 * 2 * f_clk
print(f"  4 x 128x128 x 2 ops x 500 MHz = {ops_per_s / 1e12:.1f} TOPS   (paper: 65.5 TOPS)")
P_el, P_opt = 78.0, 1.6
print(f"  full-system energy: {(P_el + P_opt) / ops_per_s / pJ:.2f} pJ/op = "
      f"{2 * (P_el + P_opt) / ops_per_s / pJ:.2f} pJ/MAC")
print(f"  PTC-only (0.25 W, 262 TOPS/W quoted at 2 GHz peak): {1 / 262e12 / fJ:.1f} fJ/op")
# What the Science SM model predicts for an N=128 in situ MVM, per op
d, a, _, _ = energies(128, 1)
ops = 2 * 128**2
print(f"  Science-SM model, N=128 in situ MVM: {a / ops / fJ:.0f} fJ per real op "
      f"(the model's digital baseline: {d / ops / fJ:.0f} fJ per real op, i.e. 6 x 100 fJ per complex MAC)")
print(f"  Per-element I/O budget in the SM model: {e_mvm / pJ:.1f} pJ per input element;"
      f" Lightmatter's realised full-system number is {(P_el + P_opt) / (cores * N_lm * f_clk) / pJ:.0f} pJ "
      f"per vector element per cycle (78 W / (4 x 128 x 500 MHz)).")
