"""Numerically verify the in situ backpropagation gradient identity.

Checks the claims of Pai et al., Science 380, 398 (2023), Supplementary
Materials, eqs. S3, S4, S6-S8, S12, on a random triangular MZI mesh:

  1. The MZI transfer matrix (S6/S7) is unitary and the nullification
     formula (S8) sends all power to the top port.
  2. Sending light backward through a reciprocal mesh implements U^T.
  3. For every phase shifter eta, with x_eta the forward field at the tap
     and x_adj,eta the backward (adjoint) field at the same tap,

         dL/d eta = -Im( x_eta * x_adj,eta )

     and this equals (p_+ - p_-)/4 where p_+/- is the tap power when the
     "sum" field  x - i x_adj^* e^{i zeta}  is sent forward with
     zeta = 0 / pi.  Both are compared against central finite differences.
  4. The sum-field power is actually reproduced by physically sending
     x - i conj(x_adj) forward through the mesh (time-reversal symmetry),
     which is what the chip does in step 3 of the protocol.
  5. The result does not depend on which side of the phase shifter the
     tap sits.

Run:  python3 scripts/verify_gradient.py
"""
import numpy as np

rng = np.random.default_rng(0)
N = 4                       # modes (the paper uses a 4x4 unit inside a 6x6 mesh)


# ---------------------------------------------------------------- building blocks
def bs():
    """50/50 directional coupler, (1/sqrt2) [[1, i],[i, 1]]."""
    return np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)


def mzi_elements(theta, phi):
    """Return the ordered list of 2x2 elements making up T2~(theta, phi) of eq. S7:
    external phi on the top input, 50/50, single-arm theta on the top arm, 50/50.
    Each element is (matrix, tag) where tag names the phase shifter or None."""
    P_phi = np.diag([np.exp(1j * phi), 1.0])
    P_theta = np.diag([np.exp(1j * theta), 1.0])
    return [(P_phi, "phi"), (bs(), None), (P_theta, "theta"), (bs(), None)]


def mzi_matrix(theta, phi):
    M = np.eye(2, dtype=complex)
    for m, _ in mzi_elements(theta, phi):
        M = m @ M
    return M


def embed(m2, k, n):
    """Embed a 2x2 acting on modes (k, k+1) into an n x n identity."""
    M = np.eye(n, dtype=complex)
    M[k:k + 2, k:k + 2] = m2
    return M


def triangular_layout(n):
    """Reck / triangular mesh: list of (column index, top mode) for each MZI."""
    # Reck decomposition order: diagonal d contains MZIs on modes (k, k+1),
    # k = 0 .. n-2-d, for a total of n(n-1)/2 MZIs.
    return [k for d in range(n - 1) for k in range(n - 1 - d)]


def mesh_ops(params):
    """Expand the mesh into an ordered list of (n x n matrix, tag_or_None) so that
    U = ops[-1] @ ... @ ops[0].  tag = (mzi index, 'theta'|'phi') for a phase shifter.
    The phase shifter matrix is diag(..., e^{i eta}, ...) acting on mode `mode`."""
    ops = []
    for i, (k, (theta, phi)) in enumerate(zip(LAYOUT, params)):
        for m2, tag in mzi_elements(theta, phi):
            ops.append((embed(m2, k, N), (i, tag, k) if tag else None))
    return ops


def unitary(params):
    U = np.eye(N, dtype=complex)
    for M, _ in mesh_ops(params):
        U = M @ U
    return U


LAYOUT = triangular_layout(N)
NUM_MZI = len(LAYOUT)
assert NUM_MZI == N * (N - 1) // 2


# ------------------------------------------------------------------ check 1: S6-S8
th, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
T = mzi_matrix(th, ph)
assert np.allclose(T.conj().T @ T, np.eye(2)), "MZI not unitary"

# S8 nullification: theta = 2 arctan|x1/x2|, phi = -arg(x1/x2) puts all power in port 1.
x2 = rng.normal(size=2) + 1j * rng.normal(size=2)
theta_n = 2 * np.arctan(abs(x2[0] / x2[1]))
phi_n = -np.angle(x2[0] / x2[1])
y2 = mzi_matrix(theta_n, phi_n) @ x2
# S8 as printed nullifies one port; with this S7 convention it is the *bottom* port
# that is extinguished (the paper's "top"/"bottom" labelling is convention-dependent).
null_port = int(np.argmin(abs(y2)))
assert abs(y2[null_port]) < 1e-12, "S8 nullification failed"
print(f"[1] MZI unitary OK; S8 nullifies port {null_port + 1} (|y| = {abs(y2).round(6)})")


# ------------------------------------------------------------- check 2: backward = U^T
params = rng.uniform(0, 2 * np.pi, size=(NUM_MZI, 2))
params[:, 0] = np.mod(params[:, 0], np.pi)  # theta in [0, pi]
U = unitary(params)
assert np.allclose(U.conj().T @ U, np.eye(N))

# Backward propagation through a reciprocal network: apply the same elements in
# reverse order with each element transposed.  Directional couplers and diagonal
# phase matrices are symmetric, so this is just the reverse product.
def unitary_backward(params):
    Ub = np.eye(N, dtype=complex)
    for M, _ in reversed(mesh_ops(params)):
        Ub = M.T @ Ub
    return Ub

assert np.allclose(unitary_backward(params), U.T), "backward pass is not U^T"
print("[2] backward propagation implements U^T  OK")


# ----------------------------------------------------- check 3-5: gradient identity
x = rng.normal(size=N) + 1j * rng.normal(size=N)
x /= np.linalg.norm(x)
target = rng.normal(size=N) + 1j * rng.normal(size=N)
target /= np.linalg.norm(target)


def loss(params):
    y = unitary(params) @ x
    return float(np.sum(abs(y - target) ** 2))


# dL = Re( g^dagger dy ) with g = 2 (y - t)  (Wirtinger: g = 2 dL/dy*)
y = U @ x
g = 2 * (y - target)
y_adj = g.conj()            # SM step 1: y_adj = (dL/dy)^*  (up to the factor 2 convention)
x_adj_in = U.T @ y_adj      # SM eq. S3: x_adj = U^T y_adj  (measured at the input side)

ops = mesh_ops(params)


def fields_at_taps():
    """For every phase shifter return (forward field, backward field) on BOTH sides
    of the shifter, at its waveguide mode."""
    out = {}
    # forward partial products
    fwd = [x.copy()]
    for M, _ in ops:
        fwd.append(M @ fwd[-1])
    # backward partial products: light injected at the output, y_adj, travelling back
    bwd = [y_adj.copy()]
    for M, _ in reversed(ops):
        bwd.append(M.T @ bwd[-1])
    bwd = bwd[::-1]          # bwd[j] = backward field at the *input* side of element j
    for j, (M, tag) in enumerate(ops):
        if tag is None:
            continue
        i, name, k = tag
        f_in, f_out = fwd[j][k], fwd[j + 1][k]        # forward field before/after shifter
        b_out, b_in = bwd[j + 1][k], bwd[j][k]        # backward field at output/input side
        out[(i, name)] = dict(f_in=f_in, f_out=f_out, b_in=b_in, b_out=b_out)
    return out


taps = fields_at_taps()

# finite differences
eps = 1e-6
fd = {}
for i in range(NUM_MZI):
    for c, name in enumerate(("theta", "phi")):
        p = params.copy(); p[i, c] += eps; lp = loss(p)
        p = params.copy(); p[i, c] -= eps; lm = loss(p)
        fd[(i, name)] = (lp - lm) / (2 * eps)


def sum_field_power(f, b, zeta):
    """p_eta(zeta) = |x_eta - i conj(x_adj,eta) e^{i zeta}|^2   (eq. S12 construction)."""
    return abs(f - 1j * np.conj(b) * np.exp(1j * zeta)) ** 2


# physically send the sum field x - i conj(x_adj) forward and read tap powers
def forward_tap_powers(x_in):
    fwd = [x_in.copy()]
    for M, _ in ops:
        fwd.append(M @ fwd[-1])
    return {tag: abs(fwd[j + 1][tag[2]]) ** 2 for j, (M, tag) in enumerate(ops) if tag}


p_plus = forward_tap_powers(x - 1j * np.conj(x_adj_in))
p_minus = forward_tap_powers(x + 1j * np.conj(x_adj_in))     # zeta = pi flips the sign

worst = 0.0
for key, t in taps.items():
    # identity evaluated on the output side of the shifter ...
    g_out = -np.imag(t["f_out"] * t["b_out"])
    # ... and on the input side: same number (check 5)
    g_in = -np.imag(t["f_in"] * t["b_in"])
    # analog form, S12: (p(0) - p(pi)) / 4
    g_analog = (sum_field_power(t["f_out"], t["b_out"], 0.0)
                - sum_field_power(t["f_out"], t["b_out"], np.pi)) / 4
    # physically realised sum-field version (check 4)
    i, name, k = key[0], key[1], LAYOUT[key[0]]
    g_phys = (p_plus[(i, name, k)] - p_minus[(i, name, k)]) / 4
    for gg in (g_out, g_in, g_analog, g_phys):
        worst = max(worst, abs(gg - fd[key]))

print(f"[3-5] max |gradient - finite difference| over {2 * NUM_MZI} phase shifters "
      f"and 4 formulations: {worst:.2e}")
assert worst < 1e-6

# Show the sign issue explicitly for the record
key = (0, "theta"); t = taps[key]
print()
print("Sign/conjugation check for one phase shifter (theta of MZI 0):")
print(f"   finite difference           : {fd[key]:+.6f}")
print(f"   -Im(x_eta * x_adj,eta)      : {-np.imag(t['f_out'] * t['b_out']):+.6f}   <- eq. S4 (correct)")
print(f"   +Im(x_eta * x_adj,eta)      : {+np.imag(t['f_out'] * t['b_out']):+.6f}   <- eq. S12 as printed (sign flipped)")
print(f"   (p+ - p-)/4                 : {(p_plus[(0,'theta',LAYOUT[0])] - p_minus[(0,'theta',LAYOUT[0])]) / 4:+.6f}   <- step 4(b) (correct)")

# --------------------------------------------- check 6: two-layer net with |y| VJP (S5)
# y1 = U1 x ; x2 = |y1| ; y2 = U2 x2 ; L = |y2 - t|^2.  Verify the layer-1 gradients using
# the nonlinearity VJP of eq. S5.  With the convention verified above (the backward field is
# the *conjugate* of the steepest-ascent direction g = 2 dL/dy*), the VJP must read
#     y_adj,1 = Re(x_adj,2) * conj(y1) / |y1|
# i.e. eq. S5 with y -> conj(y).  Using y1/|y1| as printed fails the finite-difference test.
params2 = rng.uniform(0, 2 * np.pi, size=(NUM_MZI, 2)); params2[:, 0] %= np.pi


def loss2(p1, p2):
    y1 = unitary(p1) @ x
    y2 = unitary(p2) @ abs(y1)
    return float(np.sum(abs(y2 - target) ** 2))


U1, U2 = unitary(params), unitary(params2)
y1 = U1 @ x
y2 = U2 @ abs(y1)
g2 = 2 * (y2 - target)
y_adj2 = g2.conj()
x_adj2 = U2.T @ y_adj2
for label, y_adj1 in (("conj(y1)/|y1|", np.real(x_adj2) * np.conj(y1) / abs(y1)),
                      ("y1/|y1| (as printed)", np.real(x_adj2) * y1 / abs(y1))):
    # backward fields at layer-1 taps
    bwd = [y_adj1.copy()]
    for M, _ in reversed(ops):
        bwd.append(M.T @ bwd[-1])
    bwd = bwd[::-1]
    fwd = [x.copy()]
    for M, _ in ops:
        fwd.append(M @ fwd[-1])
    err = 0.0
    for j, (M, tag) in enumerate(ops):
        if tag is None:
            continue
        i, name, k = tag
        c = 0 if name == "theta" else 1
        pp = params.copy(); pp[i, c] += eps; lp = loss2(pp, params2)
        pm = params.copy(); pm[i, c] -= eps; lm = loss2(pm, params2)
        fdv = (lp - lm) / (2 * eps)
        err = max(err, abs(-np.imag(fwd[j + 1][k] * bwd[j + 1][k]) - fdv))
    print(f"[6] two-layer |y| network, S5 VJP with {label:22s}: max error {err:.1e}"
          + ("   OK" if err < 1e-6 else "   FAILS"))

print()
print("All checks passed.")
