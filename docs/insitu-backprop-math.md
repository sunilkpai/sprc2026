# The math behind in situ backpropagation

A section-by-section breakdown of the Supplementary Materials (SM) of
Pai et al., *Experimentally realized in situ backpropagation for deep learning in
photonic neural networks*, Science 380, 398 (2023), DOI 10.1126/science.ade8450.
Equation numbers (S1 to S14) and algorithm numbers (Alg. 1 to 5) refer to the SM.
Two scripts in `scripts/` check the core claims numerically; their findings are
folded into the relevant sections and collected in section 9.

Contents

1. Physical setting and notation
2. Building blocks: MZI, vector units, matrix unit, reference arm
3. The gradient identity (the heart of the paper)
4. Three ways to read the gradient off the chip
5. The protocol as pseudocode (Alg. 1 to 5)
6. The analog update
7. Energy model and the photonic-advantage argument
8. Noise, systematic error, and the tap-coupling design
9. Errata and convention traps found while checking

---

## 1. Physical setting and notation

A layer of the photonic neural network (PNN) is a unitary $U^{(\ell)}(\boldsymbol\theta,\boldsymbol\phi)\in U(N)$
realised by a mesh of Mach-Zehnder interferometers (MZIs), followed by a
nonlinearity $f^{(\ell)}$ computed digitally:

$$y^{(\ell)} = U^{(\ell)} x^{(\ell)},\qquad x^{(\ell+1)} = f^{(\ell)}(y^{(\ell)}).$$

Every phase shifter $\eta$ (an internal $\theta$ or external $\phi$) sits next to a
3% grating tap, so the optical power $p_\eta=|x_\eta|^2$ at that point can be
imaged. The mesh is reciprocal: light can be sent in from either side.
On the experimental chip $N=6$ modes, 15 MZIs, of which a $4\times4$ block is used as
the matrix unit and the two edge diagonals as the input generator and output
analyzer (fig. S5A). The parameter count per layer is $D_\ell = N(N-1)$ phases
plus $N$ output phases $\gamma$ that are handled off-chip.

Three quantities appear throughout:

| symbol | meaning | where measured |
|---|---|---|
| $x_\eta$ | forward field at phase shifter $\eta$ when $x$ is sent in from the left | tap next to $\eta$ |
| $x_{\mathrm{adj},\eta}$ | backward field at $\eta$ when the adjoint vector $y_{\mathrm{adj}}$ is sent in from the right | same tap |
| $p_{\eta,\pm}$ | tap power when the "sum" field $x - i\,x_{\mathrm{adj}}^{*}e^{i\zeta}$ is sent forward, $\zeta=0,\pi$ | same tap |

---

## 2. Building blocks

### 2.1 Calibration model (S1, S2)

An MZI is characterised by its transmissivity $t=\sin^2\theta$ (the SM's
convention: $\theta$ is twice the internal-arm phase), measured from tap powers as
$t = p_t/(p_r+p_t)$ (S1). The heater response is fit with a cubic in voltage plus a
sinusoid,

$$\theta = p_0 v^3 + p_1 v^2 + p_2 v + p_3,\qquad t = a\sin\theta + b,\tag{S2}$$

and in practice $v^2 = q_0\theta^3+q_1\theta^2+q_2\theta+q_3$ is inverted to set a phase.
The internal $\theta$ shifters are calibrated first by routing light along
"lightwires" (fig. S2A); the external $\phi$ shifters then via "meta-MZI"
structures made of four neighbouring MZIs (fig. S2B). The linear region of
$V(\theta)$ (fig. S2E) matters later: the analog update applies a *voltage*
increment, so $\partial L/\partial V = (\partial\theta/\partial V)(\partial L/\partial\theta)$
needs a constant slope to avoid per-shifter gain correction (sec. 2.1 of the SM).

### 2.2 The MZI (S6, S7, S8)

With a differential external phase $\phi$, a 50/50 coupler, a differential internal
phase $\theta$, and a second 50/50 coupler,

$$T_2(\theta,\phi) = i\begin{pmatrix} e^{i\phi}\sin\tfrac{\theta}{2} & \cos\tfrac{\theta}{2}\\ e^{i\phi}\cos\tfrac{\theta}{2} & -\sin\tfrac{\theta}{2}\end{pmatrix},\qquad \theta\in[0,\pi],\ \phi\in[0,2\pi).\tag{S6}$$

Because the chip uses a *single-arm* internal heater rather than a push-pull pair,
the physical device is $\tilde T_2 = e^{-i\theta/2}T_2$ (S7), a global phase that is
absorbed into the downstream $\gamma$ phases.

The defining capability of an MZI is *nullification*: for any input
$(x_1,x_2)$ there is a setting that puts all the power into one output,

$$\theta = 2\arctan\left|\frac{x_1}{x_2}\right|,\qquad \phi = -\arg\frac{x_1}{x_2}.\tag{S8}$$

`scripts/verify_gradient.py` check [1] confirms that (S8) extinguishes one port
exactly; which port is "top" depends on the coupler phase convention, so the
labelling in the SM should be read as convention-dependent.

### 2.3 Vector units: generators and analyzers (S9, Alg. 1)

A vector unit is a tree of $N-1$ MZIs plus $N$ output phases. Read left to
right it is a *generator*: one input, arbitrary normalised complex output vector,
$|x\rangle = X|0\rangle$. Read right to left it is an *analyzer*: it routes an
arbitrary input vector into a single port, $X^\dagger|x\rangle = |0\rangle$ (S9).
`VEC2PHASE` in Alg. 1 is just (S8) applied sequentially: nullify port $m+1$ into
port $m$, update $x_m \leftarrow e^{i\phi_m}\sin\tfrac{\theta_m}{2}x_m + \cos\tfrac{\theta_m}{2}x_{m+1}$,
repeat. `PHASE2VEC` is the inverse. The last line of `PHASE2VEC`,
`return x exp(-i arg(x_N))`, fixes the unphysical global phase by declaring the
$N$-th element real.

Self-configuration by power minimisation (sweep $\phi$, then $\theta$) works but is
slow with a camera, so the chip instead makes four measurements at
$\theta=\pi/2$, $\phi\in\{0,\pi/2,\pi,3\pi/2\}$ and computes the relative phase as

$$\arg\frac{x_1}{x_2} = \arctan\frac{p_{3\pi/2}-p_{\pi/2}}{p_\pi - p_0},$$

a four-step quadrature (homodyne) measurement done with the MZI itself as the
interferometer (fig. S5G). For a balanced binary tree the whole analyzer settles in
$O(\log N)$ steps because all MZIs in a column are independent.

### 2.4 Matrix unit and the $\gamma$ phases (S10)

Any universal mesh (triangular, rectangular, binary-tree cascade) implements
$U|x\rangle = |y\rangle = Y|0\rangle$ with $N(N-1)/2$ MZIs, i.e. $N(N-1)$ phases
$(\theta_U,\phi_U)$, plus $N$ output phases $\gamma_U$. Multiplying by
$\mathrm{diag}(e^{i\gamma})$ is $O(N)$, so all $\gamma$ bookkeeping in both forward
and backward passes is done on the computer (Alg. 2 line 20, Alg. 3 line 8).
The overall path phase $\langle0|Y^\dagger U X|0\rangle = e^{i\phi_0}$ is not
predicted; it is measured in $O(1)$ with the reference arm.

### 2.5 Reference arm (S11)

Phases only exist relative to something. The $N$-dimensional problem is embedded
in $N+1$ dimensions with one untouched waveguide,

$$\begin{pmatrix} y \\ z\end{pmatrix} = \begin{pmatrix} U & 0\\ 0 & 1\end{pmatrix}\begin{pmatrix} x\\ z\end{pmatrix},\tag{S11}$$

and every vector-unit phase is measured against that mode. In the pseudocode this
is the line `x <- [x sqrt(1-1/N), ||x|| sqrt(1/N)]` (Alg. 2/3 line 7): a fraction
$1/N$ of the power is diverted to the reference. The homodyne alternative
(fig. S5C) splits the reference $N$ ways and interferes it at each output; it is
faster and is what the energy model in section 7 assumes.

---

## 3. The gradient identity

### 3.1 Statement (S3, S4)

Define the adjoint recursion, from the last layer $L$ down to 1,

$$y^{(\ell)}_{\mathrm{adj}} = f^{(\ell)}_{\mathrm{vjp}}\!\left(y^{(\ell)}, x^{(\ell+1)}_{\mathrm{adj}}\right),\qquad x^{(\ell)}_{\mathrm{adj}} = \left(U^{(\ell)}\right)^{T} y^{(\ell)}_{\mathrm{adj}},\tag{S3}$$

seeded with $y^{(L)}_{\mathrm{adj}} = \left(\partial L/\partial x^{(L+1)}\right)^{*}$.
Then for every phase shifter $\eta$ in layer $\ell$,

$$\frac{\partial L}{\partial\eta} = -\,\mathrm{Im}\!\left(x_\eta\, x_{\mathrm{adj},\eta}\right),\tag{S4}$$

the product of the forward field and the backward field *at the same tap*.
The update is then $\eta \leftarrow \eta - \alpha\,\partial L/\partial\eta$ (S4 writes
$+\alpha$; the sign is absorbed into the definition of the step).

### 3.2 Derivation

Write the layer unitary as $U = U_2\, D(\eta)\, U_1$ where $D(\eta)$ is the identity
except for $e^{i\eta}$ on the shifter's waveguide $k$. Then
$\partial U/\partial\eta = U_2\,(i e^{i\eta}P_k)\,U_1$ with $P_k$ the projector
onto mode $k$. For a real loss $L(y)$ define the steepest-ascent direction
$g = 2\,\partial L/\partial y^{*}$ so that $dL = \mathrm{Re}(g^\dagger\,dy)$. Then

$$\frac{\partial L}{\partial\eta} = \mathrm{Re}\!\left(g^\dagger U_2\, i e^{i\eta}P_k U_1 x\right) = \mathrm{Re}\!\left(i\,\overline{(U_2^\dagger g)_k}\; e^{i\eta}(U_1x)_k\right) = -\mathrm{Im}\!\left(a^{*}\,x_\eta\right),$$

with $x_\eta = e^{i\eta}(U_1x)_k$ the forward field just after the shifter and
$a = (U_2^\dagger g)_k$.

Now use reciprocity. Sending a vector $v$ *backward* through a reciprocal
network applies $U^{T}$, not $U^{\dagger}$ (check [2] in the script). So if the
conjugate gradient $y_{\mathrm{adj}} = g^{*}$ is injected at the output, the field
arriving at the shifter from the right is $(U_2^{T} g^{*})_k = \overline{(U_2^\dagger g)_k} = a^{*}$.
That is exactly $x_{\mathrm{adj},\eta}$, giving (S4).

Two remarks that the SM states but does not derive:

- **Tap side does not matter.** If the tap is on the input side of the shifter,
  the forward field loses a factor $e^{i\eta}$ and the backward field gains one.
  The product is unchanged (check [5]).
- **Why the third pass works.** Step 3 of the protocol sends
  $x - i\,x^{*}_{\mathrm{adj}}$ *forward*, with $x_{\mathrm{adj}} = U^{T}y_{\mathrm{adj}}$ the
  backward field measured at the *input*. For a lossless unitary,
  $U_1 x^{*}_{\mathrm{adj}} = U_1 U^{\dagger} y^{*}_{\mathrm{adj}} = D^{\dagger}U_2^{\dagger}y^{*}_{\mathrm{adj}}$,
  so the forward-propagated conjugate adjoint reproduces the conjugate of the
  backward field at *every* tap, not only at the input. This is time-reversal
  symmetry, and it is the step that requires the mesh to be lossless (or at least
  loss-balanced). It is why sec. 2.7.10 finds that beamsplitter and phase errors
  do not corrupt the gradient (they keep $U$ unitary) while loss *imbalance*
  does (check [4]).

### 3.3 The nonlinearity VJP (S5)

For $f(y)=|y|$ with real output, the chain rule gives
$g_y = g_x\odot y/|y|$ for the ascent directions. Under the convention of
section 3.2, where the backward field is the *conjugate* of the ascent direction,

$$y_{\mathrm{adj}} = \mathrm{Re}\!\left(x_{\mathrm{adj}}\right)\odot\frac{y^{*}}{|y|}.$$

The SM prints (S5) with $y/|y|$, no conjugate. `verify_gradient.py` check [6]
builds a two-layer network, $y_1 = U_1 x$, $x_2 = |y_1|$, $y_2 = U_2 x_2$, and compares
the layer-1 tap gradients with finite differences. With $y^{*}/|y|$ the error is
$4\times10^{-10}$. With $y/|y|$ as printed it is $0.9$, i.e. the gradient direction is
essentially random, because each adjoint component acquires a spurious phase
$2\arg y_i$. This did not affect the experiment (the VJP was computed by JAX
autodiff, SM sec. 2.2 and Alg. 4 line 13), but it will bite anyone implementing
from the SM text. The $\mathrm{Re}(\cdot)$ is there because $x_2$ is real-valued, so
only the real part of its adjoint is meaningful.

For holomorphic $f$ the VJP simplifies to $y_{\mathrm{adj}} = f'(y)\odot x_{\mathrm{adj}}$
(SM remark after step 5); $|y|$ is not holomorphic, hence (S5).

---

## 4. Three ways to read the gradient off the chip

All three are algebra on $p_\eta(\zeta) = \left|x_\eta - i\,x^{*}_{\mathrm{adj},\eta}e^{i\zeta}\right|^{2}$:

$$p_\eta(\zeta) = |x_\eta|^2 + |x_{\mathrm{adj},\eta}|^2 - 2\,\mathrm{Im}\!\left(x_\eta x_{\mathrm{adj},\eta}\,e^{i\zeta}\right).$$

| scheme | measurement | formula | where in SM |
|---|---|---|---|
| digital subtraction | three passes: $p_\eta$, $p_{\eta,\mathrm{adj}}$, $p_{\eta,+}=p_\eta(0)$ | $\partial L/\partial\eta = \tfrac12\left(p_{\eta,+}-p_\eta-p_{\eta,\mathrm{adj}}\right)$ | step 4(a), Alg. 4 line 15, fig. 3 |
| analog toggle | one pass, toggle $\zeta$ between $0$ and $\pi$ $K$ times, high-pass filter | $\partial L/\partial\eta = \tfrac14\left(p_\eta(0)-p_\eta(\pi)\right)$ | step 4(b), S12, fig. 2C |
| direct | not measurable: needs the complex fields | $-\mathrm{Im}(x_\eta x_{\mathrm{adj},\eta})$ | S4 |

Check [3] evaluates all of these on a random mesh and finds agreement with finite
differences to $8\times10^{-10}$. The "$-4\mathcal R(i x_\eta x^{*}_{\mathrm{adj},\eta}) = 4\mathcal I(x_\eta x_{\mathrm{adj},\eta})$"
line inside (S12) has a sign/conjugation slip; the operational result
$(p_+-p_-)/4$ on the next line is correct and is what was used.

Two scale factors sit in front of everything:

- **Wirtinger factor.** With $y_{\mathrm{adj}} = (\partial L/\partial y)^{*}$ in the
  Wirtinger sense one gets half of $g$; the measured gradient is then half the true
  one. This is a global constant absorbed into $\alpha$.
- **Normalisation.** The chip always launches unit-power vectors. Alg. 2/3 store
  the norms $q = \|x\|^2$, $q_{\mathrm{adj}} = \|y_{\mathrm{adj}}\|^2$ and rescale the
  outputs by $\sqrt q$ and the powers by $q$. The sum vector is launched as
  $(x - i x^{*}_{\mathrm{adj}})/\sqrt2$ after equal splitting (fig. 2B), so the analog
  gradient carries a further factor that is the only per-example scale not
  absorbable into the learning rate (SM sec. 2.6.2). Note that this per-vector
  max-abs normalisation is precisely a block floating point with block size $N$;
  see `lightmatter-comparison.md`.

---

## 5. The protocol as pseudocode

The five algorithms divide the work between chip and computer as follows.

| Alg. | function | on chip | on computer | complexity |
|---|---|---|---|---|
| 1 | `VEC2PHASE`, `PHASE2VEC`, `NORM` | – | sequential nullification (S8), inverse, norms | $O(N)$ |
| 2 | `MESHFORWARD` | launch $\theta_X,\phi_X$; propagate; tap powers $p_{\theta_i},p_{\phi_i}$; self-configure analyzer | reference-arm embedding, `VEC2PHASE`, `PHASE2VEC`, $\gamma$, rescale | $O(N)$ digital per vector |
| 3 | `MESHBACKWARD` | same, right to left, with $\tilde T_2^{T}$ per MZI | `VEC2PHASE(y* e^{i\gamma})` | $O(N)$ |
| 4 | `INSITUBACKPROP`, `INSITUGRADIENT` | forward passes, backward passes, sum passes | nonlinearities, VJPs (JAX), subtraction or LP/HP filtering | 3 optical passes per layer per example |
| 5 | `INSITUMINIBATCHTRAIN` | – | sampling, Adam, minibatch averaging | – |

Points worth noticing in Alg. 4:

- Line 3, `F± = X ∓ i X_adj`: the sum inputs for $\zeta = 0,\pi$ are prepared once.
- Lines 4 to 17 are a `parfor` over layers. Once the forward and backward passes
  have been run for all layers, the gradient (sum) passes for all layers are
  independent and can run simultaneously on separate meshes.
- Line 6, `d = max NORM(X)`: in analog mode every example in the batch is scaled
  by the *same* constant so the photocurrents add linearly (sec. 2.6.2), whereas in
  digital mode each example is normalised individually (line 12 to 14).
- Line 9 and 10: batch averaging is done by the integrator (`/4`, `/2` are the
  factors of section 4), the difference by the high-pass filter.

The chip demonstration used $M=1$ (pure SGD) with Adam, and fig. S4G shows why:
for this dataset the *normalised* gradient error grows with minibatch size because
the true averaged gradient shrinks while the measurement error does not.

---

## 6. The analog update (sec. 2.6)

(S12) is the analog-toggle line of section 4. What the section adds is the
signal chain:

1. Tap photodiode current $I = R\,p_\eta(\zeta(t))$, with $\zeta$ a square wave.
2. Transimpedance amplifier, gain $R_f\sim100\ \mathrm{k\Omega}$; the gain (or the APD
   bias $V_b$) *is* the learning rate, per shifter.
3. Either a high-pass filter and comparator extracting the AC amplitude
   $(p_+-p_-)/4$ (fig. S6A, single-example), or a sample-and-hold pair capturing
   $p_+$ and $p_-$ into a unity-gain differential amplifier (fig. S6B, batched).
4. A gated integrator and sample-and-hold write $\Delta V_\eta$ onto the phase
   shifter.

Batching is free by linearity: stream $M$ examples at the modulator rate
(1 GHz assumed), toggle $\zeta$ at $1/M$ GHz, and the integrator sums
$M$ contributions before one update. Fig. S7 demonstrates this with discrete
op-amps for a single MZI at $M=1,2,4$; gradient errors were on par with the
camera-based digital subtraction.

The consequence for memory: a purely analog SGD update stores nothing but the
phase voltages. Adam needs the history vector $h$ (Alg. 5 line 4), i.e. $O(D_\ell)$
analog or non-volatile memory per layer (sec. 2.2, 2.5).

---

## 7. Energy model and the photonic-advantage argument (sec. 2.7)

### 7.1 Ingredients (Table S1, 1 GHz clock)

| component | energy | note |
|---|---|---|
| modulator $E_{\mathrm{mod}}$ | 1 fJ | 16 fJ if a "digital control phase shifter" replaces the DAC |
| switch $E_{\mathrm{sw}}$ | $\le$1 fJ | |
| TIA $E_{\mathrm{TIA}}$ | 700 fJ | 7 mW at 10 GS/s, rescaled |
| 8-bit DAC $E_{\mathrm{DAC}}$ | 5 pJ | 26 mW at 5 GS/s, rescaled; 0 with digital-control shifters |
| 8-bit ADC $E_{\mathrm{ADC}}$ | 1.38 pJ | 1.73 mW at 1.25 GS/s, rescaled |
| digital 8-bit op $E_{\mathrm{OP}}$ | 100 fJ | "conservative", 45 nm, includes communication |
| optical power $E_{\mathrm{mode}}$ | 1 pJ | 1 mW per mode |

### 7.2 Composition (Tables S2 to S4)

Per layer, batch $M$, width $N$:

$$E_{\mathrm{MVM,dig}} = 6MN^2E_{\mathrm{OP}}$$

$$E_{\mathrm{MVM,alg}} = MN\left[12E_{\mathrm{OP}} + 2E_{\mathrm{mod}} + 2E_{\mathrm{DAC}} + 4E_{\mathrm{ADC}} + 4E_{\mathrm{TIA}} + 2E_{\mathrm{mode}}\right]$$

$$E_{\mathrm{grad,dig}} = 12MN^2E_{\mathrm{OP}}$$

$$E_{\mathrm{grad,alg}} = MN\left[36E_{\mathrm{OP}} + 6E_{\mathrm{mod}} + 6E_{\mathrm{DAC}} + 8E_{\mathrm{ADC}} + 8E_{\mathrm{TIA}} + 6E_{\mathrm{mode}}\right] + 4N^2E_{\mathrm{TIA}} + 20N^2E_{\mathrm{sw}}$$

The factor 6 in the digital cost is a complex multiply-accumulate (3 real
multiplies, 5 adds by Gauss's trick, rounded to 6 ops). The $4N^2E_{\mathrm{TIA}}$
term is the per-shifter gradient updater firing once per batch: it is the only
$N^2$ term on the photonic side and it is what makes batching necessary for
training advantage.

`scripts/energy_model.py` rebuilds these from Tables S1 to S3 and checks the
per-component counts of Table S4. Two things came out of that:

- **Table S3 disagrees with Table S4.** S3 prints
  $E_{\mathrm{grad,alg}} = 2E_{\mathrm{MVM,alg}} + 2E_{\mathrm{ioprep}} + 2E_{\mathrm{in}} + 2E_{\mathrm{out}} + E_{\mathrm{grad}}$,
  which gives $16MN$ ADCs, not $8MN$. The S4 counts correspond to
  $2E_{\mathrm{MVM,alg}} + 2E_{\mathrm{ioprep}} + E_{\mathrm{in}} + E_{\mathrm{grad}}$:
  forward and backward passes with full I/O, plus a sum pass that needs input
  preparation but no output readout because the gradient is read at the taps. That
  is the physically right count and the script uses it; the S3 formula is 1.42×
  higher.
- **The "2× at N=64, M≥16" claim of sec. 2.7.12 reproduces only with digital-control
  phase shifters** ($E_{\mathrm{DAC}}=0$, $E_{\mathrm{mod}}=16$ fJ). With those
  assumptions the smallest $N$ reaching 2× training advantage is 63 at $M=16$,
  48 at $M=64$, 45 at $M=256$; 4× needs $N\ge213$ at $M=16$. With an explicit 8-bit
  DAC per input the 2× threshold moves to $N=133$ at $M=16$. No advantage exists at
  $M\le4$ for any $N\le1024$.

### 7.3 What the model says

Per input element, the in situ MVM costs 21.5 pJ with Table S1 numbers, of which
18.3 pJ is conversion (DAC, ADC, TIA, modulator), 1.2 pJ digital I/O preparation
and 2 pJ optical power. Digital costs $0.6N$ pJ per input element. So inference
advantage is independent of $M$ and crosses 2× at $N\approx72$ (39 with
digital-control shifters), 4× at $N\approx144$ (77). The prose figure of
"roughly $MN\times1.54$ pJ" in sec. 2.7.4 is not derivable from Table S1.

The structural claim survives all of this: photonic cost is $O(MN)$ conversions,
digital cost is $O(MN^2)$ ops, and the crossover is set by the ratio
$E_{\mathrm{conversion}}/E_{\mathrm{OP}}\approx100$ to $200$. Everything else in
sec. 2.7 is about pushing that ratio down: digital-control phase shifters to kill
the input DAC (2.7.3), homodyne readout to avoid self-configuration (2.4.3), MEMS
shifters for zero static power (2.7.9), tunable TIA gain so the update needs no
DAC at all (2.7.6), and a laser-per-16-subdies budget to keep wall-plug losses off
the books (2.7.8).

### 7.4 Latency

Time of flight is about 1 ns (several cm of waveguide). The system is
modulator-limited at 1 GHz, and for advantage it must be *batch*-limited: the
whole argument assumes weight-stationary streaming of $M$ vectors through a fixed
mesh (2.7.1).

---

## 8. Noise, systematic error, and the tap design (sec. 2.7.10, 2.7.11, 2.7.13)

### 8.1 Which errors matter

| error | affects gradient? | why |
|---|---|---|
| beamsplitter split-ratio error, in-mesh phase error | no | $U$ stays unitary; the gradient is of the *actual* device |
| balanced loss | no | a global scale |
| loss imbalance across MZI arms | yes | breaks time-reversal symmetry (section 3.2) |
| input/output phase and amplitude quantisation | yes | modelled as $\sigma_{\mathrm{phase}}=0.025$ for an 8-bit ADC at $N=64$ |
| tap photodiode noise | yes, dominant | enters every tap measurement; modelled by $s_{\mathrm{tap}}$ |

Sec. 2.1 adds the empirical observation that gradient error grows toward
convergence (the gradient shrinks, the noise does not) and with minibatch size
(fig. S4G, H), and that output *phase* measurement error alone costs an order of
magnitude in gradient error.

### 8.2 Tap photodiode SNR (S13)

For a Si-Ge avalanche photodiode with responsivity $R$, gain $M$, excess-noise
factor $F_A(M) = k_AM + (1-k_A)(2-1/M)$, bandwidth $\Delta f$, dark current $I_d$
and TIA feedback resistance $R_f$,

$$\mathrm{SNR}(P) = \frac{(RPM)^2}{4kTF_n\Delta f/R_f + 2qM^2F_A(M)(RP+I_d)\Delta f}\approx\frac{RP}{2qF_A(M)\Delta f},$$

$$s_{\mathrm{tap}} := \sigma_{\mathrm{noise}}/\sqrt P \approx \sqrt{2qF_A(M)\Delta f/R}.\tag{S13}$$

$s_{\mathrm{tap}}$ is the noise-to-signal proportionality constant fed to the
simulations. With Table S5 (R = 0.85 A/W, M = 10, k_A = 0.05, 1 GHz) the MNIST
simulation needs $\ge500$ nW per tap ($s_{\mathrm{tap}}=0.0078$ shot-noise only,
$0.012$ with TIA input-referred noise); 1 µW gives $0.005$ / $0.007$.

### 8.3 Tap-coupling recursion (S14)

To deliver the same power $p_t$ to every tap across $C = 2N$ columns of a
rectangular mesh with loss $\alpha$ dB per MZI, the coupling ratio must grow along
the mesh:

$$P_c = 10^{-\alpha/20}(1-\xi_{c-1})P_{c-1},\qquad \xi_c = \frac{p_tN}{P_c} = \xi\,10^{-c\alpha/20}\prod_{k=1}^{c-1}(1-\xi_k)^{-1}.\tag{S14}$$

Fig. S6F evaluates this for $N=64$, $p_t/P=10^{-3}$ (30 dB), $\alpha = 0.1, 0.2$ dB:
the required coupling stays small unless component loss exceeds about 0.2 dB
per MZI, where the product term blows up.

### 8.4 All-analog inference and loss scaling (2.7.13)

Feeding $L$ optical layers in series costs $0.2NL$ dB at 0.2 dB per MZI; feeding
each layer from an equal split of the source costs $0.2N + 10\log_{10}L$ dB. For
$N=64$, $L=10$ that is 128 dB versus 22.8 dB. The "distributed" architecture of
fig. S9 also keeps switchable taps between layers so the same chip can run the
hybrid (debug) mode needed for in situ backpropagation.

---

## 9. Errata and convention traps

Collected from the checks above, ordered by consequence.

1. **(S5) conjugation.** Under the S3/S4 convention the $|y|$ VJP must use
   $y^{*}/|y|$. As printed it produces a wrong gradient direction. Experiment
   unaffected (JAX autodiff).
2. **Table S3 vs Table S4.** The training-energy formula in S3 double counts
   $E_{\mathrm{in}}$ and $E_{\mathrm{out}}$ relative to the component counts in S4
   (1.42× on $E_{\mathrm{grad,alg}}$). S4 is physically right.
3. **Fig. S8D headline** ("2× at $N=64$, $M\ge16$") requires the digital-control
   phase shifter assumption of sec. 2.7.3, which is not stated next to the claim.
4. **Sec. 2.7.4 "1.54 pJ per element"** is not derivable from Table S1
   (21.5 pJ, or 11.5 pJ with digital-control shifters).
5. **(S12) intermediate line** has $+\mathcal I$ where (S4) and the final
   $(p_+-p_-)/4$ have $-\mathcal I$. Typo; the operational formula is right.
6. **(S4) update sign** is written $\eta_t = \eta_{t-1} + \alpha\,\partial L/\partial\eta$;
   descent needs a minus, absorbed into $\alpha$.
7. **"Top port" in (S8)** and "nullify port 5 through 2" in fig. S5F depend on the
   directional-coupler phase convention; the formula is right, the label is not
   portable.
8. **Wirtinger factor of 2** between $(\partial L/\partial y)^{*}$ and the ascent
   direction; a global scale, absorbed into $\alpha$.

Nothing in the list changes a result of the paper. Items 1 to 4 would change what
a reader building from the SM gets.
