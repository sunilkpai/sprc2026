# In situ backpropagation (2023) against Lightmatter's photonic processor (2025)

This note puts the analysis in `insitu-backprop-math.md` next to
Ahmed et al., *Universal photonic artificial intelligence acceleration*,
Nature 640, 368 (2025), DOI 10.1038/s41586-025-08854-x, from Lightmatter.
The Nature main text is paywalled; what follows is built from its open
Supplementary Information (sections I to VI, Tables S1 to S4), the Lightmatter
blog post announcing the paper, the ABFP numerics paper it relies on
(Basumallik et al., arXiv:2205.06287), and secondary coverage. Anything I could
not confirm from those sources is marked as such.

Contents

1. What Lightmatter built
2. The math of a Lightmatter multiply-accumulate
3. Side by side
4. Where the two analyses agree
5. Where they diverge, and why in situ backprop is a mesh-specific idea
6. What the 2023 energy model got right and wrong in hindsight
7. Energy breakdown and a 4-bit projection
8. Open questions this raises for training on the Lightmatter-class architecture

---

## 1. What Lightmatter built

| item | value | source |
|---|---|---|
| package | six chips: 2 digital control dies (DCI) + 4 photonic tensor cores (PTC), 3D stacked on an interposer, 80 × 65 mm | SI III |
| digital dies | GlobalFoundries 12 nm, ~50 B transistors total, quad-core RISC-V control processor running NuttX | SI V, blog |
| photonic dies | 4 × (128 × 128) PTC, each 14 × 24.96 mm, ~1 M photonic components in total, 300 mm foundry silicon photonics | SI, Physics World |
| PTC contents | 128 input vector modulators (VMOD, 10-bit), 128 × 128 weight modulators (WMOD, 7-bit), TIAs and 11-bit vector ADCs (VADC) | SI I, V |
| clock | 500 MHz sustained (clock-tree limited), 2 GHz peak design | SI II, VI |
| throughput | 65.5 TOPS at 500 MHz = 4 × 128² × 2 × 0.5 GHz; 262 TOPS at 2 GHz | SI VI |
| power | 78 W electrical + 1.6 W optical (full system); 0.25 W for the PTC encode/weight path alone | SI VI |
| efficiency | 0.84 TOPS/W full system, 1.75 peak; 262 TOPS/W PTC-only | SI Table S4 |
| numerics | ABFP16: bfloat16 scales, 10-bit inputs, 7-bit weights, 11-bit ADC, measured overamplification G = 1.86 | SI I |
| weight update path | ~10 ns settle, 25 MHz transfer from DCI to PTC, double-buffered (buffer switch non-functional in silicon) | SI II |
| calibration | hourly, 10-step sequence over laser power regulators, vector stabilisers, weight and vector modulators, TIAs and ADCs | SI V |
| latency | ~200 ps time of flight; milliseconds end-to-end through PCIe 4.0 and the digital pipeline | SI VI |
| workloads | ResNet-18 (CIFAR-10, Imagenette, Imagewoof), BERT-tiny (IMDB, SQuAD), SegNet, Atari DQN, MNIST MLPs; "near 32-bit float" accuracy | SI II, GitHub repo |
| training | none on chip; weights are quantised offline, optionally with quantisation-aware training | SI I |

## 2. The math of a Lightmatter multiply-accumulate

### 2.1 Physical operation

From SI section I and V: each activation element $x_j$ is encoded by a vector
modulator (an MZI held at quadrature by a "vector stabiliser") and *split into
$N$ row lines*. On each row line it passes a weight modulator whose weight value
is encoded as the *ratio of currents drawn from the differential row lines*.
The row line photocurrents are summed and the difference is read by a TIA and
an 11-bit ADC:

$$y_i = \sum_{j=1}^{N} w_{ij}\,x_j,\qquad w_{ij} \propto \frac{I^{+}_{ij}-I^{-}_{ij}}{I^{+}_{ij}+I^{-}_{ij}}.$$

So this is an intensity-domain, incoherent crossbar. Multiplication is
attenuation, accumulation is Kirchhoff's current law, sign is dual-rail. There is
no interference between different $j$ and therefore no phase to stabilise across
the array, only per-element bias points (the quadrature lock of each VMOD and the
slope calibration of each WMOD). The Lightmatter patent literature and Nick
Harris's blog describe the same picture. I could not confirm from open sources
whether the 128-way fan-out is a passive splitter tree or a bus with taps, but the
math is the same either way.

### 2.2 Numerics: ABFP with overamplification

The PTC computes with fixed-point numbers, so every vector is scaled first.
With $s_x = \max_j|x_j|$ and $s_{w;i} = \max_j|w_{ij}|$ (both bfloat16),
$\hat x = x/s_x$, $\hat w_i = w_i/s_{w;i}$, and symmetric quantisers
$Q_b(\alpha) = \mathrm{round}\!\left(\mathrm{clamp}_{[-1,1]}(\alpha)/\delta_b\right)\delta_b$,
$\delta_b = 1/(2^{b-1}-1)$:

$$y_i = \frac{s_{w;i}\,s_x}{G}\;Q_{11}\!\left\{G\sum_{j=1}^{N}Q_7(\hat w_{ij})\,Q_{10}(\hat x_j)\right\}.$$

The braces are the PTC; the scale multiply is in the digital die.
Two facts drive the design:

- A dot product of $b_W$- and $b_X$-bit operands over $N$ terms needs
  $b_W + b_X + \log_2N - 1$ bits to be exact: 22 bits for 8/8/128. An 11-bit ADC
  therefore throws away the low half.
- Analog gain $G$ (more laser power and more TIA gain) shifts which bits the ADC
  sees: each doubling of $G$ recovers one LSB at the cost of one MSB, and the
  MSBs are rarely populated in DNN dot products. The ABFP paper's Table II shows
  ResNet-50 at tile width 128 going from 0.7% top-1 at $G=1$ to 75.2% at $G=8$
  (FP32: 76.1%). The Nature SI Table S1 shows the same at $G=4$ for the MLPerf
  set, and the silicon achieved $G = 1.86$.

The noise model is additive and value-independent: uniform (ABFP paper) or
Gaussian (SI IV) on the ADC output, with the measured error distribution having
logistic tails attributed to the interplay of vector normalisation, quantisation
and aggregation. Quantisation-aware training uses a straight-through estimator,
$\partial Q(\alpha)/\partial\alpha = 1$ inside the clamp range, and injects Gaussian
noise at the MVM output during fine-tuning.

## 3. Side by side

| | Pai et al. 2023 (Science) | Ahmed et al. 2025 (Nature) |
|---|---|---|
| encoding | coherent complex field, phase carried by a reference arm | intensity, sign by differential rails, no phase |
| linear operator | unitary $U\in U(N)$ from $N(N-1)$ phases; general matrices need SVD (two meshes + attenuators) | arbitrary real $N\times N$ matrix, one cell per element |
| parameter to matrix-element map | many-to-many: every $\theta,\phi$ touches $O(N)$ entries of $U$ | one-to-one: $w_{ij}$ is the cell |
| accumulation | interference inside the mesh | photocurrent summation on row lines |
| size demonstrated | $4\times4$ block of a $6\times6$ mesh (15 MZIs) | $4\times(128\times128)$, ~1 M photonic elements |
| rate | camera plus XY stage: 31 h for 1000 iterations | 500 MHz vector rate |
| readout | 3% grating taps imaged on an IR camera; homodyne proposed | integrated photodiodes, TIA, 11-bit ADC |
| input precision | camera-limited; 8-bit DAC assumed in the model | 10-bit DAC |
| weight precision | thermal heaters with cubic calibration, ~50 mW each | 7-bit weight DAC per cell, hourly recalibration |
| numerics | per-vector power normalisation (block floating point with block = the whole vector), unit power launched | ABFP: per-row and per-vector bfloat16 scales, 7/10/11-bit fixed point, gain $G$ |
| what runs on chip | MVM forward, MVM backward, gradient measurement | MVM forward only |
| training | in situ: gradient of every phase from 3 optical passes, no model of the device needed | offline: QAT with STE and injected noise on a GPU; weights loaded once |
| error model | tap APD shot/thermal noise $s_{\mathrm{tap}}$, loss imbalance, I/O phase error | additive ADC-referred noise, logistic tails; loss and splitter error calibrated out per cell |
| energy, realised | not applicable (proof of concept) | 1.2 pJ/op full system, 3.8 fJ/op PTC-only |
| energy, projected | 45 fJ/op for an $N=128$ in situ MVM with segmented phase shifters (84 with 8-bit input DACs); 300 fJ/op digital baseline | – |
| headline claim | training advantage $\ge2\times$ at $N\ge64$, $M\ge16$ (with digital-control shifters) | 0.84 TOPS/W full-system, comparable to A100 (0.78) on peak FP16 vs ABFP16; near-FP32 accuracy |

## 4. Where the two analyses agree

**Conversions dominate, not optics.** The 2023 model puts 18 of 21.5 pJ per
input element into DAC, ADC and TIA and 2 pJ into light. Lightmatter's own
accounting splits 262 TOPS/W (PTC encode/weight path) from 0.84 TOPS/W (whole
package): a factor of 300 between the optical multiply and the system that feeds
it. Both papers therefore reach the same design rule: amortise every conversion
over as wide a vector as possible ($O(MN)$ conversions for $O(MN^2)$ ops), and keep
weights stationary while streaming activations. The 128-wide tile in silicon is
exactly the $N\approx64$ to $128$ regime the 2023 tables identify as the crossover.

**Analog gain is a precision knob.** Sec. 2.6 and 2.7.6 of the 2023 SM propose a
tunable TIA gain (or APD bias) per gradient updater, argued from learning-rate
range and tap-coupling variation. ABFP's $G$ is the same lever applied to
inference readout, with a sharper argument: it moves the ADC's window down the
bit ladder. The SM's requirement of 50 to 60 dBΩ TIA gain to reach usable
$\Delta V$ is the training-side version of Lightmatter's "increase both TIA gain
and laser power".

**Per-vector normalisation is block floating point.** Alg. 2 and 3 of the 2023
SM normalise every launched vector to unit power and store $q=\|x\|^2$ on the
computer; the adjoint and sum vectors get the same treatment. That is ABFP with
block size $N$ and a single scale per vector. Lightmatter adds a per-row weight
scale, which a mesh cannot have (a unitary has no per-row freedom), and a
bfloat16 rather than float storage of the scales.

**Noise is additive at the detector and the network tolerates it.** The 2023
grid search over $s_{\mathrm{tap}}$, phase and amplitude error and loss variation
finds MNIST training survives $s_{\mathrm{tap}}\lesssim0.01$; ABFP's uniform-LSB
noise model and Lightmatter's Gaussian-with-logistic-tails measurement land on
the same conclusion for inference: the last bit or two are noise, and the
network does not care if the dynamic range is managed.

**Everything nonlinear stays digital.** Both architectures are hybrid by design;
the 2023 SM argues (sec. 2.5) that an optical nonlinearity buys nothing because
the $O(N)$ conversion cost is already paid, and Lightmatter runs ReLU, softmax,
norms in the DCI.

## 5. Where they diverge, and why in situ backprop is a mesh-specific idea

**In a crossbar the gradient is an outer product; in a mesh it is not.**
For $y = Wx$ with one cell per weight,

$$\frac{\partial L}{\partial w_{ij}} = \delta_i\,x_j,\qquad \delta = \left(\frac{\partial L}{\partial y}\right),$$

so once the backward MVM $\delta_{\mathrm{in}} = W^{T}\delta$ is available, the
weight gradient is free: it is the outer product of two vectors the digital side
already holds. There is nothing to measure at the cells. For a mesh, the
parameters are phases inside interferometers,

$$\frac{\partial L}{\partial\eta} = \mathrm{Re}\!\left(g^\dagger\frac{\partial U}{\partial\eta}x\right),\qquad \frac{\partial U}{\partial\eta} = U_2\,(ie^{i\eta}P_k)\,U_1,$$

and $\partial U/\partial\eta$ is a dense $N\times N$ matrix that depends on every
other phase and on every fabrication error. Computing it digitally needs a
faithful model of the imperfect device; measuring $-\mathrm{Im}(x_\eta x_{\mathrm{adj},\eta})$
at the tap needs none. That is the whole reason the 2023 paper exists: it is the
mesh's answer to a problem the crossbar never had.

**Bidirectionality is available to the mesh and not to the crossbar.** The
adjoint pass $x_{\mathrm{adj}} = U^{T}y_{\mathrm{adj}}$ is one more optical pass
because the mesh is reciprocal and lossless; time reversal then makes the sum
pass reproduce the backward field at every tap (section 3.2 of the math note).
A crossbar terminates in photodiodes, so there is no backward light; the backward
MVM is a *transposed weight load*. On the 2025 silicon that costs a 128 × 128 × 7-bit
transfer over a 25 MHz weight interface: 0.1 to 1 ms depending on the word width,
which the SI does not give, against 2 ns for a forward MVM. Training on that chip
would be weight-load-bound by five orders of magnitude unless a transposed read
path is added to the cells.

**Precision needs differ by an order of magnitude.** The 2023 data show gradient
error rising toward convergence and with minibatch size because the signal
shrinks under fixed noise (fig. S4G, H), and an 8-bit output ADC already costs
0.025 rad of phase error at $N=64$. Lightmatter needed 11 output bits plus gain
to get inference to near-FP32 on ResNet and BERT. Gradients are smaller than
activations and their errors compound over steps, so an on-chip training
analogue of ABFP would need per-vector scaling of the *adjoint* vectors (which
Alg. 3 already does) and probably a larger $G$ during the gradient pass than the
inference pass. Neither paper measures what output bit depth in situ training
needs at $N=128$.

**Stability cost.** A coherent mesh needs $N(N-1)$ phases stable to a small
fraction of a radian *relative to each other* and a reference arm, for the
duration of a forward, backward and sum pass. Lightmatter's crossbar needs each
of ~1 M elements stable at its own bias point, with an hourly recalibration and
per-cell slope matching, but no cross-element coherence. That asymmetry, more
than any energy argument, is why the crossbar reached 128 wide in 2025 and meshes
in the literature are still at 8 to 64.

**Unitary versus general.** A mesh gives $U(N)$ natively; a general real matrix
needs $U\Sigma V^{\dagger}$, two meshes and a column of attenuators, doubling
depth and loss. The 2023 energy comparison charges the digital side for a
*complex* MVM (6 ops per MAC) to be fair to the mesh's native operation, but
real DNN layers are real, so a like-for-like comparison would halve the digital
baseline or double the photonic one. Lightmatter's PTC does real matrices
directly; its ABFP scales are per row precisely because a general matrix has
per-row dynamic range.

## 6. What the 2023 energy model got right and wrong, in hindsight

Right:

- The conversion-dominated cost structure and the $O(MN)$ vs $O(MN^2)$ argument.
- The weight-stationary batching requirement (sec. 2.7.1); Lightmatter's whole
  throughput section (SI II) is about hiding the ~10 ns weight settle behind
  hundreds of MVPs.
- Homodyne rather than self-configuration readout (fig. S5C); the VMOD/VSTAB
  quadrature lock in the 2025 chip is this.
- 8-bit-class DAC/ADC energies of order 1 to 5 pJ per sample at GHz rates. The
  2025 references (a 9-bit DAC driver, 9.4- and 10.1-ENOB pipelined-SAR ADCs)
  are the same class of part.

Wrong or dated:

- **100 fJ per digital op is now pessimistic by 3 to 10×.** H100 INT8 runs at
  about 0.35 pJ/op at the wall and B200 FP4 at about 0.11 pJ/op (peak dense
  throughput over board power; see the talk's slide 6). Lightmatter's realised
  1.2 pJ/op full-system number is *above* both, which is consistent with the
  2023 model's own warning that without the input-DAC fix and at $N\le128$ the
  advantage is at best a few times, and disappears once the digital side is
  charged at modern rather than 45 nm energies.
- **1 GHz modulator rate** was assumed; silicon delivered 500 MHz sustained
  because of the digital clock tree, not the optics.
- **50 mW thermal phase shifters** were flagged as the thing to replace with MEMS.
  Lightmatter shipped ~1 M actively regulated elements with hourly calibration;
  the static-power problem was engineered around, not eliminated, and the SI
  does not break out phase-shifter power.
- **The 6-op complex MAC baseline** flatters the mesh (previous section).
- **Camera as ADC.** The 2023 error budget is dominated by a 3% tap and a
  Bobcat IR camera; the 2025 chip's dominant error is ADC-referred and logistic
  tailed. The $s_{\mathrm{tap}}$ APD model in sec. 2.7.10 is the right form for an
  integrated gradient-tap detector but has not been tested against silicon.

## 7. Energy breakdown and a 4-bit projection

`scripts/energy_breakdown.py` stacks the per-component energies of the 2023
model (Table S4 counts, Table S1 values) next to Lightmatter's measured split and
projects the model to 4-bit operation. Basis: fJ per real op, an $N\times N$ MVM
being $2N^2$ ops and an in situ VJP/grad step $4N^2$ ops per example, batch cost
divided by $M$. The figure is written to `figs/energy_breakdown.tex` and appears
in the deck.

**No DACs anywhere on the photonic side.** Every photonic bar assumes segmented
("digital control") phase shifters, SM sec. 2.7.3, for *both* the input vector
and the mesh weights: a $b$-bit value is written as $b$ binary-weighted phase
segments driven straight from logic. The SM charges this as
$E_{\mathrm{mod}}\to16E_{\mathrm{mod}}$ at 8 bits and $E_{\mathrm{DAC}}\to0$; the
4-bit projection scales the segment count with $b$. What it costs instead is
interconnect: an $N\times N$ mesh has $N(N-1)$ phases, so segmented weights need
about $N^2 b$ vertical contacts from the control die, a bump or hybrid-bond array
per weight cell.

| bits | contacts, $N=128$ | area at 40 µm microbump pitch | area at 10 µm hybrid-bond pitch |
|---|---|---|---|
| 8 | 130 048 | 208 mm² | 13 mm² |
| 4 | 65 024 | 104 mm² | 6.5 mm² |

Lightmatter's PTC is 349 mm² with 6 000 bumps (SI III) and instead streams
weights serially through 16 384 on-die 7-bit DACs at 25 MHz. At 4 bits the
segmented array fits under a PTC-sized die even at microbump pitch; at 8 bits it
needs hybrid bonding. Weight *write* energy is then $b\,E_{\mathrm{mod}}$ per
changed weight, paid once per batch and negligible against the per-op terms.

**Projection rules.** ADC energy scales as $2^{b}$ (Walden-type; the Nature SI
quotes the thermal-limited $2^{2\Delta b}$ for its own ADC, which would be 16×
more optimistic over four bits). Digital op energy is divided by 3 going to 4
bits, the B200 FP4 to H100 INT8 ratio at the wall, and doubled going to 16-bit
class. Optical power scales as $4^{b}$, since a shot-noise-limited amplitude SNR
of $2^{b}$ needs power $\propto4^{b}$; the SM's 1 mW per mode with a 3% tap is
about a 9-bit shot-noise budget at 1 GHz, so it is treated as the 8-bit base.
TIA and switch energies are bit-independent.

**4 bits is an inference story, not a training one.** Gradients need precision:
the SM's own data show gradient error growing toward convergence and with batch
size when gradients are read per example (fig. S4G, H), and the deck's own line
is that gradients want 16-bit dynamic range. So the training panel is *not*
projected to 4 bits. What it does project is the thing the SM's analog update
was designed around (sec. 2.6.2, Alg. 4 lines 9 and 10): **batch-integrated
gradient readout**. The tap photocurrent is integrated over the $M$ examples of
a batch and digitised *once per phase shifter per batch*. Two consequences:

- the gradient ADC runs at $f_{\mathrm{clk}}/M$, where 12- to 16-bit converters
  are slow and cheap (a 20 pJ SAR conversion at a few MS/s), and its energy is
  amortised over $4N^2M$ ops;
- the integrated signal grows as $M$ while shot noise grows as $\sqrt M$, so the
  gradient gains $\tfrac12\log_2 M$ bits over a single-example readout with no
  extra light: 4 bits at $M=256$, 6 bits at $M=4096$.

The per-example passes (forward, backward, sum) still read activations at 8
bits and 1 GHz; that part is inference-class. The precision-hungry part is the
one that runs slowly.

**Weight precision is free per op; activation precision is not.** A mesh holds
its weights as phases, so a 4-bit weight costs contacts and holding power but
nothing per multiply. The per-op energy is set by the activation path: the
input encoder's segment count, the output ADC's bits, and the optical power for
the output SNR. So W4A8, which is what MXFP4-weight models such as Kimi K3
(MXFP4 weights, MXFP8 activations) and DeepSeek's FP8 pipeline actually run,
costs the same as W8A8 in this model: 45 fJ per op, 5× under a B200 at FP8
(about 220 fJ per op, 4.5 PFLOPS over 1 kW). Only W4A4 collapses the ADC and
reaches 14 fJ per op, and Jalapeño's 52 fJ is an MXFP4-by-MXFP4 number, so
the like-for-like comparisons are W4A8 against B200 FP8 and W4A4 against
Jalapeño.

**Inference, $N=128$, fJ per op**

| scenario | digital prep | encode | ADC | TIA | optical | rest | total |
|---|---|---|---|---|---|---|---|
| Envise measured (encode includes its weight DACs) | | 3.8 | | | 24.4 | 1187 | 1215 |
| SM 8-bit, segmented PS | 4.7 | 0.1 | 21.6 | 10.9 | 7.8 | | 45 |
| SM 4-bit, segmented PS | 1.6 | 0.1 | 1.3 | 10.9 | 0.03 | | 14 |

Digital reference lines: the model's own 8-bit baseline 300, H100 INT8 350,
B200 FP4 110, and OpenAI's Jalapeño MXFP4 ASIC at 52 (13.4 PFLOPS at a 700 W
package, Hot Chips 2026). Groq's LPU (750 TOPS INT8 at about 300 W, 14 nm, 230 MB SRAM
per chip) sits at 400 fJ per op: it buys decode latency with SRAM-resident
weights, not energy per op. Jalapeño is the honest 4-bit target: the photonic
4-bit engine at 14 fJ per op is 3.7× under it before any shell is counted.

**Training, $N=128$, fJ per op**

| scenario | digital prep | encode | ADC (activations) | TIA (per example) | gradient readout | optical | total |
|---|---|---|---|---|---|---|---|
| SM 8-bit, per-example gradients, $M=16$ | 7.0 | 0.2 | 21.6 | 54.7 (incl. $4N^2E_{\mathrm{TIA}}$ updater) | in TIA column | 11.7 | 96 |
| batch-integrated 12-bit gradients, $M=256$ | 7.0 | 0.2 | 21.6 | 10.9 | 24.2 | 11.7 | 76 |
| batch-integrated 12-bit gradients, $M=4096$ | 7.0 | 0.2 | 21.6 | 10.9 | 1.5 | 11.7 | 53 |
| same, bf16-class activations: 8-bit mantissa, block exponent, 11-bit ADC at 7 pJ | 7.0 | 0.3 | 109 | 10.9 | 1.5 | 11.7 | 141 |

Gradient readout = $N(N-1)\,(20\ \mathrm{pJ\ ADC} + 5\ \mathrm{pJ\ TIA{+}integrator})/(4N^2M)$.
Digital reference lines: model 8-bit 300, H100 INT8 350, Cerebras WSE-3 FP16
at 180 (125 PFLOPS at about 23 kW). Cerebras is the relevant training
comparator because it keeps the weights on the wafer, which is the digital
answer to the same footprint problem the mesh has.

**bf16 activations.** Training runs bf16, and bf16 is an 8-bit significand
with an 8-bit exponent. The per-vector normalisation in Algs. 2 and 3 supplies
a block exponent, so the analog path carries the 8-bit mantissa; what it needs
beyond the 8-bit inference readout is accumulation headroom, which is why the
row above uses an Envise-class 11-bit ADC (Lagos 2022, 10.1 ENOB at 500 MS/s,
about 7 pJ per sample) rather than a 16-bit converter that does not exist at
this rate. The ADC term goes from 22 to 110 fJ per op and nothing else moves:
141 fJ per op, 1.3× under Cerebras FP16 and 5× under H100 FP16. What is lost
against true bf16 is the per-element exponent, which matters most for
gradients with outliers; ABFP's per-row and per-vector scales recover part of
it, and the gradient readout itself is batch-integrated at 12 to 16 bits, so
the exposure is in the activations of the forward and backward passes.

What the two tables say:

- **With segmented phase shifters the encode path is 0.1 fJ per op, 40× under
  Envise's measured 3.8 fJ per op.** The 2023 model's expensive term at 8 bits is
  then the ADC (48% of inference), and at 4 bits it is the TIA. There is no
  "best-of" bar because nothing in Envise's measured encode path is cheaper than
  the segmented-PS model; what Envise adds is the reality check that the rest of
  its system costs 1.19 pJ per op.
- **4-bit inference lands at 14 fJ per op, 7× under the 4-bit digital line.**
  The ADC collapses 16×, the light by 256, the four remaining segments cost
  nothing. What is left is the TIA, fixed by photodiode current and bandwidth.
- **8-bit training is 3 to 5× under the 8-bit digital line and the lever is
  batch size.** The per-phase gradient updater ($4N^2E_{\mathrm{TIA}}$ per batch)
  is 44 of the 55 fJ in the TIA row at $M=16$ and 11 of 22 at $M=64$. Inference
  advantage is independent of $M$; training advantage is set by it.
- **Batch integration is where training wins.** Reading gradients per example
  (the $M=16$ bar, and the 2023 experiment) leaves the $4N^2E_{\mathrm{TIA}}$
  updater term and forces high-rate readout. Integrating over the batch turns
  the gradient ADC into a slow 12- to 16-bit part at $f/M$, costs 1.5 fJ per op
  at $M=4096$, and adds 6 bits of gradient precision from averaging alone. The
  training bar then sits at 53 fJ per op, 6× under the 8-bit digital line and
  13× under FP16, with 8-bit-class light. An earlier version of this note
  projected 12-bit gradients by scaling the tap light by 256; that was the wrong
  lever, since the same bits come from time.
- **What integration does not fix.** Shot and thermal noise average down;
  systematic errors (tap-coupling variation, loss imbalance, phase-to-voltage
  slope) do not, and they are what the SM's fig. S4G actually measured when
  the minibatch error grew. Calibrated taps and the linear-slope requirement of
  sec. 2.1 are the price of the batch win. The integrator also has to hold a sum
  of $M$ toggled contributions without saturating; the AC-coupled scheme of
  fig. S6A only accumulates the difference term, which is what makes $M$ in the
  thousands plausible.

- **The 4-bit optical number for inference is a limit, not a design.** 4 µW per
  mode is below what a practical link budget with 0.2 dB per MZI over 256
  columns allows; the bar shows what shot noise permits, not what the S14
  recursion permits.

### 7.1 Is the ADC fast enough?

Two different ADCs are in play and they should not share a number.

| readout | rate per channel | bits needed | what exists (Murmann survey class) | energy per sample |
|---|---|---|---|---|
| activations, inference and the forward/backward passes | $f_{\mathrm{clk}}$, 1 GS/s | 4 to 8 | SAR and time-interleaved SAR, 8 bit at 1 to 2 GS/s | ~1 to 2 pJ; the SM's 1.38 pJ is at the good end |
| activations at Envise's precision | 500 MS/s | 11 (ENOB ~10) | pipelined SAR, 10.1 ENOB at 500 MS/s (Lagos 2022, cited in the Nature SI) | ~7 pJ, five times the SM's 8-bit number |
| activations at 12 bits and 1 GS/s | 1 GS/s | 12 | jitter-limited: $\mathrm{SNR}_{\mathrm{jitter}} = -20\log_{10}(2\pi f_{\mathrm{in}}\sigma_j)$ gives 11.4 bits at 100 fs rms, 8 bits at 1 ps | 20 to 50 pJ and a state-of-the-art clock |
| gradients, batch-integrated | $f_{\mathrm{clk}}/M$: 4 MS/s at $M=256$, 0.24 MS/s at $M=4096$ | 12 to 16 | any precision SAR; trivial | ~20 pJ, amortised over $4N^2M$ ops |

So: 8 bits at 1 GS/s is fine and is what the inference model assumes. Ten to
eleven bits at the same rate is where Envise sits and costs 5× the SM's ADC
number, which moves the inference ADC term from 22 to about 110 fJ per op at
$N=128$, still under the 300 fJ digital line. Twelve bits at 1 GS/s is not a
sensible target for any analog accelerator, and it does not have to be, because
the only thing that needs 12 bits is read $M$ times more slowly.

### 7.2 What batch size is realistic

The relevant $M$ is the number of vectors that pass through one weight block
between updates, per device. Current practice:

| workload | global batch per step | vectors per weight per device per step |
|---|---|---|
| LLM pretraining (Llama 3 405B, DeepSeek-V3 class) | 4M to 60M tokens | a micro-batch of 1 to 8 sequences × 4k to 8k tokens: **4k to 64k** tokens, each a vector through every projection |
| LLM fine-tuning | 0.1M to 4M tokens | 1k to 16k |
| CNN classification (ResNet-50) | 256 to 4 096 images | im2col makes every spatial position a vector: **10⁴ to 10⁶** per layer |
| the 2023 experiment | 1 | 1 |
| the 2023 SM energy analysis | 16 to 256 | 16 to 256 |

The SM's $M=16$ to $256$ was conservative by one to three orders of magnitude.
At $M \sim 10^4$ the per-batch terms ($4N^2$ updaters, gradient ADCs, weight
writes, and the block reloads of `footprint-tdm.md`) are all below a femtojoule
per op, and the training energy is the cost of three inference-class passes.
The constraint moves to the integrator: $10^4$ examples at 1 GHz is a 10 µs
integration window per gradient sample, easy for a gated integrator, but it is
also 10 µs of phase and laser stability demanded of the whole mesh, which is the
stability requirement from `rack-design.md` restated as a training spec.

## 8. Open questions this raises

1. What is the minimum output bit depth for in situ gradient measurement at
   $N=128$ to match offline QAT accuracy on ResNet-18 or BERT-tiny? The 2023
   grid search stops at $N=64$ and MNIST.
2. Can the ABFP gain trick be applied per gradient pass, with $G$ chosen from the
   adjoint vector norm, to keep $-\mathrm{Im}(x_\eta x_{\mathrm{adj},\eta})$ in the ADC
   window as the gradient shrinks toward convergence? Alg. 4 line 6 already picks
   one scale per batch; making it adaptive is a small change to the protocol.
3. For a crossbar, the only thing in situ backprop offers is the backward MVM
   $W^{T}\delta$. Is a transposed read path (a second set of row lines along
   columns, or a switchable fan-in) cheaper than a 25 MHz weight reload? If yes,
   the crossbar gets on-chip training without any of the mesh's coherence burden.
4. Does loss imbalance at the ~0.05 dB level, which the 2023 simulations tolerate,
   hold at 128 columns where the S14 coupling recursion is already fighting
   0.2 dB per MZI?
5. The PPPI metric in the Nature SI, $T\cdot2^{B}/(P\cdot A)$, weights precision
   exponentially. An $N=64$ in situ mesh with 8-bit I/O would score roughly
   $2^{77-64}\approx8000\times$ lower on precision alone before any throughput or
   area term. Whether that metric is the right one for training hardware, where
   the update precision and not the readout precision sets the floor, is worth
   arguing in the talk.

## Sources

- Pai et al., Science 380, 398 (2023), Supplementary Materials, local copy
  `refs/pai2023_science_sm.pdf`.
- Ahmed et al., Nature 640, 368 (2025), Supplementary Information, local copy
  `refs/ahmed2025_nature_si.pdf`.
- Basumallik et al., *Adaptive block floating-point for analog deep learning
  hardware*, arXiv:2205.06287, local copy `refs/basumallik2022_abfp_arxiv.pdf`.
- Lightmatter blog, *A new kind of computer*, https://lightmatter.co/blog/a-new-kind-of-computer/
- Lightmatter data repository, https://github.com/lightmatter-ai/upaia-paper-2025
- Physics World, https://physicsworld.com/a/photonic-computer-chips-perform-as-well-as-purely-electronic-counterparts-say-researchers/
- Anastasi in Tech, *Lightmatter's photonic superchip*, https://anastasiintech.substack.com/p/lightmatters-photonic-superchip
