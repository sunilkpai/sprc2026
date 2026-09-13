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
7. Open questions this raises for training on the Lightmatter-class architecture

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
| energy, projected | 84 fJ/op for an $N=128$ in situ MVM (Table S1 numbers); 300 fJ/op digital baseline | – |
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

## 7. Open questions this raises

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
