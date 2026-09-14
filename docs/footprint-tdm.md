# Silicon footprint: the mesh's real problem, and what time-multiplexing buys

A discussion note. The energy argument in `lightmatter-comparison.md` and the
rack argument in `rack-design.md` both assume a 128-wide mesh exists. The
harder question is whether it fits, and whether a mesh should hold its weights
in silicon at all. Opticore's "temporal mapping" (their technology page, built on
Hamerly et al., *Phys. Rev. X* 9, 021032, 2019 and Chen et al., *Nat. Photonics*
17, 723, 2023) is the cleanest statement of the alternative, so it is the foil.

Contents

1. How much silicon a weight costs
2. Why: one device per parameter, and the device is 100 µm long
3. What temporal mapping does instead
4. Throughput per device is the same; area and memory traffic are not
5. Combining the two: time-multiplexing the mesh at block level
6. The energy accounting once weights move
7. What this changes in the talk's argument

---

## 1. How much silicon a weight costs

| implementation | area per stored weight | throughput per mm² | note |
|---|---|---|---|
| 2023 chip MZI (625 µm × ~127 µm) | ~0.08 mm² | – | SM 1.2; thermal shifters, deep trenches, taps |
| compact MZI (200 µm × 50 µm pitch) | ~0.01 mm² | – | typical foundry PDK, EO or MEMS shifter |
| Lightmatter PTC, 128 × 128 in 349 mm² | 0.021 mm² | 0.047 TOPS/mm² at 500 MHz, 0.19 at 2 GHz | Nature 2025 SI, 65.5 TOPS over 4 × 349 mm² |
| B200 (two ~800 mm² dies, ~9 PFLOPS dense FP4) | ~10⁻⁴ mm² per MAC unit incl. SRAM (order of magnitude) | ~5.6 TOPS/mm² | datasheet peak over die area |
| Cerebras WSE-3 (46 225 mm², 44 GB SRAM, 125 PFLOPS FP16) | ~8 × 10⁻⁶ mm² per stored 8-bit weight | ~2.7 TOPS/mm² | the digital way to make weights resident: spend a wafer |

A photonic weight cell is 100 to 1 000× the area of a digital multiply-accumulate
with its local SRAM, and the realised photonic tensor core delivers 30 to 120×
less throughput per square millimetre than a GPU die. A 128-wide rectangular mesh
built from the 2023 chip's MZIs would be 80 mm long and 16 mm high, 1 300 mm²,
larger than a reticle (858 mm²). With compact cells it is about 250 mm², one
reticle for one 128 × 128 layer. That is the silicon waste. The energy per op can
be excellent and the chip still loses on cost per op, because cost is area.

Cerebras is the useful comparison for "weights in silicon": it keeps 44 GB on
the wafer, about 5.5 × 10⁹ 8-bit weights, at 21 PB/s, and that is why its
decode is not HBM-bound. A photonic mesh holding the same weights at 0.01 mm²
each would cover 55 000 m². The mesh is three orders of magnitude behind SRAM
per stored weight, and SRAM is itself two orders behind HBM.

## 2. Why: one device per parameter, and the device is 100 µm long

The mesh stores its weights as the phase state of $N(N-1)$ interferometers. Each
interferometer needs two couplers and two phase shifters, and the shortest
useful phase shifter on silicon is tens of micrometres (thermal: ~100 µm; MEMS or
EO: 50 to 500 µm depending on $V_\pi L$). The wavelength sets the floor: a
waveguide is ~0.5 µm wide, a bend radius ~5 µm, so the smallest 2 × 2 building
block is ~10⁻³ mm², and nothing on the roadmap takes it to 10⁻⁴. Meanwhile the
transistor MAC keeps shrinking. "One device per parameter" is the mesh's
structural disadvantage, and it is the same one Lightmatter's crossbar has:
16 384 weight modulators per core, each with its own DAC and calibration loop.

The mesh gets something for that silicon: the weights are *stationary*. Once
programmed, no bit of weight moves for the whole batch, which is why the 2023
energy tables have no weight-fetch term at all and why in situ backprop can
update $N^2$ parameters with $N^2$ local analog loops. The footprint and the
zero weight traffic are the same design decision.

## 3. What temporal mapping does instead

Photoelectric multiplication (Hamerly 2019): a weight and an input arrive at a
balanced homodyne detector as two coherent optical fields; the photocurrent is
proportional to their product, $I \propto \mathrm{Re}(E_x E_w^{*})$, sign
included. Accumulate the photocurrent over $N$ time steps and the detector has
computed a dot product. Stream $x_j$ in time through one modulator, fan it out to
$N$ detectors, stream $w_{ij}$ in time through one modulator per detector $i$,
and $N$ time steps give the full $N \times N$ matrix-vector product:

$$y_i = \sum_{j=1}^{N} w_{ij}\,x_j \quad\text{with}\quad O(N)\ \text{modulators},\ O(N)\ \text{detectors},\ N\ \text{time steps}.$$

No device holds a weight. The weights live in HBM, are read out as bits, and
become pulses on the way to the detector. Opticore's page says exactly this:
"an optical modulator can activate tens of billions of parameters per second,"
and "all parameters can be dynamically programmed for training." The area per
parameter is zero and $N$ is set by how many modulators and detectors fit, not
by $N^2$.

## 4. Throughput per device is the same; area and memory traffic are not

Count MACs per device per clock:

| | devices | MACs per step | MACs per device per step |
|---|---|---|---|
| mesh, $N$ wide | $N^2$ | $N^2$ | 1 |
| temporal mapping, $N$ detectors | $2N$ | $N$ | ½ |

Time-multiplexing does not make photonics more productive per device. What it
does is move the multiplexing dimension from space to time, and time is cheap
in optics: a modulator runs at 10 to 100 GS/s where a mesh phase settles at
1 GS/s at best, so a temporally mapped device delivers 10 to 100× the MACs per
device per second, and there are $N$ of them per reticle instead of $N^2$.
The trade is that every weight now costs a memory read and a modulation on
*every use*, and the whole design becomes an arithmetic-intensity problem.

## 5. Combining the two: time-multiplexing the mesh at block level

The mesh cannot be temporally mapped weight by weight; its weights are phases,
not pulses. It can be multiplexed at the block level: hold a $B \times B$ block
of a large matrix for one batch, then reload. Three things have to be true.

1. **Phase shifters switch in nanoseconds.** Thermal shifters take
   milliseconds; MEMS take microseconds; EO (LiNbO₃, BTO, silicon depletion)
   take nanoseconds. Only the last makes block-level multiplexing useful.
2. **The control die can push $B^2 b$ bits per block time.** That is the
   $N^2 b$ vertical-contact array from `lightmatter-comparison.md` sec. 7, now
   run at speed: 64² × 8 bits every 100 ns is 330 Gb/s per mesh, feasible over
   65k hybrid-bonded lines at 5 MHz each.
3. **Each block serves a batch $M$ before it is reloaded**, so the reload cost
   is amortised.

Under those conditions a 64-wide EO mesh at 250 mm² time-multiplexed across a
large weight matrix looks, from the outside, like Opticore's device: parameters
in HBM, none in silicon. From the inside it keeps the mesh's one real asset,
the $O(B)$ conversions per $B^2$ MACs within each block, and it keeps in situ
backprop, since gradients are measured per block and the block's weights are
still stationary for the duration of a batch.

A second combination is a *fixed* mesh. Replace the programmable interferometers
with a passive $N$-port transform (a star coupler is a DFT in ~0.1 mm², a
multimode interference coupler is a fixed unitary) and put all programmability
into $N$ diagonal modulators driven in time. Any matrix is a product of
diagonal and DFT-like factors, but a general one needs about $2N$ such factors
(Huhtanen and Perämäki, 2015), which brings the time multiplexing back to
$N$ steps and the throughput back to the temporal-mapping row of the table.
It pays only for structured layers: convolutions, circulant or low-rank
projections, random-feature mixers. For those it is the smallest possible
photonic MVM, and it is worth saying that Opticore's coherent VCSEL work is
already halfway there: the fixed part is free-space fan-out, the programmable
part is time.

## 6. The energy accounting once weights move

Add two terms to the 2023 model when weights are no longer stationary:

$$E_{\mathrm{MAC}} \mathrel{+}= \frac{b\,E_{\mathrm{mod}} + E_{\mathrm{fetch}}(b)}{M},$$

the weight write (segmented modulator, $b$ segments) and the memory read,
each amortised over the batch $M$ that reuses the block. With
$E_{\mathrm{mod}} = 1$ fJ the write is 8 fJ per weight, negligible above
$M \approx 8$. The fetch is not: HBM3 costs of order 2 to 4 pJ per bit at the
pins, so an 8-bit weight is 15 to 30 pJ per use before amortisation, and it needs
$M \gtrsim 300$ to fall under the 50 fJ per op that the rack argument requires.
That is the same arithmetic intensity a GPU needs to be compute-bound, and it is
why Opticore's page is as much about HBM ("hyperbonded high-bandwidth memory,"
"memory no longer needs to sit next to the processor") as about optics.

So: time-multiplexing turns the silicon-footprint problem into the
memory-bandwidth problem every accelerator already has. The mesh is then judged
on the same terms as a GPU, and its claim is a 1 to 15 fJ per op multiply
engine on the compute side of that boundary. That is a real claim, and a
smaller one than "no data movement."

## 7. What this changes in the talk's argument

- The "wide N" condition on the theory slide has an area cost the 2017 framing
  did not price: $N^2$ devices at 10⁻² mm² each. Fan-in above 128 does not fit
  on a reticle, so "N in the hundreds" is a multi-die or time-multiplexed
  statement, not a single-chip one.
- "Weights static" was counted as a win for inference. It is also the reason for
  the footprint. Giving it up at block level is the price of scaling, and it
  costs nothing in energy above a few hundred vectors per block.
- In situ backprop survives block multiplexing; temporal mapping has no
  equivalent, because there is no stationary device to attach a gradient loop
  to. Training on a temporally mapped device is offline or perturbative.
- The rack story gains a term: the HBM next to the photonic package is where
  the watts go once the weights move, and it is the same HBM a GPU has.
