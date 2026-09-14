# Transformers: where a mesh fits, where it does not, and how to tile it

A discussion note that follows from `footprint-tdm.md`. The energy chart is per
op of a weight-stationary matrix-vector product. A transformer is not only
that, so this note asks what share of the work the mesh can take, what the
batch per weight actually is, and how a 128-wide mesh serves a
28 000-wide matrix.

Contents

1. MACs per token: MLP, projections, attention
2. What GQA, MQA and the long-context tricks change
3. The batch per weight block, by regime
4. Why attention wants a different circuit
5. Tiling: scaling past one mesh
6. Weights do not fit in silicon, so decode is memory-bound
7. What this does to the claim

---

## 1. MACs per token: MLP, projections, attention

Per token, per layer, model width $d$, context $L$:

| block | MACs per token | note |
|---|---|---|
| MLP | $8d^2$ | $d\to4d\to d$, or Llama's three SwiGLU matrices at $8d/3$; same count |
| Q, K, V, O projections | $4d^2$ | $2.25d^2$ with GQA at 8 of 64 heads, $2.03d^2$ with MQA |
| attention scores and value sum | $\approx Ld$ | $2Ld$ dense, halved by the causal mask |

The MLP does not know the context length. Attention grows linearly with it per
token, quadratically per sequence. With $d=8192$ and dense causal attention:

| $L$ | MLP | projections | attention | MLP share | ceiling if the MLP were free |
|---|---|---|---|---|---|
| 8k | $5.4\times10^8$ | $2.7\times10^8$ | $6.7\times10^7$ | 61% | 2.6× |
| 32k | $5.4\times10^8$ | $2.7\times10^8$ | $2.7\times10^8$ | 50% | 2.0× |
| 128k | $5.4\times10^8$ | $2.7\times10^8$ | $1.1\times10^9$ | 29% | 1.4× |

The last column is Amdahl's law for a photonic MLP engine: the system-level
speed-up if the MLP cost nothing. It is the honest number to put beside the
per-op energy chart.

## 2. What GQA, MQA and the long-context tricks change

Grouped-query and multi-query attention reduce the number of K and V heads.
Each query head still scores against every key, so the attention MACs stay at
$\approx Ld$. What shrinks by the head ratio is the K and V projection weights
and the KV cache:

| | MHA | GQA, 8 of 64 | MQA |
|---|---|---|---|
| projection MACs per token | $4d^2$ | $2.25d^2$ | $2.03d^2$ |
| KV cache per token per layer | $2d$ | $d/4$ | $d/32$ |
| attention MACs per token | $\approx Ld$ | $\approx Ld$ | $\approx Ld$ |
| MLP share of weight-stationary work | 67% | 78% | 80% |

GQA's real effect is on decode: the KV read per token drops eightfold, and the
decode bandwidth is then dominated by the weights again, mostly the MLP. The
tricks that do cut attention *compute* are the ones that bound or compress the
key set:

| trick | attention MACs per token | MLP share at 128k |
|---|---|---|
| sliding window, $W=4$k (Mistral) | $Wd$, fixed | back to about 60% |
| multi-head latent attention (DeepSeek) | $\approx Ld$ on a 512-wide latent; tiny KV cache | about unchanged; decode much cheaper |
| sparse or block-sparse attention (NSA, DSA) | a fraction of $Ld$ | 60 to 70% |
| linear attention, state-space layers | $O(d^2)$, no $L$ dependence | MLP-like, and weight-stationary itself |
| FlashAttention | unchanged; it reorders memory traffic | none |

Every current trend shrinks attention's share of the multiplies, and the two
that go furthest make the remainder weight-stationary too. The direction of
the field favours a photonic MLP engine.

## 3. The batch per weight block, by regime

$M$ in the energy model is the number of vectors through one weight block on
one device between updates. For a transformer every token is one vector
through every weight matrix, so $M$ is tokens per device per step. Tensor and
pipeline parallelism shard the matrix, not the token stream.

| regime | $M$ per weight block |
|---|---|
| pretraining, Llama 3 405B: 16M tokens per step over ~128 data-parallel replicas | $\sim10^5$ |
| pretraining, DeepSeek-V3: 63M tokens per step | $\sim10^5$ |
| pretraining, 7B on 64 GPUs, 4M tokens per step | $\sim6\times10^4$ |
| fine-tuning, 256k tokens on 8 GPUs | $\sim3\times10^4$ |
| prefill | tokens in the prompt, $10^3$ to $10^5$ |
| decode | concurrent requests sharing the weights, 32 to 512 |
| mixture of experts | above, times $k/E$ per expert (8 of 256 for DeepSeek) |
| the 2023 SM analysis | 16 to 256 |
| the 2023 experiment | 1 |

Gradient accumulation does not change this: micro-batches are summed before
the update, so the weights are stationary for the whole global batch, which is
what the integrating tap wants. $M=4096$ in the chart is one to two orders
conservative for pretraining and about right for fine-tuning. Above a few
thousand the per-batch terms are already under 2 fJ per op, so the energy
conclusion is settled; what larger $M$ still buys is gradient SNR, $\sqrt M$,
and what it costs is integrator headroom, $\log_2 M$ bits, and a stability
window of $M/f_{\mathrm{clk}}$, 100 µs at $10^5$.

## 4. Why attention wants a different circuit

$QK^{T}$ and the attention-weighted sum of $V$ multiply two activations. Both
operands change every token. To do this on a mesh the K or V block would have
to be rewritten each token: $N^2 b$ bits per token per tile, which at 128 wide,
8 bits and 1 GHz is 130 Tb/s per mesh. That is not a design.

The circuit that fits is temporal mapping: both operands arrive as pulse
streams at a homodyne detector, the photocurrent is their product, and the
integrator sums over time. $O(N)$ devices, no state, nothing to rewrite. Its
economics are set by reuse: in prefill and training each key is reused across
the sequence, so it is compute-bound and the multiply costs a modulation and a
detector, a few femtojoules. In decode each key is used once per new token, so
it is KV-cache bandwidth at 2 to 4 pJ per bit on any hardware.

Underneath this is arithmetic intensity. An HBM-fed accelerator needs of order
100 MACs per byte fetched to be compute-bound (an H100 does about 2 PFLOPS
dense FP8 over 3.35 TB/s, roughly 300 MACs per byte at the crossover). The MLP
gets there easily: each weight byte is reused across every token in the batch.
Attention does not. Every key and value element in the cache is used once per
query token, so in decode the intensity is about one MAC per byte, two orders
of magnitude short, and the layer runs at the speed of the KV-cache read no
matter how fast the multiplier is. In prefill and training the same key is
reused across the $L$ queries of the sequence, so the intensity is $\sim L$,
and the layer is compute-bound only because FlashAttention-style tiling keeps
the $L\times L$ score matrix out of HBM. Attention layers are therefore
memory-bandwidth bound in the regime that dominates serving, and a faster
multiply, photonic or otherwise, does nothing for them there.

So the division is: mesh for the MLP and projections, temporal unit for
attention, both for training and prefill. Decode stays a memory problem.

## 5. Tiling: scaling past one mesh

A Llama-70B MLP up-projection is $8192\times28672$. At $N=128$ that is
$64\times224 = 14\,336$ tiles. Each tile $W_{ij}$ is a general real matrix, so
it is an SVD, $W_{ij} = V_{ij}\Sigma_{ij}U_{ij}^{\dagger}$: two unitary meshes
and $N$ attenuators, optical depth about $4N$.

$$y_i = \sum_j W_{ij}\,x_j .$$

- **Conversions per MAC are unchanged.** Each tile still amortises $N$ inputs
  and $N$ outputs over $N^2$ MACs. The partial-sum adds are $O(N)$ per tile in
  the digital shell, negligible against $N^2$.
- **Error does not compound across tiles.** Loss, splitter and phase error
  accumulate over the $4N$ stages inside a tile and are reset when the tile
  output is digitised. A monolithic $28\,672$-wide mesh, if it could be built,
  would compound over its whole depth. Tiling caps error at the tile.
- **Weight traffic.** About five tiles fit on a reticle at compact-MZI density.
  The rest stream from HBM per batch. A tile reload is
  $N^2 b\,E_{\mathrm{mod}}\approx130$ nJ with segmented phase shifters, and the
  HBM fetch is about 0.4 µJ. Amortised over $M=10^4$ vectors that is about
  2.5 fJ per MAC, the same arithmetic-intensity rule a GPU lives under.
- **Light delivery.** The input vector $x_j$ must reach every row of tiles:
  fan it out optically, a $d_{\mathrm{out}}/N$-way split (18 dB for 64 rows),
  or re-encode it per tile row at $E_{\mathrm{mod}}$ per element. Either is a
  laser-budget question, which the rack note already assigns to a pooled laser
  shelf.
- **Rectangular blocks.** Nothing requires square tiles, but a non-square tile
  is a rectangular SVD and the meshes are sized to the larger dimension, so
  square is the efficient choice.

## 6. Weights do not fit in silicon, so decode is memory-bound

| | weights | 128² tiles | silicon at 0.01 mm² per weight |
|---|---|---|---|
| one tile | 16k | 1 | 1.6 mm² |
| one reticle | 86k | 5 | 858 mm² |
| Llama-3 8B | $8\times10^9$ | 490k | 80 m² |
| Llama-3 70B | $7\times10^{10}$ | 4.3M | 700 m² |

Cerebras makes weights resident the digital way: 44 GB of SRAM on a 46 225 mm²
wafer, 21 PB/s, so a 70B model at 8 bits fits on two wafers and decode runs at
SRAM bandwidth. The equivalent photonic mesh would be 700 m² of silicon.

A transformer mesh accelerator is therefore block-multiplexed from HBM. Its
weight traffic per step is the same as a GPU's, its decode throughput is
bounded by the same HBM bandwidth, and "no weight traffic" survives only inside
a tile for one batch. The advantage is per op at high arithmetic intensity:
training and prefill.

## 7. What this does to the claim

- The per-op numbers on the energy chart apply to the MLP and the projections,
  61 to 92% of the multiplies depending on context length and attention
  variant, with a system ceiling of 1.4 to 2.6× from Amdahl.
- Attention needs the temporal-mapping circuit; with it, photonics covers all
  the multiplies in training and prefill.
- Decode is memory-bound on every device, and GQA moved its bottleneck from
  the KV cache to the MLP weights, which a streaming photonic MLP reads from
  the same HBM.
- Tiling is not a compromise: it caps error per tile and keeps the conversion
  cost per MAC. What it adds is weight traffic, which is the price every
  accelerator pays and which training's $M\sim10^5$ pays easily.

The one-sentence version for the talk: a photonic MLP engine for training and
prefill, worth up to 2.6× at the system level at 8k context and 1.4× at 128k,
with attention on a streaming detector and decode left to memory.
