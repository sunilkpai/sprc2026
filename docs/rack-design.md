# Rack design and footprint when the accelerator no longer needs liquid cooling

A discussion note. The premise: a photonic accelerator built on the energy
model in `lightmatter-comparison.md` sec. 7 dissipates tens of watts per package,
not a kilowatt, so direct liquid cooling is not required for the compute chips.
What still needs it, what that does to the rack, and what it does not change.
Numbers are order-of-magnitude and sourced where they can be; the ones that are
estimates say so.

Contents

1. What liquid cooling is for today
2. Where the heat is in a photonic accelerator package
3. Stability versus capacity: the cooling requirement that does not go away
4. What still wants a cold plate: lasers and transceivers
5. Rack arithmetic: same power or same throughput
6. Why the rack was dense in the first place, and why it need not be
7. What does not change: memory, network, facility PUE
8. A rack sketch
9. Open questions

---

## 1. What liquid cooling is for today

Direct liquid cooling exists because of density, not because chips are hot in
some absolute sense. The reference points:

| | value | note |
|---|---|---|
| B200 GPU package | ~1 000 W | ~9 PFLOPS dense FP4 at the wall, ~0.11 pJ/op |
| OpenAI/Broadcom Jalapeño (Hot Chips 2026) | 700 W | 13.4 PFLOPS MXFP4, ~0.052 pJ/op at the wall; 216 GiB HBM4 at 15.4 TB/s; OpenAI claims 1.7 to 1.9× tokens per kW over GB200/GB300 |
| Cerebras CS-3 (WSE-3) | ~23 kW | 125 PFLOPS FP16 vendor peak, ~0.18 pJ/op; 44 GB SRAM on the wafer at 21 PB/s, so decode is not HBM-bound |
| GB200 NVL72 rack | ~120 kW | 72 GPUs + 36 CPUs + NVLink switches; ~720 PFLOPS dense FP4, so ~0.17 pJ/op at rack level including the shell |
| air-cooled rack, conventional | 10 to 20 kW | raised-floor CRAC air |
| air-cooled rack, rear-door heat exchanger | 30 to 40 kW | water to the door, not to the chip |
| air-coolable package | ~300 to 400 W | with a large heatsink and high airflow; above that, cold plate |

An NVL72 is liquid cooled because 72 kilowatt-class packages sit in one rack so
that copper NVLink can reach all of them. Each of those two facts, the
kilowatt package and the metre-scale copper reach, is a design decision the
photonic route can revisit.

## 2. Where the heat is in a photonic accelerator package

Per 128-wide mesh at 1 GHz, from the 2023 model with segmented phase shifters
(Table S1 values) and Envise's measured split:

| block | per mesh | scaling | source |
|---|---|---|---|
| optical power in the mesh | 128 mW | 1 mW per mode | SM Table S6 |
| laser electrical draw for that light | 0.3 to 0.6 W | 20 to 45% wall-plug efficiency | SM 2.7.8 assumes 45%; commercial DFB nearer 20 to 30% |
| output ADCs (8-bit, 128 of them) | 0.2 W | 1.73 mW each at 1.25 GS/s, rescaled | SM Table S1 |
| TIAs, homodyne readout (512) | 0.4 W | 7 mW at 10 GS/s, rescaled to 1 GHz | SM Table S1 |
| segmented phase shifters, EO or MEMS actuation | ~0 static | switching energy only | SM 2.7.9 |
| segmented phase shifters, *thermal* actuation | ~800 W | 16 384 heaters × 50 mW | SM 1.2; this is the number that must not happen |
| gradient updaters (training only) | 16 384 × TIA, duty-cycled per batch | | SM 2.7.6 |
| digital shell: I/O prep, control, SRAM, host interface | the unknown | Envise: 78 W for four 128-wide cores, i.e. ~19 W per core, 25× the photonic core | Nature 2025 SI VI |
| HBM, if weights live off-package | ~25 to 30 W per stack | unchanged by photonics | vendor datasheets |

Three conclusions:

- **The photonic core is a few watts.** A mesh with EO or MEMS segments, its
  laser, ADCs and TIAs is about 1 to 1.5 W. Even ten meshes per package is a
  15 W optical engine.
- **Thermal phase shifters are disqualifying at scale.** The 2023 chip's 50 mW
  heaters, multiplied by $N^2$, put a single 128-wide mesh at 800 W of static
  dissipation, which is a GPU. Segmented phase shifters only remove the DAC;
  the actuation mechanism has to be one with zero holding power, and that is a
  prerequisite for the whole "no liquid cooling" claim.
- **The package power is set by the digital shell**, and the shell is set by what
  the accelerator has to do besides multiply: nonlinearities, normalisation,
  attention softmax, weight staging, host I/O. Envise's 78 W is the only
  measured data point; a 128-wide mesh package with four cores lands in the
  range of 50 to 100 W, and that is the number the rack design should use, not
  the 3 W photonic core.

## 3. Stability versus capacity: the cooling requirement that does not go away

Liquid cooling today is about removing heat. A photonic mesh has a different
thermal requirement: temperature *uniformity and stability*, because every phase
shifter is a thermometer.

Silicon's thermo-optic coefficient is $dn/dT\approx1.8\times10^{-4}\ \mathrm{K}^{-1}$.
For a phase segment of length $L$ at 1550 nm the drift is

$$\frac{d\phi}{dT} = \frac{2\pi}{\lambda}\frac{dn}{dT}L \approx 0.07\ \mathrm{rad/K}\ \text{per }100\ \mathrm{\mu m},$$

and the useful part of a 128-wide mesh needs relative phases held to a few
hundredths of a radian across a die that is centimetres on a side. That is a
sub-kelvin *gradient* budget, held while the digital shell next door swings its
power with the workload. The 2023 chip sat on a TEC at 30 °C; Lightmatter locks
every one of ~1 M elements with feedback and recalibrates hourly.

So the honest statement is: liquid cooling for *capacity* goes away; something
for *uniformity* may stay. A cold plate is the easiest way to pin a die to one
temperature, and a low-flow water loop at 20 W per package is a trivially
different thing from a 120 kW manifold. The alternatives are a TEC per die
(adds 1 to 2 W per die, fine at this scale) or thermally isolating the photonic
die from the digital die in the stack, which the 3D-stacked Envise package does
not do. The design choice is between "air-cooled with per-die TEC and
closed-loop phase control" and "a small facility water loop for stability".
Neither looks like an NVL72 CDU.

## 4. What still wants a cold plate: lasers and transceivers

The user's premise was "only the transceivers, potentially". The more precise
version is "the lasers", in both the compute chips and the interconnect.

- **Interconnect optics.** An 800G pluggable transceiver draws about 15 W; a
  51.2 Tb/s switch with 64 of them has about a kilowatt of optics on its
  faceplate, air-cooled today but at the limit. Co-packaged optics cut that by
  a claimed ~3.5× and move the laser out to an external laser source (ELS)
  module precisely because the laser is the temperature-sensitive part.
- **Compute-chip light.** The same lasers feed the meshes: from sec. 2, a few
  hundred milliwatts of electrical draw per mesh, a few watts per package.
  DFB and comb lasers lose efficiency and lifetime above roughly 70 °C, and the
  mesh's wavelength stability rides on the laser's temperature.
- **What this implies.** Pool the lasers. One ELS shelf per rack, or per few
  packages, with its own small water loop or a rear-door exchanger, and passive
  fibre to every compute package and every CPO port. The compute packages then
  have no laser on them, no need for a cold plate for capacity, and a TEC or
  loop only for uniformity. The liquid part of the rack shrinks to the laser
  shelf: hundreds of watts, not a hundred kilowatts.

SerDes and TIAs on the transceivers are the other warm parts, but at a few
watts each they are air-cooled today and stay that way.

## 5. Rack arithmetic: same power or same throughput

Take an air-cooled envelope of 30 kW per rack and ask what it buys at each
energy per op.

| energy per op (full system) | ops in a 30 kW air rack | vs NVL72 (720 PFLOPS at 120 kW liquid) |
|---|---|---|
| 1.2 pJ, Envise measured | 25 POPS | 29× less per rack, 7× less per kW |
| 300 fJ, 2023 model's 8-bit digital baseline | 100 POPS | 7× less per rack |
| 180 fJ, Cerebras WSE-3 FP16 | 160 POPS | 4× less per rack, but at 16-bit and with weights resident on wafer |
| 52 fJ, Jalapeño MXFP4 | 580 POPS | near parity with the 8-bit model row; the real digital target at 4 bits |
| 45 fJ, 2023 model 8-bit inference, segmented PS | 670 POPS | at parity per rack, 4× better per kW |
| 14 fJ, 2023 model 4-bit inference | 2.1 EOPS | 3× more per rack in a quarter of the power |

Read the table with sec. 2 in mind: the 45 and 14 fJ rows are the multiply
engine plus its I/O preparation, not a whole accelerator. The Jalapeño row is
the one to measure against: a purpose-built 4-bit inference ASIC already sits
at 52 fJ per op at the wall, so the photonic 4-bit engine's margin is 3.7×
*before* its own shell is counted, not the 8× against a B200. They say what the
photonic core allows, and Envise says what a first-generation shell costs. The
design target that makes the "no liquid cooling" story real is a shell under
about 50 fJ per op, which is where the 2023 model's own $12E_{\mathrm{OP}}$ per
element already sits.

Two ways to spend it:

- **Same throughput, a fifth to a tenth of the power.** One NVL72's worth of
  dense low-precision ops in an air-cooled 30 kW rack. No CDU, no manifolds, no
  leak detection, no facility water to the rack, standard 2U servers at eight
  100 W accelerators each. This is the retrofit story: any existing air-cooled
  hall can take it.
- **Same power, five to ten times the throughput.** Keep the 120 kW liquid
  rack, fill it with 100 W accelerators instead of 1 kW ones, and the rack
  holds 1 200 of them. This is the greenfield story, and it runs straight into
  the interconnect and memory questions of sec. 7, because 1 200 accelerators
  in a rack is a network problem before it is a thermal one.

Floor footprint is the same rack either way; what changes is compute per
megawatt, which is the number that sites are actually limited by.

## 6. Why the rack was dense in the first place, and why it need not be

The NVL72 exists because copper NVLink reaches one to two metres. Seventy-two
GPUs share memory over copper only if they are within a rack, so the rack is
the scale-up domain, and the thermal problem follows from that. Photonic
interconnect, which is what Lightmatter now sells as Passage and what the deck's
interconnect slide is about, breaks the reach argument: the scale-up domain can
span racks or rows. Once it does, there is no reason to put a kilowatt in every
rack unit, and the rack becomes a network and power-distribution unit rather
than a thermal one. Low-power photonic compute and photonic interconnect are the
same argument from two sides: the one removes the reason for density, the other
removes the reason for its cost.

## 7. What does not change: memory, network, facility PUE

- **Memory.** Weights and KV cache still live in HBM or SRAM; HBM stacks at 25
  to 30 W each do not care what does the multiply. Decode is bandwidth-bound,
  so for LLM serving the accelerator's memory system sets both power and
  throughput, and a 100 W photonic engine next to four 30 W HBM stacks is a
  220 W package, still air-coolable but not a 20 W one. The energy model here
  is about the MVM engine only.
- **Network.** Every rack still needs its scale-out optics, and those are the
  one place where the transceiver power per port is rising, not falling, with
  bandwidth. CPO and pooled lasers are the answer, and they are the part of the
  rack that keeps a water loop.
- **PUE.** Direct liquid cooling gets facility PUE to about 1.1; air cooling
  sits at 1.3 to 1.5. An air-cooled photonic hall therefore gives back about a
  quarter of its energy advantage in facility overhead. The total still wins by
  the ratio in sec. 5, but the comparison has to be made at the wall, not at the
  chip.
- **Reliability.** No coolant loop to the compute package removes the leak and
  corrosion failure modes; hourly recalibration and a million closed loops add
  a different class of failure. Lightmatter's SI lists a non-functional weight
  buffer switch in first silicon, which is a reminder that the digital shell is
  where first-generation problems live.

## 8. A rack sketch

- 42U air-cooled rack, 20 to 30 kW, standard hot-aisle containment, optional
  rear-door heat exchanger.
- 2U compute sleds, eight photonic accelerator packages each at 50 to 100 W
  with HBM, EO or MEMS segmented phase shifters, no on-package laser, TEC per
  photonic die.
- One 2U external laser source shelf per rack, water-cooled through a
  rear-door or small CDU loop, passive fibre fan-out to compute packages and to
  CPO switch ports. The only liquid in the rack.
- Scale-up domain over photonic interconnect spanning several racks; scale-out
  over CPO switches with pooled lasers.
- Facility: no chip-level water, no manifold in the rack, existing air-cooled
  halls reusable.

## 9. Open questions

1. What is the achievable digital shell energy per op for a real workload,
   attention included? Envise's 1.19 pJ per op is the only measurement, and it
   is 25× the photonic core. The "no liquid" story needs that under 50 fJ.
2. Is a TEC per die enough for sub-kelvin uniformity at 128 wide with a
   workload-dependent digital die in the same stack, or does uniformity alone
   justify a cold plate?
3. What does the pooled-laser shelf cost in optical loss and in power? The
   2023 model's 45% wall-plug assumption is a high-power laser number; at
   fibre-delivered milliwatts per mode the budget is set by coupling loss as
   much as by efficiency.
4. Does the scale-up domain over photonic interconnect keep the latency the
   collective operations need, or does spreading the domain over racks trade
   the thermal problem for a synchronisation one?
5. For training, sec. 7 of `lightmatter-comparison.md` puts batch-integrated
   training at 53 fJ per op at $M=4096$ with 12-bit-class gradients from 8-bit
   light. That is an air-cooled package like the inference one. What the training
   rack adds is a stability spec: a 10 µs integration window per gradient sample
   during which every phase and the laser must hold.
