# Neighbour-strip signal coupling in strip-readout MPGDs: mechanism, time structure, scaling, and an estimate for MX17

Labels used on every number: **[measured]**, **[simulated]** (by the cited authors), **[mine]** (my own calculation or estimate), **[assumed]** (MX17 input not verified).
Local source texts: `/media/ucla/work/litmix/txt/` (fetched earlier) and `/media/ucla/work/capcoupling/` (fetched for this note). They are given as links below.
My calculations: `/home/dylan/.claude/jobs/49a04248/tmp/capcalc/cap3d.py` (3D capacitance) and `net.py` (pulse model). Output of the first is `cap3d.json` in the same folder. That folder is session scratch and may be cleaned.

## What is the difference between weighting-field induction, inter-strip capacitance, resistive-layer coupling and front-end cross-talk, and which are same-time, bipolar, derivative-shaped or delayed?

### Takeaway
"Capacitive coupling" in the MPGD literature covers four different things. Only one of them gives a same-time, same-polarity, net-charge scaled copy with a size that does not depend on the amplifier: **charge sitting on a floating (high-impedance) electrode that images onto several readout electrodes**, as with the resistive strip or the floating strip. Coupling between amplifier-connected strips is a short transient that looks like a derivative after shaping. Weighting-field signals on non-collecting strips are bipolar with zero net charge, unless electrons actually land there.

### Cited findings

**(a) Weighting-field overlap (Shockley-Ramo)**
- Flierl's Garfield + Ramo calculation for a GEM (400 µm pitch, 80 µm strip width, 2 mm induction gap) **[simulated]**: while electrons are far away the first neighbour sees 80 % of the central pulse height and the fifth neighbour still 20 %; as they approach, the ratio falls continuously to a final 15 % (first neighbour) and 0.8 % (second neighbour), "because in the region close to the read-out the first neighbor shadows the signal on the following strips" — [Flierl PhD 2018, Sec. 2.4.2](/media/ucla/work/litmix/txt/flierl.txt).
- The same calculation gives a 1.5 ns time shift for a 1000-electron cloud released 5 ns after another cloud on the adjacent strip, and the signal is "nearly perfectly" recovered by subtracting 15 % of the first strip's signal **[simulated]** — [Flierl 2018, Sec. 2.4.2](/media/ucla/work/litmix/txt/flierl.txt).

**(b) Direct inter-strip capacitance and the input impedance**
- Flierl states for resistive-strip Micromegas that the coupling "is not only caused by the overlap of the weighting fields of neighboring strips, but by direct capacitive coupling of the strips" — [Flierl 2018, Sec. 6.5.1](/media/ucla/work/litmix/txt/flierl.txt).
- Hohlmann's PSpice model of a triple-GEM/MPGD: "The crosstalk pulse shape can be understood as due to a CR-differentiator"; several coupling paths add differently for a current source (a real detector) and for a voltage source (bench pulser tests) — [Hohlmann 2020, arXiv:2012.11596](https://arxiv.org/pdf/2012.11596).
- In the same model the XT current is of opposite polarity to the signal strip's current; when the facing electrode's impedance to ground is lowered (C4 from 100 pF to 1000 pF) the XT is reduced and turns bipolar, because it has two contributions of opposite sign (via the facing electrode C1, and via the interstrip C6), which can cancel completely in the idealised circuit **[simulated]** — [Hohlmann 2020](https://arxiv.org/pdf/2012.11596).
- Silicon analogue: "a fraction of each signal appears on the output of the adjacent channels", the fraction estimated as r = C_IS/C_DEC (search excerpt, full text not read) — [PAMELA, hep-ex/0510012](https://arxiv.org/pdf/hep-ex/0510012). Belle II reports a neighbour coupling k₁ of 35-40 % on the v/N side of layers 4-6 and a 20-30 % charge loss **[measured; search excerpt only]** — [Belle II SVD, arXiv:2201.09824](https://arxiv.org/pdf/2201.09824).

**(c) Coupling through a resistive layer and its insulator**
- The theorem: for electrodes in a medium with position- and frequency-dependent ε and σ connected by discrete elements, the induced signals follow from time-dependent weighting fields, which can be derived from electrostatic solutions — [Riegler, NIM A 491 (2002) 258](/media/ucla/work/capcoupling/riegler2004/p.txt). The 2004 "Extended theorems" paper covers the same quasi-static treatment (abstract only read) — [Riegler 2004, INSPIRE](https://inspirehep.net/literature/669914).
- Riegler 2016, for a Micromegas-like layer stack: a **bulk** resistive layer touching the readout gives unipolar signals, because the compensating charge "is truly coming out of the readout strips"; short time constants increase the neighbour crosstalk — [Riegler et al., JINST 11 P11002, arXiv:1602.07949](/media/ucla/work/capcoupling/riegler2016/p.txt).
- A **thin** resistive layer insulated from the strips: ∫I dt = 0, "all of the signals are strictly bipolar", because "the current compensating the point charge −Q is entirely flowing inside the thin resistive layer and no net charge is taken from or is arriving at the strips". For decreasing resistivity the central strip's signal is 'differentiated', develops an undershoot, and crosstalk increases. The time constant is T₀ = ε₀Rg — [Riegler 2016, Eqs. 186-189](/media/ucla/work/capcoupling/riegler2016/p.txt).
- Janssens et al. split the time-dependent weighting potential into a static prompt and a dynamic delayed part and compute it with FEM + Garfield++ for a 1 MΩ/□ 2D resistive-strip bulk Micromegas. On readout strips perpendicular to the resistive strips the neighbours have "progressively smaller peak amplitude values that are shifted in time", the V-shape signature, and most of the final waveform is ion signal **[simulated]** — [Janssens et al., arXiv:2304.01883](/media/ucla/work/capcoupling/janssens2023/p.txt).
- The V-shape is **[measured]**: seen on 3-4 neighbouring strips within the first 100 ns, then a tail from spreading along the resistive strips, on the layer perpendicular to them, "expected to be present in all Micromegas chambers where the resistive strips are not placed parallel to the readout-strips" — [arXiv:1406.6871](/media/ucla/work/capcoupling/a1406.6871/p.txt).
- T2K ERAM: neighbour pads are "delayed and longer in time" than the leading pad under charge spreading, unlike diffusion charge sharing **[measured]** — [Attié et al. 2022, arXiv:2106.12634](/media/ucla/work/litmix/txt/2106.12634.txt); [arXiv:2212.06541](/media/ucla/work/litmix/txt/2212.06541.txt).

**(d) Inside the front-end, cables, flex**
- A CERN design note says input resistance and interstrip capacitance together cause charge loss, and that for a given detector the preamp input impedance must be low enough not to lose the slower part of the signal (search excerpt) — [LHC electronics workshop 2002, note A12](https://lhc-electronics-workshop.web.cern.ch/2002/Tracker/A12.pdf).
- I found no measured channel-to-channel crosstalk for DREAM. The only related number is a CLAS12 design requirement of 1 % (search excerpt) — [Jefferson Lab attachment](https://mailman.jlab.org/pipermail/solid_ec/attachments/20140319/0c17a77d/attachment-0001.pdf).

**The floating-electrode class (the LMU circuit models)**
- Bortfeldt's parameter-free LTspice model of floating-strip Micromegas injects the current on the **floating anode** and reads through 7 pF to a CSA with 1 nF input coupling (table below). On the non-hit strip 5, ph5 = 0.25 ph6; "the signal on this strip begins to rise before ionization charge reaches the strip"; 10° track → +9° **[simulated]** — [Bortfeldt PhD 2014, Sec. 4.8.4, Tab. 4.1](/media/ucla/work/litmix/txt/bortfeldt.txt).
- Lösel: coupling "is mainly due to the coupling between the resistive strips, which have a high resistivity. Therefore the charge signal disperses slowly and drains to the terminal. In this step the charges couple to the neighboring resistive strips"; the schematic shows 0.29 q on each neighbour **[simulated, LTspice]** — [Lösel PhD 2017, Sec. 7.1.2, Fig. 7.3](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf).

### Inferences (mine)
- **Summary table.** All statements are my reading, derived from the sources above and the circuit algebra in the next section.

| Mechanism | Time relative to collecting strip | Polarity on neighbour | Net charge | After shaper |
|---|---|---|---|---|
| (a) Weighting field, electrons drift to strip 0 | Starts when charge moves, ends at arrival | Same sign, then returns | **0** if all electrons land on strip 0 | Bipolar, small for `t_p` much longer than the transit time |
| (a') Electrons actually landing on the neighbour | Own arrival | Same sign | Non-zero | Ordinary signal; this is sharing, not coupling |
| (b) Inter-strip C between amplifier-fed strips | Prompt, lasts ~R_in·C_node | Same sign early, returns to small static value | Small: k_LF ≈ C_is/C_in | Derivative-like, leads the main peak |
| (b') Charge on a floating electrode that images onto several strips | Same time, persists until the electrode drains | Same sign | **Non-zero** | True scaled copy |
| (c) Thin resistive layer, along-strip spreading | Delayed and longer, grows with distance | Same sign on the time scale before drain | Zero only after full drain, which is ms for MX17 | Delayed peak, V-shape |
| (d) ASIC or cable mutual capacitance | Prompt | Set by the path | Same as (b) | Derivative-like in the usual case |

- The result in Flierl's calculation that the neighbour ends at 15 % conflicts with strict Ramo: if every electron finished on the central strip, a neighbour's weighting potential there is 0 and its net charge is 0. A 15 % final value needs electrons landing on the neighbour (his 15-electron clouds have diffusion) or finite integration. I could not determine which from the text, so the "15 %" should not be read as pure induction.
- Bortfeldt's and Lösel's "charge ratio" models belong to the class (b'): charge on a node with no amplifier of its own. Read this way, their large fractions (0.25, 0.29) and the fact that amplifier-fed readout strips are not the node of injection are the same statement. It is not stated in those words in either thesis.

### Gaps
- I could not read Blum-Riegler-Rolandi, or the original Riegler 2004 paper beyond its abstract. `jkaplon.web.cern.ch` (Kaplon, time-domain cross-talk note) did not resolve.
- No paper I read separates (a), (b) and (b') quantitatively on one detector.

## What pulse shape and polarity does a non-collecting neighbour get after a charge-sensitive preamp and shaper, and what does the charge-division formula look like?

### Takeaway
The formula depends on the input impedance model. For a real CSA, Z_in ≈ R_in + 1/(sC_in), so the neighbour's charge pulse is a prompt transient of height k_HF ≈ C_is/C_node decaying with τ ≈ R_in·C_node to a small static k_LF ≈ C_is/C_in. It is a scaled copy only in the static part, or when the injection node is floating.

### Cited findings
- Bortfeldt's model values **[calculated/measured by the author]**: Cms 2.5 pF, Casas 4.1 pF (anode-anode), Ccoupl 7.0 pF, Cslsl 4.5 pF, Cslg 1.9 pF, Ccr 1.0 nF, Ccc 100 pF, for a 6.4 × 6.4 cm² detector; the values are "calculated from the detector and strip geometry" — [Bortfeldt 2014, Tab. 4.1](/media/ucla/work/litmix/txt/bortfeldt.txt).
- Bortfeldt finds a smaller deviation in the 48 × 50 cm² chamber, "caused by the larger ratio between coupling capacitance of anode- to readout-strips and capacitance of neighboring strips and the smaller strip pitch of 0.25 mm as compared to 0.5 mm" — [Bortfeldt 2014, summary](/media/ucla/work/litmix/txt/bortfeldt.txt).
- Alexopoulos et al. give the network for resistive strips: C1 mesh-to-ground ~4 nF, C2 resistive-strip-to-mesh and C3 resistive-strip-to-readout with C2:C3 ≈ 1:8, C4 readout-strip-to-ground ~1.5 pF, plus the preamp input capacitance CA **[C2:C3 is from their stated configuration]** — [Alexopoulos et al., arXiv:1011.5370](/media/ucla/work/capcoupling/a1011.5370/p.txt).
- The charge visible to the preamplifier depends on the detector capacitance relative to the amplifier input capacitance, and long-strip capacitance slows the rise; 500 ± 50 pF for SM2 strips and 30 ± 5 pF for a T-chamber, against 3.7 pF for an APV25 channel **[stated by Flierl, method not given in the passage read]** — [Flierl 2018, Sec. 6](/media/ucla/work/litmix/txt/flierl.txt).
- DREAM has per-channel CSA, shaper, 512-cell SCA; CSA decay time 5 µs; ENC below 2500 e for peaking time < 200 ns and C_det < 230 pF **[measured, poster]**; the poster also has "Open-Loop Gain > 2000 (200 fC range)" (the text extraction reads "20 00", so verify before citing) — [DREAM TWEPP 2014 poster](https://indico.cern.ch/event/299180/contributions/1659618/attachments/563093/775742/dream_twepp2014.pdf). Peaking time 16 values from 50 ns to about 1 µs (sources disagree, 900 ns or 1 µs) — [BONuS12, arXiv:2402.01904](https://arxiv.org/pdf/2402.01904).

### Inferences (mine; circuit algebra)

**Input impedance of a CSA.** With open-loop gain A(s) = A₀/(1 + s/ω_p) and feedback C_f:

Z_in(s) = 1/(s·A(s)·C_f) = **R_in + 1/(s·C_in)**, where R_in = 1/(ω_T·C_f), C_in = A₀·C_f and ω_T = A₀·ω_p.

The crossover between the two is at ω_p. For A₀ = 2000 and ω_T ≈ 10¹⁰ rad/s, ω_p ≈ 5×10⁶ rad/s (1/ω_p ≈ 200 ns), which is inside the DREAM shaping range, so neither pure limit applies **[mine; ω_T assumed]**.

**Static charge division** (t ≫ R_in·C_node, amplifiers on every strip, chain of equal strips with C_is to each neighbour, C_g to ground):

- Define a = (C_in + C_g + 2C_is)/C_is. The fraction on the n-th neighbour is q_n/q_0 = ρⁿ, with ρ = [a − √(a² − 4)]/2 ≈ C_is/(C_in + C_g + 2C_is) for small ρ.
- The total charge measured is Q·C_in/(C_in + C_g): only C_g loses signal, and C_is merely redistributes it.
- Polarity: the same as the signal, since V₁ has the same sign as V₀.

**Pure-resistive input limit** (no C_in):

I₁/I₀ = [C_is/(C_g + C_is)]·sτ/(1 + sτ), τ = R_in(C_g + C_is).

For a delta injection the neighbour charge is k′·Q·e^(−t/τ), which integrates to **zero net charge**. Physically, V₀ goes back to 0 once the current has drained into amplifier 0, so the charge that moved through C_is flows back. This is Hohlmann's "CR differentiator".

**Combined**: Q₁(t)/Q ≈ k_HF·f(t/τ) + k_LF, where f is a transient, k_HF ≈ C_is/C_node and k_LF ≈ C_is/(C_in + C_node).

**Floating-injection node** (charge q on a resistive strip with no direct amplifier, drain time ≫ the window). Then q_j = C_jE·V_E, V_E = q/C_E,total. The ratio q_neighbour/q_main = C_neighbour,E / C_main,E, independent of the amplifier and same-time. It decays only with the drain or CSA decay.

**Time dependence of the ratio.** After CSA and shaper, class (b) is a derivative-like pulse that peaks before the main pulse. Class (b') is the main pulse scaled by the constant ratio.

### Gaps
- Actual DREAM C_f, ω_T, R_in and C_in are not in any source I could read. Those numbers in the next sections are my estimates. The Flouzat and Baron papers themselves were not retrieved: only the TWEPP poster.

## How does the coupling fraction scale with strip length, pitch, strip width, insulator thickness, ground-plane distance and preamplifier input?

### Takeaway
All strip-parallel capacitances scale linearly with length, so a ratio of them is independent of length. Only ratios to a fixed C_in change with length. A fraction rising with length therefore means a fixed-capacitance term is present.

### Cited findings
- Lösel **[simulated, LTspice]**: 23.4 % at 10 cm, 29.1 % at 100 cm, rising linearly, in a model where "only the capacitors between the readout strips were varied ... due to the short range of the coupling of the resistive strips. This shows that the main coupling comes from the resistive strips" — [Lösel PhD 2017, Sec. 7.1.2, Fig. 7.6](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf). Flierl quotes this as 23-32 % for 9-150 cm — [Flierl 2018](/media/ucla/work/litmix/txt/flierl.txt).
- These percentages were applied as a time-bin-wise subtraction and improved the angle at 20-40° but not at 10° **[measured effect of a simulated number]** — [Lösel 2017, Sec. 7.2.2, Figs. 7.10, 7.14](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf).
- Rinnagel notes the cluster multiplicity is higher for the longer strips and attributes this to capacitive coupling, "slightly larger for long anode strips" **[measured multiplicity; attribution stated as likely]** — [Rinnagel MSc 2019](/media/ucla/work/litmix/txt/rinnagel.txt).
- Coupling grew with pulse height in the same thesis ("At higher pulse height the capacitive coupling increases") **[measured, multiplicity-based]**, which a linear capacitive network cannot produce (see inference) — [Rinnagel 2019](/media/ucla/work/litmix/txt/rinnagel.txt).
- The ratio of anode-to-readout coupling to strip-to-strip capacitance and the pitch control the size of the bias — [Bortfeldt 2014, summary](/media/ucla/work/litmix/txt/bortfeldt.txt).
- In 2D floating-strip designs: parallel signal fraction up to 40 %, perpendicular fraction below 1 %, "strongly depending on perpendicular strip width" **[simulated, ANSYS + Garfield++]** — [Klitzner, ICHEP 2020 slides](/media/ucla/work/litmix/txt/klitzner_ichep.txt).
- The x-layer readout strips of a 2D resistive-strip prototype are made wider to offset their weaker capacitive coupling — [arXiv:1406.6871](https://arxiv.org/pdf/1406.6871).
- C_is values in the silicon analogue fall quickly with neighbour order: 5.15, 2.38, 1.29, 0.75, 0.44 pF for successive neighbours **[measured; search excerpt]** — [arXiv:2103.14213](https://arxiv.org/pdf/2103.14213).

### Inferences (mine)

| Variable | Effect on k = C_is/C_node (amplifier-fed strips) | Effect on floating-node ratio C_nb,E/C_main,E |
|---|---|---|
| Strip length L | None to first order (both ∝ L); fixed connector and flex C add a weak rise with L | None |
| Pitch p at fixed gap | C_is falls roughly logarithmically; k falls slowly | Falls quickly once the strip no longer overlaps the neighbour |
| Strip width w | C_g ∝ w grows, so k falls | Ratio rises sharply when w exceeds the pitch minus the gap |
| Insulator thickness d | Weak effect on k | C_main,E ∝ 1/d. Thin d localises the image, so ratio falls; lateral image range ~d |
| Ground-plane distance h | Close ground raises C_g, so k falls (Hohlmann's low-impedance facing electrode, same direction) | Close mesh takes a larger share of q, ratio unchanged to first order |
| R_in | Larger R_in: larger and slower transient | None |
| C_in | k_LF ∝ 1/C_in; no effect on k_HF | None |
| Peaking time t_p | Transient amplitude after shaper ∝ ~(R_in·C_node)/t_p | Scaled copy, unchanged |

- Lösel's linear rise with length is what a fixed-C model produces only if the readout-strip capacitances grow and a length-independent ratio (the resistive part) stays constant, consistent with his statement. I could not see his element values (figure is a screenshot) and so could not reproduce the 23-29 %.
- A coupling that rises with pulse height is not linear-network behaviour. Multiplicity above threshold rises with amplitude for a fixed coupling, which is the more likely reading of Rinnagel's observation.

### Gaps
- No published scan of coupling against insulator thickness or against preamp input impedance was found.

## How does the literature treat resistive-strip Micromegas with induction through an insulator, and what is known about pad readouts with buried vias and X/Y interconnect?

### Takeaway
Resistive-strip signals are understood as capacitive pick-up from charge on the resistive strip, with time-dependent weighting fields as the rigorous description. The checkerboard pad readout has very little dedicated literature, and I found no measurement of its neighbour coupling.

### Cited findings
- Alexopoulos: charges land on resistive strips and induce through the insulator; higher resistivity means smaller charge spread and smaller C2, C3 but the same ratio ~1:8 — [arXiv:1011.5370](/media/ucla/work/capcoupling/a1011.5370/p.txt).
- Strip resistance and linear capacitance form an RC line, V(x,t) obeying a telegraph-type diffusion with C_λR_λ — [arXiv:1304.2057](/media/ucla/work/capcoupling/a1304.2057/p.txt). Lumped LTspice ladder (121 elements, 108.9 mm) of one resistive strip with C_gas and C_dielectric — [arXiv:1304.1175](/media/ucla/work/capcoupling/a1304.1175/p.txt).
- A diffusing charge in a resistive layer sized a × a: early time is insensitive to a; late-time decay constants are set by the drain — [Janssens et al.](/media/ucla/work/capcoupling/janssens2023/p.txt).
- Encapsulated-resistive-anode pads: capacitive coupling between the resistive layer and pads — [arXiv:2303.04481](/media/ucla/work/litmix/txt/2303.04481.txt).
- MIMAC 2D readout connects pads along diagonals through metallised holes into X and Y strips — [arXiv:1109.1675](https://arxiv.org/pdf/1109.1675); a patent for Micromegas beam monitors uses interconnected pads via vias across three directions and mentions matching capacitance and charge sharing — [US 10617889](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10617889). Neither gives a coupling figure (search excerpts).

### Inferences (mine)
- MX17 input numbers **[assumed, `docs/detector_spec/spec.py`]**: FR4 5 × 264 µm "residual", insulator 70.5 µm effective, ESL 10 µm, L5/L6 Ø 0.5 mm dot per cell plus a 0.098 mm bus, L3 guard ring absent inside the active area, L7 coverage 0.53 (fan-out, not a solid plane). The spec's bus-direction labels (L5 "Y strips": bus along y; L6 "X strips": bus along x) conflict with the usual naming in which an X-measuring strip runs along y; the assignment does not change the sums below.
- Checkerboard consequence: every pad's four edge neighbours belong to the *other* view. Nearest same-view pads are diagonal (centres 1.10 mm apart, corners 0.14 mm apart). So direct same-view coupling is a corner plus via-dot term, and the dominant pad-plane capacitance is to the other view, spread over 256 different combs. Those act as a shunt to virtual ground (they raise C_node, they do not transfer signal to one channel).

### Gaps
- No published measurement of neighbour coupling in a checkerboard pad / buried-via strip readout was found.

## Worked estimate for MX17 (all numbers mine unless cited)

### Takeaway
**[mine]** Direct strip-to-strip coupling is small: a neighbour coupling k = 0.3-4.5 % of the peak (best guess about 1-2 %), a derivative-like transient leading the main pulse by ~40-115 ns. The ESL-to-neighbouring-comb route when a strip is centred on a column is about 1 % per side. The large term is **geometric**: 800 µm ESL over 780 µm columns beat with a 31.2 mm period, and where a strip straddles two columns its image is shared up to 1:1. Its projected-area mean is about 22 % on the near side. This is a same-time, same-polarity scaled copy, and it varies with x.

### Cited findings
- MX17 geometry and estimate inputs: [`docs/detector_spec/spec.py`](/home/dylan/PycharmProjects/nTof_x17/docs/detector_spec/spec.py).
- ESL drain 17 ms at the frozen 2 MΩ/□, 0.85 of ESL charge images onto pads (W2 boundary) and exact 1/2 X/Y split — [`HANDOFF_CHARGE_COMPARISON_2026-08-11.md` §8](/home/dylan/PycharmProjects/nTof_x17/ntof_processing/mm_flash/HANDOFF_CHARGE_COMPARISON_2026-08-11.md). Sheet resistance is 1.4-2.6 MΩ/□ inferred, not measured — [`spec.py`](/home/dylan/PycharmProjects/nTof_x17/docs/detector_spec/spec.py).
- Measured prompt footprint 350-450 µm, against an avalanche spread of about 25 µm over the gap, attributed in the repo to "550 µm resistive strips, induction onto the pads, inter-strip coupling" — [`cloud_basics/FINDINGS.md`](/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/analysis/cloud_basics/FINDINGS.md).
- Repo measurements of the sharing: τ_X ≈ 230 ns, τ_Y ≈ 410 ns; det3 ±1 peak-time shift +37 ns (X), +87 ns (Y); c1_X measured 0.23-0.28; simulated Stage-B c1_X rising from 0.005 to 0.239 across z — [`docs/wft_reference/FEEDBACK_2026-08-11.md`](/home/dylan/PycharmProjects/nTof_x17/docs/wft_reference/FEEDBACK_2026-08-11.md).
- That same document says point-charge prompt induction to ±1 is ≈ 0 because the nearest ±1 pad is 1.10 mm diagonal and the prompt kernel is ~0.2 mm — [`FEEDBACK_2026-08-11.md`](/home/dylan/PycharmProjects/nTof_x17/docs/wft_reference/FEEDBACK_2026-08-11.md).
- The 0.80 / 0.78 pitch mismatch is known to the group as a moiré — [`HANDOFF_board_peel.md`](/home/dylan/PycharmProjects/nTof_x17/mpgd26/slides/HANDOFF_board_peel.md).

### Inferences (mine)

**Step 1: capacitances.** 3D finite-difference Laplace solve of a 2 × 2 pad periodic cell, 20 µm grid, Cu thickness neglected, multiplied by 256 pads for a comb. Stack: mesh, 150 µm gas, ESL, 70.5 µm insulator (εr 3.4), L4 pads, 264 µm FR4 (εr 4.4), L5, 264 µm, L6, 264 µm, L7 taken as ground or absent. The X bus and Y bus are placed in the gaps between dots, with no stubs modelled. Accuracy is that of the assumed stack, and I would not trust better than ±20 % from the grid and the bus geometry.

| Quantity (one 400 mm comb) | L7 ground, no ESL | No L7, no ESL | FR4 100 µm, L7 gnd | ESL floating, aligned | ESL aligned, 50 µm εr 4 |
|---|---|---|---|---|---|
| C_node (total self, comb) [pF] | 112 | 99 | 123 | 149 | 178 |
| C_is, X comb to one neighbour X comb [pF] | 6.8 | 9.5 | 4.6 | 6.6 | 6.6 |
| C to all Y combs, sum [pF] | 64 | 72 | 49 | 59 | 58 |
| C to ESL over own column [pF] | n/a | n/a | n/a | 48 | 78 |
| ESL to ESL, both neighbours [pF per 400 mm] | n/a | n/a | n/a | 0.33 | 0.17 |

Add 5-20 pF for connector and flex per channel **[assumed; nothing measured for MX17]**. C_node ≈ 100-200 pF is inside DREAM's design range (ENC specified to 230 pF, cited above). C_is is defined as C_AD/2 because the periodic cell sees both ±1 neighbours as one conductor.

Check on the model: with ESL floating and pads at virtual ground, the image fraction on pads is (48.4 + 49.9 + 1.3 + 1.0)/115.3 = **0.87**, split evenly between the X comb and the Y combs. The repo's board accounting gets 0.85, and the Alexopoulos ratio 1:8 implies 0.89.

**Step 2: direct coupling between amplifier-fed combs.** Formula: k_HF = C_is/C_node, k_LF = C_is/(C_in + C_node), Z_in = R_in + 1/(sC_in). Unknowns bracketed **[mine]**: R_in 100-1000 Ω, C_in 200-4000 pF (A₀C_f with C_f 0.1-2 pF), t_p 120-230 ns. Time-domain chain of 31 strips, frequency-domain solve, input 10 % electrons in 4 ns plus 90 % flat ion current for 150 ns (a simplification), CR-RC² shaper, 5 µs CSA decay.

| C_is, C_node [pF] | R_in [Ω] | C_in [pF] | t_p [ns] | Neighbour max / main max | at main peak | Undershoot | Lead of neighbour peak [ns] |
|---|---|---|---|---|---|---|---|
| 6.8, 130 | 300 | 1000 | 120 | 1.5 % | 1.1 % | −0.5 % | 55 |
| 6.8, 130 | 300 | 1000 | 230 | 1.3 % | 0.8 % | −0.2 % | 100 |
| 6.8, 130 | 100 | 1000 | 230 | 0.8 % | 0.6 % | −0.1 % | 83 |
| 6.8, 130 | 1000 | 1000 | 230 | 2.3 % | 1.8 % | −0.9 % | 99 |
| 6.8, 130 | 300 | 200 | 230 | 2.2 % | 2.1 % | −0.2 % | 39 |
| 6.8, 130 | 300 | 4000 | 230 | 1.1 % | 0.5 % | −0.4 % | 117 |
| 9.5, 100 (extreme high) | 1000 | 200 | 230 | 4.5 % | 4.0 % | −0.4 % | 68 |
| 4.6, 200 (extreme low) | 100 | 4000 | 230 | 0.3 % | 0.1 % | −0.1 % | 116 |

Second neighbour ≤ 0.2 % in all cases. With these inputs the neighbour signal is a prompt, partly bipolar pulse that peaks 40-115 ns **before** the main peak. So direct coupling cannot be a same-time copy at the 20 % level.

**Step 3: coupling through the ESL (floating strip, along-column).** With a strip centred on a column, the neighbour comb sees 1.0 pF of the 48.4 pF on the near column (both sides together): **~1 % per side**, same-time, same polarity, independent of the amplifier. ESL-to-ESL capacitance is 0.33 pF over 400 mm for both sides, which is negligible against 115 pF total (0.3 % of V on the next strip). This does not reproduce Lösel's 29 % strip-to-strip figure, and I cannot explain the difference without his values. A possible reason is that the pads under the strip here act as a low-impedance plane, but this is a guess.

**Step 4: geometric sharing from the 0.80/0.78 beat.** The relevant time constant is the charge equalisation across a 550 µm strip:

τ_w ≈ R□·c′·w²/π² = 2×10⁶ Ω/□ × 4.3×10⁻⁷ F/m² × (5.5×10⁻⁴ m)² / 9.87 ≈ **26 ns** (with c′ = ε₀ × 3.4 / 70.5 µm).

Along the strip, D = 1/(R□c′) ≈ 1.2 mm²/µs, which gives σ ≈ 1.5 mm at 1 µs: the Y spreading known in the repo. For comparison Riegler's T₀ = ε₀R□g is 2.7 ns with g = 150 µm: that is his geometry's constant, not the across-width one.

The beat period is 0.78 × 0.80/0.02 = **31.2 mm** (≈ 40 strips). Let δ be the offset of a strip from its column centre, |δ| ≤ 0.39 mm. Overlap with the neighbouring pad column begins at |δ| = 0.165 mm. Using projected area with charge uniform across the strip:

- 58 % of phases have overlap with the next column.
- Mean of (neighbour/main) over phases, near side: **22 %**. Mean of neighbour/(main + neighbour): 14 %.
- The FD straddling solution, δ = 0.39 mm, gives ESL capacitance 23.7 vs 25.9 pF to the two columns, i.e. about 1:1.

Time structure: same-time to within τ_w ≈ 26 ns, same polarity, net charge non-zero. It varies across the chamber with a 31 mm period.

**Comparison with existing repo knowledge.** The 22 % projected-area mean is the same size as measured c1_X 0.23-0.28, and the 350-450 µm measured footprint is about the size expected from uniform charge over 550 µm strips imaged on 0.68 mm pads. Two caveats:

- This is an upper-end projection that ignores fringe and assumes that charge is uniform across the strip width within the signal time. c1_X in `FEEDBACK_2026-08-11.md` is a fit quantity that includes diffusion and is not the same definition.
- The Stage-B simulation there has c1_X falling to 0.005 near the mesh and the "prompt induction ≈ 0" claim. Both use a point charge, and I could not determine whether the simulation includes the strip-width equalisation or the beat. A z-independent 22 % would conflict with a near-mesh c1_X of 0.005 if that value is established in data, so **this geometric term is an unverified hypothesis until compared with the existing data** (the existing track data could be binned in x modulo 31.2 mm).

**Summary for MX17 [mine]:**

| Term | Size at neighbour (of main) | Time structure | Confidence |
|---|---|---|---|
| Direct C between combs (X-X) | 0.3-4.5 % peak; best ~1-2 % | Prompt derivative-like, 40-115 ns lead, undershoot | Medium; R_in, C_in unknown |
| ESL over own column to next comb | ~1 % per side | Same-time, scaled copy | Medium |
| ESL-to-ESL (Lösel type) | ≲ 0.3 % | Same-time | Low; disagrees with Lösel by ~×100, unexplained |
| ESL straddling two columns (beat) | 0 to ~100 %, mean ~22 %, 31.2 mm period | Same-time within ~26 ns | Hypothesis; projected-area only |
| Weighting-field prompt on neighbour | ≈ 0 net | Bipolar, small | Consistent with repo |
| Y-view RC spreading | Large, delayed | Delayed, slow | Established in repo |

### Gaps
- DREAM C_f, A₀ (read from a garbled poster extraction), ω_T, R_in and C_in: unknown, bracketed here. No DREAM crosstalk measurement found.
- MX17 peaking time used for these runs: not found in the files searched, so t_p is bracketed.
- The MX17 board stack is the repo's "residual" assumption (FR4 thickness, L3 position, L7 not a plane). This is why the table gives ranges.
- The Stage-B model's treatment of across-width equalisation was not checked.
- Lösel's LTspice element values are in a screenshot and were not extracted, so the ×100 disagreement with Step 3 stays unexplained.
- Not accessed: Klitzner PhD, Bortfeldt NIM A paper, Riegler 2004 full text, Blum-Riegler-Rolandi, Kaplon's cross-talk note, Flouzat/Baron DREAM papers. Alexopoulos was read in the arXiv version.
