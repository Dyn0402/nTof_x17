# Measured and simulated neighbour-strip coupling in strip and pad MPGDs

Compiled 2026-11-09. Downloaded sources and extracted text are in `/media/ucla/work/capcoupling/` (`pdf/`, `txt/`, `img/`). Earlier texts are in `/media/ucla/work/litmix/txt/`. "Read" means I read the quoted passage in the full text. Numbers read off a figure and arithmetic are marked **(mine)**. "Secondary" means the source reports someone else's number.

Definitions differ between sources, and this matters when comparing them:
- **Neighbour/main ratio**: q(±1)/q(0), per neighbour. Used by Lösel, Flierl, Jagfeld, Scholer and Ntekas. Scholer's digitiser fixes the meaning: `qtot = qi + 2c1 qi + 2c2 qi` ([Scholer][sc] Sec. 10.2.3, Eq. 10.4).
- **Fraction of a strip's charge moved to a neighbour** (Klitzner's C_i/C_total is of this kind).
- **Peak of an edge-strip ratio distribution**: this is conditional on the edge strip having fired and on the threshold, so it is not a clean coupling constant **(mine)**.

## What coupling fractions are reported, by detector, strip length, pitch and readout type?

### Takeaway
Reported nearest-neighbour values run from about 5-10 % (GEM strips, pulse injection; simulated 10 % for a short resistive Micromegas) through 15-25 % (10 cm resistive Micromegas) and about 30 % (1-2 m resistive strips) to 67 % (one VMM3 edge-ratio peak on the longest NSW strips). No single number exists: the value depends on strip length and readout-strip width, and on how it was defined and extracted.

### Cited Findings

**Master table.** M = measured on a detector, S = simulated, E = empirical (tuned to optimise reconstruction), C = capacitance model or formula.

| # | Value | Type | Detector, geometry | Readout | How obtained | Time relation stated | Source, page |
|---|---|---|---|---|---|---|---|
| 1 | **~15 %** (edge strip vs inner neighbour), slope of q-q correlation | M | ATLAS T-type resistive Micromegas, 400 µm pitch, 300 µm wide readout strips, 10 cm class, 10° inclination | APV25 | Test beam. "the slope indicating a ∼ 15% of the charge sharing" | "inherit the neighbouring time measurement" (asserted) | [Ntekas][nt] Sec. 5.3.3, pp. 125-126 |
| 2 | **peak at ratio ≈ 6** (= 16.7 % **(mine)**) for earliest strip; last strip "not very pronounced" | M | Same | APV25 | Distribution of q ratio, Fig. 5.12 | as above | [Ntekas][nt] p. 126 |
| 3 | **~10 %** nearest-neighbour "charge leakage" | S | Same T chamber, 400 µm, 20 MΩ/cm, 5-strip model | not modelled | ANSOFT Maxwell + Simplorer | none | [Ntekas][nt] Sec. 5.3.3, p. 125; "at the 10% level" in [ATLAS proc. 2015][atl] |
| 4 | **~4** peak ratio (= 25 % **(mine)**) in the 80 µm width area; no such peak at 320 µm | M | T4C resistive Micromegas, fixed resistive geometry, four readout-strip widths | APV25 | 150 GeV/c pion beam, charge ratio 2nd/1st strip, Fig. 7.12 | none | [Ntekas][nt] Sec. 7.2, pp. 189-190 |
| 5 | **23 % at 80 µm, falling to 11 % at 320 µm** readout-strip width | S | T4C parameters | not modelled | Maxwell/Simplorer: pulse in the middle resistive strip; "in good agreement with the value measured" | none | [Ntekas][nt] pp. 190-191 |
| 6 | **23.4 % (10 cm), 29.1 % (100 cm)**; **~31.5 % at 150 cm** (read from Fig. 7.6 **(mine)**) | S | Resistive-strip Micromegas (LMU test chambers 400 µm pitch, 10 × 10 cm²; 1 m² prototype; SM2 up to 150 cm) | current ratio at readout | LTspice with resistive and readout strip capacitors; only the readout-strip capacitors scale with length | superposition described; the correction is applied per time sample | [Lösel][lo] Sec. 7.1.2, pp. 96-98 |
| 7 | **30 %** best value of a 0-40 % scan in 5 % steps | E | SM2 large resistive module, 20° | APV25 | Time-bin-wise subtraction, optimum in angle and resolution | none | [Flierl][fl] Sec. 6.5.1, ~p. 102 |
| 8 | **about 30 %** "factor for the coupling-correction" | E/secondary (method not given in the passage read) | SM2 > 1 m strips | APV25 | Cosmic facility calibration | none | [Herrmann][he] Ch. 7 (PDF p. 84) |
| 9 | **20 %** best empirical; **15 %** simulated, final 1st neighbour; **0.8 %** 2nd neighbour; 80 % early during drift | E + S | GEM, 400 µm pitch, 80 µm width, 2 mm induction gap, E_ind 2 kV/cm | APV25 | Garfield + Ramo (S); optimised on data (E). Flierl says the excess of 20 over 15 "might be afflicted by additional capacitive coupling" | S: first strip's induction precedes; shift 1.5 ns for clouds 5 ns apart | [Flierl][fl] Sec. 2.4.2 pp. 19-22; Sec. 5.3.1 p. 70 fn. 19 |
| 10 | **~5 % (text)**, about 9-10 at plateau on the Fig. 5.5 axis, up to about 35 at low charge **(mine)** | M | BESIII triple-GEM, 650 µm pitch | APV25, 25 ns | **Direct pulse injection**: special board, attenuator + 1 pF, 10-400 mV, 40 ns, read on neighbours | **"after about 15 ns"**; Fig. 5.5 shows about 4-21 ns depending on charge **(mine)** | [Farinelli][fa] Sec. 5.5, pp. 79-81 |
| 11 | **67 %** (q1/q0 peak 1.5) | M | NSW full-size quadruplet, at the longest strips | VMM3, neighbour logic on | Test beam, ratio of first to second strip assuming the edge fired only by coupling | "having the same timing" (statement) | [Scholer][sc] Sec. 8.3.3, pp. 76-77 |
| 12 | **c1 = 0.3, c2 = 0.09** (defaults); 0.3 and 0.5 tested | S | NSW digitisation | VMM | Applied to time-binned deposits, same timing by construction | same time (assumed) | [Scholer][sc] Sec. 10.2.3, Table; Sec. 10.3.9 |
| 13 | **+9° at 10° true**; non-hit strip **ph5 = 0.25 ph6** | S | Floating-strip Micromegas, 6.4 × 6.4 cm², 0.5 mm, metallic anode + coupled readout. C_asas 4.1, C_coupl 7.0, C_slsl 4.5, C_slg 1.9, C_ms 2.5 pF | ideal CSA | LTspice, parameter-free from geometry | Signal "begins to rise before ionization charge reaches the strip" | [Bortfeldt][bo] Sec. 4.8.4, pp. 69-72 |
| 14 | up to **40 %** into parallel strip; **< 1 %** perpendicular (two orders smaller); 6.8 % floating-to-parallel used; total coupled **6.4 % / 8.7 %** perp./par. of total charge | S | 2D floating-strip, 128 strips, 0.5 mm, 25 µm layers, 6.4 cm | model | ANSYS capacitance + Garfield++. Coupling *from floating strip to readout*, not neighbour. Neighbouring-strip sharing explicitly neglected | none | [Klitzner][kl] Sec. 4, pp. 53-56 |
| 15 | C(strip-strip) about **3.45 pF** (floating, x) and **4.34 pF** (y); Cxf 2.13, Cyf 0.16 pF | S/C | as #14 | none | ANSYS + strip-line formula. (Neighbour symbol names garbled in extraction; assignment **(mine)**) | n/a | [Klitzner][kl] Table 4.1 |
| 16 | **4-12 %** coupled cluster/signal cluster (non-neighbour, multiplexing board) | M | TZ detector, multiplexed adapter board | APV25 | Source: Fig. 4.12. Inter-line C by LCR meter, precision 0.1 pF | n/a | [Jagfeld][ja] Sec. 4.4, pp. 31-36 |
| 17 | **≈ 23-29 %** | Secondary | Resistive strip | APV25 | Quotes Lösel | n/a | [Jagfeld][ja] Sec. 2.3, p. 15 |
| 18 | **< 1 %** channel-to-channel (8 mV over full pulser range), chip only | M | VMM1 unbonded | VMM1 | Internal pulser, read adjacent channel. Electronic cross-talk, not strip coupling | n/a | [Iakovidis][ia] Sec. 7.4.4, p. 111; repeated in [Ntekas][nt], neighbour-channel pedestal discussion (Ch. 7) |
| 19 | **(1.66 ± 0.03) % → (1.11 ± 0.02) %** in affected sectors after mitigation; 0.24-6.40 % range; max (6.40 ± 0.42) % | M | CMS ME0 triple-GEM, **sector-to-sector**, not adjacent strips | pulser into strips, oscilloscope | Square pulse injection, ratio to injected pulse | Differentiated pulse (see below) | [Butalla & Hohlmann][me0] |
| 20 | C_strip ≈ 100 ± 20 pF/m (total strip, measured) | M | NSW Micromegas | n/a | Quoted capacitance; W/P = 0.7059 geometric fraction computed | n/a | [Alexopoulos 2024][al] Sec. 3.2 |
| 21 | Jagged strip lowers inter-strip capacitance about 30 % | S | BESIII CGEM, 650 µm | TIGER | Design statement; a search summary attributes it to simulation, not read directly | n/a | [CGEM design paper][cgem1]; [CGEM test beam][cgem2] |

Other values, not read in primary form:
- Search-result snippets only, **unverified**: FIT zigzag GEM poster "up to 14 %" on one strip; JLab SBS APV25 digital cross-talk slides. I did not open these documents.
- ALICE PASA channel cross-talk < 0.3 %, a wire-chamber front end for context only ([ALICE TPC][alice]).

Experiments where no coupling fraction was found (see Gaps): COMPASS, CLAS12 MVT, sPHENIX TPOT, T2K ERAM (as a capacitive fraction), µ-RWELL.

**Other qualitative statements.**
- NSW test chamber multiplicity 7 observed vs 4.8 ± 0.4 geometric at 21°, attributed to coupling, read out via neighbour logic. "At higher pulse height the capacitive coupling increases"; "slightly larger for long anode strips" (citing Lösel) — [Rinnagel][ri] Sec. 4.2-5.4.
- Lösel: for a resistive-strip detector "this coupling is mainly due to the coupling between the resistive strips" — [Lösel][lo] p. 96.

### Inferences
- Lösel's three points give a slope of about 0.06 %/cm between 10 and 100 cm, so about 25 % at 40 cm **(mine)**. This applies to NSW-type 0.4-0.45 mm pitch and may not transfer to a 0.78 mm pitch: Ntekas's width scan shows the value moves by a factor of 2 with geometry at fixed pitch.
- The measured ratio for an 80 µm strip (1/4) and the simulated 23 % agree. That is the best single measurement-versus-simulation comparison I found, and it is at one geometry.

### Gaps
- I found no coupling value for a 0.78 mm pitch, nor for strips of about 40 cm with that pitch, nor any DREAM/AFTER front-end measurement on a strip detector.
- Lösel's Fig. 7.6 gives no value in the text for 150 cm, and no uncertainties at all.

## Which values are direct measurements and which are simulation?

### Takeaway
Only one true on-detector injection into a strip with neighbour read-out was found (BESIII GEM, Farinelli Sec. 5.5). The ATLAS and LMU numbers are either edge-strip charge ratios from beam data or circuit simulation (LTspice or Maxwell), and the widely quoted 23-32 % is simulation only, then validated indirectly through the angle.

### Cited Findings
- Direct injection, GEM: "A signal is injected to the anode strip thanks to a special board inserted between the APV25 and the anode with an attenuator and a 1 pF capacitance" — [Farinelli][fa] p. 81.
- Injection with VMM1 tests electronics only — [Iakovidis][ia] p. 111.
- Beam charge ratios: [Ntekas][nt] Sec. 5.3.3 and 7.2; [Scholer][sc] Sec. 8.3.3.
- LTspice: [Lösel][lo] Sec. 7.1.2; [Bortfeldt][bo] Sec. 4.8.4. Maxwell/Simplorer: [Ntekas][nt]. ANSYS + Garfield++: [Klitzner][kl]. Garfield + Ramo: [Flierl][fl] Sec. 2.4.2. PSpice model of MPGD crosstalk: [PSpice model][xt].
- Indirect validation by the angle: 23.4 % correction applied to eight test chambers; "for 30° and 40° the most probable value is exactly the expected value" and 10° "still does not work very well" — [Lösel][lo] p. 100. Applied to the large module at 29.1 %, the method "works not as well" — [Lösel][lo] Fig. 7.14.
- Flierl's 30 % is an optimum in reconstruction quality on SM2, not a charge measurement — [Flierl][fl] Sec. 6.5.1.
- Pulser sign warning: PSpice results "explain why tests with a voltage pulser produce same-sign XT pulses while tests with a GEM in normal operation mode produce opposite-sign XT pulses" — [PSpice model][xt]. An external-pulser number is therefore not automatically the coupling for an avalanche signal.

### Inferences
- A subtraction factor chosen to optimise angle (Flierl 20 %, 30 %) absorbs any other effect that compresses the time ladder, such as diffusion. It is not a measurement of capacitive coupling **(mine)**.

### Gaps
- No pulse-injection measurement on a Micromegas strip (resistive or not) with neighbour read-out was found.

## What do edge-strip charge ratios look like, and is the "ghost hit" at the same time as its neighbour?

### Takeaway
Edge-strip ratio peaks are shown, at about 1/6 and 1/4 (APV25) and about 2/3 (VMM3), and they differ from middle-strip ratios. Simultaneity is **stated repeatedly but I found no histogram of edge-minus-neighbour time** in any Micromegas source. The only direct timing measurement found (GEM injection) reports a **delay of about 15 ns**, which contradicts "same time" as written elsewhere in the same thesis.

### Cited Findings
- Edge vs middle: "the charge amplitude of which is highly correlated with the amplitude of the second and one before the last strips"; for middle strips "the random relation". Edge amplitude "≤ 300 ADC" — [Ntekas][nt] p. 125.
- Timing as claim: "the strips with capacitively coupled only signal from their neighbours inherit the neighbouring time measurement as well" — [Ntekas][nt] p. 126. Also "the time measurement is equivalent to the adjoining strip time" and drift-time tails in Sec. 4.2, Fig. 4.10 — [Ntekas][nt] p. 101.
- Same statement, no numbers: "Induced signals in the first and last strips of a cluster have same arrival times as their neighbours" — [Alexeev][alx] Sec. 4, citing Ntekas as [21].
- Ghost rule: charge < 300 ADC (10 fC), ratio 15 %, "the time measured is the same" — [Farinelli][fa] App. B p. 117.
- Against that, the same thesis: "a signal of about 5% of the inducing charge on the neighboring strips after about 15 ns"; Fig. 5.5 bottom panel shows the delay vs charge — [Farinelli][fa] p. 81. That GEM injection used a 40 ns pulse, APV25 25 ns samples.
- VMM: "mostly happens when a single strip is over threshold and the neighboring strips get the charge via the capacitive coupling, therefore having the same timing as the main strip" — [Scholer][sc] Ch. 8 (no time plot in that passage). The digitiser imposes this: "The timing of the induced charge deposits is the same as for the original charge" — [Scholer][sc] Ch. 10.
- Simulations of timing: signal on strip 7 rises before ionisation arrives — [Bortfeldt][bo] p. 70; 1.5 ns shift — [Flierl][fl] p. 21.
- Effect of removal: edge cut (< 15 %) "angle bias is reduced by 5°" at 10° — [Ntekas][nt] pp. 127-131. The "small remaining bias is most likely coming from the capacitive coupling between the middle strips" — ibid.
- Other cut values: edge discarded if charge below 1/2 of neighbour — [Rinnagel][ri] Fig. 3.7; below 1/6 — [Sessa, from LITERATURE_MIXING.md][sessa] (secondary to this note).

### Inferences
- The evidence that the coupled signal is simultaneous is indirect: the angle improves when low-charge edge strips are removed. That is also what would happen if those edge strips carried diffusion-mixed charge or noise **(mine)**.
- Ntekas ascribes tails in Fig. 4.10 partly to electrons that "induce signal to the strip of interest" and are affected by diffusion, so the source itself notes a non-capacitive route to edge signal. The ratios in the table are not pure capacitance.
- A delay between signal and neighbour of the size of Farinelli's (some ns to about 20 ns on a 25 ns sampled pulse) would be below the 25 ns sample but not zero. It could also be a charge-sensitive shaping effect at low amplitude, since the main and neighbour pulses differ in size and sign of the time-walk **(mine)**. The figure's quantity is not defined in the text.

### Gaps
- No measured Δt distribution (edge minus neighbour) for a resistive Micromegas was found. Ntekas, Alexeev and Scholer assert the equality or use it as an input.
- Farinelli's Fig. 5.5 axis has no percent sign, so whether its 9-10 plateau is percent is unresolved, and the "about 5 %" in the text does not match the plotted plateau. The discrepancy cannot be resolved from the thesis.

## Is coupling the same for resistive and metallic readout, and for X and Y strips?

### Takeaway
No same-geometry resistive-versus-metallic comparison was found. Sources name the resistive layer as the main path of coupling in resistive-strip Micromegas. For two-layer readouts I found charge-sharing ratios between layers and capacitance models, but no per-layer value of neighbour coupling.

### Cited Findings
- Resistive path dominates: "the main coupling comes from the resistive strips" — [Lösel][lo] p. 97. "the effect of the capacitive coupling does not concern only the readout strips but the resistive strips as well" — [Ntekas][nt] p. 125. "high capacitive coupling" from the "resistive layer and long copper strips" — [Scholer][sc] Sec. 10.2.3.
- Flierl contrasts the GEM (weighting-field overlap) with resistive strips, where coupling is "not only caused by the overlap of the weighting fields of neighboring strips, but by direct capacitive coupling of the strips" — [Flierl][fl] Sec. 6.5.1.
- Metallic strips: floating-strip (metal anode) ph5 = 0.25 ph6, simulated — [Bortfeldt][bo]. Non-resistive ATLAS test chambers show a qualitative "capacitative induction" bias — [Sessa][sessa]. GEM: table #9, #10.
- 2D layers: resistive-strip chamber with Y layer directly under the resistive strips, X below 70 µm FR4: "the larger width of the x-layer readout strips compensates for the weaker capacative coupling" — [Lin et al.][lin] Sec. 2. Charge ratio X:Y = 1:1.4 — [Ntekas][nt] Sec. 5.2 (citing a Tmm study).
- Y view perpendicular to resistive strips: charge "starts to spread along the resistive strips, i.e. perpendicular to the y layer strips and hence creates a tail via capacitive coupling"; x signal lasts over 300 ns, y signal not longer than 200 ns in that APV25 setup, and the y signal covers more strips — [Lin et al.][lin] Sec. 3. That is a resistive time structure on one view, not the neighbour fraction.
- 2D floating strip: coupling to the perpendicular layer two orders below the parallel one — [Klitzner][kl] p. 56. A resistive 2D strip readout reports "equal charge sharing between the two readout strip planes" for one design — [Characterisation of resistive MPGDs with 2D readout][drd].
- T2K ERAM (resistive pads) distinguishes "charge sharing" and "charge spreading": "Contrary to the 'charge sharing' effect, the signals in neighbouring pads are delayed compared to the one of the leading pad in case of the 'charge spreading'" — [T2K analysis][t2k]. It gives charge ratio and time differences between pads, Fig. 8, and is not a capacitive fraction.

### Inferences
- For resistive readouts there are at least three mechanisms giving neighbour signal (induction overlap, inter-strip capacitance, resistive spreading with RC delay), so a single "coupling fraction" mixes them **(mine)**. Only the RC route is expected to be delayed; the literature above demonstrates that delay for pads and for the view across resistive strips, not for parallel neighbours.

### Gaps
- No neighbour-strip coupling measured separately for the X and Y layers of a 2D readout at different depths was found. Klitzner explicitly neglects within-layer sharing in his total, and Ntekas's T4C scan varies width, not layer.

## Do results depend on the electronics (APV25, VMM, DREAM/AFTER)?

### Takeaway
Plausibly yes, but no source separates chip from detector: the only cross-chip comparison changes strip length and chip together. Channel-to-channel electronic crosstalk is small (< 1 %) where it was measured.

### Cited Findings
- "the study presented here was done at the point of the longest strips of the DUT where CS is expected to be larger... Furthermore the two aforementioned measurements were carried out with the APV25 ... in contrast to the VMM3 used here" — [Scholer][sc] p. 77.
- VMM1 electronic crosstalk < 1 % — [Iakovidis][ia]. The same is quoted by [Ntekas][nt] in Ch. 7 ("∼ 1%", one chip not attached to a chamber).
- Neighbour logic alters what is seen: below-threshold edges read when an adjacent strip fires — [Rinnagel][ri] Sec. 4.2; [Scholer][sc] Ch. 8 ("enhances the amount of events where the µTPC method reconstructs the wrong angle").
- Applied to the time-binned charge before the VMM model, since peak-finding chips extract one value — [Scholer][sc] Sec. 10.2.3.
- APV25 corrections are per sample: "This is done as a function of time, i.e. in this case sampling steps" — [Lösel][lo] p. 100; [Jagfeld][ja] Eq. 2.4.

### Inferences
- A peak-detecting chip and a 25 ns sampled chip respond differently to a neighbour pulse that is a scaled and time-shifted copy, so an edge ratio of 0.67 (VMM3, 100/200 ns peaking) is not directly comparable to 0.15 (APV25 chip, sampled) **(mine)**.

### Gaps
- No DREAM or AFTER measurement of strip-to-strip coupling on a detector was found; T2K's AFTER use is on pads and measures RC spreading.

## Are the numbers cited accurately down the citation chain?

### Takeaway
Lösel's values are simulation and are reasonably cited, except that the "9 cm" is not in Lösel. Farinelli's "ATLAS observed an average of 15 %" traces to one Ntekas measurement on one chamber at 10°; it was not an average in the original, and the same thesis gives 10 % in simulation and 23-25 % for narrower strips.

### Cited Findings
- **Flierl:** "Simulations by [Lösel, 2017] predicted a coupling between 23 % to 32 % for chambers with a strip length between 9–150 cm" — [Flierl][fl] Sec. 6.5.1, p. 101.
- **Lösel actually states:** "For 100 cm long readout strips the capacitive coupling is 29.1 %, and 23.4 % for 10 cm long readout strips" — [Lösel][lo] Sec. 7.1.2, pp. 96-97. Fig. 7.6 (three points) adds about 31.5 % at 150 cm **(mine, from the figure)**.
- So "23-32 %" is right as a simulated range, and 32 is a rounding of a value not given in the text. "9 cm" is not Lösel's length: his small chambers are 10 cm, and 9 × 9 cm² is the size of his reference chambers, a possible source of the slip **(mine)**. It is simulation only, without uncertainties.
- **Other reuse:** Jagfeld "≈ 23 %−29 %" — [Jagfeld][ja] p. 15. Scholer's "around 30 % over the full SM2 module" cites Flierl and Herrmann; Flierl's 30 % is an optimised correction, Herrmann's is unsourced in the passage I read — [Scholer][sc] p. 77.
- **Farinelli:** "Due to the capacitance between strips, ATLAS observed an average of 15% of the inducing charge between a strip and its nearest neighbouring strip" — [Farinelli][fa] App. B p. 117, citing refs [1, 2] = Iakovidis, Ntekas.
- **Iakovidis** contains no neighbour-strip charge value. He models coupling ignoring neighbours ("any charge induction due to capacitive coupling on the neighbor strips is ignored for this study"), and gives only the < 1 % VMM1 electronic value — [Iakovidis][ia] Sec. 5, 7.4.4.
- **Ntekas** contains the ~15 % slope, from correlation plots of a T chamber at 10°, and the peak at six. "Average" does not appear; the 300 ADC threshold is from him; the "10 fC" conversion is Farinelli's.
- Ntekas is internally inconsistent: his simulation gives "∼ 10%" (p. 125) yet the data are called "in good agreement with the simulation expectations of 15% charge sharing" (p. 126). The later T4C result is 23 % (simulated, 80 µm) and 11 % (simulated, 320 µm).
- Scholer's citation of Ntekas as "a value of 16 %" is close but not what Ntekas wrote (his 15 %, ratio 6 = 16.7 % **(mine)**) — [Scholer][sc] p. 77.
- The ATLAS proceedings summary says "It was determined in simulation that charge-sharing between neighbouring strips is at the 10% level" — [ATLAS proc. 2015][atl]. So the collaboration's own summary reports 10 %, simulated, which differs from Farinelli's "observed 15 %".
- Also: Farinelli's quoted rule is ATLAS-derived but is applied by the BESIII group to GEMs, where his own injection gives about 5 % and a delay.

### Inferences
- A safe citation would be: "ATLAS test-beam edge-strip charge ratios on 400 µm resistive-strip chambers read out with the APV25 indicate a coupling of about 15 % (Ntekas 2016), rising to about 25 % for narrower strips; circuit simulations give 10-23 % depending on strip width (Ntekas) and 23-32 % for strip lengths of 10-150 cm (Lösel 2017)." **(mine)**

### Gaps
- **Not accessed:** Ketzer et al. IEEE TNS 2002, Bachmann, Iodice, Alviggi and Bianco ATLAS papers, Kuger PhD, Vanadia, Mezzadri and Da Rocha Rolo on TIGER cross-talk, Bertani, Klitzner's full Sec. 7 (only Ch. 4 read), the NTUA Maxwell/LTspice thesis (Drivas-Koulouris; record only), CLAS12 NIM A957. Altunbas 2002 was read only by keyword: it reports charge sharing between the two projections, not strip-to-strip cross-talk ([Altunbas et al.][alt]).
- CDS was blocked by an anti-bot page; Ntekas, Iakovidis and Scholer were obtained through INSPIRE instead.

[nt]: https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65
[ia]: https://inspirehep.net/files/1252741b939076ad1910144e49c84fa2
[sc]: https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b
[atl]: https://inspirehep.net/files/cf2a35f9dc497f132c073ce83c5ef49c
[lo]: https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf
[kl]: https://edoc.ub.uni-muenchen.de/24286/1/Klitzner_Felix.pdf
[he]: https://edoc.ub.uni-muenchen.de/25055/1/Herrmann_Maximilian_G.pdf
[fl]: https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf
[bo]: https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf
[ja]: https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_cjagfeld.pdf
[ri]: https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_mrinnagel.pdf
[fa]: https://arxiv.org/abs/1904.06548
[alx]: https://arxiv.org/abs/1908.06253
[me0]: https://arxiv.org/abs/2102.03342
[xt]: https://arxiv.org/abs/2012.11596
[al]: https://arxiv.org/abs/2409.19297
[lin]: https://arxiv.org/abs/1406.6871
[drd]: https://arxiv.org/abs/2402.03899
[t2k]: https://arxiv.org/abs/2212.06541
[cgem1]: https://arxiv.org/abs/1803.07797
[cgem2]: https://arxiv.org/abs/1803.07489
[alice]: https://arxiv.org/abs/1001.1950
[alt]: https://fabio.web.cern.ch/publications.res/PP_NIMA490(2002)177.pdf
[sessa]: https://inspirehep.net/files/09876cb48a49208d97e7312de1a60d94
