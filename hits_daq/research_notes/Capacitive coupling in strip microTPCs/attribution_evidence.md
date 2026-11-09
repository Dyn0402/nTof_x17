# Attribution of the µTPC angle bias to capacitive coupling: what was actually shown

Conventions. "pdf p." is the page of the PDF file; Bortfeldt's printed pages are about 14 lower, Ntekas's about 20 lower. Numbers marked **(mine)** are my arithmetic from the quoted inputs. Local text copies: `/media/ucla/work/litmix/txt/` and `/media/ucla/work/capcoupling/attr_*/`. "Bias" means reconstructed angle above true angle unless stated.

Bottom line for the writer: capacitive coupling is demonstrated to exist and to create edge "ghost" strips, and circuit-only models can produce a bias of the published size. No source separates it from diffusion and edge geometry. The one published gas-only simulation with no circuit (Ntekas 2016) and the one sharing-off simulation (Scholer 2021) both still show a bias. The phrase "understood to be due to capacitive induction" is an ATLAS formulation that later papers copy.

## 1. Origin of the attribution, and who only repeats it

### Takeaway
Two roughly simultaneous, independent roots exist around 2014: LMU (Bortfeldt, an LTspice model) and ATLAS/NTUA (Ntekas, Iodice, with a Maxwell/Simplorer circuit model and a charge-correlation measurement). Everything else I could read cites or restates these. I could not establish a pre-2014 first statement.

### Cited Findings

Evidence grading (M = measurement of the bias vs a known angle; S = simulation of the bias; C = coupling itself measured or simulated, not its effect on the angle; A = assertion; R = repeats a citation):

| Source | Grade | What was actually shown |
|---|---|---|
| Bortfeldt PhD 2014 | M + S (circuit only) | Bias measured; LTspice predicts +9.3/+4.3/+2.0/+0.6° at 10/20/30/40°. No diffusion, no on/off comparison |
| Ntekas PhD 2016 | M + S (separately) + C | Gas-only MC gives a bias without any circuit; separate 5-strip circuit gives ~10 %; edge-strip charge ratio measured |
| Iodice JINST 2015 | M + A (S cited) | 20° beam peaks near 23°; "understood"; simulations named, not described |
| Sessa 2014, Bianco 2016, Vanadia 2015/17 | M or A | Bias reported, cause stated in one sentence |
| Lösel PhD 2017 | M + C | Simulated coupling applied as a correction; slope distributions improve |
| Flierl PhD 2018 | M + S (signal only) | Garfield/Ramo induction, time-bin subtraction; bias reduced |
| Scholer PhD 2021 | S (sharing scan) | CS = 0, 0.3, 0.5 in a diffusion-on digitisation |
| Rinnagel 2019, Jagfeld 2019, Klitzner 2019 | A or R | Attribute to coupling, cite Lösel or Bortfeldt |
| Alexeev 2019 | R + A | Near-verbatim ATLAS wording, cites Ntekas; adds one sentence on diffusion |
| Farinelli 2019 | R + C | Quotes ATLAS 15 %; measures ~5 % by injection in a GEM chamber |

- Bortfeldt's thesis states the mechanism and its model: "The model is parameter-free, the applied capacity values have been calculated from the detector and strip geometry"; currents are "simplified as step functions"; the ion tail "has been neglected" (Sec. 4.8.4, pdf p. 84) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- The same section mechanism: "even on the non-hit strip 5 a signal is detected with pulse height ph5 = 0.25ph6 ... the signal on this strip begins to rise before ionization charge reaches the strip. This leads to a systematic shift of the signal timing for this and all following strips towards earlier values" — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- The LMU journal-level statement names two effects: "signals couple capacitively onto neighboring strips ... Second, for strips at the edges of the strip cluster, the ionization electrons position is not calculated correctly" (Sec. 3.2) — [Bortfeldt et al.](https://arxiv.org/pdf/1508.00802).
- ATLAS wording: "a sizable angular bias ... The peak of the distribution is at about 23°. The effect has been understood to be due to the capacitive induction of the signal on neighboring strips (the first and last strips in the cluster playing the major role) and it is well reproduced by simulations (LTSpice, ANSOFT Maxwell) ... The simulations are in very good agreement with the data." The Fig. 11 caption is softer: "understood to be mainly due to", and the text calls it "the mechanism assumed to be responsible" (p. 11) — [Iodice 2015, JINST 10 C02026](https://inspirehep.net/files/69437e5e54ad974430c3aac860b62c8d).
- The thesis behind the ATLAS statement says the coupling study was an inference from edge-strip correlations: "These results confirm the initial speculation that the earliest and/or the latest strip of the cluster are very likely to have signal coming only from the capacitive coupling with their neighbours", and these strips "inherit the neighbouring time measurement" (Sec. 5.3.3, pdf p. 146) — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- Earlier ATLAS conference statement: "A small angular bias is observed due to capacitative induction of the signal on neighbouring strips" (Sec. 4.1, quote taken from the repo review) — [Sessa 2014](https://inspirehep.net/files/09876cb48a49208d97e7312de1a60d94).
- Repetition: "corrected the results for the capacitive induction of the signal on neighboring strips" — [Bianco 2016](https://inspirehep.net/files/d600d3d110985ae967bbd4d0b7fdeacc). "It was determined in simulation that charge-sharing between neighbouring strips is at the 10% level. This effect may lead to a bias" — [Vanadia](https://inspirehep.net/files/cf2a35f9dc497f132c073ce83c5ef49c).
- BESIII copies the ATLAS sentences nearly word for word and cites Ntekas as ref. [21]: "has been understood to be due to the capacitive induction [21] ... (the first and last strips in the strip cluster playing the major role)" (Sec. 4) — [Alexeev 2019](https://arxiv.org/pdf/1908.06253).
- Farinelli's App. B attributes the source to ATLAS: "ATLAS observed an average of 15% of the inducing charge between a strip and its nearest neighbouring strip" — [Farinelli 2019](https://arxiv.org/pdf/1904.06548).
- LMU repetition: "expected due to capacitive coupling" (Sec. 4.3) — [Rinnagel 2019](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_mrinnagel.pdf); "can be attributed to random electronic noise, falsely interpreted as a signal, and capacitive coupling" (Sec. 3.5.1, pdf p. 42) — [Klitzner 2019](https://edoc.ub.uni-muenchen.de/24286/1/Klitzner_Felix.pdf).
- The abstract of Alexopoulos NIM A 617 (2010) 161 describes a 1:10 prototype and does not mention µTPC, an angle bias or coupling; only abstract and metadata were reachable — [record](https://dspace.lib.ntua.gr/xmlui/handle/123456789/32827).

### Inferences
- "Capacitive coupling" names at least three different physical things across these papers: inter-strip capacitance of readout strips (Bortfeldt), mainly resistive-to-resistive strip coupling with RC dispersion (Lösel: "mainly due to the coupling between the resistive strips"), and weighting-field induction on a GEM readout (Flierl). A paper citing one does not support another. Grounds: [Lösel 2017](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf), Sec. 7.1.2, pdf p. 105; [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf), Sec. 6.5.1, pdf p. 111.
- ATLAS's "understood" rests on a circuit simulation plus an edge-charge measurement. It does not rest on an exclusion of alternatives. The Iodice sentence and caption do not agree on whether the mechanism is "understood" or "assumed".

### Gaps
- Iodice's Fig. 11 simulated curve exists only as a plot; I could not extract numbers. The simulation behind it is described only by software names.
- Not retrieved: Bini et al. JINST 9 C02032 (2014), Iodice's earlier papers, Alexopoulos NIM A 937 (2019), Ntekas RD51 talks beyond the title slides (2013). A pre-2014 statement may exist there.
- Our repo's `LITERATURE_MIXING.md` calls the +9° "LTspice, capacitive only". That is not accurate: "Both effects have been studied with a LTspice simulation", i.e. coupling plus edge-position, with no decomposition (pdf p. 84).

## 2. Variations of parameters that could separate coupling from diffusion

### Takeaway
No source varied one such parameter cleanly and tracked the bias as a test of the cause. The closest are confounded (pitch with capacitance ratio, strip length with noise and electronics, drift field with noise and v_d). Peaking time and drift gap: nothing found.

### Cited Findings
- **Pitch and capacitance (confounded).** "The deviation between true and reconstructed track angle is smaller in the 48×50 cm² floating strip Micromegas. This is caused by the larger ratio between coupling capacitance of anode- to readout-strips and capacitance of neighboring strips and the smaller strip pitch of 0.25 mm" (Sec. 7.4, pdf p. 183). A single conclusion then says "A smaller strip pitch leads to smaller deviations" — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- **Drift field.** At the optimum 0.17 kV/cm the shift "can be fully explained by the capacitive coupling"; away from it the increase is ascribed to "noise misinterpreted as signal" and to the larger relative timing error at higher v_d (Sec. 6.1.5, pdf p. 147). Diffusion is not discussed as a cause there — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- At low field the angle can even fall below truth, "caused by a strong dependence of the electron drift velocity on the electric field" (pdf p. 114, Ch. 5) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- NSW chamber: for large drift fields "the angle increases to a value of (25 +6/−4)°" for a true (21 ± 1)° (Fig. 4.7) — [Rinnagel 2019](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_mrinnagel.pdf).
- GEM, Flierl: the angular resolution, not the bias, tracks diffusion: "the best angle resolution here was obtained at a drift voltage where this diffusion was also minimal" (Sec. 5.3.1, pdf p. 80) — [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf).
- **Strip length.** Simulated coupling 23.4 % at 10 cm and 29.1 % at 100 cm, "rises linearly" (Sec. 7.1.2, pdf p. 105 ff.) — [Lösel 2017](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf). The measured SM2 distributions are worse, but the thesis ties that to noise: "It seems to be necessary to have a noise-to-signal ratio below 5 %" — same source, near Fig. 7.14.
- Scholer finds a charge ratio of 1.5 at SM2 outer strips, "indicating CS of 67 %", against 16 % on 10 cm prototypes and ~30 % for SM2, with strip length, threshold and front end (VMM vs APV25) all different (Sec. 8.3.3, pdf p. 90-91) — [Scholer 2021](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b).
- **Readout strip width (front-end-side coupling), simulated only.** 23 % at 80 µm falling to 11 % at 320 µm in a Maxwell/Simplorer model (Sec. 7.2, pdf p. 210-211), with no bias reported as a function of width — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- **Gas.** "It must be noted that the correction depends on the gas mixture" (App. B, no numbers on the angle) — [Farinelli 2019](https://arxiv.org/pdf/1904.06548). Klitzner's Ne:CF4 measurement in a fast, differently diffusing gas: "for all inclination angles the reconstructed value is bigger than expected" below 300 V/cm, but also "All other layers reconstruct a slightly smaller angle than expected" (p. 57, p. 115, quoted via the repo survey) — [Klitzner MSc 2016](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf).
- **Drift-velocity degeneracy.** Reduced tilt angles in Klitzner's PhD: "The reconstructed angle is a little smaller on both readout strip layers. It is a consequence of a deviation between assumed and actual electron drift veloicity" (Sec. 7.4) — [Klitzner 2019](https://edoc.ub.uni-muenchen.de/24286/1/Klitzner_Felix.pdf). Rinnagel's measured v_d is itself derived from an inclined-track fit at 15° and gives a wrong angle at 21° — [Rinnagel 2019](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_mrinnagel.pdf).

### Inferences
- Any reported shift confounds a bias of times with an error in v_d and gap. Bortfeldt's "+9 predicted" is a shift at a known v_d; other authors' absolute angle depends on how v_d was fixed. Only per-track external comparison with an independently fixed v_d escapes this.
- A drift-field scan would change D_T, v_d, the time per strip and the signal-to-noise together, so even the published scans cannot separate coupling from diffusion.

### Gaps
- No bias-vs-D_T study at fixed v_d. No study varying peaking time (Scholer holds it at 200 ns as a default). No drift gap comparison of the bias.
- Klitzner's PhD Sec. 7.6 compares Ar:CO2 with Ne:CF4 (very different transverse diffusion); I read only the structure, not the angle plots, so whether the bias changes is unknown from my notes.
- Fig. 6.12 of Bortfeldt (measured vs simulated proton angles) and Fig. 7.7 are plots; their measured values were not extracted.

## 3. Simulations with coupling or diffusion switched on and off

### Takeaway
Coupling off is done twice (Ntekas gas-only MC; Scholer CS = 0) and both retain a bias. Coupling on with no diffusion is Bortfeldt. No source runs a clean diffusion-on / diffusion-off comparison with coupling held fixed. No source runs a bias decomposition across the two.

### Cited Findings
- **Ntekas gas-only MC, the most relevant simulation I found.** "only the physical processes related to the gas are implemented in the Monte Carlo simulation and not the fully equivalent electrical circuit"; "A bias in the peak reconstructed angle is also observed ... but the bias magnitude is smaller" (Sec. 5.3.2, pdf p. 144) — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- The size: "This offset becomes larger for smaller inclination angles ranging from 0.2° for the 30° case up to 4° for the 10° angle", and "the angle bias can be eliminated" by shifting the assigned position of the first and last strips (Sec. 5.3.2, pdf p. 145) — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- Its cause as stated: "purely geometrical coming from the incomplete overlap of the outer cluster strips with the particle's trace inside the MM gas gap combined with the diffusion of the ionisation electrons" (pdf p. 144) — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- **Scholer, sharing off.** "For the case of CS disabled, the distribution is symmetric around a peak value of 7° while in the case of a correct angle reconstruction a peak position at 0° would be expected. This offset most likely stems from the positions of the first and last ionizations above the corresponding strips. However it will be subject to further investigation. With CS enabled, the peaks of the distributions move towards higher angles while the distributions alse become asymmetric" (Sec. 11.3.3, pdf p. 141; only muons above 25°) — [Scholer 2021](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b).
- Scholer's digitisation includes diffusion from a per-electron model, including a double-Gaussian transverse term (Sec. 10.1.1), with drift diffusion quoted as 365 µm (transverse), 198 µm (longitudinal) at 1 cm; charge sharing is a same-time kernel (c1 = 0.3, c2 = 0.09 by default) "applied to the time-binned charge deposits" (Sec. 10.2.3) — [Scholer 2021](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b).
- Scholer explains the CS effect: "Adding CS pulls the time measurements more towards the mean time ... the time measurement of the outer strips gets biased towards the time of the inner strips" (Sec. 10.3.6, pdf p. 120); pile-up alone biases a strip "towards the time of the charge deposit with the highest charge, which is not necessarily the earliest one" (Fig. 10.12 caption, pdf p. 123) — [Scholer 2021](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b).
- Scholer's own test beam does not show a shift: "The peak of the distribution corresponds to the actual inclination of the chamber of 28°" (Sec. 8.3, pdf p. 89; repeated in proceedings) — [Scholer 2021](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b); [Scholer 2020](https://inspirehep.net/files/e7de94e1bacfaef8203072b56f9c75d7).
- **Bortfeldt LTspice.** Coupling and edge effect on, diffusion absent, four angles, four hit positions (pdf p. 85) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf). "For 10° inclination a deviation of +5° is observed" appears in a Chapter 5 summary tied to the pion-beam characterisation (pdf p. 118), while +(9.3 ± 0.8)° is the prediction for the 6.4 cm doublet in 20 MeV protons (pdf p. 145) — same source.
- **Flierl.** Simulated induction, not capacitance, "this cross-talk ... causes a systematic error in the timing measurements"; the simulation "could nearly perfectly be reconstructed by subtracting 15 % of the signal of the first strip" (Sec. 2.4.2, repo review) — [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf). The footnote records that this "has not been considered in the simulation" for inter-strip capacitance (pdf p. 80).
- **Farinelli lumps the two on purpose:** "the corrections to account for the measured capacitive and diffusion effects were obtained with the following procedures. This allowed a correction of both effects simultaneously"; the outer strips are "due to diffusion and capacitive effects" (App. B) — [Farinelli 2019](https://arxiv.org/pdf/1904.06548).
- In a parametric GEM digitiser, small-angle µTPC agreement is tuned through diffusion: "The electron diffusion is the other parameter to be tuned in order to have a good agreement of the µTPC at small incident angle" (a factor 2 on diffusion) — [arXiv:1904.06142](https://arxiv.org/pdf/1904.06142).

### Inferences
- Comparison of the two no-coupling simulations (**mine**): Ntekas 0.2° at 30° vs Scholer about 7° for angles above 25° disagree by more than an order of magnitude. Both are single-source and neither explains it. Scholer's figure axis is "Difference between muon incident angle and the angle reconstructed"; I read the +7° sign from his text that peak positions "move towards higher angles".
- Scholer's digitisation overshoots his own test beam: simulation peak ≥ +7° with CS, measured peak at the true 28°. So his sharing-off figure is not by itself a reliable absolute, though the existence of a nonzero sharing-off offset in that code is real.
- Ntekas's gas-only result at 10° corresponds to a tan ratio of 1.41 **(mine)**. This matches the magnitude of MX17's X 1.19 and Y 1.37 in order but at different angle, gap, and pitch. It supports the MX17 statement that a mechanism with no explicit capacitive term can give a bias of this kind, not that the mechanisms are the same.
- The "+9° vs +5°" comparison in our notes may not be like for like: the two numbers appear to come from different detectors, particles or chapters. I did not read the proton measurement from the plot.

### Gaps
- No diffusion off simulation located at all. Without it, Ntekas's "combined with the diffusion" and the role of diffusion in Scholer's offset remain assertions.
- Whether Ntekas's MC includes a same-time mesh/footprint term beyond diffusion: not established from the text I read.

## 4. How much of the bias do coupling-based corrections remove?

### Takeaway
Corrections reduce the bias substantially at 15-40° but none is shown to remove it fully at small angle, and residuals are either unexplained or attributed to more coupling without evidence. Reduction alone is weak evidence, because any reduction of edge weights rotates the fit in the right direction; one author says so.

### Cited Findings
- Lösel applies the simulated (not tuned) fraction time-bin by time-bin: "For 30° and 40° the most probable value is exactly the expected value. The reconstruction of the track angle works as well for 20°, but the most probable value is slightly shifted. Whereas for 10° it still does not work very well. But this is due to too few strips hit" (Sec. 7.2.2, pdf p. 110) — [Lösel 2017](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf). The uncorrected 40° already "works quite well" (same page, Fig. 7.9 caption).
- Flierl, GEM: 20 % subtraction "reduced the systematic shift of the mean reconstructed angle considerably", to (16.99 ± 0.04)° at a 15° set angle; the 20 % was "empirically found to deliver the best angle resolution, rather than the 15 % simulated before, this might be afflicted by additional capacitive coupling" (Sec. 5.3.1, pdf p. 80) — [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf). At 30° the uncorrected mean is 32°, a "significant deviation" remaining at the largest angle in the passage.
- Ntekas: "A significant improvement in the peak angle bias is obtained ... The small remaining bias is most likely coming from the capacitive coupling between the middle strips of the cluster that is not corrected" (pdf p. 150). At 10° "the angle bias is reduced by 5°" — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- Same author, earlier page: for mid-cluster strips the effect "in most cases is negligible" (pdf p. 145) and, on the corrections, "the reduction in the angle bias is naively expected since the corrections tend to rotate the reconstructed µTPC segment around its middle towards smaller inclination angles" (pdf p. 152) — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- Iodice: after charge-ratio filtering, "a significant reduction of the angle bias" (p. 11, no number in the text) — [Iodice 2015](https://inspirehep.net/files/69437e5e54ad974430c3aac860b62c8d). Alexeev: "still in a preliminary stage, but first results showed a reduction" — [Alexeev 2019](https://arxiv.org/pdf/1908.06253).
- Bortfeldt recommends calibration, not removal: an angle-dependent scale factor from data or the prediction (pdf p. 85) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- Jagfeld implements the same time-bin-wise correction (r₀ ≈ 23-29 %, from Lösel) and mentions "not properly corrected capacitive coupling" as a residual effect — [Jagfeld 2019](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_cjagfeld.pdf).

### Inferences
- Lösel's use of an externally simulated fraction is the strongest pro-coupling correction test in this set, since it is not tuned. Its reach is limited: the improvement is shown where the uncorrected bias is already small (30-40°) and fails where the bias is largest (10°).
- A time-bin-wise neighbour subtraction is a deconvolution of a same-time kernel. It would also remove same-time mixing from diffusion and footprint if the kernel width matched, so success does not identify the source. This is consistent with Flierl needing 20 % where the simulated induction gave 15 %.
- Ntekas's "middle strips" residual explanation contradicts his own statement that mid-cluster coupling is negligible and was not tested.

### Gaps
- No paper reports the residual vs true angle after a coupling correction, with an independent reference and a fixed v_d, in a way that would permit a clean accounting of residual vs. cause.

## 5. Angle dependence and depth dependence: is the shape discriminating?

### Takeaway
All sources show a shift falling fast with angle. In tan terms it is close to Δtan ∝ 1/tan θ (excess ratio ∝ 1/tan²θ), the signature of an end-of-cluster perturbation of a line fit. That shape does not distinguish coupling from diffusion or edge geometry. Depth dependence could, but no paper measures the bias as a function of depth at the mesh and cathode ends.

### Cited Findings
- Bortfeldt's predictions: +(9.3 ± 0.8)°, +(4.3 ± 0.7)°, +(2.0 ± 0.5)°, +(0.6 ± 0.2)° at 10/20/30/40° (pdf p. 145) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- The model's time-offset profile is not symmetric: strip 7 is shifted, "for strip 8, the shift increases further and then tends towards a constant value", with an edge position error on the partially hit last strip (pdf p. 85) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf).
- Ntekas: edges opposite in sign, "the distributions are highly asymmetric towards the edges of the strip pitch with an opposite effect observed between the first and the last strip of the cluster"; bias grows as cluster size falls (Sec. 5.3.2, 5.3.4) — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65).
- Klitzner MSc explicitly: "as they are more biased by neighboring strips: For 35° around nine strips are typically hit, whereas for 25.5° only around six" (p. 53, via survey) — [Klitzner MSc 2016](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf).
- Flierl gives a mixing argument at the top of the depth column for a GEM: "For clusters with a smaller inclination angle the timing difference of the incoming electrons is suppressed by the first arriving electrons, as they already occupy the width of the full cluster and the signal on all strips appears simultaneous" (pdf p. 56) — [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf).

### Inferences
All **(mine)** from the numbers above:

| True θ | Bortfeldt predicted | tan ratio | Δtan | Δtan · tan θ |
|---|---|---|---|---|
| 10° | 19.3° | 1.99 | 0.174 | 0.031 |
| 20° | 24.3° | 1.24 | 0.088 | 0.032 |
| 30° | 32.0° | 1.08 | 0.048 | 0.028 |
| 40° | 40.6° | 1.02 | 0.018 | 0.015 |

- Others: Iodice 20→23° (1.17); Flierl 15→19° MPV (1.28), 30→32° mean (1.08); Ntekas gas-only 10→14° (1.41), 30→30.2° (1.01). Flierl's uncorrected 19° is an MPV and the corrected 16.99° is a mean, so the often-quoted "19 to 17" is not like for like.
- Shape: for a least-squares line over N equal strips, displacing the two end points by δ gives a fractional slope change of about 12δ/(N²Δt), with Δt the time per strip (large-N estimate). Bortfeldt's N = 6 mm·tan θ/0.5 mm gives N = 2.1, 4.4, 6.9, 10 and implied δ/Δt ≈ 0.36, 0.39, 0.33, 0.18. So his curve is an end-perturbation curve with a nearly fixed end shift, neither additive nor constant-multiplicative.
- Both coupling and same-time spreading from diffusion produce an end shift, so both predict this 1/N² form. In the angle plane the shape does not discriminate. What would differ in the mechanism: coupling has a kernel uniform in depth, with c independent of drift distance; diffusion has width growing as √(drift distance), so a stronger effect at the cathode end than at the mesh end. Ntekas's observation that "the peak is not very pronounced" for the latest strip "mainly due to the larger ... drift path of the primary cluster electrons that smears the effect" (pdf p. 146) is the only printed hint I found of a depth asymmetry, and it is expressed in diffusion language.
- Sign of the time shift by estimator: a leading-edge estimator can only move a strip earlier (causal), so coupling should then shift only cathode-end strips, leaving the first strip fixed apart from inherited ghosts (Bortfeldt's profile). A peak or charge-weighted estimator pulls in both directions toward the mean (Scholer's description). The MX17 pattern (+84 ns at the mesh end, −65 ns at the cathode end) is two-sided. Given MX17's stated ~4 % mid-column compression and the 'point' arm's geometric end bend (+62/−44 ns, no diffusion), this fits an edge-dominated mechanism better than a uniform kernel over depth; it also resembles Ntekas's opposite-edge geometry. Neither coupling nor diffusion is singled out.
- MX17 caution: at the survey's stated median tan ≈ 0.11 (X, P0), N ≈ 4, so 12/N² ≈ 0.8 and the observed 1.19 needs δ/Δt ≈ 0.25; at tan 0.3, N ≈ 11 and it would need δ/Δt ≈ 1.9. Whether MX17's ratio is constant in tan (multiplicative, which the notes describe) or falls like 1/N² is therefore itself the discriminating information about whether MX17's effect is the same effect as in these 5-6 mm chambers. I have no MX17 ratio-vs-tan from the brief.

### Gaps
- No paper plots the bias against depth, or fit residuals against drift time at the mesh and cathode ends separately. Bortfeldt's Fig. 4.28 is the closest, for one simulated event.

## 6. Is the coupling fraction compatible with what the simulations need?

### Takeaway
Published fractions span 5 % to 67 %, with a central value about 10-30 %. They cover what the circuit models need, but they are not independent of the simulation, not consistent within single works, and not a test of the angle effect.

### Cited Findings
| Fraction to first neighbour | Type | Device | Source |
|---|---|---|---|
| ph5 = 0.25 ph6, non-hit strip | LTspice | 6 mm, floating strip | [Bortfeldt](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf) |
| ~10 % (T chamber, 400 µm, 20 MΩ/cm) | Maxwell/Simplorer | ATLAS resistive | [Ntekas](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65), pdf p. 145 |
| ~15 % (charge ratio about 6 on edge) | Data, "simulation expectations of 15 %" | ATLAS T | [Ntekas](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65), pdf p. 146 |
| 23 % at 80 µm, 11 % at 320 µm readout | Simulated | T4C | [Ntekas](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65), pdf p. 210-211 |
| 10 % level | Simulation | ATLAS MM | [Vanadia](https://inspirehep.net/files/cf2a35f9dc497f132c073ce83c5ef49c) |
| 23.4 % / 29.1 % (10 / 100 cm) | LTspice | resistive | [Lösel](https://edoc.ub.uni-muenchen.de/21251/3/Loesel_Philipp.pdf) |
| 80 % early, 15 % final; 20 % fitted | Garfield/Ramo, GEM; fit | GEM 400 µm | [Flierl](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf) |
| 5 % after ~15 ns | Injection test | BESIII GEM with APV25 | [Farinelli](https://arxiv.org/pdf/1904.06548), Sec. 5.5 |
| 16 % small / ~30 % SM2 / 67 % at longest strips | Data, ratios | ATLAS NSW | [Scholer](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b), pdf p. 90 |
| 25 % non-hit strips | Data | LMU Ne:CF4 | [Klitzner MSc](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf), p. 57 (via survey) |

### Inferences
- The same Ntekas subsection gives ~10 % from the circuit model and calls ~15 % the "simulation expectations". These are different numbers, and "the same" here is unstated.
- Scholer's 67 % assumes the outermost strip is pure CS; edge strips with partial ionisation (his own Fig. 10.10c geometry, or Ntekas's incomplete-overlap argument) inflate it. 67 % is an upper-envelope reading, not a measured coupling.
- 5 % delayed by 15 ns (Farinelli) is qualitatively unlike the same-time kernels the simulations use; a small delayed cross talk would not produce the same time-of-arrival shifts as a same-time 15-30 %.
- With c = 0.15-0.25 the ghost-strip mechanism is plausible on charge: an edge strip at the 15 % level above threshold needs main charge ≥ 7 × threshold, satisfied for a large part of MIP clusters. This leaves room for the ghost strip to explain the small-angle effect, without showing it is the dominant cause.

### Gaps
- No paper relates measured c, chamber by chamber, to a measured bias. Nor is there a published MX17 inter-strip coupling value in the material I read; I did not estimate it.

## 7. Disagreement and alternative explanations

### Takeaway
No source flatly refutes the capacitive picture, but several give explicit alternatives: diffusion with geometry (Ntekas's gas-only result, Flierl's first-arrival statement), first-electron timing with diffusion (Fang 2024), charge-weighted pile-up (Scholer), and low-charge or noisy edge strips (Bortfeldt, Klitzner). Farinelli's correction contradicts the timing premise itself.

### Cited Findings
- **Diffusion and first arrival.** A 2024 poster states that a channel's measured time, when several primary electrons contribute, is that of the first to exceed threshold, earlier than the true drift time, and proposes a diffusion-based time correction, with about 15 % better resolution in simulation (no paper found; reported bias at about one spatial resolution, ~100 µm) — [Fang, MPGD 2024](https://mpgd2024.aconf.org/presentation/68.html).
- Flierl's first-arrival statement above (pdf p. 56) is the clearest in-print physical statement of the diffusion mechanism for strips — [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf).
- **Edge geometry (non-coupling).** Ntekas's "purely geometrical" (pdf p. 144) and the position-assignment term in Bortfeldt — [Ntekas 2016](https://inspirehep.net/files/5d0a746b3db9fe13871fa68223192c65); [Bortfeldt et al.](https://arxiv.org/pdf/1508.00802).
- **Noise on edge strips.** "noise misinterpreted as signal on non-hit neighboring strips always leads to an shift of the reconstructed angles to larger values" (pdf p. 147) — [Bortfeldt 2014](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf); Klitzner names "random electronic noise" first (pdf p. 42) — [Klitzner 2019](https://edoc.ub.uni-muenchen.de/24286/1/Klitzner_Felix.pdf).
- **Charge-weighted pile-up.** Scholer's Fig. 10.12 caption, pdf p. 123 — [Scholer 2021](https://inspirehep.net/files/362b665c04e2575c975ef950e4a5c63b).
- **Time correction disputed.** Farinelli evaluates residuals only in x: "No residual in the z direction has been evaluated because the z depends on the time measurement and the transverse diffusion or the capacitive effect should not influence that variable" (App. B) — [Farinelli 2019](https://arxiv.org/pdf/1904.06548). Compare the same collaboration: "Induced signals in the first and last strips of a cluster have same arrival times as their neighbours" — [Alexeev 2019](https://arxiv.org/pdf/1908.06253).
- Alexeev adds diffusion, unquantified: "In addition the transverse diffusion plays another important role" — [Alexeev 2019](https://arxiv.org/pdf/1908.06253).
- **Bias absent in some data.** NSW test beam peak at the true 28° despite substantial sharing — [Scholer 2020](https://inspirehep.net/files/e7de94e1bacfaef8203072b56f9c75d7). Klitzner's chambers read "slightly smaller" on other layers — [Klitzner MSc 2016](https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf).

### Inferences
- Farinelli treats coupling and diffusion as position effects; ATLAS, LMU and BESIII's Alexeev treat coupling as a time inheritance. Within the literature, the time-vs-position question is not settled.
- Ion tail and shaper response are essentially unaddressed as sources: LTspice explicitly neglects the ion tail (pdf p. 84); I found nothing that tests it.
- An absent bias at 28° (NSW) is consistent with the bias being small at that angle as the 1/N² form predicts; it also limits how much weight the simulation's +7° can bear.

### Gaps
- I could not reach a peer-reviewed version of the Fang argument, and I found no study of time-walk or threshold as a cause of bias in strip µTPCs.

## 8. Outside Micromegas: GEM, µ-RWELL, drift chambers, LAr, pad or pixel readouts

### Takeaway
GEM chambers show the bias with the same wording copied in, and there a large fraction of the spread is known not to be capacitive (induction, wide diffusion). Pad and pixel systems without strip capacitance are mostly uninformative: they either measure single electrons or do not publish a one-time-per-channel angle comparison. One low-pressure directional TPC attributes a simulated bias to diffusion. I found no LAr or drift-chamber source showing an equivalent bias.

### Cited Findings
- GEM, with no resistive layer: bias at 15° and 30° (Flierl) and in BESIII (Alexeev, Farinelli), as cited above. Flierl attributes the signal spread to weighting fields of the strips, not to circuit capacitance (Sec. 2.4.2) — [Flierl 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf).
- µ-RWELL combines µTPC and centroid with no angle-bias discussion (repo review) — [Bencivenni 2020](https://arxiv.org/pdf/2007.03223).
- A µ-TPC with a 2D-pixelised anode: a straight-line fit "induces a bias lower than 4.5° on the angle due to the diffusion of electrons collected to the grid"; attribution from a MAGBOLTZ/SRIM simulation with low-pressure gas — [Maire et al. 1310.6837](https://arxiv.org/pdf/1310.6837).
- GridPix: 55 µm pixels, one electron per hit, no angle bias reported, only a time-walk correction (repo review) — [Ligtenberg et al.](https://arxiv.org/pdf/1808.04565).
- Repo survey result (not re-derived here): LAr wires show width and charge dependence on θxz but no hit-time or slope bias was found; single-electron pixels have nothing to mix (`papers/paper2/hits_daq/LITERATURE_TPC_SURVEY.md`).
- A search for an induction-wire hit-time bias found MicroBooNE's angle-dependent wire response affecting charge rather than time — [MicroBooNE, arXiv:1910.02166](https://arxiv.org/pdf/1910.02166).

### Inferences
- The direct test of the capacitive explanation would be a many-electrons-per-channel strip µTPC with strip capacitance removed, for instance a metal-pad or pixel µTPC of comparable gap and pitch with a one-time-per-channel estimator and an external reference. I found no such published result. The closest is Maire et al., and it is a simulation, at low pressure, with no capacitance in the model.
- The 1310.6837 anode is pixelised but read out as 256 strips per axis (its Sec. II), so I do not count it as capacitance-free.

### Gaps
- Not found: an explicit angle-bias statement for µ-RWELL strips, or for a pixel or pad µTPC with charge-weighted time per pixel at a comparable gap.
- LAr long-range induction with track angle: search returned only charge-response work; I did not read further.
- Not accessible: `cds.cern.ch` returned an Anubis challenge (ATL-MUON-PROC-2015-008 was read via the INSPIRE copy of Vanadia instead). The Lösel JINST paper, Iodice 2014, Bini 2014, Alexopoulos 2019 were not read.
