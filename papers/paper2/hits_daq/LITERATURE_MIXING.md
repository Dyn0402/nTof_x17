> **Extended 2026-11-08 to every TPC family:** `LITERATURE_TPC_SURVEY.md` and `literature_survey/README.md`.

# Same-time mixing in single-time-per-strip µTPC: what the literature says

Review date 2026-11-06. Source texts were downloaded and read with `pdftotext`; the working copies are in
`/media/ucla/work/litmix/` (`pdf/`, `txt/`). "Read" below means I read the passage quoted in the full text.
Page/section references are to the file I read. Arithmetic marked **(mine)** is my own, from the inputs shown.

## 1. Short answer

1. **Others did see the bias, under another name.** Every single-time-per-strip µTPC I could read reports that the
   reconstructed angle is systematically too large (the drift-time ladder is too compressed): ATLAS-type Micromegas
   (Bortfeldt, Lösel, Rinnagel, Flierl, Sessa) and BESIII triple-GEM (Alexeev, Farinelli). They call it
   "capacitive coupling / induction onto neighbouring strips" and "mis-reconstruction of the edge strips".
2. **Their explanation is not ours.** All name capacitive or weighting-field coupling of the signal to neighbours
   (and edge-strip mis-assignment). Only the GEM papers add transverse diffusion, as a second effect, in one sentence
   (Alexeev 2019, p. 8). I found no paper that states that transverse diffusion alone, on a metallic readout, compresses
   the time ladder, and none that quantifies it. That is the part of our finding I could not find in print.
3. **They corrected it empirically instead of modelling it:** drop or down-weight the first and last strip, a charge-ratio
   cut, charge-weighted fit points, subtraction of a fixed fraction (15-20 %) of the neighbour's signal time-bin by time-bin,
   an angle-dependent calibration curve, or combining with the charge centroid. Flierl (2018) is the closest to a forward
   model: a simulated time-bin-wise subtraction of the neighbour pulse (Sec. 2.4.2). No one fits full per-strip waveforms
   of a µTPC track.
4. **Why a waveform fit was not needed there:** (i) 25 ns APV25 sampling with 11-25 ns per strip at 20-30 deg, so the ladder is
   barely resolved and the figure of merit was angular resolution (2-5 deg) and mid-gap position, not an absolute angle
   **(mine, inference)**; (ii) the bias is large at small angle and shrinks at large angle, so it was handled as a calibration
   (Bortfeldt: +5 deg at 10 deg measured); (iii) the 5 mm gaps give smaller diffusion/pitch than our 28 mm / 0.78 mm.
5. **Classical TPCs, T2K ERAM and GridPix do not have the problem by construction**: pads of 4-11 mm, charge-centroid in time
   (ALICE), no µTPC (ERAM), single electrons (GridPix). Supported by the sources; the "by construction" part is my reading.
6. **Status.** Established (read): the angle bias exists in other µTPCs, and its published explanations. Inferred (not in any
   source): that the same-time mixing is the common cause, and that diffusion is a first-order effect for them.
   A rough numerical comparison (Sec. 5) puts the published angle shifts in the same range as our 1.2-1.4, which supports
   "same effect" but does not prove it.

## 2. Comparison table

σ_T = D_T·√L. Ratio = σ_T(full gap or drift)/pitch. "Δt/strip" = pitch/(tanθ·v_d) **(mine)** at the angle shown.

| System | Gap / drift | Pitch | Gas | D_T (source) | σ_T / pitch | Δt per strip | Front end | Reported angle bias / fix |
|---|---|---|---|---|---|---|---|---|
| **Ours** (Micromegas, resistive Y, metallic-like X) | 28 mm | 0.78 mm | Ar/iC4H10 95/5 | 350-400 µm/√cm (brief) | 0.75-0.86 **(mine)** | 72 ns at tanθ=0.3, v=36 µm/ns **(mine)** | DREAM, 60 ns sampling | Ladder compressed 17-24 %, tan ratio 1.2 (X) / 1.4 (Y); waveform forward fit |
| **ATLAS NSW Micromegas** (resistive strips) | 5 mm | 0.425 / 0.45 mm (test chambers 0.40) | Ar:CO2 93:7 | not found in a source I read | = 1.66·D_T[mm/√cm] for 0.425 mm **(mine)**; < 1 for D_T < 0.6 mm/√cm | 17 ns (v=66 µm/ns) to 25 ns (v=47 µm/ns) at 20 deg, pitch 0.40 **(mine)**; v from Rinnagel Fig. 3.8 and Sessa 2014 | APV25 25 ns; VMM | Angle too large from "capacitive coupling"; drop first/last strip if charge < 1/2 (Rinnagel) or < 1/6 (Sessa) of neighbour; edge-position correction |
| **LMU floating-strip Micromegas** (metallic strips, capacitively coupled readout) | 6 mm | 0.5 mm | Ar:CO2 (93:7 / 85:15) | not read | = 1.55·D_T[mm/√cm] **(mine)** | not computed | APV25 25 ns | +9 deg expected at 10 deg (LTspice), +5 deg measured; decreases with angle; angle-dependent rescaling (Bortfeldt Sec. 4.8.4) |
| **LMU GEM / Micromegas (Flierl)** | GEM 8 mm | 0.4 mm | Ar:CO2 | not read | n/a | n/a | APV25 | MPV 19 deg for 15 deg set; 32 deg mean for 30 deg set; time-bin-wise 15-20 % neighbour subtraction brings 15 deg to 16.99 deg |
| **BESIII planar / cylindrical triple-GEM** | 3 or 5 mm conversion gap | 0.65 mm | Ar:iC4H10 90:10 | ≈350 µm/√cm (Farinelli, quoting [15]) | 0.38 for 5 mm **(mine)** | v_d not read from text | APV25 25 ns | "Tilt of the original track towards larger angles"; first/last strip weighting or suppression; 3-step data-driven shift (App. B) |
| **T2K ERAM + HA-TPC** (resistive pads) | test 15 cm (up to 1 m in HA-TPC) | pads 10.09 × 11.18 mm | Ar/CF4/iC4H10 95/3/2 | 286 µm/√cm MC default, 310 fitted (arXiv:2212.06541) | 0.11 (15 cm), 0.29 (100 cm) **(mine)** | 39 ns across one pad at tanθ=0.3, v=76.8 µm/ns **(mine)** | AFTER, 40 ns sampling, 200/412 ns peaking | No µTPC. Resistive "charge spreading" delays neighbour pads; modelled with an RC waveform fit |
| **GridPix / Timepix3** (InGrid, metallic) | ≤ 20 mm | 55 µm | Ar/CF4/iC4H10 95/3/2 | 306 µm/√cm fitted (expected 318 ± 7) | 7.9 pixels at 20 mm **(mine)** | 0.2 ns across one pixel at tanθ=0.3, v=78.86 µm/ns **(mine)** | Timepix3, ToA 1.56 ns, per electron | No angle bias reported; time-walk from ToT corrected; true 3D from single electrons |
| **ALICE TPC** (MWPC, pads) | 2.5 m (94 µs × 2.65 cm/µs) | pads 4×7.5, 6×10, 6×15 mm | Ne-CO2-N2 | 220 µm/√cm (Table 1) | 0.6-0.9 in rφ at full drift **(mine)** | 85 ns across a 7.5 mm pad at tanλ=0.3, v=26.5 µm/ns **(mine)** | 5-10 MHz, shaping 190 ns FWHM | Resolution term ∝ tan²α and a threshold-loss correction; no time bias statement found |

Notes on the table:
- ATLAS and LMU diffusion cells are empty on purpose. I did not find a D_T for Ar:CO2 93:7 at the NSW drift field in a text I
  could read (the LMU theses show it only as plots). Do not fill it from memory. The ratio is given as a function of D_T.
- The T2K and GridPix Δt cells show the drift-time span of one pad or pixel for a straight track, not a physical time of arrival.
- The ALICE ratio is large at full drift but the pad is read as a charge centroid in time and space, not as one edge time per pad.

## 3. Per-experiment notes

### 3a. ATLAS NSW and related LMU Micromegas

- **Bortfeldt PhD (LMU 2014), Sec. 4.8.4, pp. 69-72** (full text read). "Additional to the finite angular resolution ... the
  reconstructed track inclinations are subject to systematic shifts. In the following section, the influence of the capacitive
  coupling of neighboring strips and the systematic mis-reconstruction of the cluster position at the borders of the charge
  clusters are discussed." An LTspice model of the strip chain gives "For 10 deg track inclination a deviation of +9 deg is
  expected, the reconstructed angles approach the true values for increasing track inclination from above." The thesis
  suggests an angle-dependent rescale `ϑ_true = f(ϑ_reco)·ϑ_reco` (Eq. 4.28), built from the simulation or from tilted data.
  Sec. 4.8.5: "Due to capacitive coupling of adjacent strips, reconstructed angles for small track inclination are too large,
  for increasing track inclination, the difference decreases." Chapter 5 summary (about p. 105): measured "+5 deg" at 10 deg.
  Diffusion appears only in the resolution discussion, not in the bias.
- **Bortfeldt et al., Nucl. Phys. B Proc. Suppl. (arXiv:1508.00802), Sec. 3.2, p. 4** (full text read). "The reconstructed track
  inclination angle is systematically shifted towards larger values ... by two effects: First, signals couple capacitively onto
  neighboring strips. Thus measured signal timing is shifted towards earlier values for the strips, hit by electrons, created
  close to the cathode. Second, for strips at the edges of the strip cluster, the ionization electrons position is not calculated
  correctly." This is the closest published statement to our mechanism: neighbour strips inherit an earlier time, and the
  effect is worst for charge from the cathode end. The cause given is capacitive coupling, not diffusion. Detector: 6 mm gap,
  0.5 mm pitch, Ar:CO2, 25 ns APV25.
- **Rinnagel MSc (LMU 2019), Sec. 3.4.1, pp. 26-27** (full text read; NSW chamber with VMM). "Typical for the distribution is a
  tail to bigger reconstructed angles because of capacitve coupling between neighboring strips and uncorrelated detector noise.
  ... If capacitve coupling occurs between neighboring VMM channels, this directly leads to a wrong track reconstruction due to
  the spread of the time information across the channels." Correction: "the first/last strip of a cluster are discarded, when its
  PDO is below half that of the neighboring strip's PDO" (Fig. 3.7). Sec. 4.3 (Fig. 4.8): both simulated and measured drift
  velocities "lead to a wrong reconstruction for the experimental angle (21 ± 1) deg", with the explanation "capacitive coupling
  and thus strips sharing charge at the borders of a cluster ... leads to an increased reconstructed angle." Drift time
  75.7 ± 0.2 ns over a 5 ± 1 mm gap gives v = 66 ± 13 µm/ns (Fig. 3.8; the gap uncertainty dominates). Gas Ar:CO2 93:7.
- **Sessa, Frascati Phys. Ser. LIX (2014) 55, Sec. 4.1** (full text read; ATLAS MM group proceedings, non-resistive test chambers,
  250 and 400 µm pitch, 5 mm gap, APV25 25 ns). "A small angular bias is observed due to capacitative induction of the signal on
  neighbouring strips. The effect, clearly, is more evident on the first and last strip of the cluster." Fix: skip the edge strip
  if its charge is more than 6 times smaller than its neighbour's, and shift the edge-strip position by a cluster-length-dependent
  term. v_d ≈ 47 µm/ns. Test angles 0-40 deg. Note: this is not a resistive chamber.
- **Lösel MSc (LMU 2013), Sec. 4.4 and 6.4** (text searched, not read in full): µTPC with 25 ns time bins; σ_Θ = 5.66 ± 0.12 deg at
  large angles with charge weighting. No bias discussion found in the passages I searched.
- **Flierl PhD (LMU 2018), Sec. 6.5.1, p. 101** (read): on resistive-strip Micromegas "Without correction it could not be
  reconstructed" at 20 deg; "the coupling of charge to neighboring strips is not only caused by the overlap of the weighting
  fields of neighboring strips, but by direct capacitive coupling of the strips"; cites Lösel (2017) for a simulated coupling of
  23 % to 32 % for strip lengths 9-150 cm (secondary citation, I did not read Lösel 2017).
- **Not found:** Klitzner PhD (2019) full text and Lösel PhD/JINST 2017 were not read. Klitzner's ICHEP 2020 slides (read) show a
  simulation chain (ANSYS capacitances, Garfield++, APV25 response) for signal formation but no angle-bias statement on the
  slides I extracted. Scholer 2020, the Iodice/Alviggi/Bianco papers and Alexopoulos NIM A 617 (2010) 161 were not retrieved;
  their content is known to me only through the citing documents above. Treat any ATLAS statement about the 5 mm resistive chamber
  that does not come from Rinnagel, Sessa or Bortfeldt as unverified.

### 3b. BESIII triple-GEM (APV25, metallic, 5 mm, Ar:iC4H10 90:10)

- **Alexeev et al., "Triple GEM performance in magnetic field", JINST (arXiv:1908.06253), Sec. 4, pp. 7-8** (full text read).
  Test chambers at 10, 20, 30, 45 deg, no field. "An angular bias is observed in the reconstructed values of the angles measured
  by the µTPC. The effect has been understood to be due to the capacitive induction of the signal on neighboring strips (the first
  and last strips in the strip cluster playing the major role). Induced signals in the first and last strips of a cluster have same
  arrival times as their neighbours, and in the µTPC reconstruction the effect is a tilt of the original track towards larger
  angles. In addition the transverse diffusion plays another important role since the signal of a single primary electron can be
  spread on more than one strip." Fix: "A data driven correction procedure ... The induced signal is identified studying the time
  information and charge ratio of the strip w.r.t. its neighbour. A subsequent weighting or suppression of the first and/or last
  strips ... still in a preliminary stage, but first results showed a reduction of the angle bias." Pitch 0.65 mm, conversion gap
  3 or 5 mm, drift field 1.5 kV/cm, APV25 25 ns. This is the one place in my set where diffusion is named as a cause. No number.
- **Farinelli PhD (arXiv:1904.06548), App. B pp. 117-124 and Sec. 4.3.3, 5.3** (full text read). App. B: "ATLAS observed an average
  of 15 % of the inducing charge between a strip and its nearest neighbouring strip"; ghost-hit rule (charge < 300 ADC, ratio 15 %,
  same time). Corrections measured at 45 deg in Ar+10 % iC4H10: shift the last strip by about 180 µm, then a linear shift by strip
  index fitted per cluster size. "Those four strips are composed by the middle one ... plus one strip and a half (on average) on
  both sides where the signal is due to diffusion and capacitive effects." Also: "the shape of the reconstructed path is not
  properly a line due to the presence of diffusion and capacitive effects" (Sec. 4.3.3). Gas dependence: "the correction depends
  on the gas mixture." Sec. 5.3: Ar+10 % iC4H10 transverse diffusion "around 350 µm in a cm" vs 250 µm for Ar+30 % CO2.
  This thesis lumps diffusion and capacitive coupling together in the correction. It does not separate them.
- **Farinelli/Alexeev, IEEE (arXiv:1803.07266)** (text searched): states the same data-driven correction and the first µTPC use in a
  strong field. No new bias numbers extracted.
- **Bencivenni et al., µ-RWELL (arXiv:2007.03223)** (read): 0.4 mm pitch, Ar:CO2:CF4 45:15:40, APV25. Uses µTPC at large angle and
  combines it with the charge centroid to get a uniform resolution. Does not discuss an angle bias.

### 3c. T2K ERAM

- **Attie et al., NIM A 1025 (2022) 166109 (arXiv:2106.12634)** (full text read). ERAM 420 × 340 mm², 32 × 36 pads of 10.09 × 11.18 mm²,
  DLC resistive foil, AFTER with 40 ns sampling and 200 or 412 ns peaking, 95/3/2 Ar/CF4/iC4H10, 15 cm drift, 0.2 T. The signal
  in neighbouring pads "is delayed and longer in time compared to the signal from direct charge deposition ... in the leading pad".
  The fit model is a per-pad charge function with σ²(t) = 2t/τ + ω², where τ = RC and ω is "associated to the transverse
  diffusion term", convolved with the electronics response I(t) (Sec. 8-9). So this is a waveform-level forward model, but it is
  used to measure the foil resistivity and the charge sharing, not to fix a track time.
  Sec. 8.1 also states: "the transverse diffusion is small and that charge spreading dominates" at 15 cm. That supports the
  hypothesis that pads ≫ σ_T.
- **Analysis paper (arXiv:2212.06541)** (read): "charge sharing" from diffusion and "charge spreading" from the resistive foil
  are separated. "Contrary to the 'charge sharing' effect, the signals in neighbouring pads are delayed compared to the one of
  the leading pad in case of the 'charge spreading'." MC default σ_trans = 286 µm/√cm, σ_long = 210 µm/√cm; data prefer +8 %
  (310 µm/√cm). The reconstruction is a full TPC: pad clusters, a position in each cluster, and a global track fit. There is no
  strip-by-strip drift-time ladder, so the hypothesis "no µTPC on pads" is confirmed.
- Hypothesis check: pad ≫ diffusion confirmed at 15 cm (ratio 0.11 **(mine)**); at 1 m, ratio 0.29 **(mine)**, still below 1.
  Dylan's "little cross-pad mixing" is a fair reading, but the paper does not make the argument in these words.

### 3d. GridPix

- **Ligtenberg et al., "Performance of a GridPix detector based on the Timepix3 chip" (arXiv:1808.04565)** (full text read).
  "The GridPix detector ... detects single ionization electrons with high efficiency"; pitch 55 × 55 µm²; ToA with 640 MHz TDC
  (1.56 ns); drift field 280 V/cm; v_d = 78.86 µm/ns; drift length up to about 20 mm; gas 95/3/2 Ar/CF4/iC4H10.
  "Diffusion is found to be the dominating error on the track position measurement both in the pixel plane and in the drift
  direction." Fitted D_T = 306 µm/√cm (expected 318 ± 7), D_L = 226 µm/√cm (expected 201 ± 5). The only time-bias item is time
  walk, corrected as a function of ToT (Sec. 5.2). Hypothesis confirmed: each hit is one electron with its own ToA, so there is
  nothing to mix. Diffusion enters as a per-hit position/time smearing, not as a bias on a track slope. (The last sentence is my
  reading.)

### 3e. Classical TPCs and drift chambers

- **ALICE TPC, Alme et al., NIM A 622 (2010) 316 (arXiv:1001.1950), space-point resolution section (Eqs. 4-7; section number not recorded)** (read). Pads 4×7.5,
  6×10, 6×15 mm²; D = 220 µm/√cm; v_d = 2.65 cm/µs; shaping 190 ns FWHM; 5-10 MHz sampling. Resolution
  σ² ∝ p0² + p_L² L_drift + p_A² tan²α, with p_A² ∝ L_pad². "The fraction of the signal below the threshold is proportional to the
  width of the response function and increases with the drift length and the track inclination angle." They replace p_L, p_A by
  versions with (1 + p1·L_drift + p2·tan²α) factors (Eq. 7). So inclination appears as a resolution term and a zero-suppression loss,
  not as a bias in the cluster time. I found no statement of an earliest-arrival bias.
- **Angular pad effect, Kobayashi et al., NIM A 767 (2014) 439 / arXiv:1404.0982** (read, Eq. 2): σ² = σ₀² + h² tan²φ / (12·N_eff). The
  inclined track spreads over h·tanφ along the pad row and the cluster position is a charge centroid. This is a resolution term from
  the finite pad height and the number of clusters, not a bias. It is the classical analogue in the pad-row direction.
- **Drift chambers.** The "leading edge = closest electron" principle: a traditional drift chamber "uses only the arrival time of the
  overall signal in determining the distance of closest approach from a sense wire" and "the first cluster arrival time is still
  the primary datum" (arXiv:1307.8101, Sec. 1.2.4, read). MEG II prototype: "the drift time of the first cluster is obtained
  with a double threshold algorithm" (arXiv:1605.07970, Sec. 3, read). These support the first-arrival principle for wire cells.
  They do not describe a pad or strip time biased by neighbouring depths.
- **Not verified:** Blum, Rolandi, Riegler "Particle Detection with Drift Chambers" and Sauli's textbooks were not retrievable as
  text. I cannot give a page reference for a classical statement that the time from an inclined track segment is biased to the
  earliest arrival.

### 3f. Other approaches (waveform / model based)

- Flierl 2018, Sec. 2.4.2 (pp. 19-22) (read): Garfield + Ramo calculation of the strip signal for a GEM with 400 µm pitch, 2 mm
  induction gap. "As long as the electrons are far away the neighbouring strips see a signal similar to the central strip, with 80 %
  of the pulse-height for the next neighbors"; the first neighbour ends at 15 % of the central strip. "For broad charge clouds ...
  this cross-talk on the other hand influences the signal shape of strips, which are reached later by the electron cloud. This causes
  a systematic error in the timing measurements." A 1000-electron cloud released 5 ns after another gives a 1.5 ns shift; "The signal
  of the second electron cloud on the other hand could nearly perfectly be reconstructed by subtracting 15 % of the signal of the
  first strip". Sec. 5.3.1 (p. 70): uncorrected MPV 19 deg at 15 deg set, mean 32 deg at 30 deg set; the 20 % empirical correction
  (15 % simulated) gives 16.99 ± 0.04 deg at 15 deg. This is the nearest published relative of our forward model: a per-strip
  waveform subtraction. It models the induction footprint (our "prompt footprint") and does not model diffusion between depths.
- Neural-network, deconvolution, Kalman, CLAS12 MVT, COMPASS, sPHENIX TPOT, Muon g-2 and ASACUSA searches returned nothing on
  point. I found no µTPC paper that models strip time as a charge-weighted or earliest-arrival mix of depths. This is a negative
  search result, not a proof of absence.

## 4. Numerical cross-check of published angle shifts (mine)

Compression of the time ladder equals the ratio tan(θ_reco)/tan(θ_true) for a fixed v_d. From the numbers quoted above:

| Source | True | Reconstructed | tan ratio | Note |
|---|---|---|---|---|
| Bortfeldt, simulation | 10 deg | 19 deg | 1.95 | LTspice, capacitive only |
| Bortfeldt, measured | 10 deg | 15 deg | 1.52 | "+5 deg" |
| Flierl GEM, uncorrected | 15 deg | 19 deg (MPV) | 1.28 | 8 mm gap, 0.4 mm pitch |
| Flierl GEM, uncorrected | 30 deg | 32 deg (mean) | 1.08 | |
| Flierl GEM, corrected | 15 deg | 16.99 deg | 1.14 | after 20 % subtraction |
| Ours | tan 0.3-0.6 | | 1.2 (X), 1.4 (Y) | 28 mm, 0.78 mm |

The published shifts fall in the range 1.1-1.9 and shrink with angle. Ours (1.2-1.4) is inside it. The comparison is rough:
different gaps, pitches, clustering and estimators, and mean vs most-probable angles. Our bias is a multiplicative ladder
compression at depth; theirs is mostly quoted as an additive angle shift at fixed angle.

## 5. Draft introduction paragraph

A micro-TPC (µTPC) reconstructs a track segment from the strip positions and one drift time per strip. Since the first
Micromegas and GEM implementations, the reconstructed angle has been found systematically larger than the true angle, most
strongly at small inclination and for the first and last strips of a cluster [L1-L4, L6]. The published explanation is capacitive
or weighting-field coupling of the signal onto neighbouring strips, which gives them the earlier time of the charge nearest the
mesh [L1, L2]; for GEM readout, transverse diffusion is named as a further contribution [L6]. The remedies are empirical:
dropping or down-weighting the edge strips [L3, L5, L6, L7], subtracting a fixed fraction of the neighbour pulse [L4], an
angle-dependent calibration [L1], or combining the µTPC with the charge centroid [L16]. Pad TPCs do not reconstruct a segment from one edge time per
pad across a short gap: T2K's pads are much wider than the diffusion [L8, L9], ALICE reads each pad as a charge centroid
in time along a long drift [L11], and GridPix records single electrons [L10].
We show that on a 28 mm gap with a 0.78 mm pitch, where σ_T at the cathode is close to the strip pitch, any single-time estimator
compresses the drift-time ladder by 17-24 %, also on a metallic readout, and that resistive charge spreading roughly doubles the
excess on the view across the strips. We therefore fit the digitised waveforms with a forward model instead of using one time per strip.

(Every statement is traceable to the sources in Sec. 7 except the last two sentences, which are our result. "Any single-time
estimator ... on a metallic readout" comes from our P0 study, not from the literature.)

**Reviewer note (2026-11-06, main session).** The first draft said pad TPCs avoid the effect because "pads are much
larger than the diffusion width". That holds for T2K (σ_T/pad 0.11–0.29) but not for ALICE, where the table above gives
0.6–0.9, as large as ours. The difference there is the estimator and the geometry: a charge-centroid time on a segment
far from the gap ends, rather than an edge time on strips near the mesh and the cathode. Our ladder compression comes
largely from the gap ends (the mixing is one-sided there), which a long drift does not have. That last point is ours
and was checked against P0 on 2026-11-06 (`p0_depth.py`, `results/p0_depth_sat_det3.json`): mid-column the data
ladder is compressed only ~4 % (model/metallic 7–8 %); the end bins carry the rest (+84 ns at the mesh end, −65 ns at
the cathode end). Part of that end bend is geometric, present with no diffusion at all ('point' arm +62 / −44 ns:
strips past the column end see one-sided charge from their width), so the paper should say "gap ends" without
attributing all of it to diffusion. Quotes spot-checked in
`/media/ucla/work/litmix/txt/`: Alexeev 2019 l.412 (diffusion), Bortfeldt arXiv:1508.00802 l.233 (earlier timing
from cathode electrons), Flierl l.1284 and l.4135 (15 % / 20 % subtraction, 16.99°).

## 6. What I could not verify

- A Magboltz D_T for Ar:CO2 93:7 at the NSW drift field, and for the LMU floating-strip gases, from a text I read. Ratios are given
  as a function of D_T. A Garfield/Magboltz run would settle it; the local `garfield_sim/` needs ROOT/Garfield, which is not on
  this desktop's venv.
- A drift velocity for the BESIII 90:10 gas at 1.5 kV/cm (read only as a plot label), so no Δt per strip for BESIII.
- Klitzner PhD (LMU 2019), Lösel PhD/JINST 2017, Scholer 2020, Iodice JINST 2014 and later, Alexopoulos NIM A 617 (2010) 161,
  Bortfeldt 2014 NIM A paper: not retrieved as text. The Lösel "23-32 %" coupling is only a secondary citation via Flierl.
- Whether any ATLAS group tested diffusion vs coupling as the cause of the angle shift (e.g. by changing gas or drift field).
  Bortfeldt's thesis shows a drift-field dependence of the resolution but I did not find a bias-vs-D_T test.
- Any classical textbook statement of the earliest-arrival bias for an inclined track on a pad or wire (Blum et al., Sauli).
- CLAS12 MVT, COMPASS, sPHENIX TPOT, Muon g-2, ASACUSA µTPC bias statements: nothing found.
- The angle ranges in the ATLAS NSW requirement (8-30 deg) came through a search summary of a thesis appendix, not a text I read.
- My ladder-compression conversion in Sec. 4 assumes the same v_d for true and reconstructed angle.

## 7. References

Read = full text downloaded and the quoted passages read. Searched = full text downloaded, only keyword-located passages read.
Abstract = abstract or search snippet only.

- [L1] J. Bortfeldt, *Development of floating strip Micromegas detectors*, PhD thesis, LMU Munich 2014.
  https://edoc.ub.uni-muenchen.de/16972/ (PDF `.../16972/1/Bortfeldt_Jonathan.pdf`). Read (Sec. 4.8.4, 4.8.5, ch. 5 summary).
- [L2] J. Bortfeldt et al., *High-Rate Capable Floating Strip Micromegas*, arXiv:1508.00802. Read (Sec. 3.2).
- [L3] M. P. Rinnagel, *Investigations of the VMM readout chip for Micromegas detectors of the ATLAS NSW upgrade*, MSc thesis, LMU 2019.
  https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_mrinnagel.pdf. Read (Sec. 3.4, 4.3).
- [L4] B. Flierl, *Particle tracking with micro-pattern gaseous detectors*, PhD thesis, LMU 2018.
  https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf. Read (Sec. 2.4.2, 5.3.1, 6.5.1).
- [L5] M. Sessa (ATLAS Micromegas group), *Performance study of Micromegas muon chambers for ATLAS upgrade*, Frascati Phys. Ser. LIX (2014) 55.
  https://inspirehep.net/files/09876cb48a49208d97e7312de1a60d94. Read (Sec. 3.2, 4.1).
- [L6] M. Alexeev et al. (BESIII CGEM), *Triple GEM performance in magnetic field*, arXiv:1908.06253. Read (Sec. 4).
- [L7] R. Farinelli, *Research and development in cylindrical triple-GEM detector with µTPC readout for the BESIII experiment*,
  PhD thesis, arXiv:1904.06548. Read (App. B, Sec. 4.3.3, 5.3, 5.5).
- [L8] D. Attié et al., *Characterization of resistive Micromegas detectors for the upgrade of the T2K Near Detector TPCs*,
  NIM A 1025 (2022) 166109, arXiv:2106.12634. Read (Sec. 2, 8, 9).
- [L9] Analysis of test beam data of a TPC with resistive Micromegas for the T2K ND upgrade, arXiv:2212.06541. Read (MC and diffusion sections).
- [L10] Ligtenberg et al., *Performance of a GridPix detector based on the Timepix3 chip*, arXiv:1808.04565. Read.
- [L11] J. Alme et al., *The ALICE TPC ...*, NIM A 622 (2010) 316, arXiv:1001.1950. Read (Table 1, Sec. on space-point resolution).
- [L12] M. Kobayashi et al., *Cosmic ray tests of a GEM-based TPC prototype ... II*, NIM A 767 (2014) 439, arXiv:1404.0982. Read (Eq. 2).
- [L13] Cluster counting in a full-length drift chamber, arXiv:1307.8101, Sec. 1.2.4. Read (that passage).
- [L14] MEG II drift chamber single-hit resolution, arXiv:1605.07970, Sec. 3. Read (that passage).
- [L15] P. Lösel, MSc thesis, LMU 2013, https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_ploesel.pdf. Searched (Sec. 4.4, 6.4).
- [L16] G. Bencivenni et al., *On the space resolution of the µ-RWELL*, arXiv:2007.03223. Read (intro, µTPC section).
- [L17] F. Klitzner, ICHEP 2020 slides, https://indico.cern.ch/event/868940/contributions/3814075/attachments/2080489/3498219/ichep_fsm2d_klitzner_v4.pdf.
  Read (first slides only; no bias statement seen).
- [L18] Alexeev et al., *Performance of the micro-TPC reconstruction for GEM detectors at high rate*, arXiv:1803.07266. Searched.
