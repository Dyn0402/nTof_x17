# Large gaseous pad TPCs: charge spreading across drift depth, and whether a depth-mixing or angle bias exists

Retrieved 2026-11-08. "Read" means I read the passage in the `pdftotext` output of the arXiv or publisher PDF.
Items marked **(derived)** are my arithmetic or toy, not in any source. Section numbers are given only where they appear in the text I read.
Not read in full, and flagged where used: ALEPH NIM papers, PEP-4 papers, Blum-Rolandi-Riegler, ALICE TDR-016.

## Geometry: drift, pads, diffusion, sampling, lever arm

### Takeaway
Every large TPC has sigma_T/pad of 0.1-1, comparable to MX17's 0.75-0.86, so a small sigma_T/pitch is **not** what protects them. What differs by roughly 10-70x is sigma_T divided by the projected track length that the sampled coordinate spans (MX17 about 0.07 at tan 0.3; large TPCs about 0.001-0.007, derived below for assumed track segments).

### Cited Findings

| Detector | Max drift | Pads (across x along track normal) | D_T | Timing | Source |
|---|---|---|---|---|---|
| ALICE MWPC | 250 cm class, v_d 2.65 cm/us | 4x7.5, 6x10, 6x15 mm | 220 um/sqrt(cm) | 5-10 MHz, 190 ns FWHM | [Alme 2010](https://arxiv.org/abs/1001.1950) Table 1, and repo `LITERATURE_MIXING.md` |
| ALICE GEM | 250 cm, 2.58 cm/us, 97 us | 152 rows; pad sizes not extracted from the text I read | D_T 209, D_L 221 um/sqrt(cm) | SAMPA, 5 MHz, 160 ns peaking | [ALICE TPC 2020](https://arxiv.org/abs/2012.09518) Table 1 |
| STAR | 209.3 cm, 5.45 cm/us | inner 2.85x11.5, outer 6.20x19.5 mm (pitch 6.7x20.0 outer); 13+32 rows | 230 um/sqrt(cm) at 0.5 T; D_L 360 | 9.4 MHz, 180 ns FWHM | [Anderson 2003](https://arxiv.org/abs/nucl-ex/0301015) Tables 1, 3 |
| ALEPH | not found | 6.2x30 mm, 21 rows at cos(theta) <= 0.8 | not found | not found | [Wiedenmann, CERN 2001 slides](https://www.mpp.mpg.de/~settles/tpc/Cern_LC.pdf) (slides, secondary) |
| PEP-4 | not found | not found | not found | not found | Only abstract-level: 8.5 atm Ar/CH4 80/20, 4 kG, per [record of Aihara et al. 1983](https://scilit.com/publications/b5eb0a2185840314d84418cb0d5063af) |
| T2K HA-TPC ERAM | 97.25 cm (full HA-TPC), 7.68 cm/us | 10.09x11.18 mm, 32x36 | 286 MC, 310 um/sqrt(cm) fitted | AFTER, 40 ns sampling, 200/412 ns peaking | [Attie 2022, 2106.12634](https://arxiv.org/abs/2106.12634); [DESY test beam, 2212.06541](https://arxiv.org/abs/2212.06541) |
| ILD LCTPC | 2.3 m | 1.3 mm pads named in the 2-hit study; about 1x6 mm typical (per CEPC paper) | 91 um/sqrt(cm) at 1 T, 30 at 3.5 T | not extracted | [ILD IDR Sec. 5, arXiv:2003.01116](https://arxiv.org/abs/2003.01116); [PoS 476, 1098](https://pos.sissa.it/476/1098/pdf) |
| LCTPC resistive MPGD prototype | prototype | 2x6 mm | D_Tr about 19 um/sqrt(cm) at high B, gas-dependent | not extracted | [Dixit et al.](https://arxiv.org/abs/physics/0703243) |
| sPHENIX | about 1 m | radial about 1 cm; azimuthal 1-2 mm zigzag; 40 layers; avg 2x12.5 mm | not given numerically in the text read | SAMPA, 80 ns peaking | [sPHENIX TDR Sec. TPC](https://indico.bnl.gov/event/7081/attachments/25527/38284/sphenix_tdr_20191010.pdf); [2110.02082](https://arxiv.org/abs/2110.02082); [1801.03087, about 40 layers](https://arxiv.org/abs/1801.03087); [2212.05541](https://arxiv.org/abs/2212.05541) |
| CEPC | 4.7 m total length, 0.3-1.8 m radius | baseline pixel 500x500 um; pad about 1x6 mm | not found | not found | [CEPC CDR](https://arxiv.org/abs/1811.10545); [PoS 476, 1098](https://pos.sissa.it/476/1098/pdf) |

- STAR quantifies the same scale directly: 3.3 mm sigma_T at full drift, and the longitudinal spread is "about 230 ns FWHM", which "sets the scale for the resolution of the tracking system in the drift direction". — [Anderson 2003, Sec. on gas](https://arxiv.org/abs/nucl-ex/0301015)
- sPHENIX TDR, on its fast gas: "With low diffusion and high drift velocity, there is a mis-match with the electronics time constant being longer than we would prefer." — [TDR, TPC chapter](https://indico.bnl.gov/event/7081/attachments/25527/38284/sphenix_tdr_20191010.pdf)
- T2K ERAM X-ray charge clouds at 3 cm: transverse RMS 540 um and longitudinal RMS 4.5 ns. — [2303.04481 Sec. 7](https://arxiv.org/abs/2303.04481)

### Inferences (derived)
- sigma_T(max) in mm and sigma_T/pitch: ALICE MWPC 3.5 (0.58-0.87 across pads); ALICE GEM 3.3 (pad width not read); STAR 3.3 (0.49 outer; 0.99 inner at 3.35 mm pitch); T2K 3.1 (0.27-0.31); ILD 0.46 at 3.5 T (0.35 across a 1.3 mm pad). MX17 0.75-0.86.
- Time bin in mm of drift: ALICE 2.65-5.3, STAR 5.8, T2K 3.1 per sample with 15-32 mm of peaking time, MX17 about 2.2 (60 ns x 36 um/ns). At 2.2 mm per sample a 28 mm MX17 column is about 13 samples.
- Lever arm in pads: ALICE GEM 152 rows (63+89 in its Table 1), STAR 45, ALEPH 21, T2K 32-36 columns, sPHENIX about 40. MX17 is about 12 strips at tan 0.3 (28 mm x 0.3 / 0.78 mm).

### Gaps
- ALICE GEM pad dimensions from a text I read; ALEPH and PEP-4 geometry, diffusion and timing from primary text; sPHENIX D_T. Not filled from memory.

## Which reconstruction, and on which axis charge mixing acts

### Takeaway
Large TPCs reconstruct **one 3-D point per pad row** from a charge-weighted cluster in (pad, time). Depth mixing along the row-normal direction is integrated, and the time is a charge-weighted mean, not an edge. T2K ERAM is the exception on the estimator: its per-pad time is the bin of the waveform maximum, and its position comes from the pad response, not from a time ladder.

### Cited Findings
- STAR, x: three-pad Gaussian fit (Eqs. 1-2), replaced by a threshold-weighted mean at large crossing angle; "The total signal is summed over all above-threshold time buckets." — [Anderson 2003 Sec. 5.1](https://arxiv.org/abs/nucl-ex/0301015)
- STAR, z: "calculated by measuring the time of arrival of the electrons in 'time buckets' and weighting the average by the amount of charge collected in each bucket." The cluster covers several buckets "because of three phenomena: the longitudinal diffusion of the drifting electrons, the shaping of the signal by the preamplifier electronics, and the track dip angle." — [Anderson 2003, z reconstruction section](https://arxiv.org/abs/nucl-ex/0301015)
- STAR, the dip-angle projection: "The ionization electrons are spread over a distance d along the beam axis, with d = L/ tan(θ) and L is the length of the pad." — [Anderson 2003](https://arxiv.org/abs/nucl-ex/0301015)
- ALICE: the space-point estimator is a centre of gravity (sigma_COG); pulser pulses are centre-of-gravity in a window of "minus two to plus two time bins around the maximum bin, as used in the offline cluster finder." — [Alme 2010 Sec. 10.5, 11.1](https://arxiv.org/abs/1001.1950)
- sPHENIX: "locates local maxima in the Z vs r-φ distribution ... Then the cluster centroid is evaluated using a weighted sum of the hits in the cluster." — [TDR, tracking chapter](https://indico.bnl.gov/event/7081/attachments/25527/38284/sphenix_tdr_20191010.pdf)
- T2K ERAM: "the charge in the pad is defined as the maximum of the waveform and the time is defined as the time bin at which the waveform reaches its maximum." — [2212.06541](https://arxiv.org/abs/2212.06541)
- ALEPH: z from pads and wires, 500 um. — [Wiedenmann slides](https://www.mpp.mpg.de/~settles/tpc/Cern_LC.pdf); a [search-level summary](https://usiena-air.unisi.it/handle/11365/35964) instead quotes 800 um at 90 deg, unresolved conflict and neither is the NIM paper.
- CEPC baseline 100 um in r-phi and 500 um in z. — [CDR](https://arxiv.org/abs/1811.10545)

### Inferences
- For dip angle, the pad-row *length* plays the role of the MX17 strip: the row integrates d = L/tan(theta) of drift. Mixing in the row-normal direction is then a blur of a row point along the track, not a compression of a drift-time ladder.
- MX17 fits a track from one time per strip; each strip mixes drift depths. Large TPCs fit a track from many (row, time) centroids, and diffusion in the row-normal direction moves charge along the track.

### Gaps
- I read no ALEPH or PEP-4 reconstruction text, only slide or abstract-level material.

## Reported depth-mixing, angle or long-cluster terms, and their treatment

### Takeaway
All reported terms are **resolution** terms (angular pad, angular wire, E×B, dip-angle projection), zero-suppression losses, or **in-plane position** bias maps. I found no statement of a track-angle-dependent drift-coordinate bias or a ladder compression in any large TPC source I read. That is a negative search result, not proof of absence.

### Cited Findings
- ALICE resolution: "σ²COG ∝ p0² + p²L LDrift + p²A tan²α" and "The fraction of the signal below the threshold is proportional to the width of the response function and increases with the drift length and the track inclination angle," handled by replacing p_L and p_A with (1 + p1 L_Drift + p2 tan²α) versions, "mostly determined by geometrical factors." — [Alme 2010 Sec. 11.1, Eqs. 4, 7](https://arxiv.org/abs/1001.1950)
- STAR dip-angle term: "The resolution deteriorates, however, with increasing dip angle because the length of path received by a pad is greater than the shaping time of the electronics (times drift velocity) and the ionization fluctuations along the particle path are not fully integrated out of the problem." The drift dependence is weak "because of the large shaping time ... (≈ 1 cm), [which] is comparable to or greater than the longitudinal diffusion width (≈ 0.5 cm)." — [Anderson 2003 Sec. 5.2](https://arxiv.org/abs/nucl-ex/0301015)
- STAR, crossing-angle: "There is an error in position determination for tracks crossing a pad row at an angle due to granularity in the ionization process (Landau fluctuations). If large clusters of ionization occur at the edge of the pad row they pull the measured centroid away from the true track center," partly cancelled by adjacent-row crosstalk that produces "an oppositely directed error." Sec. 5.1 adds a Gaussian-PRF "x-dependent bias of a few hundred µm". — [Anderson 2003, readout and Sec. 5.1](https://arxiv.org/abs/nucl-ex/0301015)
- Tail cancellation: STAR "cancellation circuitry to remove the long current tail characteristic of MWPCs"; ALICE has a Tail Cancellation Filter in the ALTRO. — [Anderson 2003](https://arxiv.org/abs/nucl-ex/0301015); [Alme 2010, electronics section](https://arxiv.org/abs/1001.1950)
- STAR channel timing: "Fluctuations on the order of 0.2 time buckets are observed between different channels", corrected by a pulser. — [Anderson 2003](https://arxiv.org/abs/nucl-ex/0301015)
- ALEPH: resolution for high-momentum tracks "dominated by the E×B and angular effects" (abstract paraphrase, full text not read). — [Amendolia et al., NIM A 283 (1989) 573](https://pure.psu.edu/en/publications/the-spatial-resolution-of-the-aleph-tpc/)
- PEP-4: spatial resolution and pad response "on drift distance and track-anode crossing angle" were studied (abstract); full text not read. — [Aihara et al. 1983](https://scilit.com/publications/b5eb0a2185840314d84418cb0d5063af)
- LCTPC GEM: "free from the E × B and the angular wire effects inherent in conventional TPCs with MWPC readout", but "the angular pad effect adds an almost constant offset to the resolution with its amount depending on the pad height as well as the track angle." — [Kobayashi 2014, arXiv:1404.0982](https://arxiv.org/abs/1404.0982)
- CEPC proceedings: pad resolution depends on the track angle to the pad rows, pixel resolution does not, cited as a theoretical formula. — [PoS 476, 1098](https://pos.sissa.it/476/1098/pdf)
- LCTPC resistive anode: "A bias of up to 150 µm is observed which may be attributed to a non-uniform RC constant due to inhomogeneities in the gap size and the resistivity of the foil. However, this bias is due to geometry only and can easily be corrected," remaining bias "negligible"; the bias is x_row − x_track vs x_track. — [Boudjemline/Dixit, physics/0510085](https://arxiv.org/abs/physics/0510085). Same group: initial bias up to 100 um, about ±20 um after correction; "attributed to point-to-point variations in the capacitance per unit area and the surface resistivity"; only track angles |phi| < 5 deg analysed. — [physics/0703243](https://arxiv.org/abs/physics/0703243)
- ALEPH distortion formula contains a Δr·tan(lambda) term in Δz_observed. — [Wiedenmann slides](https://www.mpp.mpg.de/~settles/tpc/Cern_LC.pdf). ALICE slides state δZ = dz − dr·tan(lambda) for residuals (search summary, slides not opened). — [BNL 2019 slides](https://indico.bnl.gov/event/7096/contributions/33794/attachments/26105/39435/2019-07-11_SpaceChargeCalibration.pdf)

### Inferences
- The closest published family is the angular pad/wire effect, but it is a variance term and comes from granularity, not from time-compressed depths.
- The tan(lambda) distortion terms are geometric conversions of a radial displacement into z, not a mixing effect.

### Gaps
- Blum-Rolandi-Riegler full text: retrievable copy failed (a hosted URL returned a non-PDF); only catalogue entries were seen. No page reference. The chapters "Coordinate Measurement and Fundamental Limits of Accuracy" exist per [the table of contents](https://search.library.upatras.gr/Record/978-3-540-76684-1/TOC?lng=en).
- ALEPH NIM A 294 (1990) and 360 (1995), PEP-4 IEEE papers: no full text.

## T2K ERAM specifically

### Takeaway
T2K explicitly separates undelayed diffusion "sharing" from delayed RC "spreading". It has a simultaneous waveform fit, used for RC and gain maps, not per-event tracking. It reads position at the waveform maximum, and I found no angle-dependent drift-coordinate bias reported. Its reported biases are in-plane and assigned mainly to E×B and field non-uniformity.

### Cited Findings
- "Contrary to the 'charge sharing' effect, the signals in neighbouring pads are delayed compared to the one of the leading pad in case of the 'charge spreading'." — [2212.06541, introduction of the analysis](https://arxiv.org/abs/2212.06541)
- "The signal induced by the resistive layer is smaller in amplitude, delayed and longer in time compared to the signal from direct charge deposition ... The time delay of the waveform increases with the distance of the adjacent pad to the track." "For each pad, the maximum of the waveform is used as an estimator of the charge, which corresponds to different times for the leading and adjacent pads." — [2106.12634](https://arxiv.org/abs/2106.12634)
- Per-event global waveform use is not yet done: "Studies are underway aiming at using simultaneously the full information of the waveforms of all the pads around the tracks in a global fit, to improve the track reconstruction accuracy." — [2212.06541, introduction of the analysis](https://arxiv.org/abs/2212.06541)
- Waveform fit, X-ray events only: "All the waveforms generated in one X-ray event are fitted simultaneously" (3x3 pads, t0 = time of deposition in the leading pad), for RC and gain maps; "ω associated to the transverse diffusion term is fixed at 540 µm and the longitudinal diffusion is neglected." — [2303.04481 Sec. 7](https://arxiv.org/abs/2303.04481)
- Clustering is adapted to track angle (column, diagonal, 2-by-1, 3-by-1 patterns, tangents 0, 1, 0.5, 0.3). — [2106.12634](https://arxiv.org/abs/2106.12634). Resolution 200-800 um at all angles and drifts. — [2212.06541](https://arxiv.org/abs/2212.06541)
- Bias: mean of residuals per cluster. "For the smallest and largest drift distances, biases have visible patterns with large and opposite biases at the beginning and at the end of the track. This pattern is not observed for tracks in the center of the ERAM." "The observed dependencies of the biases are not reproduced by the simulation and can point to effects related to non-uniformities in the magnetic field ... and the E×B effect." Inclined tracks: "the biases do not depend on the drift distance or on the φ angle." — [2212.06541 Sec. 9.3](https://arxiv.org/abs/2212.06541)
- An apparent inclination is defined from E×B, phi_app = atan((yL − yR)/X_ERAM), an in-plane angle. — [2212.06541 Sec. 12](https://arxiv.org/abs/2212.06541)
- External tagging in the test beam: scintillation counter trigger only; no silicon telescope is mentioned in either test-beam paper. — [2106.12634](https://arxiv.org/abs/2106.12634)

### Inferences
- T2K's "opposite biases at beginning and end" has the same form as an MX17 end effect, but it is in the pad plane, in a magnetic field, and T2K blames E×B. I would not cite it as the same mechanism.
- No drift-coordinate bias study in either paper: the sources are silent, not negative.

### Gaps
- Whether RC delay shifts the T2K track Z (it could, at the level of the leading-pad-to-neighbour delay times 77 um/ns). Neither paper tests it.

## External reference for drift coordinate or track angle

### Takeaway
Drift scale is checked externally, by lasers and silicon, at about 0.1 %. Per-track tan-lambda against an independent telescope at 1-20 % is not reported in any source I read. So hypothesis (d) is partly right, but "nobody checks" is wrong for the scale.

### Cited Findings
- ALICE drift calibration methods: "matching laser tracks with the surveyed mirror positions; matching with tracks from the Inner Tracking System (ITS); matching of the TPC primary vertices from the two halves of the TPC; matching tracks from two halves of the TPC using cosmic tracks." — [Alme 2010, drift-velocity calibration](https://arxiv.org/abs/1001.1950)
- ALICE Run 3 space-point calibration uses ITS-TPC tracks with TRD and TOF as references, restoring resolution to "a few hundred micrometers". — [Schmidt 2020, arXiv:2003.03174](https://arxiv.org/abs/2003.03174)
- STAR: "The drift velocity ... must be known with a precision of 0.1%", measured "every few hours using artificial tracks created by lasers beams"; offset tuned by matching interaction vertices between TPC halves. — [Anderson 2003 Sec. 5.2](https://arxiv.org/abs/nucl-ex/0301015)
- T2K measured v_d with drift scans and the GMC: 7.68 ± 0.03 vs 7.81 ± 0.02 cm/us. — [2106.12634](https://arxiv.org/abs/2106.12634)

### Inferences
- A track-based scale calibration (ITS match, vertex match) would **absorb** any multiplicative compression into an effective v_d. So large TPCs could not distinguish a percent-level ladder compression from a v_d shift, unless they compare with Magboltz or a surveyed laser at known depth.

### Gaps
- A published test of tan-lambda(TPC) vs tan-lambda(silicon) as a function of position within a TPC column, for any of these experiments.

## Physical scaling reasons: the four hypotheses

### Takeaway
(b) is the strongest: mixing width over projected track length is roughly 10-70x smaller. (a) is partly right, but for a charge-weighted time rather than time slices, and it does not fix gap ends. (c) holds only for scale effects and in-plane resistive bias. (d) is true at the angle level only.

### Cited Findings
- The sources above (STAR, ALICE, sPHENIX) establish charge-weighted time and many rows; T2K is peak-time on wide pads. No source states any of these four arguments. They are mine.

### Inferences (derived; none in the literature)
**Analytic.** Let x = s·z, with sigma²(z) = D²·z and Gaussian transverse spreading.
- A charge-weighted mean of z at fixed x is z = x/s + D²/s² to first order in the gradient of sigma². The offset is **independent of depth**, so the slope is unchanged. MX17: D² = 0.012 mm, s = 0.3, offset 0.14 mm.
- An early-arrival time reads z − c·sigma(z)/s, with c about 0.84 for a 20 % quantile. Its local compression is c·D/(2·sqrt(z)·s). MX17 at z = 15 mm: 4.0 %. ALICE at z = 1300 mm, s = 1: 0.08 %.

**Toy** (geometry only: no pulse shape, footprint, RC or threshold. A 20 % charge quantile stands in for a leading edge). Strip or row charge from a Gaussian, line fit over strips above 15 % of peak. Script: `/home/dylan/.claude/jobs/d56318af/tmp/padtpc_scripts/toy.py`.

| Case | sigma_end/column | tan ratio, all strips: mean / q20 | middle 60 % of track: mean / q20 |
|---|---|---|---|
| MX17 0.78 mm, s=0.3, 28 mm | 0.071 | 1.062 / 1.082 | 1.000 / 1.036 |
| same, with sigma0 = 0.3 mm | 0.080 | 1.074 / 1.087 | 1.001 / 1.031 |
| same, s = 0.6 | 0.035 | 1.011 / 1.027 | 1.000 / 1.018 |
| same, pitch 1.56 mm | 0.071 | 1.073 / 1.098 | 1.002 / 1.023 |
| same, no diffusion | 0 | 1.011 / 1.019 | 1.000 / 1.000 |
| ALICE-like, 10 mm rows, 1.0-1.6 m | 0.005 | 1.001 / 1.001 | 1.000 / 1.000 |
| STAR-like 20 mm rows | 0.005 | 1.003 / 1.003 | 1.000 / 1.000 |
| T2K-like 11 mm | 0.006 | 1.002 / 1.003 | 1.000 / 1.001 |
| sPHENIX-like 12.5 mm | 0.002 | 1.002 / 1.002 | 1.000 / 1.000 |

The toy reproduces the *ordering* (worse at small s, worse with wider strips, as the MX17 simulation finds) and the mid-column compression of about 4 % noted in the repo's P0 check. It under-predicts the MX17 total (1.06-1.09 against 1.14-1.22 simulated), because it has no pulse or footprint model. Use it for scaling only.

- **(a) Time-slice centroid gives a symmetric blur.** Partly supported, with a correction. Large TPCs use a charge-weighted cluster per row, not time slices (STAR, ALICE, sPHENIX). A charge-weighted time does not compress the interior, analytically and in the toy. Contrary evidence: (i) it does not fix the ends, since the centroid still gives 1.06 at MX17; (ii) T2K uses the time of the maximum, not a centroid, so the estimator does not explain T2K's lack of the effect; (iii) the repo's estimator-independence result covers edge, CFD and matched filter, not a charge-weighted time, so a centroid time at MX17 is untested there.
- **(b) Lever arm.** Supported, and strongest. sigma/column is 0.07 at MX17 and 0.002-0.007 for the large TPCs. The relevant compression scales as D/(sqrt(z)·s), and the fractional weight of the ends scales as sigma/column. It explains ALICE, whose sigma_T/pad matches MX17's.
- **(c) Exists but renamed and calibrated.** Weak. Named effects (angular pad, angular wire, E×B, ALICE tan²alpha) are resolution terms. Resistive-anode "bias" (Dixit) is in-plane and from RC non-uniformity. A global scale effect would be absorbed in v_d, but the toy says that scale effect is about 0.1 %, so there is little to absorb. The µTPC literature's "capacitive coupling" naming (repo `LITERATURE_MIXING.md`) is the case where (c) holds, and those are short-gap detectors.
- **(d) Nobody checks.** Partly. Scale is checked to about 0.1 %. Column-resolved tan-lambda against an external tracker is not published in anything I read; at the toy's 0.1 % it would be invisible even if tried.
- **Net answer to "why does nobody else have this?"** Large pad TPCs have a similar sigma/pad but a track roughly 10-70x longer, relative to the mixing width,, sampled by a charge-weighted time. MX17 has a short column and an edge-type time, and its gap ends take a large share of the track. µTPCs (same geometry class) do see it, as `LITERATURE_MIXING.md` records.

### Gaps
- No primary-source statement of argument (b) exists in what I read; it is derived here.
- The toy excludes electronics and threshold (ALICE's own "below threshold" term shows these have a real tan²alpha dependence, on the resolution). I did not simulate pad-TPC waveforms.
- Whether any experiment measured a row-resolved time residual versus tan-lambda in internal notes: not accessible.
