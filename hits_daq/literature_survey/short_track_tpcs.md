# Low-energy and short-track gaseous TPCs: charge spread versus drift time in track-angle reconstruction

Survey date 2026-11-08. Quotes were checked against text extracted from the arXiv PDF unless marked otherwise; page numbers are PDF pages of the arXiv version. "(derived)" marks my own arithmetic from the inputs shown. "(snippet)" means a search-result summary only: the source was not read and the number should be checked before use.

MX17 reference values, from `RECONSTRUCTION_BASIS.md` and `papers/paper2/hits_daq/LITERATURE_MIXING.md`: 28 mm column, 0.78 mm strips, D_T 350-400 µm/√cm, σ_T/pitch 0.75-0.86 at the cathode, 60 ns sampling, 72 ns per strip at tanθ = 0.3 and v = 36 µm/ns; ladder compressed 20-30 % against M3 (det3 X 1.19, Y 1.37). σ_T(cathode)/projected column length is 0.07-0.08 at tanθ = 0.3 and 0.035-0.04 at tanθ = 0.6 (derived: 0.585-0.67 mm over 8.4 or 16.8 mm).

## Geometry: drift length, pitch, σ_T, σ_T/pitch, track length in channels, sampling

### Takeaway
These detectors span σ_T/pitch from about 0.5 (MIGDAL) to about 8 (GridPix), and tracks can be only a few channels long. Most have a σ_T-to-track-length ratio as bad as MX17's or worse, but transverse diffusion coefficients are often not stated in the papers, and no paper states the ratio of mixing width to projected track length.

### Cited Findings

| System | Drift | Readout pitch | Gas | D_T / σ_T | σ_T/pitch | Track length | Time sampling | Source |
|---|---|---|---|---|---|---|---|---|
| DRIFT-IId | 50 cm per side | 2 mm wires, grouped 8 lines = 16 mm | CS2-based negative ion | "diffusion at the thermal limit"; no number read | not computed | head-tail limited: "must hit at least six wires" (snippet) | not read | [Battat 2015, p.2-3](https://arxiv.org/pdf/1410.7821); [Burgos 2008](https://arxiv.org/pdf/0809.1831) |
| MIMAC | 25 cm (5 cm in a COMIMAC test) | 424.3 µm strips, 256 per axis | 50 % iC4H10 + 50 % CHF3, 30 mbar | not read | not computed | few mm at keV ("only a few mm long") | 20 ns; v = 11.5 µm/ns at 84 V/cm gives 0.23 mm per sample, v = 22.9 µm/ns at 150 V/cm gives 0.46 mm (derived) | [Beaufort 2024, p.2-3](https://arxiv.org/pdf/2312.12842); [Tao 2019, p.5](https://arxiv.org/pdf/1903.02159) |
| NEWAGE-0.3b' | volume 31 x 31 x 41 cm³ (drift axis not confirmed from text read) | 400 µm µ-PIC, 768 + 768 strips | CF4, 76 Torr | not stated in text read | not computed | below 50 keVee "less than 1 mm", so about 2.5 strips or fewer (derived) | 100 MHz (10 ns) TOT clock | [NEWAGE 2023, p.2-3, p.20](https://arxiv.org/pdf/2301.04779) |
| MIGDAL OTPC | 3 cm | 833 µm ITO strips (120); camera for x-y | CF4, 50 Torr | 260 µm/√cm minimum at 200 V/cm | 0.54 at full drift (derived: 450 µm/833 µm) | not read in this set | 2 ns; v ≈ 13 cm/µs gives 0.26 mm per sample (derived) | [MIGDAL design paper, p.9, sections 4.4, 5.7](https://arxiv.org/pdf/2207.08284) |
| CYGNO LIME | 50 cm (z dimension of the active volume) | sCMOS image plus 4 PMTs; no strip or pad | He:CF4 60:40, atmospheric | about 125 µm/√cm transverse, 105 µm/√cm longitudinal | not applicable | alpha and muon tracks | 250 MHz slow digitiser used for muons | [CYGNO/INITIUM PhD thesis, GSSI, p.75, p.160](https://arxiv.org/pdf/2509.10890) |
| D3-Micro | 9.2 mm (45 mm for neutrons) | 50 x 400 µm pixels (FE-I3-class chip) | ArCO2 and HeCO2 | diffusion measured, with a stated systematic of order 50 % | not computed | mm-scale alpha segments | 25 ns; at 22.8 µm/ns that is 0.57 mm per bin (derived) | [Vahsen 2014, p.2-3, p.6-7](https://arxiv.org/pdf/1407.7013) |
| BEAST TPC | 10 cm (15 cm) | 250 x 50 µm pixels (FE-I4B) | He:CO2 70:30 | Magboltz, number not extracted | not computed | alpha mean 2.1 cm | 25 ns; vd chosen as 10 µm/ns, so 250 µm per bin | [Jaegle 2019, p.2-4, p.15](https://arxiv.org/pdf/1901.06657) |
| ACTAR TPC | not read | 2 x 2 mm pads, 128 x 128 | iC4H10 100 mbar (commissioning) | not read | not computed | simulated resolution quoted at 11 to 56 pads | GET, not read | [Slides (Morfouace)](https://indico.in2p3.fr/event/16443/contributions/58969/attachments/46184/57630/AT_Morfouace.pdf) (snippet); [Tracking algorithms paper (arXiv:2011.07138), Table 1](https://arxiv.org/pdf/2011.07138) |
| AT-TPC | about 1 m class; Spyral example uses buckets 10 to 560 at 6.25 MHz (snippet) | pads, not read | various | not read | not computed | long tracks plus short recoils | GET | [Spyral docs](https://attpc.github.io/Spyral/user_guide/phases/point_cloud); [PyPI example](https://pypi.org/project/attpc_spyral/0.15.0) (snippet) |
| TexAT | not read | pads 1.67 x 3.42 mm central; side strips 1.67 mm wide | not read here | not read | not computed | not stated | GET, 1-100 MHz; 40 ns buckets in one run | [Koshchiy 2020, p.3, p.7](https://arxiv.org/pdf/1906.07845) |
| MAIKo (added: strip plus time, closest in layout) | 140 mm | 400 µm µ-PIC strips | He + 7 % iC4H10, 430 hPa | not read | not computed | beam-length tracks | 100 MHz TOT clock | [Furuno 2018, p.3-4](https://arxiv.org/pdf/1809.02287) |
| GridPix (Timepix3, Nikhef) | up to about 20 mm | 55 µm | Ar/CF4/iC4H10 95/3/2 | 306 µm/√cm fitted (from LITERATURE_MIXING.md) | 7.9 at 20 mm (from LITERATURE_MIXING.md) | beam across the chip | 1.56 ns ToA | [Ligtenberg 2018](https://arxiv.org/pdf/1808.04565) |
| GasPix Timepix3 (timewalk paper) | 13.5 mm | 55 µm | CO2/DME | 29.55 µm/√mm, about 93 µm/√cm (derived unit change) | about 2.0 at 13.5 mm (derived: 109 µm/55 µm) | beam at 15° tilt, 20° rotation | Timepix3 ToA | [Tsigaridas 2019, p.3, p.6](https://arxiv.org/pdf/1902.00480) |
| IXPE GPD | 1 cm absorption gap | 50 µm pitch hexagonal; 300 x 352 pixels | pure DME, 800 mbar | minimum about 68 µm/√cm | 1.4 at full 1 cm (derived: 68 µm/50 µm) | "about 100 µm at 2 keV to slightly over 1 mm at 8 keV" | no drift-time coordinate in the passages read | [Baldini 2021, p.3-5](https://arxiv.org/pdf/2107.05496) |
| Strip NITPC polarimeter (added: strip plus time) | 13 mm | 0.121 mm strips | Ne+CO2+CH3NO2 | not read | not computed | short photoelectron tracks | 2.5 MHz, rebinned so time bin matches strip pitch | [Prieskorn 2014](https://arxiv.org/pdf/1406.4393) |

- MIMAC states that strip and sample quantisation is "no more than 500 µm" for both width and depth — [Tao 2019, p.7](https://arxiv.org/pdf/1903.02159).
- BEAST chose the velocity to equalise quantisation: "25 ns·vd = 250 µm, so vd = 10 µm/ns" — [Jaegle 2019, p.4](https://arxiv.org/pdf/1901.06657).
- NEWAGE ties its threshold to track length against pitch: tracks below 50 keVee are "less than 1 mm", "comparable to the strip pitch of 0.4 mm" — [NEWAGE 2023, p.20](https://arxiv.org/pdf/2301.04779).
- Strip NITPC: "The bin size in the time dimension must be matched to the negative ion drift velocity so that square pixels" are produced — [Prieskorn 2014](https://arxiv.org/pdf/1406.4393).

### Inferences
- Lever arm (derived, order of magnitude): IXPE σ_T(1 cm)/track is about 0.7 at 2 keV and 0.07 at 8 keV, against MX17's 0.04-0.08. NEWAGE-class and MIMAC-class keV tracks span a few channels, so any σ_T of a channel or more equals a large fraction of the track. Whether the lever arm alone predicts a bias cannot be tested here because almost no source reports both σ_T and a bias.

### Gaps
- D_T for MIMAC, NEWAGE, DRIFT, BEAST, ACTAR, AT-TPC, TexAT, MAIKo was not found in the passages read. Do not fill these from memory; a Magboltz run would be needed.
- SpecMAT: only search summaries (2916 triangular pads, 241Am characterisation); no reconstruction text read.
- D3 detector papers after 2014 and the Ghrear/Vahsen recoil-identification paper were not located (the arXiv number I tried was a wrong paper).

## How the drift coordinate and track angle are derived

### Takeaway
Two families exist. Per-channel one-time methods are used by NEWAGE, BEAST, D3, MAIKo, TexAT, AT-TPC and MIGDAL's strip plane. Time-slice methods are used by MIMAC and, in part, ACTAR. Waveform-level work is deconvolution followed by peak finding (MIGDAL, AT-TPC) or a per-bucket deconvolved charge distribution (ACTAR); I found no joint forward fit of strip x time waveforms for track geometry. TOT is mostly a charge proxy, not the time estimator.

### Cited Findings

Per-channel, one time each:
- NEWAGE: "the relative arrival times of ionized electrons in the readout strips" are "converted into relative Z positions taking account the drift velocity", then "Angles of a track in the X-Z and Y-Z planes are known by fitting the hit-positions with straight lines." TOT is used for energy deposition and head-tail, via "large TOTs at the starting point" — [NEWAGE 2023, p.7](https://arxiv.org/pdf/2301.04779).
- BEAST: z is "the relative thresholdcrossing time of the hit" (hyphen lost in extraction) and TOT is "a measurement of integrated charge" — [Jaegle 2019, p.2](https://arxiv.org/pdf/1901.06657).
- D3-Micro: 25 ns arrival-time bins and ToT give a 3D image "By using the known drift velocity in the drift gap" — [Vahsen 2014, p.3](https://arxiv.org/pdf/1407.7013).
- MAIKo: "the time over threshold was recorded for each strip, but only the middle clock between the leading edge and the trailing edge was used in the analysis." This is the only estimator in this set that uses the TOT midpoint for time. The stated reason is not given in the passage read — [Furuno 2018, p.6](https://arxiv.org/pdf/1809.02287).
- TexAT: "the wave form fitting to determine waveform maximum amplitude values and peak times", then 3D points and track fits — [Koshchiy 2020, p.10](https://arxiv.org/pdf/1906.07845).
- AT-TPC / Spyral: peak finding on each trace with SciPy, each signal becomes a point at the pad position. Z is linear between two anchors: z = (t_window - t_signal)/(t_window - t_micromegas) x l_attpc. The page does not mention waveform fitting, deconvolution, diffusion or angle bias — [Spyral docs](https://attpc.github.io/Spyral/user_guide/phases/point_cloud). CNN and conventional pipelines do baseline correction, deconvolution, then peak detection, with the CNN giving "the same results as the traditional deconvolution" — [CNN paper](https://arxiv.org/pdf/2202.12941).
- MIGDAL strips: "The time of the peak amplitude of the deconvolved ITO signal in a given strip is converted to a relative z position using the drift velocity." It "gives an approximation for the center of the charge distribution in z along a given strip" — [Tilly and Handley 2023, p.4](https://arxiv.org/pdf/2307.10477). The design paper plans two 2D deconvolutions, first electronics response and cross-talk, then "the mean response of a single electron in the induction gap" — [MIGDAL design, p.20](https://arxiv.org/pdf/2207.08284). Series resistance of ITO strips and strip coupling are modelled in the simulation with Cadence Spectre: "a nonnegligible amount of coupling between strips" — same page.

Time-slice or voxel:
- MIMAC 19F: the anode sampled "every 20 ns allows the reconstruction of a 3D cloud of primary electrons"; track depth is (t_max - t_min) x V_drift — [Tao 2019, p.5](https://arxiv.org/pdf/1903.02159). A barycentre per time slice (BCW) is the preferred fit: "The average reconstructed angle using BCW method is closest to the initial 0◦ angle" — [Tao 2020, p.7](https://arxiv.org/pdf/2003.11812).
- MIMAC keV recoils ("high gain 3D method") abandon the per-slice fit: the X-Z angle is estimated with tan θ_X = (ΔX_g/ΔZ_g)(D_L/D_T), with ΔZ_g from the deconvolved Flash-ADC and ΔX_g from the anode width times a corrective factor. The paper says a deconvolution of the strip signals is not possible because strips are binary, have thresholds, and have a non-linear weighting field — [Beaufort 2024, p.6-7](https://arxiv.org/pdf/2312.12842). This is a ratio of cloud widths, with diffusion entering explicitly.
- ACTAR: slides describe "full 3D charge reconstruction with response function de-convolution" and an "effective Z charge distribution" — [Giovinazzo slides, p.19](https://indico.in2p3.fr/event/18026/contributions/68569/attachments/51952/66963/2019-01_ACTAR-TPC_J-Giovinazzo_ENSAR-GDSpub.pdf). The journal abstract: "An effective full 3D charge distribution reconstruction method" for trajectories in any direction — [NIM A 953 record](https://oskar-bordeaux.fr/handle/20.500.12278/136910) (snippet).

Other:
- CYGNO: PMT time-over-threshold gives the track extent ΔZ (thesis uses TOT20 averaged over four PMTs), and the track sense comes from the Bragg peak position — [CYGNO/INITIUM PhD thesis, GSSI, p.188](https://arxiv.org/pdf/2509.10890). The whole-track PMT time profile has no strip or pixel division, so no one-time-per-channel step exists.
- DRIFT: head-tail from "an asymmetry in the time distribution of ionization arrival", computed from the voltage integral over the first and second half of the time profile — [Burgos 2008, p.3](https://arxiv.org/pdf/0809.1831).
- GridPix: z from each single-electron hit's ToA and a predicted velocity — [Ligtenberg 2018, p.3](https://arxiv.org/pdf/1808.04565).

Specific MIMAC and NEWAGE TOT question:
- NEWAGE: TOT is a charge and head-tail observable. I found no bias study of TOT against an angle. Head-tail power is near chance: averaged 52.4 ± 1.1 %, 52.9 ± 1.4 %, 53.6 ± 2.0 % for 50-100, 100-200, 200-400 keVee, where "Pht = 50% indicates that the detector has no sensitivity" — [NEWAGE 2023, p.10](https://arxiv.org/pdf/2301.04779).
- An earlier NEWAGE proceedings warns on TOT: "TOT is a good parameter to estimate" collected charge "as long as the longitudinal diffusion is small", and with long drift "the longitudinal diffusion cannot be ignored and TOT no longer helps" — [NEWAGE proceedings](https://arxiv.org/pdf/1109.3099). This concerns charge, not angle.
- MIMAC strips are binary per sample (fired or not), so TOT per strip is not the angle estimator. Its Z calibration issue is space charge, covered under the bias question.

Calibration of the z scale:
- AT-TPC uses external anchors: window and micromegas buckets from long tracks, or a measured drift velocity — [Spyral docs](https://attpc.github.io/Spyral/user_guide/phases/point_cloud).
- MAIKo: calculated drift velocities "agreed with the measured drift velocities within 5%" — [Furuno 2018, p.6](https://arxiv.org/pdf/1809.02287).
- MIMAC: Magboltz velocity; earlier measurement in pure CF4 found "a 12% deviation compared to the MAGBOLTZ simulation" — [Tao 2019, p.5-6](https://arxiv.org/pdf/1903.02159).
- GasPix: "the assumption of a constant drift velocity for the z-position reconstruction is not useful", so z(t_d) is calibrated with the telescope and a Chebyshev polynomial — [Tsigaridas 2019, p.5](https://arxiv.org/pdf/1902.00480).

### Inferences
- Time-slice methods are not free of mixing, they move it. A per-time-slice barycentre is largely insensitive to transverse diffusion in the track-axis position, because the transverse spread averages out, but it inherits longitudinal diffusion and shaping along z, and its ends are one-sided (see next section).
- MIMAC's D_L/D_T formula is the only reconstruction here in which diffusion is an explicit term in the angle.

### Gaps
- I could not read the full text of Giovinazzo 2020 (HAL returned HTML), Bradt 2017 (AT-TPC) or Mauss 2019 (ACTAR). It is not established whether ACTAR's published angles use the 3D deconvolved distribution or peak times.
- MSU thesis download failed, so the older AT-TPC fit method is known only from conference abstracts.
- Spyral's page does not state whether one pad can yield several points.

## Reported angle-dependent or depth-mixing bias, external validation, and IXPE

### Takeaway
A depth-mixing artefact is named explicitly in only two places: BEAST's "canoe shape" and MIMAC's biased first and last time slices. MIMAC reports the only quantified angle-dependent bias with external kinematic truth, but its z scale error has the opposite sign to MX17 and is attributed to space charge. No paper in this survey reports a ladder compression or tan ratio. IXPE's spurious modulation is a different mechanism, traced to GEM structure, though its null-source calibration is a useful methodological analogue.

### Cited Findings

Depth-mixing shapes:
- BEAST: "Because the pixel integration time is longer than the typical charge cloud size, only one entry per horizontal position is measured, leading to the canoe shape of the reconstructed tracks." This is in a figure caption, unquantified and not followed up in terms of angle — [Jaegle 2019, p.3](https://arxiv.org/pdf/1901.06657).
- MIMAC: "initial and final time slices of the track usually have a larger than average deviation from the track direction. This is because the anode samples the endpoints of the transversely diffused primary electron cloud. Removing the first and last time slice does not produce better results either." — [Tao 2020, p.7](https://arxiv.org/pdf/2003.11812). Same end-of-column localisation as MX17's end-bin finding, in a time-slice estimator. Size not given.
- MIMAC, on the role of diffusion: "the diffusion dominates the measurements" and "in fact the 3D diffusion is the physical mechanism giving access to the details of the physical track" — [Tao 2019, p.7](https://arxiv.org/pdf/1903.02159).

MIMAC z elongation (opposite sign to MX17):
- Measured 19F track depths exceed simulation, even with diffusion: "the simulated depths are much shorter than the measurements" — [Tao 2019, p.7](https://arxiv.org/pdf/1903.02159).
- Cause assigned to ion backflow in the 512 µm gap, "leading to a decrease of effective drift velocity, and thus we measure more time slices than predicted by simulations"; tested by 5 keV electrons while varying gain, and the "asymmetric factor" from the Flash-ADC is a correction used "as a first approximation" — [Tao 2019, p.8-9](https://arxiv.org/pdf/1903.02159).
- Correction is z2 = η·z1, so θ2 = arctan((1/η) tan θ1) with θ2 > θ1 for η in (0, 1] — [Tao 2020, p.3-4](https://arxiv.org/pdf/2003.11812). Uncorrected tracks are stretched in z, so tan(from drift) is too small. MX17 hit times have a compressed z, so tan is too large.
- Ions also produce a signal tail: "a signal is still detectable about 300 ns after the avalanche produced by the last primary electron" — [Beaufort 2024, p.6](https://arxiv.org/pdf/2312.12842).

MIMAC, external angle truth:
- 19F ions along the drift axis: mean reconstructed angle "small (< 1.6◦ )", spread about 10 times larger than the shift (more than 4 times at 26.3 keV), resolution "better than 15◦ down to a kinetic energy of 9.3 keV" — [Tao 2020, p.5-7](https://arxiv.org/pdf/2003.11812). The paper notes the beam was only at 0°: "not possible to inject ions at angles different from zero".
- Mono-energetic neutrons, θ_exp from kinematics. Difference from reconstruction centred near 0°. Resolution 10° (proton, 7-20 keV), 15° (carbon) at 27 keV; 16° for protons 4-8 keV. Bias: "the high gain 3D method suffers from a bias when reconstructing" carbon recoils at low angle (below 15°), and "the deviation from the expected value, in particular for θ > 45◦ , can indicate a bias in the angle reconstruction" — [Beaufort 2024, p.9-12](https://arxiv.org/pdf/2312.12842). Note θ_exp uses ionization energy and quenching factor, so it is only partly independent.
- Comparison of two methods: offset 1.5°, dispersion σ = 14° — [Beaufort 2024, p.9](https://arxiv.org/pdf/2312.12842).

Other angle checks:
- NEWAGE: σ_ang = 58.1 (+5.8/−2.8)° at 50-100 keVee, from fitting |cos γ| of neutron data against smeared simulation. Simulation-smeared, so a scale bias would be partly absorbed — [NEWAGE 2023, p.13-15](https://arxiv.org/pdf/2301.04779).
- BEAST: alpha φ peak Gaussian width 0.9° for all three sources over 2.1 cm tracks, scaled to about 2.5° at 1 cm: "our angle measurement precision is probably significantly better than 1 degree" — [Jaegle 2019, p.15](https://arxiv.org/pdf/1901.06657). This is φ; the tracks are near horizontal, so the z scale is barely used.
- D3-Micro: 3D-track pointing with an analysis-defined collimation gives σ_φ = 0.95 ± 0.04°, σ_θ = 2.2 ± 0.1°, upper limits. "Since, however, a TPC cannot measure absolute position in the drift direction (z), we cannot know the true dz , and hence cannot select on it to collimate the source in theta." — [Vahsen 2014, p.7](https://arxiv.org/pdf/1407.7013). Fit-uncertainty method gives σ_φ = 1.068 ± 0.003°, σ_θ = 1.202 ± 0.004°.
- MAIKo: resolution 1.3° for scattered 4He, from upstream-downstream halves of the same track: "θup should agree with θdown within the angular straggling" — [Furuno 2018, p.1, p.6](https://arxiv.org/pdf/1809.02287). A self-consistency test, blind to a common scale error.
- ACTAR (simulation): FWHM 2.09(4)° at range 11 pads, 0.93° at 20, 0.53° at 28, 0.40° at 37, 0.29° at 46, 0.26° at 56; "The angle reconstruction depends mostly in the granularity of the detector as well as the length of the tracks." — [tracking paper, p.9-10, Table 1](https://arxiv.org/pdf/2011.07138). Simulated, so it says nothing on mixing unless the simulation includes it. An alpha-source plot is shown without a known-angle bias figure.
- TexAT: "angular resolution determined from the simulation was approximately 3◦ at FWHM" — [Koshchiy 2020, p.10](https://arxiv.org/pdf/1906.07845).
- CYGNO LIME: cosmics triggered by two scintillator bars give "tracks with predefined and known inclination", zenith ψ = arctan(ΔZ/h_LIME) with ΔZ from PMT TOT. The check is that ψ falls in the geometrically accepted range; the thesis calls TOT "a reliable tool". No precision scale factor quoted — [CYGNO/INITIUM PhD thesis, GSSI, p.157-165](https://arxiv.org/pdf/2509.10890). It also warns that noise-RMS TOT is biased "towards larger values" for alphas, hence a 20 %-level TOT — p.187-188.
- GridPix with a telescope: mean residuals across the fiducial area have r.m.s. "8 µm in the pixel plane and 31 µm (0.4 ns) in the drift direction" — [Ligtenberg et al., Timepix3 TPC readout, p.5](https://arxiv.org/pdf/1902.01987). Beam perpendicular to the drift, so this tests offsets at fixed z, not a slope along z.
- DRIFT: no angle-bias number read. Head-tail "limited by the 2 mm readout pitch" — [review, p.13](https://arxiv.org/pdf/0905.3675).

IXPE and GPD:
- Spurious modulation: requirement below 0.5 % at 5.9 keV; "little or no sign of systematic effects at 5.9 keV", but significant at lower energy, position dependent, "circumstantially traced back to the microscopic structure of the GEM holes" — [Baldini 2021, p.9-11](https://arxiv.org/pdf/2107.05496). Gain non-uniformity is excluded and "a stretch of the effective pitch of the GEM holes in one of the two orthogonal directions as small as ∼ 1% could explain the observed modulation" — p.13.
- Residual after electronics work: GEM-dominated "about 1-2% at 2 keV"; needing 0.15 % uncertainty consumed about 60 % of calibration time, about 40 days per detector — [IXPE lessons paper, p.4](https://arxiv.org/pdf/2601.00764) (hyphen form normalised).
- Correction is empirical, not modelled: unpolarised sources at two orthogonal rotations, per-event subtraction of Stokes maps by position and energy — [Rankin 2022](https://arxiv.org/pdf/2111.14867). Baldini: "hard to model from first principles".
- Closest estimator-bias analogue, polarization leakage: errors in the reconstructed absorption point, "the impact point is typically misplaced along the photo-electron track"; there is a correlation "between the offset of the absorption point from its true position and the inferred polarization direction" — [Bucciantini 2023, p.3-4](https://arxiv.org/pdf/2302.00346). A charge-spread error along the track, but in-plane.
- A GPD lab study reports residual modulation 0.30 ± 0.15 % at 6 keV and a modulation factor consistent with GEANT4/Garfield including diffusion — [Li, Feng et al. 2015](https://arxiv.org/pdf/1509.05595). Not confirmed as the PolarLight unit.

### Inferences
- IXPE's problem does not resemble MX17's mechanism: the GPD has no drift-time ladder in the sources read, and the spurious signal comes from GEM geometry. It does resemble MX17's position in one respect: a 1 % effect was found only because a null source (unpolarised) and a known source existed.
- MIMAC is a warning against assuming a single sign. In a high-gain large-gap Micromegas with slow ions, z can be stretched, offsetting compression from mixing. Its large-gap regime differs from MX17 and its measured excess cannot be assigned to one cause.

### Gaps
- No paper here gives a tan(reco)/tan(true) ratio against an external slope, in the manner of det3 vs M3.
- CYGNO's numerical comparison of inferred ψ with acceptance was not extracted.
- PolarLight-specific spurious-modulation results not found.
- DMTPC and later D3 head-tail publications were not read.

## Per-pixel readout below diffusion: is the mixing removed, and does GridPix remove it?

### Takeaway
Pixelation alone does not remove one-time-per-channel mixing: BEAST (pixels 50 x 250 µm, one threshold time per pixel) shows it directly as the canoe shape. Single-electron GridPix removes the aggregation by construction, and telescope tests show small mean residuals, but I found no direct published test of drift-slope scale against an external angle in the way MX17 needs.

### Cited Findings
- FE-I3/FE-I4 charge-integrating pixels give one crossing time and one ToT per pixel; BEAST's caption attributes the canoe shape to that — [Jaegle 2019, p.3](https://arxiv.org/pdf/1901.06657).
- GridPix: "detects single ionization electrons with high efficiency"; "Diffusion is found to be the dominating error on the track position measurement both in the pixel plane and in the drift direction"; "A search for double hits did not yield any indication for crosstalk" — [Ligtenberg 2018, p.1, p.6](https://arxiv.org/pdf/1808.04565).
- Its only explicit time bias is time walk, corrected via ToT, with z from the velocity "found to be consistent, but because of systematic uncertainties there was no attempt at a precise determination" — [Ligtenberg 2018, p.3](https://arxiv.org/pdf/1808.04565).
- GasPix on a telescope, 13.5 mm gap, at 15° tilt and 20° rotation: time walk corrected with ToT, z calibrated against telescope tracks — [Tsigaridas 2019, p.3-5](https://arxiv.org/pdf/1902.00480). This is the nearest external calibration of the drift coordinate on a tilted pixel TPC. Its calibration absorbs any compression, so it does not report one.
- GridPix quad paper "systematic error given as the r.m.s. is 8 µm in the pixel plane and 31 µm (0.4 ns) in the drift direction" — [p.5](https://arxiv.org/pdf/1902.01987).
- Timepix OneHit mode: first threshold crossing sets the counter and later hits are ignored — [EUDET memo](https://www.eudet.org/e26/e28/e615/e851/eudet-memo-2008-40.pdf) (snippet).

### Inferences
- With single electrons each hit carries its own depth and diffusion only smears it, so the aggregate-time mechanism is absent. A pixel receiving more than one electron within its dead time would keep only the first, so a small early-time selection is conceivable. The papers read do not quantify this.
- Fine pixels with multi-electron charge (GEM on a pixel chip) keep the MX17 mechanism and add earliest-arrival selection: the pixel time is a threshold crossing of an aggregate of charge.

### Gaps
- No published angle-slope test of GridPix in tilted geometry against a telescope with a stated scale result was found.
- No study on the fraction of GridPix pixels hit by more than one electron at the cathode end of the drift.

## Hypothesis tests

### Takeaway
(a) is supported in structure but untested: no paper compares the two axes for bias. (b) cannot be tested from the literature. (c) holds only partly: canoe shape and slice-end deviations are the same family, while the other named effects are different physics. (d) is strongly supported; external-reference checks are rare and coarse.

### Cited Findings
**(a) Time-slice vs per-channel axis.**
- Per-channel: NEWAGE, BEAST, D3, MAIKo, TexAT, AT-TPC, MIGDAL strips (see section 2). Time-slice: MIMAC BCW; ACTAR deconvolved charge.
- For: MIMAC's slice barycentre has its visible artefact only at the ends — [Tao 2020, p.7](https://arxiv.org/pdf/2003.11812).
- Contrary: MIMAC moved away from per-slice fits at keV energies since strips are binary and distorted by the ion tail, replacing them with a width ratio — [Beaufort 2024, p.6-7](https://arxiv.org/pdf/2312.12842). So time slicing is not a remedy at short track.

**(b) Lever arm.** No source states mixing width over projected column length, or its effect on bias. ACTAR's resolution-vs-range table — [p.10](https://arxiv.org/pdf/2011.07138) — and D3's angular-resolution scaling are resolution trends, not bias.

**(c) Same effect under another name.**
- Same family: "canoe shape" ([Jaegle 2019, p.3](https://arxiv.org/pdf/1901.06657)); slice-end deviation ([Tao 2020, p.7](https://arxiv.org/pdf/2003.11812)).
- Different physics, sometimes corrected like a scale: MIMAC "high-gain systematic effect" (space charge); GridPix time walk; IXPE spurious modulation (GEM); polarization leakage (absorption point).
- MIGDAL treats strip coupling by deconvolution — [design paper, p.20](https://arxiv.org/pdf/2207.08284) — addressing MX17's footprint and coupling part, not mixing across depths.

**(d) Few check against an external reference at this precision.**
- External truth exists for MIMAC (0° ion beam, neutron kinematics) at 10-16° resolution; GridPix telescope for offsets; CYGNO scintillator acceptance. Self-consistent or simulated checks: MAIKo halves, D3 and BEAST source pointing, ACTAR and TexAT simulations, NEWAGE smeared simulation.
- D3 states the limit explicitly: no absolute z, so no θ collimation — [Vahsen 2014, p.7](https://arxiv.org/pdf/1407.7013).

Contrary evidence to the idea that all short-track TPCs must show compression:
- MIMAC's measured z scale error is an elongation, and the mean at 0° is under 1.6° — [Tao 2019](https://arxiv.org/pdf/1903.02159); [Tao 2020](https://arxiv.org/pdf/2003.11812).
- MIMAC neutron results are centred near 0° difference across most of the angular range — [Beaufort 2024, p.9](https://arxiv.org/pdf/2312.12842). At 10-16° resolution, a 1.2-1.4 tan ratio at moderate tanθ (about 3-4°) would be hard to exclude from the means shown without the bin values.

### Inferences
- An MX17-sized bias (about 4° at tanθ = 0.3) is below the resolution of MIMAC and far below NEWAGE (58°). The same bias in BEAST or D3 would be visible in resolution terms, but these tests use near-horizontal tracks and no absolute z, so they could not see it. Absence of report is therefore not evidence of absence.
- MAIKo is the most similar layout in this set (strips, single time per strip, tracks crossing many strips), and its method-independent check is a self-consistency test that a scale error passes. MAIKo's TOT-midpoint time is a different estimator from a leading edge and could be a cheap comparison point for MX17's existing data.

### Gaps
- No time-slice-versus-per-channel comparison on the same data.
- No bias-versus-D_T study in any of these detectors.
- A negative search result is not proof of absence: Bradt, Mauss, Giovinazzo 2020 and later D3 and SpecMAT papers remain unread.
