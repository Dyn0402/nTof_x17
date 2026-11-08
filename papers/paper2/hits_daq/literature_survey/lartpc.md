# LArTPC and wire-readout TPCs: convolution of charge spread with the time signal, as the counter-example to "nobody else has this"

Conventions used in these notes.
- Page references are arXiv PDF pages (the printed page is usually 1-2 lower). Verbatim quotes were string-matched against `pdftotext` output of the PDF, except where marked.
- **(derived)** means my arithmetic from the inputs shown; it is not in any source.
- Angles: LArTPC θxz is measured from the wire plane, so θxz = 0 is isochronous and 90° is along the drift. Our tanθ = dx/dz is measured from the drift axis, so **our tanθ = cot θxz**. Our tanθ 0.3-0.6 corresponds to θxz ≈ 73°-59° **(derived)**, which LArTPC papers call the large-angle regime.

## Headline for Paper II (verdict on hypothesis c)

- **Partly a counter-example.** LArTPC already published, and must be credited for: (1) waveform = charge convolved with a kernel in channel and time, solved jointly over channels (2018); (2) charge on neighbouring wires, from a different drift position, being "stretched in time" for inclined tracks; (3) transverse diffusion adding to the time width of a single-wire pulse, growing with θxz, with a stated mechanism (2021); (4) a per-channel regularised minimisation on the waveform, for one special case.
- **Not found in anything I read:** a reported bias of per-wire hit *time* (a compressed time ladder or too-steep angle), any measurement of it against an external reference, or a per-event fit of a depth-binned charge profile on a track. This is a negative search over the papers listed below, not a proof of absence.
- **Consequence.** "Nobody models the signal as a convolution in channel and time" is false. "Nobody has shown that same-time mixing biases the track slope from one time per channel, and corrected it against an external telescope" survives this survey. So does the depth-profile-in-the-fit claim that `RECONSTRUCTION_BASIS.md` already makes.

## Geometry: pitch, drift, diffusion, sampling, shaping, field-response extent

### Takeaway
At full drift, σ_T/pitch in LArTPCs (about 0.15-0.8, derived) is comparable to ours (up to about 1). The large difference is the lever arm: the mixing width is about 0.2-0.6 % of a full projected column in LAr against about 10 % for us.

### Cited Findings

| Detector | Wire pitch | Max drift | v_d | Sampling | Shaping | Field-response extent |
|---|---|---|---|---|---|---|
| MicroBooNE | 3 mm, plane gap 3 mm ([1802.08709](https://arxiv.org/abs/1802.08709) §1) | 2.56 m, 273 V/cm; max 2.3 ms ([2104.06551](https://arxiv.org/abs/2104.06551)) | about 1.10 mm/µs | 0.5 µs | 2 µs peaking | computed for ±10 wires; "negligible beyond ±4 wires" on V, Y; U "still sizable at ±4 wires" and contributions as far as 10 wires away ([1802.08709](https://arxiv.org/abs/1802.08709) §2, §3.2.1, pp. 14, 25) |
| ProtoDUNE-SP | 4.79 mm (G, X), 4.67 mm (U, V) | 3572 mm, 500 V/cm, 2.25 ms | 1.59 mm/µs | 2 MHz | 2 µs peaking | 21 wires, grid of 1/10 pitch; all from [2007.06722](https://arxiv.org/abs/2007.06722) Table 1, §2-4 |
| ICARUS | 3 mm ([1210.5089](https://arxiv.org/abs/1210.5089); [2407.11925](https://arxiv.org/abs/2407.11925)) | 1.5 m active width per TPC, about 1 ms ([2407.11925](https://arxiv.org/abs/2407.11925)) | 1.59 mm/µs ([1210.5089](https://arxiv.org/abs/1210.5089)) | 2.5 MHz, 0.64 mm per sample ([1210.5089](https://arxiv.org/abs/1210.5089)) | nominal shaping 1.3 µs, plus a measured tail of about 50 µs ([2407.11925](https://arxiv.org/abs/2407.11925)) | 2D Garfield field response; ±5 wires used in the waveform alignment metric ([2407.11925](https://arxiv.org/abs/2407.11925) §4.1) |
| ArgoNeuT | 4 mm, plane gap 4 mm | td = 295 µs | 1.59 mm/µs | 198 ns (5.05 MHz) | τ_int ≈ 1 µs, peaking ≈ 2τ_int; all from [1205.6747](https://arxiv.org/abs/1205.6747) | not stated there |
| SBND | 3 mm ([2002.08424](https://arxiv.org/abs/2002.08424), via abstract-level search) | APAs each 2 m from a central cathode, 500 V/cm ([2002.08424](https://arxiv.org/abs/2002.08424) §1) | not stated in text read | not stated in text read | not stated in text read | not stated in text read |

Diffusion:
- σ_T = √(2 D_T t). "The transverse diffusion constant has never been directly measured in LAr at the electric fields typical in existing LArTPC neutrino detectors" ([Putnam & Schmitz, 2205.06745, p. 3](https://arxiv.org/abs/2205.06745)). Their predictions: 12.0 (9.30) cm²/s and σ_T = 1.55 (1.36) mm·√(t/1 ms) ([2205.06745](https://arxiv.org/abs/2205.06745)).
- ICARUS: D_T estimated at 5-12 cm²/s; at the maximum drift of about 1 ms, σ_T ≈ 1.0-1.5 mm, "on the order of the wire pitch" ([2407.11925, p. 7](https://arxiv.org/abs/2407.11925)).
- MicroBooNE longitudinal: D_L = 3.74 cm²/s (collection plane) at 273.9 V/cm, against a simulation default of 6.40; from that, a predicted D_T = 5.85 +0.62/−0.33 cm²/s ([2104.06551](https://arxiv.org/abs/2104.06551) §4, §5.1).
- Older D_L values: ICARUS 4.8 ± 0.2 cm²/s (0.1-0.35 kV/cm, 42 cm drift); BNL (Li et al.) systematically higher, 5-60 mm drift ([Yang review, 2012.01319](https://arxiv.org/abs/2012.01319) §3.4; [Li et al., 1508.07059](https://arxiv.org/abs/1508.07059)).
- "The effects of diffusion are generally small": longitudinal rms ∼1.5 mm over 2.5 m, i.e. 1 tick ([Baller, 1703.04024, p. 8](https://arxiv.org/abs/1703.04024)).

Derived comparison **(all columns derived; D_T range 5-12 cm²/s from ICARUS, ignoring field dependence; tanθ = 0.3 from the drift axis, i.e. θxz = 73.3°)**:

| | σ_T at max drift [mm] | σ_T/pitch | Projected column L·tanθ [mm] | **Mixing/column = σ_T/(L tanθ)** | Drift time per wire at tanθ=0.3 | Mixing in time σ_T/(tanθ·v) |
|---|---|---|---|---|---|---|
| MX17 (28 mm, 0.78 mm, v = 36 µm/ns, σ_T ≈ 0.8 mm from the brief) | 0.8 | 1.0 | 8.4 (10.8 strips) | **9.5 %** | 72 ns | 74 ns |
| MicroBooNE | 1.52-2.35 | 0.51-0.78 | 768 | 0.20-0.31 % | 9.1 µs (18 ticks) | 4.6-7.1 µs |
| ProtoDUNE-SP | 1.50-2.32 | 0.31-0.48 | 1072 | 0.14-0.22 % | 10 µs (20 ticks) | 3.1-4.9 µs |
| ICARUS | 0.97-1.50 | 0.32-0.50 | 450 | 0.22-0.33 % | 6.3 µs (16 ticks) | 2.0-3.1 µs |
| SBND (v assumed 1.59 mm/µs from the same 500 V/cm) | 1.12-1.74 | 0.37-0.58 | 600 | 0.19-0.29 % | 6.3 µs | 2.3-3.6 µs |
| ArgoNeuT (L = 469 mm from 295 µs × 1.59) | 0.54-0.84 | 0.14-0.21 | 141 | 0.38-0.60 % | 8.4 µs (42 ticks) | 1.1-1.8 µs |

### Inferences
- σ_T/pitch alone does not separate us from LAr. MicroBooNE at full drift sits at 0.5-0.8, against our ≤1. This mirrors the reviewer note on ALICE in `LITERATURE_MIXING.md`.
- Mixing/column separates us by about 30-70× for full-column tracks.
- **Hypothesis (b), lever arm: supported as arithmetic, but it is not a universal protection.** The table assumes a full-drift track. A 3 cm track segment at full drift in MicroBooNE gives σ_T/(L tanθ) ≈ 1.5-2.4 mm / 9 mm = 17-26 % **(derived)**, larger than ours. The lever-arm argument protects long tracks; it does not protect short ones (protons, vertex regions).

### Gaps
- No primary source read gives SBND sampling or shaping. A published SBND signal-processing paper was not found; only an [APS abstract](https://archive.aps.org/apr/2024/c07/6) and talks surfaced.
- ProtoDUNE-SP and ArgoNeuT papers read do not quote a D_T.

## Is the MicroBooNE 2D deconvolution the same mathematical problem as our forward model?

### Takeaway
Same linear model class: measured waveform = charge convolved with a kernel in channel and time. Different solver, unknowns, kernel physics and use of the output. It is a true precedent for "convolution in channel and time", and not a precedent for a per-event track-constrained depth fit.

### Cited Findings
- "the raw digitized TPC signal is a convolution of the arriving ionization electron distribution, the field response describing the induced current on wires due to moving charge, and the overall electronics response" ([Adams et al., 1802.08709 §3.1, p. 17](https://arxiv.org/abs/1802.08709)).
- Eq. 3.5: M_i(t0) = ∫(… + R1(t0−t)·S_{i−1}(t) + R0(t0−t)·S_i(t) + R1(t0−t)·S_{i+1}(t) + …)dt. Fourier-transformed into a matrix equation; "a two-dimensional (2D) deconvolution involving both time and wire dimensions is performed to recover the ionization electron distribution" ([1802.08709 §3.1.2, pp. 21-22](https://arxiv.org/abs/1802.08709)).
- Assumptions, verbatim: "Ignoring the variation of the strength of the field response within one wire region" and "Equation (3.5) assumes translational invariance in the response function" ([1802.08709, p. 21](https://arxiv.org/abs/1802.08709)). R is symmetric and Toeplitz, and time-invariance R(t,t′) ≡ R(t′−t) is assumed before the FFT (§3.1.1-3.1.2).
- Regularisation: Wiener-inspired filters in time and a Gaussian filter in wire number, F(ω_w) = exp(−½(ω_w/a)²) (§3.2.2, Eq. 3.13). "the deconvolution replaces the real field and electronics response function with an effective software filter response function" (§3.1.1) ([1802.08709](https://arxiv.org/abs/1802.08709)).
- Kernel contents: field plus electronics response. "For these single-electron simulations, diffusion is omitted" (§2) ([1802.08709](https://arxiv.org/abs/1802.08709)).
- Output retains diffusion: time smearing "about 2.7 ticks and 2.3 ticks for induction and collection planes", and "A one meter drift span provides a longitudinal/time smearing of about 2 ticks" ([1802.08709 §4.3.1, p. 44](https://arxiv.org/abs/1802.08709)). MicroBooNE extracts D_L from deconvolved pulse widths ([2104.06551](https://arxiv.org/abs/2104.06551)).
- ProtoDUNE-SP uses the same form: M(w_i1, t_j1) = ∫∫ Q · R(w_i2−w_i1, t_j1−t_j2) (Eq. 4.5); "Because of the long-range induction effect, instead of a 1D deconvolution involving only the time dimension, a two-dimensional (2D) deconvolution involving both the time and wire dimensions is performed" ([2007.06722 §4.4, p. 42](https://arxiv.org/abs/2007.06722)).
- 1D lineage: ArgoNeuT and MicroBooNE: "Wire Signal = Ionization ⊗ Recombination ⊗ Diffusion and Attachment ⊗ Field Response ⊗ … Electronics Response", Fourier-deconvolved to Gaussian-like signals ([Baller, 1703.04024 §3](https://arxiv.org/abs/1703.04024); the ⊗ symbols were lost in text extraction, so this is not a verbatim quote).
- **Closest per-event fit in LAr: L1SP.** On V wires through the shorted-Y region, M = R1·S1 + R2·S2 is solved by minimising χ² = (M − R1·S1 − R2·S2)² + λ·Σ(|S1i| + |S2i|) with coordinate descent, "limited to a 1D deconvolution instead of the 2D deconvolution that is used elsewhere", and only inside ROIs ([Adams et al. Part II, 1804.02583 §3.4, pp. 29-30](https://arxiv.org/abs/1804.02583)).
- Wire-Cell 3D imaging: compressed-sensing charge solving per time slice across wire planes, slice "typically ∼2 µs" ([Qian et al., 1803.04850](https://arxiv.org/abs/1803.04850)).
- A related forward-model direction: a differentiable LArTPC simulator for gradient-based *calibration* of detector parameters, not per-event reconstruction ([Gasiorowski et al., 2309.04639](https://arxiv.org/abs/2309.04639)).

| | LArTPC 2D deconvolution | MX17 forward model |
|---|---|---|
| Linear model | M(wire, t) = S ⊗ R(Δwire, Δt) | same class |
| Unknowns | free S per wire and tick | charge per drift-depth bin along the event's track (brief) |
| Kernel source | Garfield field ⊗ electronics, position-averaged in a pitch; 2D, averaged along the wire | per-detector calibrated; diffusion, footprint, RC |
| Is diffusion in the kernel? | **No**, it stays in the deconvolved image | **Yes**, it is part of what is fitted through |
| Stationarity | time- and translation-invariant, which is what allows FFT | diffusion grows with depth, so not stationary in absolute time **(inference from the model description)** |
| Solver | linear, Fourier, Wiener-type filter, ROI | per-event constrained fit |
| Physical spread | induced current from charge not collected (bipolar, cancels) | charge physically on several strips: diffusion, footprint, RC |

### Inferences
- The inverse problem is the same type: a block-Toeplitz linear system in channel × time. A referee from the LAr community would see Adams et al. as prior art for the model equation. Cite it.
- What differs is what the mixing *is*. Adams removes the induced long-range response but leaves diffusion. In our chain the diffusion mixing is the main target, so 2D deconvolution as published would not address it. This is my reading of the kernel definition and the surviving width, not a statement in the source.
- Resistive RC spreading has no LArTPC counterpart.
- Per-event constrained fitting exists in LAr only in a restricted form (L1SP: 1D, special region, with a sparsity prior, no track geometry).

### Gaps
- I did not read Wire-Cell code or internal notes. I could not establish whether any undocumented analysis fits track-constrained charge profiles.

## Are hit times biased as a function of track angle? External reference?

### Takeaway
LAr papers quantify charge, width and efficiency against θxz in detail. I found no reported hit-time or slope bias against θxz, with or without an external reference. Their references constrain t0/x and shape, not local slope.

### Cited Findings
- θxz is the governing variable: "θ xz alone determines the shape of the TPC signal" ([1802.08709, p. 16](https://arxiv.org/abs/1802.08709)); ICARUS: "This single angle controls the shape of the ionization signal response observed within the TPC waveform" ([2407.11925, p. 28](https://arxiv.org/abs/2407.11925)).
- Part I performance metrics are charge bias, charge resolution and inefficiency (§4.3). Charge bias about 1 % for a point source ([1802.08709, p. 45](https://arxiv.org/abs/1802.08709)). "Time smearing" is defined as a width.
- **Same phenomenon, stated for charge.** The reconstructed spectrum "is dominated by the Gaussian smearing for small θ xz , or by the track topology for large θ xz in which case the contributions from the charge in neighboring wires are stretched in time and produce a triangular-like shape"; induction inefficiency at large θxz "is related to diffusion which introduces an additional smearing" ([1802.08709 §4.3.5, p. 49](https://arxiv.org/abs/1802.08709)).
- Prolonged tracks: induction signals are "low amplitude and a long duration" (§2.3), with gaps or complete track loss (§4.3.4) ([1802.08709](https://arxiv.org/abs/1802.08709)).
- ICARUS measures average waveforms in 2° θxw bins over 20°-76°, on anode-cathode-crossing muons 13-16 cm from the anode to limit diffusion; finds disagreement with nominal simulation "especially at high track angle" (quote spans a page break, read in layout text, not string-matched) and fits the field response ([2407.11925 §4.1-4.2](https://arxiv.org/abs/2407.11925)).
- ICARUS aligns tracks with a non-linearity metric on (t_i − t_i,exp) for a straight track in the angle bin, shifting up to ±5 ticks ([2407.11925 Eq. 4.1](https://arxiv.org/abs/2407.11925)). The expected slope is taken from the TPC track itself.
- MicroBooNE WireMod: data/simulation ratios of hit charge Q and width σ against x, (y, z), θXZ, θYZ, applied to waveforms; hit widths "vary by up to 50% across the drift direction" ([2111.03556, p. 5](https://arxiv.org/abs/2111.03556)). The ratio functions are for Q and σ.
- MicroBooNE dQ/dx calibration excludes |θXZ| in 75°-105° for crossing cosmics ([1907.11736 §2.3](https://arxiv.org/abs/1907.11736)).
- External references: anode/cathode-crossing t0 from the PMT flash, validated by a small cosmic tagger, correct "at least 98%" of the time ([1804.02583, p. 20](https://arxiv.org/abs/1804.02583)); ICARUS anode-cathode-crossing tracks ([2407.11925](https://arxiv.org/abs/2407.11925)); ProtoDUNE-SP has a CRT and beam instrumentation ([2007.06722](https://arxiv.org/abs/2007.06722) §2-3).
- Part II's average-response extraction notes "a Gaussian smearing of approximately 1 µs introduced by the method itself", i.e. a width, not an offset ([1804.02583, p. 21](https://arxiv.org/abs/1804.02583)).

### Inferences
- **Searched and silent:** a case-insensitive search of Parts I and II for "time bias", "timing bias", "time shift", "peak time", "hit time", "arrival time" returned nothing relevant; the t0 and drift-coordinate discussion concerns global offsets.
- A crossing-track t0 constrains the global drift coordinate but not a depth-dependent compression, except weakly through the track ends. None of the references read is used as an independent slope reference for hits, unlike our M3 telescope.
- Our 20-37 % has no LAr analogue in what I read, and is not expected for long tracks.

### Gaps
- Whether a hit-time bias study exists in theses, DUNE/SBN internal notes or LArSoft validation was not determined. A targeted search returned only slides ([Gu, point-source simulation](https://indico.fnal.gov/event/44651/contributions/192523/attachments/132023/161932/PointLikeSourceSimulation_20200724_WGu.pdf), not read beyond the search summary).
- ProtoDUNE-SP's beam angle or CRT as a slope reference for hit times: not located.

## How do Gaussian hit-finding and "a single time per wire" relate to our estimator?

### Takeaway
LAr uses a per-wire Gaussian fit to the deconvolved waveform, optionally with several Gaussians. This is closer to a peak or centroid than to a leading edge, and it is known to be inadequate for prolonged tracks.

### Cited Findings
- "The hit finding algorithm fits peaks in the wire waveforms, where a hit represents a charge deposition on a single wire at a given time" ([ProtoDUNE-SP, 2007.06722 §4.5.1, p. 43](https://arxiv.org/abs/2007.06722)). The same section notes non-Gaussian shapes for trajectories close to the wire-field plane.
- ROI fitted to a variable number of Gaussians (multiplets, "hidden" Gaussians, "crude hit"). "The hit time relative to" t0 "could be calculated using the charge weighted mean of the ROI", which loses separation information ([Baller, 1703.04024 §4, p. 12](https://arxiv.org/abs/1703.04024)).
- "Ionization electrons produced on tracks that travel roughly in the electric field direction, φ ≈ 0, will arrive over many microseconds. In this case a single Gaussian fit is clearly inadequate" ([1703.04024 §3.5, p. 10](https://arxiv.org/abs/1703.04024)).
- ICARUS: fitted pulses on collection, "The maxima of pulses are considered as the drift coordinate positions of the new hits" ([1210.5089 §2.1](https://arxiv.org/abs/1210.5089)).
- Diffusion "can influence the event images and the accuracy of the drift coordinate measurement" ([Yang, 2012.01319 §3.4, p. 11](https://arxiv.org/abs/2012.01319)); that review does not quantify it.

### Inferences
- The LAr hit is a single-time-per-channel estimator in the same sense as ours, applied after removal of field and electronics response. Mixed charge from neighbouring depths remains in the waveform.
- In mid-track, the charge from both neighbours is symmetric in time, so a Gaussian peak or centroid should be nearly unbiased. A bias remains where the mixing is one-sided, at track ends. Our P0 result fits this: mid-column compression about 4 %, with most of the rest at the end bins (`LITERATURE_MIXING.md` reviewer note). This is an inference and no LAr paper tests it.
- At equal tanθ, the LAr end-affected fraction is 2σ_T/(L tanθ) ≈ 0.5 % against about 19 % for us **(derived)**.

### Gaps
- No paper read compares peak, centroid and leading-edge hit time on LAr tracks.

## Transverse diffusion: effect on wire signal width and time; measurements from signal width

### Takeaway
MicroBooNE 2021 states the mechanism we describe, for pulse width: transverse spread overlaps adjacent clouds and adds time smearing, growing with θxz. It is the most direct precedent. Its null result is confined to near-isochronous tracks and does not transfer to our angles.

### Cited Findings
- "A measured pulse width may have contributions to the pulse width from the effects of D_T. Adjacent electron clouds begin to overlap as they spread in the yz-plane under the influence of D_T, causing additional σ_t smearing … The impact of D_T increases as a function of track θ_xz. This motivates the strict angular requirement" ([MicroBooNE, 2104.06551 §5.1, p. 18](https://arxiv.org/abs/2104.06551); subscripts normalised).
- D_L "is actually an effective diffusion coefficient that contains a small contribution from transverse diffusion" (p. 8); "As θ_xz increases, so does the intrinsic spread in x of the ionization position distribution" (p. 10) ([2104.06551](https://arxiv.org/abs/2104.06551)).
- Varying simulated D_T over 4.8 / 5.7 / 7.2 cm²/s gives measured D_L 6.26 / 6.26 / 6.25 cm²/s (Table 3). They attribute this "to the two-dimensional nature of the MicroBooNE deconvolution—which deconvolves the signal in both time and wire space, mitigating the impact of charge spread to neighboring wires—and our stringent requirement on the value of θ_xz", at θxz = ±6° ([2104.06551, pp. 19-20](https://arxiv.org/abs/2104.06551)).
- Width-based method: σ_t²(t) = σ_t²(0) + (2D_L/v_d²)t, with σ_t²(0) ∼ 1.96 µs² ([2104.06551 Eq. 3.1](https://arxiv.org/abs/2104.06551)).
- ICARUS data: wire-by-wire dQ/dx against 10-wire coarse-grained dQ/dx converge at large drift, "the first confirmation in data of the impact of transverse diffusion on the muon charge scale" ([2407.11925, p. 8](https://arxiv.org/abs/2407.11925)). Theory: [Putnam & Schmitz, 2205.06745](https://arxiv.org/abs/2205.06745), ∼4 % MPV shift for a 1 GeV muon at 2 ms and 4.7 mm spacing.

### Inferences
- At θxz = 6°, drift per wire is 3 mm × tan 6° = 0.32 mm (0.29 µs), and the mixing time is σ_T·tan 6°/v ≈ 0.14-0.22 µs **(derived)**, added in quadrature to σ_t(0) = 1.4 µs. So the null in Table 3 is expected from kinematics alone. It does not demonstrate that 2D deconvolution removes diffusion at large angle. The kernel contains no diffusion (see the deconvolution section), so the attribution in the paper is questionable for diffusion proper. It could refer to the induced component.
- Therefore a claim that transverse diffusion mixes depths into one channel's time signal is **not new as a statement about width**. Our new element is its effect on the time *value* and track slope. `LITERATURE_MIXING.md` Sec. 1 point 2 ("no paper … states that transverse diffusion alone … compresses the time ladder") remains true as worded, but should now cite this paper as the nearest precedent.

### Gaps
- No measurement of D_T from signal width at typical LAr fields was found; Putnam & Schmitz say none exists as of 2023. Later work was not established.

## Test of hypotheses (a), (b), (c), with contrary evidence

### Takeaway
(a) supported in part; (b) supported for long tracks only; (c) yes for the model and the width effect, not found for the time bias.

### Cited Findings
- **(a) Time-slice vs per-channel axis.** Wire-Cell reconstructs per time slice: "Each time slice represents a tomographic cross section across the detector" ([1803.04850, p. 3](https://arxiv.org/abs/1803.04850)). Pandora/LArSoft-type chains use per-wire hits ([2007.06722 §4.5.1](https://arxiv.org/abs/2007.06722)). So LAr uses both axes in production.
- **(b)** see the geometry table (derived).
- **(c)** see the 2D deconvolution, angle and diffusion sections above.
- Contrary evidence on (a): per-wire hit reconstruction is used in production and no slope bias is reported ([2007.06722](https://arxiv.org/abs/2007.06722); [1703.04024](https://arxiv.org/abs/1703.04024)). So the per-channel axis alone does not imply a visible bias.

### Inferences
- (a): In a time slice, a transversely symmetric diffusion cloud leaves the centroid unbiased, so the time-slice axis is the natural one to avoid the bias. Our forward fit achieves the equivalent. But the LAr per-wire chain shows the axis matters only when the lever arm is short. Paper II should present (a) and (b) together.
- Contrary evidence on (b): the short-segment case in LAr has a mixing ratio equal to or larger than ours and no bias is reported. Possible reasons, none tested in any source: nobody measures short-segment slopes against a reference; hits on short tracks are dominated by other effects; t0 and v are external to the segment, whereas a µTPC infers angle only from the ladder.
- Contrary evidence on novelty: Adams 2018 (model), Part II L1SP (per-event regularised fit), MicroBooNE 2021 §5.1 (mechanism for width), ICARUS 2024 (angle-binned waveform tuning). Suggested wording for Paper II: credit these, and limit the claim to the effect on the time ladder and slope, its measurement against an independent telescope, and the depth-resolved per-event fit.
- Also differs from LAr: a µTPC derives the local angle from one short column; LAr derives direction from long tracks and multi-plane matching.

### Gaps
- Coverage: 21 arXiv papers were downloaded; about 16 of them were read in the relevant passages and string-searched, alongside web searches. None was read cover to cover. Not covered: theses, DUNE/SBN technical notes, LArSoft documentation, pixel LArTPC (ND-LAr) papers. ICARUS 2D deconvolution and SBND signal-processing status come only from [talks](https://indico.bnl.gov/event/21492/contributions/88978/attachments/53846/92090/LArTPC_WireCellSummit_24_04_11.pdf) and the [APS abstract](https://archive.aps.org/apr/2024/c07/6) seen as search summaries, and were not verified.
- I found no source testing mid-track symmetry against end effects on LAr hit times.
