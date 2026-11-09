# Separating capacitive or induced inter-strip coupling from charge spreading in sampled strip waveforms

Scope note. All methods listed are for existing data or simulation. "Read" means the passage was read in full text; "snippet" means only a search result was seen. Paths under `/media/ucla/work/` are local extracted copies. Toy numbers marked **(toy)** are my own calculation (`/media/ucla/work/capcoupling/sep_scripts/toy.py`), with an assumed CR-RC² shape (τ = 120 ns), 20 free depth charges, peak strip at 30 σ, one track, sampled at 60 ns. They are not a measurement, and the real template, noise and tan θ mix will change them.

## What waveform signatures distinguish a coupled or induced neighbour signal from real collected charge?

### Takeaway
There is no single "coupling shape". Three different things get called coupling, and they have different signatures: a Ramo transient (bipolar, zero net), a capacitive charge divider (unipolar, same time, same shape, non-zero net), and electronic cross-talk (often bipolar, at the ~1 % level in MicroBooNE). The commonly used tests (same time, fixed charge ratio) separate coupling from **later-arriving** charge, but do not separate it from a **same-time** footprint such as σ0. Only sub-strip position dependence, depth dependence and zero-charge probes break that.

### Cited Findings
- General rule for a Ramo induced current: currents on electrodes that collect no charge are bipolar — [Riegler, Signal Formation in Various Detectors, arXiv:1406.3217](https://arxiv.org/pdf/1406.3217) (snippet). Klitzner's floating-strip thesis likewise says movement that collects nothing "has to be strictly bipolar", and shows a bipolar current on perpendicular and parallel readout strips (Klitzner PhD, LMU 2019, local `/media/ucla/work/capcoupling/txt/Klitzner_PhD.txt`, lines 2798, 2921-2922).
- Bipolar signals integrate to zero: ProtoDUNE-SP states induction signals "are bipolar and thus integrate to zero" — [DUNE, arXiv:2007.06722](https://arxiv.org/abs/2007.06722) (read).
- Before electrons land, weighting-field sharing is large and falls to a small final value. In Flierl's simulated readout the neighbour sees "80 % of the pulse-height" while the electrons are far, and "a final coupling of 15 % of the cumulative charge" at landing, with 0.8 % for the second neighbour — [Flierl PhD, LMU 2018](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf), Sec. 2 (read).
- Electronic coupling can be bipolar and small: "Calibration signal injection is observed to induce electronic crosstalk into adjacent channels with a bipolar signal size ∼1% of the calibration pulse height" — [MicroBooNE, arXiv:1804.02583](https://arxiv.org/abs/1804.02583) (read). A GEM ME0 prototype reports bipolar pulses traced to capacitive coupling between readout sectors, shaped by CR differentiation — [inspirehep file](https://inspirehep.net/files/0c1fe8f8b232f38500806196ddffa3b7) (snippet).
- Charge-divider coupling onto a neighbour, as modelled for the track method: "some ionization electron arrive earlier on the readout strips than others. Thus theses charges are first capacitively coupled to a neighboring strip and then the ionization electrons arrive on that strip. Therefore it seems like a slower signal rise, but really it is the superposition of two signals." — Lösel PhD, LMU 2017, Sec. 7.1.2 (local `/media/ucla/work/capcoupling/txt/Loesel_PhD.txt`, read). This is the clearest published statement of the inclined-track waveform signature.
- Delay. Pulser test on BESIII triple-GEM with APV25: "higher strip charge generates always a signal of about 5% of the inducing charge on the neighboring strips after about 15 ns" — [Farinelli PhD, arXiv:1904.06548](https://arxiv.org/abs/1904.06548) Sec. 5.5 (read). So in that system the injected coupling is not strictly simultaneous. BESIII also describes the induced edge signals as having "same arrival times as their neighbours" ([Alexeev et al., arXiv:1908.06253](https://arxiv.org/abs/1908.06253), read); the two statements can coexist because 15 ns is less than one 25 ns APV25 sample (my reading; neither source reconciles them).
- A delay separates resistive spreading from diffusion, not from coupling: "Contrary to the 'charge sharing' effect, the signals in neighbouring pads are delayed compared to the one of the leading pad in case of the 'charge spreading'" — [T2K ERAM, arXiv:2212.06541](https://arxiv.org/abs/2212.06541) (read).
- Linear amplitude ratio: every published coupling constant is applied as a fixed fraction (5 %, 15 %, 23-32 %, in the sources below). I found no published test of whether the ratio is constant across charge in an MPGD apart from Farinelli's measurement of ratio versus main strip charge.

### Inferences
Signature table for a strip with real charge landing at t_i (T = electronics template; W = strip waveform):

| Component | Net integral | Shape on neighbour | Time | Scales with depth? | Depends on sub-strip position? |
|---|---|---|---|---|---|
| Collected charge, later depth | ≠ 0 | T | own arrival | yes, through position | yes |
| Same-time footprint σ0 | ≠ 0 | T | same as main | no | **yes, strongly** |
| Diffusion | ≠ 0 | T summed over depths, wider | arrival of contributing depths | **yes, σ² grows** | weakens with depth |
| Capacitive divider into a charge-sensitive input | ≠ 0 (k·Q) | T of main | same (or a fixed small delay) | **no** | **no, for a fixed k** |
| Weighting-field transient | 0 | ≈ small derivative of T after shaping | during drift in gap | small | yes |
| Electronic bipolar cross-talk | ≈ 0 | derivative-like | same | no | no |

The bipolar label is not universal. A charge-sensitive preamplifier integrates the current coupled through Cc, so a capacitive divider gives a **unipolar** neighbour with the template of the main strip. Bipolar neighbours are expected from a Ramo transient, a high-impedance or differentiating path, or chip-level cross-talk. This reasoning is mine; only the individual bipolar observations above are cited. A check therefore needs both terms: W_neighbour ⊃ k0·W_main + k1·τ·dW_main/dt.

The MX17 consequence: "same time as the main strip and a fixed charge ratio" (the ghost-hit criterion) describes a same-time Gaussian footprint just as well. With σ0 = 0.30 mm at 0.78 mm pitch, a cloud centred on a strip gives neighbours of about 12 % of the main (erf arithmetic, mine), the same size as published "coupling".

### Catalogue of applicable methods (existing data only)

| # | Method | Observable | Coupling case | True spreading case | Precedent |
|---|---|---|---|---|---|
| 1 | Sub-strip phase (η) dependence | Neighbour/main ratio versus track position within strip, using the fitted or M3 position | A floor at k on both sides, independent of position | Vanishes at strip centre for a narrow footprint, large near the edge | [Belle II SVD VXD workshop plan](https://indico.mpp.mpg.de/event/3730/contributions/8295/attachments/6774/7456/8th_VXD_Workshop_10-09-15.pdf) (read: a plan, no result) |
| 2 | Depth dependence | Neighbour ratio at a given position versus drift time | Constant in depth | Grows with D²·drift | Forward-model structure itself |
| 3 | Stacked waveforms | Align many strips at given (phase, depth, tan θ) and average | Neighbour stack ∝ main stack, possibly plus derivative | Different shape (mixture of depths) | MicroBooNE full-response extraction, [arXiv:1804.02583](https://arxiv.org/abs/1804.02583) |
| 4 | Forward-fit with explicit k0, k1 | Profile likelihood of the two terms, global per chamber and condition | k > 0 | k consistent with 0 | See final section |
| 5 | Zero-charge probe, inside electronics | Waveform on a strip that cannot collect charge, conditional on neighbour | k_electronics | 0 | IPC hot-pixel method, [METIS H2RG, arXiv:2607.29438](https://arxiv.org/pdf/2607.29438) (snippet). MX17: chamber A x-view connector 8, FEU 3 channels 448-511 in run_79, disconnected at the connector |
| 6 | Zero-charge probe, geometric | Strips beyond the track's end or outside the active region | Pure k | Gaussian tail only | None published for MPGD |
| 7 | Spark or very-large-charge impulse | Neighbours around a discharge | Fixed ratio, unipolar | Spread beyond | [Bortfeldt PhD](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf) notes capacitively coupled pulses decaying with distance from a discharge (read). Saturation of the main strip voids the ratio; use only the shape and an upper bound |
| 8 | Noise correlation | Covariance between adjacent strips in pedestal runs, after common-mode subtraction | Coupled input noise gives a correlated residual | Zero | IPC: Poisson noise is correlated by capacitance but not by diffusion, see silicon section |
| 9 | Cross-view coincidence (X/Y checkerboard) | Y strips receiving a signal proportional to X, at X's time | Inter-layer term | Not produced by drift | [Lin et al., arXiv:1406.6871](https://arxiv.org/abs/1406.6871) attribute X/Y differences to capacitive coupling between resistive strips; [Banerjee et al., arXiv:1708.04087](https://arxiv.org/abs/1708.04087) report higher coupling to one plane |
| 10 | External-angle closure scan | Reconstructed versus M3 angle, scanning k in a time-dependent correction | Minimum bias at true k | — | Flierl 5 % scan, see strip section |
| 11 | Simulated k prior | SPICE or field solve of the strip geometry | Prior only | — | Lösel LTspice |

Practical limits on these for MX17: methods 5 and 8 require data that was not zero-suppressed, or a selection that handles the suppression bias (ZS is a known censoring source in this campaign). Method 8 only constrains the input network; correlated noise may arise from ASIC-adjacent channels rather than strip-adjacent ones, so compare strip-adjacent with ASIC-adjacent pairs (see the JLab channel-ordering note below).

### Gaps
- I found no published waveform-level test of a unipolar charge-divider shape on an MPGD strip neighbour. The shape claim above is inference.
- Whether bench `decoded_root` for any disconnected-channel period retains sub-threshold samples was not checked.

## How do neighbour signals behave in silicon strip APV25 data, and are there data-driven cross-talk fits?

### Takeaway
Silicon practice measures cross-talk from physics events by **selecting geometry that suppresses true sharing** (perpendicular tracks, B = 0) and reading the ratio of neighbour to seed charge in non-suppressed data. It then tunes a three-point kernel in simulation. I found no paper fitting a full cross-talk matrix from physics events. The closest pulser-free MX17 analogue is a fixed-phase selection, since MX17's drift diffusion cannot be suppressed by geometry.

### Cited Findings
- CMS method: "The cross talk is best characterized using the so called 'virgin raw data', where no thresholds are applied ... taken with magnetic field off. To suppress the charge sharing, only the tracks perpendicular to the sensor surface were selected and the amount of charge deposited on the neighboring strip was measured." Cross talk "varies by layer, but it is typically below 3% in the APV25 peak mode", expected higher in deconvolution mode — [PoS(RD09)006](https://pos.sissa.it/098/006/pdf) (read).
- CMS 2018 cosmics: "the cross talk was measured in NZS data using cosmic ray muons"; "The majority of the charge (between 75 and 86%) is collected in the leading strip and its two neighbors (6–10%)" — [CMS, arXiv:2506.17195](https://arxiv.org/abs/2506.17195) (read). Simulated by "addition of inter-strip cross talk" as part of digitisation.
- APV25 deconvolution is a weighted sum of three consecutive samples that narrows the pulse — [CMS, physics/0312132](https://arxiv.org/abs/physics/0312132) and [physics/0610259](https://arxiv.org/abs/physics/0610259) (read), original method [Gadomski et al., cited in arXiv:0911.4996](https://arxiv.org/abs/0911.4996). I did not access French et al. (NIM A466, 2001).
- Belle II SVD: charge sharing among implants is simulated with coupling constants measured on data and stored in a calibration database — [Belle II software docs](https://software.belle2.org/release-09-00-15/sphinx/_sources/svd/doc/simulation.rst) (snippet). Method proposal: parametrise η = Q_L/(Q_L+Q_R) with Q_seed = (1−2k)Q_raw, Q_L = Q_R = k·Q_raw, per strip — [Caria, 8th VXD Workshop](https://indico.mpp.mpg.de/event/3730/contributions/8295/attachments/6774/7456/8th_VXD_Workshop_10-09-15.pdf) (read, explicitly "a tentative plan").
- LHCb VELO: the discussed cross-talk is **inter-symbol** (spill between consecutive time samples and from digital headers), with a suppression algorithm, not an inter-strip matrix — [LHCb, arXiv:1405.7808](https://arxiv.org/abs/1405.7808) (read for those lines only).
- Channel adjacency: JLab GEM notes trace APV25 cross-talk to adjacent channels in the multiplexer order, so a matrix must be indexed by readout channel; one strip-level cut flags a neighbour when the mother strip ADC is about 8 times larger — [JLab slides](https://hallaweb.jlab.org/12GeV/SuperBigBite/SBS-minutes/2013/sbsWeeklyMeetingKG_20130515.pdf), [PCRE wiki](https://wiki.jlab.org/pcrewiki/images/f/f9/Weekly_1118.pdf) (snippets).
- Direct analogue in astronomy detectors. Capacitive coupling correlates Poisson noise; diffusion does not, because "If diffusion is negligible, there is no correlation in the numbers Ni for different pixels", and "while the interpixel capacitance increases the fluctuations, diffusion from one pixel to the adjacent ones would reduce them"; one H2RG study gives fd < 0.037 (3σ) — [arXiv:0905.4442](https://arxiv.org/abs/0905.4442) (read). The hot-pixel method takes the neighbour signal around a pixel with no real signal as cross-talk — [arXiv:2607.29438](https://arxiv.org/pdf/2607.29438) (snippet). The flat-field autocorrelation has no MX17 counterpart (a gas avalanche has no Poisson-limited flat), so the transferable part is the noise covariance in pedestal runs (my inference).

### Inferences
Deconvolution is a temporal, three-sample operation on a single channel. It sharpens timing but **does not address inter-strip mixing**. The word "deconvolution" in APV25 and in LAr means two different things.

The silicon recipe does not transfer directly: silicon suppresses sharing by selecting perpendicular tracks with a thin sensor, and even then a Gaussian charge cloud remains. For MX17, perpendicular tracks have a full-gap diffusion, so the analogue of "perpendicular selection" is selection in depth near the mesh combined with a fixed sub-strip phase.

### Gaps
- Could not access CMS NOTE-2007-027 (Delaere and Mirabito), the original CMS cross-talk procedure, nor the Bernardini paper (not found by name in search).
- Whether Belle II SVD published the actual data-driven constants: only the software documentation snippet was seen.
- I found no published pulser-free cross-talk matrix fit in MPGD or silicon strips; negative result of a limited search.

## How do LAr TPCs separate induced long-range signals from collected charge with 2D deconvolution?

### Takeaway
LAr writes neighbours into the response as functions R_0, R_1, … of time, computed from Garfield, and solves in wire and time with Wiener filters. It validates the response in data by **stacking aligned track waveforms** and comparing with simulation at large angle. This is the form MX17's forward model would take with a coupling term, but in LAr the neighbour response is a field effect with known physics, not a fitted parameter, and diffusion stays in the image.

### Cited Findings
- Model: M_i(t0) = ∫(… + R_1(t0−t)S_{i−1}(t) + R_0(t0−t)S_i(t) + R_1(t0−t)S_{i+1}(t) + …)dt, Fourier-transformed into a Toeplitz matrix in wire number — [Adams et al., arXiv:1802.08709](https://arxiv.org/abs/1802.08709) eqs 3.5-3.7 (read). "R_1 represents the average response function for an ionization charge drifting past in the next adjacent wire region."
- The same model in ProtoDUNE-SP: M = Q ⊗ F(w, t) ⊗ E(t) (eq. 4.5); "Because of the long-range induction effect, instead of a 1D deconvolution … a two-dimensional (2D) deconvolution involving both the time and wire dimensions is performed", in frequency space with "two Wiener-inspired filters" — [arXiv:2007.06722](https://arxiv.org/abs/2007.06722) (read).
- Validation: "Waveform signals are lined up in time across many different wires (and tracks from different events) and are added together … The result is a data-driven estimate of the full response for each plane in each of the regions"; simulations using ±3, ±10, ±50 wires were compared, and the "front porch" converged towards the data with more wires — MicroBooNE [arXiv:1804.02583](https://arxiv.org/abs/1804.02583) Sec. 3 (read).
- Neighbour induction is tested specifically at large angle: "This effect is more noticeable when an ionization track is at higher angle … a similar comparison … using tracks with 40° < θ_xz < 50°", "the simulation performs quite well in reproducing the effect of charge induced on wires neighboring the wire closest to drifting ionization"; residual discrepancies remain ("front porch", a small dip on Y) — same source.
- A natural experiment: in the "shorted" regions the wire response changes "unipolar instead of bipolar in shape", visibly exposing the field-response dependence — same source.
- The 1D versus 2D comparison in data event displays is qualitative recovery of charge — [Adams et al.](https://arxiv.org/abs/1802.08709) Fig. 11 (read).

### Inferences
The LAr equivalent of "the field response on neighbours" in MX17 would be a kernel R_{±1}(t) = k0·T(t) + k1·τ·T′(t) in the forward model. MX17 has what LAr did not need: the kernel acts per sample at known times, and the fit is per-event and regularised rather than a Fourier inverse.

LAr's stacked-waveform validation is the most transferable idea: stack at fixed tan θ and depth, compare data against the forward model with k fixed to 0, and look at the residual on the ±1 strip versus time. A pattern with the main strip's shape and independent of depth implicates coupling; one that grows with depth implicates D.

LAr's response is calculated, not fitted, so the question of degeneracy with a footprint does not arise there. Its methodology cannot be cited for the degeneracy problem.

### Gaps
- Sections of Adams on the full evaluation of topology-dependent bias were read only by section heading. ProtoDUNE-SP validation of the neighbour response was not read beyond the formula.
- Whether any LAr paper fits a wire-to-wire response from data as opposed to validating a calculated one was not established.

## How do T2K ERAM and resistive Micromegas fits treat neighbours and separate charge dispersion from coupling?

### Takeaway
They do not separate an inter-pad capacitive term: the model is diffusion plus RC dispersion. T2K and Dixit-style fits assume the neighbour's signal is RC-dispersed charge with the electronics response, and the delay of the neighbour is the handle for the RC term.

### Cited Findings
- T2K model: Q(t) = (q_e/4)[erf((x_high−x0)/(√2σ(t))) − erf((x_low−x0)/(√2σ(t)))] × [same in y], with σ(t) = √(2t/τ + ω²), τ = RC, ω the transverse diffusion term; the "leading pad" is fitted first, then neighbours are fitted "simultaneous[ly]" with a convolution of I(t) and Q(t); "neighbour pads sensitive mostly to the induced charge due to the resistive effect" — [Attié et al., arXiv:2106.12634](https://arxiv.org/abs/2106.12634) Sec. near eq. 5 (read). The X-ray paper states the capacitive mechanism as signal transfer: "Capacitive coupling between the resistive layer and the conductive pads means that charges at some place on the resistive layer are compensated by opposite charges on the pads" — [arXiv:2303.04481](https://arxiv.org/abs/2303.04481) (read). That is layer-to-pad coupling, not pad-to-pad.
- T2K charge sharing is undelayed and diffusion-driven; charge spreading is delayed — [arXiv:2212.06541](https://arxiv.org/abs/2212.06541) (read). The key characterising variables are "the charge ratio and the time difference" with the leading pad (same paper).
- Dixit: the anode is "a distributed 2 dimensional RC network", with ∂ρ/∂t = h ∂²ρ/∂x², h = 1/RC, and the position comes from bias-corrected pulse-height centroids — [Dixit et al., physics/0307152](https://arxiv.org/abs/physics/0307152) (read). Neither this nor [Dixit and Rankin, physics/0605121](https://arxiv.org/abs/physics/0605121) has an inter-strip capacitive term in the parts I read.
- On resistive strips, capacitive coupling is part of the normal signal chain, not a defect: the signal reaches the readout "via a capacitive coupling" to the resistive strips ([Lin et al., arXiv:1406.6871](https://arxiv.org/abs/1406.6871), read) and "by capacitive coupling" ([Galan et al., arXiv:1304.2057](https://arxiv.org/abs/1304.2057), read). A Riegler-type general treatment: the limiting cases are a resistive layer acting as a dielectric (induction directly on readout) and as a grounded conductor (signal from coupling only), with real cases between — [arXiv:1406.3217](https://arxiv.org/pdf/1406.3217) (snippet). A Garfield++ study of 2D resistive strip Micromegas reports neighbour peaks progressively smaller and shifted in time — [arXiv:2304.01883](https://ar5iv.labs.arxiv.org/html/2304.01883) (snippet).

### Inferences
The T2K delay argument handles only one pair: RC spreading (delayed) against drift-gas sharing (undelayed). A capacitive term that is itself undelayed sits on the "undelayed" side with σ0 and cannot be separated by this criterion. On MX17's X view, which has no RC term, the delay test has no discriminating power at all between σ0 and k.

For the Y view, a three-way degeneracy is possible among RC kernel strength, σ0 and k. The RC term has a delay and is therefore partly distinguishable; k and σ0 are not distinguished by delay. T2K's practice of measuring RC on dedicated X-ray events with a known point source has no MX17 equivalent, since no new data can be taken.

### Gaps
- No T2K paper I read mentions pad-to-pad capacitive coupling or fits it. Attié's waveform fit is stated as under study in the existing survey.
- Could not establish the size of the residual charge-sharing bias in T2K due to a coupling term; none is reported.

## What did BESIII CGEM and ATLAS NSW use to identify induced strips, and how well validated is it?

### Takeaway
Both use a same-time plus fixed-charge-ratio criterion, sometimes with a low-charge condition, and validate it only by reduction of the angle bias, not by an independent measurement of which strips are truly coupled. The criteria also lump diffusion in, as the authors themselves say.

### Cited Findings
- BESIII: "Induced signals in the first and last strips of a cluster have same arrival times as their neighbours, and in the µTPC reconstruction the effect is a tilt of the original track towards larger angles. In addition the transverse diffusion plays another important role … The induced signal is identified studying the time information and charge ratio of the strip w.r.t. its neighbour. A subsequent weighting or suppression of the first and/or last strips of the cluster has been implemented. The correction algorithm is still in a preliminary stage, but first results showed a reduction of the angle bias." — [Alexeev et al., arXiv:1908.06253](https://arxiv.org/abs/1908.06253) (read).
- ATLAS as reported by Farinelli (secondhand, the primary ATLAS note is ATL-MUON-PROC 2014-011 or similar; not read): "ATLAS observed an average of 15% of the inducing charge between a strip and its nearest neighbouring strip … If the charge measured on those strips is smaller than 300 ADC (10 fC), the charge ratio with its neighbouring strip is 15% and the time measured is the same, then this is associated to a ghost hit" — [Farinelli, arXiv:1904.06548](https://arxiv.org/abs/1904.06548) App. B (read).
- BESIII correction is empirical, and acknowledges both effects are together: it fits residuals versus strip index and shifts edge points ("The first correction applied shifts the last strip to the mean value of the others (about 180 µm)"), and notes "the correction depends on the gas mixture". Corrections cover "those strips [with] signal due to diffusion and capacitive effects" together — same source, App. B.1 (read).
- Other time-independent treatments: Klitzner applies a "hard exclusion" aimed at capacitive strips, and says a "time dependent charge correction of each cluster strip signal might be considered" for metre-long strips (Klitzner PhD, local, read).
- Time-dependent correction, the most developed form. Lösel: "The percentage of the transferred charge is calculated sampling step-wise, subtracted from the neighboring strips and added to the pulse height of the originating strip", with a simulated 23.4 % (10 cm strips; 29.1 % at 100 cm from LTspice); it fixes 30-40° exactly, 20° slightly shifted, and "for 10° it still does not work very well" (Lösel PhD, local, Sec. 7.2.2, read).
- Flierl's scan on resistive strips: empirical optimisation in 5 % steps between 0-40 % "revealed experimentally the best reconstruction of the mean angle at a charge coupling of 30 %", beyond which overcompensation appeared; simulations by Lösel predicted 23-32 %; he warns the large tail "is a hint that the charge coupling could not be resolved perfectly" — [Flierl PhD](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf) Sec. 6.5 (read).
- Bortfeldt: a capacitive-coupling simulated shift of +9° at 10° inclination, from an LTspice-type study — [Bortfeldt PhD](https://edoc.ub.uni-muenchen.de/16972/1/Bortfeldt_Jonathan.pdf) (read).

### Inferences
Validation in all these is the angle returning to the set chamber angle. That is a weak test of mechanism: the correction is effectively tuned to remove the bias, and a same-time Gaussian footprint would have produced the same residual pattern and been "corrected" equally well. Flierl's 30 % fit, with chamber angle as the reference, measures an effective strength of *all* mixing that the single-time reconstruction sees (diffusion, footprint and coupling), so its value should not be read as a capacitance.

Variance conversion (mine): the same-time footprint σ0 and a kernel (1−2k, k, k) add to the cluster variance as σ² → σ² + 2k·p². With p = 0.78 mm, MX17's σ0 = 0.30 mm corresponds to k = 0.074 if it were all coupling. Published coupling sizes of 15 % or more would by themselves imply a same-time width of about 0.43 mm at MX17's pitch (k = 0.15, zero Gaussian part), exceeding the measured 0.30 mm. So either MX17's coupling is below about 7 %, or the fitted σ0 is partly coupling and the true avalanche footprint is correspondingly narrower. This is an arithmetic bound that assumes the kernel and footprint add like this, and that the coupling in other geometries is not directly transferable (other pitch, strip length and capacitances).

### Gaps
- The original ATLAS ghost-hit source was not read; Farinelli's reference list does not map one-to-one in the text I searched.
- No independent "truth" validation of ghost hits exists in sources read, e.g. simulation with known coupling and truth labels on real test-beam data.

## What features differ for a straight inclined track where a neighbour gets both real diffused charge and coupled signal?

### Takeaway
Three features are specific to the inclined track: the neighbour precursor occurs at the *main strip's* landing time and repeats the main strip's waveform, its strength does not change along the track, and it occurs on the **early side only** in the sense that matters for a leading-edge time.

### Cited Findings
- Waveform superposition mechanism and bias direction: Lösel's quote above; Lösel: "This leads to a systematically larger angle for the reconstruction of the track angle in a single plane"; Flierl: "this cross-talk … influences the signal shape of strips, which are reached later by the electron cloud"; subtracting the influence of first-responding strips compensates the shift (Flierl, Sec. 2.4.2, read).
- Edge strips and cluster-end points are where it matters most: Alexeev ("the first and last strips in the strip cluster playing the major role"); Farinelli App. B shows residual versus strip index.
- ATLAS test-beam, Scholer: "Adding CS pulls the time measurements more towards the mean time" — quoted in `papers/paper2/hits_daq/LITERATURE_TPC_SURVEY.md` (existing survey).

### Inferences
These follow from the model and the cited mechanism; none is stated as a test in a paper.

1. Between adjacent strips the arrival-time difference is Δt = p/(v·tan θ): 141 ns at tan θ = 0.15, 70 ns at 0.3, 21 ns at 1 (v = 36.8 µm/ns). The coupled precursor occurs Δt before the strip's own arrival. At tan θ ≥ 0.5 it is under one 60 ns sample and blends with the rising edge; at tan θ ≤ 0.3 it is a resolvable leading structure.
2. Diffusion also puts earlier and later charge on the strip, but with a kernel of a Gaussian in position, dependent on the depth at which the main strip's charge came from. Near the cathode the mixing is wide, near the mesh narrow. A coupling kernel has the same form everywhere on the track. So: **the ratio of neighbouring-strip precursor amplitude to the main strip amplitude at the mesh end is the cleanest coupling observable**, because σ² → σ0² there, and any remainder beyond σ0 must come from k.
3. The coupled term is on both neighbours and symmetric at fixed time. The later strip (a charge arriving afterwards) has a precursor that the leading edge picks up; the earlier strip has a *trailing* component that a leading edge ignores. Hence a leading-edge time sees only the earlier-charge side. A charge-weighted time would see both, shifting the time toward the middle. This is why the two hit-time estimators would respond differently, and the existing plan to add a charge-weighted time (`LITERATURE_TPC_SURVEY.md`) would be sensitive to it.
4. The strip end adds a clean probe. At the end of a straight track there is a strip with no landing charge except through footprint or coupling (method 6). For a straight track the end strip's expected Gaussian tail is calculable, with the position fixed by the M3 or by the bulk of the track. The excess beyond that tail, in amplitude and with the shape of the end strip's neighbour, is k.
5. The sign of tan θ flips which strip is early, so an asymmetric response (e.g. a true delay or a chip-order effect) can be tested by comparing the two track directions.

### Gaps
- No paper was found that separated the capacitive and diffused contributions on the same inclined-track strip experimentally. Scholer's sharing-off simulation is the nearest example and its cause was left open, per the existing survey.
- Published analyses use chamber-set angles; none has an external telescope per-track slope for this test, as MX17 does.

## What is the standard statistical way to determine a coupling constant, and what is its degeneracy with the same-time Gaussian footprint?

### Takeaway
Add a kernel term to the forward model and profile it with a global likelihood over depth charges, with k, σ0, D and the template as nuisance terms, then report the k-σ0 correlation. I found no MPGD paper that does this. The toy shows a strong correlation that the sub-strip phase reduces but does not remove, and σ0 absorbs a real k with a small χ² cost per track, so the discrimination depends on how many tracks and on the sub-strip phase coverage.

### Cited Findings
- No published forward-fit determination of an inter-strip coupling constant for MPGD was found; the closest are the empirical correction scan by [Flierl](https://edoc.ub.uni-muenchen.de/22198/1/Flierl_Bernhard.pdf), the T2K waveform fit ([arXiv:2106.12634](https://arxiv.org/abs/2106.12634)), and the [Belle II η parametrisation plan](https://indico.mpp.mpg.de/event/3730/contributions/8295/attachments/6774/7456/8th_VXD_Workshop_10-09-15.pdf). This is a negative result of a limited search.
- The MX17 model records σ0 = 0.30 mm "prompt footprint: avalanche + induction, a readout constant", and retired the copy-amplitude kernels — `docs/PARAMETERS.md` (repo). `CLAUDE.md` records an earlier case where the cosmic χ² "is flat in that direction" for kernel c2/c1, a precedent for weakly constrained directions. Y-view RC is a separate kernel.

### Inferences
Recommended form, all on existing data:

W_i(t) = Σ_k q_k · f_i(σ(z_k)) · T(t−t_k) + k0 · Σ_{n=i±1}(W⁰_n − W⁰_i) + k1 · τ · Σ_{n=i±1} dW⁰_n/dt

- Include k0 and k1 as **global** per chamber and per condition (matching the project rule that anything calibrated is per detector and run condition), with q_k profiled per event. Use a non-negative constraint on k0, or report a bounded profile.
- Report the 2D profile in (σ0, k0), not a single best fit, since σ0's "SPS-measured" value itself rests on a model with k = 0. A floated σ0 with fixed k would hide the degeneracy.
- Add k0 as a free parameter to the existing pass only in a small subset first. The extended term is linear in amplitude, so profiling q_k is unchanged in structure.
- The test for k = 0 is a likelihood ratio, with the toy MC (the g4_digi / pass re-simulation route) used to calibrate because k0 ≥ 0 sits on a boundary. Inject known k into the re-simulation and check that the fit returns it, plus the bias on angle. This is a closure test and has a clear truth.

**(Toy)** Single inclined track, Fisher matrix of (σ0, k0) with 20 free q_k, for three sub-strip phases (track starting offset 0, 0.2, 0.39 mm; true σ0 = 0.3, k = 0.03):

| tan θ | σ(k0) per track | ρ(σ0, k0) |
|---|---|---|
| 0.05 | 0.013 / 0.019 / 0.028 | −0.66 / −0.83 / −0.93 |
| 0.15 | 0.023 / 0.037 / 0.061 | −0.80 / −0.92 / −0.97 |
| 0.3 | 0.027 / 0.046 / 0.065 | −0.74 / −0.89 / −0.96 |
| 0.6 | 0.028 / 0.048 / 0.068 | −0.67 / −0.94 / −0.97 |

The correlation is strong, and worst for the track phase on a strip edge; the phase dependence supports method 1. With true k = 0.05 and σ0 = 0.30 fit with k fixed to 0, the σ0 fit returns 0.35-0.38 mm, close to the variance-matching estimate of 0.388. The χ² excess per track is 0.5-5, rising to 15 for tan θ = 0.05 at the strip centre, so a modest number of tracks could detect it if the template and D are correct to better than that excess. With template and D errors of comparable size this would be masked (this was not tested in the toy).

Other parameters likely to be involved, from the MX17 model: σ0, D (via the depth dependence), the template shape (a coupling kernel mixes shapes, and a wrong template can imitate k1), the depth discretisation and (on Y) the RC kernel. Hence the recommended sequence: X view first (no RC), near-mesh depth slice, multiple sub-strip phases, then the full fit.

### Gaps
- The toy ignores template error, real noise correlations and real q_k fluctuations (Landau clusters), which all degrade the discrimination.
- Whether the existing pass's `fc_sigma0_trade` study already covers some of this degeneracy was not reviewed; it was found only as a reference in `docs/PARAMETERS.md`.
- Inaccessible or not retrieved: French et al. (2001), CMS NOTE-2007-027, Bernardini, the primary ATLAS NSW ghost-hit notes, Moore et al. (IPC autocorrelation; seen only through the snippet of an H2RG paper).
