# Front-end "hit" definitions and µTPC practice: literature survey

Purpose: before emulating other front ends on our DREAM waveforms, record exactly how each one turns a strip pulse into a hit.
Rules: every number carries a reference tag [R#] (list at the end). "Not found" means I looked and the sources I could read do not say.
"Datasheet" marks chip capability; "Used in" marks what a cited analysis actually ran. Retrieved 2026-11-05; most PDFs were read as full text, items marked (abstract/secondary) were not.

## Summary table

| Front end | Chain | Per-strip time estimator | Amplitude | Threshold / neighbour / dead time | µTPC use and reported performance | Resistive? |
|---|---|---|---|---|---|---|
| VMM3a | CSA + 3rd-order shaper, 25/50/100/200 ns peaking; discriminator + peak detector + TAC; **not sampled** [R1] | TAC ramp starts at **peak** (designed mode) or at **threshold crossing** (added later); stops on next BC-clock edge; coarse BCID (12-bit) + fine TDO (8-bit) [R1][R2] | Peak value, 10-bit ADC (~8 effective) [R1][R2] | 10-bit global DAC + 5-bit per-channel trim; neighbour logic forces 1st-order neighbours (and across chips) to be read; dead time ~250 ns [R1] | NSW MM test beam, 28°: centroid 550 µm; µTPC core 211-275 µm, with tails 318-374 µm [R3] | Yes (NSW, 5 mm gap) |
| APV25 (SRS) | CR-RC, 50 ns peaking, 128 ch, analogue pipeline; 25 ns samples, 21-27 per trigger [R4][R5][R6] | Offline fit of the samples: Fermi-Dirac rising edge (t at half height/inflexion), linear fallback [R5][R7][R8] | Max sample (Qmax) [R5] | None on chip for hit finding; SRS applies pedestal subtraction/ZS; 25 ns clock jitter vs trigger [R5][R9] | Floating-strip MM, 6 mm: angle +3/-2 deg single plane [R8]; LMU/NSW slides ~2 deg at 20 deg (not peer reviewed) [R10]; GEM 5 mm [R5] | Both (see text) |
| SAMPA | CSA + shaper (160 ns TPC), 10-bit ADC at 10 (up to 20) MS/s, DSP [R11][R12] | Sampled waveform; TPOT: "signal time" from samples, no per-strip µTPC [R12] | Sampled | DSP baseline/ZS details not found in primary text | sPHENIX TPOT: cluster position only, 390/420 µm core [R12] | TPOT yes (resistive strips, 3 mm gap) |
| TIGER | Two branches: T-shaper 60 ns, E-shaper 170 ns; two discriminators; TDC + S/H [R13] | ToA at T-branch leading-edge threshold; TDC 50 ps bins; jitter < 5 ns rms at 3 fC [R13] | ToT or S/H (E-branch), same Wilkinson ADC; 10-bit output [R13][R5] | Separate T and E thresholds with per-channel LUT equalisation; self-triggered or trigger-matched [R13][R5] | BESIII CGEM: µTPC developed with APV25; TIGER validated with charge centroid only [R5] | No (GEM, metallic) |
| STS/MUCH-XYTER2 | CSA, fast shaper + leading-edge discr., slow shaper (80-280 ns) + 5-bit ADC [R14] | 14-bit timestamp, 3.125 ns, time walk corrected offline [R14] | 5-bit | Self-triggered; 250 khit/s with pulsed reset [R14] | No µTPC use found | Not applicable |
| VFAT3 | CSA + shaper (25-100 ns) + CFD, **binary** [R15][R16] | Hit bit in 25 ns BX frame; CFD zero crossing, simulated 4.98 ns at 50 ns peaking [R16] | None (binary) | 8-bit global arming DAC + 6-bit per-channel DACs [R16] | No µTPC use found | No (GEM) |
| Timepix3 | 55 µm pixel, per-pixel ToA + ToT, data driven [R17][R18] | ToA, 1.56 ns, with ToT-based time-walk correction [R17] | ToT | Per-pixel threshold ~800 e-; sparse readout [R17][R19] | GridPix TPC, true 3-D; diffusion-limited [R17] | No (InGrid, metallic) |
| ERAM (AFTER) | Sampled waveforms, 40 ns sampling, 200 or 412 ns peaking [R20] | Waveform maximum per pad; waveform fit only for RC mapping [R20] | Waveform maximum | Pad threshold; not a µTPC | 200-250 µm, resistive pads [R20] | Yes (DLC) |

## VMM3a (ATLAS NSW Micromegas and sTGC; RD51 SRS-VMM)

**Signal chain (datasheet).** 64 channels; three-stage charge amplifier, third-order shaper, peaking time 25/50/100/200 ns, gain 0.5-16 mV/fC in eight steps; a baseline holder stabilises the output [R1]. The pulse is never sampled: a discriminator, a peak detector and a time detector follow the shaper [R1].

**Per hit.** The peak detector stores the peak height; the TAC ramp (60/100/350/650 ns) "starts either at threshold crossing or at the time of the pulse peak" and stops at the next edge of the externally supplied BC clock [R1]. Time-at-threshold "was added to VMM3a"; the TAC "was designed to be used at peak finding" [R1]. Output: 10-bit amplitude ADC (effectively ~8 bit), 8-bit fine time, 12-bit Gray-code BCID, 6-bit channel; 38 bits per hit, held in a 4-deep FIFO in continuous mode [R2]. Time-at-peak walk was measured and "can be corrected off-line"; time-at-threshold walk is large but stable, with sub-ns resolution at 200 ns peaking [R1]. The CKBC period (25 ns in ATLAS) sets the coarse bin; TDO calibration is per channel from test pulses [R2][R3].

**Threshold logic.** Sub-hysteresis discriminator, 10-bit global DAC plus 5-bit per-channel trim in 32 steps of about 1 mV; no common-mode subtraction (baseline holder only) [R1]. Neighbour logic forces the first-order neighbours (also on neighbouring chips) to be recorded whether or not they exceed threshold; hits below threshold carry a flag bit [R1][R2]. Minimum dead time about 250 ns (10-bit ADC conversion), so up to 4 Mhit/s per channel; one hit per channel in an ATLAS L0 window [R1][R21]. ART outputs only the address of the first fired channel in a BC, with no charge or fine time, and feeds the trigger [R1][R22]. A known bug: time-at-threshold can latch ("channel locks") [R1]; the L0-mode FIFO-read advance does not work, which does not matter for ATLAS [R2].

**Used in.** NSW SM2 module, SPS H8, 2018: VMM3, peaking time 100 or 200 ns, gain 9 mV/fC, neighbour logic ON, threshold 6x noise RMS (~5000 ENC), Ar:CO2 93:7 [R3]. Saturated pad signals are delayed because of the peak-finding function (VMM3a with pads, 2025) [R23], so time-at-peak is not neutral for large pulses.

**µTPC practice.** Resistive-strip Micromegas, 5 mm drift gap, strip pitch 425/450 µm, strip width 300 µm [R3][R24]. Hough transform on (strip, time) to reject outliers, then a straight-line fit, position at mid-gap [R3]. At 28° the strip multiplicity peaks at 6, drift-time spread ~110 ns; thresholds at 6x RMS lose strips (62-74 % of hits pass; narrow-Gaussian residual 211-275 µm, two-Gaussian 318-374 µm against 550 µm for the centroid) [R3]. Corrections in the literature: the first/last strip of a cluster is dropped when its PDO is below half of its neighbour's (VMM, 10x10 cm2 prototype, Rinnagel) [R25]; the same idea with a 25 % cut was used with APV25 [R8]; an older Roma Tre proceeding scales first/last-strip charge ratios by cluster length (reported via a search summary only, original not read). Angular resolution on the 10x10 cm2 prototype, 5 mm gap: +6/-4 deg with VMM3 versus +4/-2 deg with APV25, "probably due to a non-optimal time calibration" [R25]. Capacitive coupling between adjacent channels biases the reconstructed angle high [R25][R8]. The combination "µTPC + centroid" is under study [R3]. **Production Run 3 reconstruction uses the centroid**; time-based clusterisation is "being investigated"; observed resolution 200-700 µm against a 100-150 µm design [R26]. Which time mode (peak or threshold) NSW production uses: **not found**.

## APV25 (SRS) and SAMPA

**APV25.** 128 channels, CR-RC shaper of 50 ns peaking time, gain 4.6 mV/fC, 192-column analogue pipeline [R4]; 25 ns sampling, 27 samples per trigger in the GEM work, 21 samples in the NSW Module-0 work [R5][R6]. Digitisation and online pedestal subtraction by SRS/mmDAQ [R5].

**Strip to one time.** All analyses fit the rising edge offline. GRAAL (BESIII GEM): Fermi-Dirac Q(t) = Q0 + Qmax/(1+exp(-(t-tFD)/sigmaFD)), initialised from the first three samples and the 10 % point, sigma 12.5 ns; time = inflexion point; linear fit if it fails; amplitude = max sample [R5]. NSW Module-0: Fermi-Dirac over 21 samples, half-height t0 [R6]. LMU floating-strip: inverse Fermi fit, three-point (10/50/90 %) variant, weights from pulse height and rise time, Hough step, reject chi2 > 2.5, drop first/last strip below 25 % of its neighbour [R8]. Fine granularity resistive Micromegas (pads): Fermi-Dirac rise, time resolution 5.7 ns (two chambers, 10.5 cm/µs); the 25 ns trigger jitter cancels in cluster-time differences [R7]. The trigger is not synchronous to the APV clock, which adds jitter unless the trigger is injected into a channel [R5][R10].

**Performance.** Floating-strip (metallic) Micromegas, 6 mm gap, 23 MeV protons, Ne:CF4 80:20: single-plane angular resolution +3/-2 deg; µTPC improves position above 10 deg at drift field below 500 V/cm; reconstructed angle biased high at low drift field by capacitive coupling [R8]. BESIII triple-GEM (metallic, 5 mm gap): µTPC point = (strip position, t x v_drift), error pitch/sqrt(12) in x with a charge-dependent term, line fit, position at mid-gap; resolution crosses the centroid between 10 and 20 deg; "one bad µTPC point" degrades the fit; merged resolution 100-130 µm for tilted tracks; measured v_drift within 10 % of Garfield++; detector time resolution ~12 ns [R5]. LMU slides: ~2 deg at 20 deg, ~3 deg at 26.5 deg, dominated by clock jitter (conference slides, method unspecified) [R10].

**SAMPA.** 32 channels, CSA + shaper, 10-bit ADC, 10 MS/s nominal (20 MS/s possible), TPC shaping 160 ns at 20 or 30 mV/fC, 300 ns for MCH, 130 nm CMOS (abstract/secondary summaries) [R11]. sPHENIX TPOT (SAMPA v5, bulk resistive Micromegas, 3 mm drift gap, strip pitch 1 mm phi / 2 mm z zigzag): 160 ns shaping, 20 MHz sampling, 1024 ADU at 2.15 mV/ADU, gain 20 mV/fC [R12]. TPOT reports cluster sizes 3.5 (phi) and 2.6 (z), core residual 390 µm (phi) and 420 µm (z), and uses signal time versus strip only to study drift velocity/Lorentz angle, **not** a µTPC fit [R12]. DSP functions of SAMPA (baseline correction, zero suppression): not verified from a primary text.

## TIGER and STS-XYTER

**TIGER (BESIII CGEM).** 64 channels, 110 nm. One preamp feeds two shapers: T-branch 60 ns peaking for low jitter, E-branch 170 ns peaking for charge; two discriminators with separate programmable thresholds [R13]. (A search summary quoted 50 ns for the T-branch; the paper says 60 ns.) Time: TDC from four TACs and a Wilkinson ADC, 50 ps binning at 160 MHz clock, quantisation error < 50 ps rms; measured jitter < 5 ns rms at 3 fC; the requirement is ~5 ns to "enable the micro-TPC mode" [R13]. Charge: ToT or the S/H circuit sampling the E-branch inside a user window, digitised by the same ADC; 3-50 fC range [R13]. Thresholds equalised per channel by S-curve scans and a look-up table [R13]. Modes: trigger-less (every hit above threshold sent out) and trigger-matched; output decoded in 10 bits [R13][R5]. Dead time, hit rate limit and ADC effective bits: not found in the sources read (60 kHz/channel is the stated requirement [R13]).
**µTPC with TIGER.** The µTPC method was developed and validated with APV25 on planar triple-GEMs; the TIGER test beam used two chambers at normal incidence and only reported the charge centroid (cluster size vs charge, CC resolution), because the µTPC "does not work properly with orthogonal tracks" [R5]. A published µTPC resolution with TIGER: **not found**. Combined CC + µTPC target is 130 µm at 1 T (design value, not a measurement) [R13][R5].

**STS/MUCH-XYTER2.** 128 channels + 2 test channels; fast shaper with leading-edge discriminator giving a 14-bit timestamp (3.125 ns); slow shaper (80-280 ns) + continuous-time 5-bit ADC and peak detector; self-triggered; 250 khit/s per channel with pulsed reset; 0-15 fC (STS) or 1-100 fC (MUCH, GEM); time walk corrected offline; time accuracy requirement < 10 ns (GSI/AGH conference slides) [R14]. Use for µTPC reconstruction: none found; it is a design for silicon strips and GEM pads.

## VFAT3 and Timepix3 / GridPix

**VFAT3 (CMS GEM).** 128 channels of CSA + shaper + constant-fraction discriminator; the **binary** output is synchronised to the LHC clock and split into a fixed-latency trigger path and a tracking path stored in SRAM (128 channels x 1024 BX) until L1 [R15][R16]. Peaking time programmable between 25 and 100 ns; arming comparator threshold from an 8-bit global DAC with 6-bit per-channel DACs for the arming and zero-crossing comparators; CFD residual time walk < 1 ns per channel against tens of ns for the plain arming comparator (8-channel prototype); rate capability 400 kHz at 100 ns to 1 MHz at 25 ns; simulated GEM time resolution 4.98 +- 0.16 ns at 50 ns peaking with 100 +- 5 ns latency [R16]. Per hit the record is one bit per BX per channel: no amplitude, and sub-BX time was **not found**. No µTPC use found; the BX binning (25 ns) is far coarser than a 5 mm gap needs.

**Timepix3 / GridPix.** 256 x 256 pixels of 55 x 55 µm2, 130 nm; per pixel ToA (14-bit 40 MHz register refined by 4 bits to 1.5625 ns) and ToT (10-bit), data-driven sparse readout (abstract level) [R19]; 640 MHz TDC, noise ~70 e- [R17]. GridPix TPC test beam (2.5 GeV e-): threshold 800 e-, z from ToA with v_drift 78.86 µm/ns, time-walk correction as delta z = c1/(ToT + t0) with ToT > 0.15 µs cut; drift-direction resolution sigma_z^2 = (tau v_drift)^2/12 + sigma_z0^2 + D_L^2 (z - z0), hits split at ToT = 0.60 µs; diffusion dominates both planes [R17][R18]. Metallic (InGrid), no strips, so no shared-charge or neighbour-timing problem: this is the "true TPC" limit.

## AGET/AFTER: closest prior art (T2K ERAM)

T2K ND280 upgrade resistive Micromegas (DLC anode, 200 kOhm/sq, pads, AFTER ASIC, 40 ns sampling, 200 or 412 ns peaking): position from the pad response function on the **waveform maximum** per pad (200-250 µm over the drift length); a full analytical waveform fit (leading pad with the electronics response I(t), neighbours with I(t) convolved with the resistive charge-dispersion Q(t) containing RC) is used to extract the RC map, not to reconstruct each hit [R20]. So it is a prior waveform model of resistive charge sharing, not a per-hit waveform reconstruction.

## Which algorithm families exist

1. **Threshold-time + peak** (VMM3a time-at-threshold; TIGER ToA + E-branch; STS-XYTER; Timepix3 ToA + ToT). Time is the leading-edge crossing and is amplitude dependent (time walk); amplitude-corrected offline. Amplitude-dependence is repaired with ToT or peak height (Timepix3 [R17], XYTER [R14], VMM [R1]).
2. **Peak-time + peak** (VMM3a default). Time of the pulse maximum, walk smaller and stable but shaper-dependent; peak finding delays saturated pulses [R1][R23].
3. **ToA + ToT / S/H** (TIGER, Timepix3): family 1 with a charge by width or sample-and-hold.
4. **Sampled + per-strip fit** (APV25, SAMPA, DREAM, AFTER). Per-strip time from a Fermi-Dirac rising-edge fit (half height / inflexion) or a maximum. This is the family closest to ours, but the estimator is still per strip, so the neighbour-aggregation problem in `RECONSTRUCTION_BASIS.md` applies as well. Handling in the literature: drop first/last strips below 25-50 % of neighbours [R8][R25], Hough/outlier rejection [R3][R8], charge-dependent x errors [R5].
5. **Binary** (VFAT3): BX-resolved hit only; no µTPC.
6. **Waveform model fit** (ERAM RC mapping [R20]; ours): the only family that models the shared charge explicitly.

Resistive-strip µTPC with a published per-strip chain exists only for NSW Micromegas (VMM, APV25) and, with pads, ERAM; GEM and floating-strip results use metallic strips, so their capacitive-coupling bias is a different effect from resistive spreading.

## What I could not verify

- VMM3a time mode used by NSW production; a published first/last-strip rule from the NSW Run 3 chain (the 2014 Roma Tre proceeding was seen only through a search summary).
- SAMPA DSP and hit-finding details from a primary text; Timepix3 bit widths beyond the abstract; STS-XYTER from a journal paper (slides only).
- Any µTPC resolution with TIGER, STS-XYTER or VFAT3.
- Journal references for arXiv:1808.09752 and arXiv:2106.12634 (NIM A, not confirmed); drift gap and resistivity of the Rinnagel 10x10 cm2 prototype.

## References

- [R1] G. De Geronimo, G. Iakovidis, S. Martoiu, V. Polychronakos, "The VMM3a ASIC", IEEE Trans. Nucl. Sci. 2022, doi:10.1109/TNS.2022.3155818 (accepted manuscript, CERN copy).
- [R2] D. Pfeiffer et al., "Rate-capability of the VMM3a front-end in the RD51 Scalable Readout System", NIM A, arXiv:2109.10287 (2022).
- [R3] P. Scholer (ATLAS Muon), "Test beam results and performance studies of ATLAS Micromegas production modules", J. Phys. Conf. Ser. 1498 (2020) 012053, doi:10.1088/1742-6596/1498/1/012053.
- [R4] E. Terzimpasoglou, Master thesis, Bonn, arXiv:2206.09968 (2022), quoting French et al., NIM A 466 (2001) 359 for APV25.
- [R5] R. Farinelli, PhD thesis, Ferrara, "Research and development in cylindrical triple-GEM detector with µTPC readout for the BESIII experiment", arXiv:1904.06548 (2019).
- [R6] T. Alexopoulos et al., "Construction techniques and performances of a full-size prototype Micromegas chamber for the ATLAS muon spectrometer upgrade", arXiv:1808.09752 (2018).
- [R7] M. Alviggi et al., "Resistive Fine Granularity Micromegas: characterization and performance for different spark protection resistive schemes", JINST (submitted), arXiv:2411.17202 (2024).
- [R8] F. Klitzner, Master thesis, LMU Munich, "Studies of floating strip Micromegas detectors in proton and carbon ion beams with a fast gas mixture" (2016), https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_fklitzner.pdf
- [R9] same as [R7] (jitter cancellation) and [R5] (trigger injected into an APV25 channel).
- [R10] F. Klitzner talk, LMU, 2018-06-20, https://www.etp.physik.uni-muenchen.de/publications/talks/download/180620_fklitzner.pdf (slides; cited via a search summary, numbers not independently checked).
- [R11] J. Adolfsson et al., "SAMPA chip: the new 32 channels ASIC for the ALICE TPC and MCH upgrades", JINST 12 (2017) C04008, doi:10.1088/1748-0221/12/04/C04008; S. Barboza et al., JINST 11 (2016) C02088 (abstracts and secondary summaries only).
- [R12] S. Aune et al., "The sPHENIX Micromegas Outer Tracker", arXiv:2403.13789 (2024).
- [R13] F. Cossio et al., "TIGER: a front-end ASIC for timing and energy measurements", IEEE NSS/MIC 2017, doi:10.1109/NSSMIC.2017.8533025, arXiv:1903.05694.
- [R14] K. Kasinski et al. (AGH/GSI), "STS-XYTER, a 128 channel readout ASIC for silicon strips", FEE 2018 slides, https://indico.gsi.de/event/7361/contributions/32998/attachments/23795/29799/2018.05.25_-_FEE_Kasinski_v3.pdf
- [R15] A. Irshad et al. (CMS), "Irradiation of VFAT3", TWEPP-21, CMS CR-2021/227.
- [R16] D. Abbaneo et al., "Design of a constant fraction discriminator for the VFAT3 front-end ASIC of the CMS GEM detector", JINST 11 (2016) C01023.
- [R17] C. Ligtenberg et al., "Performance of a GridPix detector based on the Timepix3 chip", NIM A, arXiv:1808.04565 (2018).
- [R18] Timewalk correction for the Timepix3 chip, Tsigaridas et al., NIM A 930 (2019), arXiv:1902.00480 (search summary only).
- [R19] X. Llopart, T. Poikela et al., "Timepix3: a 65K channel hybrid pixel readout chip with simultaneous ToA/ToT and sparse readout", JINST 9 (2014) C05013, doi:10.1088/1748-0221/9/05/C05013 (abstract level).
- [R20] D. Attie et al., "Characterization of resistive Micromegas detectors for the upgrade of the T2K Near Detector Time Projection Chambers", arXiv:2106.12634 (2021).
- [R21] M. Rinnagel, Master thesis, LMU, see [R25] (L0 window and 250 ns dead time).
- [R22] ATLAS Muon Collaboration, via S. Francescato, see [R26] (ART).
- [R23] D. Zavazieva et al., "Performance of resistive MPGDs with pad readout coupled to VMM3a ASIC", arXiv:2511.00929 (2025).
- [R24] [R3] and [R26] for NSW geometry.
- [R25] M. Rinnagel, Master thesis, LMU Munich, "Investigations of the VMM readout chip for Micromegas detectors of the ATLAS New Small Wheel upgrade" (2019), https://www.etp.physik.uni-muenchen.de/publications/theses/download/master_mrinnagel.pdf
- [R26] S. Francescato for ATLAS, "ATLAS Micromegas performance studies with LHC Run3 data", JINST (MPGD2024), arXiv:2503.16735 (2025).
