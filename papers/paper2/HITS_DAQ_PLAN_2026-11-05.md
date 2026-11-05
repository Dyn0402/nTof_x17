# Paper II, last section: what a hits-only readout can achieve — analysis plan (2026-11-05)

Source: Dylan's hub note `archive/Paper II Notes.md` ("Comparison with hits based DAQ"), read against everything the
repo did with hits from March to October 2026. Companion to `OUTLINE_PLAN_2026-11-05.md` (this replaces its §6/§7 rows
and work items R1–R5, V1). No new data: everything below is computation on the stored waveforms or simulation.

## 1. What the section has to show

1. A hit readout (one time + one amplitude per strip, VMM-style) reconstructed **naively** reads tracks too steep
   and depth-wrong, and the bias is set by the sharing physics, not by the hit algorithm.
2. The **best corrections we can build on hits alone** shrink the bias but leave a resolution gap to the waveform fit
   (or close it — that is the open result, see H2c).
3. **How the gap depends on the resistive layer**: as the sharing goes to zero the hits chain approaches a classic
   µTPC; how far it gets, and what still limits it (prompt footprint, shaping), is the generalisation.

Two arms make the claim, and they need each other:

- **Arm D (data):** emulate hit DAQs on the real DREAM waveforms of pass 1. Real noise, real kernel, real reference
  (M3 on the bench, the stage angle at SPS). Headline numbers come from here. Cannot change the detector or the
  front-end bandwidth (DREAM shaping, 60 ns sampling).
- **Arm S (simulation):** synthetic waveforms with known truth and a physically parameterised kernel, sampled finely,
  with the same hit algorithms and the same reconstructions. Gives the resistivity / shaping / threshold scans. Only
  trusted after it reproduces Arm D at the measured parameters (gate S0).

## 2. What already exists (the past year, condensed)

Full inventory came from two surveys; the parts this plan uses:

| what | where | state |
|---|---|---|
| **H0 vs W, same events, five chambers, pass 1** — angle s68 H0 2.0–2.6° vs W 0.73–1.56°, H0 implied v 31–62 µm/ns (det3 46/51 vs W ~36), position 550–805 vs 430–620 µm, efficiency within 1–5 pts | `rerun/paper1_floor/hits_arm.py`, `results/fleet.json`, `figures/fleet.pdf` | **done**; note H0 already gets a per-row M3-calibrated slope scale k, i.e. it is really "H0 + global remap" |
| Production hit finder: boxcar MF gate (5 samples) at 5σ, time = 30 % leading edge, also time-of-max, ToT, thresholds crossings | `~/CLionProjects/mm_strip_reconstruction/waveform_analysis/src/WaveformAnalyzer.cpp`; regenerated hits in `rerun/rehit/` | the "DREAM-internal" hit algorithm, already a variant |
| Estimator independence: threshold, CFD 50 %, rise fit, deriv. max, MF t50, LE 20 % all compress (v 42–50 vs 36.6; residual +40 ns at mesh, −100…−130 ns at cathode) | `mx_june_cosmic_qa/24_waveform_investigation.py`, `waveform_first_threading/04_mf_ladder.py`, `05_estimator_compare.py` | det3 July only, R&D caches (gone) |
| Hit-list slope estimators (OLS, LE, peak, core-only, endpoints) all 28–30 µm/ns on the old basis | `mx_june_cosmic_qa/22_strip_timing_and_estimators.py` | hits only, runs |
| Additive tan offset / ridge (tanθ ± w/z), geometry v | `mx_june_cosmic_qa/13`, `14`, `20`, `21`, `23` | hits only; superseded physics picture |
| H2a slope remap α ≈ 0.86–0.89: σ 1.51/1.86°, threading 54/52 % | `waveform_first_threading/16_corrected_ladder.py` | det3 July |
| H2b unsharing (banded (1+c1E±1+c2E±2)⁻¹ on waveforms) + additive tan calib + head-on LDA tagger + hybrid: det3 1.75/1.86°, bias halved; same-event gate 2.42/2.60° vs W 1.20/1.14° | `mx_june_cosmic_qa/26`–`29`, `31`, `33`, `34`, `36` | det3 July, hard-coded det3 constants; **unsharing needs waveforms** |
| "hits6" hybrid on hit features only | `mx_june_cosmic_qa/35_hybrid_drift_scan.py`, `ntof_tracking/validate_transfer_bench.py` | hits only; does not transfer to near-normal beam tracks |
| Hits-readout toy: CR-RC² shaper, prompt XT, threshold 100 ADC, CFD | `mx_june_cosmic_qa/25_signal_formation_toy.py` | no RC; tuned to det3 hits |
| Digitiser into the production chain: per-electron diffusion, Polya, bundle template, **real quiet-trigger noise**, CM subtraction, hit emulation (peak ≥ 5σ) | `ntof_cosmics/g4_digi/digitise.py` | **uses the retired c1/c2 copy kernel** — needs porting to `build_matrix_rc` |
| Floor toy (ws_floor) + realism ladder L0–L8 + residual bootstrap | `rerun/model_audit/ws_floor/floor_toy.py`, `followups/fd_floor_realism/ladder.py`, `rerun/paper1_floor/boot_*.py` | X only, no RC; reaches 31–46 % of measured resolution, χ² 5–11× below data |
| First-principles response at explicit ρ_s (0.5–5 MΩ/□), DREAM shaper, measured noise, ZS, decoded_root writer | `~/PycharmProjects/MX17_Geant/response/` (`solver/`, `digitizer/`, `dream/`) | only tool where ρ_s is literal; ±1 delay +105–135 ns vs 45–60 data, so a cross-check, not the engine |
| ZS emulation on raw data (volume / survival, not tracking) | `mx_july_beam_qa/26_zs_sim_extract.py`, `sps_beam_test_26/analysis/cloud_basics/zs_emulate.py` | reusable for the threshold axis |
| Physical kernel: σ²(u)=σ0²+Dp²u+…, Y only `+2·rc_D_y·s`; rc_D_y 2.2–6.8e-4 mm²/ns, `rc_D_x = 0`, σ0 0.30 mm; D_rc = 1/(R_s c) | `wft/model.py:build_matrix_rc`, `docs/PARAMETERS.md` §2, `sps_beam_test_26/analysis/cloud_basics/FINDINGS.md` §3 | the dials for Arm S |

Lessons that constrain the design: never take v from a prior (LESSONS / R9); a hits "v" is not a velocity; corrections
fitted on det3 do not transfer (hybrid on beam: slope 0.03–0.10); unsharing works in ensemble but amplifies noise per
event; the old angle constants were on the 34 µm/ns basis.

## 3. Hit-DAQ variants to emulate (the "V" axis)

Every variant reduces a strip's waveform to at most one (or a few) `(t, A[, ToT])` per strip per event. Same events
throughout.

| id | modelled on | time | amplitude | other logic |
|---|---|---|---|---|
| V0 | DREAM internal (our `combined_hits`, H0) | 30 % leading edge, MF gate 5σ | parabolic peak | none |
| V1 | VMM, time-at-peak mode | peak time | peak (PDO, 10 bit) | threshold, **neighbour enable** (±1 read below threshold), one hit per channel + dead time |
| V2 | VMM, time-at-threshold mode | threshold crossing | peak | as V1 |
| V3 | ToA + ToT ASIC (TIGER-, Timepix-like) | threshold crossing | ToT (no peak) | as V1, no neighbour enable |
| V4 | ideal per-strip timing (upper bound for any single-time ASIC) | CFD 50 % / template-fit t0 on the full waveform | template amplitude | none |

V4 is what a perfect per-strip timer could give; if even V4 + the best correction leaves a gap, the gap is in the
data format (one time per strip), not in the ASIC.

In Arm D, V1–V3 run at DREAM shaping: the per-strip waveform is upsampled through the measured template (a per-strip
template fit gives a continuous pulse), then the ASIC logic is applied with VMM-like timing granularity. The VMM's own
shaping (25–200 ns peaking) cannot be reconstructed from 60 ns samples, so **shaping is an Arm S axis only**. VMM3a
constants (peaking options, ADC/TDC ranges, dead time, neighbour logic) to be taken from the VMM3a documentation and
cited, not from memory.

## 4. Reconstruction tiers on hits (the "H" axis)

| tier | what | needs | from |
|---|---|---|---|
| **H-naive** | line t(x) through the hits, angle with the **true v** (W's fitted v / Magboltz), earliest hit = position | hits | new, trivial; this is "show the bias" |
| H0 | as H-naive with a global slope scale k calibrated on a reference (current `hits_arm.py`) | hits + reference | done |
| H1 | H0 with the best per-strip time per variant (amplitude-weighted, core-only, endpoints) | hits | `22`, `05` |
| H2a | **angle-dependent** remap: binned tan_meas → tan_true lookup trained on one half, tested on the other; upper bound of any slope-only correction | hits + reference | extends `16` |
| H2b′ | **hits-only unsharing**: deconvolve amplitudes with the kernel, correct each strip time by its neighbours' amplitude-weighted delay, refit; + head-on tagger on hit features (hits6) | hits + kernel | rewrite of `26`–`28`, `33`, `35` on hit lists (the July version used waveforms, so it was never a hits-only tier) |
| **H2c** | forward model **through the hit emulator**: per event, generate the waveforms from (x0, tan, t0, q-profile) with `build_matrix_rc`, apply the same V algorithm, and fit the predicted hit list to the observed one (χ² on t and A, censoring for strips below threshold) | hits + kernel + emulator | new (R3) |
| W | pass-1 waveform fit | waveforms | done |

H2c is the fair "best a hit readout can do": it uses all the physics W uses, only the information a hit DAQ keeps.
If H2c gets close to W, the paper's message becomes "the model is needed, waveforms help"; if not, "the waveform is
needed". Either is a result. Compute is not a constraint.

Fairness rules: every tier sees the same events (interior_good, W ok, ≥ 1 hit fit), the same reference conventions
(`fd_s68`, tilt-map removal, |Δ| ≤ 5° window, bootstrap errors), and any reference-trained constant is
cross-validated by event-id parity. Report reference-free tiers (H-naive, H2b′, H2c) separately from
reference-trained ones (H0, H2a), since a real experiment with only hits may have no reference.

## 5. Metrics and benchmark

Per chamber, per view, per tier × variant:

- angle s68 vs |tan| (5 bins incl. head-on < 5°), signed bias vs angle, implied v vs angle (the compression line);
- ladder residual vs drift depth (per-strip time − reference-predicted time), the paper's "why hits fail" figure;
- threading (fraction within 1 mm over the full gap) and depth-resolved |cluster − reference|;
- mesh position core σ, efficiency within 5 mm;
- SPS det4 known angle (25.4° mount, X view): |tan| bias reference-free (R7);
- cost per plane (R10).

Main figure: s68 and bias vs angle, tiers as lines, W as the reference band, five chambers as small multiples or
det3 with the fleet as a table. Second: ladder residual vs depth, V0–V4 overlaid.

## 6. Arm S: the simulation and the generalisation scan

**Engine:** port `ntof_cosmics/g4_digi/digitise.py` from the copy kernel to `build_matrix_rc` (σ0, Dp, rc_D_y,
rc_D_x), keep its real-noise injection and CM subtraction, and add:
- a 1 ns internal time grid with an analytic CR-RCⁿ (or the MX17_Geant DREAM `shaper.py`) impulse response, checked to
  reproduce the measured DREAM template at the bench setting, so peaking time becomes a dial;
- the V0–V4 emulators of §3 (one module, shared with Arm D);
- straight tracks from `straight_steps()` over the bench angle distribution, plus the SPS geometry.

**Gate S0 (must pass before any scan is quoted):** at det3 bench parameters, Arm S reproduces Arm D for V0/H0 and W:
H0/W s68 ratio, H0 implied v / true v (1.28 on det3 X), the ladder residual curve (+40 / −100…−130 ns), within stated
tolerances (proposal: ratio and v within 10 %, residual curve within 20 ns). The floor toy only reaches 31–46 % of the
measured W resolution, so quote Arm S as **ratios and biases relative to W at the same point**, never as absolute
resolutions; if needed, add the residual bootstrap (`boot_extract.py`) to bring the absolute level up.

**Scan axes** (one at a time around the det3 point, then a 2-D map of the two that matter):

| axis | range | physical meaning |
|---|---|---|
| rc_D (Y) | 0, 0.25×, 0.5×, 1× (3.8e-4), 2×, 5×, 10× | resistivity × capacitance; D = 1/(ρ_s c′) maps to MΩ/□ using MX17_Geant's solved c′ (0.5–5 MΩ/□ scan) — the x axis Dylan asked for |
| rc_D_x | 0 vs = rc_D_y | resistive **pattern**: strips parallel to one view (ours) vs a continuous sheet / DLC |
| σ0 / pitch | 0.07 mm (induction floor) … 0.30 mm (ours) … 0.45 mm | prompt footprint: insulator thickness, pitch |
| peaking time | 25, 50, 100, 200 ns (+ DREAM's) | front-end shaping (VMM options) |
| threshold | 3–7σ; neighbour enable on/off | R4; also run in Arm D |
| drift v / gap | ours, ± 30 % | how many strips a pitch-time spans |

Expected deliverable: s68(H-best)/s68(W) and bias vs ρ_s, with the metallic (rc_D → 0, σ0 → floor) limit marked.

## 7. Work order

| # | item | size | depends |
|---|---|---|---|
| P1 | Hit-emulator module (V0–V4) on numpy waveforms, unit-tested on real `decoded_root` (V0 must reproduce `combined_hits` times/amplitudes on the same strips) | M | – |
| P2 | Arm D extraction: run P1 on the pass-1 interior_good events, five chambers + SPS det4 (condor, outputs on EOS; local scratch `$WORK_DATA/paper2_hits/`) | M | P1 |
| P3 | H-naive, H1, H2a on P2 + the benchmark scorer (extend `hits_arm.py` into a tier scorer) → first figures: ladder residual vs depth, bias/s68 vs angle | M | P2 |
| P4 | H2b′ hits-only unsharing + hits6 tagger, constants refit per chamber on half the events | M | P3 |
| P5 | Port g4_digi to `build_matrix_rc` + shaper + P1; gate S0 at det3 | L | P1 |
| P6 | H2c: emulator-in-the-loop fit (start on Arm S where truth is known, then Arm D) | L | P1, P5 |
| P7 | Scans of §6 in Arm S; threshold/neighbour axis in both arms | M | P5 (P6 for the H2c line) |
| P8 | `make_report.py` → `report.html` + paper figures via `figures.toml`; numbers into the §6/§7 text | S | P3–P7 |

P1–P3 alone give a complete (if pessimistic) section; P5/P7 add the generalisation; P6 decides the headline message.

## 8. Limitations to state, not chase

- Arm D's ASIC emulation runs at DREAM shaping and 60 ns sampling: it emulates the logic of a VMM, not its analogue
  front end or noise. Shaping effects come only from Arm S.
- The production model has no X RC spread (`rc_D_x = 0`); the measured X ±1 asymmetry and the delayed X copy are not
  in Arm S, so Arm S under-represents X sharing. The X hits bias in data is nonetheless as large as Y's (H0 X 2.0–2.3°).
- Arm S is validated by ratios (S0), not absolutely; 47–75 % of W's measured variance is not in any simulation we have.
- ρ_s mapping relies on MX17_Geant's c′; that simulation's ±1 delay disagrees with data by ×2, so the MΩ/□ axis is
  approximate (quote it as "D_rc, with an indicative ρ_s scale").
- Lowering the sharing is not free: the resistive layer is there for discharge protection and its spreading sets rate
  capability. The section reports the reconstruction side only.
- Reference-trained corrections (H0, H2a) assume a calibration source a hit-only experiment may not have.
