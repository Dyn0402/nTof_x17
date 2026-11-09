# Paper II, last section: what a hits-only readout can achieve — analysis plan (2026-11-05)

Source: Dylan's hub note `archive/Paper II Notes.md` ("Comparison with hits based DAQ"), read against everything the
repo did with hits from March to October 2026. Companion to `OUTLINE_PLAN_2026-11-05.md` (this replaces its §6/§7 rows
and work items R1–R5, V1). No new data: everything below is computation on the stored waveforms or simulation.

## 0. Decisions and status (updated 2026-11-05, after Dylan's review)

- **Chambers:** det3 (chamber A) carries the section, both Paper I runs (`sat_det3` = Run 1, the p2 row = Run 2);
  det4 at SPS is the reference-free known-angle check. det2/6/7 only where a cross-check is cheap (P0 used them).
- **Data first.** Arm D is most of the section; Arm S only for what data cannot vary (resistivity, footprint, shaping).
- **Every tier gets the best reconstruction we can build** (Dylan); naming a real ASIC only where the emulation is
  faithful to its logic (decided after `hits_daq/DAQ_SURVEY.md`).
- **P0 done** (`hits_daq/P0_XBIAS_2026-11-05.md`): the X hits bias is real (det3 vs M3: X 1.19, Y 1.37 ladder
  compression) and the production model reproduces it with no X resistive term. It is same-time mixing — drift
  diffusion compounded by the prompt footprint — read through one time per strip; a metallic readout keeps most of it
  (X 1.17, Y 1.22 on det3), and the RC spreading roughly doubles the excess on Y only. The story "hits fail because of
  resistive sharing" must be widened (Dylan to decide the wording in `RECONSTRUCTION_BASIS.md`).
- **Survey done** (`hits_daq/DAQ_SURVEY.md`): five hit families exist (below). It also corrects one of our prior-art
  claims: T2K ERAM takes positions from the pad response at the waveform maximum and fits full waveforms only to map RC,
  so it is a waveform model of resistive sharing, not a per-event waveform reconstruction.
- **Arm D windows done** (det3 Run 1 12,366 planes, Run 2 38,364; EOS `cosmic_bench/x17rerun/hits_daq/windows/`);
  SPS det4 6 rows running at the 11-05 wrap-up. First H2c numbers and the λ scan: `HANDOFF.md`.
- **Arm S engine changed** from the g4_digi port to re-simulation on real events (`hits_daq/resim.py`): each plane's
  own (p0, w, t0, q) regenerated through `model_waveforms` with a modified kernel plus its own fit residual, exact
  truth; P0 is its validation (the model reproduces the data's hit compression). §6's g4_digi port is not needed.
- **V0 validated 2026-11-06** (`hits_daq/V0_VALIDATION_2026-11-06.md`): with production's measured gate noise
  (rms_gate = 0.68 × rms, now `emulate.GATE_K`) V0 misses 0.03 % of production strips, adds 1.5 % marginal ones, and
  matches the ladder to 0.3–0.4 %. Numbers made before with the old 0.447 gate are preliminary.
- **Literature review done 2026-11-06** (`hits_daq/LITERATURE_MIXING.md`): every single-time µTPC we could read
  (LMU/NSW Micromegas, BESIII GEM) reports angles too large (tan ratio 1.1–1.9, shrinking with angle), attributed to
  capacitive coupling and edge strips; BESIII names diffusion as a second effect in one sentence; fixes are empirical
  (edge-strip drop, 15–20 % neighbour subtraction, angle calibration). Nobody fits µTPC waveforms, and no source shows
  that diffusion alone compresses the ladder on a metallic readout. Draft intro paragraph in its §5.
- **H2c 2026-11-06** (det3 Run 1, corrected gate): the depth profile is not the limit (mean/ampmean shapes do not
  beat amp/uniform). The Y residual is near normal only and comes from data–model timing on the weak strips 3–4
  pitches out (105–150 ns early in data, amplitudes right); the method closes on model windows (0.3°). Per-strip
  jitter error t_rise 400 ns: X 1.14 / Y 1.27° (W 1.02/0.93). Robust time cost pending; numbers in `HANDOFF.md`.
- **H2c default fixed 2026-11-06 evening: amp λ 3000, t_rise 700 ns, t_robust 0** (`h2c.py`, `h2c_run.py`). Scan
  0/250/400/600/700/800/1000 on det3 Run 1 odd ids (`hits_daq/results/h2c_trise_sat_det3.json`): X s68
  1.23→1.14→1.13→**1.11**→1.12→1.13 (flat beyond 600; the inclined X drift, 0.2–0.3 steep −0.37…−0.42, is there
  already at 400, so it is not t_rise's), Y 1.88→1.27→1.23→**1.22**→1.19→1.17 (near-normal +2.2→+1.5→+1.3° keeps falling,
  but the 0.1–0.2 bin drifts −0.15→−0.35). 700 = best X, Y overall steep +0.04; closure on model windows 1.13/1.02.
  H2c's remaining gap to W (0.09° X, 0.29° Y) is the near-normal bin, i.e. the far-strip timing below.
- **Far-strip timing (`hits_daq/far_strip_timing.py`, `results/far_strip_timing_sat_det3.json`)**: data minus pass
  model, leading-edge hit time vs lateral distance from the track's projection. Y: −51 / −176 / −244 ns at 1–2 / 2–3 /
  3–4 pitches near normal, −69 / −245 at |tan| 0.05–0.15, −99 / −264 at 0.15–0.3; amplitudes right (0.96–1.15). X:
  only near normal, −43 ns at 1–2 and −265 at 2–3 pitches (50 hits), amplitudes 19–27 % high; inclined X out-of-track
  strips are ≤ ±8 ns. So it is mostly Y, at every angle: the far charge on the resistive view arrives earlier in the data
  than the model's RC kernel brings it (or a prompt coupling adds to it); on X a smaller near-normal effect. Not yet
  compared quantitatively with Flierl's 15 % coupling; the pass model is not changed (Paper II diagnostic only).
- **Gap-end check (`hits_daq/p0_depth.py`, `results/p0_depth_sat_det3.json`), supports LITERATURE_MIXING's reviewer
  note**: per hit, t_le − true drift time against fractional depth in the plane's charge column (ends from the fitted q,
  median 720 ns vs gap/v 784), |tan| ≥ 0.15, per-plane mid-column offset removed. Mid-column (f 0.2–0.8) the slope is
  −30 ns per column in data (≈ 4 %), −55 in the model and the metallic arm (≈ 7–8 %); the end bins carry the rest: data
  +84 / +24 ns at f −0.05 / +0.05 (mesh end) and −19 / −65 ns at f 0.95 / 1.05 (cathode end); model/metallic alike. The
  amplitude-weighted ladder is dominated by those end strips. Caveat: the 'point' arm (no diffusion, floor footprint)
  also bends at the ends (+62 / −44 ns): strips past the column end see one-sided charge from the strip width alone, so
  only the excess over 'point' (≈ +20 / −25 ns) is the diffusion/footprint mixing.
- **Analysis DAG 4634728 scored 2026-11-07** (code `8000aeac`, H2c defaults amp λ3000 / t_rise 700; full det3 Run 1
  and Run 2, all event ids; 244/260 nodes, the 16 failed `R_` nodes were the `--variants` quoting bug, fixed in
  `b2685686` and rerun clean in `ana_resim_2026-11-06`; job log 292 starts = 244 + 16×3, no other failure). Tables on
  EOS `hits_daq/results/`. Score each run on its own: `score_h2c.py` keeps only event ids common to all labels, and
  Run 1 / Run 2 ids collide by accident (~800 planes).
  - Tiers (`tiers.py --from-tables`), naive / H0 / H2a s68 [°], X | Y, W = 1.04 | 1.01:
    Run 1 V0 3.18/2.06/1.75 | 3.75/2.68/2.42, V0r 3.08/2.05/1.79 | 3.38/2.42/2.05;
    Run 2 V0 3.02/2.11/1.78 | 3.68/2.61/2.40, V0r 2.93/2.10/1.77 | 3.30/2.38/2.00. Best H2a: V4e/V6r (X 1.69, Y 1.74–1.98).
    Naive steep +2.3–2.5° X, +3.0° Y on V0.
  - H2c (`score_h2c.py --bins`): Run 1 X 1.20 (steep −0.05) / Y 1.28 (+0.19); Run 2 X 1.20 (+0.11) / Y 1.29 (+0.25);
    in5 0.84–0.86. The gap to W (0.16 X / 0.27 Y) stays in |tan| < 0.05: steep +0.8–0.9 X, +1.4–1.6 Y, s68 2.0–2.7.
    (The odd-id subset of 11-06 read X 1.12 / Y 1.19: the full sets are ~0.08° worse.)
  - Arm S first look (resim, Run 1, 14 variants, |tan_true| > 0.1; median u/tan_true relative to 'point', DREAM
    family; < 1 = more compressed): X base 0.88, metal 0.92, dp0 0.97, s0_45 0.85, sheet 0.81 (rc_D_y variants leave X
    unchanged, as they must). Y base 0.80, metal 0.90, rc0 0.87, rc05 0.84, rc2 0.70, rc5 0.56, rc10 0.43, dp0 0.91.
    So without RC, diffusion + footprint still compress ~10 %; nominal RC adds ~8 % on Y and it scales steeply above.
  - **Resim scorer** `hits_daq/resim_score.py` (against exact truth, ratio = slope / true slope, |tan| > 0.1, no v):
    base X 1.18 / Y 1.34 — **the model reproduces the data's compression** (P0 data X 1.19 / Y 1.37), which is
    gate S0's ratio test passed on the hits side. Y: rc0 1.23, rc025 1.26, rc05 1.28, rc2 1.52, rc5 1.93, rc10 2.47;
    metal 1.19 (X 1.14), point 1.07 (X 1.04: strip width alone), s0_45 1.40, sheet X 1.28. V6 (ideal single time)
    within ±0.05 of V0 everywhere, so the hit algorithm is not the limit. **W reads 0.99–1.00 on every variant**,
    up to 10× RC. (Its absolute s68 there, 0.15–0.46°, is on the model with exact truth: not a resolution claim.)
    The first-look numbers above (relative to 'point') are superseded by these.
  - **SPS known angle** `hits_daq/sps_score.py`, rot25 CO₂ rows d550 / d600 / run59, inclined Y view. No column v
    on these rows (v 7.5–10 µm/ns, the column overruns the window), so v is set per row from the stage angle through
    W: everything is relative to W on the same plane. Hits / W slope: V0 1.13 / 1.10 / 1.07, V0r 1.05 / 1.04 / 1.03,
    V4 1.08–1.14, V6 1.09–1.13, V1/V2/V5 1.19–1.32; s68 V0 1.40–1.49° (V0r 1.05–1.12) vs W 1.02–1.10°.
    Compression shrinks with field (d550 → run59 at 243 V/cm). H2c on SPS: ratio 1.00–1.05 but s68 2.0–2.7°, i.e.
    untuned there (its model takes v = 33 µm/ns and t_rise from the det3 bench bundle): don't quote. The X view
    (normal incidence) and the 0° run55 row are not scored: at v ≈ 8 µm/ns, u / v blows up the hits' normal slope.
    Mount: 25.4° (measured yaw) used, `--stage` to change.
  - SPS d650/d700 resubmitted 11-07 as DAG 4635707 (`~/x17rerun/hits_daq/ana_2026-11-07`, 50 nodes, `tomorrow`);
    when done: `sps_score.py --tables <dir> --json results/ana1106_sps_score.json` and `make_report.py`.
  - **Report** `hits_daq/make_report.py` → `hits_daq/report/report.html` (four figures, tables, caveats), from the
    `results/ana1106_*.json` score files. Oddity seen, not chased: H2a's binned remap reads shallow on the steepest
    bin (X −3.5° at |tan| 0.3–0.5, Y −3.9° at 0.2–0.3), probably the map's edge statistics.
  - `score_h2c.py` now refuses labels that are not the same planes (> 5 % lost to the common cut); `--each` scores
    labels from different runs on their own planes.
- **Dylan's decisions 2026-11-08** (answers to the 11-07 list): (1) mechanism wording stays same-time mixing, and the pitch
  question is now tested (`hits_daq/gang_scan.py`); (2) H2c tuned per SPS row, as far as it goes; (3) **H2b′ dropped**; (4) shaping axis
  in Arm S (`resim_shape.py`); (5) H2c on the simulated variants (`resim.py --h2c`); (6) **det3 carries the paper, det4 where SPS adds
  something**, others only if interesting; (7) **§7 merged into §6, §8 into §9**; (8) **the floor is in Paper II** (§5 rewritten);
  (9) 60 ns depth-bin bias paragraph added to §5; (10) **mount angle 25.4°**, the measured yaw (`det4_gap`'s 25.52 is the midpoint with the config).
- **H2a edge oddity resolved 2026-11-08** (`hits_daq/h2a_edge.py`, `results/h2a_edge_sat_det3.json`): not statistics and not the flat ends
  (1–2 % of planes clamped; 40 bins still −2.7° at |tan| 0.3–0.5). H2a is median(tan_ref | u), a regression of truth on a noisy slope, so
  it shrinks toward the training angle distribution. The inverse map median(u | tan_ref) inverted is unbiased at steep angles
  (+0.06° X, −0.12° Y) but loses resolution overall (X 2.10 vs 1.75°, Y 2.36 vs 2.42°). So H2a's s68 is partly the cosmic angle prior: quote it as such; it will not hold on a different angle distribution.
- **Scans 2026-11-08** (DAG 4637865, `~/x17rerun/hits_daq/ana_2026-11-08`, code `code_hs_19d9be5b`; at wrap-up 61/77 nodes, 0 failures). Scores in `hits_daq/results/`:
  - **Pitch** (`gang_scan.py`, `results/gang_scan.json`, 16/16; det3 Runs 1+2, 1/2/3/4/6 strips summed): wider strips are worse. Data vs M3, |tan|>0.1: X 1.22/1.30/1.43/1.56/1.82, Y 1.36/1.46/1.61/1.81/2.22; at |tan| 0.2-0.3 X 1.12/1.16/1.23/1.32/1.61; geometry-only arm X 1.02/1.04/1.10/1.19/1.36, so most of the growth is the strip-end term. Coverage falls (X 0.48 at 6 strips). Answers Dylan's Q1: fine strips are not to blame.
  - **Shaping** (`resim_shape.py`, `results/shape_scan.json`, 4/4; CR-RC², 25/50/100/200 ns, S/N held at DREAM's, time at peak): X 1.198/1.187/1.175/1.179, Y 1.368/1.353/1.383/1.460, metallic Y 1.177/1.163/1.154/1.155. Barely depends on shaping: a VMM-type front end does not fix it. ENC vs shaping not modelled.
  - **H2c on simulated variants** (`resim.py --h2c`, `results/h2c_resim.json`, 39/48 **partial**): H2c slope ratio 0.96-1.00 on every variant incl. 5× RC; s68 Y 0.87-1.14° against 1.55-4.2° for the plain hits line. On simulation with the generating kernel known. On data H2c is 1.20/1.28° vs W 1.04/1.01°.
  - **Final 2026-11-09**: DAG 4637865 77/77 nodes, one start each, all exit 0, no evictions. H2c on variants complete (48/48): the 39/48 numbers hold (slope ratio 0.97-1.00, Y s68 0.88-1.15° vs V0 1.56-4.19°).
  - **SPS H2c tuning scored 2026-11-09** (`hits_daq/spstune_score.py`, `results/spstune_d600.json`; d600, 200 inclined planes, v 8.35, gap 28.6): fixing v and gap took H2c s68 from 2.60° (11-06, v 33, gap 27.9, n 3000) to 1.67-2.19°, ratio 0.99-1.02. The t_rise × S_T grid does not choose a setting: 700/15, 1500/15, 1500/40 agree (paired bootstrap, 1.72/1.67/1.73, differences ≤0.04 ± 0.21°), so the bench 700/15 stays. H2c remains above V0 (s68 1.34°, but ratio 1.12) and W (0.93°); in5 H2c 0.65 vs V0 0.505 vs W 0.70. About 21 % of fits end at |w| > 2× W, on the same planes at every grid point, with the seed hit ladder itself ~3× W there; 30 % of these planes have W itself > 5° off the stage. On the planes where W is on angle: H2c in5 0.82, V0 0.66. Rerun on all 5 inclined rows with row v: DAG 4646000 (`~/x17rerun/hits_daq/ana_2026-11-09`, stage `spsrow`, 120 nodes, tables `h2crow_sps_*`, same code `code_hs_19d9be5b`).
  - One SPS job of DAG 4635707 (`H_..d650_20`, 4635729) exited 1 once and succeeded on retry; the retry overwrote its log, cause not recoverable.
- **Capacitive coupling study 2026-11-09**: `hits_daq/report_coupling/report.html`, literature in `hits_daq/reports/`. §6 must not say same-time sharing is minor or that coupling is absent; awaiting an independent review (HANDOFF).
- **TPC literature survey 2026-11-08** (`hits_daq/LITERATURE_TPC_SURVEY.md`, index `hits_daq/literature_survey/README.md`, deck notes/mx17-tpc-literature): strip µTPCs report the bias under other names; long TPCs have Λ 7-110× smaller; LAr already has the channel-time convolution. Paper II claims the measurement, the decomposition and the per-event depth fit, not the effect.
- **(answered above)** Open question: if the mixing exists even on a metallic readout, why does no one
  else need a waveform fit? Literature review planned (HANDOFF next step 1; Dylan's T2K hypothesis: pads ≫ diffusion).

## 1. What the section has to show

1. A hit readout (one time + one amplitude per strip, VMM-style) reconstructed **naively** reads tracks too steep
   and depth-wrong, and the bias is set by the charge mixing on each strip (drift diffusion and prompt footprint in
   any µTPC, plus the resistive spreading in the view across the resistive strips — P0), not by the hit algorithm.
2. The **best corrections we can build on hits alone** shrink the bias but leave a resolution gap to the waveform fit
   (or close it — that is the open result, see H2c).
3. **How the gap depends on the resistive layer**: as the sharing goes to zero the hits chain approaches a classic
   µTPC — which P0 shows is itself compressed (metallic 1.17–1.24 against 1.21–1.54 with RC); how far it gets, and what
   still limits it (diffusion, prompt footprint, shaping), is the generalisation.

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

## 3. Hit-DAQ variants to emulate (the "V" codes) — `hits_daq/emulate.py`

Every variant reduces each strip's waveform to at most one hit `(t, a)` per event. The families are the survey's.

| code | family | time | amplitude (weight) | front-end conditions | real examples (survey) | fair to name? |
|---|---|---|---|---|---|---|
| **V0** | DREAM internal (our `combined_hits`, H0's input) | 30 % leading edge of the parabolic peak | parabolic peak | ped + common mode subtracted, boxcar 5-sample gate at 5σ | DREAM + `WaveformAnalyzer.cpp` | yes (it is ours) |
| **V1** | peak time + peak | time of the pulse maximum | peak | analog: raw − ped (no CM), threshold on the pulse, first pulse only, neighbour enable | VMM3a time-at-peak (its designed mode) | "VMM-like logic at DREAM shaping" |
| **V2** | threshold time + peak | first threshold crossing | peak | as V1 | VMM3a time-at-threshold; TIGER ToA + E-branch; STS-XYTER | as V1 |
| **V3** | ToA + ToT | first threshold crossing | time over threshold | as V1, no neighbour enable | TIGER ToT mode; Timepix3 | as V1 |
| **V4** | sampled + per-strip fit | Fermi-Dirac rising-edge fit, half height | max sample | ped + CM subtracted (as V0) | APV25 (SRS), SAMPA practice | yes for the algorithm (APV25 samples at 25 ns, we at 60) |
| **V5** | binary | threshold crossing quantised to a 25 ns bunch-crossing clock | none (unweighted) | as V1, no neighbour enable | VFAT3 | "VFAT-like"; included as the floor of the family list |
| **V6** | ideal single time | per-strip template fit (matched-filter t0) | template amplitude | as V0 | none: the upper bound for one-time-per-strip | – |

Suffix `r` = the robust ladder (iterative 3σ trimming of strips, the role of the Hough step in the NSW / APV chains).
Sub-sample times come from a monotone cubic upsampling of the 60 ns samples. The DREAM shaping is kept (it cannot be
undone from 60 ns samples), so V1–V3 and V5 emulate the **logic** of those chips, not their analogue front end;
their own shaping (VMM 25–200 ns, TIGER 60/170 ns) is an Arm S axis. V0 is validated against the production hits of the
same strips (extraction stores them). VMM3a numbers used: 10-bit peak ADC, ~250 ns dead time (one hit per channel in a
1.9 µs window), neighbour enable on (NSW test-beam setting); thresholds 5σ by default, scanned 3–7σ (R4).

## 4. Reconstruction tiers on hits (the "H" codes) — `hits_daq/tiers.py`

Every tier runs on every V family; a result is named `<V code>/<H code>` (e.g. V1/H2a).

| code | what | uses a reference? | uses the kernel? | status |
|---|---|---|---|---|
| **H-naive** | amplitude-weighted line t(x) through the hits, tan = u / v with v the bundle's calibrated drift velocity | no | no | in `tiers.py` |
| **H0** | the same line with a global slope remap tan = (u − a)/k, (a, k) from M3 on the other half of the events (event-id parity) — what `hits_arm.py` does | yes | no | in `tiers.py` |
| **H1** | per family, the best per-strip estimator variant (weighting, robust trimming, first/last-strip drop below 25–50 % of the neighbour, as the APV25 / VMM literature does) chosen on the training half | H0's | no | to build (P3) |
| **H2a** | angle-dependent remap: tan = f(u), f the binned-median map of tan_ref against u, trained on the other half — the best any slope-only correction can do | yes | no | in `tiers.py` |
| ~~H2b′~~ (dropped 11-08) | hits-only unsharing: deconvolve the hit amplitudes with the kernel, correct each strip time by its neighbours' amplitude-weighted arrival, refit; plus a head-on tagger on hit features (hits6) | no (constants from the kernel) | yes | to build (P4) |
| **H2c** | forward model through the emulator: generate the window with `build_matrix_rc` from (p0, w, t0, depth profile), apply the same V family, fit the predicted hit list to the observed one (χ² on t and a; strips below threshold censored) | no | yes | to build (P6) |
| **W** | the pass-1 waveform fit, same planes: tan = (1e3 w − w0)/(kw v) | its calibration (kw, w0 from M3) | yes | in `tiers.py` |

Scores (`tiers.py`): d = atan(tan) − atan(tan_ref); s68 over |d| ≤ 5° with the fraction inside 5° (tails),
**steep** = median of d·sign(tan_ref) (> 0 = too steep: the compression, which a plain median cancels between
signs), per |tan_ref| bin; implied v = slope of u against tan_ref. The paper-convention s68 (tilt map removed,
`floorlib`) is applied for the final numbers.

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
| ~~P4~~ | dropped 2026-11-08 | | |
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
