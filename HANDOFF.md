# Handoff

## X width excess (UNSOLVED, needs follow-up) — updated 2026-10-12 (dylan-MS-7C84)

**Resume:** OPEN PROBLEM. Bench X widens ~dry-like while v says wet; no gas fits. Next: test it with a dry-gas bench run or det3 Y.

**Goal:** explain why the X cluster width grows with drift depth faster than Magboltz allows. Until it is explained the
detector model's drift diffusion is empirical (Dd fitted from X widths), and the re-pass stays on hold (Dylan, 10-09).

**Established (FINDINGS §29; report `~/x17/cosmic_bench/cloud_basics/xwidth/report.html`):**
- Every bench dataset shows it (5 chambers, 8 runs, det3 drift scan at 6 fields and 2 gain scans). X D_T is 400–520 µm/√cm.
  Ar/iso 95/5 with the water each run's v needs (0.6–1.0 %) gives 270–325. The extra is +300–430 in quadrature.
- It does not depend on gain (det3, 425–505 V). It is present over the whole active area, with no hot spot. Only the
  early (prompt) width depends on position: ×~2 in ⟨x²⟩ at the edges. That is a separate edge/footprint effect.
- It lands at the anode (no X undershoot, rc_D_x → 0, §17). It is not track inclination (§28), and δ-rays do not
  grow with depth.
- No Ar/iso/water mixture fits det3's v(E) and D_T(E) together: joint χ² 80/36 (grid iso 0.5–8 %, water 0–1.5 %).
  An iso-poor gas (~1 %) can fit a single-field chamber (det4, det7), but not det6, and not det3's scan.
- The beam has it too. det4 run_71 X, template-free fit: +118 / +143 / +286 in quadrature at 243 / 150 / 92 V/cm,
  above the thermal floor at 92. This is indicative only: at that field the drift time is 3× the window.
- It is closer to per-depth than per-time (det6/det7 ratio 0.86 ± 0.11 vs 1.16 per-time, ~2.7σ). Bench and beam
  per-time differ ×20.

**Hypotheses left, not separable with data in hand:**
1. Magboltz's water model: water slows electrons as predicted but cools them transversely less. Both setups are wet.
2. A non-gas lateral term linear in drift depth, not in any current model.
3. (weaker) A contaminant other than water (N2, CO2, …), or isobutane depletion beyond what det3 allows.

**Next steps (follow-up):**
1. Decisive: a bench run in fresh, dry gas (new bottle, high flow, long flush), with a drift scan, one chamber.
   If X D_T stays ~480 and v rises to dry, it is hypothesis 1 (Magboltz water). If the excess persists in dry gas,
   it is hypothesis 2.
2. Cheap: det3 drift-scan **Y** view (caches already hold Y). A joint X+Y fit per field: does Y's Dd agree with X's?
3. Magboltz cross-check: the same mixtures in another code or another cross-section set (Biagi water vs LXCat), or
   published Ar/iso/H2O D_T data.
4. Beam: get a beam X template (re-extract the run_71 bundle) and redo `x_width_beam.py` with the template model.
   Use the 25.6° runs (run_62/63) for a known-angle cross-check.
5. Other contaminants: add N2 / CO2 / air columns to `make_iso_jobs.py` (no O2 on the bench, §21).
6. The edge early-width effect is separate: worth a look on its own (frame/field distortion, footprint, M3 at the edges).

**Decision for the kernel meanwhile:** take Dd from the X stack (≈ dry), not from v (§28). Never carry a v-derived water
fraction into D_T.

**Key files & commands** (`sps_beam_test_26/analysis/cloud_basics/`, run with `../../../.venv/bin/python`):
- `headon_cache.py <key> [--subrun S --base DIR]` — head-on caches → `~/x17/cosmic_bench/cloud_basics/xwidth/caches/`
  (kept). The det3 scan sub-run *waveforms* were only in job scratch (gone). Re-pull with
  `rsync -a --exclude='long_run*' --include='*/' --include='decoded_root/***' --include='m3_tracking_root/***' --exclude='*'
  lxplus:/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/Run/mx17_det3_saturday_scan_6-27-26/ <dir>/mx17_det3_saturday_scan_6-27-26/`
  and copy run_config.json in. (/media is nearly full, so use the nvme.)
- `x_width.py`, `x_width_map.py`, `x_width_beam.py` → `results/x_width*.json`
- `make_iso_jobs.py [--ext]` + lxplus `~/cloud_basics_condor/att/condor_mb_iso{,ext}.sub` (condor 4440949, 4440956) →
  `results/iso/`; `iso_fit.py` → `results/iso_fit.json`; `make_xwidth_report.py` → report.
- Refresh after the last 5 Magboltz points (dry, near-pure argon) land:
  `scp 'lxplus:cloud_basics_condor/att/magboltz_benchi*' results/iso/ && python iso_fit.py && python make_xwidth_report.py`

**Gotchas:**
- At ≳300 V/cm, v is not monotonic in water for 95/5. Inverting v → water there is ambiguous, so take the closest grid value.
- det3 gap 27.9 mm (dished cathode): scale v ×27.9/30 and D_T ×√(30/27.9) against fits made with GAP = 30.
- Stack v (GAP/T) runs ~10 % below bundle v on det2/4/7. D_T is ~insensitive (σ_drift/√gap), but the water inference is not.
- Branch bench-pin-rc is 40+ commits ahead of main (the whole cloud_basics line) and has never been merged: Dylan to decide.

## Detector model basics (cloud_basics) — updated 2026-10-12 (dylan-MS-7C84)

**Resume (pm):** see the "X width excess (UNSOLVED)" section above. It is the open blocker for freezing the kernel.

**Resume (am):** det4 pinned-D_rc arm DONE (FINDINGS §28): beam 2.2e-4 costs nothing on the bench (stack rms +0.3 %,
held-out Y +0.01–0.03 ± 0.02–0.03°); §27's 1.5–2.7× was rc_drift's assumed D_T — the joint X+Y fit pins Dd (X has no
RC) at ≈ dry on det2/3/4/7. Bench drift fields were wrong in bench_pinned (all 250 V/cm; really 347/208/243) —
`bench_water_e.py`; still 0.4–0.6 % water from v vs dry widths. Tight |tan|<0.01 brings bench σ0 to the beam 0.37.
Next: decide the bench D_T source (Magboltz water D_T, or a non-diffusion X growth term); then freeze the kernel
(det4 D_rc from beam, others bench; Dd from joint fit; σ0 from tight-|tan| stack) and run the full gate.
Older queue: det4 X vs amplification stripes, det7 bench plateau.

**10-12 (dylan-MS-7C84):** `bench_pinrc.py` (joint fit with pins + `--profile`), `bench_tanscan.py`, `bench_water_e.py`,
`compare_bench.py --base`. bench_attach's bundle path now resolves next to each plane_ratio cache (old one was session
scratch). Bench outputs `/home/dylan/x17/cosmic_bench/cloud_basics/bench_pin/` (arms in `arms/arm_det4_rc{f,p,pw,pd}.json`).

**10-11 (dylan-MS-7C84):** `attach_transfer.py` injection toy (SPS measurement code on known events, run_71 loss on/off,
paired): footprint/D_rc/c2-c1 shift ≤ 7 %, decompose b1 −27 %. Toy realistic at 243 V/cm only. `rc_drift.py` (RC fit +
Magboltz drift diffusion): beam det4 Y D_rc 2.08/2.08/1.94e-4 over 3 fields → 2.2e-4 ±10 % loss-corrected. Bench D_rc moves
by a third with the assumed D_T (σ0 with it): degenerate; det4 bench/beam 1.5–2.7× unexplained. `width_tan_split.py` inconclusive.

**10-10 pm (dylan-Yoga):** slide-note version published, notes/mx17-attachment-slides (24 slides, built by
`cloud_basics/make_attachment_deck.py`, commit 8fd794a): observable + four-mechanism fingerprints first, then a beam/bench
consistency section (shared-vs-differing setup table, bench drift scan, model-free overlay, loss rate vs field, ledger).
Honest caveat found while building it: the bench's model-free late/early ratio moves −4..+2 % because its own no-air model
falls 3–5 % (response tail + diffusion), so the bench O2 limit rests on the composition fits, not that ratio. The long-form
report notes/mx17-beam-attachment is unchanged. Slide cross-refs are `§§<id>§§` tokens resolved at build.

**10-10 am (dylan-Yoga):** DONE: gas compositions as model curves (FINDINGS §24-26, report F9-F13, notes/mx17-beam-attachment).
Beam run_71 1.51 % water + 0.078 % air (163 ppm O2), all 3 fields x 2 views fit on the high-stat grid (eta smoothed in E);
run_63 blocks 230 -> 147 ppm O2 over the night; CO2 period ~290 ppm; bench det3 6-field drift scan 0.95 % water, no air (<~10 ppm).
One physics model, setup-specific effective shapers. X: snapping rejected, footprint tails real but resolution-neutral (§23).
Next: det4 X deficit vs amplification stripes; det7 bench rising plateau; three-body O2 attachment scale (ppm is model-dependent).
Data note: re-extracted run_71 cache lives only in the laptop job tmp (regenerate: extract_det4_only.py run71_raw --cm masked --keep 12);
det3 drift-scan decoded_root pulled to ~/x17/cosmic_bench/det3/mx17_det3_saturday_scan_6-27-26/.

**Goal:** Dylan (10-09): get a physically sound, consistent detector model *before* publishing or re-passing; solve as many
consistency problems as possible and converge on one model over the coming weeks. Re-pass is ON HOLD until then.

**Done (narrative + numbers in `sps_beam_test_26/analysis/cloud_basics/FINDINGS.md` §1–12):**
- Prompt footprint σ0 ≈ 0.4 mm is readout, not gas (avalanche ~25 µm); same on bench and beam (leading edge, unbiased stacks).
- Bench drift diffusion = dry Magboltz on 4/5 chambers (det6 low). Moisture barely changes D_T.
- Y spreads as a 1-D RC line (σ² += 2·D_rc·t, D_rc 2–9e-4 mm²/ns); X does not. Y template carries the RC undershoot → use X template for both.
- NO bench attachment — my earlier claim was a max-normalised stacking artefact (retracted; use unbiased fixed-threshold stacks).
- Beam run_71: late charge decline is common to X and Y (±4 sums) and field-dependent (−5/−9/−20 % by 600 ns at 243/150/92 V/cm) → gas attachment in wet beam gas, not electronics.
- `wft.model.build_matrix_rc` (opt-in via `rc_D_y` key; production untouched; tests `wft/tests/test_rc_kernel.py`).
- Held-out bench, no refit: Y better on all 5 chambers (det2 −0.15/−0.36, det3 −0.09/−0.14, det4 −0.15/−0.20,
  det6 −0.55/−0.90, det7 −0.12/−0.28 °, all/head-on). X mixed: det6/det7 better, det2/3 neutral, det4 +0.16 worse; X χ² worse everywhere.

**In progress (lxplus condor, all 21 jobs were still IDLE at wrap-up — queue is clogged by my own 6164 is2_v2 jobs, cluster 4410513):**
- 4410533: Magboltz attachment (beam_w1p7 ± 0.05/0.1/0.2 % O2, bench_w0p5 ± O2) → `lxplus:~/cloud_basics_condor/att/magboltz_*.json`.
- 4410535: RC refits, arms rcf3 (σp0_x, σp0_y, rc_D_y), rcf4x (+rc_D_x), rcfD (+Dp) × 5 chambers → `lxplus:~/cloud_basics_condor/rc/out/arm_<det>_<arm>.json`.

**Next steps:**
1. Copy att JSONs to `cloud_basics/results/`; does 1.7 % H2O + some O2 reproduce size AND field dependence of the beam loss?
2. Bench the refit arms (`plane_bench.py`, then `compare_bench.py --start blind`) vs production and rcm. Does X want its own σ0 / an RC term?
3. X deficit hypothesis: charge on a 550 µm resistive strip snaps X to resistive-strip centres (0.80 vs 0.78 mm pitch) — a Gaussian footprint can't do that. Model it if refits don't close X.
4. Template-free beam charge-vs-depth on 25.6° runs (run_63/run_62 ZS; re-pull from EOS, mind ZS censoring).
5. Untested: leading spike = primary ionisation inside the amplification gap.
6. Then full gate (gate.sh) for the physical kernel; only then revisit the re-pass. Also make_report.py/report.html for cloud_basics.

**Gotchas / decisions:**
- Judge kernels by scale-corrected σ68 (divide each arm's tan by its own slope) with paired bootstrap; rc arms read angles 3–8 % flat.
- Stack unbiased (`CB_STACK` default): fixed threshold, no per-event normalisation.
- Bench template does not close a forward fit on beam; quote beam σ0 from the leading edge only.
- Magboltz on condor (desktop ROOT is py3.10 vs system 3.12); `v_um_ns` in magboltz JSON is ×10 µm/ns — use `v_true_um_ns`.
- plane_bench float-casts hypers: `rc_tmpl` is numeric (1.0 = X template).
- Housekeeping: offload lxplus `~/plane_ratio_condor/bench` (134 MB) with `lxstore offload`.

**Key files:** `sps_beam_test_26/analysis/cloud_basics/` (FINDINGS.md, make_rc_arms.py, compare_bench.py, beam_xy_pulse.py, …);
outputs `/home/dylan/x17/cosmic_bench/cloud_basics/` (arms/, bench/); RC payload in lxplus `~/cloud_basics_condor/rc/`.

## Per-view sharing kernel (PAPER_PLAN C2) — updated 2026-10-09 05:00 (overnight, dylan-MS-7C84)

**Resume:** ON HOLD (Dylan 10-09) — likely superseded by the physical RC kernel (see Detector model basics). Per-view Y ratio validated (gate: Y −0.04…−0.08°, head-on −0.10…−0.13°, det2/3/7); re-pass staged in `condor_campaign_pvy95/`, NOT submitted — needs Dylan's go.

**Done:**
- C1: r06 reproduces FLEET_DIGEST exactly (waveform-fit cells, five keys).
- `wft` gained opt-in per-view keys (`c2_over_c1_<view>`, `sigma_p0_<view>`, `c1_asym_<view>`) and `WFT_MODEL_FRAC`; absent = bit-identical (41 tests).
- Study tools + log in `sps_beam_test_26/analysis/plane_ratio/` (FINDINGS.md is the narrative); report `/home/dylan/x17/cosmic_bench/plane_ratio/report.html` (make_report.py).
- Candidate bundles `calib_bundle_pvy95` (+ controls `calib_bundle_prodt0`) in each golden key's wft dir; gate products `events_/alignment_/angles_/efficiency_{pvy95,prodt0,prodref}`.

**Decisions for Dylan:**
1. Submit the re-pass (`/home/dylan/x17/cosmic_bench/condor_campaign_pvy95/README_SUBMIT.md`). Mind the promote trap written there.
2. c2 < c1 gate: applied to the hyper it binds on Y (det2/3/7 improve further to y = 1.6). Hyper or observable?
3. det4/det6: their candidates help Y only against w0/kw-refreshed controls; adopting them = also the R06_GATE §4 w0/kw refresh.

**Gotchas:**
- Refitting the per-view ratio in the bench calibration does NOT work (det3 got worse); pin it.
- `wft.calibrate._event_chi2` is path-dependent (warm t0); recal.py uses a deterministic cold objective. The local calib cache is not r06's training cache (its 1.10e8 chi2 is not reproducible here).
- condor: LCG_105 numpy cannot unpickle numpy-2 caches — use LCG_108. `TAG=$(test && echo)` under set -e kills scripts.
- Condor working dir lxplus `~/plane_ratio_condor` (small JSON only), inputs on `/eos/user/d/dneff/plane_ratio/`.

## MX17 papers (two-paper plan) — updated 2026-10-08 (dylan-MS-7C84)

**Resume:** MX17 detector paper split into I (detector) + II (reco, hits vs waveforms); next: decide C2, then run C1 (reproduce r06).

**Goal:** get two companion papers written: I, the MX17 chambers and their performance (June cosmic bench + det4 in SPS H4); II, the waveform forward-model reconstruction and how much a hit-based readout loses.

**Done:**
- Plan written as tables in `mx_june_cosmic_qa/PAPER_PLAN.md`: both outlines, the work list (ids C/R/D/V, state, size, dependencies) and an n_TOF tripwire table.
- `mx_june_cosmic_qa/make_paper_plan_deck.py` (renamed from `make_paper_status_deck.py`) parses those tables, plus WAVEFORM_FIRST_THREADING §3/§10, FLEET_DIGEST and the SPS JSONs, into a 17-slide note. It is published at https://dylan-neff.web.cern.ch/notes/mx17-detector-paper-status.html (same slug as the 10-02 status note). The site commit is 61e0680 in dylan-cern-site.
- `PAPER_STATUS.md` points to PAPER_PLAN.md at the top.

**In progress / where it stopped:**
- No analysis started. Everything in the work list except R8 and D7 is open, partial or a decision.

**Next steps:**
1. Dylan decides C2: freeze r06, or build the per-plane c2/c1 ratio first. Decide before C1 finishes, so R1–R5 run once.
2. C1: pull lxplus `~/wft_campaign_r06`, run `collect_results.py --promote`, re-run `mx_june_wft` 01–04 on the five golden keys, and check that `digest.py` reproduces `FLEET_DIGEST.md`.
3. In parallel (no blockers): D7 (write the ready sections of I), D3 (timing on waveforms), R7 (SPS 25.6° known-angle test), R9 (calibration protocol), R10 (cost table).
4. Then R1 (hit tiers H0/H1 on r06, all five chambers), R2/R3/R4, R5 (the benchmark).
5. Whenever a work item changes state: edit PAPER_PLAN.md, rebuild the deck, and republish (commands below).

**Gotchas / decisions:**
- Dylan's decisions (10-08): split into two papers. II compares hits taken from the DREAM waveforms (with and without a correction) against the waveform fit. Full VMM emulation (V1) is deferred; `ntof_cosmics/g4_digi` is where it would plug in.
- The dependency is one-way: I cites II, so II goes first or the same day. The sharing kernel section lives in II.
- The July tier benchmark (WAVEFORM_FIRST_THREADING §10) used the R&D model `forward_model2`. Its kernel was physical (`hyper_v2` c2/c1 ≈ 0.17), NOT the retired c2 > c1 kernel, but it is neither the packaged `wft/` nor r06. It is a prototype only: re-run it (R5) before quoting.
- Dylan is waiting on the n_TOF MM reconstruction analysis before committing to the model. There are two watch items in the tripwire table (chamber C angle non-linearity; angle response with borrowed bundles, which looked like a transfer issue on A).
- This branch lives in its own worktree (`~/PycharmProjects/nTof_x17_paper`) because the main checkout is on `beam-off-cosmics` with others' uncommitted changes.

**Key files & commands:**
- `mx_june_cosmic_qa/PAPER_PLAN.md` — the plan (edit the tables here)
- `mx_june_cosmic_qa/make_paper_plan_deck.py` — the note generator
- `python mx_june_cosmic_qa/make_paper_plan_deck.py && python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py ~/x17/paper_status/mx17-detector-paper-status.html --slug mx17-detector-paper-status --force --deploy`
- Inputs: `mx_june_wft/FLEET_DIGEST.md`, `waveform_first_threading/WAVEFORM_FIRST_THREADING.md`, `sps_beam_test_26/analysis/{sharing_kernel,angled_kernel,spatial_resolution}/`
