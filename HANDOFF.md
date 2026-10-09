# Handoff

## Detector model basics (cloud_basics) — updated 2026-10-09 (dylan-MS-7C84)

**Resume (10-09 late, dylan-Yoga):** Dylan asked for: gas compositions shown as model curves matching data, the same for
the bench, all open questions, the X model. State:
- High-stat Magboltz grid RUNNING: condor 4410759 (697: beam/co2/bench fine grids) + 4410787 (120: bench 35/60/104/174 V/cm);
  watcher pulls into `cloud_basics/results/air_hs/`. When complete: `beam_comp_fit.py --source air_hs`,
  `driftscan_fit.py --source air_hs --emin 30`, `bench_fit.py --source air_hs`, then figures/report.
- Coarse-grid results: beam 1.55 % water + 0.070 % air (k=0, no gap spread); bench det3 drift scan (6 fields, §21)
  0.95 % water, NO air, k +0.1, gap spread 1.5 mm -- one composition fits 104-382 V/cm. 92 V/cm beam eta shape open.
- Open questions closed: faint excess = missing-sample selection (§19); 0.7 us ripple = trigger-locked pickup (§20).
- X: snapping rejected (snap_test); X footprint tails real (footprint_test) -> wft lor_frac_x/lor_gamma_x (47 tests);
  benches RUNNING condor 4410788 (`~/cloud_basics_condor/lor/bench`) -> compare_bench.py.

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
