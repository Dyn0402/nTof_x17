# Handoff

## Per-view sharing kernel (PAPER_PLAN C2) — updated 2026-10-09 05:00 (overnight, dylan-MS-7C84)

**Resume:** per-view Y ratio validated (gate: Y −0.04…−0.08°, head-on −0.10…−0.13°, det2/3/7); re-pass staged in `condor_campaign_pvy95/`, NOT submitted — needs Dylan's go.

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
