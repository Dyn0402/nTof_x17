# Handoff

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
