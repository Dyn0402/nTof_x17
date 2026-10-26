# Handoff

## det3 loss categories + saturated-strip treatment (satstudy) — updated 2026-10-26 evening (dylan-Yoga)

**Resume:** second follow-up round done, note re-worked and republished (notes/mx17-det3-loss-categories). Three
opt-in wft switches validated and pushed (85e0e6d). Next = Dylan's calls below, then paper I appendix A1 + the pairing
section of the reconstruction paper.

**10-26 evening (Dylan: characterise the bands, re-categorise MULTI, pairing for the paper, head-ons to the extreme):**
- **Noise bands** (bands.py): whole-window baseline offsets ±50–3000 ADC on a contiguous patch of strips, x and y
  together (77–95 %), 2–19 % of triggers per chamber. They follow the CHAMBER (det3 same chips on FEUs 7/8 and 3/4,
  five days apart). Null: time in run, discharges before/after, trigger spacing, SCA cell, 10–32 ms periods, HV
  (|r| ≤ 0.19), CM coupling; no decay in the window (τ ≫ 100 µs). Mechanism NOT identified (stated limitation).
  Cost det3: 92.6 → 57 % detection in the 3.9 % of events with a band = 1.4 pts. `WFT_DROP_FLAT_HITS=1`
  (hits truncated both ends) 145 → 196/252, controls 400/400 (1 view moved); `=left` is worse (192, loses a control).
- **Re-categorised budget** (summarize.recat): band 1.43, second particle 2.84 (other fitted 1.26 / M3 fitted+cut
  1.01 / merged 0.57), δ 0.93, sat blob 0.81, spark shower 0.67 / discharge 0.50 / muon+other 0.40, clean fit 0.34,
  head-on 0.26, M3 kink 0.22, edge 0.19.
- **Multi-particle** (multitrack.py sub): M3 particle fitted to single-track resolution (σ 0.61/0.56 mm, 1.06/1.29°).
  Merged class = projection overlap within the 12 mm seed gap; `WFT_TWO_TRACK_FIT=1` recovers 8/37, 0/150 controls
  split.
- **Pairing by charge** (pairing.py, overlay.py): toy 95.8 / 97.6 % (q / q + profile), real 96.4 / 98.2 %, bench
  overlay ≥ 24 mm 96.7 / 100 % (production's pairing 50 %). Both tracks found: 0 % < 12 mm, 22 % 12–18, ≥ 98 % above.
- **Head-on in wft**: `WFT_HEADON_SIG=20` + `WFT_HEADON_RESCUE=1` (column at the other view's t0, fitted, kept if
  ln(qx/qy) matches within 3·hypot(0.054, σQ/Q)). Real: 17/17, controls unmoved, 0 fits added on 742 empty triggers.
  Injection (headon2, 845 x donors): x noise ×2 gain ×0.05: 44 → 84 %, fakes ≤ 0.2 % of empty planes; end to end in
  wft (headon3, 60 donors): 43 → 67 % (k .05), 77 → 90 % (k .1), fakes 0/720.

**Decisions for Dylan:**
1. Turn on WFT_DROP_FLAT_HITS=1, WFT_HEADON_SIG=20, WFT_HEADON_RESCUE=1 (and TWO_TRACK_FIT?) in the re-processing pass?
2. Port the head-on rescue + band veto to the beam seeder (ntof_tracking.wft_beam) for n_TOF; needs an n_TOF fake test.
3. Paper: single-particle efficiency first, multi-particle with any-track credit after; pairing as its own figure.

**Gotchas:** the fit worker swallows exceptions — a NameError in _headon_rescue was silent until
test_headon.test_rescue_wiring (now in). Memory: one reco job at a time, `--jobs 2` (~3 GB); the laptop OOM-killed the
session with 3 in parallel. headon3 "other view found" column is a scoring artefact (p0 vs mid-gap truth) — ignore.

**Earlier (10-26 pm):**
**Done:**
- `rerun/satstudy/` (README = run order). Golden row sat_det3, pre-pass final_phys table: 91.2 % detected of 6 451.
- fit_failed (0.26 %) = x never seeded (2-strip cluster, 3rd strip 7–9 % < 10 % floor). MIN_STRIPS=2: 17/17 detected,
  400 controls unchanged (reco_variant.py --min-strips 2).
- spark (1.95 %) = 32 discharge / 49 muon+other / 45 shower of 126. Veto off: M3 candidate in 34/49 muon+other but
  n_dropped 24–86 → recoverable only with the n_dropped cut relaxed too.
- MULTI (1.89 %): M3 particle is a candidate in 64 % of wrong views; production picks a brighter separate track.
- Saturation: rail 3650–3900 not SAT=3550; 58 % of clipped strips hit the bottom rail (fitted as data). BUT 5 censor
  rules on 350 saturated views move agreement 71.1 → ≤ 73.1 % (drop-strips 71.4 %): failures are δ-electron blobs whose
  unsaturated edges tilt the line. Lever = track + compact-blob model, not the mask.
- Paper I: appendix stub `papers/paper1/sections/A1_loss_categories.tex` (table of classes, fractions as \requote),
  \appendix in main.tex, PAPER_PLAN row A. Dylan wants one real display per class + lost fraction.

**Next steps:**
1. Appendix A1 figures: one display per class as PDF via figures.toml (reuse make_deck.py `evd()`); re-derive the
   fractions on the re-processing pass tables; decide on other chambers' columns.
2. Recovery fixes to try in the pass: MIN_STRIPS=2 (or an absolute SNR floor beside the 10 % relative one), a discharge
   veto (chip-wide simultaneous bands / negative swing) instead of the 50-hit count, n_dropped decision (C22 #2).
3. Track + compact-blob forward model for saturated δ blobs (the 0.8 % FIT_SAT + part of DELTA).
4. Untested: MIN_STRIPS=2 fake rate on empty events; per-channel rail from pedestals (charge, §41).

**Gotchas / decisions:**
- satfit.py reproduces the table only with WFT_PIVOT_START=1; saturated fits are ~8 s/arm → sample views (350 used).
- reco_variant.py needs WFT_PIVOT_START=1 WFT_COL_GUARD=0.01 X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage; seed
  defaults are bound at import, so it wraps the functions with functools.partial. 3 runs in parallel ≈ 7.5 GB RAM.
- Classes are heuristic (by-eye checks on random examples). δ ranges approximate (argon assumed).
- /media/dylan/data ~5 GB free. paper1 does not compile locally (authblk.sty missing) — Overleaf builds it.

**Key files & commands:**
- `rerun/satstudy/{satfit,examples,sparks,railstats,reco_variant,summarize,make_deck}.py`; outputs ~/x17/cosmic_bench/satstudy/
- `../../.venv/bin/python summarize.py && ../../.venv/bin/python make_deck.py` then
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <html> --slug mx17-det3-loss-categories --force --deploy`

## Outliers vs M3, det4/det7 head-on, edge zone (C22) — updated 2026-10-25 (dylan-Yoga)

**Resume:** C22 in `rerun/CHECKS_2026-10-23.md`; report `~/x17/cosmic_bench/outliers/report/report.html`
(published notes/mx17-outliers-vs-m3.html). Dylan to decide what the evaluation cuts become (below), then back to the
C21 decisions (promote phys, PIVOT_START, vrefit).

**Done:**
- Per-event join of the 5 golden finalize_phys tables with M3 (as 03_angles), every M3 ray, hits (QA) and full-plane
  waveforms of every outlier (`rerun/outliers/`); M3 cluster trees rebuilt from raw (condor 4546889, `m3clus/`,
  EOS `/eos/user/d/dneff/x17rerun/m3clus/`, local copy `~/x17/cosmic_bench/outliers/m3clus/`).
- det3: 11.8 % of matched events disagree with M3. M3 at fault ~0.8 % (bent tracks: kink between the doublets the
  4-layer χ² < 1 cannot see; bent share of the position tail 0.65 ± 0.07). Showers/second particles are NOT M3
  failures (M3 on one particle; fit takes the other) 2.1 %, delta rays 1.0 %, geometry 3.8 % (outside active area,
  edge zone), MX17 reco on clean tracks 1.9 %, correct fits cut by n_dropped > 2: 1.8 %.
- det4/det7 head-on tail = known hardware (det4 dead bands, det7 X card 1) + edge + outside; after those both are
  det3-like. The phys increment on det4 = faint dead-band columns tilted by the narrow footprint.
- Edge zone (PAPER_PLAN D4 closed on wft): det3 X k = 0.6 within 15 mm of the X edges, head-on fake tilt 0.10–0.15
  inward; Y protected by the passivation; det2 (no degrador) opposite sign.
- Pogo check on SPS run_71 det4: CLEAN (beam-lit channels only). `rerun/pogo_check/run_sps.py`.
- SPS head-on: NOT answered — bench bundle on run_71 gives 17 / 15 % far, a model mismatch (lumpy column, one-sided X
  copy), not det4. Needs a beam bundle (re-run plan S4).

**Decisions for Dylan:**
1. Cut the reference to the active area minus the 20 mm X-edge zone before quoting resolution / far rates?
2. Treat REJECTED_OK (fit fine, n_dropped > 2) as accepted? 1.8 % (det3) – 4.9 % (det7) of events.
3. Refit MULTI events with the multi-track fit instead of the largest-cluster seed?

**Gotchas:** a 4-layer M3 χ² < 1 does NOT protect against kinks at the DUT (12 mrad → 3.5 mm at χ² 0.5). The six-point
"MX17 is the odd point" test cannot decide bent vs MX17 for a chamber mid-telescope. `rerun/outliers/common.py` would
shadow the repo `common` package — the module is `olib.py`. SPS wft fits need K ≥ T/60 + 2 (K = 40 at 14 µm/ns).
Laptop data drive has ~6 GB free.

## M1 pogo-field short check (det7 asterisk) — updated 2026-10-25 (dylan-Yoga)

**Resume:** the general pogo check is built and run on the bench fleet. Optional next: a beam reader, and the Y cross-card correlation.

**Goal:** a reusable test for M1 cards whose pogo pins bridge readout pads, one that charge sharing between
neighbouring strips cannot fool. Plus Dylan's decision on det7.

**Done:**
- **Decision (Dylan):** g_det7_long and g_det7_longer carry an asterisk because det7's M1 cards were badly seated.
  Quote det7 X from card 2 only (connectors 5-8, ch 256-511). Recorded in `qa_config.py`, RERUN_PLAN section 5
  (row "asterisk") and `rerun/det7_pogo/README.md`.
- `rerun/pogo_check/` (METHOD.md, report.html, published at notes/mx17-pogo-check.html) has four tests:
  - a jump across ≥2 quiet strips, plus identical amplitude ratio (sd ln ratio < 0.3);
  - correlation compared with the same plane's correlation at the same strip distance;
  - a cross-card null (no pad short can join two cards);
  - layout: each net must be spanned by links between pogo neighbours.
- Fleet (10 bench planes):
  - only **det7 X card 1** is POGO (10 nets, 34 channels, vertical chains);
  - **det6 Y and det7 Y** carry 361 and 261 cross-card links, which is not pogo (edge connectors 1/8);
  - det3 is clean.
- **Correction:** the census "dense 2-D block" on det7 X (ch 3-58) does not respond to tracks. It fired in 5 of
  ~35 k clean events and co-moves only in saturated events of ~430 hits.
- Interactive page notes/det7-pogo-shorts.html, regenerated by `rerun/det7_pogo/make_page.py`; it now shows the
  asterisk and the correction.

**Next steps:**
1. (optional) Find what correlates the two Y cards on det6/det7: same events at both ends? Pickup via the mesh/HV?
2. (optional) A beam reader for `accumulate.PairStats` (hit cache; single-cluster events, lower MAX_HITS).
3. The det7 net-as-virtual-channel forward-model idea is shelved now that the asterisk applies.

**Gotchas / decisions:**
- Pogo neighbours ARE strip neighbours on the linear map (±1, ±4/5/6). Every d=5 pair on a connector is a pogo pair,
  so there is no same-distance non-neighbour control inside a card. That is why the jump test and the cross-card
  null exist.
- Without the ratio cut, inclined tracks split by under-threshold strips pass the jump test (det3 X, 42 pairs).
- Noisy (pickup) channels fail the identical-ratio test even when shorted; their membership is under-reported.
- out/*.npz (70 MB) is gitignored. Regenerate with run_bench.py (~10 s per chamber).

**Key files & commands:**
- `cd rerun/pogo_check; ../../.venv/bin/python run_bench.py g_det7_long --feu-x 6 --feu-y 8 --subs long_run longer_run --out-dir out`
- `../../.venv/bin/python pogo_check.py out/*.npz --out-dir results && ../../.venv/bin/python make_report.py`

## One physical waveform model for the fleet; det4 fixed under it — updated 2026-10-25 night (dylan-Yoga)

**Resume:** C21 in `rerun/CHECKS_2026-10-23.md`. Dylan to decide: adopt `phys` as the golden bundles (promote
`wft/calib_bundle_phys` → `wft/calib_bundle`), det7 X cost, PIVOT_START on. Then vrefit (needs timing v per group) → pass.

**Done overnight:**
- det4 works under the RC model with physics-set inputs: σ0 0.238/0.343 (det3 per-event), Dp from Magboltz / unsat
  stack, beam D_rc 2.2e-4. Ladders back on timing v (X 29.8 → 31.6, Y 29.6 → 32.8 vs 33.0), χ² halved, X s68 −0.065°.
  det3 unchanged (±0.005°). Same recipe on det2/det6 neutral-to-better; det7 X +0.05° (asymmetry, see C21).
- wft: opt-in per-event footprint profile, bundle key `sigma_p0_prof_<plane>` (absent = unchanged; tests pass, 78/78).
- lxplus: `~/x17rerun/pass_phys/` (own stage1.sub with WFT_PIVOT_START=1), code_40a8b47, both arms finalized side
  by side on the 5 golden rows: `<row>/wft/final_{phys,timeD}/`, `<row>/wft/calib_bundle_{phys,timeD}` (40/40 rc 0).
  The canonical `wft/final/` and `wft/calib_bundle` were NOT written.
- det7 pogo study (`rerun/det7_pogo/`), det6 gap 28.0 ± 1.0 committed.

**Next:** (1) promote phys (a copy per golden row) or keep timeD; (2) vrefit groups need their own timing v under arm D
(port headon_cache + unsat_width to stage_job) — and their Dp from the same stack; (3) pass with WFT_PIVOT_START=1.
Open: det4/det7 head-on far tail (+1.4–3.6 pp), det7 X asymmetry, det6 X wants narrower σ0.
Key tools: `rerun/det4/{resid_map,compare,profile_arms,finalize_compare}.py`; laptop data `~/x17/cosmic_bench/det4dig/`.

## From-scratch re-run of bench + SPS on lxplus: pass machinery ready, calibration protocol needs Dylan's call — updated 2026-10-24 (dylan-Yoga)

**Resume:** v question understood (C18/C19); Dylan to decide: PIVOT_START on, arm-D protocol, det4/7 kernel gate, det7 X mask. Then finalize → vrefit → pass.

**Done 10-24 day (C18, C19 in rerun/CHECKS_2026-10-23.md; note https://dylan-neff.web.cern.ch/notes/mx17-vdrift-consistency.html):**
- Three different v's: timing (gap/T, kernel-free); reco ladder (fitted w vs M3 tan — the per-event fit never uses v);
  calibration fit with v free (flat χ², not a measurement — the whole C17 arm scatter). Ladder is arm-independent and
  t0-prior-independent.
- **det3 consistent:** pivot ladder 36.01 ± 0.13 (X) / 35.75 ± 0.15 (Y) on 1300 events vs timing 35.8 ± 0.6.
- Starting angle: χ² single-welled along w at the right p0; p0 = mesh end, ~5 mm from the charge → diagonal valley.
  `heldout_eval.py` used a BARE start (X ladders 1–4 % low; C17 arm ranking + C18 numbers made with it); production
  `wft.reco._global_start` was mostly fine. New `wft.reco.PIVOT_START` (`WFT_PIVOT_START=1`, opt-in, tests pass):
  w scan through the model-free charge centroid; same cost, fewest wrong-basin fits. `heldout_eval --start reco|pivot|bare`
  (default reco now). RC-kernel t0 is a 5 ns staircase; the ±5 ns t0 prior is swamped by χ² ~1e5.
- det2/4/6/7 don't close for 4 measured reasons: no dead masks/gains in ANY bench bundle (small effect;
  `seedfit/strip_health.py`, `mask_bundle.py`); det7 X connectors 0/1/3 channel order scrambled (angles lost there);
  v(x,y) non-uniform on det4/6/7 (ladder maps + stacked-T maps anti-correlate); RC kernel fits det4/det7 ~2× worse χ²
  than the old lp/delay kernel, whose ladder matches timing. det6: ladder × T → gap ≈ 27.5–28 mm, v ≈ 24 (timing 25.9 assumed 30 mm).

**Decisions waiting for Dylan:** (1) PIVOT_START on for the pass (recommended); (2) arm D for det3/det2; for det4/det7 gate
the RC calibration against the old kernel's χ² first; (3) mask det7 X connectors 0/1/3 for angles (scramble not mappable
with ~150 ev/connector); (4) carry dead masks as a bundle feature or not; (5) per-region v if det4/6/7 depths are used.
Code snapshots on lxplus (`~/x17rerun/code_*`) predate PIVOT_START — make a new one before using it on condor.
Local study data: `~/x17/cosmic_bench/vladder/` (laptop; caches copied from the it0 EOS rows, results JSON, report/).

**Done overnight (10-23/24), records rerun/CHECKS_2026-10-23.md C10–C17:**
- C11–C14: June re-decode + audit (1278/1278), re-hit (rr1_cm0), M3 re-audit (production rays stand), W10 path layer.
- C15 W11: masked CM not adopted (plain median everywhere); masked−plain constant shifts = CM systematic.
- W9 SPS: FEU 03 of all 38 SPS sub-runs decoded into `decoded_root_fa80c06/` (280/281; one 0-byte file); 98 RAW
  files audited, all identical. (SPS analysis path layer not done.)
- W7/W8: `rerun/pass_registry.csv` (make_registry.py): 170 rows, 20 condition groups, 5 golden + 14 v-refit groups.
- Pass DAGs: `rerun/make_pass_dag.py` phases finalize / vrefit / pass; tested end-to-end in the dry-run tree
  (vrefit on det3 700 V: hits cache → reco → alignment → cache → v fit → reco → angles → w0/kw; pass on sat_det3 +
  an HV-scan row: reco → alignment → efficiency/angles/maps with the group alignment). All rc 0.
- W6: `rerun/w12/set_w0kw.py` (w0/kw under the final bundle; the r06-inherited ones were up to 0.8° / 9 % off).
- 02_efficiency drops decoder-incomplete events from the denominator (incomplete_events.json from the stager).
- C16/C17 calibration: see below.

**Calibration, the state of play (C16, C17):**
- Pinning the reference to the waveform alignment does not work: the alignment places the plane only to 1–3 mm in z,
  which a pinned fit turns into 5–20 % of v. Iterating cannot fix it.
- With v free, methods scatter 15–25 % (det4 A vs B 27 %). A global plane offset (dz) falls into a v–dz valley.
- **v fixed at the timing v (arm D)** gives the same or lower training χ² and the best held-out resolution on every
  chamber. Recommended protocol: v from timing (vdrift_consistency, per chamber/condition), kernel σ0_x, σ0_y,
  D_rc,y from the waveform fit with p0 profiled ±2 mm, w0/kw measured under the final bundle.
- Stability of D under one more reference rebuild (C17 addendum): resolution and bias stable to 0.03°, σ0 ~5 %,
  D_rc ±4 % (det2/3/6) but ±17–19 % (det4/7, flat direction). Final arm-D bundles: `<golden rel>/wft/w12/it1/timeD/bundle`.
- Caveat: timing v has 2–4 % errors and det7 has two inconsistent values (36.8 long run vs 32.8 longer run).

**To start the pass once the protocol is chosen (all on lxplus, ~/x17rerun/pass):**
1. golden: `make_pass_dag.py finalize <code> --golden det2=it1 det3=it1 det4=it1 det6=it1 det7=it1
   --bundle-sub timeD/bundle` (arm D), then `condor_submit_dag finalize.dag`.
2. vrefit: under arm D the v-refit groups need their own timing v (port headon_cache + unsat_width to stage_job;
   14 groups) instead of the v fit in `run_fitv.sh`.
3. `make_pass_dag.py pass <code>`: 655 nodes, RC reco ~1 h per 4 k events on 16 cores.

**Gotchas (new):**
- DAGMan reads the .dag at submit: edit-and-resubmit, never edit a running DAG.
- Fit jobs must not `rm -rf $_CONDOR_SCRATCH_DIR` itself (exit 1, logs lost) — fixed in run_fit*.sh/run_arm.sh.
- RC reco is 2–3× slower than the delay kernel.
- The bundle summary now prints [RC] constants; older logs show the unused delayed-copy ones.

## X asymmetry contingency: IN FORCE — updated 2026-10-23 (dylan-Yoga)

**Resume:** mechanism closed unexplained (board 10-23); the plan is in force and its toy check is done (§43). Next: the
D1 requote on the desktop with `sigma_p0_x = 0.25` added to the r06 bundles, then the paper sections.

**Goal:** carry the unexplained X neighbour asymmetry through detector characterization, n_TOF and the papers as a
measured property and a systematic, not a model term (`sps_beam_test_26/analysis/cloud_basics/XASYM_CONTINGENCY.md`,
status table at its end).

**Done 2026-10-23:**
- Toy injection (`cloud_basics/xasym_toy.py`, FINDINGS §43, `results/xasym_toy_det3.json`): det3, production r06,
  the measured copy (0.13 × centre, 114 ns late) injected and refitted symmetric, 400 events × 2 seeds. Angle −0.05°
  overall, ∓0.19° head-on (|tan| < 0.05), +0.007° on the resolution; position pulled 60–70 µm toward the copy side
  (mostly a board offset, ±50 µm with angle). σ0_x 0.25 halves the overall cost against 0.42 (χ²/dof 6.6 vs 17.8);
  head-on unchanged. Head-on aQ the copy adds: +0.160 on σ0_x 0.25 (data +0.165), +0.118 on r06's footprint.
- n_TOF chamber A (`sept26_prelim_analysis/xasym_pointing.py`, run_145): capsule crossing moves ≤ 0.09 mm, ≤ 0.003 mm
  after re-alignment (crossing error ±0.31). C: flat ±0.1 mm → ±0.05 mm on the A+C capsule X.
- C2 decided per the plan (PAPER_PLAN C2 = ready; X1 added; tripwire row added); paper1 §8 has the asymmetry
  subsection outline, paper2 §4/§5 the mismatch + toy cost; paper1 pushed to Overleaf (4bb41fc; dash recursion workaround in papers/README.md).
- Deck republished with a toy slide: https://dylan-neff.web.cern.ch/notes/mx17-xasym-plan.html.

**Next steps:**
1. Desktop: D1 requote with `sigma_p0_x = 0.25` on the five golden keys (det3/det2/det7 r06; det4 lp_t0p, det6 lp):
   copy each bundle to `calib_bundle_r06s25` with the key added, rerun mx_june_wft 01–04, rebuild FLEET_DIGEST. §38
   predicts no overall resolution cost and slopes 0.007–0.030 closer to 1.
2. Optional: the same toy on det2/det7 (their aQ 0.08 / 0.19) for the paper's per-chamber range; needs their caches
   (desktop).
3. n_TOF: apply `xasym.correct_x` in the vertexing/alignment rerun as the ± variation; the raw-ADC session scale for
   chamber A is still unmeasured (bench range 0.44–1.57 used).

**Gotchas:**
- At |tan| > 0.15 the mesh p0 is not a usable position metric in toys: half the fits move t0 by one 60 ns bin and p0
  slides by w × 60 ns even in closure. Compare positions at the charge-centroid time.
- The laptop now compiles the papers locally (user-local TeX Live 2026, see the MPGD2026 section).
- The desktop was offline all day (Windows booted); everything here ran on the laptop's det3 calib cache and the
  run_145 stage-3 tables.

## July two-source calibration (srccal): run 224591 stale — updated 2026-10-21 (dylan-MS-7C84)

**Resume:** Deferred. Run 224591 changed on EOS after the July caches were made; re-run srccal + srccalfit with the tracking pass.

**Goal:** the July plastic-bar energy scale (Y-88 + Cs-137, `mx_july_beam_qa/SRCCAL_RESULTS_2026-07-28.md`).

**Done:**
- Tested all condor wrappers on the lock venv against their LCG_105 outputs (srccal 9 jobs, srccalfit, readpass run224401, run58). They match except the run 224591 count below. Wrappers switched (`2bb0b20`, `282fdd8`, `e4b8428`).
- Found: run 224591's raw file on EOS was rewritten 2026-08-07. Trigger count 544 today, 621 in the July caches and `srccal_runs.py`. The venv and LCG_105 agree on 544. The BL and DL edge rates are about 14 % off for that run.
- Recorded on the X17 board (open question, log entry, deferred re-run) and at the top of `SRCCAL_RESULTS_2026-07-28.md`.

**In progress / where it stopped:**
- Nothing running. The re-run is not started. The current caches for the new count are in `~/x17lockvenv_test/srccal/new/cache` on lxplus (outputs of the venv srccal jobs, cluster 4526323).

**Next steps:**
1. In the coming-months tracking pass: re-run srccal (`mx_july_beam_qa/lxplus/srccal.sub`) for the nine runs, then srccalfit, on the current EOS files.
2. Update the hand-typed 621 in the `srccal_runs.py` table (the code reads the count from the file, so the table is documentation only) and the results markdown.
3. Confirm whether 224591 feeds the energy calibration, or only the per-bar rates.

**Gotchas / decisions:**
- Run 224461 is gone from EOS; the July readpass caches are the only copy of its output.
- The reference srccalfit output uses the July caches; the venv run matched it exactly on those caches, so the mismatch is only the input.
- Dylan decided to defer the re-run; do not re-run until he asks.

**Key files & commands:**
- `mx_july_beam_qa/lxplus/srccal.sub`, `srccalfit.sub` — condor submit (run on lxplus in `~/x17qa`)
- `mx_july_beam_qa/srccal_runs.py` — run table (224591 row flagged)
- `mx_july_beam_qa/SRCCAL_RESULTS_2026-07-28.md` — results, with the open warning at the top

## Scintillator flash recovery — updated 2026-10-21 (dylan-MS-7C84)

**Resume:** done for now; two decks live. Next only if Dylan asks (wall per-bar read-out, liquids, plastic timing check).

**Goal:** answer when the X17 scintillators recover from the n_TOF γ-flash (collaborator question, future data
taking in the first µs), as a time and a neutron energy, with what efficiency and precision.

**Done (record: `ntof_scint_recovery/ANALYSIS_PLAN.md`, Round 7 on top, then 6, 5, …):**
- **Collaborator deck** (Dylan's "share" version, plain language, 10 slides + appendix):
  <https://dylan-neff.web.cern.ch/notes/scint-flash-recovery-summary.html> (`make_share_deck.py`).
  Recovery = shape-test time: **walls 2.5 µs / 299 keV** (~82 % of MIPs seen then, 90 % from 4 µs), **plastics
  0.4 µs / 9.4 MeV** (~83 %, 90 % from 0.8 µs; signal size known only from 2 µs), **liquids "not clear: did not
  work as intended"**. Slides: set-up (MPGD26 plan view), traces, the full/half-intensity idea, spectra, shape
  mismatch, signal size, particles seen, pile-up (4 bars per wall channel; one bar per channel ≈ 95 % at 2.5 µs),
  precision; appendix: the ~1 µs bump = fast neutrons (`bump.py`).
- **Detailed note** <https://dylan-neff.web.cern.ch/notes/scint-flash-reconstruction.html> (`make_early_deck.py`):
  MIP view, real-pulse linearity, what rests on injection; per-size results and long-time recovery as appendices.
- New code: `linearity.py` (dedicated-vs-parasitic and vs-20–60 µs spectrum shape; wall bar ends),
  `mip_eff.py` (MIP spectrum × gain × injected efficiency; gain bounds before 2 µs), `bump.py`, `make_share_deck.py`.
- Bug fixed in `early.py`: gain uncertainty now relative in σ(E) (was absolute; ×3 understated at gain 0.33).

**In progress / where it stopped:** nothing running. The last injection condor job (cluster 4521469) from the
earlier round was never remerged; it changes nothing visible (README commands if wanted).

**Next steps (only if asked):**
1. Wall per-bar read-out estimate assumes even occupancy over the 4 bars of a group; could check with MM tracks.
2. Plastic timing is injection-only (no real-pulse check); liquids' linearity not established (ded/par 10–20 % off at all times).
3. MeVee anchor (3.45 MeVee = median of ALL plastic coincidences) sits ~15 % below the through-going-muon median: revisit if MeVee numbers are reused.

**Gotchas / decisions:**
- Dylan: recovery time = shape test (ded vs par), quoted with the MIP efficiency at that moment; not the 90 % MIP time.
- The ded/par shape test is the strong evidence (Dylan: "exactly what we needed"); the vs-20–60 µs test is invalid before ~2.5 µs (particle mix changes).
- Plastic MIP spectrum: the wall–plastic coincidences have a 15–33 % low shoulder (clipping tracks); use the fitted through-going peak.
- Wall t = 0 is right: the walls are gated off during the flash and reopen at ~0.6 µs (divert gate), t0 from the divert-off runs.
- Wall channels are 4 read-out groups of 4 bars per arm (16 groups total), not 16 single bars.
- slidedoc `Plot.text(x, y, …)` takes DATA coordinates, not pixels.
- `templates.npz` is force-added despite `*.npz` in .gitignore. `pkill -f <pattern>` kills your own shell if the pattern is in the command.
- Condor: judge completion by `xrdfs ls` file counts on EOS, not `condor_q`. Production = runs ≥ 224488; count only live channels.
- Local products: `/media/dylan/data/x17/scint_stack/recovery/{inject,gain,g4mix,reco,early,linearity,deck,pulsehist_v3,summaries,raw}`.

**Key files & commands** (all in `ntof_scint_recovery/README.md`):
- `python -m ntof_scint_recovery.{early,linearity,mip_eff}` then `make_early_deck` / `make_share_deck`
- `python -m ntof_scint_recovery.bump <local flash_*.root>` (appendix numbers)
- publish: `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <deck>.html --slug <slug> --force --deploy`
- dataset tool `ntof_processing/flash_pull/`; condor submit dir lxplus `~/x17rec`.

## B4C target (July) pair expectation — updated 2026-10-20 (dylan-Yoga)

**Resume:** Done: B4C night (run_39/40) can't show pairs; deck live. Optional: free 185 GB of sim output, chase occupancy gap.

**Goal:** Stephan (2026-10-19) asked whether the early-July non-³He targets, especially the B4C night on the
scintillator trigger, could give pairs. Answer with existing data + Geant4.

**Done:**
- Inventory: B4C = run_38 (flash only), run_39 + run_40 (93 k scint triggers, 2635 pulses); "Marex" runs 25–37.
- Geant4 (`MX17_Full_Geant` main, merged from `b4c-slab`): EAR2 beam on a 2.5 mm B4C disc (Ø100/Ø50), measured
  EGAF ¹⁰B(n,γ) cascade, Born IPC per ¹¹B line, ³He reference pairs in the same geometry.
- Result: 1.6 IPC pairs/pulse made, 0.03 triggered two-chamber pair events in the whole night; leptons of
  ≤ 11.45 MeV transitions stop before the plastics. ³He capsule Al already makes ~24× more such pairs per pulse.
- `ntof_b4c_target/` (README has everything), report `/media/dylan/data/x17/b4c_target/report.html`,
  slide note https://dylan-neff.web.cern.ch/notes/b4c-target-pairs.html

**In progress / where it stopped:** nothing half-done.

**Next steps (optional):**
1. Ask Dylan before deleting `/eos/experiment/ntof/data/x17/full_sim/b4c_july/ipc_r*` (148 GB step trees;
   reduced parts in `/eos/user/d/dneff/scratch/b4c_sim/parts*` are all the analysis needs).
2. Open: sim chamber occupancy (478 keV Compton electrons, ~1900 lit/arm/pulse) vs data's few % multi-chamber
   fraction — criteria not matched. Would need the data criterion applied to the sim (or a 3 keV-like one to data).

**Gotchas / decisions:**
- Geant4 11.2 ParticleHP ¹⁰B(n,γ) is not energy-conserving; `/process/had/particle_hp/use_photo_evaporation true`
  conserves but has the wrong branching → cascade generated from EGAF via `--gamma-lines E1+E2:w`.
- Geant4 MT (-t >1) aborts with ROOT "gInterpreter not initialized" in this build: run -t 1.
- lxplus can't push to GitHub (no creds): fetch the lxplus worktree branch over ssh from the desktop and push there.
- Data multi-chamber fraction uses the hits-tree event count (71 k), not all 93 k triggers.

**Key files & commands:**
- `python -m ntof_b4c_target.expect` / `.deck_data` / `.make_report` / `.make_b4c_deck`
- `ntof_b4c_target/lxplus/` — condor job + reduction scripts (cluster outputs under `full_sim/b4c_july/`)
- lxplus build: `/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_b4c` (worktree, branch b4c-slab)

## MPGD2026 proceedings (JINST) — updated 2026-10-20 (dylan-Yoga)

**Resume:** first draft is on Overleaf; Dylan says it "looks good to start". Next: fill the cyan `\askDylan` items, then polish.

**Goal:** 5-page JINST proceedings for the MPGD2026 oral talk ("Searching for X17 at n_TOF with large-area Micromegas
detectors in µTPC mode", 3 Sept 2026). **Deadline 17 Jan 2027**, submitted on jinst.sissa.it → "submit a proceedings contribution".
Limit is 5 pages, excluding authors/affiliations, acknowledgements and references.

**Done:**
- `papers/mpgd26_proceedings/main.tex`: full draft on the official JINST template (jinstpub.sty, line numbers, `\notoc`).
  It follows the talk: motivation → MX17 chamber + resistive-strip sharing → waveform forward fit → bench performance
  (93.1 % on 21,953 muons, σ68 1.19°/1.23°, 33 ns, SPS 176 ± 35 µm) → n_TOF setup and campaign (41.8 M events, 17.9 TB,
  436,373 pulses, 78.6 %) → γ flash (chamber usable again after 2.0 µs vs DREAM 5.0 ms, ×2.4e3) → thermal window →
  preliminary capsule pointing (A and C converge ~9 mm off axis). 5 figures from `mpgd26/slides/assets/img/`, 11 refs.
  Body is pages 2–6.
- Linked to Overleaf: https://www.overleaf.com/project/6ad72207b122b98dea142d04 (remote `overleaf-mpgd26_proceedings`).
- Overleaf git token is now on all three machines: laptop and Ubuntu desktop `~/.git-credentials`; Windows
  `C:\Users\Dylan\.git-credentials` with a store helper scoped to git.overleaf.com (Credential Manager stays the default).

**Next steps:**
1. Dylan: affiliation and email, co-authors and collaboration wording, EUROLABS/other grant numbers, DREAM reference (two options in the bib).
2. Decide the physics framing (see the `\askDylan` in §4). No absolute X17/IPC yields are quoted on purpose: the rate tables may be ~100× high (`sept26_prelim_analysis/IPC_MISSING.md`).
3. Re-render `chamber_exploded.png` with larger labels for print; the other figures are slide-style plots and read fine.
4. Check page count on Overleaf (real newtx fonts), and fix the one 7.5 pt overfull line in §2's forward-model paragraph.
5. Before submission: remove all `\askDylan`/`% TODO Dylan`, then submit.

**Gotchas / decisions:**
- The laptop now has a user-local TeX Live 2026 (`~/texlive/2026`, on PATH via ~/.bashrc), so it compiles all three papers as Overleaf does.
  jinstpub.sty for `\RequirePackage{mathptmx}` (same Times metrics). Never commit that swap; Overleaf compiles the real one.
- `\askDylan{}` inside `\emailAdd`/`\affiliation` overflows TeX's input stack (mailto link), so those placeholders are plain text.
- The capsule offset quoted (~9 mm) is the September per-run imaging (−9.3 ± 0.3 mm), which matches the talk's "~1 cm".
- `figures/efficiency_breakdown.png` is copied but unused (spare, in case a figure is swapped).
- Windows desktop ssh from the laptop: `ssh -o HostKeyAlias=desktop-win-lan dylan@192.168.0.89` (shared IP with Ubuntu).

**Key files & commands:**
- `papers/mpgd26_proceedings/main.tex`, `README.md` (workflow), `figures/`
- `papers/overleaf.sh pull mpgd26_proceedings` / `push mpgd26_proceedings`

## MX17 papers on Overleaf — updated 2026-10-19 (dylan-MS-7C84)

**Resume:** Paper I is linked to Overleaf (git bridge). Next: share with Stephan + Track changes, then link Paper II.

**Goal:** Overleaf as the editing/review home for the two MX17 papers, kept in step with `papers/<paper>/` by `papers/overleaf.sh` (git subtree).

**Done:**
- Created Overleaf project "MX17 Paper I" (id 6ad61186fe591228d14dac68) via Claude in Chrome (Dylan's browser "dyn040294").
- `papers/overleaf.sh link paper1 https://git.overleaf.com/6ad61186fe591228d14dac68` done; repo and Overleaf `main` in step; pushed to origin main.
- Git token stored by Dylan in `~/.git-credentials` (600, store helper for git.overleaf.com).

**Next steps:**
1. Overleaf, Paper I: Share, add Stephan as Editor; Review, Track changes on for everyone; settings: pdfLaTeX, main document `main.tex`. Check the project compiled and the files arrived.
2. Create a blank "MX17 Paper II" project and `papers/overleaf.sh link paper2 <url>` (commit first; the script needs a clean tree).
3. Delete `~/Desktop/overleaf_git_token.txt` (token already stored).

**Gotchas / decisions:**
- Overleaf's git bridge uses branch `main`, not `master`; `overleaf.sh` fixed (7dda16c).
- GitHub Sync not used: it maps a whole repo root to a project, and the papers are subfolders of this monorepo. Direct git bridge is simpler.
- The account shows "free plan" but has the Git integration section and token generation, and it works.
- Did not link the UCLA/CERN institutional-account banners (account change, Dylan's call).
- Tracked changes don't travel over git; accept/reject in Overleaf before `pull`.

**Key files & commands:**
- `papers/overleaf.sh` (pull / push / sync / link), `papers/README.md` (workflow)

## Future-X17 meeting deck (19 Oct) — updated 2026-10-19 (dylan-MS-7C84)

**Resume:** round 3 republished; Dylan says slides "look much better". Next: any further feedback, else done.

**Goal:** informal-meeting deck on MX17 status and where to search for X17 next (LNL, neutron facilities).

**Done:**
- Rounds 1–3 of Dylan's 19 Oct feedback (verbatim + item log in `talks/future_oct26/FEEDBACK_2026-10-19.md`).
- Round 3: slide 4 tables from Geant4 truth (`g4_breakdown`): singles per chamber A–D, e⁺e⁻ pairs same / adjacent /
  opposite (opposite = 0, < 2 k at 95 %), and both-legs-reach-wall. Cover stat = data; slide 4 = Geant4 × digitised
  reco eff. Kernel maps on slides 2–3; scint maps with a σ 8 mm kernel (slide 5) + one-line wall explanation
  (MM efficiency cancels; bands = 4 SiPM read-out groups). Legends added (slide 6 bars, LNL slide); skill rule
  "plots readable without hover" (claude-skills a0fe46e).
- Live: https://dylan-neff.web.cern.ch/notes/mx17-future-x17.html (12 main slides + B1–B9).

**Next steps:**
1. Apply any further feedback from Dylan.

**Gotchas / decisions:**
- Geant4 sim arms: 2 = A (+z), 3 = C (−z), 0 = D (+x), 1 = B (−x). G4 has external conversions only (IPC ≈ +18 %).
- `ntof_scint_stack/ana.py`: `calibrated()` was factored out of `run_arm` for `map_tracks`; map_tracks reproduces
  eff_map counts exactly, but the full `ana` run was not re-run after the refactor.
- Caches (delete to rebuild): `<OUT>/cache/{g4_breakdown.json,scint_map_tracks.parquet,singles_height.csv}`,
  PNGs in `<OUT>/img`; OUT = /media/dylan/data/x17/future_oct26.
- Open from round 2: Demokritos card is "?" (no study); pairs slide is vs ToF, not pointing origin (no per-leg directions).

**Key files & commands:**
- `talks/future_oct26/make_deck.py` — generator; `.venv/bin/python talks/future_oct26/make_deck.py`
- `.venv/bin/python talks/future_oct26/render_check.py <html> <outdir>` — crops each frame; look for clipped bottoms
- `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py /media/dylan/data/x17/future_oct26/mx17-future-x17.html --slug mx17-future-x17 --force --deploy`
  then commit the site repo (branch `master`).

## Detector model basics: dead strips, gap, v(E), saturation, X asymmetry — updated 2026-10-19 pm (dylan-MS-7C84)

**Resume:** FINDINGS §35–42 + slide note <https://dylan-neff.web.cern.ch/notes/mx17-basics-oct19.html>
(`cloud_basics/make_basics_deck.py`). Kernel freeze hold LIFTED 2026-10-23 (r06 frozen, XASYM_CONTINGENCY.md). Next: Dylan's read of the note; then the
amplitude-dependent template (saturation charge) and the D_rc-vs-delay test of the X asymmetry mechanism.

**Done 2026-10-19 (pm):**
- §39 inclined-track dead-strip test (`strip_gain_audit.holes`, `bad_set`): det7 ~20 more dead X channels (67–250);
  masks now used by gap_extent and rc_diffusion. `gap_extent.py --satmask` keeps saturated events. gaps.py adopted
  mean of the two selections: det3 28.08 ± 0.19, det2 31.12 ± 0.42, det4 30.48 ± 0.44, det7 29.44 ± 1.12.
- §40 `vdrift_consistency.py`: det3 scan meshes (one water 0.955 %, χ² 4/5; timing+Magboltz gap 28.25 [27.75, 28.75]).
  det2/det4 D_T-match gaps equal the timing-free ones; det3's D_T excess (+160…+260) is at every field. E = V/gap now
  (was V/2.88 everywhere). Beam: ladder v is estimator-dependent (7.77 vs 10.0 µm/ns) → no det4 beam gap test.
  `beam_comp_fit.py --gap` exists but its χ²(gap) is multi-minimum: do not use as a profiler.
- §41 `sat_closure.py`: censoring keeps angle/position (resolution unchanged); charge 5–12 % low, not fixed by
  `wft.model.SAT_SOLVE` (opt-in) → template amplitude dependence.
- §42 X asymmetry: det3 model-free one-sided −1 copy delayed ~110 ns at the time-symmetric angle; no 31.2 mm beat;
  det4 beam test band-dependent (gain-confounded, inconclusive). Hypothesis: resistive strip/pad offset + across-strip
  RC (τ = (w/2)²/D_rc). Another session's quiet-event study is in `/media/dylan/data/x17/nb_asym_raw` (not mine).

**Next steps:**
1. Amplitude-dependent X/Y pulse template from bright unclipped pulses; re-run `sat_closure.py` (target: charge bias ~0).
2. X asymmetry: per-chamber delay vs 1/D_rc (det2, det6 need a time-symmetric sample at their own tilt);
   hand the hypothesis to the simulation side; read nb_asym_raw's crosstalk result before any model term.
3. det3's D_T excess: the one open bench inconsistency. Candidate checks: iso fraction (iso_fit.py grid), det3
   amplification-region spread (σ0 vs D_T degeneracy at fixed σ0).
4. Re-run unsat_width/bench_fit with E = V/gap (bench_fit.py CH still uses 347/208/243 V/cm).
5. Older list still open: asymmetric X footprint in build_matrix_rc (wait for 2.), per-event censored width fit,
   SPS run_71 det4 masking in beam channel numbering.

**Gotchas:**
- The head-on health test judges few channels on det2/6/7: always use `bad_set` (adds the inclined-track holes).
- `xw_clean.py` clean subsets need `XW_MIN_EV=60` on det4.
- `beam_comp_fit.py` without `--gap` keeps the old field labels; with `--gap` it writes `beam_comp_fit_<src>_gap<g>.json`.
- / (nvme) is nearly full: outputs on /media/dylan/data (= ~/x17/cosmic_bench).

**Key files:** `cloud_basics/{strip_gain_audit,gainmap_export,gap_extent,gaps,vdrift_consistency,sat_closure,
xasym_tests,xasym_samechannel,xasym_boundary,make_basics_deck}.py`; `wft/model.py` (SAT_SOLVE).

## Detector spec page for Stephan's review — updated 2026-10-16 (dylan-Yoga)

**Resume:** page live with comments on; next: Dylan posts one test comment, sends Stephan the link, fold answers into spec.py + Geant4.

**Goal (Dylan):** before publication, a visual, clickable page with every detail of the MX17 module as we model it,
so Stephan can check it and answer the open questions directly on the site.

**Done:**
- Live: <https://dylan-neff.web.cern.ch/notes/mx17-detector-spec.html> (unlisted). Source `docs/detector_spec/`
  (README there): side view of the full Geant4 stack, top view with zoom insets, the amplification region to scale,
  17 open questions ranked by impact, all 97 numbers with a status (9 unknown, 31 assumed), the four chambers,
  a glossary (calendering defined).
- Questions asked as Dylan requested: resistive strip width/pitch (550/800 µm from memory, vs the 780 µm readout
  pitch), strip thickness (10 µm?), sheet resistance, kapton + glue under the strips, mesh spec (19/48 µm is a
  P2 placeholder) and calendering, window mylar 60 µm single/double aluminised and nm of Al, PCB stack-up,
  field cage, the ~19 mm Y dead bands.
- Comments: the budget page's Apps Script (`x17_facility_search/budget/feedback_apps_script.gs`) now routes
  `page: "detector"` to a "Detector" tab of the v2 budget feedback sheet. Dylan redeployed it (same URL) on
  2026-10-16; verified `?page=detector` answers and the budget GET is unchanged.

**Next steps:**
1. Dylan posts one test comment from the page, then deletes that row in the sheet's "Detector" tab; sends Stephan the link.
2. When answers arrive: read the "Detector" tab (Drive `read_file_content`), update `docs/detector_spec/spec.py`
   AND `MX17_Geant/shared/MX17ModuleGeometry.hh` (+ `design/NEEDED_INPUTS.md`), then
   `python3 docs/detector_spec/build_page.py --publish`. Mark handled rows `status` = done.

**Gotchas / decisions:**
- Flagged on the page: Geant4 puts the L3 guard ring + 264 µm FR4 between the kapton and the pads; the response
  model laminates kapton straight onto the pads. The FR4 thickness is a residual of the CAD's 1.70 mm board.
- The page only POSTs after a GET returns `page: "detector"`; otherwise it falls back to mailto, so detector
  comments can never land in the budget tab.
- Not tested: a live POST (would write a real sheet row). Mock-backend test passed.
- Side-view band heights are log-scaled; a true-scale bar sits beside it.

**Key files & commands:**
- `docs/detector_spec/spec.py` — all data; `template.html` — the page; `build_page.py [--publish]`.
- `DETECTOR_FEEDBACK_ENDPOINT=http://127.0.0.1:8766/mock python3 docs/detector_spec/build_page.py` — mock test.

## UCLA disk rescue + reuse — updated 2026-10-17 (dylan-MS-7C84)

**Resume:** Dylan runs `sudo bash /home/dylan/x17_local/rescue/reuse_disk.sh test` (erase + full write/read test), then read `reuse_disk.log`; if the media-error count didn't move, run `... format`.

**Goal:** get the 26 unreadable files off the failing UCLA NVMe (ADATA SX8200PNP, `/dev/nvme1n1p2`, NTFS),
then decide whether the disk can be reused as scratch space for analysis files that have a copy elsewhere.
Not X17 analysis; the notes live here because this repo is the one reachable from every machine.

**Done:**
- 26 files (26.67 GB) rescued, **1,028,096 B unreadable** (99.996 % recovered). Method: each file's NTFS cluster
  runs (`ntfsinfo`) → a read-only device-mapper device → `ddrescue -d` (pass `-n`, then `-r3`).
- Copied to `/eos/experiment/ntof/data/x17/research_old/ucla/<relpath>`. **All 26 match by name, size and md5.**
- Records + scripts on EOS in `research_old/ucla/_rescue_2026-10-16/`: `damage_report.txt` (what each hole hit),
  ddrescue `maps/`, `files.tsv`, `smart_before.txt`, the scripts, and the rescue's own `HANDOFF.md`.
- Damage: all ROOT files open. Each hole costs whole compressed blocks (baskets), which wipes 1–2 branches over an
  entry range: about 2.4–3 % of entries in the big Cooper-Frye trees, 22–34 % in the proton-only ones, and all of
  one branch in the two small AMPT files. Dca_xy_QA files lose 1 and 5 objects of about 2,000. The text files lose 23 (11.5 GB CSV)
  to 456 lines. Dylan declined partial-basket recovery (2026-10-16).

**Done 2026-10-17:**
- Whole disk compared with EOS `research_old/` by name and size: x17match (18,292 files), Saclay (91,464),
  x17_resp (16), Research leftovers (31, = `ucla/`) all match; mx17_response_sim matches except the 13 tiny
  `results_meshfield_nc1_20261016` logs Dylan said to drop. Other top-level items: chkdsk `found.000`, an old
  mirror of `~/upgrade-kit` (the home copy is newer), Windows junk.
- Deleted (Dylan's OK): staging `/media/dylan/data/ucla_rescue/` (data disk now 31 GB free) and the disk's
  `android/avd` (pixel_9_api34) + `gradle/` cache. `~/.bashrc` still points `ANDROID_AVD_HOME`/`GRADLE_USER_HOME`
  at `/media/ucla`: Gradle rebuilds its cache on the next build; recreate the AVD in Android Studio.
  Android builds fail while the disk is unmounted for the test.
- Decisions: erase from Linux, reformat **NTFS** (Windows access). Firmware: Dylan wants it updated (2026-10-17).
  Do it on Windows with ADATA SSD ToolBox **before** `reuse_disk.sh test`, so the test runs on the new firmware.
  The disk holds nothing needed now, so a failed flash loses no data. Record the new version here.
- Wrote `~/x17_local/rescue/reuse_disk.sh` (checks the drive serial 2K34291HANDC first):
  `test` = SMART → umount → blkdiscard → `badblocks -wsv -t random` → blkdiscard → SMART;
  `format` = GPT + one NTFS partition "ucla" with the old serial `CE90B08890B07913`, so the fstab UUID still
  matches; it also changes the fstab line to `defaults,nofail 0 0` (no boot hang if the disk dies). Log: `reuse_disk.log`.

**In progress / where it stopped:**
- Dylan is switching to Windows now to do the firmware (said at wrap-up 2026-10-17). On Windows the only step is
  ADATA SSD ToolBox; `~/x17_local/rescue/` (script, logs) is on the Ubuntu partition and not visible there.
- Next: Dylan boots Windows, updates firmware in ADATA SSD ToolBox (currently 42B2S7JA), then back on Ubuntu runs `test` (sudo, tmux, hours long). Not started.

**Next steps:**
1. After `test`: compare "Media and Data Integrity Errors" in `smart_pre_erase.txt` vs `smart_post_test.txt`
   and check `badblocks_found.txt` is empty. Unchanged + empty → `sudo bash .../reuse_disk.sh format`. Otherwise retire it.
2. Use it only for data with a copy elsewhere. Monthly: read everything back + check SMART; retire if errors on new data.
3. When the laptop is on: move `x17/sept26_fullpass` + `x17/sept26_slim` back from `yoga:/media/dylan/data/x17_hold/`
   to `/media/dylan/data/x17/` (29 GB; 31 GB free), verify names/sizes, then remove from the laptop.

**Gotchas / decisions:**
- Diagnosis: wear 2 %, spare 100 %, all logged errors are unrecovered reads (status 0x281), and the bad files date
  from 2022–23 → most likely charge loss on data left unused for years, not worn flash. An SSD replaces a bad spot only when it is rewritten,
  so an erase should clear it. The error count rose during the rescue because of our own retries, not because the drive got worse.
- Holes are 4 KiB-aligned and zero-filled in the rescued files; text holes collapse many lines into one NUL line.

**Key files & commands:**
- `~/x17_local/rescue/` (Ubuntu only; copy on EOS above): `rescue.sh pass1|pass2|finalize|status`, `parse_runs.py`,
  `damage_report.py`, `HANDOFF.md`.
- `sudo smartctl -a /dev/nvme1n1` — health; compare the media-error count.

## Track passport (tracking + MM dE/dx + scintillators per track) — updated 2026-10-15 (dylan-Yoga)

**Resume:** published 2026-10-15 as a 26-slide pedagogical note (principle → simulation → real detector): https://dylan-neff.web.cern.ch/notes/track-passport.html (`ntof_calorimetry/make_passport_deck.py`). The long report is `/media/dylan/data/x17/calorimetry/passport/report.html`.

**Goal:** Dylan asked for as much information as possible about each track, from tracking and energy together. It is a demonstrator of the set-up, not X17 physics.

**Done (results in `ntof_calorimetry/MM_EDEP.md`, "Track passport"):**
- `mm_edep_ana` ran for A+C. The beam MM calibration is fixed (beam-sim k; sim bias −0.46 → 0.00), giving per-track σ = 0.16 (A) and 0.22 (C). Stage 4 was rerun at σ 0.16.
- `track_passport`: 15,444 run_145 tracks (12,818 with an MM deposit). Cosmic path length: MM sec^0.97/1.03 (Heed 1.07), wall sec^1.0/1.3. The MM tags tracks with several particles in the gap (AUC 0.84); the wall cannot (0.47). Data and Geant4 agree that MM and wall are uncorrelated. The track angle improves the plastic energy for steep tracks (1.40 → 1.07 MeV) and for multi-particle tracks (1.70 → 1.08 MeV). Capsule-vs-elsewhere AUC from the geometry is 0.75. Example cards for 12 data tracks are in the report.

**Next steps:**
1. (done 2026-10-15: logged on the x17 board as a log entry and an output.)
2. Add time information: MM t0 vs scintillator pulse times per track (stack `*_dt_on`), as a fourth timing measurement.
3. Get a Geant4 reduction with the liquid (`stack_edep.py` on branch `lnl`). Geant4 "through the plastic" is T > 6 MeV for now.
4. The 2026-10-14 slow-tag AUCs (0.50/0.59/0.63) did not reproduce (now 0.74/0.66/0.80). Trace why.

**Gotchas / decisions:**
- The desktop data are reachable from the laptop through Windows on the same disk: `ssh desktop-win` → `D:\x17\...` (Ubuntu `/media/dylan/data`). Copied to the laptop so far: `calorimetry/`, `sept26_prelim/{kcal,stage3_fullpass/tracks_run_145_*}`, `scint_stack/{tracks/stack_run_145,ana,geometry.csv}` and `ntof_cosmics/g4_angle/neutrons_nose`.
- The desktop's pickles are numpy 2; the laptop venv has numpy 1.26 (the shim segfaults). The laptop copies were re-saved through parquet in a numpy-2 env, and the originals are kept as `*.pkl.np2`.
- Beam electrons scatter between the gap and the wall, so the clean path-length test is the cosmic one.
- Data counter energies are moved onto Geant4's scale (wall_plastic scale fits) before the kNN estimators are applied.

**Key commands:** `PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.track_passport` (~10 min), then `-m ntof_calorimetry.make_passport_report` and `-m ntof_calorimetry.make_passport_deck`; republish with `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py /media/dylan/data/x17/calorimetry/passport/deck/track-passport.html --slug track-passport --force --deploy`. The deck caches the beam-calibration histograms in `passport/deck_cache_beamcal.pkl`; delete it after rerunning the analysis.
- Site repo: pull before any add-note --deploy. A deploy from a stale tree pushes stale pages (the 15 Oct pair-display note had been deployed from the laptop uncommitted while origin was 11 commits ahead; committed now).

## Pair event displays — updated 2026-10-15 (dylan-Yoga)

**Resume:** 7 3D pair displays done and published; next: render more triggers / other runs if wanted (needs desktop stage 3).

**Goal:** 3D GIF event displays of selected pair candidates for the 15 Oct meeting. Each shows TPC charge, tracks, the fired scintillators and the capsule at its inferred pose.

**Done:**
- `sept26_prelim_analysis/pair_event_display.py` (main d55b293), four steps:
  - `candidates`: rebuilds run_145 stat090_0000 stage 3 from the is2_v2 reco, then pairs via `det_a_scint.match_run` + `det_a_intra.pairs` (both unchanged) plus two-arm A–C pairs;
  - `render`: GIF + MP4 + PNG per trigger;
  - `report`: report.html gallery;
  - `note`: self-contained note version.
- Rendered (in `/media/dylan/data/x17/sept26_prelim/pair_displays/gif/`):
  - same-chamber `wall2`: A 24576, A 52903, C 4748, C 49942;
  - two-arm: 38414, 23580, 18841.
- Published: https://dylan-neff.web.cern.ch/notes/pair-event-displays.html (md5 checked).
- 24576 for the slides: https://dylan-neff.web.cern.ch/files/x17_pair_24576.{gif,mp4,png}. The same files are in CERNBox at `/eos/user/d/dneff/x17/pair_displays/`.
- A Google Doc of links, "X17 pair event display 24576 — GIF download links", is in My Drive. Drive got a link doc rather than the GIF because the connector only takes inline base64 and this laptop has no rclone or Drive mount.

**Next steps:**
1. For more events or runs, run on the desktop. Point the code at the real `stage3_is2_v2_cf` and the slim parquet (the script currently rebuilds stage 3 locally, for this one sub-run only).
2. Waveforms for new triggers: `lxplus_cut_script()` in the module makes the lxplus cut script (LCG_105). Copy the npz files into `pair_displays/wf/`.

**Gotchas / decisions:**
- k here is the capsule-focus value: kcal_is2_v1 × (1.009, 1.003) × 1.12, giving A 1.103, C 1.081. The kcal JSONs are only on the desktop. Muon-scale angles are about 12 % smaller.
- TPC charge comes from decoded_root, not hits. Depth = (t − 140 ns − t0)·v/k, where 140 ns is the bundle template peak. It lines up with the fitted tracks on all 7 events.
- None of the 7 passes the vertex cut. Two-arm pairs at ≥170° are probably single through-going particles. C 4748 has Δt0 −281 ns. The page lists these per event.
- pyvista off-screen needs `pl.render()` before `screenshot()`, or every frame is identical. A per-frame GIF palette makes the legend flicker, so the code builds one shared palette.
- `eosexperiment.cern.ch` does not resolve off-site. Cut the files on lxplus instead.

**Key files & commands:**
- `PYTHONPATH=. .venv/bin/python -m sept26_prelim_analysis.pair_event_display render --events s:24576,t:38414` (s: = same-chamber, t: = two-arm)
- outputs in `/media/dylan/data/x17/sept26_prelim/pair_displays/` (wf/ = cut waveforms, 111 MB)

## Repo merges and cleanup — updated 2026-10-14 (dylan-MS-7C84)

**Resolved 2026-10-15:** `cut_scan.csv` was truncated by `yield_gate.py scan --runs run_86` (it rewrote the table with run_86 only). The committed table is correct: a full re-scan reproduces it exactly. The working copy was restored from git; `yield_gate.py` now keeps the other runs' rows on a subset run (branch `yield-gate-table-merge`). The cosmic analysis this table feeds is the is2_v1 check (see `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §15), superseded by is2_v2 (§19–20).

**Goal:** get the x17 repo's branches merged into main and pushed, and clean up the leftover branches.

**Done:**
- `worktree-sps-attachment-transfer` merged into main (e9dac9d, pushed). The HANDOFF.md conflict was resolved by keeping both logs; the branch's 10-08 to 10-11 sections sit before the 10-09 cosmic-bench section.
- `bench-pin-rc` and `origin/mx17-paper-status` merged into main (7a98ab7, pushed). HANDOFF.md: the 10-12 X-width section is at the top; the 10-12 cloud_basics section replaces 10-11.
- Garfield test fix and drift-consistency scripts committed (46aeaad, pushed).
- Local branches that were merged into main were deleted (beam-off-cosmics, same-chamber-oct, y-fix-regen, among others).

**In progress / where it stopped:**
- Still attached and merged: bench-pin-rc, is2v2-compare, mx17-paper-status, two-track-joint-fit, worktree-capsule-scale-kernel, worktree-sps-attachment-transfer. The same branches exist in the `nTof_x17_paper` and `nTof_x17_tt` clones, which are checked out on them.

**Next steps:**
1. Clean up the attached worktree branches once Dylan confirms (see the workspace handoff for the same question).

**Gotchas / decisions:**
- `beam-off-cosmics`, `same-chamber-oct` and `y-fix-regen` disappeared from the `nTof_x17_paper` and `nTof_x17_tt` clones during the sweep. The cause was not found. All three are ancestors of `main` in this repo, so no commits were lost.
- `main` merges were done without review, as Dylan asked.

**Key files & commands:**
- `HANDOFF.md` — the two merged logs
- `git worktree list` — attached branches

## X width excess: done with the data we have — updated 2026-10-17 (dylan-Yoga)

**Resume:** superseded by the 2026-10-19 section above (gap table fixed via gap_extent; kernel freeze on hold).

**Goal:** explain the X drift-width excess so the detector-model kernel can be frozen. NO NEW DATA: the detectors are gone.

**Done 2026-10-16/17** (§30 desktop, §31–32 laptop, §33 lxplus, §34 laptop):
- §30–31: saturation biases v low and D_T high (cut at 3500 ADC); `invert` water-branch bug fixed; det3 gap 27.9 mm.
- §32: det3 excess at every field, cut and |tan| cut; det4/det7 D_T falls ~100 with the amplitude cut.
- §33: closure toy (xw_sim.py, condor 4474928): the estimator has no amplitude bias. Unsat closure D_T +3…+17.
  Centring on the true track inflates by +90…+135.
- §34: signed-tan scan: det3 shows no field tilt (δ ≈ +0.02 everywhere, floor still ~350 against the toy's 298); the
  dish would need a 0.05 rad tilt spread, about 4× the map's steepest gradient. Amplitude terciles: det3 +57…+84 in
  every tercile. det4/det7 classes differ in both directions; det7's faint tercile is edge/empty events. No residual
  baseline. Model-free neighbour/peak ratio: on det4/det7 the core is narrower than linear below ~2000 ADC; det3
  follows the toy. No compression below the ceiling.

- Slide note (14 slides, open consistency questions + ideas on slide 13): https://dylan-neff.web.cern.ch/notes/x-width-excess.html,
  built by `cloud_basics/make_xwidth_deck.py` from results/*.json; rebuild + `add-note.py … --slug x-width-excess --force --deploy`.

**Next steps:**
0. Cheapest consistency fix first: unsat_width.GAP_STUDY uses 30 mm for det4/det6, but GAP_STUDY (K = 22) measured det6 27.85
   (det4 25.55, amplitude-dominated). Put one gap table in one place, requote det6 (v −7 %, D_T +4 %), rebuild the deck.
   The other open questions (det2 vs det3 at 347 V/cm, water per chamber/night, run_71 unsat cut, δ vs alignment) are on slide 13.
1. Freeze the kernel: Dd per chamber from the unsaturated stack at the 3500 cut, with +100 µm/√cm (det3, unexplained)
   and ±100 µm/√cm (det4/det6/det7, amplitude-dependent footprint) as stated systematics. Then the kernel arms on
   unsaturated stacks, and the gate.
2. Update make_model_deck.py slides 4 & 18 (saturation dominant; det3 excess unexplained; det4/7 amplitude-dependent
   footprint), rebuild, republish.
3. Optional, from existing data: does the wft forward model (whose neighbour copies scale linearly) fit det4's
   faint events worse than its bright ones? That would show whether the non-linearity is in the copies or in σ_p0.

**Gotchas:**
- unsat_width.py writes one JSON: never run two at once. Keys: `<key>[@sat<N>][@ref][@tan<T>]`; `--recompare` re-does
  only the Magboltz comparison. `--tan` sets x_width.TAN before the pool forks.
- x_width.det_of / drift_hv resolve `<key>__<subrun>` caches (from headon_cache.py --subrun).
- Any v → water inversion at ≳300 V/cm must use `invert(...)['branches']`.
- `pkill -f` with a pattern that also matches your own shell's command line kills that shell: use pgrep + kill PIDs.
- `o22_long_det3`/`o22_det3` fits sit at v 22.69 ± 0.00 (stuck at a bound): ignore them.
- xw_sim: `prep` rebuilds inputs.pkl and wipes the tune, so run `tune --merge` after it (the tune_*.json are committed).
  On lxplus a replica takes ~3 s. If `.bashrc: Permission denied` shows up there, the AFS token has expired: run `aklog`.
- xw_tan/xw_amp/xw_compress run on the laptop's 8 cores in minutes (inputs.pkl + the head-on caches). Not on lxplus:
  the caches are local.

**Key files:** `sps_beam_test_26/analysis/cloud_basics/` — `unsat_width.py`, `x_width.py`, `rc_diffusion.py` (CB_CENTRE),
`bench_water_e.py`, `headon_cache.py`. Caches: `~/x17/cosmic_bench/cloud_basics/xwidth/caches/` on both the desktop
(original set) and now the laptop (the §32 set; det2/det6 not built here, they need event caches or ref-only).

## X width excess (UNSOLVED, superseded by the section above) — updated 2026-10-12 (dylan-MS-7C84)

**Resume:** OPEN PROBLEM. Bench X widens ~dry-like while v says wet; no gas fits. Next: det3 Y joint fit, Magboltz cross-check, contaminants, beam X template. NO NEW DATA (detectors gone).

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
0. **No new data can be taken: the detectors are gone (2026-10-16).** A dry-gas run is not possible; work with what we have.
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

## MM energy deposit per track (dE/dx) — updated 2026-10-14 (dylan-MS-7C84)

**Resume (2026-10-15):** steps 1–3 below are done (A+C ana, beam k fixed, combo at σ 0.16). The combination continues in the "Track passport" section above.

**Goal:** Dylan's ask: how accurately a reconstructed track's energy deposit in the MM gap is measured, first in
sim then on n_TOF data, and whether it helps the scintillators identify the track's energy. (The old M2 note only
said 1 vs 2 MIPs don't separate; that was not the question.)

**Done:** full record and numbers in `ntof_calorimetry/MM_EDEP.md`. Headlines: sim accuracy 10 % (A, contained) /
16 % (C); data add a 1D-along-x gain structure (removable with an in-situ map) plus a random per-track gain jitter
(σ 0.22–0.27 cosmics, 0.09 beam after map, from the truth-free halves test) → ~13–17 % beam, ~25 % cosmics.
Cosmic map does not transfer to beam. MM unusable < ~9 ms after the flash. Data: tracks firing no scintillator
carry +29 % MM charge (Geant4 predicts +15 %).

**In progress / where it stopped:**
- All four sims are on disk (`sim_{cosmic,beam}_{A,C}.parquet`); beam C finished at wrap-up, not yet analysed.
- `mm_edep_ana.chamber('A')` ran (→ `<calo>/mm_edep/results/chamber_A.json`); `main()` (A+C) not yet run.
- No report yet.

**Next steps:**
1. `PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.mm_edep_ana` (A+C).
2. Fix the beam calibration: use the beam sim's own `window_curve` / k for beam (see MM_EDEP.md "Known issues").
3. `python -m ntof_calorimetry.mm_edep_combo <σ>` with the measured σ (≈0.15 beam); add data category table
   (code for it is in the session, re-derive: stack_run_145 pulses vs WALL_THR/PLAS_THR, pointing at plastic).
4. Investigate the data "none" excess being 2× Geant4 (second particles? out-of-time tracks?).
5. `make_report` (report.html in `<calo>/mm_edep/`) and a slide note (slide-note skill), log on the x17 board.

**Gotchas / decisions:**
- Heed runs only on lxplus (desktop PyROOT is built for py3.10, system is 3.12). Reach lxplus only as `lxplus`.
- Bundles' per-channel gains are all 1.0. χ² tracks charge, not quality. Don't `pkill -f` (kills own shell).
- Disks: / and /media/dylan/data each have ~3.5 GB free.

**Key files & commands:**
- `ntof_calorimetry/{heed_edep,mm_edep_sim,mm_edep_data,mm_edep_ana,mm_edep_combo}.py`, `MM_EDEP.md`
- outputs `/media/dylan/data/x17/calorimetry/{heed,mm_edep}/`

## Stage-3 y fix — updated 2026-10-14 (dylan-MS-7C84)

**Resume:** DONE. Regenerated, audited and republished overnight 10-13→14. Remaining items are optional cleanup (below).

**Goal:** HANDOFF_Y_FLIP_FIX.md: fix the 09-07 y mirror at the source, guard it, regenerate everything, audit,
correct the record.

**Done 2026-10-14 (desktop):**
- `Y_FIX_REGEN.sh` ran end to end. Every live stage-3 set is `y_convention: "up"` and was proved an exact y
  mirror row by row: fullpass, is2_v2, is2_v2_cf (292 each), campaign (287, `--allow-drift`: its ToF columns
  predated the 09-11 trigtime rebuild), cosmic k_run_147/150. All guards pass; cosmic verdict "UP"; tests 10/10.
- Data disk full (3.8 GB): `regen_stage3 --work` stages and backs up on / (`~/x17_local/yfix_work/`,
  `<dir>/pre_yfix` symlinks). Largest set first.
- Grids that had been sized on the mirrored heights, found and fixed: `capsule_tof.Y_EDGES`
  (now −120…+60) and `capsule_y.Y0_GRID` (now −90…+90).
- The Athens side view compared the crossing with the gas centroid. It now uses the Geant4 nose
  (`CAD_CROSSING_Y = −27.8`), so the CAD shift is −3.5 mm.
- Added to the chain: late same-chamber pair sets (`same_chamber_late`, needed by `same_chamber_compare`) and
  `tracking_qa`.
- Results: capsule −31.3 / −34.6 (band crossing); ray trace −32.3 (A+C, campaign); G4 −28 recovered.
  Same-chamber excess mirrored, counts/angles unchanged. Liquids B, C now rise toward their PMT. Plastics 0/8.
  Full table: `Y_FLIP_AUDIT.md` §3–4.
- Republished: y-orientation (new "fixed" slide; evidence slides read the frozen `pre_fix/` inputs),
  same-chamber-pairs, capsule-pose, scint-stack; pair-vertex-y erratum; track-QA shards + tiers; board log;
  question closed.

**Optional next steps:**
1. Delete the backups (`~/x17_local/yfix_work/*/pre_yfix`, ~30 GB on /, plus the `pre_yfix` symlinks) once
   satisfied. Each old table is exactly recoverable by negating the five y columns.
2. `stage3_is2_v1_pilot` is still mirrored (superseded, no reader). Regenerate it if space allows:
   `.venv/bin/python -m sept26_prelim_analysis.regen_stage3 <out>/stage3_is2_v1_pilot --work ~/x17_local/yfix_work --combine [--allow-drift]`.
3. `mx_june_wft/drift_consistency/` (untracked) and the `garfield_test.py` / `cut_scan.csv` edits predate
   this session and were left alone.

**Gotchas / decisions:**
- The y-orientation deck's evidence slides must keep reading `ntof_cosmics/results/y_orientation/pre_fix/`.
  The regenerated files show the fixed frame.
- The band crossing images the capsule NOSE, not the gas centroid. Compare it with the Geant4 crossing (−28),
  never with 0.8.
- `same_chamber_compare` reads `~/x17_local/same_chamber_late/` (ToF ≥ 12 ms), not `same_chamber/`.

**Key files:** `Y_FIX_REGEN.sh`, `Y_FLIP_AUDIT.md`, `COORDINATES.md`, `sept26_prelim_analysis/regen_stage3.py`.

## Capsule pose in the beam — updated 2026-10-15 (dylan-MS-7C84)

**Resume:** biased, gas-fixed Geant4 library running (cluster 4467046); next: steps+reduce, digitise, pool, refit, rebuild deck.

**Goal:** where the ³He capsule sits relative to the beam and the chambers (sets the ³He/IPC/X17 yield; feeds the 1 : 8
pair-rate question). Full physics state: Correction box of `sept26_prelim_analysis/HANDOFF_CAPSULE_POSE_FIT.md`.

**Where the result stands:**
- Real EAR2 beam since 14 Oct (Lambda2D was the resolution function). Capsule ~1 cm below drawn; leans with its top
  toward B and A (φ ≈ 135°, θ ~7–15°); sideways beam/capsule degenerate (³He sees 20–53 % of the beam).
- z lean confirmed without D (15 Oct): A/C band-slope depth, `capsule_pose data_depth` (+4.6 mm toward A from h −80 to
  −20; D +4.5); stable to ~0.4 mm across ToF slices / run halves / wall tracks.
- Caveat: z moves below the source height, x above it; one rigid tilt fits neither shape well. Unmodelled hardware:
  the thin Al hanger tube above the capsule bends toward the A–D corner (Dylan: likely sets the tilt). Not to be modelled
  unless asked; check it first when a shape doesn't fit.
- Deck (published): https://dylan-neff.web.cern.ch/notes/capsule-pose.html (`make_capsule_pose_deck.py`).

**Running / where it stopped:**
- **Cluster 4467046** on lxplus: 200 x 5M EAR2 neutrons, 19.5 m, `--bias-capsule 20`, built from MX17_Full_Geant e3def59
  (gas primary-ionization fix + capsule-Al capture bias) in `/afs/cern.ch/work/d/dneff/git/MX17_Full_Geant_v2`.
  ~25 min/job, ~4 GB/file (~0.8 TB) -> `/eos/experiment/ntof/data/x17/full_sim/neutrons_thermal_ear2_L19p5_bcap20/`.
  Submit dir `~/condor/mx17_neutrons_ear2_L19p5_bcap20`. Worth ~1700 analog jobs for the library.
- Analog extension (cluster 4465059, old gas) REMOVED at Dylan's request. Its 400 partial files were moved (not deleted)
  to `full_sim/neutrons_thermal_ear2_L19p5/_killed_ext_2026-10-15/` (362 GB); delete only with Dylan's OK.
- Analog EAR2 libraries done: 19.5 m digitised (`g4_digi/g4_{A,C}_ear2_L19p5/`); 19.95 m steps/reduce on EOS
  (`full_sim/angle_scale/{steps,neutrons}_ear2_L19p95`), not pulled/digitised (old gas; low priority now).
- Local background waiters die with this session; nothing local is running.

**Next steps:**
1. When 4467046 is done (`condor_q`; check holds): on lxplus `cd ~/condor && bash make_ear2_subs.sh
   neutrons_thermal_ear2_L19p5_bcap20 ear2_L19p5_bcap20 && condor_submit steps_ear2_L19p5_bcap20.sub && condor_submit
   reduce_ear2_L19p5_bcap20.sub`. Reduce peaked ~12 GB on 1.2 GB files; these are ~4 GB: test one job's memory first or
   raise MEM in make_ear2_subs.sh. `lxstore status` before/after.
2. Pull `angle_scale/steps_ear2_L19p5_bcap20` -> `/media/dylan/data/x17/ntof_cosmics/g4_digi/` and
   `neutrons_ear2_L19p5_bcap20` -> `.../g4_angle/` (local disk has only ~3 GB free: check sizes; steps were 0.5 GB/100
   analog files, biased will be larger), then `bash ntof_cosmics/g4_digi/run_ear2.sh ear2_L19p5_bcap20`.
3. Pool: `POSE2_LIB=nose,ear2_L19p5,ear2_L19p5_bcap20`; rerun `capsule_pose shear`, `pose2` / `pose2z` / `pose2d`
   (families: `POSE2_XB=0 POSE2_XC=-16,-18,-20`; `-5 / -10,-15`; `-10 / -10,-15`) into a new out dir with copies of
   data*.csv (as `capsule_pose_ear2_pool/`); regenerate `make_pose2_report.py` and the deck (point POSE2 at the new dir).
4. If the toy is still the limit, extend the biased run (more jobs, same sub with new seeds).
5. Still needed from Dylan: collimator, capsule tip height/holder, survey/photo.

**Gotchas / decisions:**
- Library size: the toy uses capsule tracks captured within r < 8 mm; the real beam puts only 61 % there (93 % for
  Lambda2D). Pooling libraries is valid (the toy translates each track to its own capture point); `library()` resolves
  the reco tan signs per part (raw y-tan sign differs between digitiser vintages; both focus equally).
- `run_digi` needs the run's own reduction (`G4DIGI_REDUCED`; `run_ear2.sh` sets it) or it silently digitises ~1 event.
  ADC/e now comes from that reduction's median track (was fixed 247 e-); ear2_L19p5 was digitised with 247 (~3 % high
  amplitude, harmless for pointing). Gas-fixed libraries will give ~10 % fewer e-.
- `pose2d` (toy prediction of the A/C depth curve) is library-limited (pooled toy sd 1.2–1.4 mm vs data 0.15); the deck
  uses the depth data reading, not its χ². Pooled x-shear toy sd 0.3–0.6 mm.
- Geant4 11.2 (lxplus) records LS showers in ~10x more steps than 11.3: biased files are big. Not a port bug (cosmics
  identical old/new).
- uproot can't read EventTree's `capture_vol` char arrays; use PyROOT (`PYTHONPATH=~/Software/root_6_40/lib`,
  `bytes(e.capture_vol)`), or the reduce parquet.
- Never quote the capture source as the beam position; D z only from wall-coincident tracks ≥ 25 ms.

**Key files & commands:**
- `sept26_prelim_analysis/capsule_pose.py` — steps `pose2`, `data_z`, `pose2z`, `data_depth`, `pose2d`; env
  `POSE2_XB/XC/THETA/PHI/ALPHA/L/LIB` (LIB comma-list pools).
- `sept26_prelim_analysis/make_capsule_pose_deck.py` → `<out>/capsule_pose/deck/capsule-pose.html`; publish with
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <html> --slug capsule-pose --force --deploy` (pull the
  site repo first).
- `ntof_cosmics/g4_digi/condor/make_ear2_subs.sh <simdir> <tag> [first_job]`, `ntof_cosmics/g4_digi/run_ear2.sh <tag>`.
- MX17_Full_Geant: `--bias-capsule <f>`; its HANDOFF.md has the validation numbers.

## Same-chamber pairs + the y mirror — updated 2026-10-13 (dylan-MS-7C84)

**Resume:** y mirror fixed AND regenerated (10-14, see the section above); pairs 1:8 still open.

**Goal:** same-chamber (intra) pair physics against Geant4 for Dylan's Thu 2026-10-15 talk. Along the way we found
that stored global y is upside down.

**Done (10-12 → 10-13):**
- Pair results on the is2_v2 re-pass at the capsule-focus k (`kcal_is2_v2_cf`, provisional):
  - correlated pairs are 0.15 % of late events;
  - Geant4 says ~78 % of them are Al e⁺e⁻;
  - data ≈ 1/8 of Geant4;
  - the 10-12 vertex excess was retracted.
  - Record: `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md`; report `<out>/same_chamber/report.html`.
- 10-13 follow-up (`capsule_tof.py`, `same_chamber_scatter.py`, deck `make_same_chamber_deck.py` →
  notes/same-chamber-pairs.html):
  - **Time window.** The < 12 ms cut is not where the pairs went. 87 % of beam captures fall before 12 ms, but 92 %
    of capsule-pointing tracks come after it (delayed neutrons). Late data are as capsule-pure as Geant4.
  - **Vertexing.** Al e⁺e⁻ cannot vertex: p·β·θ = 0.40 MeV·rad, a 33 mm miss, and only 2.3 % pass with perfect
    tracks. X17 legs miss by 9–11 mm.
  - **Expectation.** The expected count already includes the opening angle and the < 20 mm loss.
- **The y mirror.** `build_tracks.IN_PLANE_SIGN_Y` flips the y position; it should flip tan y.
  - Proven by cosmic-muon timing (run_149, p = 4e-11), 8/8 plastic PMT timing, liquid B, the capsule nose and the
    wiring.
  - `COORDINATES.md` (new) holds the frame and the rule. CLAUDE.md has a warning, build_tracks a warning comment.
  - Slide note: notes/y-orientation.html (`ntof_cosmics/make_y_orientation_deck.py`); checks in
    `ntof_cosmics/y_orientation_check.py`.
- The capsule pose fit was done by another session (315a50d, `HANDOFF_CAPSULE_POSE_FIT.md` Outcome):
  - one compact capture source on the nose, at frame x −10.3, z −3.6, at every ToF;
  - revised 10-13 night: that is NOT established as the beam position. Either the beam/frame is ~10 mm off the
    alignment, or the capsule is far off a centred beam and lower than drawn (³He yield 1–15 %); needs a survey;
  - neither reading explains 1 : 8 alone.

**Next steps:**
1. The y fix is its own section above: done 10-14.
2. The 1 : 8 pair rate:
   - re-digitise the Geant4 pairs with is2_v2;
   - Geant4 with delayed thermal neutrons on the capsule;
   - check for a busy-event two-track loss.
3. Talk on Thursday: present the pair fraction, Geant4's breakdown, chamber A's opening angle, and the 1 : 8 as open.
   No vertex excess. Heights are quotable again (fixed 10-14).

**Gotchas / decisions:**
- Read stage 3 through `build_tracks.y_up(df, path)` (flag-aware); `capsule_pose`'s own mirror is gone.
- Pair work: t0 ≤ 300 and ToF ≥ 12 ms. Early-time mixing made the 10-12 artefact.
- In-beam A–C straight lines are not muons (no timing direction). Use beam-off run_149 for anything that needs muons.
- The wall mean-of-ends time is position-independent. Wall-group cable offsets contaminate u slopes, so fit with
  per-group offsets.

**Key files & commands:**
- `python -m sept26_prelim_analysis.capsule_tof [beam|g4|tof|tilt]` → `<out>/capsule_tof/`
- `python -m sept26_prelim_analysis.same_chamber_scatter` → `<out>/same_chamber/scatter_*.csv`
- `python ntof_cosmics/y_orientation_check.py cosmic|pmt` → `ntof_cosmics/results/y_orientation/`
- Decks: `python -m sept26_prelim_analysis.make_same_chamber_deck`, `python ntof_cosmics/make_y_orientation_deck.py`,
  then `add-note.py <html> --slug same-chamber-pairs|y-orientation --force --deploy`

## Large q_sum tracks (NNLS runaway) — updated 2026-10-02 (dylan-MS-7C84)

**Goal:** explain the ~25–30 % of gated tracks with q_sum > 1e6 ADC (STATUS
2026-09-10), and find out whether they corrupt geometry as well as charge.

**Done:**
- Mechanism found. The NNLS depth-charge profile in `wft/model.py:chi2_plane`
  is unregularised. Depth bins whose column is ~0 inside the data window get
  unbounded q. Two routes:
  - time-censored (71 %): window 20×60 ns, last sample 1140 ns, while the grid
    spans 1080 ns after t0;
  - space-censored (29 %): the slope walks deep bins off the strip window.
- Campaign census, 14.26 M gated tracks:
  - 30.3 % have a runaway plane;
  - 17.0 % are in the late class (t0 > 300 ns), whose geometry is wrong;
  - 9.8 % of scint-coincident tracks are in the late class.
- Refit, run_145 tag 000, all four arms, 3 181 fits: production against a
  "guarded" fit that drops columns below 1 % of the largest.
  - Time-censored runaways: median |Δtan| 0.20, t0 moved in 82 %, flat fits 37 → 8 %.
  - Late normal fits move too (47 % beyond 0.05), so the class boundary is t0, not q_sum.
- Written up in STATUS.md (2026-10-02 entry) and as OCTOBER_2026.md item O10.
- Report: `<out>/qsum_runaway/report.html`. Slide note:
  https://dylan-neff.web.cern.ch/notes/qsum-runaway.html

**In progress / where it stopped:** nothing half-done. Follow-ups were
proposed, but Dylan has not picked any yet.

**Next steps (proposed, in order):**
1. Arbitrate production against the guarded geometry on late arm-A tracks, using
   scintillator pointing (`det_a_scint` / `scint_stack` extrapolation). Measure
   which fit points at the wall group or plastic bar that fired.
2. Quantify how many late-class tracks feed the pair and opening-angle tables,
   and whether a t0 cut changes the spectra or the perpendicular excess.
3. Prototype the real fix for O10:
   - only offer depth bins whose pulse peaks inside the window, or add a depth-continuity prior;
   - require early fits to stay unchanged and late fits to show no χ² loss;
   - store `q_obs`.
4. Work out what the late tracks physically are (out-of-time particles?), and
   what fraction of the space class is coherent ringing.

**Gotchas / decisions:**
- The guarded fit is a probe, not a fix. It is worse on χ² for late tracks.
  It is a monkeypatch of `wft.model.chi2_plane` inside `qsum_runaway` workers only.
- The run_145 data window is 20 samples, not 32. `wm.NSAMP` defaults to 32 in
  a fresh process and is only set inside `fit_plane`. Diagnostics outside a fit
  must use the window's own `W.shape[1]`. This cost one wrong classification here.
- An event can have several candidate windows per plane. Always key examples on
  (arm, event_id, plane, win).
- Don't classify by summing q across fits: a few 1e22 values swamp everything.
  Classify per fit by its dominant bin.
- A χ² explained-fraction does not separate noise windows from tracks, because
  window noise dominates every class.

**Key files & commands:**
- `sept26_prelim_analysis/qsum_runaway.py`:
  - `census` (~2 min; reads stage3_fullpass/tracks_campaign.parquet);
  - `refit --n 300 --jobs 14` (~12 min on 16 cores; writes `<out>/qsum_runaway/`).
- `python -m sept26_prelim_analysis.make_qsum_runaway_figures`, then
  `python -m sept26_prelim_analysis.make_qsum_runaway_report`.
- `python -m sept26_prelim_analysis.make_qsum_runaway_deck`, then
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py <out>/qsum_runaway/deck/qsum-runaway.html --slug qsum-runaway --force --deploy`.
- `<out>` = `/media/dylan/data/x17/sept26_prelim` (`python -m sept26_prelim_analysis.paths`).

## Beam-off cosmics: angle response + in-situ reco — updated 2026-10-12 (dylan-MS-7C84)

**Resume:** muons say is2 scale 1.00, capsule focus 1.12 (A and C). Next: measure in-beam muons directly in is2_v2.

**Goal:** a reconstruction that measures angles correctly at n_TOF, head-on included, so same-chamber pairs can be
reconstructed.

**Read first:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §22 (latest), then §20 (is2_v2 full pass vs production,
report `results/fullpass_is2_v2/report.html`) and §21 (capsule focus, 0.8 cut rejected). Earlier: §13–§19.

**Done:**
- 10-10/11 (another session, no wrap-up at the time): is2_v2 condor 4410513 finished (6,464/6,466; the missing
  run_104 stat090_0016 A/C have empty input, and production has no A/C there). Stage 3 is in
  `/media/dylan/data/x17/sept26_prelim/stage3_is2_v2` (stores the 0.8 superset as `gated`; the analysis cut is 0.6).
  Wall-confirmed vs production: A ×1.49, C ×1.82. Merged to main (f0c80cd).
- 10-12 (§22):
  - **Geometry ruled out for the capsule scale.** Distance errors cancel or push A and C oppositely. The source sits
    at global x −9.5 mm (joint (s,dx) fit = imaging −9.3), but with it fitted s = 1.12 on both arms.
  - **Depth origin ruled out:** median charge depth is at mid-gap. Note: a p0/t0 slide does NOT cancel against the
    A–C truth.
  - **Population ruled out:** s 1.08–1.14 across time window, capsule height, wall, χ², charge and |tan y| splits.
  - **Physical RC kernel (cloud_basics) on the run_149 A–C truth:** A y resolution −7 %, x S-shape a third flatter.
    C gets no gain (det7 constants make y +10 %). Raw scale moves < 2 %.
  - The opt-in RC kernel files (`wft/model.py`, `wft/calib.py` + tests) were brought from `mx17-paper-status` into
    main.
- 10-12 (later): slide note updated with §19–22: https://dylan-neff.web.cern.ch/notes/ac-insitu-angles.html
  (`ntof_cosmics/make_insitu_deck.py`, slides 13–21: is2_v2 pass, focus, steep tracks, scale by population,
  the muon-vs-capsule tension, RC kernel, problems, next steps). c549bcf on main.

**Next steps:**
0. (Cheapest, proposed to Dylan, not started) Measure the in-beam A–C muons' scale directly in the is2_v2
   stage-2/3 output instead of inferring it through production (the pooled norm in
   `/media/dylan/data/x17/ntof_cosmics/inbeam_through_goers/pooled_norm.csv` was measured with the production bundle).
   Muons at 1.00 ⇒ the 12 % is real; ~1.1 ⇒ it was the inference.
1. Run `focus_offset` logic on the g4_digi sim chain (`g4_*_is2_full`) with the data's dx. Then vary the simulated
   source (capsule walls, material behind the axis) to see what turns muon-correct angles into s 1.12. Or find an
   electron-specific angle truth.
2. Decide whether stage 3 `pointing()` should use the measured axis (x ≈ −9.5 mm) instead of 0. Today
   `dca_axis_mm` / `in_bore` are off-centre by about a capsule radius. Check who cuts on them before changing it.
3. Physical kernel at n_TOF, if pursued: n_TOF-gas Dp (Magboltz 90/10 at 243/278 V/cm), det6's own template for C,
   §28 joint Dd, then a refit. Only A's y and head-on resolution is likely to gain.
4. Still open: D in-situ bundle (v ≈ 28); B's HV in run_149; adopting is2_v2 as production (Dylan).

**Gotchas / decisions:**
- `capsule_focus.py` assumes the axis at x = 0. Its tan-sign halves are dominated by the −9.5 mm offset; use
  `focus_offset.py`. Exclude 1–10 ms (flash junk, miss 58–82 mm).
- The physical kernel was compared at each label's own TRAIN scale (`insitu_paired.py`); raw scale shifts are in the
  `cmp` v_scale column.
- Bench chamber map at n_TOF: A = det3, B = det2, C = det6 (runs det7's r06 kernel in is2_C), D = det7.
- Raw-tan constants (TAN_MAX etc.) change meaning with v. Never use the capsule band as the beam scale.

**Key files & commands:**
- `ntof_cosmics/focus_split.py`, `focus_offset.py` → `ntof_cosmics/results/focus_split/` (CSVs + logs).
- `ntof_cosmics/insitu_paired.py`, `~/scratch/ntof_insitu/recipe_rc.sh` (copy in `results/insitu_rc/`, with the
  paired CSVs). Bundles `~/scratch/ntof_insitu/bundles/rcm{3_A,7_C,6_C}`, reco tables `reco_rcm*ts_s3_*.parquet`.
- `python -m ntof_cosmics.focus_offset --src /media/dylan/data/x17/sept26_prelim/stage3_is2_v2` (~15 min, 1 core).

## Scintillator stack mapped by the MM tracks — updated 2026-10-06 (dylan-MS-7C84)

**Resume:** scint-stack maps done (whole-face, PMTs drawn); next: why vertical liquids don't answer near their PMT, arm D's extra unbiased tracks.

**Goal:** characterise every scintillator (efficiency and response heat maps), on the single-track imaging calibration.

**Done:**
- Code in `ntof_scint_stack/`; data `/media/dylan/data/x17/scint_stack/` (`paths.spell('scint')`). Calibration: per-run imaging `k` (tanx = k·tan_raw), extrapolation `u + L(α a + λ k tan) − δ` fitted on the wall group edges; α ≈ 0, λ 0.59–0.76. README has it.
- **Statistics (this session):** coverage was always the whole full pass (34 runs, 13.3 M gated tracks; run_79/81 excluded). The thin sample was the *unbiased* tag (another arm triggered): ~99 % of triggers fire one arm. Main maps now `*_full` layers on `all_late` (every late trigger, whole face, boundary-tolerant `_tol` probes, 25 mm): 134k–934k tracks/arm vs 2–9k unbiased. Unbiased shape is its own slide (100 mm).
- `checks.py`: trigger thresholds MEASURED (0.5 % low edge of each arm's own triggers, flat to ~1 mV campaign-wide; now in `extract.WALL_THR/PLAS_THR`; run_79 read-back had plastic 3–6 mV high). Time-cut scan: LATE_MS stays 10 ms (earlier = <15 % more stats, efficiency not converged).
- `ana` writes `funnel` (cut flow per arm); deck has a samples slide.
- **PMTs drawn on the maps.** Liquids from Geant4 (`MX17_Full_Geant/include/SimConfig.hh` `ls_rot_deg`, 17–18 July survey): A, D horizontal with PMT at +u; B, C vertical, PMT up. Dylan recalled B right / D top; Geant AND the data (A/D answer 12–23× more on the +u half, B/C no u gradient) disagree — told Dylan; switch is `make_deck.LIQ_PMT`. Plastic PMTs on top (Dylan's report; not in Geant, not testable).
- Slide note republished: https://dylan-neff.web.cern.ch/notes/scint-stack.html; long report regenerated.

**Next steps:**
1. ~~Why the vertical liquids (B, C) show no rise toward their PMT (C: 0.07 % top vs 0.34 % bottom) while A, D do.~~ **Answered 2026-10-13: the stage-3 y mirror.** In true y C is 0.34 % top vs 0.07 % bottom, a rise toward the PMT like A and D (`Y_FLIP_AUDIT.md` #3; confirm on the regenerated scint stack).
2. Arm D: a third of its late tracks sit on other-arm triggers (A–C 4–18 %) — real particles or chamber pick-up? D's unbiased maps rely on them.
3. Join run_149 cosmic tracks to the slim → MIP-clean efficiencies.
4. Plastic bar placement / D wall group-0/2 cabling against the as-built drawings.

**Gotchas / decisions:**
- Self-triggered maps read near 1 by construction (trigger = wall-sum AND plastic); they show holes, not levels. Quote unbiased numbers.
- Dead end: loosening LATE_MS below 10 ms (`checks.late_scan`): not worth it.

**Key commands:** extract → ana → checks → make_figures → make_report → make_deck → add-note (see `ntof_scint_stack/README.md`).

## Calorimetry feasibility + MM dE/dx — updated 2026-10-14 (dylan-MS-7C84)

**Resume:** scintillator proof of concept done 2026-10-14 (wall + plastic, muon-calibrated, vs beam e⁻ and Geant4): https://dylan-neff.web.cern.ch/notes/scint-calorimetry.html (`ntof_calorimetry/make_wp_deck.py`). Earlier feasibility note: https://dylan-neff.web.cern.ch/notes/calorimetry.html (`make_deck.py`). Open: C1 +15 % electron offset, PMT bases for LNL.

**Wall + plastic result (2026-10-14, `wall_plastic.py`, record in `PLAN.md` "Reopened"):** data ÷ Geant4 at ≥ 40 ms: plastic A1/A2/C2 1.05/1.03/1.02, **C1 1.15** (pure gain offset, cause unknown), wall A/C 0.95/0.96. Plastic is linear 3.4→8 MeV and is 20 mm thick; the wall measures dE/dx (~0.5 MeV flat), a 1-vs-2-particle tag. Plastic PMTs lose 3–8 % gain 12–25 ms after the flash (LNL: rate-proof bases or a pulser). Run-to-run stability 0.5–0.8 %. Geant4 side: `MX17_Full_Geant/scripts/stack_edep.py` on branch `lnl` (737e9a2, deliberately not merged to its main). Rebuild: `python -m ntof_calorimetry.wall_plastic`, then `python -m ntof_calorimetry.make_wp_deck`, then `add-note.py … --slug scint-calorimetry --force --deploy`. Outputs `/media/dylan/data/x17/calorimetry/wp/` (disk ~3.6 GB free).

**Goal:** find what energy information the n_TOF data can give, and test MM dE/dx as a concept.

**Done (all in `ntof_calorimetry/`, report `/media/dylan/data/x17/calorimetry/report.html`):**
- C1 plastic: the srccal keVee line reads the cosmic MIP at 0.76–0.97 of Bichsel 3.41 MeV (Geant4
  agrees: 3.375). New MIP-anchored scale `c1/calib_plastic_e.json` (4–7 %). Trigger threshold
  2.1–2.9 MeVee. Linear to 1.45 MIP.
- C4 liquids: MIP = 20–28 mV against a 16–18 mV n_TOF ZS threshold, so the liquids are threshold-limited.
  Their source scale is ×5–7 wrong. Usable: A u≥0 at 65–86 %, D u≥50 at 50–70 %.
- In-beam A–C through-goers at ≥10 ms reach the liquid at 0.10 of the cosmic expectation, so they are
  mostly not penetrating.
- M1/M2: raw road charge (`mm_charge.py`). FWHM/MPV ~1, and a 2-MIP flag reaches only 40 % at 10 %
  mistag, so this line is KILLED.
- C2 (condor 4409644): 1.2 MeV is lost before the plastic, and half of 4 MeV electrons reach it.
- C3 (condor 4409645, 10^7 pair sim, IPC reweighted to ipc_born): plastic energy gives a Z² gain over
  θ_open of 1.05–1.08, so KILLED. The true soft-leg T would give ×2.5–4: a design lesson.

**In progress / where it stopped:** nothing half-done. All condor jobs finished; outputs are on EOS
`full_sim/calorimetry/` and pulled to `<calo>/c2`, `<calo>/c3`.

**Next steps (optional):**
1. Measure a plastic bar (20 vs 25 mm). 25 mm would move the MIP scale by 21 %.
2. C1.4 gain vs time since flash: needs an in-beam energy reference (H-capture Compton edge?).
3. Tell the through-going-background owner that beam through-goers are not penetrating.

**Gotchas / decisions:**
- Beam through-goers are not a MIP sample. Use run_149 cosmics (clock-matched) for scintillator MIPs.
- Plastic MIP fits need per-event truncation at the trigger threshold (A's threshold is at 0.85–0.9 of
  the peak) and a σ/MPV prior.
- C3 metric: fine 2D bins on the IPC MC fake gains. Use coarse bins, pool bins with <10 MC events, and
  check the split halves.
- An apparent attachment fall in C was a selection artefact (per-track normalisation on in-window
  tracks).
- `pkill -f <pattern>` and `pgrep` loops match their own shell command line: don't.

**Key files & commands:**
- `ntof_calorimetry/README.md` — run order; `PLAN.md` — the plan plus "Results so far"
- `ntof_calorimetry/condor/` — C2/C3 submitters and reducer (deployed at
  `/afs/cern.ch/user/d/dneff/condor/calorimetry/scripts`)
- `PYTHONPATH=. .venv/bin/python -m ntof_calorimetry.make_report`

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

**Resume:** CLOSED 2026-10-23: r06 frozen per XASYM_CONTINGENCY.md, pvy95 not adopted (quoted as a known Y improvement). Was ON HOLD (Dylan 10-09) — likely superseded by the physical RC kernel (see Detector model basics). Per-view Y ratio validated (gate: Y −0.04…−0.08°, head-on −0.10…−0.13°, det2/3/7); re-pass staged in `condor_campaign_pvy95/`, NOT submitted — needs Dylan's go.

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

## MX17 papers (two-paper plan) — updated 2026-10-19 (dylan-MS-7C84)

**Resume:** LaTeX outlines of Papers I+II are in `papers/`; next: Dylan sends 2 Overleaf git URLs, run `overleaf.sh link`.

**Goal:** get two companion papers written: I, the MX17 chambers and their performance (June cosmic bench + det4 in SPS H4); II, the waveform forward-model reconstruction and how much a hit-based readout loses. Stephan edits in Overleaf with tracked changes (production/design input for I).

**Done:**
- 10-08: plan as tables in `mx_june_cosmic_qa/PAPER_PLAN.md` (outlines, work list C/R/D/V, n_TOF tripwires); slide note https://dylan-neff.web.cern.ch/notes/mx17-detector-paper-status.html via `make_paper_plan_deck.py`. C1 (reproduce r06) was done 10-09.
- 10-19 (commit e6448d5, on main): `papers/paper1/`, `papers/paper2/` — LaTeX outlines, one file per section with a state box, intent, placeholder text (`\tbw`), `\askStephan`/`\askDylan` boxes, `\analysis{id}` margin notes, `\requote` numbers. Paper I has a new §2 "Design and construction" (incl. production & QA, Stephan's) seeded with the 17 spec-page questions. Both compile (12 / 8 pages, pdflatex+bibtex).
- `papers/make_figures.py` + `figures.toml` per paper: figures/<id> is a copy (current), a faded "SUPERSEDED" stamp (hits-based: edge turn-on, timing) or a grey placeholder. 11 real figures, rest placeholders.
- `papers/overleaf.sh link|pull|push|sync <paper>` (git subtree). Tested both ways on a local stand-in remote: link, Stephan-edit pull, local push, refusal when Overleaf is ahead. NOT yet run against real Overleaf.
- Workflow documented in `papers/README.md`; pointers in CLAUDE.md layout and PAPER_PLAN.md.

**In progress / where it stopped:**
- Waiting on Dylan to create the two blank Overleaf projects and store a git token (Dylan has Overleaf premium).

**Next steps:**
1. Dylan: create blank projects "MX17 Paper I"/"MX17 Paper II", copy git URLs (Menu → Integrations → Git), make a token (Account Settings → Git integration), store it: `git config --global credential.https://git.overleaf.com.helper store` then one `git ls-remote <url>` (user `git`, password token).
2. `papers/overleaf.sh link paper1 <url>` and `... paper2 <url>`; commit + push origin main (the link adds a merge commit). Then Dylan shares with Stephan as Editor, turns Track changes on, sets main.tex / pdfLaTeX.
3. Dylan: send Stephan the spec page link and the Overleaf link (spec-page section of this file).
4. Fill placeholders as figures appear: edit `figures.toml` (source + state = "current"), `make_figures.py`, commit, `overleaf.sh push`. Candidates needing data not on the desktop: sps_resolution (residuals.npz), kernel_forms (stacks_run71_raw.npz), sps_setup (mpgd26 inputs).
5. Open decisions still blocking text: C2 (freeze r06 vs per-plane c2/c1, pvy95 re-pass staged), C4 (journal, author list; then switch document class from plain article).
6. Analysis items as in PAPER_PLAN: D1 requote, D2, D3, D4, R1–R7, R9, R10.

**Gotchas / decisions:**
- Overleaf git carries plain text, not tracked changes/comments: accept/reject Stephan's changes in Overleaf before pulling; never push repo edits into a section with pending tracked changes. Figures are always safe to push.
- overleaf.sh `link` joins histories with a commit carrying git-subtree trailers (a plain `git subtree merge` refuses unrelated histories — tried, failed). Never forces.
- Paper figures are tracked despite the global `*.png` ignore (`!papers/*/figures/*.png` in .gitignore).
- Unconfirmed in the text: gas mixtures per setup (bench Ar/iC4H10 95/5? SPS Ar/CF4/iC4H10 88/10/2?) — left as `\askDylan`. refs.bib has 5 entries from memory; replace with INSPIRE BibTeX. D_RC value in II §2 has no unit.
- Section state boxes duplicate PAPER_PLAN states: change both.
- Unrelated: uncommitted cloud_basics changes in the tree belong to another session; not touched.

**Key files & commands:**
- `papers/README.md` — the workflow
- `.venv/bin/python papers/make_figures.py all [--list|--run|--only id]`
- `cd papers/paper1 && latexmk -pdf main.tex` (or pdflatex/bibtex)
- `papers/overleaf.sh sync paper1`
- `mx_june_cosmic_qa/PAPER_PLAN.md` — states and work list; `make_paper_plan_deck.py` + add-note.py to republish the status note

## Cosmic-bench efficiency + gain maps (A–E) — updated 2026-10-09 (dylan-MS-7C84)

**Resume:** both PDFs built and accepted "good enough"; nothing pending — rerun scripts only if a colleague wants a different kernel.

**Goal:** per-chamber efficiency and relative-gain maps of the June cosmic bench, one page each, in the MPGD26 slide-27 format, for a colleague.

**Done:**
- `mx_june_wft/report/make_bench_effmaps.py` → `~/x17/cosmic_bench/Analysis/bench_efficiency_maps/cosmic_bench_efficiency_maps_s4mm.pdf`.
  Gaussian σ = 4 mm (my reading of "4 mm kernel"; hard r = 4 mm not tried), 0.5 mm step, viridis 0–100 %, within 5 mm of M3 ref.
  Detector-local frame (x horizontal, y vertical; inverse of alignment θ ≈ 90°), fixed axes −20…420 mm, fixed page size, method text in footer.
- `mx_june_wft/report/make_bench_gainmaps.py` → `…/bench_efficiency_maps/gain/cosmic_bench_gain_maps_s8mm.pdf`.
  Per-muon Q = √(x_q_sum·y_q_sum)/√(1+tx²+ty²) from wft events; Gaussian mean of ln Q, σ = 8 mm, /chamber median, log colour 0.3–3.
- Datasets: A det3 `g_det3_wknd` (21 953 rays, 93.1 %), B det2 `g_det2` = 6-22 `long_run` (19 045, 91.3 %), C det6 `g_det6_long` (9 628, 74.9 %), D det7 `g_det7_long` (9 428, 57.1 %), E det4 `g_det4` (12 259, 41.6 %).
- Options not used: det3 `sat_det3` (7.1 k rays, 93.3 %); det2 `o22_long_det2` = `longer_run` (3.7 k, 92.0 %, ~3 muons/kernel).

**Next steps (only if asked):**
1. Hard-disc r = 4 mm variant if that is what "4 mm kernel" meant; larger σ for det6/7/4.
2. Investigate the ×2.5 gain rise toward the edges (seen in both wft Q and summed hit integral; probably edge fringe field, cf. script 32).

**Gotchas / decisions:**
- Data-dir changes (outside repo): 6-22 `long_run` (det2_det3) now has `m3_tracking_root_v2` + EOS reprocessed hits as `combined_hits_root`; old hits kept as `combined_hits_root_june_orig`. Re-ran `01_alignment`/`02_efficiency` for `g_det3_wknd` (its 08-12 outputs were stale vs the 08-21 r06 reco), `g_det2`, and `o23_long_det4` (alignment only; that key is a tiny `longer_run`, efficiency fails — hits lack `significance`; irrelevant).
- Peak-strip amplitude is NOT a usable gain estimator for det3/det2 at operating HV (~40 % of events saturated); raw waveforms for g_det3_wknd/g_det2 are not local.
- Gain is measured on detected muons only → survivor-biased where efficiency collapses (det4, det7). Gain is relative per chamber, not comparable across chambers.
- Per-ray CSV caches in the output dir (`<key>_rays.csv`, `<key>_box.json`) — delete to force recompute.

**Key files & commands:**
- `.venv/bin/python mx_june_wft/report/make_bench_effmaps.py [--sigma 4]`
- `.venv/bin/python mx_june_wft/report/make_bench_gainmaps.py [--sigma 8]`

> **Read first:** `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md`. This branch
> is one of two threads on same-chamber pairs; the other (single-track angle
> truth, the seeder, v) lives on `beam-off-cosmics`.

## Two-track limit: relative angle of the pairs — updated 2026-10-07 (dylan-MS-7C84)

**Goal:** answer Dylan's question about the two-track slide note: what relative angle do the modelled pairs have, does angle help to separate them, and what angle do real pairs have? Answer it from data, and add slides to `notes/two-track-limit.html`.

**Done:**
- `sept26_prelim_analysis/pair_angle.py` (new):
  - `asimov`: R5, the noise-free Δχ² on a grid of mesh separation d × divergence D over the drift column. Done; output `r5_asimov.csv`.
  - `oracle`: the same grid with noise; running, see below.
  - `data`: donor pointing, R3 pairs with their divergence, event-bench divergence, det_a_intra real vs mixed pairs, and an IPC point-source toy. Done.
  - Outputs go in `~/x17/sept26_prelim/two_track_limit/pair_angle/`.
- `sept26_prelim_analysis/make_pair_angle_slides.py` (new): six slides, wired into `make_two_track_deck.build` before the closing slide. The oracle slide says "pending" until `r5_oracle.parquet` exists.
- The deck is built and checked visually (render in the session scratchpad), but **not yet republished**.
- Results:
  - A common-vertex pair diverges over the 30 mm gap by only 30/L = 0.13 of its separation, so "close" means "parallel".
  - Angle helps a lot. Oracle partial: at d = 0, D = 1 mm resolves 52 % and D = 2 mm resolves 100 %. R3 in chamber A, ideal fit, under 1 mm: 24 % of near-parallel pairs against 90 % of pairs diverging 3 mm or more.
  - Bench donors come from different triggers. In y, their slopes scatter ±0.14 about the pointing line (in x, ±0.05), so 80 % of coincident bench overlays closer than 12 mm diverge by ≥ 3 mm in one view. Below ~3 mm the bench is optimistic for genuine pairs.
  - Real same-trigger pairs in chamber A look like mixed pairs: median opening angle 31–49°, and 0 of 296 k pairs closer than 24 mm, where mixing predicts 1.6 %.
  - IPC toy: 5.2 % of same-chamber M1 pairs (0.4 % of E0) are within 12 mm.
  - Side finding: the fixed chain resolves only 76 % of diverging chamber-A pairs at 3–12 mm, against ~97 % for the ideal fit. Suspected cause: the grid search seeds only parallel line pairs. Not tested.

**Update 2026-10-07:**
- The oracle finished (2 Oct 19:43): `r5_oracle.parquet`; the angle-oracle slide is filled in.
- F-rescan cluster 4348153 merged (A and C, 56/56 shards each). The ladder reproduces the fixed runs exactly at A 1200 / C 2400.
- Result: pick = matched in both chambers. A 1000 fails (0.83 %); C 2000 fails (0.76 %). So C's threshold is not too strict.
- The operating slide's title and callout are now computed from the ladder ("Real triggers confirm the matched thresholds").
- Deck and report rebuilt. **Not yet republished.**

**Next steps:**
1. ~~Republish~~ done 2026-10-07 16:10: the deck now also carries the same-chamber context slides (`make_same_chamber_slides.py`: origin, timeline, chain map, bench progression, pairing, the T2 thread, the join), and its beam-angle-scale slides say the gap is explained (electron scattering, Geant4; cosmic in-situ scale stands). Live at https://dylan-neff.web.cern.ch/notes/two-track-limit.html.
2. Optionally, add a relative-tan dimension to the fixed chain's grid search (`WFT_TWO_TRACK_SEARCH=grid` in `wft/reco.py`) and re-run the R3 split. This targets the 76 % vs ~97 % gap on diverging A pairs at 3–12 mm.
3. Ship decision (Dylan): bundles with profc pairing, fixed-chain env + F on condor, rescue floor, full re-pass.

**Gotchas / decisions:**
- Divergence on real pairs is 30 mm × |Δtan| from the two donors' fits. It carries about ±1.9 mm of measurement error, so the <1 and 1–3 mm classes are blurred.
- The fitted tan is compressed (pointing slope 1/294 in A x and 1/387 in C x, against the geometric 1/235). The common-vertex line uses the geometric value.
- The oracle threshold is R2's own 99th percentile on its tan 0.3, view-x singles (A 25.1, C 29.1), so the new curves are directly comparable with R2.
- The IPC toy ignores capsule size, the pinwheel offsets, dead strips and efficiency. It is a prior, not a measurement.
- This work sits in the git worktree `~/PycharmProjects/nTof_x17_tt`. The main checkout is on `beam-off-cosmics`.

**Key files & commands:**
- `sept26_prelim_analysis/pair_angle.py`: the analysis (`asimov | oracle | data | table`).
- `sept26_prelim_analysis/make_pair_angle_slides.py`: the slides.
- `~/PycharmProjects/dylan-cern-site/scripts/slidedoc.py`: gained `Plot.scatter`.
