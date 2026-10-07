# Handoff

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

## Beam-off cosmics: angle response + in-situ reco — updated 2026-10-07 (dylan-MS-7C84)

**Resume:** G4→wft digitiser built: electrons reco like muons; 5–13 % capsule-view data/sim residual. Next: read the 100-file reruns.

**Read first:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §13. §12 is superseded; §10g and §11 still hold.

**Goal:** a reconstruction that measures angles correctly at n_TOF (head-on included), so same-chamber pairs can
be reconstructed; decide whether the cosmic in-situ scale (`is2_A`/`is2_C`) is the beam angle scale.

**Done (2026-10-07, evening/night):**
- Inventory: the campaign applies run_145's capsule k on v = 42.6 bundles (A 1.27, C 1.62, D 1.77; tan = k·tan_raw).
  `campaign_tracks.py`'s docstring says `/ k`; the code multiplies.
- §12 `g4_angle/capsule_estimator.py`:
  - With an ideal line, the capsule estimators read ~1 on the G4 beam population.
  - Data capsule k is flat in time from 10 to > 60 ms, and the same in every quality split.
- §13 `ntof_cosmics/g4_digi/` digitiser:
  - G4 steps are converted to electrons, go through the bundle's kernel and template, are injected into real
    quiet run_145 raw ADC, then pass CNS, emulated hits, and the unchanged seeder plus `wft.reco`.
  - Muon gates pass on A and C.
  - **Electrons vs the ideal line reco like muons (1–2 %).** The falling capsule response is mostly the estimator
    (reco-charge window, spread, failed fits).
  - The data/sim residual remains: band A 1.19 vs 1.14 ± .02, C 1.19 vs 1.06; response 8–12 % lower, 3–6σ.
- lxplus AFS quota incident fixed:
  - single-gun ROOT moved to EOS `full_sim/angle_scale/single_root/`;
  - job scripts clear their scratch;
  - `lxstore` tool and rule in `~/.claude/CLAUDE.md`.

**In progress / where it stopped:**
- **Condor cluster 4404719 on lxplus** (submitted 2026-10-07; the local reruns were stopped for an OS switch).
  - 200 jobs: arms A and C × 100 nose step files, `is2` bundles.
  - Each writes `/eos/experiment/ntof/data/x17/full_sim/angle_scale/digi/g4_{A,C}_is2_full/sNNN.parquet`.
  - Job dir `~/condor/mx17_g4_digi/` (code.tar.gz, bundles, is2_t0.parquet, digi.sub, logs/). Check with
    `condor_q 4404719` and `ls .../digi/g4_A_is2_full | wc -l`.
  - Rebuild the package: see `ntof_cosmics/g4_digi/condor/run_digi_job.sh`. code.tar.gz = wft, ntof_tracking
    (`__init__`, `wft_beam`, `run145_target_imaging`), common, mx17_m1_map.csv, ntof_cosmics/g4_digi.

**Next steps:**
1. Pull and compare:
   `rsync -a lxplus:/eos/experiment/ntof/data/x17/full_sim/angle_scale/digi/ /media/dylan/data/x17/ntof_cosmics/g4_digi/`,
   then `PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/compare_data.py --sim-a g4_A_is2_full --sim-c g4_C_is2_full`
   (`analyse.load` concatenates `<label>/*.parquet`).
   Update §13's table with the full-stat numbers.
2. Chase the residual:
   - is2 data on run_145 split early/late (a non-capsule population?);
   - prod-bundle (v 42.6) digitised reruns: does the sim reproduce production's k_arm 1.27/1.62?
   - busy (non-quiet) overlay triggers as a systematic;
   - single-gun trends from `steps_single/`.
3. Then Dylan's decision: adopt is2 for beam, with a ~5–10 % systematic. Make it a re-pass with a version tag;
   consumers are the opening angle, the 170° cut, `slope_reliable`, T1's F (§11).
4. §11 items 2–5 are unchanged (X17 opening-angle compression, combined split-ab, ambient-neutron mode).

**Gotchas / decisions:**
- Never quote the wall (0.89/0.92) or the capsule k as beam angle truth. Both are estimator-on-population; compare
  to the g4_digi forward model.
- The digitiser's response model IS the fit model. It tests ionisation shape, noise, seeding and truncation, not
  model-vs-chamber (that is calibrated on cosmics).
- Charge scale: A 11.8, C 10.8 ADC/e (data median x_q_sum 1553/1585). Sim foot A 16.35, C 16.4; D_PERP 234.6
  (from muon guns).
- ~18–23 % wrong-sign x fits at |tan| > 0.35, in sim and data alike.
- lxplus: AFS home is code-only. Condor scratch leftovers get copied back to AFS. Check with `lxstore status`.
- Still valid: never use the ref-pinned v; never run `k_arm` on cosmic runs; `~/scratch/ntof_insitu` holds the
  is2 bundles and the run_149 waveforms.

**Key files & commands:**
- `ntof_cosmics/g4_digi/digitise.py` (model), `run_digi.py muons|g4 --arm A|C --bundle is2_A|is2_C`,
  `analyse.py <label>`, `compare_data.py`.
- `ntof_cosmics/g4_digi/extract_steps.py` + `condor/` (lxplus job dir `~/condor/mx17_digi_steps/`).
  Steps on EOS `full_sim/angle_scale/steps_nose/` (100) and `steps_single/` (15); local copies under
  `/media/dylan/data/x17/ntof_cosmics/g4_digi/`.
- `ntof_cosmics/g4_angle/capsule_estimator.py --data` (§12 tables).

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
1. Why the vertical liquids (B, C) show no rise toward their PMT (C: 0.07 % top vs 0.34 % bottom) while A, D do.
2. Arm D: a third of its late tracks sit on other-arm triggers (A–C 4–18 %) — real particles or chamber pick-up? D's unbiased maps rely on them.
3. Join run_149 cosmic tracks to the slim → MIP-clean efficiencies.
4. Plastic bar placement / D wall group-0/2 cabling against the as-built drawings.

**Gotchas / decisions:**
- Self-triggered maps read near 1 by construction (trigger = wall-sum AND plastic); they show holes, not levels. Quote unbiased numbers.
- Dead end: loosening LATE_MS below 10 ms (`checks.late_scan`): not worth it.

**Key commands:** extract → ana → checks → make_figures → make_report → make_deck → add-note (see `ntof_scint_stack/README.md`).
