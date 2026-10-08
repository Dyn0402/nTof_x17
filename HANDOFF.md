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

## Beam-off cosmics: angle response + in-situ reco — updated 2026-10-08 (dylan-MS-7C84)

**Resume:** is2_v1 re-pass staged, NOT launched. A raw-tan gate cut (TAN_MAX) needs Dylan's decision first; then the B/D options.

**Read first:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §14 (today), then §13. §10g and §11 still hold; §12 is
superseded. Slide note of the review: <https://dylan-neff.web.cern.ch/notes/ac-insitu-angles.html>.

**Goal:** a reconstruction that measures angles correctly at n_TOF, head-on included, so same-chamber pairs can be
reconstructed.

**Done (2026-10-08):**
- Review before the re-pass, built from data:
  - `repass_readiness.py all` → `ntof_cosmics/results/repass_readiness/`;
  - `make_insitu_deck.py` → 15-slide note, published.
- Cosmic closure as delivered: is2 A is right to −6/+8 %; production A is 11–25 % steep. C x is non-linear (1.03 → 0.89).
- In-beam muons: is2 is right to 1–4 %.
- **New blocker:** `wft.reco.TAN_MAX = 0.6` acts on the bundle's RAW tan.
  - True-angle reach: production A 0.76 / C 0.97; is2 ≈ 0.58.
  - Smoke sub-run, C: gated tracks −42 % against m3 (the production bundle with the 3-strip seeder); scintillator-confirmed tracks −10 %.
  - The smoke test compared condor with local only, so it could not catch this.
- B/D feasibility: `bd_feasibility.py all` → `ntof_cosmics/results/bd_feasibility/report.html`.
  - **D x truth = D's own wall on cosmics:** 1.525 × production raw (bootstrap 1.50–1.56). Implied v ≈ 28, not 36.6.
    Ran with `cosmic_wall_scale.py --arm D`, a new option; A's outputs are unchanged.
  - **B is nearly blind to cosmics:** it lights on 12.7 % of the crossings D predicts (C: 64.6 %), and its reco keeps 3 %.
    So B–D lines are not a practical truth for D. Under beam, B's hit efficiency is comparable to the other chambers.

**In progress / where it stopped:** nothing half-done. Dylan will choose among the options later.

**Next steps (Dylan's choices, in the note's order):**
1. TAN_MAX: keep ≈ 0.58 true as the stated acceptance, OR make it a bundle-carried true-angle cut and validate
   |tan| 0.6–1.0. Validate with g4_digi guns and corner-cutting cosmics: A–D/C–D lines obey tan₁·tan₂ = 1, so both sit at |tan| ≈ 1.
   Then re-run the smoke sub-run.
2. A yield gate for the real is2 bundles on other periods (run_86, run_110, run_156). Use `seed_beam_test` scint/compare.
3. A pilot pass: one sub-run per run.
4. Interim (is2_v1 alone) vs combined with T1's `xy_pairing`. Then launch: `cd ~/sept26_stage2_is2_v1 && condor_submit stage2_fullpass.sub`.
   Stage 3 goes into `stage3_is2_v1`, never into production.
5. D: an in-situ bundle with v ≈ 28, iterated to wall s = 1.
   - Needs D waveforms (FEU 01/02) for the clock-matched run_149 sub-runs.
   - Mask the dead channels in the wall fit.
   - y has no truth: carry x's scale with a 3–5 % systematic.
6. B: check B's HV in run_149 before concluding that the chamber is blind. Otherwise keep B as the hit-mode tag.

**Gotchas / decisions:**
- Any constant in raw-tan units changes meaning when v changes: TAN_MAX, and also `TAN_MIN_SLOPE` and `FLOOR_TAN`.
  When bundles change, compare yields against production, not only condor against local.
- `cosmic_wall_scale` (`fit_wall_u`) is valid on cosmics only; it is degenerate on beam. D's per-|tan| bins are unusable.
- Never use the capsule band as the beam scale. Its only legitimate use is the run-to-run ratio (gas).
- The muon scale depends on the sep cut, so compare only at equal cuts. Never quote the wall (0.89/0.92) as beam truth.
- Never run `k_arm` on cosmic runs. Never use the ref-pinned v.
- run_126 has no k in either chain.
- Production C used `calib_bundle_prelim` (det6 lp, v 42.6). Local copies are in `~/scratch/ntof_insitu/bundles/fp145_{A,C}`.
- lxplus: AFS is for code only. Check `lxstore status` before and after a large submission.

**Key files & commands:**
- `ntof_cosmics/repass_readiness.py all` (needs the smoke products in `sept26_prelim/smoke_is2_v1` and
  `~/scratch/ntof_insitu`), then `make_insitu_deck.py`, then
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py ntof_cosmics/results/deck/ac-insitu-angles.html --slug ac-insitu-angles --force --deploy`.
- `ntof_cosmics/bd_feasibility.py all`; `cosmic_wall_scale.py build|ana --arm D` → `/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale/arm_D/`.
- `ntof_cosmics/g4_digi/`; `inbeam_through_goers.py pooled`; `sept26_prelim_analysis/k_insitu.py`;
  `condor/make_stage2_campaign.py --version`.

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
