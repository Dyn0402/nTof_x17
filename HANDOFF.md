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

**Resume:** The in-situ re-pass (is2_v1, A and C) is built, smoke-tested and staged on lxplus. It is **not
submitted**: whether to adopt is2 for beam is Dylan's decision.

**Read first:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §13: "Chasing the residual", then "Consequence for
adopting is2", then "The in-situ re-pass is built". §10g and §11 still hold; §12 is superseded.

**Goal:** a reconstruction that measures angles correctly at n_TOF, head-on included, so same-chamber pairs can be
reconstructed.

**Findings (2026-10-08):**
- Full-stat g4_digi: data/sim band residual A +3.6 %, C +10 %.
  - It is not time, charge, a non-pointing population or the kernel. C is the same under r06-det7 and lp.
  - The band depends on the source model: about 1.00 at fixed source, rising with source width.
- Production's per-run k_arm band varies with the gas: A ±1.2 %, C ±3.8 %, D ±4.3 %. In-beam muons confirm it.
- In-beam A–C muons, pooled and gas-normalised: is2 reads beam-period muons right to 1–4 %. Its tans are
  slightly too large, if anything.
  - So **the band residual is not an angle-scale error** (~3σ A, ~7σ C).
  - The in-beam muons are the scale truth for beam.
- The y-view residual (3–4× x) is still open. The likely cause is an under-modelled source along the capsule's long axis.

**Re-pass prepared (not launched):**
- k: `sept26_prelim_analysis/k_insitu.py --version is2_v1` → `/media/dylan/data/x17/sept26_prelim/kcal_is2_v1/`.
  - k = muon norm (A 0.976, C 0.962) × band(run)/band(run_145).
  - B and D get no k.
- Reco: `make_stage2_campaign.py --full-pass --arms A,C --insitu … --min-strips 3 --version is2_v1`.
- Smoke cluster 4405910 (run_145 stat090_0000) matches the local is2 reco: identical events, 99.5 % of fits
  bit-close. The remainder are multi-candidate flips.
- Stage 3 with `--kcal` works.
- Package: lxplus `~/sept26_stage2_is2_v1` (6466 jobs, ≈4300 CPU-h, ~14 GB on EOS). Launch:
  `condor_submit stage2_fullpass.sub`.

**Next steps:**
1. Dylan: adopt is2 for beam? If yes, launch the package and run stage 3 into `stage3_is2_v1`, never into
   production. The systematic is ±2–3 % from the norm, plus the x/y spread (A 3 %).
2. y-view residual: model the source along the capsule's long axis (v).
3. Busy-overlay systematic and single-gun trends (`steps_single/`), both optional.
4. Seeder min 3 against T1's `xy_pairing`, plus the consumers in §11 (opening angle, 170° cut, `slope_reliable`, F).

**Gotchas / decisions:**
- Never use the capsule band as the beam scale. Its only legitimate use is the run-to-run ratio (gas).
- The muon scale depends on the sep cut, so compare only at equal cuts.
- Never quote the wall (0.89/0.92) as beam truth.
- Never run `k_arm` on cosmic runs.
- Never use the ref-pinned v.
- Production C used `calib_bundle_prelim` (det6 lp, v 42.6), not r06k_C. Local copies are in
  `~/scratch/ntof_insitu/bundles/fp145_{A,C}`.
- lxplus: AFS is for code only. Check `lxstore status` before and after a large submission.

**Key files & commands:**
- `ntof_cosmics/g4_digi/`:
  - `digitise.py`, `run_digi.py --sim-bundle`;
  - `compare_data.py --data is2|prod`;
  - `residual_checks.py`.
- `ntof_cosmics/inbeam_through_goers.py pooled` → `/media/dylan/data/x17/ntof_cosmics/inbeam_through_goers/pooled_norm.csv`.
- `sept26_prelim_analysis/k_insitu.py`, `campaign_tracks.py --kcal --arms`, `condor/make_stage2_campaign.py --version`.

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
