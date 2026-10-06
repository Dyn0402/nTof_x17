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

## Beam-off cosmics: angle response + in-situ reco — updated 2026-10-06 (dylan-MS-7C84)

**Resume:** Fix the chamber reco with the A–C cosmic line as truth. Next: beam purity check of seeder min 3 (run_145/147).

**Goal:** a reconstruction that measures angles correctly in all cases (head-on included) at n_TOF, so that
same-chamber coincident pairs can be reconstructed. run_149 through-goers give truth: the line through
chambers A and C.

**Done:**
- Pooled run_149 (87 sub-runs). The old handoff's slope ratio was inverted: s/j = k_b/k_true, so A's tans
  read ~10 % too LARGE. Report: `ntof_cosmics/results/tracking/pooled/report.html`.
- The bulk angle-scale error is `wft_beam.make_bundle` replacing the bench v (fitted with the kernel)
  with the 42.6 prior. The in-situ geometric v is A 38.6/37.7 and C ≈ 28 µm/ns.
- **Head-on failure = beam seeder `MIN_STRIPS_BEAM = 5`** (bench uses 3). With min 3: A–C pairs +66 %,
  near-normal tracks ×5, near-normal σ_tan 0.2 → 0.03–0.06 (`ntof_cosmics/seed_test.py`).
- Ruled out with bench M3 truth: S/N ÷8, n_TOF 20-sample framing, real n_TOF noise. Also ruled out:
  the template, ZS, the fit window, the w-scan range, the t0 prior.

**In progress / where it stopped:**
- Nothing running. Remaining non-linearity: A x is mildly S-shaped; chamber C (old det6 lp kernel) is
  non-linear with 10–17 % core tails.

**Next steps:**
1. Seeder min 3 on a beam sub-run (run_145/147): gate pass rates, χ²/dof, junk and isochronous
   deposits, near-normal capsule-track yield. The 5-strip minimum was chosen for beam junk.
2. In-situ kernel for chamber C (start from r06; objective = free-fit closure vs truth, not the
   ref-pinned χ²).
3. Campaign re-pass (condor) with seeder 3, geometric v and new bundles. Then redo the beam/cosmic
   comparison.

**Gotchas / decisions:**
- Never use the ref-pinned fitted v (χ²(v) valley, ANALYSIS_STATE S8); take v from free fits against truth.
- Near-normal tracks are under-represented in any sep-selected sample (circular). Judge head-on
  on one-track-per-arm events with no sep cut.
- No writes to `/media/dylan/data`. Work dir `~/scratch/ntof_insitu` (17 GB of run_149 A/C
  waveforms, caches, bench test scripts `degrade*.py`, `noise_inject.py`). Durable, outside the repo.
- Never run `k_arm` on cosmic runs (it overwrites kcal/).

**Key files & commands:**
- `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §7–9: full record and numbers.
- `ntof_cosmics/insitu_calib.py`: truth / cache / profile / hyper / reco / score / t0meas / mkbundle /
  corridor / joint / implied. `WFT_BEAM_BASE=~/scratch/ntof_insitu/beam/`; `--work ~/scratch/ntof_insitu`.
- `ntof_cosmics/seed_test.py --work ~/scratch/ntof_insitu --subs ~/scratch/ntof_insitu/subs.txt --min 3`
- `ntof_cosmics/angle_response.py`, `make_pooled_report.py`: the pooled report.
- If the work dir is gone: `~/scratch/ntof_insitu/fetch.sh` (needs `kinit`), then `insitu_calib.py truth --subs`, then `cache`.

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
