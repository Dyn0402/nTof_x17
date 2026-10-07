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

**Resume:** beam/cosmic angle gap SOLVED (electron scattering, Geant4). Next: adopt the cosmic in-situ scale, then T1 + seeder min 3.

**Read first:**
- `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §10d–10g (record, numbers) and §11 (next steps in detail).
- `sept26_prelim_analysis/SAME_CHAMBER_PAIRS.md` (how this thread feeds same-chamber pairs).
- Slide note: https://dylan-neff.web.cern.ch/notes/beam-off-cosmics.html. Section 2 (slides 12–22) is
  2026-10-07, built by `ntof_cosmics/make_deck.py` + `deck_angle.py`.

**Goal:** a reconstruction that measures angles correctly in all cases (head-on included) at n_TOF, so
same-chamber coincident pairs can be reconstructed. Truth: run_149 cosmic through-goers (the A–C line).

**Done (2026-10-07):**
- Clock match on all of run_149's n_TOF overlap.
  - cos_0000–0034 × n_TOF 224678–687: 43 pairs, 93 % matched, core MAD 8–23 ns.
  - DREAM timestamps come from an lxplus extract; straddling sub-runs are trimmed.
- Cosmic wall test (`cosmic_wall_scale.py`): cosmics at A's wall read 1.15 × raw (bootstrap 1.12–1.19),
  matching the A–C line (1.11). Beam reads 0.89 campaign-wide, 0.91–0.93 after 20 ms, and 0.77–0.87 at
  10–20 ms. The gap is real.
- In-beam muons (`inbeam_through_goers.py`): A–C through-goers in beam runs read within 2–5 % of
  beam-off muons, although under beam their gain is ~1/3 lower. So gain, beam noise and space charge
  are not the cause.
- Geant4 (`g4_angle/`, condor): the data's wall estimator with an IDEAL reconstruction gives:

  | population | wall estimator |
  |---|---|
  | beam-capture electrons (median 3 MeV) | 0.59 |
  | electrons > 4 MeV | 0.92 |
  | electrons 2–4 MeV | 0.63 |
  | muons | 1.00 |

  **The beam/cosmic gap is electron scattering between the gap and the wall.** The wall is not angle
  truth for beam electrons, and the cosmic in-situ scale stands.
- Dylan's questions, answered on slides 19–21 and 17:
  - The A–C cosmic rate matches the muon flux: 578/h expected, 189/h observed, ε ≈ 0.57/chamber,
    zenith shape matches (`ac_cosmic_rate.py`).
  - Activation with T½ ≥ minutes is ≤ 0.2 Hz (`activation_bound.py`).
  - The late triggers (> 30 ms) are one T½ = 23.5 ± 0.8 ms exponential, not beam captures (t⁻⁴) and not
    ¹²B (20.2 ms disfavoured, Δχ² 125). Best reading: thermal-neutron die-away in the hall
    (`late_trigger_clock.py`).
  - Gain is lower under beam, but the angle scale doesn't follow it.
- The no-writes-to-/media rule is lifted (Dylan); the `_guard`s are no-ops.

**In progress / where it stopped:** nothing running, nothing half-done. Condor clusters 4402864/4402865
finished; outputs on EOS `full_sim/angle_scale/` and local `/media/dylan/data/x17/ntof_cosmics/g4_angle/`.

**Next steps (detail in tracking handoff §11):**
1. **Decide and adopt the beam angle calibration:** the cosmic in-situ scale (bundles `is2_A`, `is2_C`)
   for beam; retire the wall- and capsule-based k as truth.
   - Check what stage 3 / `k_arm` currently apply on the campaign and what changes downstream
     (opening angles, 170° cut, T1).
   - Needs Dylan's OK before touching campaign products.
2. **X17 opening-angle check:** the ideal gap line reads electrons shallower than their emission
   direction (median gap/gun 0.73 at 5 MeV, 0.91 at 8 MeV).
   - Check that the pair/IPC simulation used for the opening-angle spectrum carries this physics,
     i.e. comes from Geant4 hits, not from truth directions.
3. **Combined split-ab:** seeder min 3 + T1 `xy_pairing`, then re-derive T1's F on the in-situ bundles.
   Then the campaign re-pass (O4).
4. **Ambient thermal-neutron source in Geant4** (capture vertices in the arms' structure, τ ≈ 34 ms):
   the late-trigger population is missing from every sim. It matters for backgrounds, the wall maps and
   the data's exact 0.92.
5. Optional:
   - a `wft`-digitised forward model of Geant4 electron tracks (closes the data's 0.92 against the
     ideal fit);
   - a Geant4 flash run (> 13.6 MeV neutrons, RadioactiveDecay kept to 100 ms) to bound ¹²B;
   - the in-beam muon and cosmic wall tests on C and D;
   - the 10–20 ms dip.
6. ~~X17 board~~ done (log entry 2026-10-07). The two-track note (T1 worktree) now carries the T2 slides with the gap explained, and `SAME_CHAMBER_PAIRS.md` is updated on both branches.

**Gotchas / decisions:**
- **Never quote the wall scale (0.89/0.92) or capsule k_arm as the beam angle truth.** Both are set
  by electron scattering and population, not by the reconstruction.
- Two wall estimators:
  - beam needs the u-binned median test (`wall_edge_scale`);
  - cosmics need the edge likelihood (`ntof_scint_stack.ana.fit_wall_u`);
  - `fit_wall_u` is degenerate on beam: it returns 0.52 at > 20 ms.
- A–C through-goers in beam runs need sep < 10–20 mm; looser cuts are diluted by accidentals.
- Geant4 HitTree: cut t < 1e8 ns (RadioactiveDecay). Single guns aimed near the wall's outer edge carry
  wall-edge truncation; use them for trends only.
- The lxplus MX17_Full_Geant checkout is at 3d97437 (07-23 build, the one that made the nose campaign);
  local is ahead (23c5b5d).
- Still valid from earlier:
  - never use the ref-pinned v;
  - never run `k_arm` on cosmic runs (it overwrites kcal/);
  - the work dir `~/scratch/ntof_insitu` holds the run_149 waveforms (refetch with its `fetch.sh`,
    needs kinit).

**Key files & commands:**
- `ntof_cosmics/clock_match.py --run run_149 --subrun <sub> --ntof <run>`
  - n_TOF partials: `/media/dylan/data/x17/beam_july/ntof_data/`
  - DREAM ts: `.../dream_ts/run_149/`
- `ntof_cosmics/cosmic_wall_scale.py build|ana|report` → `/media/dylan/data/x17/ntof_cosmics/cosmic_wall_scale/`
- `ntof_cosmics/inbeam_through_goers.py`, `ac_cosmic_rate.py`, `late_trigger_clock.py`, `activation_bound.py`
- `ntof_cosmics/g4_angle/`:
  - `reduce_gap_wall.py` + `condor/` (lxplus job dir `~/condor/mx17_angle_scale/`);
  - `analyze.py` → `/media/dylan/data/x17/ntof_cosmics/g4_angle/report.html`.
- Slides: `.venv/bin/python ntof_cosmics/make_deck.py`, then
  `python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py ntof_cosmics/results/deck/beam-off-cosmics.html --slug beam-off-cosmics --force --deploy`

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
