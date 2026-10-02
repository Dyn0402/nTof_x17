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

## Beam-off cosmics: tracking — updated 2026-10-02 (dylan-MS-7C84)

**Goal:** track the beam-off cosmic runs through the campaign reco to measure
the through-going background (170° cut, capsule DCA, arm-to-arm Δt).

**Done:**
- run_149/cos_0000 reconstructed (condor 4354841) and tracked with k borrowed
  from run_147 and run_150; report `ntof_cosmics/results/tracking/report.html`.
- Headline (~40 clean events): only 74–77 % of clean A–C through-goers pass
  170°; a capsule-free slope check says the borrowed k reads A ~8 % shallow,
  C ~12 % steep.

**In progress / where it stopped:**
- condor cluster **4355060** (1,028 jobs, the other 86 sub-runs of run_149)
  running on lxplus; tarballs land in `/eos/user/d/dneff/x17/cosmics_fullpass`.
- Pooling across sub-runs is not written yet.

**Next steps:** see `ntof_cosmics/HANDOFF_TRACKING_2026-10-02.md` §5 (fetch/
build/analyse loop, then pool, re-measure slope ratios, redo 170° with cosmic k).

**Gotchas / decisions:**
- **Never write to `/media/dylan/data`** while Dylan's backup is running (and
  in general check free space): `~/x17` symlinks there and every `paths.out`
  default resolves there. `cosmic_tracks.py` refuses `/media` outputs.
- Never run `k_arm` on cosmic runs (no `--out`, overwrites kcal/ silently).

**Key files & commands:**
- `ntof_cosmics/cosmic_tracks.py` — fetch / build / analyse
- `ntof_cosmics/make_tracking_report.py` — the report
- `sept26_prelim_analysis/condor/make_stage2_campaign.py --full-pass --tags-json <json>`
  — packaging; the tag lists used are in `ntof_cosmics/results/tracking/condor/`.
  Package dest must be off `/media`; after building, edit the shipped
  `stage2_fullpass.sub` EOS_STAGE2_OUT to `cosmics_fullpass`.
