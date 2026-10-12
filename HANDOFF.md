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
1. Why the vertical liquids (B, C) show no rise toward their PMT (C: 0.07 % top vs 0.34 % bottom) while A, D do.
2. Arm D: a third of its late tracks sit on other-arm triggers (A–C 4–18 %) — real particles or chamber pick-up? D's unbiased maps rely on them.
3. Join run_149 cosmic tracks to the slim → MIP-clean efficiencies.
4. Plastic bar placement / D wall group-0/2 cabling against the as-built drawings.

**Gotchas / decisions:**
- Self-triggered maps read near 1 by construction (trigger = wall-sum AND plastic); they show holes, not levels. Quote unbiased numbers.
- Dead end: loosening LATE_MS below 10 ms (`checks.late_scan`): not worth it.

**Key commands:** extract → ana → checks → make_figures → make_report → make_deck → add-note (see `ntof_scint_stack/README.md`).

## Calorimetry feasibility + MM dE/dx — updated 2026-10-08 (dylan-MS-7C84)

**Resume:** plan executed and closed; note https://dylan-neff.web.cern.ch/notes/calorimetry.html (`ntof_calorimetry/make_deck.py`). Open: C1.4 + bar thickness.

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
