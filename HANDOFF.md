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

## Beam-off cosmics: angle response + in-situ reco — updated 2026-10-09 (dylan-MS-7C84)

**Resume:** is2_v2 two-sided full pass on condor 4410513 (multi-day). When it lands: pull, kcal_is2_v2, stage 3, compare.

**Goal:** a reconstruction that measures angles correctly at n_TOF, head-on included, so same-chamber pairs can be
reconstructed.

**Read first:** `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §19 (latest: pilot vs production, two-sided on run_86,
launch), then §18/§17 (two-sided search), §16 (pilot), §15–§13. Report: `ntof_cosmics/results/pilot_is2_v1/report.html`.
Slide note of the review: <https://dylan-neff.web.cern.ch/notes/ac-insitu-angles.html>.

**Running on lxplus — condor cluster 4410513 (is2_v2, Dylan's call 2026-10-09):**
- Two-sided slope search (`WFT_TWO_SIDED=1`), TAN_MAX 1.0 raw, min 3 strips, in-situ bundles is2_A/is2_C, arms A,C,
  JobFlavour testmatch (72 h). 6,466 jobs. Built at 939dc2b (clean tree).
- Output: EOS `/eos/user/d/dneff/x17/sept26_fullpass_is2_v2`. Package: lxplus `~/sept26_stage2_is2_v2`
  (logs in `log/<run>/`), local copy `/media/dylan/data/x17/sept26_prelim/pkg_is2_v2`.
- Expected ~20 h median, up to ~55 h per job, so the whole pass will take days (6,466 jobs × ~20 h, queue-limited).
- **State at wrap-up (2026-10-09 afternoon):** 1,723 running, 4,426 idle, ~315 done, 2 held. AFS quota 29 %.
- **The 2 held jobs (procs 1748/1749) are run_104 `stat090_0016`, A and C.** Both fail on input: the decoded ROOT
  files (`…260730_11H29_000_03.root`, `…_07.root`) are empty (no `nt` tree), so this is not a reco bug and a re-run
  won't help. Check whether production also lacks this sub-run, then `condor_rm 4410513.1748 4410513.1749`.

**Next steps:**
1. Check `condor_q 4410513 -totals`, `condor_q 4410513 -hold -af ProcId HoldReason` and `lxstore status`. A hold for
   anything other than the run_104 empty input → read `~/sept26_stage2_is2_v2/log/<run>/<tag>.err`. Re-run failures
   with a new package built with `--done-list`.
2. When done: pull with `fetch_stage2.sh` (REMOTE=lxplus:/eos/user/d/dneff/x17/sept26_fullpass_is2_v2, plus LOCALPKG
   and OUT) into `/media/dylan/data/x17/sept26_prelim/fullpass_is2_v2`.
3. Make `kcal_is2_v2` = kcal_is2_v1 × (A 1.009, C 1.003) (the two-sided cosmic-closure shift), or re-derive.
4. Stage 3: `campaign_tracks --fullpass …/fullpass_is2_v2 --kcal …/kcal_is2_v2 --arms A,C --tan-max-true 0.6
   --out …/stage3_is2_v2`. Consider a wider cut: two-sided purity held to 0.6–0.8 on run_86.
5. Compare with production `stage3_fullpass` (not `stage3_campaign`, which has ~1/15 of the statistics) using
   `ntof_cosmics/pilot_compare.py` (repoint PILOT; steps match/summary/bins/same) and regenerate the report with
   `make_pilot_report.py`.
6. Still open: which k is right on A (in-situ is 0.85× production on the same tracks; board question); D in-situ
   bundle (v ≈ 28); B's HV in run_149.

**Key results (details §19):**
- Pilot is2_v1 (one-sided): confirmed yield 1.5–1.8× production per run in |true tan| < 0.6; its directions win
  2:1 on the same tracks. A's per-bin deficit is the k difference, not mirrors.
- run_86 one- vs two-sided: confirmed A 1,283 → 1,255, C 1,026 → 1,053; gated A 4,045 → 2,612, C 2,879 → 1,870;
  wall purity A 0.38 → 0.59, C 0.49 → 0.66. CPU ~20×.

**Gotchas / decisions:**
- Raw-tan constants (TAN_MAX, `TAN_MIN_SLOPE`, `FLOOR_TAN`, `W_SCAN_HALF`) change meaning with v.
- Sidecars before 7cd9db7 have no `tan_max_raw`; `build_tracks` assumes 0.6 for them.
- Arms without k (B, D, and 9 run/arms in stage3_fullpass) get `in_acceptance` False / are skipped.
- Never use the capsule band as the beam scale. Never run `k_arm` on cosmics. run_126 has no k.
- `yield_gate.py summary`'s report step throws KeyError without the is2 chain; the CSV is written first.
- Superseded: is2_v1 pilot (cluster 4409382, `sept26_fullpass_is2_v1`) and the staged one-sided `~/sept26_stage2_is2_v1`.
- lxplus SSH goes through a mux master. If it hangs, retry, or `kinit`.

**Key files & commands:**
- `sept26_prelim_analysis/condor/make_stage2_campaign.py --full-pass --arms A,C --insitu A=…/is2_A,C=…/is2_C
  --min-strips 3 --tan-max 1.0 --two-sided --flavour testmatch --version is2_v2 --dest <dir> [--done-list …]`.
- `wft/reco.py` (TAN_MAX, two-sided), `sept26_prelim_analysis/build_tracks.py` (`--tan-max-true`).
- `ntof_cosmics/pilot_compare.py`, `make_pilot_report.py`, `yield_gate.py`.

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
