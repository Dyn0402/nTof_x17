# Fixing the n_TOF tracking audit: the roadmap (started 2026-11-22)

The tracking audit of 2026-11-21 found three problems in the track sample, several code errors and a set of estimator
biases. The findings are 11 to 23 in `README.md`, the report is `audit_report/report.html`, and the three code reviews
are kept as they were written in `audit_report/reviews/`. This file is the plan for fixing all of them, in the order we
do them. Each item names the finding it comes from (README number, or the review and its item number), what we change,
and how we check it. The status column is updated as we go.

All the work uses the data we already have. Most fixes are at table level: we rebuild stage 3 and the analysis view
from the existing stage-2 reconstruction outputs. Only phase 4 touches the waveform reconstruction itself.

Status: `done` = committed; `wip` = in progress; `todo`; `decide` = waits for Dylan.

Phase 4 (decided 2026-11-22): fix the code, test each change on a few sub-runs against pass 1, then Dylan decides on a
pass 2 from the numbers.

## Phase 0. Write down everything the audit found

| # | item | status |
|---|---|---|
| 0.1 | The three code reviews saved verbatim (`audit_report/reviews/`) | done |
| 0.2 | README finding 23: the review items the summary left out | done |

## Phase 1. Code fixes that need no rebuild to be correct

These change code only. Their effect on the products comes with the rebuild in phase 2.

| # | item | from | change | status |
|---|---|---|---|---|
| 1.1 | Scint-stack y slope has the wrong sign | 20 | `vt_view.py` and `map_view.py` write `tan_raw_y` with the reco sign (the stage-3 contract), so `extract.py` is right again | done |
| 1.2 | Beam-off selected as "time is null" | 13 | `pass1_tracks.cosmic_tags()` / `cosmic_filter()` (registry class); `line_residuals` and `bd_map` use it | done |
| 1.3 | Cosmic lines would be built on mapped slopes on a re-run | stage-3 review 7 | `pass1_tracks.nomap()`; `line_residuals` and `bd_map` build on the `_nomap` twins (censored-window cut kept) | done |
| 1.4 | `map_view` would map a mapped view; `vt_view --force` would overwrite the mapped view | stage-3 review 7 | both refuse a table that has `_nomap` columns | done |
| 1.5 | Bow block re-maps a paired winner with the other pair's coordinate | 21 | `wft/reco.py`: `apply_bow(skip=...)`, paired winners keep their partner's mapping (takes effect in any new reco; pass 1 is recomputed at table level in 2.3) | done |
| 1.6 | Stage-1 merge duplicates two rows | 22 | `build_tracks`: one stage-1 row per event (classified row kept), `validate='many_to_one'` | done |
| 1.7 | `FOOT_SIM['C']` 16.4 against 17.3 in the Geant4 geometry | 17, imaging review 7 | `capsule_tof.FOOT_SIM['C'] = 17.3` | done |

## Phase 2. Rebuild stage 3 and the analysis view once, with every table-level fix

One rebuild carries all of these. Its order is: slim products, stage 3, v(t) view, map refit, map, censored-window cut,
the new quality flags. The old stage 3 stays on EOS beside the new one until the comparison is done.

| # | item | from | change | status |
|---|---|---|---|---|
| 2.1 | Sub-runs with two slim files read from the first only | 12 | every reader takes every file (`paths.slim_files`, `concat_slim`); trigtime, slim export and stage 1 regenerated for the 77 sub-runs into `$WORK_DATA/ntof_pass1/context_r2/` (old rows identical; 3.39 M → 6.95 M triggers with a time) | done |
| 2.2 | Stage 3 rebuilt with the context columns | 12 | `ntof_stage3.py --stage1-dir/--trigtime-dir context_r2` into `stage3_r2/`: 543/543 built; untimed production tracks 14.72 % → 0.68 % (true gaps between n_TOF segments) | done |
| 2.3 | Bow-partner tangents in pass 1 | 21 | `ntof_stage3.bow_patch` recomputes every paired candidate's tangent with `wft.reco.bow_pairs`; run_145 0000: 0.6 % (A) / 0.2 % (C) of late gated tracks change | done (in the rebuild) |
| 2.4 | `dca_axis_mm`, `in_bore` about the nominal axis | 22, reco review 2 | `quality_join` adds `dca_source_mm` about `SOURCE_XZ_MM`; axis columns kept | done |
| 2.5 | Bow mesh height not applied to positions | 22, stage-3 review 4 | tested on run_145 late pointing: half-width A 12.44 → 12.48 mm, C 13.46 → 13.46 mm. No gain, so not applied; `x/y_bow_h` are carried in stage 3 | done (not applied) |
| 2.6 | κ rescale also scales the bow tilt term | stage-3 review 6 | `vt_join` rescales (tan + a)·r − a with the carried `x/y_bow_a`; measured change 0.0001–0.0004 rms (A, C), 0.003 (D y) | done (in the rebuild) |
| 2.7 | Map refit on real cosmics | 13 | `run_r2_chain.sh` (line_residuals, angle_map fit, bd_map on `stage3_r2_train`, `merge_maps`, map_view) | todo (next) |
| 2.8 | Post-flash ringing fits pass the gate | 14 | Dylan: fold into `gated`, but look for a smarter rule first. Charge-only variants (flat, run-MIP, amplitude, 3-variable likelihood) are equivalent on A/C; a per-event waveform-noise threshold does not beat them (`low_charge_study.py`). Waveform ringing study (`rerun/ntof/ringing/`): removable, but the flag adds 0.04–0.08 purity before 8 ms only. Choice asked 11-22 | decided 11-23 (Dylan): before 8 ms the charge cut (q >= 600 ADC) AND the own-window waveform flag; from 8 ms the waveform flag alone. Needs the flag: from a pass 2, or a ~300-job waveform pass |
| 2.9 | D contamination; C windows partly censored | 15, reco review 9 | `quality_join` `d_faint`: amp_norm < 0.25 or x window ≥ 0.8 censored, in `gated` (late D: keeps 98.7 % of reference, 20–30 ms purity 0.51 → 0.75); χ² never helped. C partial censoring: not done | done (D) |
| 2.10 | Where the new cuts live | — | decided: folded into `gated` (pre-cut value kept as a column), in a `quality_join.py` step after `censor_join` | todo |
| 2.11 | The new stage 3 on EOS and in the ledger (HELD by Dylan 11-23 until the pass-2 decision; downstream rerun held too) | — | decided: in place at `Analysis_pass1/stage3/`, the old set moved to `stage3_r1_pre_audit/`; MD5SUMS, retire record | todo |

## Phase 3. Estimators and analysis scripts

| # | item | from | change | status |
|---|---|---|---|---|
| 3.1 | `crossing` fails abruptly with background | 17, imaging review 2 | `capsule_tof.crossing` refits within 100 mm of the first crossing. Injection (early tracks added): A −10.92 stays −10.86 at +40 % (was −17.31), D stays −4.8 to +20 % (was −8.6). The clean values move C −10.93 → −11.14, D −3.86 → −4.86 | done |
| 3.2 | D z from `capsule_tof tilt` is not a measurement | 16, imaging review 1 | `tilt` takes D at ≥ 25 ms with the wall (as `capsule_pose` data_z); the clean D rule (2.9) to be added | wip |
| 3.3 | The −1.7 mm is quoted downstream | imaging review 1 | `compare_chains` copies `capsule_tof tilt`, so it follows on the rerun; `PASS1_DOWNSTREAM.md` text in 5.2 | todo (5.1) |
| 3.4 | `campaign_imaging` biases | 17, imaging review 3 | `k_arm.coincident_tracks_view` (view `gated`, ToF ≥ 20 ms, wall and plastic fired, no plastic-position truncation) is the default sample (`X17_IMAGING_SOURCE=reco` keeps the old one); the reported variant is the re-centred window. run_145: A −10.80, C −10.73, D −5.07 (old: −7.70, −9.80, −6.11) | done |
| 3.5 | Focus split about the nominal axis | 17, imaging review 4 | both take the miss about `pass1_tracks.SOURCE_XZ_MM` (−10.9, −4.5) | done |
| 3.6 | Pose fit: stale `Z_BD`, library-size term | imaging review 7, 9 | `Z_BD = {D: (−4.5, 1.0)}` from `SOURCE_XZ_MM`; `chi2_x` adds each toy crossing's library error (`xlib_<arm>`) in quadrature to `SIG_X_SYS` | done |
| 3.7 | `track_quality` null keeps part of the pointing; stale cache | 17, imaging review 8, 9 | full-shuffle null; the cache is keyed on the view (directory, table count, newest table) | done |
| 3.8 | `capsule_point` axis from the is2 era | imaging review 9 | `SOURCE_XZ_MM` | done |
| 3.9 | Same-chamber rate fit from 12 ms | 14 | `same_chamber_compare` with `FIT_MS` 12 on the new `gated`; check the pair fraction is flat from 12 ms | todo |
| 3.10 | Scint stack and calorimetry on the corrected y sign | 20 | rerun `ntof_scint_stack` extract + ana, `ntof_calorimetry/wall_plastic` | todo |
| 3.11 | y crossing and the trigger plastic clip | imaging review 9 | stated limitation (−0.5 to −1.3 mm, toy only; README finding 23) | done (limitation) |

## Phase 4. Reconstruction-level fixes (wft/, the beam driver)

These change how tracks are found and fitted, so they act only through a new reconstruction of the n_TOF data. We fix
the code, test each change on a few sub-runs against pass 1, and then decide whether a pass 2 is worth it.

| # | item | from | change | status |
|---|---|---|---|---|
| 4.1 | x–y coincidence uses the bench `tc_tol` | reco review 4 | studied 11-23 (`xy_match/README.md`): the matching is replayable from the candidates table; an n_TOF likelihood is 1–4 points better on overlays but not better where it disagrees with pass 1 (junk events); two-track swaps move the opening angle by 0.1–0.2°. Keep pass-1 matching; n_TOF `tc_tol` + matching-score column; pass 2 stores candidate depth profiles | done (study); score column todo |
| 4.2 | Out-of-time tracks are biased, not rejected | reco review 6 | `t0_railed` stays a flag (Dylan 11-23) until study (c) is done: `t0_study/` (pile-up, accidental pairs, railed tracks), then re-reco of the test sub-runs with a wider t0 box and pinned children. Injection test done 11-23 (`t0_study/README.md`): tc follows the arrival (A slope 1 from −120 to +420 ns), t0 does not (≤ +90 ns), angle within 0.5° inside −120…+300 ns | wip |
| 4.3 | Two-track children ignore the t0 pin; `split_child` not in stage 3 | 22, reco review 8 | pin the children (reco); `split_child` is carried into stage 3 from the rebuild on (`x/y_split_child`) | todo (pin) |
| 4.4 | Seeder keeps the five widest clusters | reco review 10 | rank by significance | todo |
| 4.5 | Head-on rescue not wired into the beam driver | 22, reco review 11 | wire `headon_seeds` / `_rescue_` windows into `wft_beam` | todo |
| 4.6 | `quality_ok` reference median set by the ringing | 14, reco review 3 | reference from late tracks | todo |
| 4.7 | Candidates carry no `bow_h` | stage-3 review 4 | write it | todo |
| 4.9 | D oscillating channels (19 x, 42 y unmasked) | ringing study 11-23 | add to D's hot list; test on a re-reco of a few sub-runs | todo |
| 4.10 | A/C ringing correction | ringing study 11-23 | `deflect` v1, gated, mode `deflect_hits`, plus the own-window waveform flag; not on D | todo (pass 2) |
| 4.11 | Candidate depth profiles not stored | xy_match 11-23 | write the 20-bin profile per candidate (enables `profile_distance` matching at table level) | todo (pass 2) |
| 4.12 | Plane-wide seeding floor loses the second track | t0 injection 11-23 | a strip must pass 10 % of the strongest strip in the whole plane (`SIG_FLOOR_LOCAL_MM = 0`), so a bright track hides a fainter one: two in-time tracks > 50 mm apart are both found 76 % (A) / 71 % (C) of the time; a 30 mm local floor gives 87 % / 84 % on the same injected pairs (`t0_study/README.md`). Measure its effect on single tracks and fakes, then decide; also check that g4_digi pairs see the same loss | todo (test) |
| 4.8 | Smaller | reco review 12 | events table pairs the plane winners; `tan_err` at its floor; `dt_xy` bench key | todo |

## Phase 5. Rerun the downstream and write it up

| # | item | status |
|---|---|---|
| 5.1 | `PASS1_DOWNSTREAM.sh` on the new view (~6 h), `compare_chains`, the before/after page | todo |
| 5.2 | README, `PASS1_DOWNSTREAM.md`, `docs/PARAMETERS.md`, HANDOFF, the analysis board | todo |
