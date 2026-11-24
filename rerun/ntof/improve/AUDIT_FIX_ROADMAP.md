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
| 2.7 | Map refit on real cosmics | 13 | `run_r2_chain.sh` (line_residuals, angle_map fit, bd_map on `stage3_r2_train`, `merge_maps`, map_view) | done 11-23 (`angle_map_r2.json`, `stage3_r2_final`); a spline map is better, see 6.1 |
| 2.8 | Post-flash ringing fits pass the gate | 14 | Dylan: fold into `gated`, but look for a smarter rule first. Charge-only variants (flat, run-MIP, amplitude, 3-variable likelihood) are equivalent on A/C; a per-event waveform-noise threshold does not beat them (`low_charge_study.py`). Waveform ringing study (`rerun/ntof/ringing/`): removable, but the flag adds 0.04–0.08 purity before 8 ms only. Choice asked 11-22 | decided 11-23 (Dylan): before 8 ms the charge cut (q >= 600 ADC) AND the own-window waveform flag; from 8 ms the waveform flag alone. Needs the flag: from a pass 2, or a ~300-job waveform pass |
| 2.9 | D contamination; C windows partly censored | 15, reco review 9 | `quality_join` `d_faint`: amp_norm < 0.25 or x window ≥ 0.8 censored, in `gated` (late D: keeps 98.7 % of reference, 20–30 ms purity 0.51 → 0.75); χ² never helped. C partial censoring: not done | done (D) |
| 2.10 | Where the new cuts live | — | decided: folded into `gated` (pre-cut value kept as a column), in a `quality_join.py` step after `censor_join` | done 11-22/23 (`quality_join.py`; the flags are in `stage3_r2_final`). The charge rule itself is still off (`LOW_CHARGE_RULES` = none): 6.4 |
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
| 4.2 | Out-of-time tracks are biased, not rejected | reco review 6 | `t0_railed` stays a flag (Dylan 11-23) until study (c) is done: `t0_study/` (pile-up, accidental pairs, railed tracks), then re-reco of the test sub-runs with a wider t0 box and pinned children. Injection test done 11-23 (`t0_study/README.md`): tc follows the arrival (A slope 1 from −120 to +420 ns), t0 does not (≤ +90 ns), angle within 0.5° inside −120…+300 ns. 11-24: accidental shares of the rate fit stand (A 6.7 %, C 10.5 %; no constant-source accidentals, `look11.py`); a ±360 ns box gains little (prior holds t0) | done (study) |
| 4.3 | Two-track children ignore the t0 pin; `split_child` not in stage 3 | 22, reco review 8 | pin the children (reco); `split_child` is carried into stage 3 from the rebuild on (`x/y_split_child`) | code in (`reco.TWO_TRACK_T0_PRIOR`), test `phase4/` |
| 4.4 | Seeder keeps the five widest clusters | reco review 10 | rank by significance | code in (`seed.SEED_RANK=charge`), test `phase4/`. **Re-audit 11-24: correct first.** The cap of five is applied in `seed_candidates` before the wide-band veto of `wft_beam`, and a wide noise band has a large summed charge, so it takes a slot and is then dropped. Rank by significance and cap after the veto (6.6) |
| 4.5 | Head-on rescue not wired into the beam driver | 22, reco review 11 | wire `headon_seeds` / `_rescue_` windows into `wft_beam` | code in (`--headon-rescue`), test `phase4/` |
| 4.6 | `quality_ok` reference median set by the ringing | 14, reco review 3 | reference from late tracks | code in (stage 3, `X17_QUALITY_REF_LATE_MS`) |
| 4.7 | Candidates carry no `bow_h` | stage-3 review 4 | write it | done (candidates `bow_h`) |
| 4.9 | D oscillating channels (19 x, 42 y unmasked) | ringing study 11-23 | add to D's hot list; test on a re-reco of a few sub-runs | bundles made (`ringing/make_osc_bundles.py`), test `phase4/`. **Re-audit 11-24: first look says net loss.** In `all`, 21-30 real late D tracks are removed by the new flags against at most 8 gained (4 acquisitions of run_145). A table-level flag on the pass-1 bundle is the alternative (6.5). Judge `osc` on `gated_uncensored` and `gated_precut` too |
| 4.10 | A/C ringing correction | ringing study 11-23 | `deflect` v1, gated, mode `deflect_hits`, plus the own-window waveform flag; not on D | code in (`ntof_reco --ringing deflect_hits`), test `phase4/`. Re-audit 11-24: the corrected-event flag was wrong for the first event of each reader (fixed, redeploy); **the per-track waveform flag of rule 2.8 is not written yet** (6.6) |
| 4.11 | Candidate depth profiles not stored | xy_match 11-23 | write the 20-bin profile per candidate (enables `profile_distance` matching at table level) | code in (`reco.STORE_PROFILES`); on only in `all`: turn it on in every configuration (it changes no fit) |
| 4.12 | Plane-wide seeding floor loses the second track | t0 injection 11-23 | a strip must pass 10 % of the strongest strip in the whole plane (`SIG_FLOOR_LOCAL_MM = 0`), so a bright track hides a fainter one: two in-time tracks > 50 mm apart are both found 76 % (A) / 71 % (C) of the time; a 30 mm local floor gives 87 % / 84 % on the same injected pairs (`t0_study/README.md`). Measure its effect on single tracks and fakes, then decide. g4_digi pairs go through the same seeder (P(2|2) A 0.48, C 0.42), so data/G4 already includes the loss: a fix gains efficiency (the same seeder is used, but whether the digitised relative amplitudes are realistic was not checked; re-run `g4_digi` with the floor if it is adopted) | code in, test `phase4/`. Re-audit 11-24: no visible gain in the data of `all` (158 A / 103 C two-track events from 12 ms in the five sub-runs): judge it with the injection test on the new code |
| 4.8 | Smaller | reco review 12 | events table pairs the plane winners; `tan_err` at its floor; `dt_xy` bench key | todo |

## Phase 5. Rerun the downstream and write it up

| # | item | status |
|---|---|---|
| 5.1 | `PASS1_DOWNSTREAM.sh` on the new view (~6 h), `compare_chains`, the before/after page | todo |
| 5.2 | README, `PASS1_DOWNSTREAM.md`, `docs/PARAMETERS.md`, HANDOFF, the analysis board | todo |

## Phase 6. The re-audit of 2026-11-24

Dylan asked for a second audit before the test campaign runs, and for the implications of the bench drift-geometry
result for n_TOF. Report: `reaudit/report.html` (README there; the tests are in `reaudit/tests/`). The items below are
what it adds. Nothing of 6.1-6.5 needs a new reconstruction.

| # | item | from | change | status |
|---|---|---|---|---|
| 6.0 | The test chain failed on every sub-run; no comparison script | T1, T2 | `ntof_stage3.py --allow-missing`, `phase4/stage3_chain.sh`, `phase4/compare.py`; `ref` on code `e86028b5` reproduces `stage3_r2_final` (8 acquisitions, largest difference 1.3e-6 mm) | done |
| 6.1 | The 50 mm cell map is noisy in its per-cell scales | T6, W2a | a penalised spline map (tilt as the gradient of one potential, one scale per view; D from the B-D lines and the capsule pointing). Held-out clean width, 1e-3: A 21.2 / 19.6 -> 15.7 / 16.5, C 24.3 / 28.7 -> 18.2 / 22.0, D 34.8 / 33.9 -> 25.6 / 26.0. At the noise of a track (learning curve). Candidate: `reaudit/tests/W2a_map_r3/angle_map_r3_candidate.json`, `apply_r3.py` | decide (Dylan): wire into `map_view.py` as r3 |
| 6.2 | D efficiency hole at x_local 0-60 mm, whole campaign; five channels hotter from run_133 (pedestal of 4 August) | T7, W2b | fiducial exclusion in D efficiency and shape analyses, or an efficiency that changes at run_133. No new mask | todo (table) |
| 6.3 | B noisy box (x -52..-44, y -200..-160 mm) through run_132 | T7, W2b | exclude it for B as a position point through run_132 | todo (table) |
| 6.4 | Decided table-level rules that are still off | T1 | charge rule q >= 600 (`LOW_CHARGE_RULES`), late quality reference (4.6), matching score (4.1), per-track wall time | todo (table) |
| 6.5 | D oscillating channels | W2b | flag windows dominated by oscillating channels (before 30 ms) on the pass-1 bundle, instead of the hot list | todo (table), after the `osc` single configuration |
| 6.6 | Before the test campaign is submitted | T1, T2 | seed rank by significance and cap after the veto (4.4); `--ringing` writes the waveform flag; profiles in every configuration; a configuration with the n_TOF x-y tolerance in the bundles; D on run_104; 2-3 beam-off cosmic sub-runs; D single switches before `all`; `tomorrow` for D; redeploy the code; about 10,000-12,000 core-hours | todo (decide with Dylan) |
| 6.7 | No scale correction for D | W2b, W3 | the beam pointing scale (A 0.990, C 0.965, D 0.946) is consistent with an estimator effect (Geant4 truth 0.92-0.99); the B-D cosmic lines give D 0.997 +- 0.004; D's scale does not follow v(t) (0.04 +- 0.08) | done (no change) |
| 6.8 | Per-track timing | T9 | a timing correction gains 0.4-3.3 % held out; x and y errors of a track are uncorrelated. No correction, no joint x-y fit | done (no change) |
| 6.9 | Drift-time maps | T4 | stable in A, B, C, D; a gap map on A only; not a substitute for an angle truth | done (study) |
| 6.10 | Geant4 digitisation side findings | W3 | pass-1 reco of the digitised events reads 3.4 % (nose) and 9 % (EAR2) steeper than the truth line; the y view of C in the EAR2 product is uncorrelated with the truth | open |
| 6.11 | The three n_TOF candidates of the bench audit (`rerun/paper1_gap/audit/`, HANDOFF item 3) | bench audit, W2a, T4 | drift-time correction for D: −13 % in the z view, against −28 % in both views for the spline map; bench fine map of A with no dish: A x 17.6 %, A y 2-9 %, against −35 / −24 % for the spline map; gap map into the v(t) scale: the drift time correlates with the cosmic scale at only −0.3 on A and C and is not a gap map on C and D. The spline map (6.1) covers all three | done (no separate item); the drift-time route stays a reference-free check |
