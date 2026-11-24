# Completeness of the phase-4 plan (T1_phase4_review, 2026-11-24)

Every actionable item of the three verbatim reviews (`rerun/ntof/improve/audit_report/reviews/`), of findings 11-23
(`rerun/ntof/improve/README.md`) and of the recommendations in `t0_study/README.md`, `xy_match/README.md` and
`rerun/ntof/ringing/README.md`, mapped to the roadmap (`AUDIT_FIX_ROADMAP.md`) and to the phase-4 test campaign
(`phase4/make_jobs.py`: configurations ref, all, floor30, t0prior, seedrank, headon, ringing, osc; sub-runs run_145
stat090_0000-0002, run_104 stat090_0003, run_139 stat090_0003; D not run on run_104).

"Code" = what I found in the code (grep/read, 2026-11-24, HEAD feeeaaf9). "Campaign" = whether a campaign configuration
exercises it. Level: T = table-level (no re-reconstruction needed), R = reconstruction-level.

Abbreviations: RR = reco_gating.md review, S3 = stage3_tables.md review, IM = imaging_estimators.md review,
F = README finding, T0 = t0_study/README.md, XY = xy_match/README.md, RG = ringing/README.md.

## A. Reconstruction-level items

| source | item | roadmap | status (roadmap / code) | campaign | level |
|---|---|---|---|---|---|
| RR 1 (l.20-36), F 14 | gate accepts post-flash ringing; no cost / Δχ² / charge-balance threshold rejects a pair | 2.8 (table cut) + 4.10 (waveform correction) | 2.8 decided 11-23: q >= 600 AND own-window waveform flag before 8 ms, flag alone after. Code: `quality_join.LOW_CHARGE_RULES = ('none',)` (quality_join.py:60), so `low_charge` is all False in `stage3_r2_final` (meta confirms `low_charge (none)`). The decided rule is NOT implemented | ringing config tests the waveform correction only; the charge cut is not applied by `stage3_chain.sh` (quality_join default `--low-charge none`) | T (charge part) / R or ~300-job waveform pass (flag part) |
| RG "Recommendation for a pass 2" (l.364-369) | per-track waveform flag column (own window, >= 1 channel > 5σ after `deflect`, both views) | 2.8 (needs the flag) | NOT in code: `ntof_reco --ringing deflect_hits` installs the correction and the hits filter (ntof_reco.py:98-103, deflect.py `install_hits_filter`) but writes no per-track flag | no (no configuration produces the flag, so 2.8's decided rule cannot be evaluated on the campaign) | R (or a separate waveform pass) |
| RG l.364-367 | A/C: `deflect` v1, gated, mode `deflect_hits` | 4.10 | code in | yes (`ringing`, `all`; A/C only) | R |
| RG l.370-373 | D: no correction, no waveform test | 4.10 | code refuses D (ntof_reco.py:100-101) | yes (D excluded from `ringing`) | R |
| RG l.372-374, l.378 | D: add 19 x / 42 y oscillating channels to the hot list, measure on a re-reco | 4.9 | bundles made (`make_osc_bundles.py`) | yes (`osc`, `all`), but only run_139 and run_145 (D not run on run_104); the channel list was derived from the same three runs | R |
| RR 3 (l.42-45), F 14, RG l.382-385 | `quality_ok` reference median set by the ringing | 4.6 | code in (`build_tracks.QUALITY_REF_LATE_MS`, build_tracks.py:414-428), stage-3 only | NOT in any configuration: no cfg sets `X17_QUALITY_REF_LATE_MS`; `stage3_chain.sh` only passes it "if set". Since it is stage-3 only it can be built from `ref` outputs | T (stage 3 from existing stage 2) |
| RR 4 (l.47-49), XY rec. 3 (l.132-133), F 23 | x-y coincidence uses the bench `tc_tol` (197/456/726 ns); use `TC_TOL_NTOF` (151/296/535) in the bundles' `xy_pairing` in a pass 2 | 4.1 ("n_TOF tc_tol") | table flag `xy_time_ok` exists (quality_join.py:65, 191). Bundles still carry the bench value (local `bundles/pf/*`: A 197.37, C 456.41, D 725.72); `ntof_bundle.py` and `make_osc_bundles.py` do not change it | NO | R (reco gate) / T as a flag only |
| RR 4 (l.50), F 23 | upper column bound 1.6 T (1243/1708 ns) is beyond the 1200 ns grid and never rejects | NONE (4.1 does not mention it) | `wft/reco.py:647` U_MAX_FRAC = 1.60 unchanged | NO | R (`plausible` is in the reco gate); a table-level cut on `x/y_q_uend` is possible |
| T0 "Possible mitigations" (l.246-247) | upper column-end bound tied to the gap | NONE | not in code | NO | R or T (as above) |
| RR 4 (l.52) | pairing cost has three correlated timing terms | 4.1 (study: keep pass-1 matching) | decided: keep | n/a | — |
| XY rec. 2 (l.128-131) | per-track matching score (n_TOF likelihood + gain of best alternative) as a stage-3 column | 4.1 | "score column todo"; not in quality_join / build_tracks | n/a | T |
| XY rec. 4 (l.134-137) | store each candidate's 20-bin depth profile | 4.11 | code in (`reco.STORE_PROFILES`, reco.py:813, 1956) | partly: only in `all` (no standalone cfg); no table-level `profile_distance` matching script consumes it | R |
| T0 l.173-174 | second cause of lost second tracks: both views fitted, x-y association builds only one track, the leftover combination fails the coincidence | NONE (4.12 covers only the seeder cause) | not in code | NO (and the campaign has no injection, so P(2\|2) cannot be measured by it) | R (select_tracks), partly T (replayable from candidates table, XY l.8, 30) |
| T0 l.168-172, l.176-184 | plane-wide seeding floor loses the second track; 30 mm local floor | 4.12 | code in (`seed.SIG_FLOOR_LOCAL_MM`) | yes (`floor30`, `all`): single-track and fake effect. The pair gain was measured by injection on `ref`-equivalent code only, never with `all` (seedrank + cap 5 + floor interact) | R |
| RR 10 (l.79-80), F 23 | seeder keeps the five widest clusters | 4.4 | code in (`seed.SEED_RANK=charge`, seed.py:38, 284-288) | yes (`seedrank`, `all`) | R |
| RR 8 (l.71-73), F 22, T0 l.36-41 | two-track children ignore the t0 pin | 4.3 | code in (`reco.TWO_TRACK_T0_PRIOR`, reco.py:809, 1163, 2151) | yes (`t0prior`, `all`) | R |
| RR 8, F 23 | `split_child` not in stage 3 | 4.3 | done (build_tracks.py:232-233) | n/a | T |
| RR 11 (l.82-84), F 22 | head-on rescue not wired into the beam driver | 4.5 | code in (`--headon-rescue`, wft_beam.py:323-326) | yes (`headon`, `all`) | R |
| RR 6 (l.60-65), F 23, T0 | out-of-time tracks biased, not rejected; wider t0 box | 4.2 | study done; ±360 ns box gains little; `t0_railed` flag only (quality_join.py:9-13, "decision pending") | no (by decision) | R (box) / T (flag) |
| RR 7 (l.67-69), S3 4, F 22 | candidates carry no `bow_h`; stage 3 does not apply it | 4.7 / 2.5 | done (column); 2.5 tested, not applied | yes, as a column in every cfg (always on) | R (column) / T (applying it) |
| RR 12 (l.87), F 23 | events table pairs the plane winners (not track 0 in 24/31 %) | 4.8 | todo | NO | R (stage-2 output); T if stage 3 does not read the events table |
| RR 12 (l.88), F 23 | `tan_err` at its floor, no per-track information | 4.8 | todo (`tan_err = hypot(ew·1e3/v, tan_floor)`, wft/reco.py:607) | NO | R (or T, recomputable from `ew` if stored) |
| RR 12 (l.89) | `cal.dt_xy.get(..., -18.8)` reads bench keys | 4.8 | todo (wft/reco.py:1720, 1988 still bench default); inert today | NO | R (inert) |
| S3 3 (l.45-48), F 22 | D (and 2-5 % of C) in early runs: the raw 0.6 reco slope cut maps to 0.48-0.50 for 25-35 % of D's tracks (runs 79, 104) | NONE (no roadmap item) | not addressed; bundles still `tan_max = 0.6` (raw) | NO, and untestable on the campaign: D is not run on run_104 and runs 139/145 are clean | R (raise `WFT_TAN_MAX`) or T (per-run tighter acceptance cut, e.g. \|tan\| < 0.48 on D early runs) |
| S3 "checked" (l.77), F 23 | D's fitted column starts ~55 ns late: 1.8 mm of depth or a 6 % error in D's v_t | NONE | not addressed | NO | T (depth reference / v_t, if decided) — cannot be separated with existing data per the review |
| RR 9 (l.75-77), F 23 | C x windows 10-30 % censored point at 0.50 vs 0.58 | 2.9 ("C partial censoring: not done") | not done (censor_join removes fully censored windows only; `x_cens_frac` is carried) | NO | T |
| T0 l.235-238 | D not done in the t0 study (needs `d_faint` and the osc channels first) | NONE | open | NO | T / injection |
| T0 l.231 | injection test on one sub-run (run_145 0000), A and C only | NONE | stated limitation | NO | injection (R-like, desktop) |

## B. Table-level items (no re-reconstruction)

| source | item | roadmap | status | campaign | level |
|---|---|---|---|---|---|
| S3 1, F 20 | view `tan_raw_y` sign; scint stack mirrors y | 1.1, 3.10 | 1.1 done; 3.10 (rerun scint stack + calorimetry) todo | n/a | T |
| S3 2, F 12, F 13 | two-slim sub-runs; null time read as beam-off | 2.1, 2.2, 1.2 | done | n/a | T |
| S3 5, F 22 | stage-1 merge duplicates two rows | 1.6 | done | n/a | T |
| S3 6, F 23 | κ rescale scales the bow tilt term | 2.6 | done | n/a | T |
| S3 7, F 22/23 | re-run hazards (map twice, `vt_view --force`, residual-map fit) | 1.3, 1.4 | done | n/a | T |
| RR 2, F 22 | `dca_axis_mm` / `in_bore` about the nominal axis | 2.4 | done (`dca_source_mm`) | n/a | T |
| RR 5, F 21 | `apply_bow` partner bug | 1.5 (reco), 2.3 (table) | done | the reco fix is in the code all cfgs run, so `ref` on e86028b5 is NOT byte-identical to pass 1 for paired tracks (ringing README l.122-123: tangents differ on 6 % of candidates) | R (fixed) / T (patched) |
| F 15, RR 9 | D contamination | 2.9 | `d_faint` done; "demonstration" clean selection not adopted | osc tests the reco side | T |
| F 14 | same-chamber rate fit from 12 ms | 3.9 | todo | n/a | T |
| 2.10, 2.11 | where cuts live; stage 3 on EOS | 2.10, 2.11 | todo / held | n/a | T |
| 2.7 | map refit on real cosmics | 2.7 | roadmap says "todo (next)"; COMMON.md says `stage3_r2_final` already carries `angle_map_r2.json` (roadmap status appears stale) | stage3_chain holds the r2 map fixed | T |
| IM 1, F 16 | D z from `capsule_tof tilt` not a measurement | 3.2 | wip ("clean D rule (2.9) to be added") | n/a | T |
| IM 1, F 23 | −1.7 mm copied into `chains.csv` / `PASS1_DOWNSTREAM.md` | 3.3 | todo (5.1) | n/a | T |
| IM 2, F 17 | `crossing` fails with background | 3.1 | done | n/a | T |
| IM 3a-d, F 17 | `campaign_imaging` biases | 3.4 | done (3d hot strips run_145 only: not addressed, 0.14 mm) | n/a | T |
| IM 4, F 17 | focus split about the nominal axis | 3.5 | done | n/a | T |
| IM 5, F 11 | "45-58 % in focus" is not a purity | none needed (interpretation) | documented | n/a | — |
| IM 6, F 16/23 | D z caveats (band not straight, foot in dead region, inside/outside −5.2/−2.2) | NONE beyond 3.2 | open, stated | n/a | T |
| IM 7, F 17/23 | FOOT_SIM C; library-size term | 1.7, 3.6 | done | n/a | T |
| IM 8, IM 9 | track_quality null, stale cache, `Z_BD`, `capsule_point` axis | 3.7, 3.6, 3.8 | done | n/a | T |
| IM 9, F 23 | y crossing / trigger plastic clip | 3.11 | limitation | n/a | — |
| IM 9 | depth from band slopes diluted by background (not measured) | NONE | open | n/a | T |
| IM 9 | `capsule_focus` null binning (display only) | NONE | open, cosmetic | n/a | T |
| F 18 | source height is a flat line, not used in pose fits | NONE | observation | n/a | — |
| XY rec. 1 | keep the pass-1 matching | 4.1 | decided | n/a | — |
| XY l.114-122 | likelihood as a quality cut | none (not proposed) | — | n/a | — |
| T0 l.242-243 | pair-level \|Δtc\| < 300 ns cut | NONE | "not decided" | n/a | T |
| T0 l.244-245 | per-track wall time in the view (±1000 ns slim) | NONE | not done (`wall_time.py` is a study product) | n/a | T |
| T0 l.246 | per-track arrival time from the stored depth profile | 4.11 (enabler only) | not done | partly (`all` stores profiles) | T after a pass with profiles |
| T0 l.235-237 | rebuild the `look9` \|Δtc\| template from the injection response | NONE | not done (11-24 settled the fraction by look10/11, so possibly moot) | n/a | T |
| T0 l.181-183 vs roadmap 4.12 | do g4_digi pairs see the same seeder loss? | 4.12 text | roadmap says checked (P(2\|2) A 0.48, C 0.42); t0 README still says "We have not checked that yet" | n/a | — |
| RG l.376-381 | local excursions < 1.25 ms: cause open, no correction | none | stated limitation | n/a | — |
