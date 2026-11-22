# Fixing the n_TOF tracking audit: the roadmap (started 2026-11-22)

The tracking audit of 2026-11-21 found three problems in the track sample, several code errors and a set of estimator
biases. The findings are 11 to 23 in `README.md`, the report is `audit_report/report.html`, and the three code reviews
are kept as they were written in `audit_report/reviews/`. This file is the plan for fixing all of them, in the order we
do them. Each item names the finding it comes from (README number, or the review and its item number), what we change,
and how we check it. The status column is updated as we go.

All the work uses the data we already have. Most fixes are at table level: we rebuild stage 3 and the analysis view
from the existing stage-2 reconstruction outputs. Only phase 4 touches the waveform reconstruction itself.

Status: `done` = committed; `wip` = in progress; `todo`; `decide` = waits for Dylan.

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
| 2.1 | Sub-runs with two slim files read from the first only | 12 | the four readers take every file; regenerate the slim export, trigtime and stage 1 for the 77 sub-runs in `audit_report/slim_files.csv` | wip (mapping the chain) |
| 2.2 | Stage 3 rebuilt with the context columns | 12 | `ntof_stage3.py` on the regenerated products; check that untimed production tracks fall from 14.7 % to ~0 | todo |
| 2.3 | Bow-partner tangents in pass 1 | 21 | recompute each track's bow tangent from its own (w, p0) pair at stage 3, with the bundle's bow block | todo |
| 2.4 | `dca_axis_mm`, `in_bore` about the nominal axis | 22, reco review 2 | add `dca_source_mm` about the measured source; keep the axis columns, documented as such | todo |
| 2.5 | Bow mesh height not applied to positions | 22, stage-3 review 4 | test p0 − tan·h on the late capsule pointing and on the A–C lines; apply only if it improves both | todo |
| 2.6 | κ rescale also scales the bow tilt term | stage-3 review 6 | rescale only the speed part of the tangent (small: 0.002 rms on C, D) | todo |
| 2.7 | Map refit on real cosmics | 13 | `line_residuals build`, `angle_map fit`, `bd_map` on the registry's cosmic sub-runs, then `map_view` | todo |
| 2.8 | Post-flash ringing fits pass the gate | 14 | `low_charge` = min(x_q_sum, y_q_sum) < 600 ADC; taken out of `gated` like `censored_window` (pending 2.10) | decide |
| 2.9 | D contamination; C windows partly censored | 15, reco review 9 | a D rule (width, censored fraction, χ² against the late reference) tuned on D's wall-confirmed late tracks, as the censored rule was; the same test on C | todo |
| 2.10 | Where the new cuts live | — | flags in a `quality_join.py` step after `censor_join`, with the pre-cut `gated` kept | decide |
| 2.11 | The new stage 3 on EOS and in the ledger | — | upload, MD5SUMS, ledger rows; retire record for the old set | decide |

## Phase 3. Estimators and analysis scripts

| # | item | from | change | status |
|---|---|---|---|---|
| 3.1 | `crossing` fails abruptly with background | 17, imaging review 2 | a second fit within 100 mm of the first crossing; injection test as in the review | todo |
| 3.2 | D z from `capsule_tof tilt` is not a measurement | 16, imaging review 1 | D sample = late, clean (2.9), wall-confirmed; quote with the band-shape caveats of imaging review 6 | todo |
| 3.3 | The −1.7 mm is quoted downstream | imaging review 1 | `compare_chains.py`, `PASS1_DOWNSTREAM.md` take the new D z | todo |
| 3.4 | `campaign_imaging` biases | 17, imaging review 3 | read the view (`gated`, map, censored cut), ToF ≥ 20 ms, window centred on the source, hot-strip drop on every run | todo |
| 3.5 | Focus split about the nominal axis | 17, imaging review 4 | `focus_split.py`, `capsule_focus.py` about the measured source | todo |
| 3.6 | Pose fit: stale `Z_BD`, library-size term | imaging review 7, 9 | `Z_BD` from the pass-1 D z; `SIG_X_SYS` adds the Geant4 library's statistical error per crossing | todo |
| 3.7 | `track_quality` null keeps part of the pointing; stale cache | 17, imaging review 8, 9 | full shuffle null; cache keyed on the view's md5 | todo |
| 3.8 | `capsule_point` axis from the is2 era | imaging review 9 | measured pass-1 source | todo |
| 3.9 | Same-chamber rate fit from 12 ms | 14 | `same_chamber_compare` with `FIT_MS` 12 on the new `gated`; check the pair fraction is flat from 12 ms | todo |
| 3.10 | Scint stack and calorimetry on the corrected y sign | 20 | rerun `ntof_scint_stack` extract + ana, `ntof_calorimetry/wall_plastic` | todo |
| 3.11 | y crossing and the trigger plastic clip | imaging review 9 | toy correction or a stated limitation | todo |

## Phase 4. Reconstruction-level fixes (wft/, the beam driver)

These change how tracks are found and fitted, so they act only through a new reconstruction of the n_TOF data. We fix
the code, test each change on a few sub-runs against pass 1, and then decide whether a pass 2 is worth it.

| # | item | from | change | status |
|---|---|---|---|---|
| 4.1 | x–y coincidence uses the bench `tc_tol` | reco review 4 | n_TOF tolerance from the late x–y column-time difference | decide |
| 4.2 | Out-of-time tracks are biased, not rejected | reco review 6 | flag tracks with t0 at the ±180 ns box edge | decide |
| 4.3 | Two-track children ignore the t0 pin; `split_child` not in stage 3 | 22, reco review 8 | pin the children; carry `split_child` | decide |
| 4.4 | Seeder keeps the five widest clusters | reco review 10 | rank by significance | decide |
| 4.5 | Head-on rescue not wired into the beam driver | 22, reco review 11 | wire `headon_seeds` / `_rescue_` windows into `wft_beam` | decide |
| 4.6 | `quality_ok` reference median set by the ringing | 14, reco review 3 | reference from late tracks | decide |
| 4.7 | Candidates carry no `bow_h` | stage-3 review 4 | write it | decide |
| 4.8 | Smaller | reco review 12 | events table pairs the plane winners; `tan_err` at its floor; `dt_xy` bench key | decide |

## Phase 5. Rerun the downstream and write it up

| # | item | status |
|---|---|---|
| 5.1 | `PASS1_DOWNSTREAM.sh` on the new view (~6 h), `compare_chains`, the before/after page | todo |
| 5.2 | README, `PASS1_DOWNSTREAM.md`, `docs/PARAMETERS.md`, HANDOFF, the analysis board | todo |
