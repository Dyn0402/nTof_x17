# Retired 2026-11-20: the n_TOF track products before pass 1

n_TOF pass 1 (`NTOF_PASS_PLAN_2026-11-02.md`; stage 3 final on EOS `july_beam/Analysis_pass1/stage3/` 2026-11-20,
ledger clean 15,748 / 15,748) replaces every earlier n_TOF reconstruction. The sets below are **retired**: no new
analysis reads them. They are kept, unchanged, as references for the comparisons in `PASS1_DOWNSTREAM.md` and for
the history (`rerun/history/LESSONS.md`). Nothing was deleted.

| set | what it was | where (desktop unless noted) | replaced by |
|---|---|---|---|
| `stage3_fullpass` | production stage 3 (Sep full pass, v 42.6 prior, seeder min 5, per-run k) | `/media/dylan/data/x17/sept26_prelim/stage3_fullpass` (24 GB); pre-y-fix copy on EOS `desktop_scratch/x17_local/yfix_work/` | `stage3_pass1` |
| `reco_fullpass` | its stage 2, merged per sub-run | `sept26_prelim/reco_fullpass` (38 GB) | `reco_pass1` |
| `stage3_is2_v2`, `fullpass_is2_v2` | the is2_v2 re-pass (A, C; in-situ bundles, two-sided, two-track off) and its stage 2 | `sept26_prelim/stage3_is2_v2` (14 GB), `sept26_prelim/fullpass_is2_v2` (14 GB); stage 2 on EOS `analysis/sept26_fullpass_is2_v2` | `stage3_pass1` / `reco_pass1` |
| `stage3_is2_v2_cf` | is2_v2 at the capsule-focus k (1.12 × is2) | `~/x17_local/stage3_is2_v2_cf` (14 GB) | `stage3_pass1` (v(t) angles, no k) |
| `stage3_campaign`, `stage3_is2_v1_pilot` | early campaign stage 3; the is2_v1 pilot (still y-mirrored) | `sept26_prelim/` | — |
| `kcal`, `kcal_is2_v1`, `kcal_is2_v2`, `kcal_is2_v2_cf` | per-run angle-scale k from the capsule band | `sept26_prelim/` | nothing: pass 1 has k = 1, the angle comes from v(t) with κ |
| is2 bundles (`is2_A`, `is2_C`, `r06k7`, …) | the in-situ bundles of the is2 re-pass | `~/scratch/ntof_insitu/bundles` | frozen pass-1 bundles, EOS `Analysis_pass1/bundles/` (`freeze/FROZEN_2026-11-07.md`) |
| `ntof_cosmics/results/tracking/pooled/resolution.csv` | single-track angle resolution against the A–C line, is2 era | repo | `rerun/ntof/stage3/resolution/resolution_pass1.csv` (`resolution_line.py`) |
| `rerun/vt_all` n_TOF part (`vt_ntof.py`) | v(t) feasibility from per-sub-run k on is2_v2 | repo | P7a v(t): `rerun/ntof/vt/`, `tan_vt` in stage 3 |
| g4_digi libraries `g4_{A,C}_is2_full`, `g4_{A,C}_ear2_*` | Geant4 through the is2 reconstruction (copy-kernel signal model) | `/media/dylan/data/x17/ntof_cosmics/g4_digi/` | `$WORK_DATA/ntof_pass1/g4_digi/g4_{A,C}_<tag>_pass1` (`run_pass1.sh`, RC-kernel generator) |

How code chooses: `sept26_prelim_analysis/paths.py` `STAGE3` / `RECO` (default `stage3_pass1` / `reco_pass1`;
`X17_STAGE3=stage3_fullpass X17_RECO=reco_fullpass` for the retired production chain). The studies with an
explicit chain switch (`same_chamber_compare`, `same_chamber_physics`, `capsule_tof`, `focus_split`) keep the retired
chains as named options for comparison.

Studies of the retired chains themselves are left as they are and are not migrated: `ntof_cosmics/pilot_compare.py`,
`make_pilot_report.py`, `make_fullpass_report.py`, `repass_readiness.py`, `seed_beam_test.py`, `yield_gate.py`,
`angle_response.py`, `inbeam_through_goers.py`, `late_trigger_clock.py`, `ac_tof.py`, `g4_angle/capsule_estimator.py`,
`sept26_prelim_analysis/qsum_runaway.py`, the two-track bench (`intra_bench`, `two_track_synth`, `condor/two_track/`),
`regen_stage3`, `campaign_tracks`, `merge_fullpass`. They document how the retired chain behaved.

Deleting any of the sets above is Dylan's decision (they total ~140 GB on `/media/dylan/data` and `~`).
