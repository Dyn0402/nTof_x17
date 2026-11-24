# n_TOF tracking re-audit, 2026-11-24

Report: [`report.html`](report.html), built by `make_report.py`. The figures are git-ignored PNGs in `figures/`. A copy
of the report with its figures is on EOS, `/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/reports/reaudit_2026-11-24/`.

Dylan asked for a second audit of the n_TOF tracking before the phase-4 test campaign runs, and for the implications of
the bench drift-geometry result (`rerun/paper1_gap/audit/`) for n_TOF. Ten tests were run on the existing data. The
baseline is the rebuilt analysis view `stage3_r2_final`.

## Results in one place

- **`ref` reproduces pass 1.** Code `e86028b5` with every switch off, 8 acquisitions of A and C (condor 4729151): 20,251
  tracks in both sets, all gated tracks matched, largest difference 1.3e-6 mm.
- **The phase-4 chain was broken and is fixed.** `ntof_stage3.py --allow-missing`, `phase4/stage3_chain.sh`,
  `phase4/compare.py`. `deflect.py` marked the first event of each new reader as corrected; fixed.
- **The campaign needs changes before it is submitted**: section 3 of the report (seed rank, D job flavour, cost of
  10,000 to 12,000 core-hours, cosmic sub-runs, items in no configuration).
- **A spline angle map (r3 candidate) beats the current map on held-out cosmic lines**: A 21.2 / 19.6 -> 15.7 / 16.5,
  C 24.3 / 28.7 -> 18.2 / 22.0, D 34.8 / 33.9 -> 25.6 / 26.0 (clean width, 1e-3 in tan, x / y view). It is at the noise
  of a single track. Not applied: `tests/W2a_map_r3/angle_map_r3_candidate.json` and `apply_r3.py`.
- **The drift geometry at n_TOF**: the tilt field is curl-free (one potential costs nothing); the scale is common to
  the two views on C, not on A; drift-time maps are stable in all four chambers but are not a gap map on C and D;
  capsule pointing confirms the map (r 0.93 A, 0.70 C, 0.94 D) and adds information only for D.
- **Per-track timing** gains 0.4 to 3.3 %; the x and y errors of a track are uncorrelated. No timing correction, no
  joint x-y fit.
- **Chamber D**: efficiency hole at x_local 0 to 60 mm for the whole campaign; five channels hotter from run_133 (new
  pedestal of 4 August); angle scale does not follow v(t); adding the oscillating channels to the hot list loses real
  tracks. **Chamber B**: noisy box through run_132.
- **The beam pointing scale (0.946 to 0.990) is consistent with an estimator effect** (Geant4 truth gives 0.92 to 0.99).
  No scale correction for D.

## Tests

`tests/COMMON.md` is the brief common to all tests. Each `tests/<id>/` holds the scripts and the small result tables;
the caches were under `$WORK_DATA/claude-jobs/eb8200b3/<id>/` (regenerable from `stage3_r2_final`).

| id | question |
|---|---|
| `T1_phase4_review` | review of the phase-4 code and campaign (`q1_code_findings.md`, `completeness_table.md`) |
| `T2_ref_compare` | `ref` against pass 1; `all` against pass 1 (`results/`) |
| `T4_time_maps` | drift-time maps of A, B, C, D |
| `T5_pointing_map` | angle map from capsule pointing |
| `T6_surface_map` | potential + common scale against free cells |
| `T7_stability` | the sample sub-run by sub-run (`eras.csv`, `outliers.csv`) |
| `T9_track_timing` | per-track timing against the angle error |
| `W2a_map_r3` | the spline map candidate (`r3lib.py`, `apply_r3.py`, `r3_validation.csv`) |
| `W2b_chamber_D` | chamber D and B items (`results.txt`) |
| `W3_g4_scale` | the pointing scale estimator on Geant4 truth (`results.txt`) |

Rebuild the report: `.venv/bin/python rerun/ntof/improve/reaudit/make_report.py`.
