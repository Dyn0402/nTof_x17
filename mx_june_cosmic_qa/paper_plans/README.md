# Paper plans — execution guide

Plans for the analyses still missing for the June det3 micro-TPC paper (see
`../PAPER_STATUS.md` for the full audit). Each plan is self-contained: exact input
paths, verified column names, formulas, acceptance checks, and known traps. They are
written to be executable without re-deriving context — read the plan top to bottom
before writing any code.

| plan | what | priority | est. effort |
|---|---|---|---|
| `PLAN_37_m3_pointing_deconvolution.md` | M3 telescope pointing error at the detector plane + intrinsic-resolution deconvolution | **1 (do first — changes every quoted σ)** | ~half day |
| `PLAN_38_xy_charge_balance.md` | Charge balance between X and Y strip layers | 2 | ~half day |
| `PLAN_39_spark_deadtime.md` | Dead time per spark → efficiency-ceiling model | 3 | ~half day |
| `PLAN_40_gap_skeptic_tests.md` | Three hardening tests for the drift-gap/attachment story | 4 | ~1 day (3 sub-tasks) |
| `PLAN_41_publication_figures.md` | Publication-figure remakes + housekeeping | 5 (after 37–40) | ~1 day |
| `PLAN_42_time_resolution.md` | micro-TPC time resolution — detector 33 ns (≈1 mm drift) + absolute 37.7 ns (scint trigger + applied ftst); LaTeX report | **DONE 7-11** | done |

## Shared conventions (read once, applies to every plan)

- **Python**: `../.venv/bin/python` (repo-root `.venv`; NOT `../venv`). Run all scripts
  FROM `mx_june_cosmic_qa/`: `../.venv/bin/python <script>.py <run_key> [--veto=50]`.
- **Run keys** live in `qa_config.py` (`RUNS` dict). Headline det3 run key: `sat_det3`
  (= `mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V`, det3 top slot,
  FEU 7=X / 8=Y, drift 1000 V, resist 490 V). det2 validation key: `o22_long_det2`.
- **Output roots** (per run/det):
  `~/x17/cosmic_bench/Analysis/<run>/<subrun>/<det>/alignment_tpc_veto50/` — put each
  plan's outputs in a new subdirectory there (pattern: how scripts 31–36 use
  `CFG.out_dir(...)` / the alignment dir). Never write into `*_prev2_backup/`.
- **New scripts** continue the numbered series (37, 38, 39, 40) in `mx_june_cosmic_qa/`,
  same style as 31–36: docstring header stating purpose + inputs + outputs, `qa_config`
  for paths, matplotlib figures saved at dpi≥150, plus a small CSV/JSON of the numbers.
- **M3 rays**: load via `cosmic_bench_analysis/M3RefTracking.py`, passing
  `chi2_cut=qa_config.M3_CHI2_CUT, min_nclus=qa_config.M3_MIN_NCLUS` explicitly (do not
  rely on the class defaults, which are historical/pre-June and shared with other
  packages). The recipe (chi2<1.0 & NClus=4, since 2026-07-14) is the standard everywhere
  -- see `../det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md`.
- **Spark veto**: analyses use the veto50 cache (`cache/event_results_veto50.pkl`) or
  a >50-strips/event multiplicity veto. Sub-veto discharges (30–50 strips) survive it.
- **Never assume the FEU↔plane map** — read it from `qa_config.py` per run.
- **Hits have NO common-noise subtraction**; per-strip `amplitude` is usable (pulse max
  above pedestal-ish baseline) but FEU 6/8 have large common mode in raw *waveforms*.
  Hits-level amplitude sums are fine; waveform-level work needs the pedestal + per-chip
  median subtraction pattern from `24_waveform_investigation.py`.
- **Geometry constants**: pitch 0.78 mm, drift gap 30 mm, E = HV/3.0 cm, DREAM 32×60 ns
  samples, ADC 12-bit (saturation at 4095, baseline ~220).
- **Sanity anchor numbers** (det3 sat run, M3 v2): eff 92.9 ± 0.2 % (5 mm fid),
  σx/σy = 0.76/0.83 mm, v_geom(1000 V) = 34.30 ± 0.28 µm/ns, T_sat 691 ns,
  λ_att 15.4 mm, ~15.3k matched events. If your event counts or baseline numbers are
  wildly off these, stop and find out why before proceeding.
- **When done**: append a dated results section to the bottom of the plan file itself
  (numbers + output paths), and add a one-line entry to `DET3_WEEKEND_ANALYSIS.md`'s
  open-follow-ups section marking it done.
