# v(t) within a row: det6 / det7, 6-26 night (HANDOFF 2026-10-30 step 3, GC)

Generated report: `report.html` (figures in `figures/`; numbers in `results/summary.json`). Copy with the corrected
tables: `/eos/user/d/dneff/x17rerun/vt_correction/`.

## Verdict
**The drift velocity moves within these rows, and it is the gas, not the chamber.** The correction is a table-level
v → v·r(t) in the angle mapping, measured with the same M3-ladder estimator that gives every row its kw. It is applied
to the pass-1 tables without refitting.

- **v rises through the night; it does not fall.** The handoff's "falls ~35 %" read the rows in the wrong order. In
  clock order: short_run 01:00, longer_run 02:00, the 530→505 V rows 04:00–06:00, the golden long_run 06:00–10:20, the
  HV scan 10:30–15:10.
- **det6:** 17.3 → 31.5 µm/ns (+82 %) from 01:15 to a knee at ~10:30, then flat. Inside the rows: short_run +6 %,
  longer_run +15 %, **golden long_run +20 %** (24.2 → 29.0).
- **det7 drifts the same way:** 31.5 → 39.1 µm/ns (+24 %), with the knee at the same clock time. Inside the rows:
  +2.5 / +6 / +4.4 %.
- **Same time-shape.** det6's deficit 1 − v/v_plateau is **2.80 ± 0.07 × det7's** at every time before the scan
  (r = 0.95, χ² 32/25, 26 blocks). One cause common to both chambers, with det6 seeing ~3× more of it: the gas.
- **The ladder is v, not gain.** On the 8 plateau rows of each chamber (charge varying ×55 on det6, ×20 on det7), v is
  flat to 1.0 % / 0.9 % rms, with a slope of +0.9 / −1.2 % per decade of charge. The 530→505 V rows also sit on the
  time curve.
- **Not attachment.** det6 had an early electronegative component: at fixed HV its gain rose ×25 and its charge-arrival
  shape u50/u_end went from 0.23 to 0.49, both settled by ~4 h. det7 had neither. Yet v kept rising in both until
  ~10.5 h, +20 % on det6 between 4 and 10 h with u50/u_end flat.

## Numbers (applied model: `night`, built on all events)
| row | r(t) range | max Δθ at 30° | max Δz at the cathode | mid-gap shift p95 | angle s68 x / y (°), before → after |
|---|---|---|---|---|---|
| det6 short_run | 0.948–1.042 | 1.28° | 1.5 mm | 0.14 mm | 2.024→1.975 / 2.879→2.886 |
| det6 longer_run | 0.923–1.075 | 1.87° | 2.2 mm | 0.23 mm | 1.682→1.631 / 1.976→1.953 |
| det6 long_run (golden) | 0.910–1.086 | 2.18° | 2.5 mm | 0.25 mm | 1.504→1.462 / 1.680→1.649 |
| det7 short_run | 0.979–1.020 | 0.52° | 0.6 mm | 0.07 mm | 2.210→2.188 / 1.411→1.417 |
| det7 longer_run | 0.968–1.029 | 0.78° | 0.9 mm | 0.12 mm | 2.456→2.442 / 1.540→1.528 |
| det7 long_run (golden) | 0.971–1.023 | 0.71° | 0.8 mm | 0.10 mm | 2.502→2.486 / 1.548→1.539 |

**Held-out test.** Each model is built on the even event_id half and scored on the odd half by the residual scale of
that half's own time blocks. Summed block χ² over the overnight rows, both views, ~40 blocks:

| model | χ² |
|---|---|
| one v per row (pass 1) | 322 |
| a line per row | 26.6 |
| piecewise blocks | 33.4 |
| one exponential per chamber (`night`) | **24.3** |

So the three time models are equivalent, and one v per row is not. On det6 long_run the block rms goes from 5.6 / 5.8 %
to 0.9 / 1.4 % (x / y).

## Method
1. Clock: the acq_tag stem (minute) + the DREAM timestamp × 10 ns, from the pass-1 event tables (`vt_common.clock_h`).
2. Estimator (`vt_common.ladder`): the one in `rerun/w12/set_w0kw.py`, on the row's own 03_angles `per_event.npz`, after
   the tan-form bow, t′ = (tan_ref + a)·S. Over the whole row it reproduces the applied kw to ≤ 1σ on all 48 rows.
   Per block of ~300 tracks in the kw window, it gives v_lad(t) = v_drift × kw(t) (`vt_measure.py` →
   `results/blocks.json`).
3. Correction (`vt_apply.py`): tan = (w − w0)/(kw·v·r(t)·S) − a. This is `wft.reco.v_rel_t`'s `v_map['t']`, written
   per row as `vt_map.json`. The mid-gap position p0 + 15 mm·tan and the depth z = v·r·u follow from it. w does not
   depend on v (GC), so nothing is refitted.
4. Outputs per row in `/eos/user/d/dneff/x17rerun/vt_correction/<run>/<subrun>/<det>/`:
   - `events_vt.parquet`: event_id, acq_tag, t_h, v_rel, x/y_tan_vt, x/y_theta_deg_vt, x/y_mid_vt, z_end_vt. Join on
     event_id, which is unique within these rows.
   - `vt_map.json`.

Run: `./fetch.sh <dir>` (~23 MB through `ssh lxplus`), then `python vt_measure.py <dir>`,
`python vt_apply.py <dir> --out <out>` and `python make_report.py`. All of it takes ~2 min on the laptop.

## Limitations (no new data can be taken)
- **What in the gas changed is not identified.** Water drying out would raise v in both chambers. det6's early
  attachment and gain loss is an extra electronegative component that det7 did not see. Neither the bench gas order and
  flow on 6-26 nor the composition is recorded; the bench composition is known only from a fit (0.95 % H₂O, no O₂).
- **The plateaus differ:** det6 31.5 against det7 39.1 µm/ns at the same 700 V drift. That is a chamber-side
  difference, not addressed here.
- **The column-end depth does not confirm the ladder.** In det6's long_run, z_end = v·u_end falls slightly with time
  under one v and rises after the correction. The column duration is gain- and attachment-confounded: GC found the
  ladder moving 1.68× faster than 1/T on det6. The absolute depth scale carries this disagreement.
- **The `night` exponential is an interpolator, not a gas model.** It misses det6's knee (χ² 82/38) and its τ is not
  a time constant. Inside each row it is as good as a line.
- **The resolution gain is small** (det6 long_run x 1.50 → 1.46°). The correction is mainly to the angle *scale* at
  the row ends, up to ~2° at 30° on det6.
- **Short rows.** short_run has one block per half, so its held-out score is empty. The 20-min HV-scan rows get the
  night curve; their within-row change is ≤ 1–2 %.
- **Scope.** The other GC candidates are not done: det4 golden 6-24 (~7 %), det4 6-23, det2 6-23 long_run.
