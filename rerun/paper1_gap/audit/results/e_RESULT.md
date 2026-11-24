# Task E: shape of the drift lines from the waveforms (audit tag `e`, 2026-11-24)

Scripts (all in `rerun/paper1_gap/audit/`): `e_lib.py` (extraction, regressions), `e_01_extract.py`, `e_02_zglobal.py`,
`e_03_main.py` (slice fits), `e_04_report.py` (all numbers -> `results.json`), `e_05_headon.py`, `e_06_figs.py`, `e_07_tables.py`
(-> `tables.md`, every table quoted below). Figures: `fig_zk_calibration.png`, `fig_delta_profiles.png`, `fig_gradients.png`.
All in `$WORK/audit/e/`. Units of gradients: 1e-4 per 100 mm (dimensionless slope per 100 mm of position).

## Method as run
- Data: the four caches (run 1 headon+incl, run 2 headon+incl), merged by event id (4,219 + 12,748 events with waveforms) and matched to the
  pass-1 table by `event_id` (index). The match is exact: `tan_ref` and `ref_mesh` of the pickles equal the table's to 1e-16.
  Matched to the standard selection (`GM.load_table`): head-on 55.6 % / 53.8 %, inclined 65.0 % / 62.3 % of the cache (run 1 / run 2), the rest fails the selection.
- Per event, view, sample k: charge centroid of strips with W > 3 sigma_noise inside a window +-(3 + 6|tan|) mm around the predicted landing
  position (ref_mesh + tan z(t), z(t) from a first-pass calibration, iterated once); >= 2 strips. Time t = 60 ns k - t0_fit (the fit pins t0 = C - 10 ns ftst, so t is
  trigger-referenced; ftst makes the phase vary in 10 ns steps). 60 ns bins from -60 to 900 ns.
- Per region and time bin: robust line (3.5 sigma clip) `cen - ref_mesh = z_k tan + delta_k`. Rows are re-weighted so +tan and -tan have equal weight in
  every |tan| bin (balance): the intercept is then exactly free of any odd nonlinearity of the centroid in tan. (There is one: the centroid slope drops ~10 % for |tan| > 0.3, see below.)
- Fit delta_k = a + b z_k + q z_k^2 over the bins with t in 0-810 ns, **z reached: 2.8-22.8 mm (X), 3.1-23.5 mm (Y)**. The calibration z_k(t) is
  smooth and monotonic (`fig_zk_calibration.png`: z = 1.5 mm at t = -30 ns, 12 mm at 450 ns, 23 mm at 870 ns, flattening at the end where the charge pile-up at the cathode mixes depths).
  What is extrapolated: b is the slope at z = 0 (2.8-3.1 mm below the first bin); D = 2 g q = c_slope - h_slope assumes the quadratic holds to g = 28.2 mm (5-9 mm beyond the range);
  `s_far` (slope at the end of my range, no extrapolation) is also in `tables.md`; `s_mean` over the full column is b + q g (extrapolated), `sl` is the plain straight-line slope of delta(z) over my range (not extrapolated).
- Errors: bootstrap over events (200, whole regression + fit redone), gradients by weighted line through slice values. **The slice values scatter around a straight line more than their
  errors** (chi2/dof 2-11 for b, up to 8 for D), so I quote gradient errors times sqrt(chi2/dof) when > 1 ("scaled").

## Results (both runs pooled unless said; table T1 in tables.md has per-run values)
| quantity | measured (scaled err) | audited-fit prediction | flat cathode | cathode parallel |
|---|---|---|---|---|
| X view, d b_x/dU | +86 +- 11 | +78 (+- 7) | | |
| Y view, d b_y/dW | +151 +- 14 | +131 (+- 9) | | |
| X view, d b_x/dW (cross) | +5 +- 11 | -1 | | |
| Y view, d b_y/dU (cross) | +12 +- 7 | -1 | | |
| X view, d D_x/dU | **+73 +- 51** (bootstrap +-18) | +28 | -78 | 0 |
| Y view, d D_y/dW | **-133 +- 15** | -109 | -131 | 0 |
| X, d D_x/dW (cross) | -4 +- 20 | -1 | | |
| Y, d D_y/dU (cross) | -33 +- 15 | -1 | | |
Significance (measured - prediction, errors incl. the joint-fit errors): b_x +0.7 s, b_y +1.2 s; D_x: audited +0.9 s, flat +2.9 s, parallel +1.4 s;
D_y: audited -1.1 s, flat -0.1 s, **parallel -8.8 s**. Cross terms of b are 0 (|z| < 1.6 s); D_y vs U is -2.0 s from 0, small and one of many numbers.

Per run (stat. errors, unscaled): b_x 114+-16 / 82+-11, b_y 145+-14 / 151+-9, D_x 10+-39 / 78+-24, D_y -147+-32 / -127+-18 (run 1 / run 2): consistent.
Per-slice values at U = 0: b_x = -0.0046 +- 0.0016 (audited -0.0016), at W = 0: b_y = -0.0043 +- 0.0013 (audited -0.0040). Curvature q is
+1.8e-4 /mm at the centre of X (D_x = +0.010 +- 0.004: the cathode slope is larger than the mesh slope by 0.01 there).

Mean-slope consistency (item 5): my straight-line slope of delta(z) over the range has gradient X +116.4 +- 2.7, Y +89.9 +- 2.2; the pass-1 tables' w0/v gradient (per-slice robust ladder, both runs) is X +107.1 +- 1.4,
Y +96.0 +- 1.4. They agree to 8 % (X, +3.0 s) and -6 % (Y, -2.3 s). So the method is understood at the ~10 % level; the small residual is not explained
(candidates: the straight-line fit of the pass weights time bins differently from my 60 ns bins; selection). The audited joint-fit mean-slope gradient ((h+c)/2: X 92, Y 77) is *lower* than both the tables (107, 96) and my values.

Weight a in w0 = v[(1-a) h + a c]: a = (grad w0/v - grad b)/grad D = **0.28 +- 0.05 (X)** and **0.41 +- 0.02 (Y)** (statistical, b and D anti-correlated so the combination is tight). Over the systematic variants:
X 0.26-0.40 (0.67 with the narrow window, 0.09 inclined-only), Y 0.34-0.44. So a ~ 0.3-0.4, not 0.5: the straight-line fit weights the mesh end more than the cathode end.
(A plain time-uniform weighting of my own range would give 0.46-0.47.)

## Verdicts
1. **Mesh-end slope gradient vs the mesh dish: yes.** b_x gradient +86 +- 11 (dish +78), b_y +151 +- 14 (dish +131); 0.7 s and 1.2 s. Y is 15 % higher than the dish, not significant. Head-on tracks alone give 64 +- 11 and 142 +- 9. This is an independent confirmation of kuu and kww, with a different observable and different systematics. Cross gradients are zero as predicted.
2. **Curvature along y: yes, sign and size as predicted.** D_y gradient -133 +- 15; the "parallel cathode" value 0 is excluded at 8.8 s and in every variant (range -93 to -184, head-on -109, leaving out an end slice -113/-153). Flat cathode (-131) and audited (-109) are both consistent (0.1 s, 1.1 s). The data cannot separate flat from audited along y (2e-3 apart, error 1.5e-3 plus systematics).
3. **Curvature along x: undecided between "follows the board" and "parallel"; flat cathode disfavoured but not excluded.** D_x gradient +73 +- 51 (scaled; bootstrap alone +-18): audited +28 is +0.9 s, parallel 0 is +1.4 s, flat -78 is +2.9 s away. The result is dominated by the U = -1.32 slice (x 30-98 mm, nearest the board edge): its D is -0.0136 against +0.009 to +0.013 in the other four slices (`fig_gradients.png`). Without it the gradient is -23 +- 25 (flat -78 is 2.2 s away, audited +28 is -2.0 s away, 0 is -0.9 s away); without slice 5, +118 +- 27. The head-on-only estimator gives +115 +- 24. Over the systematic variants the central value runs from +10 to +131. What is robust is the **level**: D_x = +0.007 to +0.013 in four of five slices, i.e. the cathode slope exceeds the mesh slope at all U > -0.8 by about 0.01, but its change across the chamber is not determined. The 4x4 plane fit gives +49 +- 21 along U and +48 +- 23 along W (the latter should be ~0), chi2 32/13.
4. **Weight a = 0.28 (X), 0.41 (Y)**, systematic range 0.26-0.67 (X), 0.34-0.44 (Y).

## Systematics (table T5 in tables.md)
- Threshold 2 / 5 sigma, saturated events kept or only saturated samples dropped, half-sample bins, fit range 30-690 or 90-810 ns, dead-strip windows kept: b moves by <= 8, D by <= 20 (x) / 40 (y), well inside the scaled errors.
- **Window half width** is the largest systematic on D: (2+4|tan|) mm: D_x +19, D_y -182; (5+9|tan|) mm: D_x +57, D_y -97 (default 3+6|tan| gives +73, -133). A narrow window truncates the charge cloud (Y: RC spreading) and bends delta(z). b moves only 94 / 88 and 158 / 137.
- Inclined-only (|tan| > 0.12): b, D unchanged within doubled errors (D_x +38 +- 39, D_y -184 +- 39). Balanced |tan| < 0.3 gives the same as default.
- **Sign of tan:** one-sided samples (head-on plus +tan, or plus -tan, not balanced) give a larger D_x (+113, +131) and smaller D_y magnitude (-108, -114); the head-on-only estimator behaves the same (+115, -109). Balancing in sign is exact for odd terms, and the centroid is non-linear in tan: the residual of the line is about -0.3 mm (+tan) and +0.2 mm (-tan) at |tan| 0.3-0.45, ~ -0.04 / +0.03 mm at |tan| 0.06-0.2. So one-sided and head-on-only estimators carry a leak that the balanced one does not; I regard them as a check, and as a systematic of about +-0.004 on D_x and +-0.002 on D_y. The effective z_k itself depends slightly on |tan|, which also feeds head-on-only.
- Run 1 against run 2 agree within errors for every quantity.
- Dead strips: windows containing a masked strip (X list from `GM.MASK_X`, Y dead 510 of the bundle) are dropped; keeping them changes b by -4 and D by +6. Edges: all rows are >= 30 mm inside the active area. Leaving out the outermost slice at U = -1.32 is the one big effect (above). The z_k curves of the outer slices (-1.3 U and -1.2 W) are ~1 mm higher than the central ones at equal time (`fig_zk_calibration.png`): z_k contains the local mesh height h and the local velocity, so this is expected, not checked further.
- Saturated events: 2-7 % shifts, in or out.
- Y RC spreading: the centroid is symmetric by construction; no sign of bias next to the dead strip, but the Y D estimate depends on window width (above), so it carries the RC-truncation uncertainty.
- Within-bin errors of delta_k (0.011 mm per bin, all events) are correlated between bins; all errors are bootstrap on events, so this is included. Bins in z are not independent either.

## What the data cannot decide
- Whether the cathode follows the board along x: only the level (D_x ~ +0.01) is measured, its gradient depends on one edge slice and the choice of estimator (+10 to +131, scaled error 50).
- Flat versus audited cathode along y (the two differ by 2e-3).
- The behaviour of the lines within 2.8 mm of the mesh and above 23 mm: the first and last samples mix depths; b and D are extrapolations there (b by 3 mm, D by 5-9 mm).
- An absolute value of the mesh and cathode slope at any point: the centroid regression includes the local mesh height, the alignment (a free intercept per bin) and the |tan|-dependent z_k.
- The 6-8 % mismatch between my straight-line slope gradient and the table w0/v gradient remains unexplained.
