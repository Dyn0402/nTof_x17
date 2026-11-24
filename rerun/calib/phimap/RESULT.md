# Per-mounting angle maps: production fitter and results (bench re-finalization, step 3, 2026-11-24)

Code: `rerun/calib/phimap/` (`rows.py` loader, `mountings.py`, `lib.py` fitter, `driver.py`, `post.py`, `report.py`, `mdreport.py`, `verify_bundle.py`).
Outputs (large, not in git): `/media/ucla/work/claude-jobs/d006ab29/phimap/` with `blocks/<unit>[_tilt].json` (the `bow['phi']` blocks, validated with `wft.bow_phi.validate`),
`row_constants.json` (per row: map block, w0 and kw per view with the map applied, flags), `final/<row>.parquet` (all 156 rows, every gate-table event), `units/<unit>.json`, `result.json`.
Small files copied to `rerun/calib/phimap/results/`.

## Summary

- **Four mountings reach the r3 targets within errors** (A 6-27/28, A 6-25, B 6-22, D 6-26; table "Against the r3 targets"), now with the production-form fit, the row constants refit inside every fold, railed tracks excluded and a lambda grid that is never at its edge. The largest difference is +1.1 sigma (B X, 0.684 against 0.675). Gains 22-35 % on A and B, 16-21 % on D (X for x >= 200 mm only), 16 % on the C long row.
- **C per row** gives 1.125 -> 0.939 (X) and 1.415 -> 1.193 (Y) on the long row (16 %), 5.6 / 8.6 % on the longer row (1700 unrailed events only), and on the three r3 rows together 1.183 -> 1.038 / 1.480 -> 1.316 (12 % / 11 %), as the leave-one-row-out gain of r3 (1.037 / 1.285).
- **Do not install:** A 6-22 (81 % of its tracks are railed; on all events the map gains nothing, -2 %), A 6-23 (the pass-1 angles are uncorrelated with the M3 reference, r = 0.01: the angle is unusable that night), B 6-17 (104 gate events), and **E** (joint map: X 0.2 %, Y 3.7 %, 6-24 X is worse by 0.07 +- 0.03 deg, and the edge zone is worse by 0.5 / 1.0 deg). These stay on the bow block.
- **Edge zone is a problem for four maps** (A 6-25 Y +0.28, C longer Y +0.84, D Y +0.12, E +0.55 / +1.0 deg, production map clamped at the domain edge against pass 1). `row_constants.json` carries `x_edge_use_pass1` / `y_edge_use_pass1` and the parquet an `interior` column: use the map only inside the 30 mm interior for those views, or taper. Needs a decision.
- **Reproduction:** the block installed in the row's real bundle (saved and reloaded through `CalibrationBundle`) and `wft.reco.bow_pairs` give the fitter's tangent to 5e-14 (the fitter's own spline algebra) and identically (0.0) the columns written to the parquet, on every unit.
- **Lambda never sits at the grid edge** (10^-6 ... 10^3, chosen 10^-5 ... 10^2); K is never at its maximum (24); K_S (maximum 16) hits the maximum once in A 6-27/28 and once in B.

## What was fitted

- **Form** (production, `wft.bow_phi` / `bow_pairs`): tan = (w - w0_r) / (c1_r (1 + S_p(U, W))) - A_p(U, W), c1 = kw v_drift of the row, A_x = dPhi/dU + cx, A_y = dPhi/dW + cy (one shared potential, kappa = 1), S_x and S_y independent scale maps, all in tan units; (U, W) = (p0 + 15 mm z - 200)/100 with z = (w - w0)/c1, as `bow_pairs`. A and S are fitted directly in this form (Gauss-Newton on the product (1 + S)(tan_ref + A), 4-6 rounds to 1e-5 in tan; the means of A and 1 + S are moved into the row constants after every round). No fixed vy/vx.
- **Selection** (as d_lib/r3): single track (`n_tracks == 1`, except the tables where the column is 0 for every event: det3 on 6-23 and the 100 V drift-scan row, selected without it and flagged `ntracks_unavailable`), x and y ok and quality_ok, no spark, M3 chi2 < 1 and >= 4 clusters per view, reference mid-gap position (ref_mesh + 14 mm tan_ref) inside the chamber's active area minus 30 mm, |tan_ref| <= 0.6 (never binding, M3 reaches 0.38). Spline domain = that interior, rounded outward to 0.01 (U -1.70 ... 1.69, W about -1.52 ... 1.50); a track outside is clamped to it.
- **Railed tracks:** a view is railed when its `q_uend` reaches the end of the fit's depth grid, K x 60 ns (K from the registry: 24, or 30 at <= 500 V). Railed events are excluded per view from every fit and from all statistics except where stated. Railed fractions (any view): A 6-22 80 %, C 21 / 31 / 35 % (long / longer / short) and 24 % over C, E 13.5 %, D 1.3 %, A 6-23 41 %, others < 1 %. The handoff quotes slightly different numbers (it does not say which view or definition; A 6-22 X 70 %, D short 14 % and E 6-24 9-10 % are the X view or the mean of the views here).
- **Model selection** per fit, on the training part only: 4 inner time blocks per row, K in {6, 9, 12, 16, 20, 24}, K_S in {0, 4, 6, 8, 10, 12, 14, 16}, lam in 10^-6 ... 10^3, K^2 + 2 K_S^2 <= N_train/15; the score is s68(X) + s68(Y) of the held-out inner blocks, in the linear a2 form (row constants from the inner training part).
- **Row constants** (w0, kw per view) are the robust line of w against (1 + S)(tan_ref + A) on the row's unrailed events (3.5 MAD clip). In every cross-validation fold they come from the row's events outside the block, so each CV angle is from a map AND constants trained without the event's time block (r3 refit them on the test block).
- **Cross-validation:** 5 time blocks inside every row (rank in (acq_tag, timestamp) over the whole gate table), so every fold has training events of every row. Quoted numbers are the CV ones; `x_tan_map` is the production map (in sample).
- **Units:** one map per mounting; C one map per row with >= 2000 gate events (long 4981, longer 2191); the other C rows (short and the 20 + 2 HV-scan rows) take the map of the C row nearest in time (`borrowed`; the constants are always the row's own, or the nearest row's when it has < 60 usable events: `const_src`).
- **Other drift fields:** A 6-27/28 has six rows at 100-1100 V besides the 1000 V main field. They get the tilt part only unless the scale part helps out of sample: pooled it helps (X 1.100 -> 0.995, -0.105 +- 0.017; Y 1.030 -> 0.986, -0.045 +- 0.014), so they use the full map. It does not help at 100 and 300 V (pass 1 is 8.9 and 1.7 deg there, and no variant gains), so those two rows are marked not recommended.

## Mountings (`mountings.csv`: one line per registry row)

| mounting | chamber | map unit(s) | rows | gate events | main drift V | rows by role | railed (any view) | alignment spread (x off, y off, theta) | decision |
|---|---|---|---|---|---|---|---|---|---|
| B_0617 | B | none | 10 | 104 | 1000 | nomap 10 | 0.0 % | 0.00, 0.00 mm, 0.00 deg | only 104 gate events in total (< 2000): stays on the bow block |
| B_0622 | B | B_0622 | 19 | 21411 | 1000 | train 19 | 0.0 % | 0.10, 0.10 mm, 0.10 deg |  |
| A_0622 | A | A_0622 | 19 | 22186 | 1000 | train 19 | 79.7 % | 0.50, 0.30 mm, 0.10 deg |  |
| A_0625 | A | A_0625 | 1 | 6279 | 500 | train 1 | 0.0 % | 0.00, 0.00 mm, 0.00 deg |  |
| A_0623 | A | none | 16 | 3999 | 600 | nomap 16 | 40.8 % | 0.00, 0.20 mm, 0.10 deg | pass-1 angles are uncorrelated with the M3 reference (corr 0.01): no angle map is possible |
| A_0627 | A | A_0627 | 26 | 29467 | 1000 | train 20, tilt_only 6 | 0.2 % | 0.10, 0.10 mm, 0.15 deg |  |
| E | E | E | 17 | 5739 | 600 | tiny 10, train 7 | 13.5 % | 1.60, 7.90 mm, 0.95 deg |  |
| C_0626 | C | C_long, C_longer | 24 | 14419 | 700 | borrowed 22, train 2 | 23.9 % | 0.10, 0.10 mm, 0.10 deg |  |
| D_0626 | D | D_0626 | 24 | 9364 | 700 | train 24 | 1.3 % | 0.10, 0.10 mm, 0.20 deg |  |

## Angle s68 [deg]: pass 1 against the map, 5-time-block cross-validation

Events: gate events of the unit's training rows (railed tracks of the view excluded; D X only for reference x >= 200 mm). `map CV` = each event from a map and row constants trained without its time block; `map (production)` = the shipped map on its own training events (in sample, for reference only).

| unit | view | N | pass 1 | map CV | gain (CV) | map (production) | far fraction (> 3 s68 of pass 1): pass 1 -> CV | K / K_S / lam (full fit) |
|---|---|---|---|---|---|---|---|---|
| A_0622 | X | 6612 | 5.804 +- 0.078 | 2.709 +- 0.047 | 53.3 % (-3.095 +- 0.079) | 2.642 | 0.7 % -> 0.1 % | 9 / 6 / 1 |
| A_0622 | Y | 5881 | 5.101 +- 0.070 | 2.139 +- 0.038 | 58.1 % (-2.962 +- 0.071) | 2.064 | 1.3 % -> 0.2 % | 9 / 6 / 1 |
| A_0625 | X | 6278 | 0.876 +- 0.011 | 0.681 +- 0.009 | 22.3 % (-0.196 +- 0.010) | 0.668 | 1.3 % -> 0.8 % | 16 / 8 / 0.01 |
| A_0625 | Y | 6279 | 1.030 +- 0.017 | 0.667 +- 0.009 | 35.3 % (-0.363 +- 0.017) | 0.635 | 1.2 % -> 0.4 % | 16 / 8 / 0.01 |
| A_0627 | X | 26497 | 0.924 +- 0.006 | 0.689 +- 0.004 | 25.5 % (-0.236 +- 0.006) | 0.679 | 1.3 % -> 0.9 % | 16 / 14 / 0.01 |
| A_0627 | Y | 26497 | 0.815 +- 0.006 | 0.607 +- 0.004 | 25.5 % (-0.208 +- 0.005) | 0.598 | 1.5 % -> 1.0 % | 16 / 14 / 0.01 |
| B_0622 | X | 21411 | 0.942 +- 0.007 | 0.670 +- 0.005 | 28.9 % (-0.272 +- 0.006) | 0.652 | 1.7 % -> 1.2 % | 16 / 14 / 0.01 |
| B_0622 | Y | 21411 | 0.957 +- 0.006 | 0.721 +- 0.005 | 24.7 % (-0.236 +- 0.006) | 0.707 | 1.6 % -> 1.2 % | 16 / 14 / 0.01 |
| C_long | X | 4772 | 1.125 +- 0.017 | 0.939 +- 0.015 | 16.5 % (-0.186 +- 0.016) | 0.884 | 3.1 % -> 2.4 % | 9 / 8 / 0.01 |
| C_long | Y | 4057 | 1.415 +- 0.022 | 1.193 +- 0.020 | 15.7 % (-0.222 +- 0.021) | 1.140 | 2.4 % -> 1.7 % | 9 / 8 / 0.01 |
| C_longer | X | 1747 | 1.225 +- 0.028 | 1.157 +- 0.030 | 5.6 % (-0.069 +- 0.024) | 1.025 | 3.5 % -> 3.1 % | 9 / 4 / 0.001 |
| C_longer | Y | 1699 | 1.597 +- 0.043 | 1.460 +- 0.039 | 8.6 % (-0.137 +- 0.029) | 1.381 | 3.0 % -> 2.6 % | 9 / 4 / 0.001 |
| D_0626 | X | 3628 | 0.924 +- 0.020 | 0.777 +- 0.014 | 16.0 % (-0.148 +- 0.017) | 0.749 | 2.6 % -> 2.0 % | 16 / 8 / 1 |
| D_0626 | Y | 8962 | 0.959 +- 0.010 | 0.756 +- 0.010 | 21.2 % (-0.203 +- 0.010) | 0.720 | 3.2 % -> 2.8 % | 16 / 8 / 1 |
| E | X | 4629 | 2.162 +- 0.045 | 2.158 +- 0.037 | 0.2 % (-0.005 +- 0.025) | 2.148 | 9.8 % -> 10.3 % | 6 / 6 / 0.1 |
| E | Y | 4700 | 1.458 +- 0.034 | 1.403 +- 0.037 | 3.7 % (-0.054 +- 0.026) | 1.355 | 12.2 % -> 12.7 % | 6 / 6 / 0.1 |

## Against the r3 targets (r3 rows: long, longer, short; r3 included railed tracks)

`excl.` = railed tracks of the view excluded from fit and statistics (production treatment); `incl.` = all selected events (C with the railed rescale; D and E railed tracks are only excluded in the production output, shown here for comparison with r3). r3 refit the two row constants on every test block; here the constants also come from the training blocks only.

| subset | view | r3 pass 1 | r3 map | ours pass 1 (excl.) | ours CV (excl.) | ours pass 1 (incl.) | ours CV (incl.) | (ours incl. - r3) / sigma |
|---|---|---|---|---|---|---|---|---|
| A 6-27/28 (sat + p2 long rows) | X | 0.923 | 0.688 | 0.923 +- 0.006 | 0.689 +- 0.005 | 0.923 +- 0.006 | 0.689 +- 0.005 | +0.2 |
| A 6-27/28 (sat + p2 long rows) | Y | 0.805 | 0.599 | 0.805 +- 0.007 | 0.597 +- 0.005 | 0.805 +- 0.007 | 0.597 +- 0.005 | -0.3 |
| A 6-25 | X | 0.876 | 0.681 | 0.876 +- 0.011 | 0.681 +- 0.009 | 0.876 +- 0.011 | 0.681 +- 0.009 | -0.0 |
| A 6-25 | Y | 1.030 | 0.666 | 1.030 +- 0.017 | 0.667 +- 0.009 | 1.030 +- 0.017 | 0.667 +- 0.009 | +0.1 |
| B 6-22 (long, longer, short) | X | 0.961 | 0.675 | 0.961 +- 0.008 | 0.684 +- 0.006 | 0.961 +- 0.008 | 0.684 +- 0.006 | +1.1 |
| B 6-22 (long, longer, short) | Y | 0.966 | 0.735 | 0.966 +- 0.008 | 0.734 +- 0.006 | 0.966 +- 0.008 | 0.734 +- 0.006 | -0.1 |
| D 6-26 (long, longer, short; X: x >= 200) | X | 0.968 | 0.802 | 0.966 +- 0.024 | 0.802 +- 0.020 | 0.968 +- 0.023 | 0.802 +- 0.019 | -0.0 |
| D 6-26 (long, longer, short; X: x >= 200) | Y | 1.028 | 0.829 | 1.020 +- 0.015 | 0.804 +- 0.012 | 1.028 +- 0.015 | 0.809 +- 0.012 | -1.1 |
| E 6-23 (joint map) | X | 2.405 | 2.354 | 2.332 +- 0.069 | 2.262 +- 0.057 | 2.327 +- 0.069 | 2.261 +- 0.056 | -1.0 |
| E 6-23 (joint map) | Y | 1.627 | 1.377 | 1.544 +- 0.062 | 1.419 +- 0.062 | 1.549 +- 0.063 | 1.424 +- 0.062 | +0.6 |
| E 6-24 (joint map) | X | 2.141 | 2.120 | 2.025 +- 0.051 | 2.096 +- 0.053 | 2.141 +- 0.057 | 2.171 +- 0.050 | +0.7 |
| E 6-24 (joint map) | Y | 1.436 | 1.324 | 1.419 +- 0.045 | 1.403 +- 0.045 | 1.436 +- 0.040 | 1.421 +- 0.042 | +1.6 |
| C long row (r3: all three C rows) | X | 1.183 | 0.932 | 1.125 +- 0.017 | 0.939 +- 0.015 | 1.128 +- 0.016 | 0.942 +- 0.014 | +0.6 |
| C long row (r3: all three C rows) | Y | 1.480 | 1.228 | 1.415 +- 0.022 | 1.193 +- 0.020 | 1.379 +- 0.019 | 1.165 +- 0.017 | -2.8 |

Chamber C, the three r3 rows together (own map for long and longer, the map of the nearest row in time for short), s68 [deg]; r3: pass 1 1.183 / 1.480, ONE map for the three rows 0.932 / 1.228, leave-one-row-out 1.037 / 1.285:

| railed tracks | view | N | pass 1 | map CV | map CV x railed rescale |
|---|---|---|---|---|---|
| excluded | X | 7157 | 1.180 +- 0.016 | 1.029 +- 0.014 | 1.029 +- 0.014 |
| excluded | Y | 6381 | 1.521 +- 0.021 | 1.333 +- 0.020 | 1.333 +- 0.020 |
| included | X | 7985 | 1.183 +- 0.016 | 1.038 +- 0.012 | 1.042 +- 0.012 |
| included | Y | 7985 | 1.480 +- 0.020 | 1.316 +- 0.016 | 1.324 +- 0.018 |

## Rows that were not in the training of their map

| unit | role | view | N | pass 1 | map CV | gain |
|---|---|---|---|---|---|---|
| A_0627 | tilt_only | X | 2967 | 1.524 +- 0.042 | 0.995 +- 0.028 | 34.7 % (-0.529 +- 0.040) |
| A_0627 | tilt_only | Y | 2927 | 1.301 +- 0.033 | 0.986 +- 0.024 | 24.2 % (-0.315 +- 0.029) |
| C_long | borrowed | X | 5693 | 1.091 +- 0.016 | 0.792 +- 0.011 | 27.4 % (-0.298 +- 0.016) |
| C_long | borrowed | Y | 5358 | 1.271 +- 0.017 | 1.020 +- 0.017 | 19.8 % (-0.251 +- 0.018) |
| C_longer | borrowed | X | 725 | 1.451 +- 0.058 | 1.334 +- 0.056 | 8.1 % (-0.117 +- 0.040) |
| C_longer | borrowed | Y | 694 | 2.028 +- 0.087 | 1.872 +- 0.070 | 7.7 % (-0.156 +- 0.056) |
| E | tiny | X | 425 | 2.288 +- 0.145 | 2.051 +- 0.157 | 10.4 % (-0.237 +- 0.132) |
| E | tiny | Y | 442 | 2.034 +- 0.187 | 1.473 +- 0.109 | 27.6 % (-0.561 +- 0.154) |

Tilt part only (S = 0) against the full map on the rows at another drift field of A 6-27/28 (CV, unrailed):

| drift V | rows | N | X pass 1 | X tilt only | X full | Y pass 1 | Y tilt only | Y full |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 | 458 | 8.876 | 9.148 | 9.107 | 6.741 | 5.936 | 5.922 |
| 300 | 1 | 448 | 1.722 | 1.734 | 1.748 | 1.835 | 1.713 | 1.721 |
| 500 | 1 | 532 | 0.872 | 0.794 | 0.724 | 0.947 | 0.814 | 0.724 |
| 700 | 1 | 509 | 0.847 | 0.724 | 0.622 | 0.781 | 0.637 | 0.582 |
| 900 | 1 | 515 | 0.892 | 0.707 | 0.614 | 0.817 | 0.639 | 0.588 |
| 1100 | 1 | 465 | 0.894 | 0.724 | 0.703 | 0.836 | 0.622 | 0.637 |

Pooled decision for A_0627: X: pass 1 1.524, tilt only 1.100, full 0.995, full - tilt = -0.105 +- 0.017; Y: pass 1 1.301, tilt only 1.030, full 0.986, full - tilt = -0.045 +- 0.014 -> scale part used: True.

## Railed tracks

| row (C) | view | railed events | factor 1/slope | slope |
|---|---|---|---|---|
| all other C rows (< 150 railed events each), pooled | X | | 1.034 | 0.967 |
| all other C rows (< 150 railed events each), pooled | Y | | 1.071 | 0.934 |
| g_det6_long | X | 209 | 1.047 | 0.955 |
| g_det6_long | Y | 924 | 1.109 | 0.902 |
| _det6_det7_overnight_6-26-26__longer_run__mx17_6 | X | 444 | 1.036 | 0.965 |
| _det6_det7_overnight_6-26-26__longer_run__mx17_6 | Y | 492 | 1.083 | 0.923 |
| 7_det6_det7_overnight_6-26-26__short_run__mx17_6 | X | 175 | 1.062 | 0.942 |
| 7_det6_det7_overnight_6-26-26__short_run__mx17_6 | Y | 188 | 1.024 | 0.977 |

Railed C tracks, residual x sign(tan_ref) [deg] (a negative mean = flatter than the reference):

| view | N | pass 1 mean / s68 | map CV mean / s68 | map CV x factor, mean / s68 |
|---|---|---|---|---|
| X | 1046 | -0.334 / 1.188 | -0.331 / 1.026 | +0.018 / 1.144 |
| Y | 1471 | -0.383 / 1.483 | -0.441 / 1.296 | +0.036 / 1.389 |

## Rows with >= 300 gate events whose CV angle is worse than pass 1 by more than 2 sigma (paired bootstrap)

| row | unit | role | view | N | pass 1 | map CV | diff |
|---|---|---|---|---|---|---|---|
| g_det4 | E | train | X | 2773 | 2.025 | 2.096 | +0.071 +- 0.034 |

## Mid-gap position residual [mm] and edge zone

Position: s68 of (p0 + 14 tan) - (ref_mesh + 14 tan_ref), pass 1 -> map CV (unrailed events of the long rows). Edge zone: events inside the active area but within 30 mm of its border (never in a fit; production map, clamped to the domain; unrailed): angle s68 pass 1 -> map.

| unit | position X | position Y | edge X: N, pass 1 -> map (diff) | edge Y: N, pass 1 -> map (diff) |
|---|---|---|---|---|
| A_0622 | 1.612 -> 0.784 | 1.464 -> 0.704 | 862, 10.67 -> 4.87 (-5.79 +- 0.34) | 673, 9.59 -> 4.96 (-4.63 +- 0.35) |
| A_0625 | 0.415 -> 0.386 | 0.441 -> 0.403 | 1189, 1.51 -> 1.56 (+0.05 +- 0.06) | 1189, 1.48 -> 1.76 (+0.28 +- 0.06) |
| A_0627 | 0.555 -> 0.523 | 0.534 -> 0.511 | 2077, 1.22 -> 1.05 (-0.17 +- 0.03) | 2077, 1.20 -> 0.95 (-0.25 +- 0.03) |
| B_0622 | 0.546 -> 0.512 | 0.521 -> 0.493 | 3017, 3.92 -> 2.89 (-1.04 +- 0.06) | 3017, 2.37 -> 2.06 (-0.31 +- 0.04) |
| C_long | 0.472 -> 0.445 | 0.444 -> 0.411 | 1417, 2.03 -> 1.89 (-0.14 +- 0.07) | 1185, 1.60 -> 1.69 (+0.09 +- 0.06) |
| C_longer | 0.451 -> 0.448 | 0.516 -> 0.475 | 473, 2.67 -> 2.92 (+0.25 +- 0.20) | 437, 1.94 -> 2.77 (+0.84 +- 0.13) |
| D_0626 | 0.535 -> 0.498 | 0.543 -> 0.530 | 738, 1.04 -> 1.10 (+0.06 +- 0.04) | 1217, 1.16 -> 1.28 (+0.12 +- 0.04) |
| E | 0.867 -> 0.861 | 0.711 -> 0.703 | 695, 2.32 -> 2.87 (+0.55 +- 0.14) | 711, 2.31 -> 3.31 (+1.01 +- 0.16) |

## Model selection: edge hits of the inner CV (6 fits per unit: full + 5 folds)

Grids: K in {6, 9, 12, 16, 20, 24}, K_S in {0, 4, 6, 8, 10, 12, 14, 16}, lam in 10^-6 ... 10^3 (10 values), K^2 + 2 K_S^2 <= N/15.

| unit | lam at grid edge | K at grid max | K_S at grid max | lam range chosen | K chosen | K_S chosen |
|---|---|---|---|---|---|---|
| A_0622 | 0 | 0 | 0 | 0.1 .. 100 | 9, 16, 16, 9, 12, 9 | 6, 0, 4, 0, 0, 0 |
| A_0625 | 0 | 0 | 0 | 0.0001 .. 0.1 | 16, 12, 9, 12, 9, 16 | 8, 8, 10, 8, 8, 6 |
| A_0627 | 0 | 0 | 1 | 0.01 .. 0.1 | 16, 16, 16, 16, 16, 20 | 14, 14, 12, 12, 16, 14 |
| B_0622 | 0 | 0 | 1 | 0.001 .. 0.1 | 16, 16, 12, 16, 12, 16 | 14, 14, 12, 16, 8, 8 |
| C_long | 0 | 0 | 0 | 0.0001 .. 0.1 | 9, 9, 12, 9, 12, 12 | 8, 6, 6, 8, 4, 6 |
| C_longer | 0 | 0 | 0 | 0.0001 .. 10 | 9, 6, 6, 6, 6, 6 | 4, 4, 4, 4, 4, 4 |
| D_0626 | 0 | 0 | 0 | 1e-05 .. 10 | 16, 12, 12, 12, 12, 16 | 8, 6, 8, 4, 8, 6 |
| E | 0 | 0 | 0 | 0.1 .. 10 | 6, 6, 6, 6, 6, 9 | 6, 10, 10, 10, 10, 4 |

## Reproduction of the fitter by wft.reco.bow_pairs (block installed in the row bundle)

| unit | row | n events | max abs diff tan X | max abs diff tan Y |
|---|---|---|---|---|
| A_0622 | mx17_det2_det3_overnight_6-22-26__long_run__mx17_3 | 11972 | 1.1e-16 | 1.1e-16 |
| A_0625 | c26__mx17_det3_day_6-25-26__long_run__mx17_3 | 6279 | 1.7e-16 | 1.7e-16 |
| A_0627 | ght_6-27-26__long_run_p2_det1_sanity_check__mx17_3 | 14066 | 1.7e-16 | 8.3e-17 |
| B_0622 | mx17_det2_det3_overnight_6-22-26__long_run__mx17_2 | 10963 | 2.5e-16 | 1.1e-16 |
| C_long | g_det6_long | 4981 | 5.6e-17 | 1.1e-16 |
| C_longer | 17_det6_det7_overnight_6-26-26__longer_run__mx17_6 | 2191 | 1.1e-16 | 2.2e-16 |
| D_0626 | g_det7_long | 2680 | 2.2e-16 | 2.2e-16 |
| E | g_det4 | 3128 | 6.7e-16 | 4.8e-14 |

## E with one map per day (diagnostic, not shipped)

| map | view | N | pass 1 | map CV | gain |
|---|---|---|---|---|---|
| E_0623 | X | 1856 | 2.332 +- 0.069 | 2.256 +- 0.074 | 3.2 % (-0.076 +- 0.044) |
| E_0623 | Y | 1851 | 1.544 +- 0.062 | 1.380 +- 0.058 | 10.6 % (-0.164 +- 0.041) |
| E_0624 | X | 2773 | 2.025 +- 0.051 | 2.094 +- 0.050 | -3.4 % (+0.069 +- 0.029) |
| E_0624 | Y | 2849 | 1.419 +- 0.045 | 1.432 +- 0.044 | -0.9 % (+0.012 +- 0.032) |

## Differences from r3 larger than 2 sigma, and what they are

None for a like-for-like row set. The largest are:

- **C Y (-2.8 sigma, ours better):** the r3 line is the three C rows together with ONE map (time-block CV, optimistic: every test block shares rows with the training). We quote the long row alone (own map). Over the three rows with per-row maps we get 1.038 / 1.316 (r3 leave-one-row-out 1.037 / 1.285, the fair comparison), and no number is better than r3's honest one.
- **B X (+1.1 sigma, ours worse by 0.009):** the row constants come from the training blocks here; r3 refit them on the test block (two numbers measured on the events being scored). Same direction, smaller than 2 sigma, on D (X 0.802 equal, Y 0.809 against 0.829, -1.1 sigma).
- **A 6-25 and A 6-27/28 reproduce to 0.001 deg.** The larger K_S (up to 16) of the extended grid brings nothing beyond r3's K_S = 8 (0.689 / 0.597 against 0.688 / 0.599).
- **E** is not comparable: one joint map here, one per day in r3. With one map per day (diagnostic, `phimap_diag`): 6-23 Y 1.544 -> 1.380 (10.6 %), X 3.2 %; 6-24 X +0.069 +- 0.029 (worse), Y -0.9 %. r3's Y gains (15 % and 8 %) shrink by the constants (no refit on the test block) and by pooling.

## Things that look wrong or need a decision

1. **E: install nothing, or Y only.** The joint map does not help X (2.162 -> 2.158) and makes 6-24 X worse by 2 sigma (2.025 -> 2.096; the one row above); Y gains 3.7 % (2 sigma) but the edge zone is worse by 1.0 deg (6 sigma). Pass 1 itself is 1.4-2.2 deg there. Recommended: E stays on the bow block (`x_map_recommended = false`; Y is `true` by the 2-sigma rule only with `y_edge_use_pass1`).
2. **Edge zone** (events inside the active area but within 30 mm of its border, production map clamped to the domain, against pass 1): worse on A 6-25 Y (1.48 -> 1.76), C longer Y (1.94 -> 2.77), D Y (1.16 -> 1.28), E X and Y; better on A 6-27/28 (-0.17 / -0.25), B (-1.04 / -0.31), C long X (-0.14). The tilts are largest at the edges, and a clamped spline cannot follow them. A rule has to be chosen before installing (map only in the interior, taper, or fit the map on a wider domain, which needs events at the edge: the gate selection has none by construction). The flags are in `row_constants.json`, the `interior` column in every parquet.
3. **A 6-22 stays on the bow block.** 80 % of the tracks are railed (K = 24, 1440 ns, wet gas). On the unrailed 20 % the map takes pass 1 from 5.8 / 5.1 deg to 2.7 / 2.1, but those are the fast tracks and not representative; on all events it changes 3.40 -> 3.47 (X) and 2.66 -> 2.69 (Y). No map helps without the refit with K = 27 the handoff mentions. The block is shipped (`blocks/A_0622.json`) but `map_recommended = false`.
4. **A 6-23: no map possible.** `corr(tan_ref, pass-1 tan)` = 0.02 (X) and 0.01 (Y) on the 3100 gate events of the long row; the pass-1 angles of det3 that night carry no information (the registry note says "degraded bottom-slot night"). The alignment says it is the same slot as 6-22 (offsets differ by 0.5 mm, theta by 0.1 deg); I kept them as separate mountings because the foil changes with every mounting, which no test here can decide, and the 6-23 data cannot be tested anyway.
5. **Railed-track rescale on C** (factor 1/slope, per row where >= 150 railed events, else pooled; slopes X 0.955 / 0.965 / 0.942, Y 0.902 / 0.923 / 0.977 per row, 0.967 / 0.934 pooled): it removes the mean bias of the railed tracks (-0.33 / -0.44 deg -> +0.02 / +0.04) but raises their s68 by about 10 % (1.03 -> 1.14, 1.30 -> 1.39), because it multiplies the noise too. On all C events the s68 changes by +0.004 / +0.008 deg (no gain). It is shipped as `x_resc`, `x_tan_final` and left to the user of the angle: use `*_tan_final` where the angle slope matters, `*_tan_map` where the width matters. D and E railed tracks are only flagged (`*_excl`), as decided.
6. **Rows worse than pass 1 beyond 2 sigma** (>= 300 gate events): one, E 6-24 X (+0.071 +- 0.034). No row worse in A, B, C, D. The far fraction (|residual| > 3 s68 of pass 1) falls on every unit except E (+0.5 pp).
7. **D X for x < 200 mm** is flagged `x_excl` (and `x_view_valid_from_mm = 200` in `row_constants.json`); the X map there is an extrapolation of the potential and must not be used.
8. **Borrowed C rows** gain 27 % (X) and 20 % (Y) with the long-row map (20 rows, 6300 events, 04:19-14:51) and 8 % with the longer-row map (2 rows, 947 events). Those rows have 20-450 gate events each, so their constants are measured on few events.
9. **The 26 rows with no map** (B 6-17: 10 rows, A 6-23: 16 rows) have a parquet with NaN map columns (`map_variant = none`) and `map_recommended = false`.
10. The 100 V and 300 V drift-scan rows of A 6-27 (pass 1: 8.9 / 6.7 deg and 1.7 / 1.8 deg) have no usable angle with or without a map.
11. K is never at its grid maximum (24) and lam never at an edge; K_S reaches its maximum (16) in one fold of A 6-27/28 and one of B; no larger K_S was tried.

## Not done

- No bundle was changed and no pass product was written: the blocks and constants are inputs to the installation step (`rerun/w12/set_w0kw.py --bow-json` pattern, `bow_finalize.py`).
- The gate "position residuals unchanged" is checked as the mid-gap position residual (improves 3-8 %, A 6-22 by a factor 2); the map does not touch p0.
- No Paper I numbers were requoted.
