# A1 - fine structure of the drift-tilt maps: reproducible? curl-free?  (audit tag a1, 2026-11-24)

Scripts (all new, `rerun/paper1_gap/audit/`): `a1_lib.py` (maps, stats, finite differences, Helmholtz system), `a1_part1.py`,
`a1_part2.py` (curl test + 200-replica event bootstrap), `a1_agg.py`, `a1_part2d.py` (Phi map, 50 mm scan, fold),
`a1_part3.py`, `a1_synth.py` (control with a pure gradient field), `a1_fig.py`, `a1_hp.py`. Outputs in `$WORK/audit/a1/`
(json, png, logs). Selection: `GM.load_table`, |tan_ref| <= 0.6 in the fitted view, `CO.line`-type robust line
(own copy `rline`, identical iteration, threshold 30 events; checked equal to `CO.line` for the chamber-wide fit).
tau = w0 / v_run (dimensionless, tan), U, W in units of 100 mm. Cells: n x n over U in [-1.7, 1.7] (340 mm), W in
[-1.51, 1.51] (302 mm); NaN below 30 events. Note: only 19 of the 256 cells at n = 16 are valid in both runs (run 1 has 4,654 ladder events),
and 62 / 80 / 86 cells at n = 8 / 10 / 12, so n = 16 is not interpretable.

## Verdict (three questions)

1. **The fine structure is real and reproducible.** After subtracting the best weighted plane, the tau_x and tau_y maps of
   run 1 and run 2 correlate with r = 0.97-0.99 (n = 4, 6, 8), 0.95-0.96 (n = 10), 0.91-0.97 (n = 12). The noise-free signal
   rms (sqrt of the run 1 x run 2 covariance) is 4.1 -> 8.0e-3 (x) and 5.9 -> 7.0e-3 (y) in tan, i.e. 0.23-0.46 deg (x) and 0.34-0.44 deg (y), against
   mean cell errors of 0.7-1.7e-3. The signal is 4-6 times the statistical error. Most of it lives on scales >= 57 mm.
2. **The tilt field is curl-free, at plane level and at fine scale, to the precision of the data.** Plane level: c_x - b_y = 0.0001 +- 0.0002 (run 2) and
   0.0006 +- 0.0004 (run 1), event-level statistical errors; 0.4 and 1.7 sigma. Fine scale: Cx (from tau_x, run a) against Cy
   (from tau_y, run b != a): r = 0.92-0.95 for n = 6-12, equal to the noise ceiling sqrt(r_xx r_yy) (curl-free fraction 0.96-1.00, +- 0.02-0.03).
   The Helmholtz fit leaves a "solenoidal" part that is **statistically consistent with the floor the method itself
   produces on a noise-free pure gradient field** (control, below). Excess solenoidal variance at n = 8: 0.1 +- 0.3e-6 against a total of 52e-6, i.e. < 1.5 % (2 sigma). A gradient field is not rejected; a non-gradient field (an instrument effect of one view)
   would give r_xy ~ 0, and r_xy is 0.92.
3. **Below 42 mm there is a small, real, reproducible remainder, about 0.15-0.2 deg rms (2.5-3.6e-3 in tan)**, not the 0.4 deg of the whole non-planar map.
   The cross-run signal rms grows from 6.8e-3 (42 mm) to 8.0e-3 (28 mm) in x (+17 +- 11e-6 in variance) and does not grow in y (-9 +- 12e-6). It is flat within errors after a quadratic is removed
   (x 6.8 -> 7.3e-3, y 7.6 -> 6.95e-3). A high-pass test (cell minus mean of its 4 neighbours, run 1 x run 2) falls slower than the h^4 expected from a smooth field:
   variance 37e-6 (n = 8) -> 16.5 +- 3.5e-6 (n = 12) for x, 32 -> 16.4 +- 3.1e-6 for y; smooth extrapolation gives ~7e-6, so an excess of ~9e-6 (2.5 sigma) points to structure at 28-56 mm of ~2.7e-3 rms.
   The 28-42 mm cells are noisy (n = 12 high-pass noise 5e-6 against 16-21e-6 variance) and cell averaging attenuates what is there.
   No peak at the 50 mm connector period (below).

## Part 1 - cell maps and reproducibility

v_run (um/ns): run 1 x 37.04 +- 0.07, y 36.50 +- 0.07; run 2 x 37.74 +- 0.04, y 37.14 +- 0.04.
Residual = tau minus weighted plane. rms = weighted rms (run 1 / run 2), err = mean cell error (run 1 / run 2),
r = Pearson run 1 vs run 2 (cell bootstrap error), sig = sqrt(max(cov,0)) of the run 1 x run 2 covariance (cell-bootstrap error on the covariance in brackets, 1e-6).

| view | n (cell mm) | cells | rms x1e3 | err x1e3 | r | sig x1e3 | sig deg | cov (1e-6) |
|---|---|---|---|---|---|---|---|---|
| x | 4 (85) | 16 | 4.2 / 3.8 | 0.9 / 0.5 | 0.977 +- 0.013 | 4.06 | 0.23 | 16.5 +- 3.9 |
| x | 6 (57) | 36 | 5.6 / 5.9 | 1.4 / 0.8 | 0.975 +- 0.007 | 5.68 | 0.33 | 32.2 +- 6.1 |
| x | 8 (42) | 62 | 6.9 / 6.8 | 1.7 / 1.0 | 0.968 +- 0.009 | 6.78 | 0.39 | 46.0 +- 7.4 |
| x | 10 (34) | 80 | 7.6 / 7.4 | 1.9 / 1.1 | 0.962 +- 0.009 | 7.68 | 0.44 | 59.0 +- 8.3 |
| x | 12 (28) | 86 | 7.8 / 7.8 | 2.1 / 1.2 | 0.965 +- 0.007 | 7.96 | 0.46 | 63.4 +- 8.6 |
| x | 16 (21) | 19 | 8.5 / 9.4 | 2.3 / 1.4 | 0.957 +- 0.015 | 8.88 | 0.51 | 78.8 +- 17.5 |
| y | 4 | 16 | 4.9 / 4.6 | 0.8 / 0.5 | 0.987 +- 0.007 | 5.86 | 0.34 | 34.3 +- 10.0 |
| y | 6 | 36 | 6.0 / 6.0 | 1.2 / 0.7 | 0.983 +- 0.005 | 7.65 | 0.44 | 58.5 +- 14.2 |
| y | 8 | 62 | 6.9 / 6.8 | 1.4 / 0.8 | 0.967 +- 0.009 | 7.62 | 0.44 | 58.1 +- 9.6 |
| y | 10 | 80 | 6.6 / 6.8 | 1.6 / 1.0 | 0.946 +- 0.010 | 7.17 | 0.41 | 51.4 +- 8.1 |
| y | 12 | 86 | 6.9 / 7.0 | 1.7 / 1.1 | 0.914 +- 0.016 | 7.03 | 0.40 | 49.5 +- 7.3 |
| y | 16 | 19 | 9.0 / 8.1 | 2.1 / 1.4 | 0.965 +- 0.012 | 8.07 | 0.46 | 65.2 +- 13.0 |

(The cells valid at each n differ, so the sequence in n mixes scale and coverage; n = 16 covers only the 19 best-filled cells.)
After removing a weighted quadratic instead of a plane (`sigtab.json`): x 4.0, 5.7, 6.8, 7.4, 7.3 (n = 16: 5.1 +- 1.8, r = 0.55 +- 0.23); y 6.2, 8.0, 7.6, 7.3, 6.95 (1e-3, n = 4 ... 12).
Signal variance added by finer cells (plane-removed, 1e-6): x: +16 (85->57), +14 (57->42), +13 (42->34), +4 (34->28); y: +24, -0.4, -7, -2.
The tau signal saturates at about 7-8e-3 (0.4-0.46 deg) by 34-42 mm cells; it does not keep growing as cells shrink to 28 mm (the n = 16 points are not reliable).

v_rel maps (v / v_run, run 1 x run 2): minus mean: signal 0.029-0.033 (x), 0.021-0.026 (y), r = 0.66-0.97; minus quadratic: signal 0.0105 (n = 4), 0.013, 0.015, 0.014, 0.016 (x, n = 4-12), y 0.0096, 0.011, 0.012, 0.015, 0.012,
r 0.83 -> 0.4-0.7 (x: 0.83, 0.76, 0.58, 0.49, 0.69; y: 0.68, 0.47, 0.53, 0.56, 0.40). So v_rel carries a large reproducible quadratic-like part (about 3 %) and a poorly reproducible non-quadratic part
(~1-1.6 %, partly noise: cell errors are 1-3 %). Details in `part1.json` (keys `vrel_*`).

Plane coefficients (tan per 100 mm), event-level fit w = a + b U + c W + v t, errors statistical only (they ignore the non-planar structure, which is strongly correlated in space):
| run, view | a | b = d tau/dU | c = d tau/dW |
|---|---|---|---|
| run 1 x | -0.0019 | 0.0105 +- 0.0003 | 0.0004 +- 0.0003 |
| run 1 y | -0.0030 | -0.0002 +- 0.0002 | 0.0106 +- 0.0003 |
| run 2 x | -0.0020 | 0.0107 +- 0.0002 | -0.0003 +- 0.0002 |
| run 2 y | -0.0031 | -0.0004 +- 0.0001 | 0.0104 +- 0.0002 |
Cell-map plane fits (weighted, errors scaled by sqrt(chi2/dof), chi2/dof = 17-138 because of the structure), n = 8: x: b 0.0100 / 0.0104 +- 0.0011, c 0.0001 / -0.0002 +- 0.0011;
y: b -0.0001 / 0.0000 +- 0.0010, c 0.0105 / 0.0098 +- 0.0012 (run 1 / run 2). The two views have the same own-axis curvature (0.0105 per 100 mm): an isotropic bowl, tau = 0.0105 (U, W).

## Part 2 - curl test

### (a) plane level
Required: c_x = d tau_x/dW equals b_y = d tau_y/dU. From the event-level planes above: c_x - b_y = +0.0006 +- 0.0004 (run 1, 1.7 sigma), +0.0001 +- 0.0002 (run 2, 0.4 sigma).
From the cell planes (n = 8) c_x - b_y = +0.0002 (run 1), -0.0001 (run 2), with event-bootstrap std 0.0008 / 0.0003 and with the structure-inflated cell-plane errors ~0.0015: < 0.3 sigma.
Both are 1-2 orders below the own-axis gradient 0.0105 (1.0 +- 3 %). n = 4: run 2 gives c_x - b_y = -0.0010 +- 0.0003 (bootstrap; 3.4 sigma) but the bootstrap does not include the model error of the plane on
16 cells; n = 6, 10, 12: <= 1.4 sigma (table `agg.json`). Caveat: for a non-quadratic Phi, independent weighted plane fits of the two views need not give equal c_x and b_y on a 340 x 302 mm rectangle (plane-fit leakage), so the plane-level test is only a consistency check; the fine-scale test (b, c) is the stronger one.

### (b) finite-difference curl test (plane-subtracted maps, corners, Cx(run a) against Cy(run b != a)); event-bootstrap errors (200 replicas)
| n | corners | r_xx (Cx 1 vs 2) | r_yy | r_xy (mean of two pairings) | curl-free fraction | slope Cy = s Cx (errors on both, through 0) | cov-slope sqrt(cyy/cxx) | rms Cx / Cy (1e-3 per 100 mm) | control slope (pure gradient) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 9 | 0.991 +- 0.022 | 0.992 +- 0.008 | 0.924 +- 0.017 | 0.933 +- 0.016 | 1.46 +- 0.08 | 1.41 +- 0.08 | 5 / 7 | 1.56 |
| 6 | 25 | 0.988 +- 0.023 | 0.986 +- 0.011 | 0.948 +- 0.014 | 0.961 +- 0.012 | 1.26 +- 0.05 | 1.24 +- 0.06 | 10 / 12 | 1.22 |
| 8 | 47 | 0.940 +- 0.038 | 0.961 +- 0.019 | 0.916 +- 0.018 | 0.964 +- 0.018 | 1.15 +- 0.04 | 1.11 +- 0.05 | 14 / 15 | 1.12 |
| 10 | 59 | 0.927 +- 0.040 | 0.968 +- 0.021 | 0.950 +- 0.024 | 1.002 +- 0.021 | 1.07 +- 0.04 | 1.08 +- 0.04 | 17 / 18 | 1.09 |
| 12 | 53 | 0.899 +- 0.059 | 0.953 +- 0.031 | 0.928 +- 0.031 | 1.002 +- 0.028 | 1.14 +- 0.06 | 1.14 +- 0.06 | 18 / 21 | 1.04 (circular) |

Expected slope 1: the measured slope exceeds 1 at coarse n (1.46, 1.26, 1.15). This is **not** a non-gradient signal: pushing a noise-free, pure gradient field (spline of the n = 12 Phi) through the same cell averaging, plane subtraction and finite
differences gives the same slopes (1.56, 1.22, 1.12, 1.09, 1.04); the excess is the method's bias (the plane subtraction acts differently on the two views). At n = 10, 12 the slope is 1.07 +- 0.04 and 1.14 +- 0.06, about 1.5-2.5 sigma from 1 and within 0.05-0.1 of the control.
The curl-free fraction (r_xy against the noise ceiling) is 0.93-1.00 with errors of 0.02-0.03; the corners overlap (shared cells), so the bootstrap over events, which includes this, is the number to use, not a Fisher error.
Figure `fig_curl_scatter.png`.

### (c) Helmholtz decomposition (weighted least squares, Phi on the (n+1)^2 corners; null modes (the constant and the checkerboard) removed by minimum norm, so dof = data - rank)
| n | maps | chi2/dof both runs (dof) | run 1 / run 2 alone | solenoidal cross-cov (run 1 x run 2), 1e-6 | control floor (noise-free pure gradient) | total cross-cov, 1e-6 | solenoidal fraction of the variance |
|---|---|---|---|---|---|---|---|
| 4 | plane-subtr. | 5.9 +- 0.8 (41) | 7.1 / 16.3 | 1.62 +- 0.28 | 1.77 | 27 | 6.4 +- 1.0 % (floor 6.5 %) |
| 6 | plane-subtr. | 2.65 +- 0.42 (97) | 2.9 / 5.3 | 1.28 +- 0.29 | 1.73 | 46 | 2.8 +- 0.6 % (floor 3.6 %) |
| 8 | plane-subtr. | 2.02 +- 0.29 (173) | 1.85 / 3.5 | 1.24 +- 0.33 | 1.10 | 52 | 2.4 +- 0.6 % (floor 1.9 %) |
| 10 | plane-subtr. | 1.50 +- 0.37 (241) | 0.93 / 1.67 | 0.06 +- 0.26 | 0.35 | 54 | 0.1 +- 0.5 % (floor 0.6 %) |
| 12 | plane-subtr. | 1.62 +- 0.35 (289) | 0.94 / 1.47 | 0.13 +- 0.22 | 0.02 | 56 | 0.2 +- 0.4 % (floor 0.03 %) |
Full (not plane-subtracted) maps give the same numbers within 10 % (`agg.json`, keys `none_*`). Expected noise rms of the solenoidal residual 0.6-1.4e-3, observed 1.4-1.7e-3 (n = 4-12): the chi2/dof above 1 (up to 3.5 for run 2 alone at n = 8) mostly
means the line-fit cell errors are 1.2-1.9 times too small for run 2 (or a small solenoidal part), but the noise-free cross-covariance, which does not depend on the errors, shows
no excess over the floor of the method at any n. Excess at n = 8: 0.14 +- 0.33e-6, 2 sigma limit < 0.8e-6, i.e. a solenoidal rms < 0.9e-3 tan (< 0.05 deg), < 1.5 % of the variance. The test cannot see a solenoidal part below that level.

### (d) Phi (mm), the "relief (h + c)/2" of the long-wavelength limit
Runs combined, non-quadratic part (a quadratic fitted over the corners and removed): n = 8: rms 0.40 mm, peak-to-peak 1.69 mm, statistical error per corner 0.034 mm (max 0.10); n = 12: rms 0.35 mm, p-p 1.56 mm, error 0.035 mm (max 0.10).
Run 1 against run 2 (independent maps): r = 0.964 (n = 8), 0.942 (n = 12), cross-run signal rms 0.38 mm (n = 8), 0.31 mm (n = 12). The quadratic part (mm, U, W in mm from the centre): UU 4.4e-5, WW 3.3e-5, UW ~0 (n = 8) = the bowl of tau.
Map (`fig_phi.png`): the non-quadratic relief is dominated by the two long edges y = +-150 mm, where it reaches +-1 mm and alternates along x with a period of 100-150 mm (high at x ~ -50 and +/-20..40, low at the x edges and near +100), against a flat band (|rel| < 0.25 mm) for |y| < 100 mm.
Its amplitude near the edges is the least constrained (fewest events) but it is reproduced by both runs. The 2-D map has no obvious ridge along a connector boundary.
Periodicity (`fig_period_scan.png`, `fig_fold50.png`, `per.json`, `fold.json`): an event-level scan of a sinusoid (period 15-150 mm, after removing a degree-4 polynomial in the coordinate and a quadratic in the other) for each view and coordinate. tau_x against x shows
coherent (same amplitude and phase in both runs) power at periods of ~73 and ~99 mm (amplitudes 1.6e-3 and 2.2e-3 in tan) and rising amplitude at > 120 mm; at 49.9 mm the amplitude is 1.6e-3 (run 1) / 0.9e-3 (run 2), phase difference 0.43 rad, the same size as its neighbours at 40-60 mm, at the 70th percentile of the coherent amplitude between 20 and 100 mm.
tau_y against y: 0.17e-3 / 0.11e-3 at 49.9 mm (noise 0.3e-3), 15th-22nd percentile. Folding the residual on 49.9 mm (10 phase bins): x chi2 against flat = 29 / 27 for 9 dof (run 1 / run 2), y 10 / 2.5. So the x view has non-flat phase structure but no specific 50 mm line; the y view does not.
**No evidence for a 50 mm connector periodicity.**

## Part 3 - systematics on the offset maps (`part3.json`)
Own-axis plane gradients b_x (tau_x vs U) and c_y (tau_y vs W), 1e-3 tan per 100 mm (run 1 / run 2); sig = cross-run signal rms (1e-3); n = 8, errors ~1.1 (gradients), ~0.9 (sig; cov error 7-10e-6):
| variant | b_x | c_y | sig_x | sig_y | corr. of residual maps with base |
|---|---|---|---|---|---|
| base | 10.0 / 10.4 | 10.5 / 9.8 | 6.78 | 7.62 | 1 |
| cubic term | 10.0 / 10.4 | 10.4 / 9.8 | 6.69 | 7.62 | 0.998-1.000 (rms diff 0.2-0.5e-3) |
| tan_ref symmetrised per cell | 10.2 / 10.4 | 10.4 / 9.8 | 6.70 | 7.61 | 0.996-1.000 (rms diff 0.3-0.7e-3) |
| \|tan_ref\| <= 0.3 | 10.1 / 10.3 | 10.5 / 9.8 | 6.74 | 7.63 | 0.998-1.000 (0.1-0.4e-3) |
| index from reconstructed mid-gap position | 10.0 / 10.4 | 10.4 / 9.8 | 6.75 | 7.59 | 0.998-0.999 (full tau 0.999-1.000) |
n = 4: all variants change b_x, c_y by <= 0.1e-3 and sig by <= 0.03e-3 (base sig_x 4.06, sig_y 5.86). Run 1 (reference-indexed) against run 2 (reconstructed-indexed) residuals: r = 0.966 (x), 0.968 (y), the same as reference vs reference (0.968, 0.967).
Neither the acceptance leak nor the choice of position index creates or removes any of the structure. The cubic term is not significant in the cells.

## Caveats
- Everything here is a statement about the two golden runs of one chamber, mounted identically (top slot), tracked by the same M3 telescope. A reference-side effect (M3 slope error varying with position) would be reproducible and would not have to be curl-free either; the result that it is curl-free argues against such an effect and against single-view electronics, but cannot separate "tilt of the drift lines" from any other gradient of a scalar.
- Corner-based correlations share cells (neighbouring corners are correlated); I used the event bootstrap (200 replicas, rows of the table resampled, both views together) for all quoted errors on r, slopes and solenoidal numbers. Cell bootstrap errors are used in Part 1 (cells assumed independent; the high-pass and sig errors are therefore approximate).
- The cell errors (line fit) look 1.2-1.9 times too small for run 2 (chi2/dof of the gradient description 1.5-3.5 alone); statistics based on cross-run products do not depend on this.
- The control field was built from the n = 12 Phi, so the n = 12 control is circular; n = 4-10 are fair.
- n = 16 has 19 valid cells only; do not use it for the scale argument.
- The Phi absolute level (the constant) and the checkerboard mode are undetermined; the quadratic part of Phi is not discussed here (it equals the plane of tau).
