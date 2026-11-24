# Audit test B: fine maps of T, v, hp, tilt, and the two-surface relations (2026-11-24)

Scripts (all in `rerun/paper1_gap/audit/`): `b_01_maps.py` (cell maps, n = 4, 6, 8, 10, both runs, U in [-1.7,1.7], W in [-1.51,1.51], cells indexed by the
mid-gap track position), `b_lib.py`, `b_02_part1.py`, `b_03_parts2_5.py`, `b_04_addenda.py`. Logs and json: this directory (`part1_*.json`, `part2_5.log`,
`part2_5.json`, `addenda.log`, `addenda.json`). Figures: `fig_fine_maps_n8.png`, `fig_scatter_n8.png`, `fig_timeblocks.png`.
Run 1 has 4,654 rows (3,220 with a colsum T), run 2 14,067 (9,318). At n = 8 that is about 50 (run 1) and 150 (run 2) events per cell for T, so the cell errors are 1-3 %.

Conventions. "Fine" = residual after the weighted quadratic surface (plane for w0, tau = w0 / chamber v). "Tot" = the map minus its mean. All slopes and
covariances between the runs are the cross-run form k = [cov(y1, x2) + cov(y2, x1)] / [2 cov(x1, x2)]: noise in a cell is independent between runs and drops out.
Errors are bootstrap over cells (3000 draws); the 16th-84th percentile range is given where the distribution is skewed (denominators close to zero).
Cells with fewer than 20 T events are dropped (n = 8: 57-59 of 64 remain; for the tilt tests the gradient also removes the edge columns, 40-43 cells).

## Part 1. Are the fine structures reproducible?

Residual rms, mean statistical error, run1 x run2 Pearson r of the residual maps, noise-free signal rms sqrt(max(cov, 0)). (full-map r is the r before removing the surface.)

| n | map | rms (run 1 / 2) | mean err (1 / 2) | r full | r residual | signal rms |
|---|---|---|---|---|---|---|
| 4 | T (ns) | 15.7 / 17.6 | 9.4 / 5.3 | 0.97 | 0.88 ± 0.05 | 16.2 ± 2.6 |
| 6 | T | 20.0 / 20.3 | 14.1 / 8.0 | 0.90 | 0.73 ± 0.09 | 17.4 ± 1.9 |
| 8 | T | 27.8 / 23.5 | 17.9 / 11.0 | 0.88 | 0.72 ± 0.07 | 21.9 ± 2.2 (2.7 %) |
| 10 | T | 26.8 / 25.4 | 20.7 / 13.4 | 0.84 | 0.62 ± 0.07 | 20.4 ± 2.0 |
| 4 | v_x (um/ns) | 0.43 / 0.44 | 0.27 / 0.16 | 0.97 | 0.83 ± 0.06 | 0.41 ± 0.05 |
| 6 | v_x | 0.63 / 0.50 | 0.41 / 0.24 | 0.94 | 0.76 ± 0.07 | 0.50 ± 0.05 |
| 8 | v_x | 0.77 / 0.63 | 0.49 / 0.33 | 0.85 | 0.54 ± 0.15 | 0.51 ± 0.08 (1.4 %) |
| 10 | v_x | 0.75 / 0.71 | 0.47 / 0.40 | 0.85 | 0.58 ± 0.07 | 0.51 ± 0.05 |
| 4 | v_y | 0.42 / 0.43 | 0.26 / 0.15 | 0.92 | 0.68 ± 0.14 | 0.36 ± 0.08 |
| 8 | v_y | 0.66 / 0.63 | 0.43 / 0.29 | 0.84 | 0.55 ± 0.11 | 0.46 ± 0.06 (1.3 %) |
| 10 | v_y | 0.68 / 0.71 | 0.45 / 0.35 | 0.81 | 0.61 ± 0.09 | 0.53 ± 0.07 |
| 4 | hp_x (mm) | 0.25 / 0.30 | 0.24 / 0.14 | 0.89 | 0.60 ± 0.15 | 0.22 ± 0.06 |
| 6 | hp_x | 0.55 / 0.38 | 0.38 / 0.23 | 0.69 | 0.28 ± 0.27 | 0.24 ± 0.13 |
| 8 | hp_x | 0.54 / 0.52 | 0.47 / 0.31 | 0.83 | 0.52 ± 0.10 | 0.35 ± 0.06 |
| 10 | hp_x | 0.58 / 0.60 | 0.50 / 0.39 | 0.70 | 0.19 ± 0.11 | 0.21 ± 0.08 |
| 4 | hp_y | 0.18 / 0.17 | 0.28 / 0.15 | 0.92 | 0.39 ± 0.12 | 0.11 ± 0.02 |
| 8 | hp_y | 0.62 / 0.46 | 0.54 / 0.32 | 0.70 | 0.11 ± 0.16 | 0.18 ± 0.11 |
| 10 | hp_y | 0.62 / 0.50 | 0.58 / 0.39 | 0.67 | 0.02 ± 0.12 | 0.08 ± 0.09 |
| 4 | w0_x (um/ns, plane removed) | 0.16 / 0.15 | 0.035 / 0.021 | 1.00 | 0.98 ± 0.02 | 0.157 ± 0.020 |
| 8 | w0_x | 0.26 / 0.26 | 0.061 / 0.039 | 0.99 | 0.97 ± 0.01 | 0.257 ± 0.020 |
| 8 | w0_y | 0.29 / 0.31 | 0.050 / 0.032 | 0.99 | 0.97 ± 0.01 | 0.277 ± 0.024 |
| 10 | w0_x / w0_y | 0.29 / 0.25 | 0.066 / 0.056 | 0.98 / 0.99 | 0.96 / 0.93 | 0.28 / 0.25 |

Answers.
- **T, v_x, v_y and w0 have reproducible fine structure at every n.** At n = 8 the signal rms of fine T is 2.7 % (22 ns), of v about 1.3-1.4 %, and of w0 about 0.26 um/ns
  (0.7 % of v, i.e. a tilt of 7 mrad rms). The residual rms is larger than the statistical error in all cases. The w0 fine map is by far the cleanest (r = 0.97).
- **hp (the position route) is marginal.** hp_x has a reproducible fine part of 0.2-0.35 mm only at n = 4 and 8 (r 0.5-0.6); at n = 6 and 10 it is not significant. hp_y has no significant fine structure at n >= 6 (r 0.02-0.15, signal 0.08-0.18 mm, errors as large as the signal). The mesh-height fine map is not measured well enough to carry a test by itself.
- Caveat on n = 4: removing 6 surface parameters from 16 cells removes most of the structure, so r residual at n = 4 is partly an artefact of the fit; n = 8 is the cleanest "fine" level.

### Charge-time proxies (free in every pass-1 table)

Cross-run, n = 8, proxy averaged over X and Y. "r_nf" is the noise-free correlation cov(proxy1, T2) / sqrt(cov(p1, p2) cov(T1, T2)). Slope = d ln T / d ln proxy (cross-run, residual maps in relative units).

| proxy | r (proxy run A, T run B), full / residual | r_nf resid | slope d ln T / d ln proxy, full | residual |
|---|---|---|---|---|
| q_u90 | 0.92 / 0.81 | 0.98 | 0.90 ± 0.03 | 0.92 ± 0.04 |
| q_uend | 0.89 / 0.72 | 0.93 | 1.12 ± 0.04 | 0.92 ± 0.08 |
| q_u50 | 0.83 / 0.57 | 1.02 | 0.86 ± 0.06 | 1.2 ± 0.4 |

Fine-structure reproducibility of the proxies itself (n = 8): u90 r = 0.91 (x), 0.93 (y), signal 18-19 ns; uend 0.89 (x), 0.67 (y); u50 only 0.35-0.45.
**q_u90 is a usable proxy of the colsum T**, in the full map and in the residual map. Its fine map correlates with the fine T map to r_nf = 0.98 (the noise-free value, within the errors consistent with 1), with a slope of 0.9 in d ln,
and its own run-to-run reproducibility is as good as T (r 0.9). `q_uend` is slightly worse in y (broad tail), `q_u50` is not usable for fine structure. For other chambers and rows, use u90 (X and Y averaged). What I did not test: the proxy against T on another chamber, or whether the u90 slope of 0.9 is the same there.

## Part 2. Gap or gas with noise-free estimators

slope = d ln v / d ln T, cross-run. Expected: gap change at fixed voltage -0.405 (beta = 0.68); gas change -1. beta = -s / (1 + s).
The audited number (4 x 4, within-run) was -0.50 ± 0.02. Table: slope ± bootstrap error (beta from the 16-84 % range).

| n | v | total | smooth (quadratic) | fine |
|---|---|---|---|---|
| 4 | v_x | -0.69 ± 0.07 (beta 1.8-3.0) | -0.72 ± 0.04 | -0.56 ± 0.08 (beta 1.0-1.8) |
| 4 | v_y | -0.40 ± 0.08 (0.5-0.9) | -0.39 ± 0.07 (0.5-0.8) | -0.53 ± 0.09 (0.8-1.6) |
| 6 | v_x | -0.70 ± 0.05 | -0.76 ± 0.03 | -0.60 ± 0.09 (1.1-2.2) |
| 6 | v_y | -0.47 ± 0.06 (0.7-1.1) | -0.46 ± 0.04 (0.7-1.0) | -0.46 ± 0.10 (0.6-1.3) |
| 8 | v_x | -0.61 ± 0.06 (1.3-2.0) | -0.74 ± 0.02 (2.6-3.0) | -0.44 ± 0.10 (0.5-1.2) |
| 8 | v_y | -0.44 ± 0.06 (0.6-1.0) | -0.45 ± 0.03 (0.7-0.9) | -0.38 ± 0.07 (0.45-0.84) |
| 8 | v mean of x, y | -0.52 ± 0.05 (0.9-1.3) | -0.59 ± 0.02 | -0.41 ± 0.08 (0.5-0.95) |

Findings.
- Within-run slopes (noise biased towards 0) are 0.0-0.15 smaller in magnitude than the cross-run ones, as expected.
- **The fine part behaves like a gap change at n = 8 and perhaps n = 6** (v_x -0.44 ± 0.10, v_y -0.38 ± 0.07, mean -0.41 ± 0.08 against -0.405). At n = 4 the fine slopes are -0.5 to -0.6 (beta 1.0-1.5), and the gas value -1 is excluded for the fine part at every n (by 4-8 sigma).
- **The smooth part does not behave like one number for both views.** The smooth v_y slope is -0.39 to -0.46 (gap-like, beta 0.6-0.85). The smooth v_x slope is -0.72 to -0.76 with an error of 0.02-0.04, which is neither the gap value (-0.405, excluded at 8-20 sigma) nor the gas value (-1, excluded at 7-16 sigma). Something besides a pure gap change moves v_x on long wavelengths (v_x varies by an extra 25-35 % of what the gap would give). The audited -0.50 is the average of two different slopes in the two views.
- Therefore the claim "it is the gap, not the gas" holds for the fine part and for the Y view at all scales; it is **not** supported for the smooth X-view velocity map. A possible contribution is the long-wavelength tilt itself (a tilt and a curvature of tau along x acts on the X ladder) but I did not establish a cause (Part 4 finds no focusing signal that could do it).
- beta from these slopes: fine part beta = 0.69 (0.51-0.95) at n = 8, consistent with the Magboltz 0.68. The audited joint fit gave 1.0 ± 0.2, which agrees with the n = 4/6 numbers.

### Tan range

Chamber-wide ladder v (um/ns), range |tan| <= 0.6 / <= 0.3 / 0.1-0.6 (the data reach only about |tan| 0.38, so 0.6 and 0.3 hold 96-99 % of the same tracks):
- run 1 x: 37.035 ± 0.068 / 37.009 ± 0.073 / 36.998 ± 0.071; y: 36.504 / 36.509 / 36.437 (± 0.07).
- run 2 x: 37.739 ± 0.041 / 37.748 ± 0.044 / 37.708 ± 0.042; y: 37.139 / 37.182 / 37.039 (± 0.04).
Differences between ranges are at most 0.1 um/ns (0.3 %), consistent with noise; the ladder is linear enough.
Map correlation (4x4 and finer): r(map 0.6, map 0.3) = 0.99-1.00, r(map 0.6, map 0.1-0.6) = 0.97-0.99; the rms difference of the maps is 0.04-0.2 um/ns, below the cell error of the narrower range (0.16-0.5). The run-to-run reproducible part of the range difference is 0.0-0.11 um/ns (signal rms; r between -0.56 and +0.58, noise-compatible), against a v map signal of 0.4-0.5. **A non-linear ladder with a position-dependent tan mix does not fake the v map** (it could at most create about 10-20 % of its amplitude, and no reproducible pattern is seen).

## Part 3. Which electrode carries the structure

tau = w0 / v (dimensionless); delta g in mm from delta ln T / (1 + beta) x 28.2 (beta = 0.68, 1.0) or from ln(v T) (beta free, v = mean of v_x, v_y).
k from tau = k d(delta g)/dx (central differences between cells; cathode +0.5, mesh -0.5), cross-run. Attenuation of short wavelengths (0.8 at 100 mm, 0.54 at 50 mm) lowers |k| at n = 8 (cells of 42 mm, wavelength of the finest visible mode about 85 mm): the expected value for either electrode is therefore about +-0.4.

| n | part | view | dg from T (0.68) | dg from T (1.0) | dg from v T |
|---|---|---|---|---|---|
| 6 | fine | x | +0.62 ± 0.25 | +0.74 ± 0.29 | +0.84 (range 0.41-1.44) |
| 6 | fine | y | +0.59 ± 0.21 | +0.70 ± 0.25 | +0.87 (0.5-1.45) |
| 8 | fine | x | **+0.67 ± 0.12** | +0.80 ± 0.14 | +0.56 ± 0.25 (0.40-0.78) |
| 8 | fine | y | **+0.46 ± 0.16** | +0.55 ± 0.19 | +0.32 (0.12-0.76) |
| 4 | fine | x, y | +0.35, +0.43 (errors 0.65, only 8 cells) | | |
| 8 | total | x | +0.80 ± 0.16 | +0.96 ± 0.19 | +0.51 ± 0.34 |
| 8 | total | y | -0.01 ± 0.11 | -0.01 ± 0.14 | -0.01 ± 0.12 |

Direct tests (fine, n = 8; mesh -1 / cathode 0):
- hp (mean of x and y) against delta g: +0.09 ± 0.14 (from T), +0.17 ± 0.30 (v T); n = 6: -0.09 ± 0.19, -0.08 ± 0.9. **The mesh value -1 is excluded at 6 sigma (n = 8, T); the cathode value 0 fits.** The total maps (smooth + fine) give -0.2 to -0.3: the smooth dish is a mesh effect (see below).
- tau against grad hp (n = 8, x: +0.85 ± 2.1, y: +0.8 ± 54) and the two-regressor fit (tau = a grad h + b grad dg): a = +0.5 ± 8, b = +0.4 ± 6: **not constrained**; the hp map is too noisy (r residual 0.1-0.5) for a gradient to be formed from it.

(i) Quadratic parts (analytic, cell-refit bootstrap). Gap curvature G from the fitted surface of ln T (beta 0.68), mm per (100 mm)^2; tilt slope = d tau / dU from the w0 plane, k = 100 (d tau / dU) / G:

| n | G_uu run 1 / 2 | G_ww run 1 / 2 | tilt slope x, y (mm / (100 mm)^2) | k_x | k_y |
|---|---|---|---|---|---|
| 4 | +0.05 ± 0.5 / +0.11 ± 0.4 | -1.44 / -1.42 (± 0.5) | 0.98-1.00, 1.03-1.04 | undefined (G_uu = 0) | -0.72, -0.73 |
| 6 | +0.31 ± 0.21 / +0.18 ± 0.23 | -1.39 / -1.29 (± 0.3) | 1.00-1.01, 0.98-1.04 | +3.2, +5.7 | -0.75, -0.76 |
| 8 | +0.35 ± 0.25 / +0.20 ± 0.19 | -1.41 / -1.43 (± 0.25) | 1.00-1.04, 0.98-1.06 | +2.8, +5.1 | -0.76, -0.69 |

(beta = 1.0: G_ww -1.2 and k_y -0.8 to -0.9. v T map: G_ww -1.0, k_y -0.85 to -1.1; k_x diverges because G_uu is about 0.) This reproduces the audited joint fit (G_uu +0.28, G_ww -1.09; tilt slopes 0.92 / 0.77; k_x 3.3, k_y -0.7).
Along y the smooth part has k_y = -0.7 to -0.9 (a mesh-like dish: the board bends, the cathode stays flat; k = -0.5 would be a pure mesh, the excess is within the sizeable error of G_ww of 0.25-0.5 and the T-to-g conversion). Along x the gap curvature is only +0.2 ± 0.2, so the smooth tilt (1.0 mm per (100 mm)^2) is carried by both electrodes together (k >= 3 means the cathode follows the board), as the joint fit found.

**Verdict for Part 3.** The two scales behave differently.
- The smooth dish (quadratic part) sits on the mesh/board along y (k_y -0.75 ± about 0.15) and on both electrodes along x.
- **The fine structure sits on the cathode** (or at least not on the mesh): k = +0.67 ± 0.12 (x) and +0.46 ± 0.16 (y) at n = 8 from T, +0.6 ± 0.2 at n = 6, and the mesh-height map shows no -1 slope against the fine gap. Mesh (-0.5) is excluded at 9 sigma (x) and 6 sigma (y) at n = 8, but the values for the v T map are weaker (+0.56 ± 0.25, +0.3).
- That sign explains why the total y regression is 0 (-0.01 ± 0.11): the smooth part has k = -0.75 and the fine part +0.5, and they cancel in the y variance. For x the total is +0.8.
- Caveats. (a) The fine gap map is built from ln T with the beta of the gap. A cell-level correlation of T with the signal amplitude would also create a correlation with the tilt if the gain map looks like the tilt map (Addendum 6: the gain map correlates with w0_x at r = 0.2-0.4, so some of the x result may come from gain; per-cell correction changes T by 0.2-1 % only, see below). (b) The finite-difference gradient attenuates by an unknown amount; the value k = 0.67 at n = 8 is above 0.5 but with beta = 0.68 and noise in the denominator the error is +-0.12 and the result is within 1.5 sigma of 0.5. (c) The data cannot decide a mixture of both electrodes, only that mesh-only is excluded.

## Part 4. Focusing

Derivation. Let tau(x) = tau0 + a x (the audit sign convention: a drift line leaves the mesh going up along dx/dz = -tau). Charge uniform in z, arrival time proportional to z (path length differences ignored).
A track at mesh position x_c with slope t deposits charge at x_c + t z; following the line down to the mesh, the landing position is x_L(z) = x_c + (t + tau(x_c)) z + a (t + tau0/2) z^2.
The straight-line fit of x_L against time over z in [0, g] has the slope of the z term plus g times the coefficient of z^2 (the linear regression of z^2 on a uniform z has slope g).
It is (t + tau(x_c)) + a g (t + tau0/2). Cells are indexed by the mid-gap position x_m = x_c + t g / 2, so tau(x_c) = tau(x_m) - a g t/2. The coefficient of t is then 1 + a g - a g / 2:

    v_fit = v (1 + (1/2) g d tau / dx)         f = 1/2 exactly (f = 1 if the cells were indexed at the mesh; the audit uses mid-gap, 14 mm).

Expected size: the smooth tilt slope is 0.0100 per 100 mm, so (1/2) g dtau/dx = 0.14 %, below the 1.4 % error floor. Only the fine tilt structure can matter: the rms of g d tau / dx for the fine part is 0.12-0.4 % (so f = 0.5 adds at most 0.06-0.2 % rms).
Test: R = ln v + beta / (1 + beta) ln T (v after the gap term), regressed on X = g d tau / dx (cross-run, finite differences). Results (beta = 0.68; beta = 1.0 in `part2_5.log`):

| n | part | f_x | f_y | removed at f = 0.5 (x / y) |
|---|---|---|---|---|
| 6 | fine | +0.54 ± 0.73 | +0.45 ± 0.57 | 4 % / 8 % of the reproducible R variance |
| 8 | fine | -0.08 ± 0.46 | +0.53 ± 0.43 | 6 % / 7 % |
| 6 | total | +0.42 ± 1.26 | +1.73 ± 0.88 | 1 % / 1 % |
| 8 | total | -0.26 ± 0.65 | +1.51 ± 0.60 | 2 % / 2 % |

**Nothing is seen.** All f values are consistent with 0.5 and with 0 within 1-2 sigma, except the total y at n = 8 (+1.5 ± 0.6, 2.5 sigma above 0.5, and it flips to +1.0 ± 0.7 for beta = 1.0; the x total goes to -0.63 ± 0.57). The errors (0.4-0.7) are too large to measure f. Even a true f = 0.5 would remove only 2-8 % of the reproducible variance of R, which has a signal rms of 0.5-1.5 %. **Focusing does not explain the extra variation of v** (including the smooth v_x velocity map of Part 2), and the data are too noisy to test it.

## Part 5. One model line

Fraction of the reproducible (run-cross) variance of each fine map explained by the others (cross-run regression; one common gap field plus one mesh-height field).

| map | explained by | n = 6 | n = 8 |
|---|---|---|---|
| T | v_x, v_y | 0.91 [0.74-1.09] | 0.70 [0.52-1.01] |
| v_x | T | 0.90 ± 0.14 | 0.68 ± 0.31 |
| v_y | T | 0.79 [0.52-1.2] | 0.63 ± 0.20 |
| v_x | T and v_y | 0.97 | 0.82 [0.69-1.11] |
| v_y | T and v_x | 0.95 | 0.79 [0.67-1.06] |
| hp_x | hp_y | 1.0 (poorly known) | 0.52 [-0.07-0.98] |
| hp_x | T (gap) | 0.36 [0.0-0.8] | 0.00 [0.00-0.08] |
| tau_x | grad h + grad(dg)/2 (fixed model) | 0.92 ± 0.09 | 0.34 ± 0.22 |
| tau_y | same | 1.04 ± 0.10 | 0.17 ± 0.47 |
| tau_x | grad h alone / grad dg alone | 0.82 / 0.28 | 0.29 / 0.44 |

Statement per map.
- **T and v (one gap field).** The fine T, v_x and v_y maps share one field: 70-90 % of the reproducible variance of each is explained by the others (n = 8: 63-82 %, n = 6: 79-97 %); the remainder (10-35 %) is of the size of the errors (the 16-84 % ranges cross 1). No evidence for reproducible structure outside the one gap field.
- **hp.** hp_x and hp_y are consistent with one mesh field (explained 52 % at n = 8, large errors), and the mesh field is uncorrelated with the gap field (slope +0.1 ± 0.14, explained 0 % of hp_x by T). That is a statement about the mesh height field, not a test of consistency. hp is too noisy at fine scale to constrain delta h.
- **tau.** At n = 6 the fixed two-surface relation tau = grad h + grad(dg)/2 (h from hp, dg from v T) explains 92-100 % of the tilt variance; at n = 8 only 17-34 % with errors of 0.2-0.5, and most of that comes from the grad h term. The n = 6 result is not reproduced at n = 8, and both are dominated by the noise in hp (the h gradient is the least known input). The honest statement is: **the data do not contradict a two-surface description of the tilt, and they cannot confirm it at n = 8**. A free two-regressor fit has errors of order 2-50 and fixes nothing.
- Overall: **the T, v maps are self-consistent with one gap field; the tilt maps (reproducible at r = 0.97, signal 0.26 um/ns) are explained by the gap gradient only to 25-45 % (grad dg alone, n = 8), and the rest of the tilt structure is not predicted by T or v.** The mesh-height map is too noisy to say whether it carries the rest.

## Addendum 6. Gain confound of the T map

- Event-level slope of T on ln q: 22.6 / 18.8 ns (run 1 / 2) for the mean of the two views; per view x 18.2 / 15.4, y 26.8 / 21.3 ns. The per-view correction gives the same T map as the global one (T maps identical to 0.1 ns and the surface coefficients to 0.01 mm).
- Gain map (cell-mean ln q): rms over cells 0.06 (n = 4) to 0.09 (n = 8) (about 9 % of the amplitude), reproducible between runs (r 0.86 full, 0.80 residual, signal rms 0.074 at n = 8). A smooth part (U 0.02, W 0.03, W^2 0.04 per (100 mm) units in ln q) plus fine structure.
  Event-level slopes turn 0.09 in ln q into 0.09 x 20 ns = 1.8 ns = 0.2 % of T, i.e. negligible against the 2.7 % fine T structure.
- (a) Corrected T map against ln q map: the cross-run slope on cell-mean ln q is +0.4 ± 68 (n = 8), +24 ± 98 (n = 6), +53 ± 181 ns per ln q (n = 4): consistent with 0 and also with the event-level 20 ns, i.e. the cell data cannot say whether the global correction is sufficient or not. Raw cross-run r(T_amp, gain) is 0.00 to 0.08 over the full map; for the fine parts at n = 8 r = +0.26 (slope d ln T / d ln q = +0.13 ± 0.05, against the event-level +0.028): a hint that a **fine-scale** correlation of T with gain exists and is 5 times stronger than the event-level slope. It cannot be told whether this is a gain effect on T or a real correlation of gain with gap.
- (b) Redoing the T map with the cell-level slope (fine, +0.158 ± 0.053 per ln q at n = 8) changes the gap coefficients (mm of gap, beta 0.68; run 1 / run 2; n = 8): tilt along x G_u +0.54 / +0.51 -> +0.50 / +0.44 (-8 to -14 %); y slope G_w -0.30 / -0.34 -> -0.38 / -0.40; y dome G_ww -1.38 / -1.42 -> -1.53 / -1.58 (+11 %); G_uu +0.38 / +0.21 -> +0.43 / +0.30. For beta = 1.0 all values are 0.84-0.9 times these. With the full-map cell slope (+0.018 ± 0.087) nothing changes (differences < 0.04 mm). The fine T signal falls from 2.74 to 2.48 % (r 0.70 -> 0.67). So **the global event-level correction suffices for the smooth surfaces (dome and tilt unchanged within 0.1 mm, errors 0.2-0.3 mm); a cell-level correction would alter them by 10 %, within the errors**. The fine d ln v / d ln T becomes -0.44 / -0.40 (x / y) instead of -0.41 / -0.36: closer to the gap value.
- (c) v against ln q. Event level (high vs low charge halves in every 4 x 4 cell): d ln v / d ln q = +0.008 ± 0.006, +0.002 ± 0.006 (run 1 x, y), +0.004 ± 0.003, +0.012 ± 0.004 (run 2): v depends at most on 1 % per ln q, so the gain map (9 %) gives under 0.1 % in v. Cell level (cross-run, full): d ln v_x / d ln q = +0.04 ± 0.06, d ln v_y / d ln q = +0.10 ± 0.05 (n = 8); fine -0.04 ± 0.02 and -0.02 ± 0.03. The gain correlation with v is thus at most 0.1 x 0.09 = 0.9 % for v_y on long wavelengths (borderline against the 1.4 % floor), and negligible in the fine structure.
- The gain map looks like **neither the gap map nor clearly the tilt map**. Cross-run r(gain, ln T) is -0.02 to +0.07 on full maps (no similarity to the gap map; the gain W^2 coefficient is positive where the gap dome is negative), and r(gain, w0_x) = +0.28-0.36, r(gain, w0_y) = +0.18-0.40 (a gradient in gain along U and W similar to the tilt planes), but the fine part of w0_x correlates only at r = 0.2-0.3 and w0_y not at all (-0.1 to +0.15). The gain has a smooth gradient of about 2-3 % per 100 mm along both axes, the same direction as the tilt plane; a link (for example a drift field dependence of the gain from the same tilt) is possible but not established.

## Addendum 7. Time signature inside each run

Blocks of equal event number (8 for run 1, 10 for run 2) in `acq_tag`, `timestamp` order; block values in `addenda.log`, figure `fig_timeblocks.png`.
- Run 1: T 796-808 ns, v_x 36.8-37.35, v_y 36.16-36.80 (a rising v_y over the run, +1.6 % first to last block). Scatter of ln T over blocks 0.51 % against a mean statistical error of 0.82 %: **no resolved variation of T**. Slopes d ln v / d ln T: v_x -0.17 ± 0.54, v_y -0.92 ± 0.45 (OLS, bootstrap over blocks).
- Run 2: T 794 -> 783 ns (a steady fall of 1.4 % over the night), v_x 37.48 -> 37.81, v_y 37.00 -> 37.39. Scatter of ln T 0.71 % against an error of 0.53 %: a resolved drift. Slopes: v_x -0.38 ± 0.16 (Deming, accounting for noise in both: -0.59), v_y -0.06 ± 0.24 (Deming -0.36). Taking the first 3 and last 3 blocks (by hand, from the table): T -1.1 %, v_x +0.9 %, v_y +0.8 % (ratio -0.7 to -0.8, noisy).
- **Run 1 against run 2, chamber-wide:** d ln T = -1.65 ± 0.33 %, d ln v_x = +1.88 ± 0.21 %, d ln v_y = +1.72 ± 0.23 %, so d ln v / d ln T = **-1.14 ± 0.26** (x) and **-1.05 ± 0.25** (y). This is the gas-like value (-1), and differs from the gap value (-0.405 to -0.5) by 2.8 sigma. Mean ln q barely changes (10.755 -> 10.772, +1.7 %, which is 0.03 % of T by the event-level slope).
- Within a run the lever arm is small (T moves 0.5-1.4 %, with block errors 0.5-0.8 %), and the within-run slopes (-0.2 to -0.9, errors 0.2-0.5) can neither separate -0.4 from -1 nor contradict either. What the data say: the change between the two runs is gas-like (the two mountings are the same, the same slot), and the within-run drift in run 2 is intermediate (-0.4 to -0.8). The implication for the whole analysis: **the chamber-wide level differs between the runs in the gas direction**, while the spatial structure (Part 2 fine part) is gap-like. That is consistent with a slowly changing gas composition (about 1.7 % in v between the two days) plus a position-dependent gap.

## What this audit cannot say
- No test of hp at fine scale: the position route has errors of 0.3-0.5 mm per cell at n = 8, as large as the structure.
- The k = +0.5 result for the fine structure relies on beta (0.68 or 1.0) and on the cathode-like sign persisting in the v T map (weaker: +0.56 ± 0.25 and +0.3 for y).
- The smooth v_x slope of -0.74 is unexplained; nothing in Parts 3, 4 or Addendum 6 accounts for it.
- All errors are over cells (64 cells at n = 8, about 45 usable); the two runs share the same mounting, so a systematic of the chamber (not noise) is common to both and invisible to the cross-run estimators.
