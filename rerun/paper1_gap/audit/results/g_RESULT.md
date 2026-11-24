# Audit test G: does the bench fine angle map predict the n_TOF in-situ angle map? (2026-11-24)

Scripts (all new, `rerun/paper1_gap/audit/`): `g_common.py` (selection, frame mapping, cell fits), `g_compare.py` (cell-by-cell comparison), `g_direct.py`
(held-out sigma on the n_TOF A-C lines), `g_poly.py` (smooth surfaces against the fine shape), `g_bow.py` (quadratic level, mirror and shift scans),
`g_final.py` (figures and tables). Products here: `compare.json`, `direct_m{0,30}.csv`, `poly_m{0,30}.md`, `bow_level.csv`, `total_gradients.csv`,
`tables.md` (all numbers below), figures `fig1_maps.png` ... `fig4_bow_gradient.png`. Nothing was written to `ntof_pass1/` or the repo's stage 3.

## Verdict

* **Chamber A: yes, the bench fine map transfers, partly, with a larger amplitude at n_TOF.** On the 31,387 true beam-off A-C lines
  (`ana_r2`, 40 runs), held-out, the bench map (A-C-independent, measured against M3 in June) improves the single-view sigma by 10.8 % (x) and 7.3 % (y)
  as it is (unit amplitude), and by 12.7 % and 8.8 % with one free amplitude per view. The in-situ 8 x 8 map gives 14.6 % and 12.0 %. So the bench map
  recovers 61-74 % of the in-situ gain as it is and 73-87 % with a fitted amplitude (tilt amplitude alpha = 1.3-1.5, stable between parities to 0.1;
  the scale part transfers weakly, beta = 0.2-0.7). A smooth plane or quadratic surface gives nothing on A (x: 28.7 -> 29.0 / 27.1), so the gain is the
  fine, edge-localised structure, which the bench also has. The cell-by-cell correlation of the tilt maps is moderate: r = 0.52 +- 0.12 (x) and 0.51 +- 0.14 (y)
  with the bench taken to the active-area edge (0.20 +- 0.16 and 0.74 +- 0.07 with the standard 30 mm interior), against n_TOF noise ceilings of 0.97 and 0.83.
* **Chamber C: no.** The bench map does nothing at unit amplitude (x 32.5 -> 32.6, y 37.1 -> 39.8, i.e. worse), and 1.6 % / 2.1 % with a free amplitude
  (alpha = 0.3-0.6), against 16.2 % / 15.5 % for the in-situ map. The tilt correlation is -0.03 +- 0.13 (x) and 0.62 +- 0.08 (y) cell by cell, but the y
  correlation is carried by a few outer cells and does not turn into a held-out gain. What helps C at n_TOF is a smooth surface (quadratic: x 32.4 -> 29.9,
  y 37.1 -> 32.5, almost the whole in-situ gain), and the bench map adds little on top of it (quad + bench 29.4 / 31.3).
* **Where the distortion comes from.** For A the fine structure is a property of the chamber: it is reproduced on the bench between two runs on two days
  (r = 0.99 over 36 common cells), survives transport and the change of orientation, and shows up in the n_TOF cosmic lines at the same places, but its
  edge tilts are 1.3-1.5 times larger at n_TOF. What changes with the mounting is the smooth part: the quadratic bow block, which is a bench property of the
  horizontal board, is mostly absent for A at n_TOF (the residual gradient undoes 76 % of the block's gradient along y), and present at full size for C. This
  fits the README (the bow helps C and is neutral on A) and the picture that A's dish is gravity sag of a horizontal board, C's is not. For C the n_TOF
  residual is smooth and large-scale and has no bench counterpart that we can measure with the statistics we have.
* Consequence for use: the A bench map (more precise, 1 mrad per cell) could stand in for the in-situ A map only at about 60-85 % of its gain, so it is
  an independent confirmation of the in-situ map's existence and shape for A, not a replacement. For C the bench map must not be used.

## 1. Coordinate mapping (derived, not fitted)

The cell index of the n_TOF map is (x_local, y_local) in 50 mm cells, and the map residual is r = (global d_p/d_z of the track) - (joined A-C slope j), per
view p, with the slope taken from the unmapped v(t) angle (`rerun/ntof/improve/angle_map.py:clean_lines`, `line_residuals.py:build`, lines 70-71).

| quantity | bench chamber (det3 = A, det6 = C) | n_TOF local / global | source |
|---|---|---|---|
| x position | strip coordinate x_p0 [mm], 0..398.58 | x_local = -(x_p0 - 199.29) for A and C | `build_tracks.py:262`, `IN_PLANE_SIGN = -1` (line 66, measured 2026-08-20 on the target image, applied to the position only) |
| y position | y_p0 | y_local = +(y_p0 - 199.29) for A and C | `build_tracks.py:263`, `run145_target_imaging.py:97` (`IN_PLANE_SIGN_Y = +1`), COORDINATES.md section 2 (channel 0 at the bottom) |
| raw angle | tan = + d(strip coordinate)/dz_up, z_up = height above the readout (COORDINATES.md 4b; w rises with the M3 tangent on 245/247 pairs) | the same fit (x_tan_theta, y_tan_theta): d(strip)/d(distance from the strip plane). In `local_and_global` the track point at depth `gap` is x_local - tx gap, i.e. strip + tx gap, so d(strip)/d(depth) = + x_tan_theta | `build_tracks.py:265-271`, `geometry.py:129-135` (local z = -w) |
| x slope in the map | e_x = x_tan_theta - tan_ref_x | A and C: d_x/d_z = + tanx = + x_tan_theta (A looks along -z with R = identity, C along +z with R = 180 deg about y; the two sign flips cancel) | checked on the tables: slope of d_x/d_z against x_tan_theta is 1.000 (A), 0.958 (C, the v(t) factor) |
| y slope in the map | e_y = y_tan_theta - tan_ref_y | A: d_y/d_z = tany = **- y_tan_theta**; C: d_y/d_z = - tany = **+ y_tan_theta** (TAN_SIGN_Y = -1, `run145_target_imaging.py:98`, `build_tracks.py:266`; A and C differ because the depth axis points along -z for A and +z for C) | checked on the tables: slope of d_y/d_z against y_tan_theta is -0.994 (A), +0.960 (C) |
| map entries | a_bench(strip -> local), s_bench | a_ntof = SGN * a_bench with SGN = +1 (x, A and C), -1 (y, A), +1 (y, C); s_ntof = s_bench (a sign flip of both r and j leaves s unchanged) | derived above |

The n_TOF scale s also carries the factor v_bundle / (kappa v_t) of the pass-1 angle (the C factor is 0.958 on x), a global scale that I remove by taking the
median over cells; the bench s has no such factor. Cells were indexed by the reconstructed (x_p0, y_p0), not by the mid-gap position.

**Cross-check by correlation (8 mirror/sign combinations, tilt maps; flip of x position, of y position, of the sign of a).** The derived combination is the
best one for A x (r = 0.52) and C y (0.62) at the 0 mm margin, and for A x (0.20) at the 30 mm margin. Disagreements, all flagged:
* A y: the derived combination (r = 0.51 / 0.74) ties with flipping both positions and the sign of a (0.53 / 0.74), because an odd map is invariant under this
  point inversion. In the held-out sigma the inverted combination is better by 1.5 % of sigma (gain 9.3 % against 7.8 %), about 1 sigma, so it does not
  overrule the derivation. The plain sign flip of a alone is excluded (the held-out sigma gets 20 % worse).
* C x: the best Pearson combination flips x (r = 0.41, against -0.03 derived at the 0 mm margin; 0.66 against 0.63 at 30 mm). In the held-out sigma
  none of the 8 combinations helps (derived +0.2 %, flipped x -6 to -8 %), so the 0.41 is the maximum of 8 noise correlations over 57 cells.
* C y at the 30 mm margin: best (flip y position and sign) 0.62 against derived 0.56; held-out all combinations are negative (-4.9 % to -16 %).
* Shift scan of the bench shape by whole cells (+-3 cells in x and y): the unshifted position is the best for A x and A y (gain 13.0 % and 7.8 %) and for C x; only C y
  prefers a shift of (3, -3) cells, with a gain still negative (-2.6 %). The registration of A is therefore right to within a cell (50 mm).

## 2. Bench residual maps beyond the bow block

Selection: `GM.load_table` replicated with each chamber's own active area (`common/mx17_active_area.TRUE_ACTIVE_BY_DET`): single track, M3 chi2 < 1, 4 clusters,
interior margin 30 mm (standard) or 0 mm (to reach the outer cells, where the n_TOF distortion sits). Residual r = x_tan_theta - tan_ref per view, |tan_ref| < 0.45, |r| < 0.15 (the n_TOF
clean-line cuts), per cell the robust line r = a + s tan_ref of `angle_map.robust_line`; at least 60 events per cell (n_TOF: 80). Rows: det3 6-27 saturday (4.6 k events at
30 mm margin, 36 cells) and 6-28 p2 (14.1 k, 59 cells); det6 long_run (5.0 k, 55 cells) and longer_run (2.2 k, 13-16 cells). The bow block of the bundles is the same in
both rows of a chamber and the same as the n_TOF bundle (`bench_bundles/A`, `pf/A_*`), so "beyond the block" is the same on both sides.
Per-cell bench errors (bootstrap): a 0.7-0.8e-3 (A), 1.8-2.3e-3 (C); s 0.6-0.7 % (A), 1.6-2.0 % (C).

Reproducibility (Pearson r over the cells filled in both halves):

| chamber view | bench row 1 vs row 2 (tilt a) | bench tag-parity halves (a) | bench halves (s) | n_TOF run-parity halves (a) | n_TOF halves (s) |
|---|---|---|---|---|---|
| A x | 0.99 (36 cells) | 0.99 | 0.92 | 0.97 | 0.79 |
| A y | 0.99 (36) | 0.98 | 0.77 | 0.83 | 0.76 |
| C x | 0.14 (13 cells only) | 0.70 | 0.84 | 0.97 | 0.96 |
| C y | 0.84 (16) | 0.93 | 0.87 | 0.90 | 0.93 |

For C the second bench row fills only 13-16 cells, so the row split is not a usable ceiling for C; the time halves of the pooled rows (which are not independent runs) are given instead.
The A tilt map is the same on two days of bench data; the n_TOF map is stable between run parities too. The noise ceilings are therefore not what limits the A comparison.

## 3. Cell-by-cell comparison (median-free: raw values, cells with >= 80 n_TOF lines and bench data)

Map: `angle_map_r2.json` (the final in-situ map; the earlier `angle_map.json` gives the same picture, plane gradients differ by < 10 %). Bench maps pooled over the two rows.
Errors on r: bootstrap over cells (the cells are spatially correlated, so these errors are optimistic).

| chamber view | margin | cells | r(a) | slope a_ntof / a_bench | r(a) / bench-halves ceiling | cross r(a) (bench half x n_TOF half, noise-free) | r(s) | slope s | cross r(s) |
|---|---|---|---|---|---|---|---|---|---|
| A x | 0 mm | 61 | +0.52 +- 0.12 | +1.05 +- 0.22 | 0.53 | +0.61 | +0.39 +- 0.13 | +0.40 +- 0.12 | +0.40 |
| A y | 0 mm | 60 | +0.51 +- 0.14 | +0.67 +- 0.15 | 0.54 | +0.59 | -0.12 +- 0.17 | -0.09 +- 0.10 | +0.18 |
| C x | 0 mm | 57 | -0.03 +- 0.13 | -0.03 +- 0.14 | -0.03 | +0.13 | +0.32 +- 0.12 | +0.24 +- 0.10 | +0.37 |
| C y | 0 mm | 57 | +0.62 +- 0.08 | +0.76 +- 0.13 | 0.65 | +0.40 | +0.51 +- 0.12 | +0.41 +- 0.10 | +0.42 |
| A x | 30 mm | 46 | +0.20 +- 0.16 | +0.40 +- 0.31 | 0.20 | +0.46 | +0.44 +- 0.16 | +0.51 +- 0.16 | +0.44 |
| A y | 30 mm | 47 | +0.74 +- 0.07 | +0.90 +- 0.12 | 0.78 | +0.67 | +0.35 +- 0.13 | +0.24 +- 0.10 | +0.37 |
| C x | 30 mm | 39 | +0.63 +- 0.16 | +1.18 +- 0.24 | 0.65 | +0.13 | +0.06 +- 0.16 | +0.07 +- 0.19 | +0.33 |
| C y | 30 mm | 39 | +0.56 +- 0.10 | +0.75 +- 0.18 | 0.59 | +0.40 | +0.24 +- 0.18 | +0.21 +- 0.14 | +0.41 |

(Full table with all ceilings, rms values and the 8 combinations: `tables.md`, `compare.json`.) Reading it: the A tilt agrees in sign and roughly in amplitude (slope 0.7-1.05) with
a moderate correlation. The n_TOF A x map has a much larger edge tilt (+0.04 to +0.06 at x_local = +175, -0.03 to -0.04 at -175) than the bench (+0.01 to +0.02, -0.01 to -0.02), which is why the
correlation is 0.5 and not near the ceiling; in the interior the two agree with slope close to 1 (Fig. 2). The C correlations are unstable between the two margins (x: -0.03 / 0.63, with a
cross r of 0.13 both times, i.e. nothing survives once within-map noise is removed). The scale maps agree weakly at best (r = -0.12 to 0.51); no scale transfer can be claimed for A, and for C
y the 0.51 is not confirmed by the held-out test (beta = 0.3).

Figures: `fig1_maps.png` (the maps side by side in the n_TOF frame), `fig2_scatter_m0.png` and `fig2_scatter_m30.png` (cell scatter with errors and the OLS line).

## 4. Direct test on the n_TOF cosmic lines (held-out single-view sigma)

Lines `ana_r2/track_quality/lines.parquet` (31,387 A-C lines, 40 beam-off runs, built on the unmapped v(t) slopes). Train on one run parity, test on the other, both ways,
mean of the two; sigma = MAD of (corrected slope - joined slope) over all test lines with |j| < 0.45 ("all"); the clean-line sigma is in `tables.md`. Errors: bootstrap over test lines
(about 0.4e-3; the gains below have errors of about 1.5 percentage points, and the two model columns share the same test lines). Bench map: 0 mm margin; the 30 mm margin is in `tables.md`.
Units 1e-3 in tan.

| chamber view | no map | global only (c + gamma j) | in-situ 8x8 | bench map as is | global + bench shape | bench, free amplitude | bench a only, free amplitude | gain in-situ | gain bench as is | gain bench free amp | share of in-situ gain (free amp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A x | 28.6 | 28.7 | 24.5 | 25.6 | 25.6 | 25.0 | 25.1 | 14.6 % | +10.8 % | +12.7 % | 87 % |
| A y | 26.2 | 26.0 | 23.1 | 24.3 | 24.4 | 23.9 | 24.8 | 12.0 % | +7.3 % | +8.8 % | 73 % |
| C x | 32.5 | 32.4 | 27.2 | 32.6 | 32.8 | 31.9 | 32.4 | 16.2 % | -0.4 % | +1.6 % | 10 % |
| C y | 37.1 | 37.1 | 31.4 | 39.8 | 39.2 | 36.3 | 37.0 | 15.5 % | -7.1 % | +2.1 % | 14 % |

With the standard 30 mm interior for the bench map the A gains are 7.3 % (x) and 7.5 % (y) as it is, 7.2 % and 8.8 % with a free amplitude; C 1.0 % / -7.9 % as is, 4.2 % / 1.2 % with a free amplitude.
The README numbers of the in-situ map (A x 0.031 -> 0.026, A y 0.029 -> 0.025, C x 0.035 -> 0.028, C y 0.039 -> 0.032) were on the first line set (which contained 14,742 production events); on
the true beam-off lines of `ana_r2` the held-out values are those of the table (for example A x 0.0286 -> 0.0245), i.e. the same 12-16 % gains.

Fitted amplitudes of the bench shape (train parity 0 / 1), 0 mm margin: tilt alpha: A x 1.46 +- 0.05 / 1.54 +- 0.02, A y 1.33 +- 0.06 / 1.34 +- 0.02, C x 0.36 / 0.29, C y 0.52 / 0.58;
scale beta: A x 0.35 / 0.19, A y 0.71 / 0.62, C x 0.42 / 0.43, C y 0.29 / 0.32. (At the 30 mm margin the C alphas are unstable: 1.47 for x, 0.27-0.34 for y.)

Does the fine structure add to a smooth surface? Held-out sigma with polynomial surfaces for a(x,y) and s(x,y) (`g_poly.py`, bench shape at 0 mm margin):

| chamber view | const | plane | quad | bench shape | plane + bench | quad + bench | in-situ 8x8 |
|---|---|---|---|---|---|---|---|
| A x | 28.7 | 29.0 | 27.1 | 25.0 | 24.5 | 23.8 | 24.5 |
| A y | 26.0 | 25.8 | 25.1 | 23.9 | 23.4 | 23.1 | 23.1 |
| C x | 32.4 | 31.2 | 29.9 | 31.9 | 30.8 | 29.4 | 27.2 |
| C y | 37.1 | 34.9 | 32.5 | 36.3 | 33.4 | 31.3 | 31.4 |

For A the quadratic with the bench shape reproduces the in-situ map (23.8 against 24.5, 23.1 against 23.1) and no smooth surface alone does; for C a quadratic does most of the work and the in-situ map still
beats it by 2-3e-3 in x. Figure `fig3_sigma_m0.png` (and `_m30`).

## 5. Quadratic level: does the bench bow block over- or under-correct at n_TOF?

If the dish were gravity sag of a horizontal board it would vanish for a vertical chamber. Pass 1 applies the bench block (same numbers in `bench_bundles` and the bench rows). Gradients are plane
fits over the cells common to the n_TOF and the bench map, along the view's own axis, in 1e-3 tan per 100 mm (`total_gradients.csv`, errors by bootstrap over cells), with the block, the tilt of the bench
total (block + bench residual) and of the n_TOF total (block + n_TOF residual) in the n_TOF frame:

| chamber view | block | bench total | n_TOF total | n_TOF residual gradient / block gradient | regression of the n_TOF residual on the block: f - 1 (f = 1: block right at n_TOF, f = 0: block absent) | same, n_TOF minus bench residual |
|---|---|---|---|---|---|---|
| A x | -5.3 +- 0.4 | -5.0 +- 1.2 | +5.7 +- 1.9 | -1.77 | -1.98 +- 0.26 | -1.78 +- 0.23 |
| A y | -8.7 +- 0.2 | -8.0 +- 1.3 | -1.6 +- 1.1 | -0.76 | -0.87 +- 0.12 | -0.72 +- 0.14 |
| C x | -14.9 +- 0.6 | -7.7 +- 1.9 | -18.9 +- 2.4 | +0.11 | +0.11 +- 0.14 | +0.74 +- 0.19 |
| C y | +18.1 +- 1.2 | +15.1 +- 1.6 | +15.6 +- 1.9 | -0.07 | -0.20 +- 0.09 | +0.04 +- 0.07 |

* **A over-corrects at n_TOF.** In y the residual cancels 76 % of the block's gradient (the total is 18 % of the block, f = 0.1-0.3), in line with a gravity-sag origin of A's dish. In x
  the block gradient is small (5e-3 per 100 mm, rms 7e-3 over the plane against a residual rms of 2e-2), so the ratio of -1.8 is ill-conditioned; the n_TOF total gradient has the opposite sign to the bench's with
  similar size. The regression error is statistical only and the cells are correlated, so the exact ratio in x should not be quoted; the sign (over-correction) is clear.
* **C is not over-corrected.** y: the total gradient at n_TOF (15.6) equals the bench total (15.1) and the block (18.1), so the dish is present at full size and is not gravity sag. x: the n_TOF total (-18.9) is larger than the block (-14.9); the
  bench total (-7.7, from the two rows with different fields) is smaller than the block, so the C x bench value is itself unstable, and the n_TOF minus bench ratio of +0.74 +- 0.19 is
  mostly that.
* The block's scale part: the n_TOF scale residual anti-correlates with the block's scale gradient in all four cases (regression f - 1 = -0.6 to -0.85, r = -0.58 to -0.77), i.e. the block's scale gradient is also mostly absent at n_TOF.
  Since the n_TOF angle is the v(t) angle with kappa and the scale pattern is correlated with the timing-velocity map, this is less clean than the tilt.

Figure `fig4_bow_gradient.png`.

## 6. Limits of the comparison

* The n_TOF truth is the joined A-C line. It depends only on the two positions (469 mm lever arm), not on either chamber's angle, so the residual map isolates each chamber's angle bias. Position errors that
  depend on the angle (tan h up to 0.9 mm, finding 22 of the README; 1 mm = 0.002) leak into j at the 0.002 level, an order of magnitude below the A tilts. The bench truth is M3.
* The n_TOF angle is the timing v(t) angle times v_bundle / (kappa v_t): it has a different scale from the ladder angle on the bench, the drift field and gas differ, so a tilt in tan units is field-independent
  (BOW_TRANSFER) but the scale map is not, and I compare s only after removing the median. Scale transfer cannot be tested cleanly.
* n_TOF clean lines are cut at |r| < 0.15, |j| < 0.45 and the other chamber's view and own other view within 0.10; the bench uses the same |r|, |tan| cuts but M3 chi2 and cluster cuts. n_TOF resolution per cell
  (a 0.4-1.5e-3 error per cell, 80-760 lines) is comparable to the bench error for A (0.7-0.8e-3) and better than for C (1.8-2.3e-3).
* The bench C statistics are poor: 6.6 k events (interior to the active edge) in 57 cells, and the second row covers only the centre. A positive C result would have needed more than we have; the negative C result is
  therefore weaker than the positive A result, but the C bench halves do reproduce (a: 0.70-0.93), so noise alone does not explain a zero held-out gain. The C bench rows also differ in drift field (v = 24 and 20 um/ns).
* Cell correlations use a few tens of spatially correlated cells; the quoted bootstrap errors are optimistic. The held-out sigma test is the decisive one, since no bench information is fitted on n_TOF except the one or two
  amplitudes (and the chamber-wide constants, which the in-situ map also has).
* The 30 mm interior of the standard bench selection removes most of the outer cells, where the n_TOF distortion is largest; the 0 mm margin includes cells with poorer M3 acceptance at the edge (n = 36-100 per cell).
* The bench rows used are the two named in the task; the det3 6-25 day row and the 6-22 det2/det3 rows were not used.
* Nothing here proposes a new measurement; the answer for C, if it matters, would have to come from simulation or from the existing in-situ map.
