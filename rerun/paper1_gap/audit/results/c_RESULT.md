# Audit C: does chamber A's fine angle map belong to the chamber and to the drift-field geometry? (2026-11-24)

Scripts: `rerun/paper1_gap/audit/c_lib.py`, `c_01_mount.py`, `c_02_fit.py`, `c_03_res.py`, `c_04_cells.py`, `c_05_gap_dish.py`, `c_06_figs.py`, `c_07_tables.py`, `c_08_flipnull.py`, `c_09_fig5.py`.
Outputs here: `kfit.json`, `res625.json`, `cells.json`, `gapdish.json`, `flipnull.json`, `mount.json`, `tables.md` (all tables below), figures `fig1_fine_maps_n4.png`, `fig1_fine_maps_n6.png`, `fig2_fine_scatter.png`, `fig3_k_values.png`, `fig4_s68.png`, `fig5_gap_dish_maps.png`.

## Verdict (numbers below)

1. **Bound to the chamber? Only partly, and the test cannot prove more.** Run 1 and run 2 (same mounting) agree on the 4x4 fine map of w0 (slope 0.89 to 0.95 +- 0.06, correlation 0.98 to 0.99): the map is reproducible and real. The 6-25 mounting has an own fine map of the same size (sd 0.0042 against 0.0040 in X, 0.0071 against 0.0058 in Y, tan units) but its pattern correlates with the golden one only at 0.60 (X) and 0.64 (Y) for 4x4 cells, 0.43 and 0.64 for 6x6, with slope 0.63 and 0.79 (4x4), 0.38 and 0.69 (6x6). Noise does not explain it (noise is 1 to 6 % of the cell variance). The correlation has little discriminating power, however: the golden map mirrored in U or W correlates with the 6-25 map equally well or better (X 4x4: -0.63 to +0.60; Y 4x4: -0.72 to +0.69), because 16 cells are dominated by a few low-order modes. In the amplitude fit the mirrored golden maps explain the 6-25 row as well as the unmirrored one or better (X 4x4: reduction of the residual sum of squares 1.9 % for the true orientation, 6.2 % and 5.2 % for the two single mirrors; Y 4x4: 9.7 % against 8.6 %, 6.5 %, 10.6 %). So the data do not show that the golden fine pattern is the one present in the 6-25 mounting.
2. **Geometric (field independent in tan units)? Not supported for the fine map; the plane part is neither geometric nor fixed rate in Y.** After division by the same-mounting control (the control is itself 1.3 to 1.9, see section 2), k_fine = 0.44 +- 0.05, 0.35 +- 0.03, 0.34 +- 0.03 in X and 0.83 +- 0.04, 0.68 +- 0.04, 0.57 +- 0.03 in Y for n = 4, 6, 8. The geometric value is 1 and the fixed-rate value 1.63 (v_golden/v' = 1.65 X, 1.62 Y, not 2.3: the measured v ratio is 0.59, not 0.44). Both are excluded in X; Y is between 0 and 1, closer to the geometric value at n = 4 and falling with n. The amplitude therefore points to "partial transfer", not to a fixed rate, and the attenuation grows with finer cells, which is what a mounting-dependent fine pattern gives. k_plane (the dish tilt) is 1.12 +- 0.03 in X (geometric within 12 %) but 2.05 to 2.25 in Y (the golden plane has to be doubled; above both references).
3. **Usable on another mounting? No for the fine map, with the golden bow block only partly valid.** On the 6-25 row the golden w0 map in tan units reduces s68 by 0.077 +- 0.010 deg in X at 4x4 (0.876 to 0.799) and 0.016 +- 0.009 deg in Y (1.030 to 1.014), but almost all of it comes from the plane part (plane alone: -0.040 X, -0.034 Y); the fine part adds -0.013 (X) and +0.018 (Y) at 4x4, and nothing at 6x6 and 8x8 (X 6x6: -0.015, Y: +0.015). The own in-sample 2-fold map reaches 0.744 (X) and 0.743 (Y) at 4x4, 0.731 and 0.712 at 6x6. Inside the golden mounting the other run's map gives 0.82 to 0.84 (X) and 0.71 to 0.72 (Y) at 4x4 from pass-1 values of 0.91 to 0.93 and 0.80 to 0.81, so the fine map there recovers about 0.09 deg, against 0.00 to 0.02 deg (the fine part alone) in the other mounting. The 16 to 21 % improvements of the audited analysis are a same-mounting property.

Things the brief did not expect, plainly stated:
- The v ratio is 0.59 (v = 22.0 X, 21.8 Y against 37.0 to 37.7), not 0.44. The fixed-rate reference is therefore v_g/v' = 1.65, not 2.3. At 0.97 % water and 355 V/cm the Magboltz tables give v = 36.15, at 178 V/cm 15.97 (ratio 0.44); the 6-25 gas must have contained less water (0.68 % reproduces v = 22.0 at 177.5 V/cm, and 0.89 % reproduces the golden 37.4 at 355).
- The same-mounting control gives k_fine = 1.3 to 1.9, not 1 minus a few percent. A 4x4 cell map is the cell average of the true map; the finer structure inside the cells projects on it with a coefficient above 1. Noise attenuation is small (section 2: 0.6 to 7 % of the cell variance), so it is the resolution of the cells that sets the control. All "k/ctrl" numbers divide by this control; the raw k are also given.
- The 6-25 pass-1 angle is worse than the golden rows (s68 0.876 / 1.030 deg against 0.91 to 0.93 / 0.80 to 0.81) and the golden bow block (identical content in all three bundles, source_row sat_det3) is applied to it. In Y the plane of the drift tilt is twice the golden one, so the pass-1 bow correction is about half of what the 6-25 mounting needs.


## 0. What differs in the 6-25 mounting

| | run 1 (6-27 sat) | run 2 (6-28 p2) | 6-25 day |
|---|---|---|---|
| theta_align_deg | 89.45 | 89.45 | 89.75 |
| alignment z_x / z_y (mm) | 714 / 714 | 713 / 714 | 242 / 243 |
| x_offset / y_offset (mm) | -236.8 / -209.1 | -236.8 / -209.1 | -210.1 / -190.0 |
| selected events (golden-style cuts) | 4654 | 14067 | 6279 |
| ladder slope v_x (um/ns) | 37.04 +- 0.07 | 37.74 +- 0.04 | 22.01 +- 0.04 |
| ladder slope v_y (um/ns) | 36.50 +- 0.07 | 37.14 +- 0.04 | 21.82 +- 0.05 |
| ladder offset w0_x (um/ns) | -0.067 +- 0.010 | -0.066 +- 0.006 | +0.073 +- 0.005 |
| ladder offset w0_y (um/ns) | -0.157 +- 0.009 | -0.158 +- 0.005 | -0.210 +- 0.007 |
| corr(pass-1 tan, tan_ref) x / y | +0.991 / +0.985 | +0.986 / +0.979 | +0.988 / +0.980 |
| bundle v_drift / w0 (x,y) / kw (x,y) | 35.8 / -0.002,-0.165 / 1.037,1.015 | 37.2 / -0.007,-0.163 / 1.009,0.991 | 22.4 / +0.177,-0.222 / 1.004,0.996 |
| bundle bow block | same block (source_row sat_det3, identical content) | same block (source_row sat_det3, identical content) | same block (source_row sat_det3, identical content) |

## 2. Amplitude fits on the 6-25 row (pooled golden map, bootstrap of the 6-25 events, 150 resamples)

Raw k, then ctrl = mean of the two golden-to-golden controls (run1 map on run2, run2 map on run1), then k/ctrl. References: geometric = 1 (in tan units), fixed rate = v_golden/v' (shown per row).

| view | n | v_g/v' | k_plane | k_fine | k_s | ctrl k_plane | ctrl k_fine | ctrl k_s | k_plane/ctrl | k_fine/ctrl | k_s/ctrl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| X | 4 | 1.65 | 1.14 +- 0.02 | 0.76 +- 0.07 | 0.68 +- 0.06 | 1.02 / 1.01 | 1.63 / 1.86 | 1.11 / 1.39 | 1.12 +- 0.03 | 0.44 +- 0.05 | 0.55 +- 0.08 |
| X | 6 | 1.65 | 1.12 +- 0.02 | 0.55 +- 0.05 | 0.49 +- 0.05 | 0.98 / 1.00 | 1.62 / 1.53 | 0.99 / 1.25 | 1.13 +- 0.03 | 0.35 +- 0.03 | 0.44 +- 0.07 |
| X | 8 | 1.65 | 1.11 +- 0.02 | 0.47 +- 0.04 | 0.38 +- 0.05 | 1.01 / 0.99 | 1.39 / 1.42 | 0.97 / 1.17 | 1.11 +- 0.03 | 0.34 +- 0.03 | 0.35 +- 0.06 |
| Y | 4 | 1.62 | 2.22 +- 0.03 | 1.33 +- 0.05 | 1.46 +- 0.09 | 1.12 / 1.04 | 1.59 / 1.61 | 1.37 / 1.32 | 2.05 +- 0.08 | 0.83 +- 0.04 | 1.09 +- 0.09 |
| Y | 6 | 1.62 | 2.22 +- 0.03 | 0.95 +- 0.04 | 1.02 +- 0.07 | 1.09 / 0.98 | 1.33 / 1.47 | 1.07 / 1.13 | 2.14 +- 0.12 | 0.68 +- 0.04 | 0.93 +- 0.08 |
| Y | 8 | 1.62 | 2.19 +- 0.03 | 0.74 +- 0.03 | 0.80 +- 0.07 | 0.97 / 0.97 | 1.28 / 1.32 | 1.01 / 1.10 | 2.25 +- 0.04 | 0.57 +- 0.03 | 0.76 +- 0.08 |

Control checks (same mounting and field; expected 1 up to noise attenuation): noise variance of the golden fine map relative to its cell variance, 1 - lambda:

| view | n | noise/variance golden pooled | noise/variance run 1 | run 2 | noise/variance 6-25 own |
|---|---|---|---|---|---|
| X | 4 | 0.015 | 0.051 | 0.021 | 0.041 |
| X | 6 | 0.015 | 0.068 | 0.019 | 0.057 |
| Y | 4 | 0.006 | 0.021 | 0.007 | 0.014 |
| Y | 6 | 0.007 | 0.025 | 0.009 | 0.022 |

Fixed-rate parametrisation vs tan parametrisation on the same events (4x4, pooled golden map): the two are the same linear model with the column rescaled by v_g/v', so RSS is identical.

| view | k_fine+plane (tan) | k' (um/ns map) | k' v_g/v' | RSS tan | RSS rate | v_g/v' |
|---|---|---|---|---|---|---|
| X | 1.092 | 0.662 | 1.092 | 689.4 | 689.4 | 1.648 |
| Y | 2.117 | 1.302 | 2.117 | 761.0 | 761.0 | 1.626 |

## Cell-level comparison of the maps (slope of second on first, correlation; event bootstrap, 100 resamples)

| view | n | pair | A_fine slope | A_fine corr | A_plane slope | S slope | S corr |
|---|---|---|---|---|---|---|---|
| X | 4 | run1_vs_run2 | 0.89 +- 0.06 | 0.98 +- 0.02 | 1.02 +- 0.04 | 0.91 +- 0.07 | 0.97 |
| X | 4 | gold_vs_r625 | 0.63 +- 0.06 | 0.60 +- 0.05 | 1.11 +- 0.03 | 0.59 +- 0.06 | 0.50 |
| X | 4 | run1_vs_r625 | 0.57 +- 0.08 | 0.59 +- 0.06 | 1.13 +- 0.04 | 0.58 +- 0.08 | 0.53 |
| X | 4 | run2_vs_r625 | 0.64 +- 0.08 | 0.60 +- 0.06 | 1.11 +- 0.03 | 0.57 +- 0.09 | 0.48 |
| Y | 4 | run1_vs_run2 | 0.95 +- 0.05 | 0.99 +- 0.01 | 0.99 +- 0.04 | 0.97 +- 0.12 | 0.91 |
| Y | 4 | gold_vs_r625 | 0.79 +- 0.05 | 0.64 +- 0.03 | 1.96 +- 0.05 | 0.92 +- 0.12 | 0.55 |
| Y | 4 | run1_vs_r625 | 0.74 +- 0.07 | 0.62 +- 0.04 | 1.95 +- 0.07 | 0.74 +- 0.19 | 0.43 |
| Y | 4 | run2_vs_r625 | 0.80 +- 0.05 | 0.65 +- 0.03 | 1.96 +- 0.05 | 0.90 +- 0.11 | 0.56 |
| X | 6 | run1_vs_run2 | 1.04 +- 0.05 | 0.97 +- 0.02 | 1.00 +- 0.03 | 0.86 +- 0.08 | 0.93 |
| X | 6 | gold_vs_r625 | 0.38 +- 0.04 | 0.43 +- 0.04 | 1.08 +- 0.03 | 0.50 +- 0.06 | 0.40 |
| X | 6 | run1_vs_r625 | 0.36 +- 0.08 | 0.39 +- 0.09 | 1.08 +- 0.05 | 0.45 +- 0.07 | 0.39 |
| X | 6 | run2_vs_r625 | 0.39 +- 0.04 | 0.44 +- 0.04 | 1.08 +- 0.03 | 0.49 +- 0.06 | 0.40 |
| Y | 6 | run1_vs_run2 | 0.90 +- 0.04 | 0.98 +- 0.01 | 0.96 +- 0.04 | 0.77 +- 0.08 | 0.84 |
| Y | 6 | gold_vs_r625 | 0.69 +- 0.04 | 0.64 +- 0.02 | 2.04 +- 0.05 | 0.60 +- 0.09 | 0.40 |
| Y | 6 | run1_vs_r625 | 0.62 +- 0.07 | 0.62 +- 0.04 | 1.98 +- 0.09 | 0.38 +- 0.11 | 0.28 |
| Y | 6 | run2_vs_r625 | 0.71 +- 0.04 | 0.65 +- 0.02 | 2.06 +- 0.06 | 0.65 +- 0.10 | 0.43 |

## 3. Resolution on the 6-25 row: s68 of theta - theta_ref (deg), |tan_ref| <= 0.6, difference to pass 1 with paired bootstrap error

| arm | X s68 | X diff | Y s68 | Y diff |
|---|---|---|---|---|
| pass1 | 0.876 | +0.000 +- 0.000 | 1.030 | +0.000 +- 0.000 |
| const | 1.065 | +0.189 +- 0.010 | 1.305 | +0.275 +- 0.012 |
| gold4 w0+v | 0.799 | -0.077 +- 0.010 | 1.014 | -0.016 +- 0.009 |
| gold4 w0 | 0.823 | -0.053 +- 0.010 | 1.014 | -0.016 +- 0.009 |
| gold4 v | 1.043 | +0.167 +- 0.009 | 1.326 | +0.296 +- 0.011 |
| gold4 plane | 0.836 | -0.040 +- 0.009 | 0.996 | -0.034 +- 0.007 |
| gold4 fine | 1.049 | +0.172 +- 0.010 | 1.366 | +0.336 +- 0.013 |
| gold4 w0 (fixed rate) | 0.827 | -0.049 +- 0.013 | 0.889 | -0.141 +- 0.012 |
| gold4 w0+v (fixed rate) | 0.827 | -0.050 +- 0.012 | 0.878 | -0.152 +- 0.012 |
| own2fold4 w0+v | 0.744 | -0.132 +- 0.010 | 0.743 | -0.287 +- 0.014 |
| own2fold4 w0 | 0.793 | -0.084 +- 0.010 | 0.776 | -0.254 +- 0.013 |
| gold6 w0+v | 0.823 | -0.054 +- 0.011 | 1.039 | +0.009 +- 0.011 |
| gold6 w0 | 0.822 | -0.055 +- 0.011 | 1.017 | -0.013 +- 0.011 |
| gold6 v | 1.047 | +0.170 +- 0.009 | 1.344 | +0.314 +- 0.011 |
| gold6 plane | 0.837 | -0.040 +- 0.009 | 1.002 | -0.028 +- 0.007 |
| gold6 fine | 1.036 | +0.160 +- 0.010 | 1.383 | +0.353 +- 0.012 |
| gold6 w0 (fixed rate) | 0.870 | -0.007 +- 0.014 | 0.923 | -0.107 +- 0.014 |
| gold6 w0+v (fixed rate) | 0.880 | +0.003 +- 0.015 | 0.919 | -0.111 +- 0.014 |
| own2fold6 w0+v | 0.731 | -0.145 +- 0.010 | 0.712 | -0.318 +- 0.014 |
| own2fold6 w0 | 0.786 | -0.091 +- 0.010 | 0.755 | -0.275 +- 0.014 |
| gold8 w0+v | 0.843 | -0.033 +- 0.011 | 1.039 | +0.009 +- 0.011 |
| gold8 w0 | 0.835 | -0.041 +- 0.011 | 1.028 | -0.002 +- 0.011 |
| gold8 v | 1.046 | +0.170 +- 0.009 | 1.348 | +0.318 +- 0.010 |
| gold8 plane | 0.835 | -0.041 +- 0.009 | 1.002 | -0.028 +- 0.007 |
| gold8 fine | 1.043 | +0.167 +- 0.012 | 1.394 | +0.364 +- 0.014 |
| gold8 w0 (fixed rate) | 0.900 | +0.024 +- 0.014 | 0.966 | -0.064 +- 0.014 |
| gold8 w0+v (fixed rate) | 0.925 | +0.049 +- 0.015 | 0.966 | -0.064 +- 0.014 |
| own2fold8 w0+v | 0.731 | -0.145 +- 0.009 | 0.721 | -0.309 +- 0.014 |
| own2fold8 w0 | 0.783 | -0.094 +- 0.009 | 0.751 | -0.279 +- 0.013 |

Reference inside the golden mounting (map of one run on the other, same arms; s68 deg):

| view | direction | pass1 | const | 4 w0+v | 6 w0+v | 8 w0+v |
|---|---|---|---|---|---|---|
| X | run1->run2 | 0.929 | 1.047 | 0.836 | 0.791 | 0.766 |
| X | run2->run1 | 0.905 | 1.034 | 0.819 | 0.767 | 0.737 |
| Y | run1->run2 | 0.809 | 0.934 | 0.723 | 0.672 | 0.651 |
| Y | run2->run1 | 0.796 | 0.932 | 0.714 | 0.659 | 0.627 |

## 4b/5. Gap proxy and dish (cell maps by the reference position at mid-gap, as cellobs.py)

| n | pair | quantity | slope (second on first) | corr |
|---|---|---|---|---|
| 4 | run1_vs_run2 | v_rel (both views) | 0.90 +- 0.07 | 0.97 +- 0.02 |
| 4 | run1_vs_run2 | q_u90 rel | 0.99 +- 0.04 | 0.99 +- 0.01 |
| 4 | run1_vs_run2 | v x q_u90 product | 0.98 +- 0.10 | 0.96 +- 0.03 |
| 4 | run1_vs_run2 | hp_x, plane removed (mm) | 0.75 +- 0.14 | 0.78 +- 0.13 |
| 4 | run1_vs_run2 | hp_y, plane removed (mm) | 0.67 +- 0.08 | 0.92 +- 0.08 |
| 4 | gold_vs_r625 | v_rel (both views) | 0.73 +- 0.07 | 0.49 +- 0.04 |
| 4 | gold_vs_r625 | q_u90 rel | 0.45 +- 0.03 | 0.39 +- 0.03 |
| 4 | gold_vs_r625 | v x q_u90 product | 0.42 +- 0.08 | 0.40 +- 0.07 |
| 4 | gold_vs_r625 | hp_x, plane removed (mm) | 1.17 +- 0.17 | 0.94 +- 0.06 |
| 4 | gold_vs_r625 | hp_y, plane removed (mm) | 1.10 +- 0.18 | 0.86 +- 0.07 |
| 6 | run1_vs_run2 | v_rel (both views) | 0.90 +- 0.06 | 0.97 +- 0.02 |
| 6 | run1_vs_run2 | q_u90 rel | 0.97 +- 0.03 | 0.98 +- 0.01 |
| 6 | run1_vs_run2 | v x q_u90 product | 0.93 +- 0.07 | 0.94 +- 0.04 |
| 6 | run1_vs_run2 | hp_x, plane removed (mm) | 0.47 +- 0.13 | 0.54 +- 0.13 |
| 6 | run1_vs_run2 | hp_y, plane removed (mm) | 0.63 +- 0.08 | 0.80 +- 0.09 |
| 6 | gold_vs_r625 | v_rel (both views) | 0.57 +- 0.07 | 0.37 +- 0.04 |
| 6 | gold_vs_r625 | q_u90 rel | 0.36 +- 0.03 | 0.26 +- 0.02 |
| 6 | gold_vs_r625 | v x q_u90 product | 0.38 +- 0.09 | 0.30 +- 0.07 |
| 6 | gold_vs_r625 | hp_x, plane removed (mm) | 0.88 +- 0.10 | 0.86 +- 0.07 |
| 6 | gold_vs_r625 | hp_y, plane removed (mm) | 0.95 +- 0.14 | 0.77 +- 0.08 |

Amplitudes (sd over cells, relative): 4x4 gold: v 0.022, q_u90 0.048, product 0.027, mean q_u90 609 ns; 4x4 r625: v 0.034, q_u90 0.055, product 0.028, mean q_u90 1051 ns; 6x6 gold: v 0.026, q_u90 0.051, product 0.028, mean q_u90 609 ns; 6x6 r625: v 0.039, q_u90 0.070, product 0.035, mean q_u90 1056 ns

Dish of the position route, h = h0 + lu U + lw W + kuu U^2/2 + kww W^2/2 + kuw U W (mm, U,W in 100 mm). Error = event bootstrap (formal error scaled by sqrt(chi2/dof) is about 1.5 to 2 times larger):

| n | view | mounting | kuu | kww | kuw | chi2/dof |
|---|---|---|---|---|---|---|
| 4 | X | gold | 0.52 +- 0.10 | 1.30 +- 0.12 | 0.03 +- 0.05 | 4.5 |
| 4 | X | r625 | 0.82 +- 0.13 | 1.19 +- 0.18 | 0.05 +- 0.07 | 3.4 |
| 4 | X | run1 | 0.41 +- 0.19 (formal) | 1.31 +- 0.22 (formal) | 0.08 +- 0.10 (formal) | 1.3 |
| 4 | X | run2 | 0.54 +- 0.22 (formal) | 1.33 +- 0.24 (formal) | 0.01 +- 0.11 (formal) | 4.5 |
| 4 | Y | gold | 0.87 +- 0.12 | 1.49 +- 0.17 | -0.05 +- 0.06 | 2.9 |
| 4 | Y | r625 | 1.15 +- 0.19 | 1.65 +- 0.27 | 0.01 +- 0.10 | 4.9 |
| 4 | Y | run1 | 1.11 +- 0.19 (formal) | 1.88 +- 0.25 (formal) | -0.15 +- 0.11 (formal) | 1.1 |
| 4 | Y | run2 | 0.84 +- 0.15 (formal) | 1.28 +- 0.20 (formal) | -0.04 +- 0.08 (formal) | 2.4 |
| 6 | X | gold | 0.75 +- 0.11 | 1.37 +- 0.12 | 0.03 +- 0.05 | 2.4 |
| 6 | X | r625 | 0.74 +- 0.15 | 0.97 +- 0.17 | -0.00 +- 0.07 | 2.1 |
| 6 | X | run1 | 0.59 +- 0.19 (formal) | 1.38 +- 0.19 (formal) | 0.06 +- 0.10 (formal) | 1.2 |
| 6 | X | run2 | 0.82 +- 0.17 (formal) | 1.37 +- 0.16 (formal) | 0.03 +- 0.09 (formal) | 2.5 |
| 6 | Y | gold | 1.01 +- 0.11 | 1.47 +- 0.15 | -0.05 +- 0.06 | 2.3 |
| 6 | Y | r625 | 0.96 +- 0.17 | 1.70 +- 0.21 | -0.06 +- 0.09 | 3.4 |
| 6 | Y | run1 | 1.17 +- 0.19 (formal) | 1.53 +- 0.26 (formal) | -0.20 +- 0.11 (formal) | 1.2 |
| 6 | Y | run2 | 0.92 +- 0.14 (formal) | 1.35 +- 0.18 (formal) | -0.01 +- 0.08 (formal) | 2.1 |

## Mirror (orientation) null of the fine map on the 6-25 row (`c_08_flipnull.py`)

k_fine and the reduction of the residual sum of squares when the fine column is added (plane column kept), for the golden map evaluated at the true position and at mirrored positions (U to -U, W to -W; the plane is mirrored with it).

| view | n | true | mirror U | mirror W | mirror U and W |
|---|---|---|---|---|---|
| X | 4 | k 0.77, dRSS 1.9 % | k -1.46, 6.2 % | k 1.32, 5.2 % | k -0.87, 2.3 % |
| X | 6 | 0.55, 2.3 % | -0.85, 5.1 % | 0.72, 3.8 % | -0.64, 3.1 % |
| Y | 4 | 1.33, 9.7 % | 1.22, 8.6 % | -1.03, 6.5 % | -1.34, 10.6 % |
| Y | 6 | 0.95, 10.0 % | 0.93, 10.3 % | -0.80, 8.0 % | -0.93, 10.7 % |

Cell correlation of the fine maps (4x4) with the 6-25 own map: X true 0.60, mirror U -0.51, mirror W 0.52, both -0.63; Y 0.64, 0.69, -0.63, -0.72. Note that the fit with the plane column of the true orientation is not included in this table.


## Part 4 (a): k_s and beta

Measured k_s (6-25 raw / ctrl): X 0.68 / 0.55, 0.49 / 0.44, 0.38 / 0.35 (n = 4, 6, 8); Y 1.46 / 1.09, 1.02 / 0.93, 0.80 / 0.76 (table in section 2). Expected from the gap model, S = -beta dg/g with beta from the Magboltz tables of `rerun/paper1_gas/gasinv.py` (`GasGrid('bench','air_merged')`, central difference in ln E over +-5 %):
- golden: water 0.888 % reproduces v = 37.4 at 355 V/cm, beta(355) = 0.58 (0.69 at 0.97 %);
- 6-25: water 0.684 % reproduces v = 22.0 at 177.5 V/cm, beta(177.5) = 1.28 (1.43 at the golden water 0.89 %);
- ratio beta_625/beta_golden = 2.2 (2.4 at equal water), shown as the dotted line in `fig3_k_values.png`.
The measured k_s (0.35 to 1.1 after the control) is below this in both views (X far below, by more than 10 sigma), so the v_rel map does not scale with beta of the 6-25 field. The v-map correlation with the golden one is also only 0.49 (4x4) and 0.37 (6x6; slope 0.73 +- 0.07 and 0.57 +- 0.07), against 0.97 between run 1 and run 2. The beta values depend strongly on the water content (0.30 to 0.58 at 355 V/cm for 0.68 to 0.89 %), which the 6-25 row does not fix independently, so the 2.2 is only indicative. We have no independent measurement of the 6-25 water content.

## Part 4 (b): q_u90 and the product map

Proxy quality: the golden 4x4 q_u90 map (|tan_ref| <= 0.3 in both views, 20 % trimmed mean of the mean of the two views) against the colsum T map of `cellobs.json` (mean of runs 1 and 2): correlation 0.995, slope q_u90 = 1.13 T (run 1: 0.988 / 1.127, run 2: 0.994 / 1.134). Amplitudes: sd over cells 0.048 for q_u90 against 0.042 for T. The proxy is good in the golden mounting.

Between mountings (4x4; error = event bootstrap, 100 resamples; golden to 6-25, with run 1 to run 2 as control):

| quantity | slope run1 -> run2 | corr | slope golden -> 6-25 | corr | predicted slope (gap) |
|---|---|---|---|---|---|
| v_rel | 0.90 +- 0.07 | 0.97 | 0.73 +- 0.07 | 0.49 | beta ratio 2.2 |
| q_u90 rel | 0.99 +- 0.04 | 0.99 | 0.45 +- 0.03 | 0.39 | (1+beta) ratio 1.45 |
| v x q_u90 | 0.98 +- 0.10 | 0.96 | 0.42 +- 0.08 | 0.40 | 1 |

At 6x6 the 6-25 slopes fall to 0.57, 0.36, 0.38 with correlations 0.37, 0.26, 0.30. So the product (gap) map is NOT the same in the two mountings (correlation 0.40, not near 1), and v and q_u90 do not change amplitude as the different beta predicts (the amplitude over cells is 0.034 against 0.022 for v and 0.055 against 0.048 for q_u90, i.e. ratios 1.5 and 1.15, but the patterns differ). Caveats: q_u90 is a charge-time quantile (mean 609 ns golden, 1051 ns in 6-25, ratio 1.73), is not the strip-sum duration T, depends on gain and threshold (mesh 495 against 490 V) and on the electronics, and we only checked it against T in the golden mounting. A small chamber-bound change of the map between mountings (the board, the cathode foil, the drift field homogeneity) would equally break the correlation. The result is exploratory.

## Part 5: position route and dish

Mesh height hp (slope of p0 - ref_mesh against tan_ref per cell, all |tan_ref|), plane removed (the rigid tilt in the telescope differs between mountings). Golden against 6-25, 4x4: X correlation 0.94 +- 0.06, slope 1.17 +- 0.17; Y 0.86 +- 0.07, 1.10 +- 0.18. 6x6: X 0.86 +- 0.07, 0.88 +- 0.10; Y 0.77 +- 0.08, 0.95 +- 0.14. The position-route map of the mesh is therefore the same in both mountings (slope compatible with 1), and it is a stronger agreement than between run 1 and run 2 of the golden mounting (slope 0.47 to 0.75, corr 0.54 to 0.92), where the map's own noise limits it. The dish curvatures (mm per (100 mm)^2, h = ... + kuu U^2/2 + kww W^2/2, bootstrap errors; tables above): golden X kuu 0.52 +- 0.10, kww 1.30 +- 0.12; 6-25 X 0.82 +- 0.13, 1.19 +- 0.18; golden Y 0.87 +- 0.12, 1.49 +- 0.17; 6-25 Y 1.15 +- 0.19, 1.65 +- 0.27 (4x4). At 6x6 they are X 0.75 +- 0.11 / 1.37 +- 0.12 against 0.74 +- 0.15 / 0.97 +- 0.17, Y 1.01 +- 0.11 / 1.47 +- 0.15 against 0.96 +- 0.17 / 1.70 +- 0.21. The differences are 0 to 2 sigma, so the dish of the mesh is the same in the two mountings within the errors (the fit chi2/dof of 2 to 5 means these errors are optimistic; the formal errors are 1.5 to 2 times larger). Bundle bow block of all three rows: h_x kuu 0.80, kww 1.31; h_y 0.97, 1.67.

Consequence: the mesh dish (position route) transfers, but the drift-tilt plane in Y does not (k_plane = 2.1 +- 0.1). The drift tilt depends on grad h + grad c, so the cathode must differ between the mountings in Y (a different cathode foil tension or mounting), or the y tilt has a contribution that is not geometry. We cannot separate the two with the existing data.

## Caveats

- The k fits use the bilinear cell maps evaluated at the reconstructed mid-gap position; the cell maps are averaged over 100 to 1,500 events per cell (6-25 own: 390 per 4x4 cell, 100 per 6x6 cell).
- The 6-25 table has 6,279 selected events (8,877 rows before the cuts), not 7,500. Run 1 has 4,654 and run 2 14,067 after the same cuts.
- The fixed-rate and the tan description are the same linear model for a fixed v' (the column is rescaled by v_g/v'), so their chi2 are identical (RSS 689.4 X, 761.0 Y, section 2); only the value of k discriminates. A tan-unit k of 1.09 (X) and 2.12 (Y) for plane plus fine together corresponds to a rate-unit k' of 0.66 and 1.30. A mixed mechanism (part geometric, part rate) cannot be separated from a genuine mounting change with one extra mounting.
- Only one extra mounting exists (6-25); the chamber was moved, so mounting changes (board strain, cathode foil) and a field dependence cannot be separated.
- The control divisor assumes the same cell-resolution response in the golden and in the 6-25 maps; it is a first-order correction, and the raw k are given beside it.
- Statistical errors are bootstraps over events; the cell-to-cell scatter of the fine map beyond statistics (rchi2 of the dish fits 2 to 5) is not in the errors of the dish.
