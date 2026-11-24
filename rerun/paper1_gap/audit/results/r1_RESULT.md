# R1: can the drift tilt be predicted from the drift-time map plus the static mesh? (audit tag r1, 2026-11-24)

Scripts (`rerun/paper1_gap/audit/`): `r1_lib.py`, `r1_lib2.py` (loading, P-spline gap field, regressors), `r1_02_select.py`/`r1_02b_select.py` (K and penalty), `r1_03_part1.py`,
`r1_04_part2.py` (map statistics), `r1_05_ab.py` (free a, b), `r1_06_part3.py` (golden tracking), `r1_07_part3_625.py` (6-25 tracking), `r1_08_part4.py`, `r1_09_resid.py`, `r1_10_planesplit.py` (diagnostics), `r1_11_figs.py`.
Outputs here: `part1.json`, `part2_maps.json`, `part2_ab.json`, `part3_golden.json`, `part3_625.json`, `part4_split.json`, `part4_planesplit.json`, `part4_resid.json`, logs, pickles.
Figures: `fig1_gap_maps.png`, `fig2_tilt_golden.png`, `fig3_tilt_625.png`, `fig4_fine_scatter.png`, `fig5_s68.png`.

## Method in brief
- Gap field: ln of the amplitude-walk-free T (golden, colsum) or q_u90 (all three datasets) fitted on the reconstructed mid-gap position with a 12 x 12 tensor-product cubic B-spline, second-difference penalty, robust clip at 3.5 MAD (about 99 % of events kept). delta g = 28.2 mm x delta ln / (1 + beta), beta = 0.68 (golden), 1.28 (6-25, agent C's gas inversion; it is only indicative, so a free amplitude is also fitted).
- K and the penalty were chosen on the golden runs only, by the fraction of the weighted tilt variance explained at n = 8 when the T field of run 1 predicts the tilt of run 2 and the reverse (a = 1, b = 0.5). Scan K = 4...12, lambda = 1e-4...10: explained fraction 0.49 (mesh only) to 0.65. The maximum is a broad plateau, K = 10-12, lambda = 0.03-0.3 (0.63-0.65). Frozen: **K = 12, lambda = 0.1** (lambda is relative: penalty = lambda x N / K^2 x second differences). Used unchanged for 6-25.
- Mesh term: bundle bow block. Default `h` (cross-view mean) as in the brief; per-view `h_x`, `h_y` and a hp-cell quadratic as variants. Gradient in tan = dh/dU / 100.
- Tilt prediction tau = a dh/dx + b d(delta g)/dx with a = 1, b = 0.5 nominal. All errors: bootstrap over events (the spline is refitted, 100-150 replicates, and a random replicate of the gap field is paired with each replicate of the tilt events); cell-level statistics use a cell bootstrap.
- Measured tilt: w0 / v from `CO.line` in n x n cells at the reconstructed position, |tan_ref| <= 0.6 (the 6-25 cuts as agent C).
- Tracking arms reproduce the committed ceilings exactly (direct 8x8 from the other run: X 0.736 / 0.776, Y 0.627 / 0.651; 6-25 own 2-fold 4x4: 0.744 / 0.743; golden direct map on 6-25: 0.799 / 1.014).

## Part 1. Gap fields (`part1.json`, `fig1_gap_maps.png`)
rms of delta g (mm, mean map error from the bootstrap in brackets) and quadratic coefficients (mm per (100 mm)^2, h = c0 + gu U + gw W + guu U^2/2 + gww W^2/2 + guw UW):

| set | N | rms delta g | err | guu | gww | guw | fine rms (quadratic removed) |
|---|---|---|---|---|---|---|---|
| T run 1 | 3220 | 0.81 | 0.25 | +0.48 +- 0.16 | -1.13 +- 0.18 | +0.05 +- 0.10 | 0.42 +- 0.04 |
| T run 2 | 9318 | 0.78 | 0.14 | +0.35 +- 0.08 | -1.20 +- 0.10 | +0.03 +- 0.04 | 0.43 +- 0.03 |
| q_u90 run 1 | 4644 | 0.93 | 0.13 | +0.53 +- 0.08 | -1.56 +- 0.09 | +0.09 +- 0.04 | 0.43 +- 0.02 |
| q_u90 run 2 | 14040 | 0.91 | 0.07 | +0.59 +- 0.04 | -1.43 +- 0.04 | +0.03 +- 0.02 | 0.45 +- 0.01 |
| q_u90 6-25 (beta 1.28) | 6264 | 0.88 | 0.10 | +0.65 +- 0.06 | **+0.58 +- 0.07** | **-0.51 +- 0.03** | 0.53 +- 0.02 |

Comparisons on a 23 x 21 grid (map after removing its mean; r, slope of B on A; errors from independent bootstrap replicates):
- Run 1 against run 2: T total r 0.93 +- 0.04, slope 0.88 +- 0.08; T fine (quadratic removed) r 0.78 +- 0.10, slope 0.78 +- 0.11. q_u90 total r 0.98 +- 0.01, slope 0.96 +- 0.03; q_u90 fine r 0.95 +- 0.03, slope 0.99 +- 0.05. The gap map is reproducible within the golden mounting.
- T against q_u90 (same run): total r 0.94 +- 0.04 / 0.98 +- 0.01, fine r 0.83 +- 0.08 / 0.93 +- 0.03; slope 1.06 / 1.14 total. q_u90 is a usable stand-in for T and less noisy.
- Golden against 6-25: **total r 0.29 +- 0.03, slope 0.38 +- 0.04; fine r 0.25 +- 0.04, slope 0.42 +- 0.07.** The gap map is different in the other mounting. The difference (6-25 minus golden, in ln q): gww +0.136 +- 0.006, guw -0.045 +- 0.003, guu +0.020 +- 0.005, tilt terms 0 within 0.003. In words: the golden gap is a bowl along y (gww -1.4 mm), the 6-25 gap is a dome along y (gww +0.6 mm) with a twist (guw -0.5 mm), and the fine structure also differs (rms 0.53 against 0.43 mm).

## Part 2. Prediction against measurement
### (a) Full maps and fine part, n = 8, a = 1, b = 0.5, T fields, per-view differences negligible (`part2_maps.json`)
Measured tilt of the destination run against prediction (mesh `h` + 0.5 grad delta g); Pearson r and slope of measured on predicted (cell bootstrap):

| gap field -> tilt | view | full r | full slope | fine r | fine slope |
|---|---|---|---|---|---|
| T run 1 -> run 2 | X | 0.84 +- 0.03 | 0.87 +- 0.07 | 0.53 +- 0.09 | 0.66 +- 0.15 |
| T run 1 -> run 2 | Y | 0.79 +- 0.04 | 0.98 +- 0.10 | 0.48 +- 0.10 | 0.80 +- 0.16 |
| T run 2 -> run 1 | X | 0.83 +- 0.03 | 0.92 +- 0.09 | 0.54 +- 0.10 | 0.69 +- 0.16 |
| T run 2 -> run 1 | Y | 0.78 +- 0.06 | 1.02 +- 0.13 | 0.40 +- 0.15 | 0.68 +- 0.25 |
| q run 1 -> run 2 | X | 0.86 +- 0.03 | 0.87 +- 0.06 | 0.59 +- 0.09 | 0.76 +- 0.15 |
| q run 1 -> run 2 | Y | 0.80 +- 0.04 | 1.13 +- 0.11 | 0.53 +- 0.10 | 0.91 +- 0.17 |
| q run 2 -> run 1 | X | 0.85 +- 0.03 | 0.88 +- 0.07 | 0.58 +- 0.09 | 0.76 +- 0.16 |
| q run 2 -> run 1 | Y | 0.83 +- 0.05 | 1.12 +- 0.11 | 0.53 +- 0.10 | 0.95 +- 0.18 |
| **6-25, own q, golden mesh** | X | 0.90 +- 0.02 | 0.87 +- 0.06 | 0.45 +- 0.12 | 0.70 +- 0.22 |
| **6-25** | Y | 0.92 +- 0.02 | 0.87 +- 0.05 | 0.67 +- 0.07 | 0.69 +- 0.09 |

(n = 4, 6 give the same picture, in `part2_maps.json`.) Predicting the fine tilt from the gap field alone has r about 0.5 in the golden mounting. The prediction field is noisy: the reliability of the predicted gradient (cross-run correlation of the two gap-gradient maps) is 0.71-0.82 for T and 0.90-0.95 for q_u90 (`part4_split.json`, IV entries), which caps r near sqrt(reliability) x (measured reproducibility 0.97) = 0.8-0.9, so noise in the field does not by itself explain r = 0.5.

### (b) Plane part (tan per 100 mm, event-level; own-axis and cross gradient)
| | measured own | predicted own (h / per-view h_x,h_y, with 0.5 gap) | measured cross | predicted cross |
|---|---|---|---|---|
| golden X (run 1, run 2) | 0.0105, 0.0107 +- 0.0003 | 0.0119 / 0.0107 (T) | +0.0004, -0.0003 | -0.0001...+0.0003 |
| golden Y | 0.0106, 0.0105 | 0.0092 (h) / 0.0107 (per-view) | -0.0002, -0.0004 | -0.0007...-0.0012 |
| 6-25 X | 0.0117 +- 0.0003 | 0.0121 | -0.0032 +- 0.0003 | -0.0028 |
| 6-25 Y | **0.0201 +- 0.0003** | **0.0209** | -0.0024 +- 0.0002 | -0.0042 |

(Per-view numbers: mesh own 0.0080 (X) and 0.0167 (Y) plus gap halves 0.0027 and -0.0060, quoted from the plane fits of the regressors; the cross-view `h` gives 0.0088 and 0.0149.) The Y tilt plane, twice as steep in 6-25, is predicted to within 4 %: the mesh dish is the same, the gap dome along y adds +0.006 instead of subtracting 0.006.

### (c) Free coefficients tau = a grad h + b grad delta g (`part2_ab.json`, event level, mesh `h`)
| data | view | a | b |
|---|---|---|---|
| golden T, cross-run (run 1 -> 2, 2 -> 1) | X | 0.97 +- 0.07, 1.04 +- 0.05 | 0.35 +- 0.06, 0.35 +- 0.05 |
| golden T, cross-run | Y | 1.01 +- 0.06, 1.02 +- 0.05 | 0.40 +- 0.08, 0.39 +- 0.06 |
| golden q_u90, cross-run | X | 0.92 +- 0.05, 0.91 +- 0.04 | 0.38 +- 0.05, 0.38 +- 0.04 |
| golden q_u90, cross-run | Y | 1.21 +- 0.06, 1.16 +- 0.04 | 0.51 +- 0.05, 0.54 +- 0.04 |
| 6-25, own q_u90 (golden mesh, beta 1.28) | X | 1.15 +- 0.05 | 0.25 +- 0.05 |
| 6-25 | Y | 1.08 +- 0.04 | 0.34 +- 0.02 |

a and b are strongly correlated (0.8 in Y) because the dish of the gap and of the mesh are both planes in the gradient. With a = 1 fixed and the plane part of the gap gradient separated from the rest (`part4_planesplit.json`, per-view mesh):
- golden T: b(plane part) 0.43 +- 0.15, 0.58 +- 0.14 (X), 0.54 +- 0.14, 0.49 +- 0.06 (Y); b(fine part) 0.35 +- 0.06, 0.34 +- 0.06 (X), 0.40 +- 0.08, 0.40 +- 0.07 (Y).
- golden q_u90: b(plane) 0.34 +- 0.06, 0.38 +- 0.06 (X), 0.42 +- 0.04, 0.48 +- 0.04 (Y); b(fine) 0.38 +- 0.05, 0.38 +- 0.03 (X), 0.51 +- 0.05, 0.55 +- 0.04 (Y).
- 6-25: b(plane) 0.63 +- 0.06 (X), 0.24 +- 0.02 (Y); b(fine) 0.18 +- 0.06 (X), 0.35 +- 0.03 (Y).
So a = 1 holds (0.9-1.2) and the plane part of the gap enters with b near 0.5. The fine part enters with b = 0.35-0.55 in the golden mounting (the expected 0.5 is excluded at about 2 sigma for T in X and Y, not for q_u90 in Y), and with 0.2-0.35 in 6-25. The cross-run (instrumental-variable) estimate that removes the noise of the predictor is b = 0.44 +- 0.10 (X) and 0.44 +- 0.11 (Y) for T at n = 8, and 0.38 +- 0.08 / 0.53 +- 0.10 for q_u90 (plain least squares 0.32 / 0.36 and 0.36 / 0.48). So part of the shortfall against 0.5 is attenuation from the noisy field.

Separate long (K = 6 projection) and short wavelength parts (`part4_split.json`): b(long) 0.29-0.44, b(short) 0.43-0.94, with errors of 0.1-0.3 on b(short). There is no sign that the short wavelengths are attenuated more (which tanh(pi g / lambda) would give): the shortfall sits in the long part, where the dish on the mesh and the cathode is not exactly a = 1, b = 0.5 shared.

Velocity exponent from the same fit (v_rel = (g / <g>)^-kappa): golden kappa = 1.02 +- 0.06, 1.08 +- 0.08 (X, T) and 0.73 +- 0.07, 0.70 +- 0.08 (Y) against beta = 0.68; 6-25 kappa = 1.23 +- 0.07 (X), 0.86 +- 0.07 (Y) against the assumed 1.28 (`abkappa` entries). The X view in the golden mounting again wants a larger exponent than Y (as agent B found for the smooth v_x).

## Part 3. Tracking (s68 of theta - theta_ref in degrees, |tan_ref| <= 0.6; paired bootstrap, 200 replicates; full list in `part3_golden.json`, `part3_625.json`; `fig5_s68.png`)
Fraction = (pass 1 - arm) / (pass 1 - direct 8x8 w0+v from the other run) for the golden runs, and against the own 2-fold 4x4 w0+v map for 6-25. `T same` uses the field of the run itself (no reference used, legitimate), `other` the other run's.

### Golden mounting
| arm | run 1 X | run 1 Y | run 2 X | run 2 Y |
|---|---|---|---|---|
| pass 1 | 0.905 | 0.796 | 0.929 | 0.809 |
| constants only | 1.034 | 0.932 | 1.047 | 0.934 |
| mesh only (`h`) | 0.896 | 0.807 | 0.909 | 0.824 |
| mesh + gap, T of other run | 0.863 (-0.042 +- 0.013) | 0.755 (-0.041 +- 0.011) | 0.852 (-0.077 +- 0.008) | 0.764 (-0.044 +- 0.006) |
| mesh + gap, T of same run | 0.853 (-0.052 +- 0.012) | 0.752 (-0.044 +- 0.010) | 0.856 (-0.074 +- 0.007) | 0.781 (-0.028 +- 0.006) |
| mesh + gap, q_u90 other / same | 0.855 / 0.852 | 0.735 / 0.740 | 0.846 / 0.849 | 0.758 / 0.754 |
| mesh + gap + v_rel from the gap (T other / same) | 0.833 / 0.821 | 0.740 / 0.747 | 0.830 / 0.829 | 0.754 / 0.769 |
| (a, b) fitted on the other run, same-run field | 0.853 | 0.759 | 0.860 | 0.770 |
| per-view mesh + gap (T other) | 0.865 | 0.746 | 0.856 | 0.768 |
| direct 4x4 w0 / w0+v (other run) | 0.857 / 0.816 | 0.731 / 0.717 | 0.860 / 0.837 | 0.748 / 0.730 |
| direct 8x8 w0 / w0+v (other run, ceiling) | 0.779 / 0.736 | 0.653 / 0.627 | 0.807 / 0.776 | 0.675 / 0.651 |

Fraction of the 8x8 direct-map gain (w0+v, 0.169 / 0.169 / 0.154 / 0.157 deg) recovered, tilt only (T, other run): run 1 X 0.25 +- 0.07, Y 0.24 +- 0.06; run 2 X 0.50 +- 0.04, Y 0.28 +- 0.03. With the velocity map from the gap: 0.43 +- 0.06, 0.33 +- 0.05, 0.65 +- 0.04, 0.35 +- 0.03. With q_u90: 0.30 / 0.36 (run 1), 0.54 / 0.32 (run 2). The tilt-only prediction is as good as the direct 4x4 w0 map (0.29-0.45 / 0.38-0.39) and recovers 30-60 % of the 8x8 w0 map gain. Averaged over the four cases the reference-free prediction (tilt + v_rel) recovers about 45 % (range 33-65 %) of the committed 16-21 % improvement; a mesh term alone recovers 0-13 % (it is already mostly in pass 1 through its bow block). Same-run and other-run fields agree within errors, so the field is not tuned to the tilt.

### 6-25 mounting (own q_u90 field, golden mesh from the bundle; the golden direct map is the transfer the audit said fails)
| arm | X | Y |
|---|---|---|
| pass 1 | 0.876 | 1.030 |
| constants only | 1.065 | 1.305 |
| golden direct 4x4 w0+v (C: 0.799 / 1.014) | 0.799 (-0.077 +- 0.009) | 1.014 (-0.016 +- 0.009) |
| mesh only (`h`) | 0.850 | 0.908 |
| **mesh + gap, nominal 0.5, beta 1.28** | **0.819 (-0.058 +- 0.010)** | **0.795 (-0.235 +- 0.015)** |
| mesh + gap, free b (a = 1), fitted on half, tested on other half (b = 0.31, 0.36) | 0.817 | 0.790 |
| mesh + gap, free a and b | 0.809 | 0.788 |
| ... + free plane (uses reference) | 0.809 | 0.785 |
| mesh + gap + v_rel (beta 1.28) | **0.756 (-0.120 +- 0.010)** | 0.792 (-0.238 +- 0.015) |
| mesh + gap, free b, + v_rel | 0.759 | 0.784 |
| per-view mesh + gap nominal / + v_rel | 0.826 / 0.765 | 0.797 / 0.806 |
| own 2-fold direct 4x4 w0 / w0+v (ceiling, C: 0.744 / 0.743) | 0.793 / 0.744 | 0.776 / 0.743 |
| own 2-fold direct 6x6 / 8x8 w0+v | 0.731 / 0.731 | 0.712 / 0.721 |

Fraction of the own 2-fold 4x4 w0+v gain (0.132 / 0.287 deg) recovered: nominal tilt only X 0.44 +- 0.06, Y 0.82 +- 0.03; free b X 0.45 +- 0.05, Y 0.84 +- 0.03; with v_rel X 0.91 +- 0.05, Y 0.83 +- 0.03. Against the own tilt-only ceiling (4x4 w0: 0.793 / 0.776) the nominal prediction reaches 0.819 / 0.795 (X 70 %, Y 93 % of that gain). The prediction needs no free number: the free amplitude changes s68 by at most 0.002 (Y) and 0.002-0.010 (X).
The gain in 6-25 Y is the plane, which the mesh term alone half provides (0.908) and the gap dome completes (0.795). The golden direct map, which has the golden plane, gives nothing there (-0.016 +- 0.009).

## Part 4. What is missing (`part4_resid.json`, `fig2_tilt_golden.png`, `fig3_tilt_625.png`, `fig4_fine_scatter.png`)
Residual = measured - predicted (a = 1, b = 0.5, per-view mesh), n = 8, golden cross-run (T field; q_u90 in brackets):
- Variance of the full tilt map explained: X 0.67 (0.72), Y 0.53 (0.63); of the fine part only X 0.00 (0.18), Y 0.15 (0.31). 6-25 (own q): full X 0.78, Y 0.83; fine X 0.12, Y 0.32.
- The residual is **not noise**: the plane-removed residual maps of run 1 and run 2 correlate with r = 0.77 (X) and 0.91 (Y) for T (0.91 and 0.92 for q_u90), and 68 % (X) and 80 % (Y) of the reproducible fine variance of the measured map is left in the residual for T (71 % and 69 % for q_u90). In 6-25 two random halves of the events give residual correlation 0.90 (X) and 0.93 (Y) with 95 % and 63 % of the reproducible fine variance left. The signal is about 7e-3 in tan rms (0.4 deg).
- The residual is still a **gradient field**: cross-run curl test of the residual, r(dRes_x/dW, dRes_y/dU) = 0.83 +- 0.04 for T and 0.92 +- 0.02 for q_u90, slope 0.79 +- 0.06 and 0.87 +- 0.04, against 0.96 +- 0.01 and slope 0.87 +- 0.03 for the measured maps. So the missing part is a smooth potential, not an instrument artefact of one view.
- Not a wrong mesh model: bundle `h`, per-view `h_x`/`h_y` and the hp-cell quadratic (X kuu 0.75, kww 1.40; Y 1.04, 1.44) give residual statistics identical to 0.01 at n = 8. The position-route hp map has no fine structure of its own that can be used (agent B).
- Not noise attenuation of the gap gradient alone: the IV estimate raises b only from 0.32-0.36 to 0.44 for T; a short-wavelength attenuation would make b(short) < b(long), the opposite is found (0.4-0.9 against 0.3-0.4, large errors).
- Candidate missing pieces, which these data cannot separate: (i) the predicted gradient noise (T field reliability 0.7-0.8, so the fine r of 0.5 is partly noise; q_u90 does better: fine r 0.53-0.59, and b(fine) of 0.5 in Y); (ii) the gap-to-ln-amplitude conversion (beta, and the fact that T and q_u90 average over the column; kappa fits suggest beta_eff above 0.68 in X); (iii) a part of the cathode or mesh relief that does not change the column duration, for example a height change correlated between the surfaces (g unchanged, tilt changed); (iv) the reproducible fine structure having scale 28-57 mm, close to the 42 mm cell, where the spline (K = 12 gives 30 mm knots) and the field noise limit the prediction.

## Verdicts
1. **Does tau = grad h + grad g / 2 hold?** Largely yes, with a = 1 and b near 0.5 for the smooth (plane) part and b = 0.35-0.55 for the fine part, and with the gap field coming only from timing. Golden, cross-run: a = 0.97-1.04 (T, mesh `h`, X and Y, +-0.05-0.07); b(total) = 0.35 +- 0.06 (X), 0.39 +- 0.07 (Y) for T, 0.38 +- 0.04 (X) and 0.52 +- 0.04 (Y) for q_u90; the IV estimate with the noisy-field attenuation removed is 0.44 +- 0.10 (T). In 6-25 the same law predicts the Y tilt plane that is twice the golden one (0.0209 predicted against 0.0201 +- 0.0003 measured) from a gap field that has a different sign of the y curvature; b = 0.25 +- 0.05 (X), 0.34 +- 0.02 (Y) with beta = 1.28 assumed, plane part b = 0.63 (X), 0.24 (Y) per-view mesh or 0.48 / 0.38 with `h`. The fine part is only partly predicted (r = 0.5 golden, 0.45-0.67 in 6-25) and about 70 % of the reproducible fine variance stays in the residual, which is a smooth gradient field. The two-surface picture is not contradicted, but it is incomplete at the 0.2-0.3 deg level.
2. **Angle-resolution gain without reference tracks.** Golden mounting: pass 1 improves from 0.905 / 0.796 / 0.929 / 0.809 to 0.853-0.865 / 0.746-0.760 / 0.852-0.860 / 0.764-0.781 with the tilt prediction (-0.04 to -0.08 deg, 4-9 sigma), and to 0.821-0.833 / 0.736-0.752 / 0.829-0.838 / 0.754-0.769 with the velocity map from the same gap field: a recovery of 25-50 % of the tilt-only 8x8 map gain and 33-65 % of the full 8x8 gain (committed ceiling), equal to or better than a direct 4x4 w0 map. 6-25 mounting: Y 1.030 -> 0.795 (-0.235 +- 0.015) recovering 82 +- 3 % of the own 2-fold direct 4x4 gain, X 0.876 -> 0.819 (44 +- 6 %) and 0.756 (91 +- 5 %) with the velocity map. The reference-free prediction beats the transferred golden direct map (X 0.799, Y 1.014) in Y and is comparable in X. At n_TOF this requires the q_u90 map (or T) of the n_TOF data and the static mesh model of the bundle.
3. **What limits it.** In the golden mounting: the noise of the gap field (q_u90 field reliability 0.9-0.95, T 0.7-0.8), which sets the fine-part r near 0.5, a smooth missing piece (about 70 % of the reproducible fine tilt, curl-free at r = 0.83-0.92) and the beta that converts ln T into delta g (kappa_X 1.0-1.1, kappa_Y 0.7 in the golden mounting). In 6-25: the same, plus an unknown beta (1.28 assumed from a gas inversion with no independent water measurement) and the saturation of 87 % of its events, which affects q_u90 in a way not tested here. The plane is predicted well in both mountings.

## Caveats
- K and lambda were tuned on golden tilt maps (both directions), so the golden cross-run numbers are mildly optimistic; 6-25 was untouched (frozen K = 12, lambda = 0.1). The default mesh term (`h`) was fixed before the runs; per-view `h` is shown beside it and changes s68 by at most 0.01 except for the golden Y mesh-only arm (0.849 against 0.824).
- Pass 1 already contains the quadratic bow block (a quadratic tilt of small amplitude); the "mesh only" arm is applied to the raw rate, as in `track_grid.py`, so it is not an increment on pass 1 in the golden mounting.
- The v_rel arm assumes beta (0.68 and 1.28) as nominal, no fit, and in 6-25 it gives a large gain in X (0.819 to 0.756) although agent C found that the 6-25 v map does not scale with beta; with an event-level fit of the exponent kappa = 1.23 +- 0.07 (X) and 0.86 +- 0.07 (Y), the X value agrees with 1.28, the Y value is lower. The gain should be taken as shown, but beta for 6-25 is not independently known.
- "Same-run" fields are legitimate here because T and q_u90 use no reference, but T and the tilt share the same waveform fit, so a small noise correlation is possible; other-run results are the same within errors.
- Fine-part statistics at n = 8 use 40-60 cells; r and slope errors are 0.1-0.2.
- The event-level `a` is not identified when the free plane is added (the mesh gradient is a plane), so the "+ free plane" arm is only a bound on what a refit plane could add (nothing, within 0.003 deg).
