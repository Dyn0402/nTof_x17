# a2 RESULT: smooth (B-spline) angle maps against 42 mm cells, and how robust the gain is

Scripts (all in `rerun/paper1_gap/audit/`): `a2_lib.py` (maps, CV, bootstrap), `a2_repro.py`, `a2_part1.py`, `a2_part2.py`, `a2_part3.py`, `a2_final.py` (final map and `evaluate()`), `a2_report.py` (tables.md and figures). Everything is in this directory. Figures: `fig_s68_vs_K.png`, `fig_learning_curve.png`, `fig_gain_vs_tan.png`. Full tables: `tables.md` (appended below). Final map: `final_map_det3.json`.

## Verdicts

0. **Reproduction: exact.** `track_grid.table/cells`, `TT.bilinear/robust_const/metrics` called from my own script give pass 1, 4x4, 6x6, 8x8, 10x10 for both runs and views with maximum difference 0.0000 deg from `results/track_grid.json`.
1. **Part 1.** Smooth splines beat every cell grid. Trained on run 1 and tested on run 2: X 0.706, Y 0.617 deg (phi, K=12), against 8x8 cells 0.776 / 0.651 and pass 1 0.929 / 0.809. Trained on run 2, tested on run 1: X 0.678, Y 0.588 (K=16) against cells 0.736 / 0.627 and pass 1 0.905 / 0.796. Differences phi - cells8: X -0.070 +- 0.005 and -0.058 +- 0.009, Y -0.034 +- 0.005 and -0.038 +- 0.007 (paired bootstrap). Against pass 1 the gain is -0.22 deg in X (24 %) and -0.19 to -0.21 deg in Y (24-26 %). The s68 is flat for K >= 12 (K = 12 to 26 differ by less than 0.01 deg); K = 5 to 9 is clearly worse. Inner CV picks K = 12 to 16.
2. **Shared potential: not needed, not harmful.** `phi` (one gradient field) and `indep` (separate surfaces) agree within the errors: indep - phi = -0.003 +- 0.003 (X), -0.002 +- 0.003 (Y) for run 1 -> run 2, and -0.005 +- 0.005, +0.005 +- 0.004 for run 2 -> run 1. `phi+` (extra non-gradient surfaces, 30x penalty) adds nothing (differences up to 0.004 +- 0.002). So the data are consistent with a gradient field but cannot show that the constraint helps: the gain from a shared Phi is at most a few 0.001 deg. The phi map has half the offset coefficients, so it is the more economical choice.
3. **The v map (S) matters.** With S = 0 the s68 is 0.03 to 0.04 deg worse in X and 0.015 to 0.02 deg worse in Y than K_S = 4 to 8. K_S = 4, 6, 8 differ by less than 0.01. Inner CV prefers K_S = 8 slightly.
4. **Part 2: learning curve is flat.** s68 at the largest N: A (run 2 + half of run 1 -> other half of run 1) 0.674 / 0.589; B (run 1 + half of run 2 -> other half of run 2) 0.693 / 0.611; union 5-fold 0.694 / 0.600 (X / Y). Fitting s68^2 = s_inf^2 + c/N gives **s68_inf = 0.685 +- 0.012 deg (X) and 0.597 +- 0.010 deg (Y)** (mean over the three sets and three fit windows; error = half of their spread 0.670 to 0.696 and 0.582 to 0.610; split-to-split scatter 0.007). A map from the full statistics is therefore at most 0.01 to 0.02 deg better than from ~15,000 events: the remaining resolution is per-event noise, not map error. Caveat: the asymptote differs between sets by more than the split scatter (A is lower than B), because the test events differ (run 1 vs run 2); and with random half splits the train and test events of one run are interleaved in time.
5. **Part 3a: the gain is stable in time.** Time splits inside run 2 (first/second half, odd/even files, random half) give phi 0.685 - 0.709 (X) and 0.610 - 0.619 (Y), the same as across runs, gain over pass 1 -0.223 to -0.234 (X) and -0.191 to -0.202 (Y), all with errors 0.01. Run 1 (only 2,000 to 2,600 training events) gives phi 0.67 - 0.73 / 0.59 - 0.61 while the 8x8 cells are worthless there (0.78 - 0.92, equal to pass 1): the splines need about 10 times fewer events than the cells.
6. **Part 3e: the pass-1 baseline is fair.** Refitting only the two constants on the test run changes pass 1 by at most 0.01 deg (X 0.929 -> 0.920, Y 0.809 -> 0.801); the pure `const` arm (no bow block) is 1.03 - 1.05 / 0.93 - 0.95. The quoted gain against pass 1 is the gain against the best pass-1-like baseline, in-sample bow block notwithstanding (the bow is a quadratic, 3 numbers; the pass-1+2 constants arm shows nothing to be gained from it).
7. **Part 3b, c, d.** See the tables below. The gain is largest at small |tan_ref| and in the outer region (X run 1 -> run 2: |tan| < 0.1: 0.959 -> 0.699; centre 0.890 -> 0.680, outer 0.950 -> 0.719), where position-dependent offsets are largest; tails (|dtheta| > 2 deg) fall from 4.9 % to 2.2 % (X, |tan| < 0.1). Indexing by the reference position changes s68 by at most 0.002 deg (e.g. phi X 0.7058 -> 0.7062): the reconstructed position is good enough, no reference is needed in production. The map also improves the mid-gap position (p0 + 14 tan - reference): s68 0.554 -> 0.522 mm (X), 0.542 -> 0.517 mm (Y) run 1 -> run 2 (0.558 -> 0.522 and 0.507 -> 0.482 the other way), a 5 % effect.
8. **Part 4: what to ship.** Splines with a shared potential Phi: K = 16 (CV optimum over the union; K = 12 to 20 are equivalent within 0.005), K_S = 8 for the v map (or 6), second-difference penalty lambda small (CV minimum is at the lower grid edge 1e-3 for K = 16, i.e. nearly unpenalised: the cubic basis with K = 16 already smooths; lambda at the grid edge is a caveat, a smaller lambda was not tried), the v map included. Cross-validated expected angle resolution on the union (5 time blocks, maps from the other four, two constants refit per block): **X 0.686 deg, Y 0.599 deg**, against pass 1 0.923 / 0.805 and 8x8 cells 0.753 / 0.637 in the same CV: gain 26 % (X) and 26 % (Y) over pass 1, and 9 % / 6 % over 8x8 cells. Asymptote from Part 2: 0.685 +- 0.012 / 0.597 +- 0.010, so more data would not help. The final map is in `final_map_det3.json` (K=16, K_S=8, lambda=0.001, kappa = vy/vx = 0.985 fixed, Phi coefficients, cx, cy, S_x, S_y coefficients, knots, v used: robust ladder slopes of the union); `evaluate()` in `a2_final.py` reproduces the fitted map to 9e-14.

## Caveats

- The two runs are chamber A in the same conditions on consecutive days; the maps transfer between them, but a different chamber state (HV, gas) is not tested. The cross-run and the time-split tests agree, so the map is stable within these data.
- Bootstrap errors on differences resample test events with the predicted angles fixed (constants not refit inside the bootstrap). Errors do not include the training-set variance, which is why the learning-curve scatter is quoted separately.
- Penalty lambda often sits at the lower edge of the grid (1e-3) (flagged "yes" in the tables). Doing so costs nothing on the plateau in K; I did not extend the grid.
- The `phi` model uses a fixed kappa = vy/vx from a pilot fit to link the two views (a_y = kappa dPhi/dW). The gradient-field hypothesis is not tested independently of this assumption.
- The union CV selected K, K_S, lambda on the same CV folds that give the quoted union numbers (small optimism, probably < 0.005 deg; the cross-run numbers use CV only inside the train run and have no such bias).
- A 14,976 / 14,977 duplicate N row in the union learning table is only fold-size rounding.
- Part 3d: the position residual uses p0 + 14 tan, so it changes only through the angle; s68 differences of 0.03 mm.

---
## Part 1: unbinned spline maps, cross-run (maps chosen by inner CV in the train run only)

### train run1 -> test run2  (n_test = 14067)

**x view**; the spline rows use K_S and lambda chosen by inner CV (K_S, lambda in the last columns); "d vs pass1" and "d vs cells8" are s68 differences with paired-bootstrap errors (400 resamples of test events; the two constants are not refit inside the bootstrap).

| arm | s68 (deg) | d vs pass1 | d vs cells8 | CV s68 (train) | K_S | lambda | lambda at grid edge |
|---|---|---|---|---|---|---|---|
| pass1 | 0.929 +- 0.007 | +0.000 +- 0.000 | +0.154 +- 0.006 | | | | |
| const | 1.047 +- 0.009 | +0.118 +- 0.006 | +0.271 +- 0.008 | | | | |
| pass1c | 0.920 +- 0.007 | -0.009 +- 0.003 | +0.144 +- 0.006 | | | | |
| cells4 | 0.837 +- 0.007 | -0.092 +- 0.005 | +0.062 +- 0.005 | | | | |
| cells6 | 0.790 +- 0.006 | -0.139 +- 0.006 | +0.014 +- 0.004 | | | | |
| cells8 | 0.776 +- 0.007 | -0.154 +- 0.006 | +0.000 +- 0.000 | | | | |
| cells10 | 0.826 +- 0.007 | -0.104 +- 0.007 | +0.050 +- 0.005 | | | | |
| indep K=5 | 0.789 +- 0.007 | -0.140 +- 0.007 | +0.013 +- 0.005 | 0.771 | 8 | 0.001 | yes |
| indep K=7 | 0.745 +- 0.006 | -0.184 +- 0.006 | -0.030 +- 0.005 | 0.729 | 8 | 0.001 | yes |
| indep K=9 | 0.722 +- 0.007 | -0.207 +- 0.007 | -0.054 +- 0.006 | 0.708 | 6 | 0.001 | yes |
| indep K=12 | 0.702 +- 0.006 | -0.227 +- 0.007 | -0.073 +- 0.005 | 0.695 | 8 | 0.00562 | no |
| indep K=16 | 0.705 +- 0.006 | -0.224 +- 0.007 | -0.070 +- 0.005 | 0.697 | 8 | 0.178 | no |
| indep K=20 | 0.699 +- 0.005 | -0.230 +- 0.007 | -0.076 +- 0.005 | 0.697 | 8 | 0.178 | no |
| indep K=26 | 0.705 +- 0.006 | -0.224 +- 0.007 | -0.070 +- 0.005 | 0.699 | 8 | 1 | no |
| phi K=5 | 0.813 +- 0.007 | -0.116 +- 0.006 | +0.038 +- 0.005 | 0.783 | 8 | 0.00562 | no |
| phi K=7 | 0.770 +- 0.006 | -0.159 +- 0.007 | -0.006 +- 0.005 | 0.740 | 8 | 0.001 | yes |
| phi K=9 | 0.725 +- 0.007 | -0.204 +- 0.007 | -0.051 +- 0.006 | 0.702 | 8 | 0.001 | yes |
| phi K=12 | 0.705 +- 0.006 | -0.224 +- 0.007 | -0.070 +- 0.005 | 0.687 | 8 | 0.0316 | no |
| phi K=16 | 0.703 +- 0.006 | -0.226 +- 0.007 | -0.072 +- 0.005 | 0.685 | 8 | 0.001 | yes |
| phi K=20 | 0.707 +- 0.006 | -0.222 +- 0.007 | -0.068 +- 0.006 | 0.695 | 8 | 0.001 | yes |
| phi K=26 | 0.716 +- 0.006 | -0.213 +- 0.008 | -0.059 +- 0.006 | 0.696 | 8 | 1 | no |
| phiplus K=5 | 0.781 +- 0.006 | -0.148 +- 0.006 | +0.006 +- 0.005 | 0.764 | 8 | 0.001 | yes |
| phiplus K=7 | 0.742 +- 0.006 | -0.188 +- 0.007 | -0.034 +- 0.005 | 0.726 | 8 | 0.001 | yes |
| phiplus K=9 | 0.717 +- 0.006 | -0.212 +- 0.007 | -0.058 +- 0.005 | 0.700 | 6 | 0.001 | yes |
| phiplus K=12 | 0.706 +- 0.006 | -0.223 +- 0.007 | -0.069 +- 0.006 | 0.693 | 4 | 0.0316 | no |
| phiplus K=16 | 0.702 +- 0.006 | -0.227 +- 0.007 | -0.073 +- 0.005 | 0.694 | 8 | 0.0316 | no |
| phiplus K=20 | 0.705 +- 0.006 | -0.225 +- 0.007 | -0.071 +- 0.005 | 0.696 | 8 | 5.62 | no |
| phiplus K=26 | 0.707 +- 0.006 | -0.222 +- 0.007 | -0.069 +- 0.005 | 0.701 | 8 | 31.6 | no |

Inner-CV-best K per variant (x+y): {'indep': 12, 'phi': 12, 'phiplus': 12}.

Paired differences w.r.t. `phi` at its CV-best K: indep: -0.003 +- 0.003, phiplus: +0.001 +- 0.003, cells10: +0.121 +- 0.006, cells8: +0.070 +- 0.005

**y view**; the spline rows use K_S and lambda chosen by inner CV (K_S, lambda in the last columns); "d vs pass1" and "d vs cells8" are s68 differences with paired-bootstrap errors (400 resamples of test events; the two constants are not refit inside the bootstrap).

| arm | s68 (deg) | d vs pass1 | d vs cells8 | CV s68 (train) | K_S | lambda | lambda at grid edge |
|---|---|---|---|---|---|---|---|
| pass1 | 0.809 +- 0.008 | +0.000 +- 0.000 | +0.157 +- 0.007 | | | | |
| const | 0.934 +- 0.008 | +0.126 +- 0.008 | +0.283 +- 0.008 | | | | |
| pass1c | 0.801 +- 0.008 | -0.008 +- 0.003 | +0.149 +- 0.007 | | | | |
| cells4 | 0.730 +- 0.007 | -0.078 +- 0.006 | +0.079 +- 0.004 | | | | |
| cells6 | 0.679 +- 0.006 | -0.129 +- 0.006 | +0.028 +- 0.003 | | | | |
| cells8 | 0.651 +- 0.005 | -0.157 +- 0.007 | +0.000 +- 0.000 | | | | |
| cells10 | 0.690 +- 0.007 | -0.119 +- 0.007 | +0.038 +- 0.005 | | | | |
| indep K=5 | 0.678 +- 0.006 | -0.131 +- 0.007 | +0.027 +- 0.004 | 0.664 | 8 | 0.001 | yes |
| indep K=7 | 0.654 +- 0.005 | -0.154 +- 0.007 | +0.003 +- 0.004 | 0.644 | 8 | 0.001 | yes |
| indep K=9 | 0.627 +- 0.006 | -0.182 +- 0.008 | -0.025 +- 0.004 | 0.614 | 6 | 0.001 | yes |
| indep K=12 | 0.616 +- 0.006 | -0.193 +- 0.008 | -0.036 +- 0.004 | 0.598 | 8 | 0.00562 | no |
| indep K=16 | 0.615 +- 0.006 | -0.193 +- 0.008 | -0.036 +- 0.004 | 0.598 | 8 | 0.178 | no |
| indep K=20 | 0.617 +- 0.005 | -0.192 +- 0.008 | -0.034 +- 0.004 | 0.600 | 8 | 0.178 | no |
| indep K=26 | 0.617 +- 0.006 | -0.192 +- 0.008 | -0.035 +- 0.005 | 0.599 | 8 | 1 | no |
| phi K=5 | 0.679 +- 0.006 | -0.130 +- 0.007 | +0.027 +- 0.004 | 0.675 | 8 | 0.00562 | no |
| phi K=7 | 0.657 +- 0.006 | -0.152 +- 0.007 | +0.005 +- 0.004 | 0.643 | 8 | 0.001 | yes |
| phi K=9 | 0.626 +- 0.006 | -0.182 +- 0.008 | -0.025 +- 0.004 | 0.614 | 8 | 0.001 | yes |
| phi K=12 | 0.618 +- 0.006 | -0.191 +- 0.008 | -0.034 +- 0.004 | 0.593 | 8 | 0.0316 | no |
| phi K=16 | 0.619 +- 0.006 | -0.190 +- 0.008 | -0.033 +- 0.005 | 0.606 | 8 | 0.001 | yes |
| phi K=20 | 0.626 +- 0.005 | -0.183 +- 0.008 | -0.026 +- 0.004 | 0.605 | 8 | 0.001 | yes |
| phi K=26 | 0.637 +- 0.005 | -0.171 +- 0.008 | -0.014 +- 0.005 | 0.617 | 8 | 1 | no |
| phiplus K=5 | 0.675 +- 0.006 | -0.133 +- 0.007 | +0.024 +- 0.004 | 0.662 | 8 | 0.001 | yes |
| phiplus K=7 | 0.651 +- 0.005 | -0.158 +- 0.007 | -0.001 +- 0.004 | 0.638 | 8 | 0.001 | yes |
| phiplus K=9 | 0.626 +- 0.006 | -0.183 +- 0.008 | -0.025 +- 0.004 | 0.611 | 6 | 0.001 | yes |
| phiplus K=12 | 0.622 +- 0.006 | -0.187 +- 0.008 | -0.030 +- 0.005 | 0.592 | 4 | 0.0316 | no |
| phiplus K=16 | 0.618 +- 0.005 | -0.190 +- 0.008 | -0.033 +- 0.005 | 0.600 | 8 | 0.0316 | no |
| phiplus K=20 | 0.626 +- 0.005 | -0.183 +- 0.008 | -0.025 +- 0.005 | 0.607 | 8 | 5.62 | no |
| phiplus K=26 | 0.629 +- 0.005 | -0.179 +- 0.007 | -0.022 +- 0.005 | 0.611 | 8 | 31.6 | no |

Inner-CV-best K per variant (x+y): {'indep': 12, 'phi': 12, 'phiplus': 12}.

Paired differences w.r.t. `phi` at its CV-best K: indep: -0.002 +- 0.003, phiplus: +0.004 +- 0.002, cells10: +0.072 +- 0.005, cells8: +0.034 +- 0.004

### train run2 -> test run1  (n_test = 4654)

**x view**; the spline rows use K_S and lambda chosen by inner CV (K_S, lambda in the last columns); "d vs pass1" and "d vs cells8" are s68 differences with paired-bootstrap errors (400 resamples of test events; the two constants are not refit inside the bootstrap).

| arm | s68 (deg) | d vs pass1 | d vs cells8 | CV s68 (train) | K_S | lambda | lambda at grid edge |
|---|---|---|---|---|---|---|---|
| pass1 | 0.905 +- 0.013 | +0.000 +- 0.000 | +0.169 +- 0.011 | | | | |
| const | 1.034 +- 0.017 | +0.129 +- 0.011 | +0.298 +- 0.015 | | | | |
| pass1c | 0.906 +- 0.013 | +0.001 +- 0.001 | +0.170 +- 0.011 | | | | |
| cells4 | 0.816 +- 0.013 | -0.090 +- 0.009 | +0.079 +- 0.009 | | | | |
| cells6 | 0.764 +- 0.012 | -0.141 +- 0.010 | +0.028 +- 0.006 | | | | |
| cells8 | 0.736 +- 0.012 | -0.169 +- 0.011 | +0.000 +- 0.000 | | | | |
| cells10 | 0.706 +- 0.011 | -0.200 +- 0.012 | -0.031 +- 0.006 | | | | |
| indep K=5 | 0.765 +- 0.011 | -0.141 +- 0.012 | +0.028 +- 0.008 | 0.781 | 8 | 0.001 | yes |
| indep K=7 | 0.720 +- 0.010 | -0.186 +- 0.012 | -0.017 +- 0.008 | 0.738 | 8 | 0.001 | yes |
| indep K=9 | 0.696 +- 0.010 | -0.210 +- 0.013 | -0.040 +- 0.009 | 0.712 | 8 | 0.001 | yes |
| indep K=12 | 0.667 +- 0.010 | -0.239 +- 0.013 | -0.070 +- 0.009 | 0.693 | 8 | 0.001 | yes |
| indep K=16 | 0.679 +- 0.012 | -0.226 +- 0.013 | -0.057 +- 0.009 | 0.695 | 6 | 0.0316 | no |
| indep K=20 | 0.677 +- 0.010 | -0.228 +- 0.012 | -0.059 +- 0.009 | 0.696 | 8 | 0.178 | no |
| indep K=26 | 0.679 +- 0.010 | -0.226 +- 0.012 | -0.057 +- 0.008 | 0.698 | 8 | 1 | no |
| phi K=5 | 0.786 +- 0.012 | -0.119 +- 0.011 | +0.050 +- 0.008 | 0.807 | 8 | 0.00562 | no |
| phi K=7 | 0.738 +- 0.011 | -0.168 +- 0.012 | +0.002 +- 0.008 | 0.766 | 8 | 0.001 | yes |
| phi K=9 | 0.691 +- 0.010 | -0.215 +- 0.012 | -0.046 +- 0.009 | 0.716 | 8 | 0.001 | yes |
| phi K=12 | 0.668 +- 0.010 | -0.237 +- 0.012 | -0.068 +- 0.009 | 0.697 | 8 | 0.001 | yes |
| phi K=16 | 0.672 +- 0.011 | -0.234 +- 0.013 | -0.064 +- 0.009 | 0.695 | 8 | 0.00562 | no |
| phi K=20 | 0.671 +- 0.010 | -0.234 +- 0.013 | -0.065 +- 0.009 | 0.694 | 8 | 0.001 | yes |
| phi K=26 | 0.680 +- 0.011 | -0.225 +- 0.014 | -0.056 +- 0.010 | 0.698 | 8 | 0.00562 | no |
| phiplus K=5 | 0.757 +- 0.014 | -0.149 +- 0.012 | +0.020 +- 0.008 | 0.776 | 8 | 0.001 | yes |
| phiplus K=7 | 0.711 +- 0.011 | -0.195 +- 0.013 | -0.026 +- 0.009 | 0.733 | 8 | 0.001 | yes |
| phiplus K=9 | 0.684 +- 0.010 | -0.222 +- 0.013 | -0.052 +- 0.008 | 0.707 | 8 | 0.001 | yes |
| phiplus K=12 | 0.671 +- 0.010 | -0.234 +- 0.013 | -0.065 +- 0.009 | 0.696 | 8 | 0.001 | yes |
| phiplus K=16 | 0.671 +- 0.010 | -0.234 +- 0.013 | -0.065 +- 0.009 | 0.695 | 8 | 0.001 | yes |
| phiplus K=20 | 0.674 +- 0.011 | -0.231 +- 0.013 | -0.062 +- 0.010 | 0.693 | 8 | 0.001 | yes |
| phiplus K=26 | 0.680 +- 0.011 | -0.225 +- 0.013 | -0.056 +- 0.009 | 0.698 | 8 | 0.00562 | no |

Inner-CV-best K per variant (x+y): {'indep': 12, 'phi': 16, 'phiplus': 16}.

Paired differences w.r.t. `phi` at its CV-best K: indep: -0.005 +- 0.005, phiplus: -0.001 +- 0.004, cells10: +0.034 +- 0.008, cells8: +0.064 +- 0.009

**y view**; the spline rows use K_S and lambda chosen by inner CV (K_S, lambda in the last columns); "d vs pass1" and "d vs cells8" are s68 differences with paired-bootstrap errors (400 resamples of test events; the two constants are not refit inside the bootstrap).

| arm | s68 (deg) | d vs pass1 | d vs cells8 | CV s68 (train) | K_S | lambda | lambda at grid edge |
|---|---|---|---|---|---|---|---|
| pass1 | 0.796 +- 0.013 | +0.000 +- 0.000 | +0.169 +- 0.011 | | | | |
| const | 0.932 +- 0.012 | +0.136 +- 0.011 | +0.305 +- 0.012 | | | | |
| pass1c | 0.790 +- 0.013 | -0.007 +- 0.003 | +0.163 +- 0.011 | | | | |
| cells4 | 0.717 +- 0.012 | -0.079 +- 0.008 | +0.090 +- 0.008 | | | | |
| cells6 | 0.661 +- 0.009 | -0.135 +- 0.010 | +0.034 +- 0.006 | | | | |
| cells8 | 0.627 +- 0.009 | -0.169 +- 0.011 | +0.000 +- 0.000 | | | | |
| cells10 | 0.610 +- 0.010 | -0.186 +- 0.011 | -0.017 +- 0.005 | | | | |
| indep K=5 | 0.676 +- 0.011 | -0.120 +- 0.012 | +0.049 +- 0.007 | 0.677 | 8 | 0.001 | yes |
| indep K=7 | 0.638 +- 0.011 | -0.159 +- 0.012 | +0.011 +- 0.007 | 0.650 | 8 | 0.001 | yes |
| indep K=9 | 0.612 +- 0.010 | -0.184 +- 0.013 | -0.015 +- 0.007 | 0.621 | 8 | 0.001 | yes |
| indep K=12 | 0.592 +- 0.008 | -0.204 +- 0.012 | -0.035 +- 0.007 | 0.601 | 8 | 0.001 | yes |
| indep K=16 | 0.593 +- 0.009 | -0.204 +- 0.013 | -0.034 +- 0.007 | 0.606 | 6 | 0.0316 | no |
| indep K=20 | 0.590 +- 0.010 | -0.206 +- 0.013 | -0.037 +- 0.007 | 0.606 | 8 | 0.178 | no |
| indep K=26 | 0.587 +- 0.010 | -0.209 +- 0.012 | -0.039 +- 0.007 | 0.611 | 8 | 1 | no |
| phi K=5 | 0.682 +- 0.012 | -0.114 +- 0.012 | +0.055 +- 0.008 | 0.678 | 8 | 0.00562 | no |
| phi K=7 | 0.642 +- 0.010 | -0.154 +- 0.012 | +0.015 +- 0.007 | 0.650 | 8 | 0.001 | yes |
| phi K=9 | 0.607 +- 0.012 | -0.189 +- 0.013 | -0.019 +- 0.008 | 0.620 | 8 | 0.001 | yes |
| phi K=12 | 0.591 +- 0.010 | -0.206 +- 0.013 | -0.036 +- 0.007 | 0.609 | 8 | 0.001 | yes |
| phi K=16 | 0.587 +- 0.009 | -0.210 +- 0.013 | -0.040 +- 0.007 | 0.605 | 8 | 0.00562 | no |
| phi K=20 | 0.591 +- 0.010 | -0.206 +- 0.013 | -0.036 +- 0.007 | 0.607 | 8 | 0.001 | yes |
| phi K=26 | 0.600 +- 0.009 | -0.196 +- 0.013 | -0.026 +- 0.007 | 0.614 | 8 | 0.00562 | no |
| phiplus K=5 | 0.668 +- 0.011 | -0.128 +- 0.012 | +0.042 +- 0.007 | 0.673 | 8 | 0.001 | yes |
| phiplus K=7 | 0.639 +- 0.011 | -0.157 +- 0.012 | +0.012 +- 0.007 | 0.644 | 8 | 0.001 | yes |
| phiplus K=9 | 0.603 +- 0.010 | -0.193 +- 0.013 | -0.023 +- 0.007 | 0.618 | 8 | 0.001 | yes |
| phiplus K=12 | 0.591 +- 0.009 | -0.206 +- 0.013 | -0.036 +- 0.007 | 0.608 | 8 | 0.001 | yes |
| phiplus K=16 | 0.589 +- 0.009 | -0.207 +- 0.013 | -0.038 +- 0.007 | 0.606 | 8 | 0.001 | yes |
| phiplus K=20 | 0.593 +- 0.009 | -0.204 +- 0.013 | -0.034 +- 0.007 | 0.611 | 8 | 0.001 | yes |
| phiplus K=26 | 0.597 +- 0.010 | -0.200 +- 0.013 | -0.030 +- 0.008 | 0.615 | 8 | 0.00562 | no |

Inner-CV-best K per variant (x+y): {'indep': 12, 'phi': 16, 'phiplus': 16}.

Paired differences w.r.t. `phi` at its CV-best K: indep: +0.005 +- 0.004, phiplus: +0.002 +- 0.003, cells10: +0.023 +- 0.006, cells8: +0.040 +- 0.007

## Part 2: learning curve of the `phi` map (K and lambda re-chosen by inner CV at every N; K_S = 6)

**A: train = run 2 + random half of run 1, test = other half of run 1.** Mean over 4 random splits (s68 in deg, scatter between splits in brackets); cells 8x8 trained on the same N for comparison (nan where cells have < 40 events).

| N_train | n_test | K chosen | phi x | phi y | cells8 x | cells8 y |
|---|---|---|---|---|---|---|
| 1024 | 2327 | [12] | 0.711 (0.011) | 0.630 (0.005) | nan | nan |
| 2049 | 2327 | [9, 12, 16, 20] | 0.703 (0.014) | 0.626 (0.009) | 0.944 | 0.823 |
| 4098 | 2327 | [12, 16] | 0.688 (0.009) | 0.604 (0.013) | 0.745 | 0.649 |
| 8197 | 2327 | [16] | 0.680 (0.009) | 0.590 (0.006) | 0.735 | 0.628 |
| 16394 | 2327 | [16] | 0.674 (0.006) | 0.589 (0.007) | 0.736 | 0.626 |

**B: train = run 1 + random half of run 2, test = other half of run 2.** Mean over 4 random splits (s68 in deg, scatter between splits in brackets); cells 8x8 trained on the same N for comparison (nan where cells have < 40 events).

| N_train | n_test | K chosen | phi x | phi y | cells8 x | cells8 y |
|---|---|---|---|---|---|---|
| 730 | 7034 | [9, 12, 16, 20] | 0.735 (0.005) | 0.661 (0.009) | nan | nan |
| 1460 | 7034 | [9, 12] | 0.723 (0.008) | 0.641 (0.008) | 1.045 | 0.928 |
| 2921 | 7034 | [12, 16] | 0.709 (0.006) | 0.625 (0.003) | 0.855 | 0.706 |
| 5843 | 7034 | [16] | 0.702 (0.004) | 0.616 (0.003) | 0.760 | 0.650 |
| 11687 | 7034 | [12, 16] | 0.693 (0.007) | 0.611 (0.003) | 0.757 | 0.647 |

**U: union of both runs, 5-fold (a different held-out fold per seed), train subsampled.** Mean over 4 random splits (s68 in deg, scatter between splits in brackets); cells 8x8 trained on the same N for comparison (nan where cells have < 40 events).

| N_train | n_test | K chosen | phi x | phi y | cells8 x | cells8 y |
|---|---|---|---|---|---|---|
| 936 | 3745 | [12, 16] | 0.729 (0.009) | 0.645 (0.011) | nan | nan |
| 1872 | 3745 | [12, 16] | 0.717 (0.007) | 0.628 (0.010) | 0.982 | 0.871 |
| 3744 | 3745 | [12, 16] | 0.706 (0.005) | 0.616 (0.007) | 0.790 | 0.656 |
| 7488 | 3745 | [12, 16] | 0.701 (0.009) | 0.609 (0.008) | 0.756 | 0.641 |
| 14976 | 3745 | [16] | 0.694 (0.000) | 0.597 (0.000) | 0.754 | 0.639 |
| 14977 | 3744 | [16] | 0.694 (0.002) | 0.601 (0.011) | 0.756 | 0.635 |

Asymptote s68_inf from s68^2 = s68_inf^2 + c/N (deg), per split and fit window; mean over splits (+- scatter), and the spread between windows:

| set | view | fit window | s68_inf | scatter between splits | c (deg^2) | s68 at largest N |
|---|---|---|---|---|---|---|
| A | x | all N | 0.676 | 0.007 | 54.74 | 0.674 |
| A | x | N>=N_max/4 | 0.670 | 0.008 | 102.96 | 0.674 |
| A | x | 3 largest | 0.670 | 0.008 | 102.96 | 0.674 |
| B | x | all N | 0.696 | 0.006 | 43.93 | 0.693 |
| B | x | N>=N_max/4 | 0.689 | 0.008 | 83.53 | 0.693 |
| B | x | 3 largest | 0.689 | 0.008 | 83.53 | 0.693 |
| U | x | all N | 0.695 | 0.007 | 48.20 | 0.694 |
| U | x | N>=N_max/4 | 0.691 | 0.005 | 81.59 | 0.694 |
| U | x | 3 largest | 0.691 | 0.005 | 81.59 | 0.694 |

**x view: s68_inf = 0.685, range over sets and windows 0.670 - 0.696, split scatter 0.007.**

| A | y | all N | 0.590 | 0.008 | 57.73 | 0.589 |
| A | y | N>=N_max/4 | 0.582 | 0.004 | 102.51 | 0.589 |
| A | y | 3 largest | 0.582 | 0.004 | 102.51 | 0.589 |
| B | y | all N | 0.610 | 0.001 | 49.35 | 0.611 |
| B | y | N>=N_max/4 | 0.607 | 0.005 | 65.34 | 0.611 |
| B | y | 3 largest | 0.607 | 0.005 | 65.34 | 0.611 |
| U | y | all N | 0.602 | 0.007 | 52.60 | 0.600 |
| U | y | N>=N_max/4 | 0.596 | 0.010 | 92.89 | 0.600 |
| U | y | 3 largest | 0.596 | 0.010 | 92.89 | 0.600 |

**y view: s68_inf = 0.597, range over sets and windows 0.582 - 0.610, split scatter 0.006.**

## Part 3: robustness

### (a), (e) time splits and baselines (phi: K, lambda chosen by inner CV in the train part; K_S = 6; paired-bootstrap errors on differences, 400 resamples)

| split | n_train/n_test | K | view | pass1 | const | pass1+2 const | cells8 | phi | phi - pass1 | phi - cells8 |
|---|---|---|---|---|---|---|---|---|---|---|
| run1->run2 | 4654/14067 | 12 | x | 0.929 | 1.047 | 0.920 | 0.776 | 0.706 | -0.223 +- 0.007 | -0.070 +- 0.005 |
| run1->run2 | 4654/14067 | 12 | y | 0.809 | 0.934 | 0.801 | 0.651 | 0.617 | -0.191 +- 0.008 | -0.034 +- 0.005 |
| run2->run1 | 14067/4654 | 16 | x | 0.905 | 1.034 | 0.906 | 0.736 | 0.678 | -0.227 +- 0.013 | -0.058 +- 0.009 |
| run2->run1 | 14067/4654 | 16 | y | 0.796 | 0.932 | 0.790 | 0.627 | 0.588 | -0.208 +- 0.013 | -0.038 +- 0.007 |
| run1 first->second | 2327/2327 | 12 | x | 0.918 | 1.048 | 0.919 | 0.893 | 0.727 | -0.191 +- 0.018 | -0.165 +- 0.022 |
| run1 first->second | 2327/2327 | 12 | y | 0.800 | 0.937 | 0.786 | 0.785 | 0.614 | -0.187 +- 0.018 | -0.171 +- 0.018 |
| run1 second->first | 2327/2327 | 12 | x | 0.889 | 1.013 | 0.892 | 0.846 | 0.673 | -0.216 +- 0.018 | -0.173 +- 0.020 |
| run1 second->first | 2327/2327 | 12 | y | 0.790 | 0.928 | 0.792 | 0.769 | 0.591 | -0.199 +- 0.016 | -0.178 +- 0.017 |
| run1 oddfiles->evenfiles | 2038/2616 | 12 | x | 0.910 | 1.017 | 0.913 | 0.904 | 0.697 | -0.212 +- 0.018 | -0.207 +- 0.019 |
| run1 oddfiles->evenfiles | 2038/2616 | 12 | y | 0.789 | 0.928 | 0.790 | 0.816 | 0.607 | -0.182 +- 0.016 | -0.209 +- 0.016 |
| run1 evenfiles->oddfiles | 2616/2038 | 12 | x | 0.894 | 1.039 | 0.899 | 0.859 | 0.697 | -0.197 +- 0.019 | -0.161 +- 0.017 |
| run1 evenfiles->oddfiles | 2616/2038 | 12 | y | 0.804 | 0.937 | 0.785 | 0.731 | 0.588 | -0.217 +- 0.020 | -0.143 +- 0.020 |
| run1 random-half-A->B | 2327/2327 | 12 | x | 0.902 | 1.039 | 0.908 | 0.916 | 0.685 | -0.217 +- 0.019 | -0.231 +- 0.019 |
| run1 random-half-A->B | 2327/2327 | 12 | y | 0.787 | 0.937 | 0.781 | 0.779 | 0.605 | -0.182 +- 0.015 | -0.174 +- 0.017 |
| run2 first->second | 7033/7034 | 16 | x | 0.937 | 1.047 | 0.928 | 0.777 | 0.708 | -0.229 +- 0.011 | -0.068 +- 0.008 |
| run2 first->second | 7033/7034 | 16 | y | 0.809 | 0.926 | 0.803 | 0.646 | 0.615 | -0.195 +- 0.009 | -0.032 +- 0.006 |
| run2 second->first | 7034/7033 | 16 | x | 0.920 | 1.042 | 0.917 | 0.752 | 0.693 | -0.227 +- 0.011 | -0.058 +- 0.008 |
| run2 second->first | 7034/7033 | 16 | y | 0.809 | 0.941 | 0.799 | 0.658 | 0.616 | -0.193 +- 0.010 | -0.042 +- 0.007 |
| run2 oddfiles->evenfiles | 6922/7145 | 12 | x | 0.942 | 1.055 | 0.929 | 0.776 | 0.709 | -0.233 +- 0.011 | -0.067 +- 0.007 |
| run2 oddfiles->evenfiles | 6922/7145 | 12 | y | 0.809 | 0.932 | 0.798 | 0.660 | 0.619 | -0.191 +- 0.010 | -0.041 +- 0.007 |
| run2 evenfiles->oddfiles | 7145/6922 | 16 | x | 0.920 | 1.036 | 0.908 | 0.754 | 0.685 | -0.234 +- 0.010 | -0.068 +- 0.008 |
| run2 evenfiles->oddfiles | 7145/6922 | 16 | y | 0.807 | 0.939 | 0.802 | 0.647 | 0.610 | -0.198 +- 0.011 | -0.037 +- 0.006 |
| run2 random-half-A->B | 7034/7033 | 12 | x | 0.928 | 1.040 | 0.916 | 0.766 | 0.704 | -0.224 +- 0.011 | -0.063 +- 0.007 |
| run2 random-half-A->B | 7034/7033 | 12 | y | 0.819 | 0.945 | 0.810 | 0.654 | 0.616 | -0.202 +- 0.011 | -0.038 +- 0.007 |

### (b) where the gain comes from (cross-run, both directions)

Tail columns: fraction of events with |theta - theta_ref| > 2 deg and > 4 deg (in per cent).

**run1->run2**

| view, subset | n | pass1 s68 | cells8 s68 | phi s68 | pass1 >2/>4 deg | cells8 >2/>4 | phi >2/>4 |
|---|---|---|---|---|---|---|---|
| x |tan| 0-0.1 | 6539 | 0.959 | 0.808 | 0.699 | 4.94 / 0.58 | 3.55 / 0.60 | 2.23 / 0.60 |
| x 0.1-0.2 | 5033 | 0.906 | 0.734 | 0.676 | 3.58 / 0.60 | 2.30 / 0.62 | 1.69 / 0.62 |
| x 0.2-0.3 | 2112 | 0.876 | 0.770 | 0.751 | 3.74 / 0.43 | 2.32 / 0.43 | 2.27 / 0.43 |
| x 0.3-0.6 | 383 | 0.987 | 0.910 | 0.902 | 4.96 / 0.26 | 2.87 / 0.26 | 1.83 / 0.26 |
| x centre | 4699 | 0.890 | 0.728 | 0.680 | 3.21 / 0.45 | 1.87 / 0.43 | 1.51 / 0.45 |
| x outer | 9368 | 0.950 | 0.801 | 0.719 | 4.80 / 0.61 | 3.42 / 0.64 | 2.30 / 0.63 |
| x all | 14067 | 0.929 | 0.776 | 0.706 | 4.27 / 0.55 | 2.90 / 0.57 | 2.03 / 0.57 |
| y |tan| 0-0.1 | 7625 | 0.799 | 0.647 | 0.611 | 2.75 / 0.35 | 1.64 / 0.34 | 1.34 / 0.33 |
| y 0.1-0.2 | 5060 | 0.781 | 0.630 | 0.596 | 2.35 / 0.55 | 1.38 / 0.51 | 1.23 / 0.57 |
| y 0.2-0.3 | 1207 | 0.919 | 0.751 | 0.724 | 4.39 / 0.33 | 2.65 / 0.33 | 2.49 / 0.33 |
| y 0.3-0.6 | 175 | 1.012 | 0.923 | 0.855 | 4.00 / 0.00 | 2.86 / 0.00 | 2.86 / 0.00 |
| y centre | 4699 | 0.799 | 0.646 | 0.602 | 2.04 / 0.38 | 1.40 / 0.40 | 1.43 / 0.40 |
| y outer | 9368 | 0.813 | 0.654 | 0.627 | 3.13 / 0.44 | 1.77 / 0.39 | 1.41 / 0.42 |
| y all | 14067 | 0.809 | 0.651 | 0.617 | 2.77 / 0.42 | 1.65 / 0.40 | 1.41 / 0.41 |

**run2->run1**

| view, subset | n | pass1 s68 | cells8 s68 | phi s68 | pass1 >2/>4 deg | cells8 >2/>4 | phi >2/>4 |
|---|---|---|---|---|---|---|---|
| x |tan| 0-0.1 | 2076 | 0.922 | 0.743 | 0.680 | 4.87 / 0.39 | 2.75 / 0.39 | 2.02 / 0.39 |
| x 0.1-0.2 | 1692 | 0.873 | 0.677 | 0.615 | 3.19 / 0.41 | 1.89 / 0.30 | 1.48 / 0.30 |
| x 0.2-0.3 | 754 | 0.950 | 0.780 | 0.775 | 3.71 / 0.00 | 2.12 / 0.00 | 1.72 / 0.00 |
| x 0.3-0.6 | 132 | 1.035 | 0.915 | 0.857 | 5.30 / 0.00 | 4.55 / 0.00 | 3.03 / 0.00 |
| x centre | 1498 | 0.929 | 0.737 | 0.694 | 3.87 / 0.47 | 2.54 / 0.33 | 2.14 / 0.40 |
| x outer | 3156 | 0.903 | 0.736 | 0.673 | 4.18 / 0.25 | 2.31 / 0.25 | 1.65 / 0.22 |
| x all | 4654 | 0.905 | 0.736 | 0.678 | 4.08 / 0.32 | 2.39 / 0.28 | 1.80 / 0.28 |
| y |tan| 0-0.1 | 2545 | 0.786 | 0.608 | 0.563 | 2.63 / 0.43 | 1.30 / 0.39 | 1.18 / 0.39 |
| y 0.1-0.2 | 1600 | 0.778 | 0.627 | 0.595 | 2.75 / 0.62 | 1.38 / 0.62 | 1.38 / 0.62 |
| y 0.2-0.3 | 450 | 0.861 | 0.705 | 0.684 | 2.44 / 0.00 | 1.56 / 0.00 | 1.33 / 0.00 |
| y 0.3-0.6 | 59 | 0.906 | 0.721 | 0.756 | 8.47 / 1.69 | 8.47 / 1.69 | 8.47 / 1.69 |
| y centre | 1498 | 0.785 | 0.630 | 0.569 | 2.14 / 0.40 | 1.40 / 0.33 | 1.27 / 0.33 |
| y outer | 3156 | 0.795 | 0.628 | 0.596 | 3.01 / 0.51 | 1.46 / 0.51 | 1.39 / 0.51 |
| y all | 4654 | 0.796 | 0.627 | 0.588 | 2.73 / 0.47 | 1.44 / 0.45 | 1.35 / 0.45 |

### (c), (d) index by the reference position; mid-gap position residual

| direction | view | phi, recon. index | phi, ref index | cells8 recon | cells8 ref | position s68 (mm) pass1 | pass1+2c | cells8 | phi |
|---|---|---|---|---|---|---|---|---|---|
| run1->run2 | x | 0.706 | 0.706 | 0.776 | 0.774 | 0.554 | 0.548 | 0.531 | 0.522 |
| run1->run2 | y | 0.617 | 0.618 | 0.651 | 0.652 | 0.542 | 0.537 | 0.521 | 0.517 |
| run2->run1 | x | 0.678 | 0.677 | 0.736 | 0.735 | 0.558 | 0.559 | 0.526 | 0.522 |
| run2->run1 | y | 0.588 | 0.590 | 0.627 | 0.625 | 0.507 | 0.506 | 0.484 | 0.482 |

## Part 4: final map of the union (CV: 5 time blocks over run 1 then run 2)

| variant | K | K_S | lambda | CV s68 x | CV s68 y | lambda at edge |
|---|---|---|---|---|---|---|
| phi | 16 | 8 | 0.001 | 0.686 | 0.599 | True |
| phi | 20 | 8 | 0.001 | 0.688 | 0.601 | True |
| indep | 12 | 6 | 0.001 | 0.690 | 0.600 | True |
| phi | 16 | 6 | 0.001 | 0.689 | 0.601 | True |
| phiplus | 16 | 6 | 0.001 | 0.690 | 0.601 | True |
| phiplus | 20 | 6 | 0.001 | 0.691 | 0.601 | True |
| phi | 12 | 8 | 0.001 | 0.689 | 0.605 | True |
| indep | 16 | 6 | 0.00562 | 0.690 | 0.603 | False |
| phi | 20 | 6 | 0.001 | 0.693 | 0.601 | True |
| phi | 16 | 4 | 0.001 | 0.692 | 0.604 | True |
| phi | 12 | 6 | 0.00562 | 0.693 | 0.605 | False |
| indep | 20 | 6 | 0.178 | 0.693 | 0.605 | False |
| indep | 26 | 6 | 1 | 0.694 | 0.605 | False |
| phi | 26 | 8 | 0.001 | 0.692 | 0.607 | True |
| phi | 20 | 4 | 0.001 | 0.695 | 0.604 | True |
| phiplus | 12 | 6 | 0.001 | 0.695 | 0.605 | True |
| phi | 12 | 4 | 0.001 | 0.696 | 0.607 | True |
| phi | 26 | 6 | 0.00562 | 0.696 | 0.607 | False |
| phiplus | 26 | 6 | 0.00562 | 0.696 | 0.608 | False |
| phi | 26 | 4 | 0.001 | 0.700 | 0.608 | True |
| phiplus | 9 | 6 | 0.001 | 0.703 | 0.617 | True |
| phi | 9 | 8 | 0.001 | 0.711 | 0.618 | True |
| indep | 9 | 6 | 0.001 | 0.711 | 0.620 | True |
| phi | 9 | 6 | 0.00562 | 0.717 | 0.621 | False |
| phi | 9 | 4 | 0.001 | 0.720 | 0.623 | True |
| phi | 16 | 0 | 5.62 | 0.724 | 0.620 | False |
| phi | 20 | 0 | 5.62 | 0.723 | 0.622 | False |
| phi | 26 | 0 | 31.6 | 0.726 | 0.623 | False |
| phi | 12 | 0 | 1 | 0.724 | 0.626 | False |
| phi | 9 | 0 | 0.178 | 0.748 | 0.642 | False |

Reference arms in the same 5-block CV (maps from the other four blocks): pass1: x 0.923 / y 0.805, pass1c: x 0.920 / y 0.801, cells8: x 0.753 / y 0.637, cells6: x 0.778 / y 0.673, cells10: x 0.727 / y 0.623

Shipped: `phi|16|8`, evaluator check max |difference| = 9.24e-14; file `final_map_det3.json`.