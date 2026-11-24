# Task D: does the fine angle map work on the other four chambers, and is the gain real without the M3 telescope?

Scripts: `rerun/paper1_gap/audit/d_lib.py`, `d_00_rows.py`, `d_01_part1.py`, `d_02_part2.py`, `d_02b_regress.py`, `d_02c_var.py`, `d_02d_diff.py`, `d_03_maps.py`, `d_04_part34.py`, `d_05_part1_tables.py`, `d_06_part2_tables.py`, `d_07_part34_tables.py`.
Numbers: `part1.json`, `part2*.json`, `part34.json`, `rows.json` in this directory. Figures: `fig_part1.png`, `fig_part2.png`, `fig_part34.png`.

Conventions. Selection = `GM.load_table` with each chamber's own active area (`TRUE_ACTIVE_BY_DET`, 30 mm margin). Views x, y in Part 1, 3, 4 are the chamber views. In Part 2 the axes "X" and "Y" are the M3 axes (M3 x is measured mostly by the chamber y view, M3 y by the chamber x view; all chambers sit at about 90 deg). The loader reproduces chamber A's `track_grid.json` to 0.001 deg or better (table at the end; N differs by 1 event for p2, 14066 against 14067). Rows whose train set is small (det2 short, det6 short, det7 short, det4) give maps with many empty cells and are only quoted as such.

## Verdicts

1. **Cross-validated gain per chamber** (s68 change of the w0+v map against the pass-1 angle, deg; train set with at least 4000 events where one exists).
   - B (det2, same conditions, 11k train events): X -0.18 / Y -0.18 at n=6, -0.225 / -0.209 at n=8, -0.237 / -0.230 at n=10 (errors 0.013-0.016), i.e. 19-24 % of 0.99 / 0.97 deg. Still falling at n=10 (about 110 events per cell). With only 2.2k train events it saturates at n=4-6 (-0.12 / -0.12) and then overfits (n=10 is worse than pass 1 by +0.11 / +0.15).
   - A (det3, same slot and HV, 14k train): X -0.141 / Y -0.135 at n=6, -0.169 / -0.169 at n=8, -0.199 / -0.185 at n=10 (reproduces the committed numbers). With the 4.6k-event sat row as train the optimum is n=8 (-0.154 / -0.157) and n=10 overfits (-0.104 / -0.119).
   - D (det7): Y view -0.147±0.019 at n=6 (14 %, train 2.7k events); X view -0.11±0.04 at n=6 on the full area, and -0.09±0.03 (n=4) on the good half x >= 200 mm; no gain beyond n=4-6 on X. Both are limited by 1-2.7k train events.
   - C (det6): X -0.095±0.026 at n=4 and -0.069±0.029 at n=6 (5-7 %), Y -0.035±0.036 (not significant). The w0-only arm gives nothing.
   - E (det4): X none (+0.1 to -0.05), Y -0.12±0.04 at n=4-6 for 6-24 to 6-23 only; s68 is 1.5-2.4 deg, so no map of this size can matter.
   - Against pass 1, only B, A and (Y of) D gain clearly; C gains a little; E gains nothing. (Constants-only angles are worse than pass 1 in B, A, C, E, because pass 1 already contains the bow block.)
   - **At a different mesh/drift condition the gain is lost.** Chamber A map from the 6-27/28 rows (490 V, drift 1000 V, top slot) applied to 6-25 (drift about 500 V, bottom slot): X -0.06 (n=4) then +0.05 at n=8; Y -0.12 (n=4) to -0.02 (n=10). The reverse (6-25 map to 6-27/28): -0.01 to -0.02 (X) and -0.01 to -0.02 (Y) at every n. For the same slot and HV the gain is -0.14 to -0.20. So the map is a property of the chamber at one mounting and one drift field, not a transferable chamber constant. Rows with a different mesh voltage in the same night do not exist; the nearest test is the drift field and slot change above.
   - Saturation: it depends on events per cell, not on the chamber. With about 110-180 train events per cell (B, A with 11-14k events at n=8-10) the gain is still growing slowly; with 40-70 events per cell (n=8 with 2-5k events) it saturates and then turns to overfitting.

2. **Gain without M3 (det6 - det7, 2176 matched events in long + longer): only partly confirmed.**
   - The matching is exact (M3 slopes of the two tables identical, max difference 0), and both chambers' `tan_ref` equals the M3 slope rotated by the row's `theta_align_deg` to 1e-16.
   - The naive budget fails. With w0+v maps n=6 on both chambers, s68(d)^2 falls by 0.78±0.11 (M3 x) and 0.77±0.21 deg^2 (M3 y); the falls of the two chambers against M3 add to 2.29±0.13 and 1.78±0.13. Observed minus predicted: +1.51±0.16 and +1.01±0.23. Mapping only det7 raises s68(d)^2 by +0.84±0.14 and +0.74±0.17 (predicted -0.64, -0.19). Mapping only det6 lowers it by 0.31±0.14 and 0.67±0.21 (predicted 1.65, 1.58). The same holds with the variance instead of s68^2 (clipped at 5 deg; observed minus predicted -1.18±0.07 and -0.61±0.08), so it is not an effect of non-Gaussian tails.
   - The reason is visible in the corrections themselves: the corrections of det6 and det7 are strongly correlated (r = 0.68 on M3 x, 0.14 on M3 y at n=6) and 79 % (M3 x) and 57 % (M3 y) of the variance of the correction is common to both chambers. A correction common to the two chambers cancels in d, whether it is a telescope error or a distortion that the two chambers share. d cannot test that part.
   - The part of the correction that differs between the chambers is tested and is real. The slope of d(const) on (D6 - D7), where D is the correction, is -0.98±0.10 (M3 x) and -0.47±0.19 (M3 y) at n=6 w0+v; -0.85±0.10 and -1.24±0.19 at n=8; -0.97±0.13 and -1.54±0.19 for w0 only n=6. A map of M3 origin would give 0, a real chamber distortion -1. The statistics are small, so the M3 y value at n=6 is 2.8 sigma from -1 on the low side and 1.4 sigma from -1 at n=8.
   - Verdict: the differential part of the gain (about a quarter of the correction variance on M3 x, a half on M3 y) is real without M3. For the common part (about 60-80 %) this data set has no M3-free test; the common part is the part that reaches the 16-21 % gain of chamber A. With the faulty half of det7 X removed the conclusion is the same but noisier (matched events 891).

3. **Is the structure chamber-bound? Partly; the data do not give a clean answer.**
   - Same chamber, same slot and conditions: the residual tilt maps (tau = w0/v minus the best plane, n=6) agree with r = 0.98-0.99 (A sat vs p2, ceiling 0.97-0.98), 0.89-0.95 (B), 0.83-0.85 (C), 0.92 (D). So the structure is a reproducible property of a chamber in one mounting.
   - Same chamber, other slot and drift field (A, 6-25 vs 6-27/28): r = 0.30-0.46 (X) and 0.44-0.64 (Y) against a ceiling of 0.95-0.99. A large part of the structure changes with the mounting or the drift field.
   - Different chambers in the same telescope slot (tau, n=6, telescope frame, best row pair): B-A +0.13±0.04 (X), +0.19±0.03 (Y); B-E +0.24±0.17, +0.47±0.18; B-D -0.17±0.13 (Y); A-E +0.65±0.19, +0.68±0.13; A-D -0.10±0.08 (Y); E-D -0.10±0.13 (Y); and in the bottom slot A(6-25)-C +0.49±0.05, +0.41±0.05. A mirrored map gives chance correlations between about -0.5 and +0.5 because the maps are smooth and have few independent cells, so only A-C and A-E stand out, and E has a noise ceiling of 0.8. The telescope-tied hypothesis predicts positive r for all pairs; the B-D, A-D, E-D pairs (D is the chamber with the damaged X half) are zero or negative, and B-A is small.
   - So: the fine structure is mostly chamber-bound (r near 1 within a mounting, near zero for 3 of 7 chamber pairs and 0.1-0.2 for a fourth), but the bottom-slot pair A-C (r about 0.45) and the smaller B-A and A-E values leave room for a part tied to the slot or telescope. The common plane (point 4) is the strongest candidate. We cannot separate the two with these data (chamber A is the only chamber seen in two slots).

4. **Curl test (n=6 planes through tau_x, tau_y): the planes are curl-free within errors on the chambers that are well measured.**
   - B long: tau_x/dW - tau_y/dU = +0.25±0.25 (1e-3 per 100 mm) against own-axis gradients of 12.2 and 12.8. A 6-27/6-28: +0.96±0.60 and +0.41±0.21 against 10.6 / 8.0-8.3. A 6-25: -0.25±0.35 against 10.4 / 20.1. C long: +1.80±0.63 (2.9 sigma, 8 % of the own-axis gradient) and C longer +1.10±1.04. D (x >= 200 mm): +1.6±1.1 (long) and +6.4±1.9 (longer, 3.3 sigma, 7 cells in X). E: 0.6±12.8 and -10±6 (uninformative).
   - Individual rows are within 2 sigma, apart from C long (2.9) and D longer (3.3). The cross gradient is below 10 % of the own-axis gradient in all chambers with errors under 1.
   - Own-axis gradients are positive in all chambers (tau = (w0/v) grows with position along its own axis), about 10-20e-3 per 100 mm; chamber A's Y gradient doubles between the top slot (8-8.3) and the bottom slot (20.1) while the X gradient stays (10.4-10.6).

## Per-row summary

| row | chamber (public) | N selected | mesh V | drift V | slot, z_x / z_y (mm) | theta_align (deg) | v_x / v_y (um/ns) | w0_x / w0_y (um/ns) | bundle v_drift (um/ns) |
|---|---|---|---|---|---|---|---|---|---|
| det2:long 6-22 long | det2 (B) | 10963 | 495 | 1000 | top, 712 / 713 | 89.2 | 39.1 / 39.6 | -0.222 / -0.215 | 38.3 |
| det2:longer 6-22 longer | det2 (B) | 2159 | 495 | 1000 | top, 712 / 713 | 89.2 | 38.7 / 39.5 | -0.181 / -0.223 | 39.3 |
| det2:short 6-22 short | det2 (B) | 1108 | 495 | 1000 | top, 712 / 713 | 89.2 | 39.0 / 39.4 | -0.202 / -0.238 | 39.0 |
| det3:d625 6-25 day | det3 (A) | 6279 | ~500 (not on disk) | 500 (ambiguous) | bottom, 242 / 243 | 89.75 | 22.0 / 21.8 | +0.073 / -0.210 | 22.4 |
| det3:sat 6-27 sat | det3 (A) | 4654 | 490 | 1000 | top, 714 / 714 | 89.45 | 37.0 / 36.5 | -0.067 / -0.157 | 35.8 |
| det3:p2 6-28 p2 | det3 (A) | 14066 | 490 | 1000 | top, 713 / 714 | 89.45 | 37.7 / 37.1 | -0.066 / -0.158 | 37.2 |
| det4:623 6-23 long | det4 (E) | 1610 | 495 | 600 | top, 715 / 715 | 89.25 | 35.6 / 38.7 | -0.361 / -0.514 | 35.9 |
| det4:624 6-24 day | det4 (E) | 3128 | 495 | 600 | top, 713 / 714 | 90.05 | 30.8 / 32.5 | -0.070 / -0.213 | 33.0 |
| det6:long 6-26 long | det6 (C) | 4981 | 495 | 700 | bottom, 242 / 242 | 89.7 | 25.0 / 26.3 | +0.071 / -0.415 | 24.2 |
| det6:longer 6-26 longer | det6 (C) | 2191 | 495 | 700 | bottom, 241 / 242 | 89.75 | 18.9 / 19.7 | +0.000 / -0.277 | 19.9 |
| det6:short 6-26 short | det6 (C) | 813 | 495 | 700 | bottom, 241 / 242 | 89.7 | 16.8 / 17.3 | -0.018 / -0.225 | 17.9 |
| det7:long 6-26 long | det7 (D) | 2680 (1032 with x>=200) | 495 | 700 | top, 712 / 713 | 90.1 | 38.0 (38.0) / 38.3 | -0.049 (+0.174 x>=200) / -0.179 | 36.8 |
| det7:longer 6-26 longer | det7 (D) | 1508 (562 with x>=200) | 495 | 700 | top, 713 / 713 | 90.1 | 35.4 (35.3) / 34.8 | -0.018 (+0.146 x>=200) / -0.118 | 33.9 |
| det7:short 6-26 short | det7 (D) | 949 (340 with x>=200) | 495 | 700 | top, 714 / 713 | 90.1 | 32.6 (32.9) / 32.5 | -0.045 (+0.085 x>=200) / -0.128 | 31.8 |
## Part 1: cross-validated map gain

Figure: fig_part1.png (w0+v map, change of s68 against pass 1, one line per group of pairs).

![](fig_part1.png)

### Summary per chamber (mean over the listed test rows; s68 in deg, change against pass 1 with error)

| group | view | N train | pass 1 | const | n=4 w0+v | n=6 w0+v | n=8 w0+v | n=10 w0+v | n=4 w0 | n=6 w0 | n=8 w0 | n=10 w0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B (det2) long -> longer/short | x | 10963, 10963 | 0.988 | 1.134 (+0.146±0.018) | 0.847 (-0.141±0.013) | 0.805 (-0.183±0.014) | 0.763 (-0.225±0.015) | 0.751 (-0.237±0.015) | 0.880 (-0.108±0.014) | 0.827 (-0.161±0.014) | 0.796 (-0.192±0.015) | 0.777 (-0.211±0.016) |
| B (det2) long -> longer/short | y | 10963, 10963 | 0.969 | 1.174 (+0.205±0.030) | 0.845 (-0.124±0.015) | 0.789 (-0.180±0.016) | 0.759 (-0.209±0.016) | 0.738 (-0.230±0.015) | 0.861 (-0.107±0.015) | 0.815 (-0.154±0.015) | 0.784 (-0.185±0.015) | 0.765 (-0.204±0.016) |
| B (det2) longer -> long (2.2k train) | x | 2159 | 0.956 | 1.082 (+0.126±0.010) | 0.835 (-0.121±0.007) | 0.823 (-0.133±0.007) | 0.991 (+0.035±0.010) | 1.066 (+0.110±0.010) | 0.860 (-0.096±0.007) | 0.853 (-0.103±0.007) | 1.016 (+0.060±0.011) | 1.069 (+0.113±0.010) |
| B (det2) longer -> long (2.2k train) | y | 2159 | 0.970 | 1.175 (+0.205±0.014) | 0.848 (-0.122±0.008) | 0.802 (-0.168±0.008) | 0.949 (-0.021±0.011) | 1.123 (+0.153±0.013) | 0.860 (-0.110±0.008) | 0.812 (-0.158±0.008) | 0.957 (-0.013±0.010) | 1.123 (+0.153±0.013) |
| A (det3) p2 -> sat (same slot, same HV) | x | 14066 | 0.905 | 1.034 (+0.129±0.011) | 0.816 (-0.090±0.010) | 0.764 (-0.141±0.010) | 0.736 (-0.169±0.011) | 0.706 (-0.199±0.012) | 0.857 (-0.048±0.010) | 0.809 (-0.096±0.012) | 0.779 (-0.127±0.012) | 0.758 (-0.148±0.013) |
| A (det3) p2 -> sat (same slot, same HV) | y | 14066 | 0.796 | 0.932 (+0.136±0.011) | 0.717 (-0.079±0.008) | 0.661 (-0.135±0.009) | 0.627 (-0.169±0.011) | 0.611 (-0.185±0.011) | 0.731 (-0.065±0.008) | 0.677 (-0.119±0.009) | 0.652 (-0.144±0.010) | 0.637 (-0.160±0.011) |
| A (det3) sat -> p2 (same slot, same HV) | x | 4654 | 0.929 | 1.047 (+0.118±0.006) | 0.837 (-0.092±0.005) | 0.790 (-0.139±0.006) | 0.775 (-0.154±0.007) | 0.825 (-0.104±0.007) | 0.860 (-0.069±0.006) | 0.817 (-0.112±0.006) | 0.807 (-0.122±0.007) | 0.852 (-0.077±0.007) |
| A (det3) sat -> p2 (same slot, same HV) | y | 4654 | 0.809 | 0.934 (+0.126±0.008) | 0.730 (-0.078±0.005) | 0.679 (-0.130±0.006) | 0.651 (-0.157±0.006) | 0.690 (-0.119±0.007) | 0.748 (-0.061±0.005) | 0.701 (-0.108±0.006) | 0.675 (-0.134±0.006) | 0.706 (-0.103±0.007) |
| A (det3) p2/sat -> 6-25 (other slot, drift 500 V) | x | 14066, 4654 | 0.876 | 1.065 (+0.189±0.007) | 0.812 (-0.064±0.008) | 0.873 (-0.003±0.009) | 0.926 (+0.049±0.010) | 0.978 (+0.102±0.010) | 0.817 (-0.059±0.009) | 0.859 (-0.018±0.009) | 0.899 (+0.023±0.010) | 0.964 (+0.087±0.009) |
| A (det3) p2/sat -> 6-25 (other slot, drift 500 V) | y | 14066, 4654 | 1.030 | 1.305 (+0.275±0.009) | 0.914 (-0.116±0.008) | 0.939 (-0.091±0.010) | 0.974 (-0.056±0.011) | 1.011 (-0.019±0.010) | 0.914 (-0.116±0.009) | 0.945 (-0.085±0.010) | 0.972 (-0.058±0.010) | 1.007 (-0.023±0.010) |
| A (det3) 6-25 -> sat/p2 (other slot, drift 500 V) | x | 6279, 6279 | 0.917 | 1.041 (+0.123±0.006) | 0.896 (-0.021±0.005) | 0.904 (-0.013±0.005) | 0.913 (-0.005±0.006) | 0.907 (-0.010±0.006) | 0.909 (-0.009±0.005) | 0.902 (-0.015±0.005) | 0.898 (-0.019±0.005) | 0.893 (-0.024±0.005) |
| A (det3) 6-25 -> sat/p2 (other slot, drift 500 V) | y | 6279, 6279 | 0.802 | 0.933 (+0.131±0.007) | 0.781 (-0.022±0.005) | 0.781 (-0.021±0.006) | 0.783 (-0.019±0.006) | 0.791 (-0.011±0.007) | 0.776 (-0.027±0.005) | 0.763 (-0.039±0.006) | 0.770 (-0.032±0.006) | 0.777 (-0.026±0.006) |
| E (det4) 6-24 -> 6-23 | x | 3128 | 2.405 | 2.617 (+0.212±0.044) | 2.355 (-0.050±0.038) | 2.397 (-0.007±0.053) | 2.425 (+0.020±0.046) | 2.518 (+0.113±0.045) | 2.450 (+0.045±0.039) | 2.439 (+0.034±0.044) | 2.514 (+0.109±0.043) | 2.555 (+0.150±0.044) |
| E (det4) 6-24 -> 6-23 | y | 3128 | 1.627 | 2.150 (+0.523±0.058) | 1.510 (-0.117±0.040) | 1.505 (-0.122±0.046) | 1.554 (-0.073±0.047) | 1.966 (+0.338±0.055) | 1.559 (-0.068±0.042) | 1.587 (-0.041±0.044) | 1.612 (-0.016±0.045) | 1.958 (+0.331±0.055) |
| E (det4) 6-23 -> 6-24 | x | 1610 | 2.141 | 2.284 (+0.144±0.033) | 2.196 (+0.055±0.033) | 2.236 (+0.095±0.035) | 2.298 (+0.157±0.036) | 2.281 (+0.140±0.033) | 2.202 (+0.061±0.037) | 2.238 (+0.098±0.038) | 2.283 (+0.142±0.032) | 2.285 (+0.144±0.033) |
| E (det4) 6-23 -> 6-24 | y | 1610 | 1.436 | 1.840 (+0.404±0.037) | 1.328 (-0.108±0.027) | 1.353 (-0.084±0.031) | 1.674 (+0.238±0.042) | 1.827 (+0.390±0.035) | 1.403 (-0.033±0.030) | 1.419 (-0.017±0.031) | 1.684 (+0.248±0.042) | 1.833 (+0.396±0.035) |
| C (det6) long -> longer/short | x | 4981, 4981 | 1.339 | 1.802 (+0.463±0.034) | 1.244 (-0.095±0.026) | 1.270 (-0.069±0.029) | 1.272 (-0.067±0.035) | 1.309 (-0.030±0.031) | 1.320 (-0.019±0.029) | 1.321 (-0.018±0.030) | 1.342 (+0.003±0.030) | 1.393 (+0.054±0.031) |
| C (det6) long -> longer/short | y | 4981, 4981 | 1.788 | 2.142 (+0.354±0.046) | 1.753 (-0.035±0.036) | 1.748 (-0.040±0.037) | 1.793 (+0.005±0.040) | 1.816 (+0.028±0.041) | 1.835 (+0.047±0.038) | 1.838 (+0.050±0.039) | 1.851 (+0.063±0.041) | 1.907 (+0.119±0.040) |
| C (det6) longer -> long | x | 2191 | 1.128 | 1.712 (+0.584±0.018) | 1.118 (-0.010±0.013) | 1.099 (-0.028±0.014) | 1.413 (+0.285±0.022) | n/a | 1.254 (+0.126±0.013) | 1.247 (+0.120±0.015) | 1.470 (+0.342±0.018) | n/a |
| C (det6) longer -> long | y | 2191 | 1.379 | 1.759 (+0.380±0.022) | 1.310 (-0.069±0.013) | 1.302 (-0.077±0.015) | 1.523 (+0.144±0.021) | n/a | 1.386 (+0.007±0.017) | 1.372 (-0.007±0.017) | 1.547 (+0.168±0.020) | n/a |
| D (det7) long -> longer/short | x | 2680, 2680 | 1.591 | 1.624 (+0.032±0.034) | 1.499 (-0.092±0.037) | 1.482 (-0.110±0.036) | 1.538 (-0.053±0.037) | 1.497 (-0.094±0.032) | 1.525 (-0.066±0.033) | 1.519 (-0.072±0.034) | 1.551 (-0.040±0.034) | 1.549 (-0.042±0.031) |
| D (det7) long -> longer/short | y | 2680, 2680 | 1.008 | 1.268 (+0.260±0.023) | 0.910 (-0.098±0.016) | 0.861 (-0.147±0.019) | 0.874 (-0.134±0.018) | 0.976 (-0.032±0.018) | 0.953 (-0.055±0.017) | 0.910 (-0.098±0.020) | 0.919 (-0.089±0.020) | 1.013 (+0.005±0.020) |
| D (det7) long -> longer/short, x >= 200 mm | x | 1032, 1032 | 0.949 | 0.918 (-0.031±0.031) | 0.857 (-0.093±0.030) | 0.871 (-0.079±0.032) | 0.921 (-0.028±0.032) | n/a | 0.857 (-0.092±0.029) | 0.880 (-0.069±0.031) | 0.923 (-0.027±0.030) | n/a |
### Reproduction of chamber A (committed track_grid.json)

| run (test) | view | arm | this loader | committed track_grid.json |
|---|---|---|---|---|
| run1 (det3:sat) | x | pass1 | 0.9054 | 0.9054 |
| run1 (det3:sat) | x | 4 w0+v | 0.8158 | 0.8158 |
| run1 (det3:sat) | x | 6 w0+v | 0.7642 | 0.7642 |
| run1 (det3:sat) | x | 8 w0+v | 0.7362 | 0.7363 |
| run1 (det3:sat) | x | 10 w0+v | 0.7059 | 0.7057 |
| run1 (det3:sat) | x | 8 w0 | 0.7786 | 0.7787 |
| run1 (det3:sat) | y | pass1 | 0.7963 | 0.7963 |
| run1 (det3:sat) | y | 4 w0+v | 0.7170 | 0.7171 |
| run1 (det3:sat) | y | 6 w0+v | 0.6611 | 0.6611 |
| run1 (det3:sat) | y | 8 w0+v | 0.6268 | 0.6268 |
| run1 (det3:sat) | y | 10 w0+v | 0.6110 | 0.6101 |
| run1 (det3:sat) | y | 8 w0 | 0.6524 | 0.6525 |
| run2 (det3:p2) | x | pass1 | 0.9291 | 0.9291 |
| run2 (det3:p2) | x | 4 w0+v | 0.8373 | 0.8373 |
| run2 (det3:p2) | x | 6 w0+v | 0.7898 | 0.7898 |
| run2 (det3:p2) | x | 8 w0+v | 0.7755 | 0.7755 |
| run2 (det3:p2) | x | 10 w0+v | 0.8255 | 0.8256 |
| run2 (det3:p2) | x | 8 w0 | 0.8070 | 0.8071 |
| run2 (det3:p2) | y | pass1 | 0.8087 | 0.8086 |
| run2 (det3:p2) | y | 4 w0+v | 0.7304 | 0.7305 |
| run2 (det3:p2) | y | 6 w0+v | 0.6791 | 0.6792 |
| run2 (det3:p2) | y | 8 w0+v | 0.6514 | 0.6515 |
| run2 (det3:p2) | y | 10 w0+v | 0.6897 | 0.6896 |
| run2 (det3:p2) | y | 8 w0 | 0.6748 | 0.6749 |
## Part 2: det6 - det7

![](fig_part2.png)

### Matching

| sub-run | train | N det6 | N det7 | matched | max abs diff of M3 slopes (both tables) | corr M3 slope x / y | max abs (rotated tan_ref - m3_tan), det6 / det7 | theta det6 / det7 | rms(tan_ref_x det6 - det7), chamber frame | rms after rotation to M3 |
|---|---|---|---|---|---|---|---|---|---|---|
| long | longer | 4981 | 2680 | 1472 | 0.0e+00 | 1.000000 / 1.000000 | 1e-16 / 6e-17 | 89.7 / 90.1 | 0.00083 | 2e-17 |
| longer | long | 2191 | 1508 | 704 | 0.0e+00 | 1.000000 / 1.000000 | 6e-17 / 6e-17 | 89.75 / 90.1 | 0.00073 | 2e-17 |
| short | long | 813 | 949 | 334 | 0.0e+00 | 1.000000 / 1.000000 | 6e-17 / 6e-17 | 89.7 / 90.1 | 0.00081 | 2e-17 |
### s68 and budget (X, Y = M3 axes)

#### all selected events, test sets: long, longer (matched events 2176)

| det6 / det7 angle | s68(d) X-axis | s68(d) Y-axis | s68(det6-M3) X / Y | s68(det7-M3) X / Y |
|---|---|---|---|---|
| pass 1 / pass 1 | 1.824±0.038 | 2.204±0.054 | 1.457 / 1.149 | 1.062 / 1.592 |
| constants / constants | 1.899±0.038 | 2.306±0.057 | 1.858 / 1.660 | 1.263 / 1.622 |
| w0+v map n=6 / w0+v map n=6 | 1.681±0.038 | 2.132±0.062 | 1.345 / 1.082 | 0.975 / 1.561 |
| w0+v map n=6 / constants (det6 mapped only) | 1.815±0.036 | 2.156±0.054 | 1.345 / 1.082 | 1.263 / 1.622 |
| constants / w0+v map n=6 (det7 mapped only) | 2.110±0.046 | 2.461±0.061 | 1.858 / 1.660 | 0.975 / 1.561 |
| w0+v map n=8 / w0+v map n=8 | 1.758±0.040 | 2.200±0.057 | 1.493 / 1.238 | 1.068 / 1.544 |
| w0+v map n=8 / constants (det6 mapped only) | 1.819±0.043 | 2.236±0.060 | 1.493 / 1.238 | 1.263 / 1.622 |
| constants / w0+v map n=8 (det7 mapped only) | 2.003±0.046 | 2.355±0.063 | 1.858 / 1.660 | 1.068 / 1.544 |
| w0 map n=6 / w0 map n=6 | 1.754±0.030 | 2.190±0.061 | 1.440 / 1.216 | 1.013 / 1.543 |
| w0 map n=6 / constants (det6 mapped only) | 1.871±0.037 | 2.184±0.062 | 1.440 / 1.216 | 1.263 / 1.622 |
| constants / w0 map n=6 (det7 mapped only) | 2.105±0.040 | 2.391±0.061 | 1.858 / 1.660 | 1.013 / 1.543 |
| w0 map n=8 / w0 map n=8 | 1.848±0.040 | 2.221±0.061 | 1.573 / 1.320 | 1.099 / 1.532 |
| w0 map n=8 / constants (det6 mapped only) | 1.891±0.043 | 2.242±0.058 | 1.573 / 1.320 | 1.263 / 1.622 |
| constants / w0 map n=8 (det7 mapped only) | 2.000±0.048 | 2.316±0.064 | 1.858 / 1.660 | 1.099 / 1.532 |

Change of s68(d)^2 (deg^2) against the constants/constants configuration, observed and predicted from the falls of s68^2 of the two chambers against M3 (predicted = -(fall6 + fall7)); diff = observed - predicted.

| map | configuration | axis | observed change | predicted change | observed - predicted | fall det6 | fall det7 |
|---|---|---|---|---|---|---|---|
| w0+v n=6 | both mapped | x | -0.78±0.11 | -2.29±0.13 | 1.51±0.16 | 1.65±0.12 | 0.64±0.06 |
| w0+v n=6 | both mapped | y | -0.77±0.21 | -1.78±0.13 | 1.01±0.23 | 1.58±0.10 | 0.19±0.09 |
| w0+v n=6 | det6 only | x | -0.31±0.14 | -1.65±0.12 | 1.33±0.15 | 1.65±0.12 | 0.00±0.00 |
| w0+v n=6 | det6 only | y | -0.67±0.21 | -1.58±0.10 | 0.92±0.21 | 1.58±0.10 | 0.00±0.00 |
| w0+v n=6 | det7 only | x | 0.84±0.14 | -0.64±0.06 | 1.49±0.15 | 0.00±0.00 | 0.64±0.06 |
| w0+v n=6 | det7 only | y | 0.74±0.17 | -0.19±0.09 | 0.94±0.18 | 0.00±0.00 | 0.19±0.09 |
| w0+v n=8 | both mapped | x | -0.52±0.11 | -1.68±0.12 | 1.16±0.15 | 1.22±0.11 | 0.46±0.06 |
| w0+v n=8 | both mapped | y | -0.48±0.18 | -1.47±0.13 | 1.00±0.20 | 1.22±0.09 | 0.25±0.09 |
| w0+v n=8 | det6 only | x | -0.30±0.12 | -1.22±0.11 | 0.93±0.14 | 1.22±0.11 | 0.00±0.00 |
| w0+v n=8 | det6 only | y | -0.32±0.19 | -1.22±0.09 | 0.91±0.19 | 1.22±0.09 | 0.00±0.00 |
| w0+v n=8 | det7 only | x | 0.40±0.11 | -0.46±0.06 | 0.86±0.12 | 0.00±0.00 | 0.46±0.06 |
| w0+v n=8 | det7 only | y | 0.23±0.14 | -0.25±0.09 | 0.48±0.16 | 0.00±0.00 | 0.25±0.09 |
| w0 n=6 | both mapped | x | -0.53±0.11 | -1.95±0.13 | 1.42±0.15 | 1.38±0.11 | 0.57±0.06 |
| w0 n=6 | both mapped | y | -0.52±0.21 | -1.53±0.13 | 1.00±0.23 | 1.28±0.09 | 0.25±0.09 |
| w0 n=6 | det6 only | x | -0.11±0.13 | -1.38±0.11 | 1.28±0.14 | 1.38±0.11 | 0.00±0.00 |
| w0 n=6 | det6 only | y | -0.55±0.21 | -1.28±0.09 | 0.73±0.21 | 1.28±0.09 | 0.00±0.00 |
| w0 n=6 | det7 only | x | 0.82±0.11 | -0.57±0.06 | 1.39±0.12 | 0.00±0.00 | 0.57±0.06 |
| w0 n=6 | det7 only | y | 0.40±0.13 | -0.25±0.09 | 0.65±0.15 | 0.00±0.00 | 0.25±0.09 |
| w0 n=8 | both mapped | x | -0.19±0.11 | -1.37±0.12 | 1.17±0.15 | 0.98±0.11 | 0.39±0.05 |
| w0 n=8 | both mapped | y | -0.38±0.19 | -1.30±0.13 | 0.91±0.22 | 1.01±0.09 | 0.28±0.10 |
| w0 n=8 | det6 only | x | -0.03±0.12 | -0.98±0.11 | 0.95±0.14 | 0.98±0.11 | 0.00±0.00 |
| w0 n=8 | det6 only | y | -0.29±0.18 | -1.01±0.09 | 0.72±0.19 | 1.01±0.09 | 0.00±0.00 |
| w0 n=8 | det7 only | x | 0.39±0.11 | -0.39±0.05 | 0.78±0.12 | 0.00±0.00 | 0.39±0.05 |
| w0 n=8 | det7 only | y | 0.05±0.13 | -0.28±0.10 | 0.33±0.15 | 0.00±0.00 | 0.28±0.10 |

#### all selected events, test sets: long, longer, short (matched events 2510)

| det6 / det7 angle | s68(d) X-axis | s68(d) Y-axis | s68(det6-M3) X / Y | s68(det7-M3) X / Y |
|---|---|---|---|---|
| pass 1 / pass 1 | 1.887±0.034 | 2.228±0.054 | 1.521 / 1.175 | 1.053 / 1.593 |
| constants / constants | 1.986±0.039 | 2.358±0.054 | 1.913 / 1.663 | 1.260 / 1.605 |
| w0+v map n=6 / w0+v map n=6 | 1.733±0.035 | 2.178±0.066 | 1.415 / 1.098 | 0.953 / 1.532 |
| w0+v map n=6 / constants (det6 mapped only) | 1.912±0.038 | 2.194±0.056 | 1.415 / 1.098 | 1.260 / 1.605 |
| constants / w0+v map n=6 (det7 mapped only) | 2.208±0.048 | 2.476±0.057 | 1.913 / 1.663 | 0.953 / 1.532 |
| w0+v map n=8 / w0+v map n=8 | 1.843±0.042 | 2.256±0.064 | 1.575 / 1.237 | 1.019 / 1.538 |
| w0+v map n=8 / constants (det6 mapped only) | 1.928±0.045 | 2.283±0.065 | 1.575 / 1.237 | 1.260 / 1.605 |
| constants / w0+v map n=8 (det7 mapped only) | 2.115±0.047 | 2.404±0.064 | 1.913 / 1.663 | 1.019 / 1.538 |
| w0 map n=6 / w0 map n=6 | 1.794±0.036 | 2.214±0.064 | 1.509 / 1.224 | 0.996 / 1.516 |
| w0 map n=6 / constants (det6 mapped only) | 1.951±0.042 | 2.225±0.066 | 1.509 / 1.224 | 1.260 / 1.605 |
| constants / w0 map n=6 (det7 mapped only) | 2.217±0.048 | 2.438±0.062 | 1.913 / 1.663 | 0.996 / 1.516 |
| w0 map n=8 / w0 map n=8 | 1.913±0.042 | 2.269±0.073 | 1.654 / 1.319 | 1.062 / 1.546 |
| w0 map n=8 / constants (det6 mapped only) | 1.969±0.046 | 2.276±0.070 | 1.654 / 1.319 | 1.260 / 1.605 |
| constants / w0 map n=8 (det7 mapped only) | 2.119±0.050 | 2.350±0.067 | 1.913 / 1.663 | 1.062 / 1.546 |

Change of s68(d)^2 (deg^2) against the constants/constants configuration, observed and predicted from the falls of s68^2 of the two chambers against M3 (predicted = -(fall6 + fall7)); diff = observed - predicted.

| map | configuration | axis | observed change | predicted change | observed - predicted | fall det6 | fall det7 |
|---|---|---|---|---|---|---|---|
| w0+v n=6 | both mapped | x | -0.94±0.12 | -2.34±0.14 | 1.40±0.16 | 1.66±0.13 | 0.68±0.06 |
| w0+v n=6 | both mapped | y | -0.82±0.20 | -1.79±0.12 | 0.97±0.22 | 1.56±0.10 | 0.23±0.08 |
| w0+v n=6 | det6 only | x | -0.29±0.14 | -1.66±0.13 | 1.37±0.15 | 1.66±0.13 | 0.00±0.00 |
| w0+v n=6 | det6 only | y | -0.74±0.19 | -1.56±0.10 | 0.82±0.20 | 1.56±0.10 | 0.00±0.00 |
| w0+v n=6 | det7 only | x | 0.93±0.15 | -0.68±0.06 | 1.61±0.15 | 0.00±0.00 | 0.68±0.06 |
| w0+v n=6 | det7 only | y | 0.57±0.15 | -0.23±0.08 | 0.80±0.16 | 0.00±0.00 | 0.23±0.08 |
| w0+v n=8 | both mapped | x | -0.55±0.12 | -1.73±0.13 | 1.18±0.16 | 1.18±0.12 | 0.55±0.05 |
| w0+v n=8 | both mapped | y | -0.47±0.18 | -1.45±0.13 | 0.98±0.20 | 1.24±0.09 | 0.21±0.08 |
| w0+v n=8 | det6 only | x | -0.23±0.14 | -1.18±0.12 | 0.95±0.15 | 1.18±0.12 | 0.00±0.00 |
| w0+v n=8 | det6 only | y | -0.35±0.19 | -1.24±0.09 | 0.89±0.19 | 1.24±0.09 | 0.00±0.00 |
| w0+v n=8 | det7 only | x | 0.53±0.12 | -0.55±0.05 | 1.08±0.13 | 0.00±0.00 | 0.55±0.05 |
| w0+v n=8 | det7 only | y | 0.22±0.15 | -0.21±0.08 | 0.43±0.16 | 0.00±0.00 | 0.21±0.08 |
| w0 n=6 | both mapped | x | -0.73±0.11 | -1.98±0.14 | 1.25±0.16 | 1.38±0.13 | 0.60±0.06 |
| w0 n=6 | both mapped | y | -0.66±0.18 | -1.55±0.12 | 0.89±0.20 | 1.27±0.09 | 0.28±0.08 |
| w0 n=6 | det6 only | x | -0.14±0.15 | -1.38±0.13 | 1.25±0.16 | 1.38±0.13 | 0.00±0.00 |
| w0 n=6 | det6 only | y | -0.61±0.19 | -1.27±0.09 | 0.66±0.20 | 1.27±0.09 | 0.00±0.00 |
| w0 n=6 | det7 only | x | 0.97±0.14 | -0.60±0.06 | 1.57±0.15 | 0.00±0.00 | 0.60±0.06 |
| w0 n=6 | det7 only | y | 0.38±0.14 | -0.28±0.08 | 0.66±0.16 | 0.00±0.00 | 0.28±0.08 |
| w0 n=8 | both mapped | x | -0.28±0.12 | -1.38±0.13 | 1.10±0.16 | 0.92±0.12 | 0.46±0.05 |
| w0 n=8 | both mapped | y | -0.41±0.19 | -1.21±0.13 | 0.80±0.22 | 1.03±0.10 | 0.19±0.09 |
| w0 n=8 | det6 only | x | -0.07±0.14 | -0.92±0.12 | 0.86±0.15 | 0.92±0.12 | 0.00±0.00 |
| w0 n=8 | det6 only | y | -0.38±0.18 | -1.03±0.10 | 0.65±0.20 | 1.03±0.10 | 0.00±0.00 |
| w0 n=8 | det7 only | x | 0.55±0.13 | -0.46±0.05 | 1.01±0.13 | 0.00±0.00 | 0.46±0.05 |
| w0 n=8 | det7 only | y | -0.04±0.14 | -0.19±0.09 | 0.15±0.16 | 0.00±0.00 | 0.19±0.09 |

#### det7 x >= 200 mm only, test sets: long, longer (matched events 891)

| det6 / det7 angle | s68(d) X-axis | s68(d) Y-axis | s68(det6-M3) X / Y | s68(det7-M3) X / Y |
|---|---|---|---|---|
| pass 1 / pass 1 | 1.932±0.060 | 1.563±0.050 | 1.506 / 1.051 | 1.144 / 0.981 |
| constants / constants | 2.011±0.076 | 1.601±0.059 | 1.770 / 1.248 | 1.281 / 0.964 |
| w0+v map n=6 / w0+v map n=6 | 1.725±0.110 | 1.449±0.112 | 1.417 / 1.027 | 0.995 / 0.805 |
| w0+v map n=6 / constants (det6 mapped only) | 1.903±0.063 | 1.457±0.048 | 1.417 / 1.027 | 1.281 / 0.964 |
| constants / w0+v map n=6 (det7 mapped only) | 2.035±0.126 | 1.617±0.128 | 1.770 / 1.248 | 0.995 / 0.805 |
| w0+v map n=8 / w0+v map n=8 | 2.037±0.138 | 1.389±0.122 | 1.555 / 1.219 | 1.203 / 0.840 |
| w0+v map n=8 / constants (det6 mapped only) | 1.978±0.072 | 1.543±0.063 | 1.555 / 1.219 | 1.281 / 0.964 |
| constants / w0+v map n=8 (det7 mapped only) | 2.068±0.133 | 1.606±0.119 | 1.770 / 1.248 | 1.203 / 0.840 |
| w0 map n=6 / w0 map n=6 | 1.954±0.117 | 1.619±0.114 | 1.503 / 1.155 | 0.998 / 0.791 |
| w0 map n=6 / constants (det6 mapped only) | 2.014±0.059 | 1.584±0.051 | 1.503 / 1.155 | 1.281 / 0.964 |
| constants / w0 map n=6 (det7 mapped only) | 2.047±0.130 | 1.585±0.115 | 1.770 / 1.248 | 0.998 / 0.791 |
| w0 map n=8 / w0 map n=8 | 2.087±0.114 | 1.501±0.109 | 1.654 / 1.207 | 1.162 / 0.839 |
| w0 map n=8 / constants (det6 mapped only) | 2.040±0.058 | 1.608±0.055 | 1.654 / 1.207 | 1.281 / 0.964 |
| constants / w0 map n=8 (det7 mapped only) | 2.114±0.130 | 1.582±0.114 | 1.770 / 1.248 | 1.162 / 0.839 |

Change of s68(d)^2 (deg^2) against the constants/constants configuration, observed and predicted from the falls of s68^2 of the two chambers against M3 (predicted = -(fall6 + fall7)); diff = observed - predicted.

| map | configuration | axis | observed change | predicted change | observed - predicted | fall det6 | fall det7 |
|---|---|---|---|---|---|---|---|
| w0+v n=6 | both mapped | x | -1.07±0.44 | -1.78±0.24 | 0.71±0.44 | 1.13±0.19 | 0.65±0.16 |
| w0+v n=6 | both mapped | y | -0.46±0.31 | -0.78±0.12 | 0.32±0.30 | 0.50±0.10 | 0.28±0.08 |
| w0+v n=6 | det6 only | x | -0.42±0.25 | -1.13±0.19 | 0.71±0.27 | 1.13±0.19 | 0.00±0.00 |
| w0+v n=6 | det6 only | y | -0.44±0.15 | -0.50±0.10 | 0.06±0.15 | 0.50±0.10 | 0.00±0.00 |
| w0+v n=6 | det7 only | x | 0.10±0.47 | -0.65±0.16 | 0.75±0.47 | 0.00±0.00 | 0.65±0.16 |
| w0+v n=6 | det7 only | y | 0.05±0.35 | -0.28±0.08 | 0.33±0.35 | 0.00±0.00 | 0.28±0.08 |
| w0+v n=8 | both mapped | x | 0.11±0.57 | -0.91±0.26 | 1.02±0.52 | 0.72±0.16 | 0.19±0.19 |
| w0+v n=8 | both mapped | y | -0.63±0.34 | -0.29±0.14 | -0.34±0.32 | 0.07±0.11 | 0.22±0.10 |
| w0+v n=8 | det6 only | x | -0.13±0.22 | -0.72±0.16 | 0.59±0.24 | 0.72±0.16 | 0.00±0.00 |
| w0+v n=8 | det6 only | y | -0.18±0.18 | -0.07±0.11 | -0.11±0.17 | 0.07±0.11 | 0.00±0.00 |
| w0+v n=8 | det7 only | x | 0.23±0.46 | -0.19±0.19 | 0.43±0.48 | 0.00±0.00 | 0.19±0.19 |
| w0+v n=8 | det7 only | y | 0.02±0.33 | -0.22±0.10 | 0.24±0.32 | 0.00±0.00 | 0.22±0.10 |
| w0 n=6 | both mapped | x | -0.22±0.48 | -1.52±0.24 | 1.30±0.49 | 0.88±0.18 | 0.65±0.16 |
| w0 n=6 | both mapped | y | 0.06±0.34 | -0.53±0.13 | 0.59±0.33 | 0.22±0.09 | 0.30±0.09 |
| w0 n=6 | det6 only | x | 0.01±0.22 | -0.88±0.18 | 0.89±0.26 | 0.88±0.18 | 0.00±0.00 |
| w0 n=6 | det6 only | y | -0.05±0.15 | -0.22±0.09 | 0.17±0.16 | 0.22±0.09 | 0.00±0.00 |
| w0 n=6 | det7 only | x | 0.15±0.48 | -0.65±0.16 | 0.79±0.48 | 0.00±0.00 | 0.65±0.16 |
| w0 n=6 | det7 only | y | -0.05±0.32 | -0.30±0.09 | 0.26±0.31 | 0.00±0.00 | 0.30±0.09 |
| w0 n=8 | both mapped | x | 0.31±0.49 | -0.69±0.28 | 1.00±0.48 | 0.40±0.17 | 0.29±0.20 |
| w0 n=8 | both mapped | y | -0.31±0.32 | -0.32±0.13 | 0.02±0.31 | 0.10±0.10 | 0.22±0.09 |
| w0 n=8 | det6 only | x | 0.12±0.20 | -0.40±0.17 | 0.52±0.25 | 0.40±0.17 | 0.00±0.00 |
| w0 n=8 | det6 only | y | 0.02±0.16 | -0.10±0.10 | 0.12±0.16 | 0.10±0.10 | 0.00±0.00 |
| w0 n=8 | det7 only | x | 0.43±0.45 | -0.29±0.20 | 0.72±0.48 | 0.00±0.00 | 0.29±0.20 |
| w0 n=8 | det7 only | y | -0.06±0.32 | -0.22±0.09 | 0.17±0.31 | 0.00±0.00 | 0.22±0.09 |

#### det7 x >= 200 mm only, test sets: long, longer, short (matched events 1009)

| det6 / det7 angle | s68(d) X-axis | s68(d) Y-axis | s68(det6-M3) X / Y | s68(det7-M3) X / Y |
|---|---|---|---|---|
| pass 1 / pass 1 | 1.965±0.053 | 1.579±0.049 | 1.528 / 1.091 | 1.133 / 0.988 |
| constants / constants | 2.078±0.069 | 1.643±0.054 | 1.815 / 1.278 | 1.274 / 0.966 |
| w0+v map n=6 / w0+v map n=6 | 1.974±0.130 | 1.445±0.088 | 1.462 / 1.056 | 0.910 / 0.819 |
| w0+v map n=6 / constants (det6 mapped only) | 1.981±0.066 | 1.470±0.049 | 1.462 / 1.056 | 1.274 / 0.966 |
| constants / w0+v map n=6 (det7 mapped only) | 2.218±0.126 | 1.623±0.098 | 1.815 / 1.278 | 0.910 / 0.819 |
| w0+v map n=8 / w0+v map n=8 | 2.252±0.117 | 1.514±0.103 | 1.628 / 1.232 | 1.192 / 0.896 |
| w0+v map n=8 / constants (det6 mapped only) | 2.066±0.072 | 1.580±0.059 | 1.628 / 1.232 | 1.274 / 0.966 |
| constants / w0+v map n=8 (det7 mapped only) | 2.268±0.106 | 1.690±0.095 | 1.815 / 1.278 | 1.192 / 0.896 |
| w0 map n=6 / w0 map n=6 | 2.088±0.114 | 1.567±0.088 | 1.554 / 1.196 | 0.940 / 0.827 |
| w0 map n=6 / constants (det6 mapped only) | 2.074±0.066 | 1.570±0.052 | 1.554 / 1.196 | 1.274 / 0.966 |
| constants / w0 map n=6 (det7 mapped only) | 2.253±0.121 | 1.617±0.097 | 1.815 / 1.278 | 0.940 / 0.827 |
| w0 map n=8 / w0 map n=8 | 2.320±0.119 | 1.514±0.103 | 1.694 / 1.262 | 1.171 / 0.868 |
| w0 map n=8 / constants (det6 mapped only) | 2.124±0.061 | 1.620±0.055 | 1.694 / 1.262 | 1.274 / 0.966 |
| constants / w0 map n=8 (det7 mapped only) | 2.255±0.104 | 1.684±0.093 | 1.815 / 1.278 | 1.171 / 0.868 |

Change of s68(d)^2 (deg^2) against the constants/constants configuration, observed and predicted from the falls of s68^2 of the two chambers against M3 (predicted = -(fall6 + fall7)); diff = observed - predicted.

| map | configuration | axis | observed change | predicted change | observed - predicted | fall det6 | fall det7 |
|---|---|---|---|---|---|---|---|
| w0+v n=6 | both mapped | x | -0.42±0.51 | -1.95±0.23 | 1.53±0.53 | 1.15±0.20 | 0.79±0.13 |
| w0+v n=6 | both mapped | y | -0.61±0.26 | -0.78±0.12 | 0.17±0.26 | 0.52±0.10 | 0.26±0.07 |
| w0+v n=6 | det6 only | x | -0.39±0.27 | -1.15±0.20 | 0.76±0.28 | 1.15±0.20 | 0.00±0.00 |
| w0+v n=6 | det6 only | y | -0.54±0.14 | -0.52±0.10 | -0.02±0.14 | 0.52±0.10 | 0.00±0.00 |
| w0+v n=6 | det7 only | x | 0.60±0.50 | -0.79±0.13 | 1.40±0.50 | 0.00±0.00 | 0.79±0.13 |
| w0+v n=6 | det7 only | y | -0.07±0.28 | -0.26±0.07 | 0.20±0.27 | 0.00±0.00 | 0.26±0.07 |
| w0+v n=8 | both mapped | x | 0.75±0.53 | -0.85±0.24 | 1.60±0.48 | 0.64±0.19 | 0.20±0.16 |
| w0+v n=8 | both mapped | y | -0.41±0.30 | -0.25±0.14 | -0.16±0.30 | 0.12±0.12 | 0.13±0.08 |
| w0+v n=8 | det6 only | x | -0.05±0.26 | -0.64±0.19 | 0.59±0.27 | 0.64±0.19 | 0.00±0.00 |
| w0+v n=8 | det6 only | y | -0.20±0.17 | -0.12±0.12 | -0.09±0.17 | 0.12±0.12 | 0.00±0.00 |
| w0+v n=8 | det7 only | x | 0.83±0.40 | -0.20±0.16 | 1.03±0.41 | 0.00±0.00 | 0.20±0.16 |
| w0+v n=8 | det7 only | y | 0.16±0.27 | -0.13±0.08 | 0.29±0.27 | 0.00±0.00 | 0.13±0.08 |
| w0 n=6 | both mapped | x | 0.04±0.46 | -1.62±0.24 | 1.66±0.48 | 0.88±0.21 | 0.74±0.13 |
| w0 n=6 | both mapped | y | -0.24±0.27 | -0.45±0.12 | 0.21±0.27 | 0.20±0.09 | 0.25±0.08 |
| w0 n=6 | det6 only | x | -0.02±0.24 | -0.88±0.21 | 0.86±0.26 | 0.88±0.21 | 0.00±0.00 |
| w0 n=6 | det6 only | y | -0.24±0.15 | -0.20±0.09 | -0.03±0.15 | 0.20±0.09 | 0.00±0.00 |
| w0 n=6 | det7 only | x | 0.76±0.48 | -0.74±0.13 | 1.49±0.47 | 0.00±0.00 | 0.74±0.13 |
| w0 n=6 | det7 only | y | -0.09±0.28 | -0.25±0.08 | 0.16±0.28 | 0.00±0.00 | 0.25±0.08 |
| w0 n=8 | both mapped | x | 1.06±0.54 | -0.67±0.25 | 1.74±0.50 | 0.42±0.18 | 0.25±0.16 |
| w0 n=8 | both mapped | y | -0.41±0.31 | -0.22±0.14 | -0.19±0.31 | 0.04±0.11 | 0.18±0.08 |
| w0 n=8 | det6 only | x | 0.19±0.22 | -0.42±0.18 | 0.62±0.25 | 0.42±0.18 | 0.00±0.00 |
| w0 n=8 | det6 only | y | -0.07±0.15 | -0.04±0.11 | -0.04±0.15 | 0.04±0.11 | 0.00±0.00 |
| w0 n=8 | det7 only | x | 0.77±0.39 | -0.25±0.16 | 1.02±0.40 | 0.00±0.00 | 0.25±0.16 |
| w0 n=8 | det7 only | y | 0.14±0.27 | -0.18±0.08 | 0.32±0.27 | 0.00±0.00 | 0.18±0.08 |
### Same budget with the clipped variance

| sample | map | axis | rms/s68 of r6 | rms/s68 of d | var(d) fall obs | pred | obs-pred (both) | det6 only obs-pred | det7 only obs-pred |
|---|---|---|---|---|---|---|---|---|---|
| full | w0+v6 | x | 0.98 | 1.03 | 0.63±0.05 | 1.81±0.08 | -1.18±0.07 | -1.14±0.06 | -0.99±0.05 |
| full | w0+v6 | y | 0.99 | 1.06 | 0.14±0.17 | 0.75±0.18 | -0.61±0.08 | -0.65±0.07 | -0.24±0.05 |
| full | w0+v8 | x | 0.98 | 1.03 | 0.34±0.05 | 1.12±0.06 | -0.79±0.06 | -0.72±0.05 | -0.54±0.04 |
| full | w0+v8 | y | 0.99 | 1.06 | 0.29±0.08 | 0.75±0.06 | -0.46±0.06 | -0.41±0.06 | -0.24±0.03 |
| full | w06 | x | 0.98 | 1.03 | 0.35±0.04 | 1.49±0.07 | -1.14±0.06 | -1.02±0.05 | -0.99±0.05 |
| full | w06 | y | 0.99 | 1.06 | 0.35±0.08 | 0.85±0.07 | -0.50±0.06 | -0.45±0.05 | -0.22±0.04 |
| full | w08 | x | 0.98 | 1.03 | 0.14±0.05 | 0.91±0.06 | -0.77±0.05 | -0.67±0.04 | -0.55±0.04 |
| full | w08 | y | 0.99 | 1.06 | 0.28±0.06 | 0.64±0.05 | -0.36±0.05 | -0.30±0.05 | -0.17±0.02 |
| xhalf | w0+v6 | x | 1.02 | 0.91 | 0.40±0.21 | 1.55±0.23 | -1.15±0.20 | -0.95±0.07 | -0.55±0.12 |
| xhalf | w0+v6 | y | 1.09 | 1.10 | 0.61±0.18 | 0.95±0.16 | -0.34±0.11 | -0.16±0.05 | -0.02±0.06 |
| xhalf | w0+v8 | x | 1.02 | 0.91 | -0.12±0.23 | 1.08±0.20 | -1.20±0.18 | -0.57±0.06 | -0.04±0.03 |
| xhalf | w0+v8 | y | 1.09 | 1.10 | 0.47±0.19 | 0.83±0.16 | -0.35±0.10 | -0.12±0.05 | 0.02±0.02 |
| xhalf | w06 | x | 1.02 | 0.91 | 0.00±0.20 | 1.13±0.23 | -1.13±0.20 | -0.95±0.07 | -0.60±0.12 |
| xhalf | w06 | y | 1.09 | 1.10 | 0.13±0.17 | 0.36±0.14 | -0.23±0.10 | -0.12±0.04 | -0.01±0.04 |
| xhalf | w08 | x | 1.02 | 0.91 | -0.66±0.25 | 0.64±0.20 | -1.29±0.18 | -0.61±0.06 | -0.09±0.04 |
| xhalf | w08 | y | 1.09 | 1.10 | 0.13±0.18 | 0.38±0.14 | -0.24±0.09 | -0.08±0.04 | 0.01±0.02 |
### Differential and common part of the correction (slope -1 = real)

| sample | map | axis | slope of d on (D6-D7), differential part (real: -1; telescope: 0) | slope of (r6+r7)/2 on (D6+D7)/2, common part | share of correction variance that is common | slope of r6 on D6 | slope of d on D6 (joint fit) | slope of d on D7 (joint fit, real: +1) | corr(D6, D7) |
|---|---|---|---|---|---|---|---|---|---|
| full | w0+v6 | x | -0.98±0.10 | -1.15±0.04 | 0.79 | -1.16±0.05 | -0.96±0.10 | 0.83±0.20 | 0.68 |
| full | w0+v6 | y | -0.47±0.19 | -0.45±0.16 | 0.57 | -1.38±0.10 | -1.32±0.17 | -0.55±0.23 | 0.14 |
| full | w0+v8 | x | -0.85±0.10 | -1.18±0.06 | 0.73 | -1.12±0.07 | -0.85±0.11 | 0.73±0.23 | 0.54 |
| full | w0+v8 | y | -1.24±0.19 | -0.99±0.13 | 0.59 | -1.27±0.11 | -1.34±0.20 | 0.30±0.35 | 0.24 |
| full | w06 | x | -0.97±0.13 | -1.17±0.04 | 0.83 | -1.19±0.06 | -0.94±0.14 | 0.72±0.21 | 0.72 |
| full | w06 | y | -1.54±0.19 | -1.23±0.14 | 0.58 | -1.51±0.12 | -1.70±0.21 | 1.00±0.32 | 0.18 |
| full | w08 | x | -0.80±0.12 | -1.19±0.05 | 0.75 | -1.10±0.07 | -0.80±0.12 | 0.65±0.24 | 0.58 |
| full | w08 | y | -1.43±0.21 | -1.24±0.16 | 0.58 | -1.39±0.15 | -1.49±0.24 | 0.77±0.33 | 0.21 |
| xhalf | w0+v6 | x | -0.64±0.10 | -0.95±0.07 | 0.71 | -0.82±0.07 | -0.67±0.10 | 1.15±0.24 | 0.54 |
| xhalf | w0+v6 | y | -1.55±0.46 | -1.86±0.48 | 0.51 | -1.77±0.45 | -1.62±0.45 | 0.69±0.55 | 0.04 |
| xhalf | w0+v8 | x | -0.41±0.11 | -1.19±0.10 | 0.52 | -0.80±0.08 | -0.42±0.11 | 1.60±0.76 | 0.18 |
| xhalf | w0+v8 | y | -1.38±0.42 | -1.81±0.46 | 0.49 | -1.59±0.39 | -1.42±0.40 | -0.86±2.12 | -0.05 |
| xhalf | w06 | x | -0.61±0.15 | -0.92±0.08 | 0.74 | -0.79±0.10 | -0.64±0.15 | 1.03±0.26 | 0.59 |
| xhalf | w06 | y | -1.24±0.45 | -1.68±0.55 | 0.49 | -1.48±0.46 | -1.37±0.47 | -0.97±1.19 | -0.05 |
| xhalf | w08 | x | -0.32±0.13 | -1.18±0.10 | 0.52 | -0.75±0.10 | -0.32±0.12 | 0.35±0.74 | 0.17 |
| xhalf | w08 | y | -1.27±0.47 | -1.68±0.54 | 0.49 | -1.48±0.46 | -1.37±0.48 | -3.19±2.74 | -0.10 |
## Parts 3 and 4

![](fig_part34.png)

### Part 3, best row pair per chamber pair (tau, telescope frame)

| chamber pair | view | n | best row pair | r | common cells | ceiling | r / ceiling | null range |
|---|---|---|---|---|---|---|---|---|
| det2-det3 | x | 6 | det2:long / det3:p2 | +0.13±0.04 | 25 | 0.98 | +0.14 | -0.37 .. +0.19 |
| det2-det3 | x | 8 | det2:long / det3:p2 | +0.06±0.03 | 49 | 0.97 | +0.06 | -0.29 .. +0.22 |
| det2-det3 | y | 6 | det2:long / det3:p2 | +0.19±0.03 | 25 | 0.99 | +0.19 | -0.41 .. +0.45 |
| det2-det3 | y | 8 | det2:long / det3:p2 | +0.14±0.03 | 49 | 0.98 | +0.14 | -0.38 .. +0.44 |
| det2-det4 | x | 6 | det2:long / det4:624 | +0.24±0.17 | 18 | 0.78 | +0.31 | -0.41 .. +0.25 |
| det2-det4 | x | 8 | det2:long / det4:624 | +0.13±0.19 | 15 | 0.60 | +0.22 | -0.29 .. +0.02 |
| det2-det4 | y | 6 | det2:long / det4:624 | +0.47±0.18 | 18 | 0.87 | +0.54 | -0.68 .. +0.58 |
| det2-det4 | y | 8 | det2:long / det4:624 | +0.46±0.20 | 15 | n/a | n/a | -0.49 .. +0.24 |
| det2-det7 | y | 6 | det2:long / det7:long | -0.17±0.13 | 13 | 0.97 | -0.18 | -0.53 .. +0.60 |
| det2-det7 | y | 8 | det2:long / det7:long | +0.26±0.10 | 12 | n/a | n/a | -0.32 .. +0.56 |
| det3-det4 | x | 6 | det3:p2 / det4:624 | +0.65±0.19 | 18 | 0.78 | +0.84 | -0.11 .. +0.13 |
| det3-det4 | x | 8 | det3:p2 / det4:624 | +0.64±0.22 | 14 | 0.60 | +1.05 | -0.24 .. +0.21 |
| det3-det4 | y | 6 | det3:p2 / det4:624 | +0.68±0.13 | 18 | 0.88 | +0.77 | -0.49 .. +0.37 |
| det3-det4 | y | 8 | det3:p2 / det4:624 | +0.62±0.20 | 14 | n/a | n/a | -0.23 .. +0.35 |
| det3-det6 | x | 6 | det3:d625 / det6:long | +0.49±0.05 | 25 | 0.94 | +0.51 | -0.49 .. +0.61 |
| det3-det6 | x | 8 | det3:d625 / det6:long | +0.41±0.05 | 32 | 0.91 | +0.45 | -0.31 .. +0.48 |
| det3-det6 | y | 6 | det3:d625 / det6:long | +0.41±0.05 | 25 | 0.96 | +0.42 | -0.54 .. +0.21 |
| det3-det6 | y | 8 | det3:d625 / det6:long | +0.39±0.05 | 32 | 0.94 | +0.41 | -0.44 .. +0.31 |
| det3-det7 | y | 6 | det3:p2 / det7:long | -0.10±0.08 | 14 | 0.97 | -0.11 | -0.25 .. +0.06 |
| det3-det7 | y | 8 | det3:p2 / det7:long | +0.03±0.07 | 12 | n/a | n/a | -0.10 .. +0.36 |
| det4-det7 | y | 6 | det4:624 / det7:long | -0.10±0.13 | 12 | 0.86 | -0.12 | -0.06 .. +0.12 |
### Part 3, all row pairs

#### 3a same chamber, tau = w0/v
| row A | row B | view | n | frame | r (tau residual maps) | common cells | noise ceiling | r / ceiling | mirrored-map null |
|---|---|---|---|---|---|---|---|---|---|
| det2:long | det2:longer | x | 6 | chamber | +0.89±0.04 | 32 | n/a | n/a | -0.66 .. +0.23 |
| det2:long | det2:longer | x | 6 | tel | +0.89±0.05 | 19 | n/a | n/a | -0.76 .. +0.20 |
| det2:long | det2:longer | x | 8 | chamber | +0.87±0.07 | 20 | n/a | n/a | -0.81 .. +0.23 |
| det2:long | det2:longer | y | 6 | chamber | +0.95±0.03 | 32 | n/a | n/a | -0.79 .. +0.57 |
| det2:long | det2:longer | y | 6 | tel | +0.93±0.04 | 19 | n/a | n/a | -0.82 .. +0.56 |
| det2:long | det2:longer | y | 8 | chamber | +0.77±0.08 | 20 | n/a | n/a | -0.68 .. +0.33 |
| det3:d625 | det3:sat | x | 6 | chamber | +0.42±0.08 | 36 | 0.95 | +0.45 | -0.45 .. +0.33 |
| det3:d625 | det3:sat | x | 6 | tel | +0.30±0.06 | 25 | 0.95 | +0.31 | -0.46 .. +0.37 |
| det3:d625 | det3:sat | x | 8 | chamber | +0.45±0.05 | 58 | 0.93 | +0.49 | -0.52 .. +0.38 |
| det3:d625 | det3:sat | x | 8 | tel | +0.29±0.06 | 42 | 0.93 | +0.31 | -0.37 .. +0.40 |
| det3:d625 | det3:sat | y | 6 | chamber | +0.60±0.04 | 36 | 0.97 | +0.62 | -0.68 .. +0.64 |
| det3:d625 | det3:sat | y | 6 | tel | +0.44±0.04 | 25 | 0.97 | +0.45 | -0.47 .. +0.55 |
| det3:d625 | det3:sat | y | 8 | chamber | +0.49±0.05 | 58 | 0.96 | +0.51 | -0.60 .. +0.50 |
| det3:d625 | det3:sat | y | 8 | tel | +0.41±0.05 | 42 | 0.96 | +0.43 | -0.52 .. +0.45 |
| det3:d625 | det3:p2 | x | 6 | chamber | +0.46±0.04 | 36 | 0.96 | +0.48 | -0.52 .. +0.40 |
| det3:d625 | det3:p2 | x | 6 | tel | +0.38±0.05 | 25 | 0.96 | +0.39 | -0.46 .. +0.33 |
| det3:d625 | det3:p2 | x | 8 | chamber | +0.44±0.05 | 64 | 0.94 | +0.46 | -0.50 .. +0.36 |
| det3:d625 | det3:p2 | x | 8 | tel | +0.31±0.05 | 49 | 0.94 | +0.32 | -0.35 .. +0.35 |
| det3:d625 | det3:p2 | y | 6 | chamber | +0.63±0.02 | 36 | 0.99 | +0.64 | -0.70 .. +0.65 |
| det3:d625 | det3:p2 | y | 6 | tel | +0.45±0.03 | 25 | 0.99 | +0.45 | -0.52 .. +0.59 |
| det3:d625 | det3:p2 | y | 8 | chamber | +0.58±0.02 | 64 | 0.97 | +0.59 | -0.66 .. +0.61 |
| det3:d625 | det3:p2 | y | 8 | tel | +0.45±0.03 | 49 | 0.97 | +0.46 | -0.46 .. +0.55 |
| det3:sat | det3:p2 | x | 6 | chamber | +0.98±0.02 | 36 | 0.97 | +1.01 | -0.81 .. +0.39 |
| det3:sat | det3:p2 | x | 6 | tel | +0.98±0.01 | 25 | 0.97 | +1.01 | -0.79 .. +0.26 |
| det3:sat | det3:p2 | x | 8 | chamber | +0.97±0.01 | 58 | 0.96 | +1.01 | -0.70 .. +0.29 |
| det3:sat | det3:p2 | x | 8 | tel | +0.98±0.01 | 42 | 0.96 | +1.02 | -0.71 .. +0.18 |
| det3:sat | det3:p2 | y | 6 | chamber | +0.99±0.01 | 36 | 0.98 | +1.01 | -0.79 .. +0.70 |
| det3:sat | det3:p2 | y | 6 | tel | +0.99±0.01 | 25 | 0.98 | +1.01 | -0.74 .. +0.56 |
| det3:sat | det3:p2 | y | 8 | chamber | +0.96±0.02 | 58 | 0.97 | +0.99 | -0.71 .. +0.57 |
| det3:sat | det3:p2 | y | 8 | tel | +0.96±0.01 | 42 | 0.97 | +0.99 | -0.66 .. +0.48 |
| det4:623 | det4:624 | x | 6 | chamber | +0.46±0.34 | 18 | n/a | n/a | -0.56 .. +0.63 |
| det4:623 | det4:624 | x | 8 | chamber | +0.62±0.24 | 14 | n/a | n/a | -0.04 .. +0.37 |
| det4:623 | det4:624 | y | 6 | chamber | +0.56±0.21 | 18 | n/a | n/a | -0.21 .. +0.38 |
| det4:623 | det4:624 | y | 8 | chamber | +0.60±0.25 | 14 | n/a | n/a | -0.54 .. +0.47 |
| det6:long | det6:longer | x | 6 | chamber | +0.85±0.07 | 33 | n/a | n/a | -0.44 .. +0.42 |
| det6:long | det6:longer | x | 6 | tel | +0.84±0.07 | 18 | n/a | n/a | -0.24 .. +0.46 |
| det6:long | det6:longer | x | 8 | chamber | +0.68±0.13 | 22 | n/a | n/a | -0.22 .. +0.60 |
| det6:long | det6:longer | y | 6 | chamber | +0.83±0.06 | 33 | n/a | n/a | -0.57 .. +0.19 |
| det6:long | det6:longer | y | 6 | tel | +0.80±0.08 | 18 | n/a | n/a | -0.53 .. +0.22 |
| det6:long | det6:longer | y | 8 | chamber | +0.59±0.16 | 22 | n/a | n/a | -0.13 .. +0.20 |
| det7:long | det7:longer | y | 6 | chamber | +0.92±0.05 | 18 | n/a | n/a | +0.12 .. +0.35 |

#### 3a same chamber, v
| row A | row B | view | n | frame | r (tau residual maps) | common cells | noise ceiling | r / ceiling | mirrored-map null |
|---|---|---|---|---|---|---|---|---|---|
| det2:long | det2:longer | x | 6 | chamber | +0.46±0.15 | 32 | n/a | n/a | +0.07 .. +0.48 |
| det2:long | det2:longer | x | 6 | tel | +0.58±0.13 | 19 | n/a | n/a | +0.10 .. +0.49 |
| det2:long | det2:longer | x | 8 | chamber | +0.75±0.10 | 20 | n/a | n/a | +0.27 .. +0.65 |
| det2:long | det2:longer | y | 6 | chamber | +0.47±0.14 | 32 | n/a | n/a | +0.07 .. +0.34 |
| det2:long | det2:longer | y | 6 | tel | +0.45±0.15 | 19 | n/a | n/a | +0.00 .. +0.31 |
| det2:long | det2:longer | y | 8 | chamber | +0.62±0.16 | 20 | n/a | n/a | +0.10 .. +0.45 |
| det3:d625 | det3:sat | x | 6 | chamber | -0.17±0.11 | 36 | 0.83 | -0.20 | -0.27 .. +0.14 |
| det3:d625 | det3:sat | x | 6 | tel | +0.07±0.10 | 25 | 0.83 | +0.09 | -0.09 .. +0.18 |
| det3:d625 | det3:sat | x | 8 | chamber | -0.19±0.09 | 58 | 0.84 | -0.23 | -0.46 .. +0.03 |
| det3:d625 | det3:sat | x | 8 | tel | +0.13±0.11 | 42 | 0.84 | +0.16 | -0.01 .. +0.17 |
| det3:d625 | det3:sat | y | 6 | chamber | +0.06±0.09 | 36 | 0.88 | +0.07 | -0.02 .. +0.04 |
| det3:d625 | det3:sat | y | 6 | tel | +0.26±0.08 | 25 | 0.88 | +0.29 | +0.09 .. +0.37 |
| det3:d625 | det3:sat | y | 8 | chamber | +0.10±0.09 | 58 | 0.75 | +0.13 | -0.13 .. +0.09 |
| det3:d625 | det3:sat | y | 8 | tel | +0.17±0.10 | 42 | 0.75 | +0.23 | +0.20 .. +0.25 |
| det3:d625 | det3:p2 | x | 6 | chamber | -0.18±0.09 | 36 | 0.90 | -0.21 | -0.20 .. -0.10 |
| det3:d625 | det3:p2 | x | 6 | tel | +0.19±0.08 | 25 | 0.90 | +0.21 | -0.02 .. +0.19 |
| det3:d625 | det3:p2 | x | 8 | chamber | -0.18±0.08 | 64 | 0.86 | -0.20 | -0.22 .. -0.00 |
| det3:d625 | det3:p2 | x | 8 | tel | +0.19±0.07 | 49 | 0.86 | +0.22 | +0.03 .. +0.17 |
| det3:d625 | det3:p2 | y | 6 | chamber | +0.24±0.06 | 36 | 0.93 | +0.26 | -0.16 .. +0.05 |
| det3:d625 | det3:p2 | y | 6 | tel | +0.38±0.06 | 25 | 0.93 | +0.41 | +0.09 .. +0.40 |
| det3:d625 | det3:p2 | y | 8 | chamber | +0.10±0.08 | 64 | 0.87 | +0.11 | -0.14 .. +0.04 |
| det3:d625 | det3:p2 | y | 8 | tel | +0.31±0.07 | 49 | 0.87 | +0.35 | +0.07 .. +0.30 |
| det3:sat | det3:p2 | x | 6 | chamber | +0.83±0.09 | 36 | 0.80 | +1.04 | +0.22 .. +0.48 |
| det3:sat | det3:p2 | x | 6 | tel | +0.83±0.08 | 25 | 0.80 | +1.03 | +0.13 .. +0.48 |
| det3:sat | det3:p2 | x | 8 | chamber | +0.70±0.11 | 58 | 0.82 | +0.86 | +0.28 .. +0.53 |
| det3:sat | det3:p2 | x | 8 | tel | +0.73±0.08 | 42 | 0.82 | +0.90 | +0.22 .. +0.44 |
| det3:sat | det3:p2 | y | 6 | chamber | +0.75±0.08 | 36 | 0.88 | +0.85 | +0.44 .. +0.64 |
| det3:sat | det3:p2 | y | 6 | tel | +0.73±0.07 | 25 | 0.88 | +0.83 | +0.30 .. +0.63 |
| det3:sat | det3:p2 | y | 8 | chamber | +0.79±0.08 | 58 | 0.76 | +1.04 | +0.50 .. +0.59 |
| det3:sat | det3:p2 | y | 8 | tel | +0.78±0.06 | 42 | 0.76 | +1.03 | +0.37 .. +0.56 |
| det4:623 | det4:624 | x | 6 | chamber | +0.30±0.25 | 18 | n/a | n/a | -0.01 .. +0.25 |
| det4:623 | det4:624 | x | 8 | chamber | +0.10±0.28 | 14 | n/a | n/a | -0.09 .. +0.25 |
| det4:623 | det4:624 | y | 6 | chamber | +0.43±0.32 | 18 | n/a | n/a | -0.10 .. +0.02 |
| det4:623 | det4:624 | y | 8 | chamber | +0.52±0.27 | 14 | n/a | n/a | -0.18 .. +0.15 |
| det6:long | det6:longer | x | 6 | chamber | +0.75±0.08 | 33 | n/a | n/a | +0.34 .. +0.58 |
| det6:long | det6:longer | x | 6 | tel | +0.72±0.10 | 18 | n/a | n/a | +0.46 .. +0.57 |
| det6:long | det6:longer | x | 8 | chamber | +0.74±0.14 | 22 | n/a | n/a | -0.19 .. +0.54 |
| det6:long | det6:longer | y | 6 | chamber | +0.79±0.08 | 33 | n/a | n/a | +0.40 .. +0.60 |
| det6:long | det6:longer | y | 6 | tel | +0.75±0.10 | 18 | n/a | n/a | +0.51 .. +0.56 |
| det6:long | det6:longer | y | 8 | chamber | +0.69±0.18 | 22 | n/a | n/a | +0.24 .. +0.60 |
| det7:long | det7:longer | y | 6 | chamber | +0.64±0.17 | 18 | n/a | n/a | +0.04 .. +0.56 |

#### 3b different chambers, same slot, tau
| row A | row B | view | n | frame | r (tau residual maps) | common cells | noise ceiling | r / ceiling | mirrored-map null |
|---|---|---|---|---|---|---|---|---|---|
| det2:long | det3:sat | x | 6 | tel | +0.04±0.06 | 25 | 0.97 | +0.04 | -0.38 .. +0.18 |
| det2:long | det3:sat | x | 8 | tel | -0.01±0.04 | 42 | 0.95 | -0.01 | -0.36 .. +0.20 |
| det2:long | det3:sat | y | 6 | tel | +0.18±0.06 | 25 | 0.98 | +0.19 | -0.40 .. +0.41 |
| det2:long | det3:sat | y | 8 | tel | +0.13±0.06 | 42 | 0.97 | +0.14 | -0.34 .. +0.42 |
| det2:long | det3:p2 | x | 6 | tel | +0.13±0.04 | 25 | 0.98 | +0.14 | -0.37 .. +0.19 |
| det2:long | det3:p2 | x | 8 | tel | +0.06±0.03 | 49 | 0.97 | +0.06 | -0.29 .. +0.22 |
| det2:long | det3:p2 | y | 6 | tel | +0.19±0.03 | 25 | 0.99 | +0.19 | -0.41 .. +0.45 |
| det2:long | det3:p2 | y | 8 | tel | +0.14±0.03 | 49 | 0.98 | +0.14 | -0.38 .. +0.44 |
| det2:long | det4:624 | x | 6 | tel | +0.24±0.17 | 18 | 0.78 | +0.31 | -0.41 .. +0.25 |
| det2:long | det4:624 | x | 8 | tel | +0.13±0.19 | 15 | 0.60 | +0.22 | -0.29 .. +0.02 |
| det2:long | det4:624 | y | 6 | tel | +0.47±0.18 | 18 | 0.87 | +0.54 | -0.68 .. +0.58 |
| det2:long | det4:624 | y | 8 | tel | +0.46±0.20 | 15 | n/a | n/a | -0.49 .. +0.24 |
| det2:long | det7:long | y | 6 | tel | -0.17±0.13 | 13 | 0.97 | -0.18 | -0.53 .. +0.60 |
| det2:long | det7:long | y | 8 | tel | +0.26±0.10 | 12 | n/a | n/a | -0.32 .. +0.56 |
| det2:longer | det3:sat | x | 6 | tel | +0.33±0.08 | 19 | n/a | n/a | -0.56 .. +0.30 |
| det2:longer | det3:sat | y | 6 | tel | +0.33±0.11 | 19 | n/a | n/a | -0.42 .. +0.43 |
| det2:longer | det3:p2 | x | 6 | tel | +0.38±0.07 | 19 | n/a | n/a | -0.56 .. +0.29 |
| det2:longer | det3:p2 | y | 6 | tel | +0.33±0.10 | 19 | n/a | n/a | -0.43 .. +0.47 |
| det2:longer | det4:624 | x | 6 | tel | +0.34±0.18 | 16 | n/a | n/a | -0.37 .. +0.24 |
| det2:longer | det4:624 | y | 6 | tel | +0.40±0.21 | 16 | n/a | n/a | -0.69 .. +0.54 |
| det2:longer | det7:long | y | 6 | tel | -0.05±0.15 | 13 | n/a | n/a | -0.42 .. +0.46 |
| det3:d625 | det6:long | x | 6 | tel | +0.49±0.05 | 25 | 0.94 | +0.51 | -0.49 .. +0.61 |
| det3:d625 | det6:long | x | 8 | tel | +0.41±0.05 | 32 | 0.91 | +0.45 | -0.31 .. +0.48 |
| det3:d625 | det6:long | y | 6 | tel | +0.41±0.05 | 25 | 0.96 | +0.42 | -0.54 .. +0.21 |
| det3:d625 | det6:long | y | 8 | tel | +0.39±0.05 | 32 | 0.94 | +0.41 | -0.44 .. +0.31 |
| det3:d625 | det6:longer | x | 6 | tel | +0.51±0.10 | 19 | n/a | n/a | -0.48 .. +0.68 |
| det3:d625 | det6:longer | y | 6 | tel | +0.53±0.10 | 19 | n/a | n/a | -0.53 .. +0.29 |
| det3:sat | det4:624 | x | 6 | tel | +0.61±0.19 | 18 | 0.77 | +0.79 | -0.08 .. +0.14 |
| det3:sat | det4:624 | x | 8 | tel | +0.67±0.23 | 13 | 0.59 | +1.12 | -0.35 .. +0.25 |
| det3:sat | det4:624 | y | 6 | tel | +0.68±0.15 | 18 | 0.87 | +0.79 | -0.43 .. +0.36 |
| det3:sat | det4:624 | y | 8 | tel | +0.64±0.21 | 13 | n/a | n/a | -0.19 .. +0.45 |
| det3:sat | det7:long | y | 6 | tel | +0.07±0.08 | 13 | 0.96 | +0.07 | -0.14 .. +0.06 |
| det3:sat | det7:long | y | 8 | tel | -0.00±0.09 | 12 | n/a | n/a | -0.12 .. +0.31 |
| det3:p2 | det4:624 | x | 6 | tel | +0.65±0.19 | 18 | 0.78 | +0.84 | -0.11 .. +0.13 |
| det3:p2 | det4:624 | x | 8 | tel | +0.64±0.22 | 14 | 0.60 | +1.05 | -0.24 .. +0.21 |
| det3:p2 | det4:624 | y | 6 | tel | +0.68±0.13 | 18 | 0.88 | +0.77 | -0.49 .. +0.37 |
| det3:p2 | det4:624 | y | 8 | tel | +0.62±0.20 | 14 | n/a | n/a | -0.23 .. +0.35 |
| det3:p2 | det7:long | y | 6 | tel | -0.10±0.08 | 14 | 0.97 | -0.11 | -0.25 .. +0.06 |
| det3:p2 | det7:long | y | 8 | tel | +0.03±0.07 | 12 | n/a | n/a | -0.10 .. +0.36 |
| det4:624 | det7:long | y | 6 | tel | -0.10±0.13 | 12 | 0.86 | -0.12 | -0.06 .. +0.12 |

#### 3b different chambers, same slot, v
| row A | row B | view | n | frame | r (tau residual maps) | common cells | noise ceiling | r / ceiling | mirrored-map null |
|---|---|---|---|---|---|---|---|---|---|
| det2:long | det3:sat | x | 6 | tel | +0.12±0.10 | 25 | 0.81 | +0.15 | +0.44 .. +0.52 |
| det2:long | det3:sat | x | 8 | tel | +0.16±0.10 | 42 | 0.80 | +0.19 | +0.34 .. +0.44 |
| det2:long | det3:sat | y | 6 | tel | +0.31±0.11 | 25 | 0.72 | +0.43 | +0.33 .. +0.49 |
| det2:long | det3:sat | y | 8 | tel | +0.17±0.11 | 42 | 0.62 | +0.27 | +0.26 .. +0.40 |
| det2:long | det3:p2 | x | 6 | tel | +0.20±0.08 | 25 | 0.87 | +0.23 | +0.35 .. +0.44 |
| det2:long | det3:p2 | x | 8 | tel | +0.16±0.07 | 49 | 0.82 | +0.20 | +0.29 .. +0.48 |
| det2:long | det3:p2 | y | 6 | tel | +0.26±0.10 | 25 | 0.76 | +0.34 | +0.42 .. +0.50 |
| det2:long | det3:p2 | y | 8 | tel | +0.14±0.09 | 49 | 0.72 | +0.19 | +0.30 .. +0.43 |
| det2:long | det4:624 | x | 6 | tel | +0.16±0.15 | 18 | 0.57 | +0.27 | +0.02 .. +0.07 |
| det2:long | det4:624 | x | 8 | tel | +0.13±0.19 | 15 | n/a | n/a | -0.46 .. -0.19 |
| det2:long | det4:624 | y | 6 | tel | -0.35±0.17 | 18 | 0.51 | -0.69 | -0.19 .. -0.05 |
| det2:long | det4:624 | y | 8 | tel | -0.44±0.21 | 15 | n/a | n/a | -0.48 .. +0.07 |
| det2:long | det7:long | y | 6 | tel | -0.13±0.16 | 13 | 0.67 | -0.20 | -0.35 .. +0.12 |
| det2:long | det7:long | y | 8 | tel | -0.11±0.17 | 12 | n/a | n/a | -0.24 .. +0.43 |
| det2:longer | det3:sat | x | 6 | tel | +0.22±0.13 | 19 | n/a | n/a | +0.31 .. +0.44 |
| det2:longer | det3:sat | y | 6 | tel | +0.20±0.16 | 19 | n/a | n/a | +0.11 .. +0.26 |
| det2:longer | det3:p2 | x | 6 | tel | +0.28±0.11 | 19 | n/a | n/a | +0.21 .. +0.37 |
| det2:longer | det3:p2 | y | 6 | tel | +0.36±0.16 | 19 | n/a | n/a | +0.16 .. +0.29 |
| det2:longer | det4:624 | x | 6 | tel | -0.18±0.18 | 16 | n/a | n/a | -0.34 .. +0.04 |
| det2:longer | det4:624 | y | 6 | tel | -0.24±0.21 | 16 | n/a | n/a | -0.43 .. -0.13 |
| det2:longer | det7:long | y | 6 | tel | -0.02±0.21 | 13 | n/a | n/a | -0.40 .. -0.16 |
| det3:d625 | det6:long | x | 6 | tel | +0.63±0.07 | 25 | 0.90 | +0.70 | +0.21 .. +0.39 |
| det3:d625 | det6:long | x | 8 | tel | +0.42±0.08 | 32 | 0.80 | +0.52 | +0.12 .. +0.35 |
| det3:d625 | det6:long | y | 6 | tel | +0.65±0.06 | 25 | 0.90 | +0.72 | +0.19 .. +0.67 |
| det3:d625 | det6:long | y | 8 | tel | +0.51±0.08 | 32 | 0.77 | +0.67 | +0.24 .. +0.61 |
| det3:d625 | det6:longer | x | 6 | tel | +0.29±0.12 | 19 | n/a | n/a | +0.12 .. +0.51 |
| det3:d625 | det6:longer | y | 6 | tel | +0.58±0.10 | 19 | n/a | n/a | +0.34 .. +0.42 |
| det3:sat | det4:624 | x | 6 | tel | -0.05±0.12 | 18 | 0.52 | -0.09 | -0.19 .. -0.06 |
| det3:sat | det4:624 | x | 8 | tel | -0.17±0.15 | 13 | n/a | n/a | -0.27 .. -0.11 |
| det3:sat | det4:624 | y | 6 | tel | +0.15±0.18 | 18 | 0.59 | +0.25 | -0.53 .. +0.10 |
| det3:sat | det4:624 | y | 8 | tel | +0.12±0.18 | 13 | n/a | n/a | -0.41 .. -0.12 |
| det3:sat | det7:long | y | 6 | tel | -0.22±0.12 | 13 | 0.77 | -0.29 | -0.27 .. +0.13 |
| det3:sat | det7:long | y | 8 | tel | +0.12±0.20 | 12 | n/a | n/a | -0.27 .. +0.02 |
| det3:p2 | det4:624 | x | 6 | tel | +0.08±0.12 | 18 | 0.56 | +0.14 | -0.04 .. +0.08 |
| det3:p2 | det4:624 | x | 8 | tel | -0.31±0.15 | 14 | n/a | n/a | -0.30 .. -0.18 |
| det3:p2 | det4:624 | y | 6 | tel | +0.14±0.21 | 18 | 0.63 | +0.22 | -0.47 .. +0.19 |
| det3:p2 | det4:624 | y | 8 | tel | +0.07±0.18 | 14 | n/a | n/a | -0.36 .. -0.01 |
| det3:p2 | det7:long | y | 6 | tel | -0.18±0.13 | 14 | 0.81 | -0.23 | -0.23 .. -0.08 |
| det3:p2 | det7:long | y | 8 | tel | +0.12±0.16 | 12 | n/a | n/a | -0.23 .. +0.03 |
| det4:624 | det7:long | y | 6 | tel | -0.34±0.23 | 12 | 0.55 | -0.62 | -0.01 .. +0.25 |
### Part 4

(all gradients in 1e-3 per 100 mm; tau = w0/v is dimensionless)

| row | N cells x / y | d tau_x/dU | d tau_x/dW | d tau_y/dU | d tau_y/dW | cross difference (tau_x/dW - tau_y/dU) | significance | sum (sign check) |
|---|---|---|---|---|---|---|---|---|
| det2:long | 36 / 36 | +12.22±0.21 | +0.02±0.18 | -0.23±0.18 | +12.80±0.20 | +0.25±0.25 | +1.0 sigma | -0.21±0.26 |
| det2:longer | 32 / 32 | +11.85±0.99 | +0.15±0.71 | +0.33±0.75 | +13.93±0.84 | -0.17±0.98 | -0.2 sigma | +0.48±1.08 |
| det2:short | 8 / 8 | +19.19±3.71 | -2.35±3.58 | -13.09±4.73 | +20.13±3.74 | +10.74±5.78 | +1.9 sigma | -15.45±6.08 |
| det3:d625 | 36 / 36 | +10.35±0.25 | -1.95±0.26 | -1.69±0.20 | +20.05±0.26 | -0.25±0.35 | -0.7 sigma | -3.64±0.32 |
| det3:sat | 36 / 36 | +10.63±0.29 | +0.84±0.35 | -0.12±0.53 | +7.95±0.65 | +0.96±0.60 | +1.6 sigma | +0.73±0.67 |
| det3:p2 | 36 / 36 | +10.57±0.18 | +0.10±0.16 | -0.31±0.13 | +8.31±0.14 | +0.41±0.21 | +1.9 sigma | -0.21±0.20 |
| det4:623 | 19 / 19 | +15.33±14.43 | -0.96±4.21 | +9.39±3.93 | +25.68±2.13 | -10.35±5.77 | -1.8 sigma | +8.43±5.75 |
| det4:624 | 31 / 31 | +14.97±20.07 | +1.03±11.83 | +0.42±5.17 | +18.64±3.56 | +0.61±12.80 | +0.0 sigma | +1.45±13.03 |
| det6:long | 36 / 36 | +18.06±0.50 | -0.07±0.55 | -1.87±0.39 | +22.12±0.50 | +1.80±0.63 | +2.9 sigma | -1.94±0.73 |
| det6:longer | 33 / 33 | +16.02±0.85 | -0.51±0.89 | -1.61±0.73 | +21.84±1.18 | +1.10±1.04 | +1.1 sigma | -2.12±1.25 |
| det7:long | 27 / 27 | +5.50±6.76 | -0.38±3.07 | +0.26±0.64 | +14.38±0.48 | -0.65±3.09 | -0.2 sigma | -0.12±3.18 |
| det7:long|xhalf | 11 / 27 | +5.66±2.04 | +1.84±0.85 | +0.26±0.64 | +14.38±0.48 | +1.57±1.05 | +1.5 sigma | +2.10±1.08 |
| det7:longer | 18 / 18 | -29.06±146.68 | -8.93±98.38 | -0.39±1.07 | +14.27±0.91 | -8.54±98.39 | -0.1 sigma | -9.32±98.38 |
| det7:longer|xhalf | 7 / 18 | -6.10±5.65 | +6.00±1.68 | -0.39±1.07 | +14.27±0.92 | +6.39±1.93 | +3.3 sigma | +5.61±2.05 |
| det7:short | 8 / 8 | +14.47±9.10 | +3.93±4.31 | +7.40±4.59 | +13.50±2.42 | -3.47±6.07 | -0.6 sigma | +11.32±6.51 |
## Caveats

- Maps use cells with at least 40 events; empty cells are filled with the mean (as in track_grid). Rows with small train sets are therefore weak, and the table says so by N train.
- Constants (w0, v) are re-measured on the test row against the M3 reference in every arm, including constants-only in Part 2; this is two numbers per view and is not an M3-free quantity.
- Part 2 uses 2176 matched events (long, longer); d contains multiple scattering between the two chambers.
- The correlation errors in Part 3 are bootstrap errors over events only; the mirrored-map null range (about +-0.4) is the better yardstick for significance.
- Chamber voltages for 6-25 are not on disk (notes: resist about 500 V, drift 500 V, ambiguous); other rows from each run_config.json.
- Chamber D (det7) x < 200 mm is the faulty half; Part 3 and 4 give D X from the good half (x >= 200 mm).
