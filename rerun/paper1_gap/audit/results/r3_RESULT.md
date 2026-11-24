# R3: per-mounting angle maps on all 14 gate rows (audit tag r3, 2026-11-24)

Scripts (`rerun/paper1_gap/audit/`): `r3_lib.py` (loading, normalisation, generic map fit with a base model, inner CV), `r3_run.py <mounting>` (all CV arms), `r3_lc.py` (learning curve), `r3_gap.py` (gap maps, saturation, row consistency), `r3_report.py` (tables with paired bootstrap). Outputs here: `cv_*.json/npz`, `lc_*.json`, `gapstats.json`, `tables.md` (all per-arm tables, both CV schemes), `summary.json`, figures `figs/gap_{A,B,C,D,E}.png`.

## Method
- Mountings: B = det2 (long, longer, short); A625 = det3 6-25 (bottom slot, own mounting); A628 = det3 sat + p2; E623, E624 = det4 (two mountings, offsets and theta differ); C = det6 (3 rows); D = det7 (3 rows), run with all events and as Dx200 (x >= 200 mm only, both views). Alignment inside B, A628, C, D agrees between rows (z within 1 mm, offsets within 0.05 mm, same theta); see `cv_*.json` rows. But for C the rows do NOT share the physical state (see flags).
- Selection and reference exactly as `d_lib.load` (chamber's own active area, 30 mm margin, |tan_ref| <= 0.6). Position index = reconstructed mid-gap position. Pass 1 on A628 reproduces 0.923 / 0.805 (a2).
- Rows differ in drift velocity (C: 25.0, 18.9, 16.8 um/ns), so each row is normalised first: w_n = (w - c0)/c1 with its own robust ladder constants. Maps are then in tan units (a = tilt, s = relative scale), shared potential phi with kappa = 1 (a2 code, `a2_lib.fit_map`). Constants are refit per row on every test set (as in production).
- Direct map: phi + scale maps S (or S = 0); K in {6,9,12,16}, K_S in {0,4,8}, lambda in {0.01,0.1,1,10}, with K^2 + 2 K_S^2 <= N_train/15, chosen by 4-block inner CV on the training part only. Outer CV: 5 time blocks over the union ("time5") and leave-one-row-out ("loro", multi-row mountings; tables in `tables.md`).
- Reference-free: gap field = P-spline (K = 12, lambda = 0.1 as r1; K = 8 if N < 4000) of ln q_u90 (amplitude walk removed, per-row mean removed; y view only for D all-events), fitted on all events of the mounting (no reference used); mesh = bundle bow block per view (`h_x`, `h_y`). tilt = grad h + b (G0/(1+beta)) grad(delta ln q), G0 = 28.2 mm; b = 0.5 nominal, or fitted on the training blocks (a = 1 fixed, event level); b is quoted at beta = 0.68 amplitude. Velocity map v_rel = (g/<g>)^(-beta).
- beta: from the slope d ln v_rel / d ln q of the direct S map against the q field (beta = -s/(1+s)), jackknife over the 5 fold maps; used when slope < 0, > 3 sigma and 0.2 < beta < 2, else 0.68. Used: B 0.80, A625 1.38, A628 0.94, E623 1.59, E624 0.68 (no S map chosen), C 0.68 (slope -0.85 gives beta 5.5, not credible), D 0.68 (slope -0.72 +- 0.30), Dx200 0.68.
- Both: reference-free base, then a direct map of the remainder, selected by inner CV on the training part.
- Errors: paired bootstrap over events (400 resamples, predictions fixed); fitted b and slopes: jackknife over folds.

## Verdicts

**1. Per-mounting direct map (time-block CV).** Gain over pass 1, all with errors about 0.01 deg when N > 5000:
- A628 (18.7k events): X 0.923 -> 0.688 (-0.235 +- 0.006), Y 0.805 -> 0.599 (-0.206 +- 0.007): 25-26 %, matches a2 (0.686 / 0.599).
- B (14.2k): X 0.961 -> 0.675 (-30 %), Y 0.966 -> 0.735 (-24 %). A625 (6.3k): X 0.876 -> 0.681 (-22 %), Y 1.030 -> 0.666 (-35 %). So B, A625, A628 all reach 0.6-0.74 deg: **the same resolution on three mountings of two chambers.**
- C (8.0k): X 1.183 -> 0.932 (-21 %), Y 1.480 -> 1.228 (-17 %). D (5.1k, x half faulty): Y 1.028 -> 0.829 (-19 %), X on the good half 0.968 -> 0.802 (-17 %; Dx200 trained on the good half only 0.796); X on the whole D 1.617 -> 1.463 (-9.5 %) because the faulty half is not fixable by a map.
- E (det4, 1.6k and 3.1k events): X no gain (-0.05 +- 0.05, -0.02 +- 0.04), Y -15 % (E623, -0.25 +- 0.05) and -8 % (E624, -0.11 +- 0.03). Pass 1 is 1.4-2.4 deg here, so the angle noise is dominated by something else and N is small.
- The S map (velocity) matters: S = 0 (tilt only) is 0.01-0.04 deg worse in A, B, D, 0.1 deg worse in C.
- Learning-curve asymptote (s68^2 = s_inf^2 + c/N, 2 seeds x 3 windows; the K grid is restricted by N/15, which makes steps in the curve): B 0.63 +- 0.03 / 0.72 +- 0.02; A625 0.67 +- 0.03 / 0.65 +- 0.03; A628 0.65 +- 0.04 / 0.58 +- 0.02; C 0.89 +- 0.06 / 1.20 +- 0.05, against full-statistics 0.675 / 0.735, 0.682 / 0.666, 0.688 / 0.600, 0.930 / 1.222. So a map from all events is within 0.01-0.05 deg of the noise limit; mountings below ~5000 events (D, E) are on the steep part of the curve.

**2. Reference-free (timing + mesh only).** Tilt + velocity with b fitted recovers about half of the direct gain on the large mountings: B X -0.14 +- 0.01 (ratio to direct 0.49), Y -0.10 (0.44); A628 X -0.10, Y -0.08 (0.44, 0.38); A625 X -0.11 (0.56), Y -0.24 (0.67); C X -0.13, Y -0.14 (0.53, 0.54); D Y -0.13 (0.66), D X good half -0.11 (0.65); E623 Y -0.17 +- 0.05 (0.68), E624 Y -0.07 (0.58); E X zero. Reference-free s68: B 0.82 / 0.86, A628 0.82 / 0.73, A625 0.77 / 0.79, C 1.05 / 1.34. Nominal b = 0.5 tilt only gets only 0.2-0.5 of the direct gain (e.g. B X 0.854 vs pass 1 0.961). **So roughly 40-65 % of the achievable gain (10-25 % of the s68) comes from timing alone, and it needs no reference tracks.** The gain exists in every mounting where the direct map works, including the ones golden r1 never saw (B, C, D, E Y).
- Fitted b (a = 1; jackknife over blocks; at beta = 0.68 amplitude): B 0.37 +- 0.02 (x 0.34, y 0.38); A625 0.22 +- 0.01; A628 0.46 +- 0.00 (x 0.37, y 0.51; chamber A earlier 0.35-0.54); E623 0.17 +- 0.05; E624 0.09 +- 0.05; C 0.18 +- 0.04 (x 0.09 +- 0.04, y 0.36 +- 0.04; loro 0.16 +- 0.08); D 0.47 +- 0.02; Dx200 0.31 +- 0.01 (x 0.54 +- 0.07, y 0.27 +- 0.02). b is clearly below 0.5 on B, A625, C and E and differs between mountings and views, so the amplitude is empirical (different drift voltages: C 700 V, E 600 V, B/A 1000 V, A625 500 V) and has to be fitted per mounting, which needs reference tracks (only one number per mounting, a few hundred events suffice).
- Measured beta from the S map (d ln v / d ln q = -0.44 +- 0.08 B, -0.58 +- 0.01 A625, -0.48 +- 0.02 A628, -0.61 +- 0.09 E623) gives 0.8-1.6, above the Magboltz 0.68 at A's operating point; the S map includes angle-dependent response, so this is an effective number. The velocity term helps (e.g. B X 0.854 -> 0.822 with fitted b).

**3. Both.** Reference-free first, then a direct map of the remainder, brings nothing beyond the direct map on large mountings: B X 0.673 vs 0.675, Y 0.729 vs 0.735; A628 0.689 / 0.602 vs 0.688 / 0.599; A625 0.685 / 0.659 vs 0.681 / 0.666; C 0.925 / 1.223 vs 0.932 / 1.228 (all differences within 0.01, errors 0.005-0.01). For the small mountings it helps a little, within errors: D X(half) 0.787 vs 0.802, D Y 0.820 vs 0.829, Dx200 Y 0.875 vs 0.869 (no change), E624 X 2.054 vs 2.120 (-0.07, not verified against its paired error, about 0.04), E623 Y 1.395 vs 1.377. Conclusion: the direct map is at its noise limit with >= 5000 events and the timing information is redundant there; below ~3000 events the timing prior is useful as a regulariser but the gain is at most 0.03-0.07 deg.

## Summary
(see the table below; "relative gain" = change of s68 against pass 1; time-block CV; ref-free = tilt + velocity with fitted b)

## Gap maps (`figs/gap_A.png` ... `gap_E.png`; delta g in mm for beta 0.68; left full, right with quadratic removed)
- B (N 14230): flat along x; bowl along y (-1.5); tilt y +0.41 mm/100 mm; folds rms 0.36 mm; p2p 3.2 mm. Fine-structure p2p 1.7 mm.
- A625 (N 6279): dome along x (+0.9); dome along y (+0.8); twist (-0.7); tilt x +0.51, y -0.22 mm/100 mm; folds rms 0.68 mm; p2p 5.8 mm. Fine-structure p2p 3.6 mm.
- A628 (N 18720): dome along x (+0.5); bowl along y (-1.5); tilt x +0.50, y -0.35 mm/100 mm; folds rms 0.43 mm; p2p 3.5 mm. Fine-structure p2p 1.8 mm.
- E623 (N 1610): bowl along x (-0.5); dome along y (+1.3); tilt x +0.44, y -1.06 mm/100 mm; folds rms 0.35 mm; p2p 3.7 mm. Fine-structure p2p 2.0 mm.
- E624 (N 3128): bowl along x (-0.7); dome along y (+0.8); tilt x +0.43, y -1.06 mm/100 mm; folds rms 0.38 mm; p2p 4.0 mm. Fine-structure p2p 1.9 mm.
- C (N 7985): dome along x (+1.9); dome along y (+1.0); twist (-0.6); tilt x +0.45, y -1.05 mm/100 mm; folds rms 0.50 mm; p2p 6.5 mm. Fine-structure p2p 2.4 mm.
- D (N 5137): flat along x; flat along y; twist (+0.4); tilt y -0.26 mm/100 mm; folds rms 0.43 mm; p2p 2.0 mm. Fine-structure p2p 1.7 mm.
- Dx200 (N 1934): bowl along x (-0.3); dome along y (+0.7); twist (+0.7); tilt y -0.28 mm/100 mm; folds rms 0.39 mm; p2p 2.5 mm. Fine-structure p2p 1.8 mm.
(Convention as r1: negative curvature of delta g = "bowl", positive = "dome"; curvature in mm per (100 mm)^2; p2p = 1st to 99th percentile of the map inside the covered area. A628 reproduces r1 (gww -1.5, dome along x +0.5); A625 reproduces r1's different mounting (dome in y, twist). The y curvature changes sign between B/A628 (bowl) and E, C, A625, D (dome or flat): the gap shape is a property of the mounting.)

## Flags
1. **C: the three rows are not one mounting in practice.** Alignment is the same, but v falls 25.0 -> 18.9 -> 16.8 um/ns and the saturated fraction goes 92 % -> 33 % -> 0 % across the rows. Gap maps of the rows: long vs longer r = 0.92, but short vs long -0.05 and short vs longer 0.01 (full maps; short has 813 events). The leave-one-row-out gain is much smaller than the time-block gain: X 1.183 -> 1.037, Y 1.480 -> 1.285 (-12 %, -13 %), per row (held out) long 1.128 -> 0.993, longer 1.237 -> 1.020, short 1.440 -> 1.288; the time5 numbers (0.932 / 1.228) are optimistic because every test block shares rows with the training. Beta is not measurable on C. C maps should be made per row or per hour, not per night.
2. Mountings B, A628, D behave: LORO ~ time5 (B X 0.697 vs 0.675, Y 0.742 vs 0.735; A628 0.696 / 0.610 vs 0.688 / 0.599; D 1.471 / 0.850 vs 1.463 / 0.829). A map trained on the other rows works on a held-out row, including the 1100-event short rows of B and D.
3. **E: no map can matter** (s68 1.4-2.4 deg, X unchanged). Not a map failure; pass 1 itself is poor there (earlier agents). E623 and E624 are different mountings (offsets 216.6/178.6 vs 215.0/186.3, theta 89.25 vs 90.05).
4. **D X faulty half**: with all events X pass 1 is 1.617 and the map only gets 1.463; on the good half 0.968 -> 0.802. Quote D X on x >= 200 mm only. D gap field uses the y view only (X half is bad). D rows' gap maps agree (r 0.83-0.93).
5. **Saturation.** Saturated-strip fraction per row: B 0.52/0.49/0.50; A625 0.87; A628 0.26/0.29; C 0.92/0.33/0.0; D 0.75/0.68/0.62; E 0.0. q_u90 is lower for events with saturated strips: ln q shift -0.025 (B), -0.049 (A625), -0.029 (A628), -0.058 (C long), -0.024 (D), all +-0.002-0.006, i.e. 0.4-0.8 mm in delta g, comparable to the whole fine-structure rms (0.35-0.5 mm). But the position dependence is weak: the field from unsaturated events only agrees with the full field at r = 0.99 (B, N 6.9k), 0.998 (A628, N 13k) full map, fine r 0.97 / 0.99; poorer for small unsaturated samples (A625 832 events fine r 0.82; C 2685 events fine r 0.64, D 1481 fine r 0.86). The map of the saturated fraction correlates with the q map at |r| <= 0.32 (B 0.32, C -0.27, D 0.14, A 0.04, 0.03). So saturation shifts q_u90 as an offset but does not create the gap map, except possibly in C where the saturated fraction changes between rows. Not tested: whether the reference-free arm degrades with unsaturated-only fields.
6. Caveats: (a) per-row constants c0, c1 used for the normalisation come from the whole row (two numbers per row, refit in production anyway); (b) the gap field and beta are fitted on all events of the mounting (reference-free for the field; beta is one scalar from the direct map, mild leak); (c) b fitted on training blocks only; (d) inner K, K_S, lambda grids are coarse and lambda sits at the grid edge in several folds; (e) bootstrap errors do not include training variance (the learning-curve scatter is the better guide: 0.01-0.04 deg); (f) G0 = 28.2 mm assumed for all chambers, absorbed by b; (g) the drift voltages differ (D/E/C 600-700 V vs 1000 V), so b and beta are not comparable across chambers.

## Verdict
- **Worth shipping per mounting: A, B, C, D (good X half and Y)**, expected s68 about 0.6-0.7 deg (A, B), 0.8 deg (D, good half, Y 0.83), 0.93 / 1.23 deg (C, but see flag 1: use per-row maps; row-held-out gain 12 %). Gain 17-35 % over pass 1 when the mounting has >= 5000 events; the learning curve says ~5000 events already reach within 0.02-0.04 deg of the asymptote on A/B, and ~2000 events (Dx200, rows of D) give 17-23 %.
- **Not worth it: E** (no X gain, Y 8-15 % on 1.4-1.6 deg, small N).
- **Timing alone** (reference-free, with one fitted amplitude b per mounting and view) gives 40-65 % of the direct gain, 10-25 % of the s68 (A628 0.82 / 0.73; B 0.82 / 0.86). It adds nothing once a direct map with >= 5000 events exists, and a modest, error-limited help (0.03-0.07 deg) below ~3000 events. It is the only option for mountings without reference tracks, but the amplitude b (0.1-0.5) must be calibrated per mounting.
# R3 tables (time-block CV: 5 blocks; loro = leave one row out)

## Summary (time5)

| mounting | view | N | pass 1 | direct map CV (phi + S) | direct, S = 0 | reference-free (tilt + v, b fitted) | reference-free (tilt, b = 0.5) | both | asymptote | gain direct | gain ref-free | ref-free / direct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | X | 14230 | 0.961 +- 0.008 | 0.675 +- 0.006 | 0.709 +- 0.006 | 0.822 +- 0.008 | 0.854 +- 0.008 | 0.673 +- 0.006 | 0.626 +- 0.034 | +29.7 % (diff -0.285 +- 0.008) | +14.5 % (diff -0.139 +- 0.007) | 0.49 |
| B | Y | 14230 | 0.966 +- 0.008 | 0.735 +- 0.006 | 0.740 +- 0.007 | 0.864 +- 0.007 | 0.883 +- 0.008 | 0.729 +- 0.006 | 0.715 +- 0.015 | +23.9 % (diff -0.231 +- 0.008) | +10.6 % (diff -0.102 +- 0.007) | 0.44 |
| A625 | X | 6279 | 0.876 +- 0.011 | 0.681 +- 0.010 | 0.742 +- 0.010 | 0.767 +- 0.010 | 0.825 +- 0.010 | 0.685 +- 0.009 | 0.667 +- 0.028 | +22.3 % (diff -0.195 +- 0.011) | +12.4 % (diff -0.109 +- 0.009) | 0.56 |
| A625 | Y | 6279 | 1.030 +- 0.017 | 0.666 +- 0.008 | 0.705 +- 0.010 | 0.788 +- 0.012 | 0.789 +- 0.011 | 0.659 +- 0.009 | 0.646 +- 0.030 | +35.3 % (diff -0.364 +- 0.017) | +23.5 % (diff -0.242 +- 0.016) | 0.67 |
| A628 | X | 18720 | 0.923 +- 0.006 | 0.688 +- 0.005 | 0.725 +- 0.006 | 0.819 +- 0.007 | 0.852 +- 0.007 | 0.689 +- 0.005 | 0.646 +- 0.038 | +25.5 % (diff -0.235 +- 0.006) | +11.2 % (diff -0.104 +- 0.006) | 0.44 |
| A628 | Y | 18720 | 0.805 +- 0.007 | 0.599 +- 0.005 | 0.621 +- 0.005 | 0.728 +- 0.006 | 0.746 +- 0.006 | 0.602 +- 0.005 | 0.575 +- 0.021 | +25.6 % (diff -0.206 +- 0.007) | +9.6 % (diff -0.078 +- 0.005) | 0.38 |
| E623 | X | 1610 | 2.405 +- 0.075 | 2.354 +- 0.077 | 2.406 +- 0.071 | 2.424 +- 0.079 | 2.501 +- 0.077 | 2.346 +- 0.085 | n/a | +2.1 % (diff -0.051 +- 0.047) | -0.8 % (diff +0.019 +- 0.042) | n/a |
| E623 | Y | 1610 | 1.627 +- 0.068 | 1.377 +- 0.051 | 1.471 +- 0.060 | 1.457 +- 0.069 | 1.499 +- 0.069 | 1.395 +- 0.050 | n/a | +15.4 % (diff -0.251 +- 0.048) | +10.5 % (diff -0.170 +- 0.045) | 0.68 |
| E624 | X | 3128 | 2.141 +- 0.057 | 2.120 +- 0.048 | 2.120 +- 0.048 | 2.121 +- 0.052 | 2.158 +- 0.050 | 2.054 +- 0.043 | n/a | +1.0 % (diff -0.021 +- 0.038) | +0.9 % (diff -0.020 +- 0.038) | n/a |
| E624 | Y | 3128 | 1.436 +- 0.040 | 1.324 +- 0.042 | 1.324 +- 0.042 | 1.372 +- 0.036 | 1.489 +- 0.047 | 1.278 +- 0.039 | n/a | +7.8 % (diff -0.112 +- 0.031) | +4.5 % (diff -0.065 +- 0.026) | 0.58 |
| C | X | 7985 | 1.183 +- 0.016 | 0.932 +- 0.010 | 1.049 +- 0.014 | 1.051 +- 0.012 | 1.163 +- 0.014 | 0.925 +- 0.012 | 0.890 +- 0.063 | +21.2 % (diff -0.251 +- 0.013) | +11.2 % (diff -0.132 +- 0.012) | 0.53 |
| C | Y | 7985 | 1.480 +- 0.020 | 1.228 +- 0.014 | 1.341 +- 0.016 | 1.343 +- 0.017 | 1.384 +- 0.016 | 1.223 +- 0.017 | 1.203 +- 0.054 | +17.0 % (diff -0.252 +- 0.016) | +9.2 % (diff -0.136 +- 0.014) | 0.54 |
| D | X | 5137 | 1.617 +- 0.033 | 1.463 +- 0.029 | 1.476 +- 0.034 | 1.506 +- 0.033 | 1.520 +- 0.030 | 1.433 +- 0.030 | n/a | +9.5 % (diff -0.154 +- 0.023) | +6.9 % (diff -0.111 +- 0.021) | 0.72 |
| D | X (x>=200 events of D) | 1934 | 0.968 +- 0.023 | 0.802 +- 0.019 | 0.802 +- 0.022 | 0.860 +- 0.019 | 0.878 +- 0.020 | 0.787 +- 0.020 | n/a | +17.1 % (diff -0.166 +- 0.019) | +11.2 % (diff -0.108 +- 0.017) | 0.65 |
| D | Y | 5137 | 1.028 +- 0.015 | 0.829 +- 0.013 | 0.846 +- 0.013 | 0.896 +- 0.016 | 0.930 +- 0.015 | 0.820 +- 0.012 | n/a | +19.3 % (diff -0.199 +- 0.015) | +12.8 % (diff -0.132 +- 0.015) | 0.66 |
| Dx200 | X | 1934 | 0.968 +- 0.023 | 0.796 +- 0.018 | 0.796 +- 0.018 | 0.885 +- 0.026 | 0.899 +- 0.017 | 0.781 +- 0.021 | n/a | +17.7 % (diff -0.171 +- 0.019) | +8.5 % (diff -0.082 +- 0.020) | 0.48 |
| Dx200 | Y | 1934 | 1.124 +- 0.030 | 0.869 +- 0.024 | 0.869 +- 0.024 | 1.018 +- 0.026 | 1.058 +- 0.027 | 0.875 +- 0.023 | n/a | +22.6 % (diff -0.254 +- 0.028) | +9.4 % (diff -0.106 +- 0.029) | 0.42 |

## Mounting B: B (det2), rows ['det2:long', 'det2:longer', 'det2:short'], N = 14230

beta: {'slope': -0.44431136749560346, 'err': 0.07938699429099091, 'slope_x': -0.5605331181698332, 'err_x': 0.09227711642293959, 'slope_y': -0.3293682561986238, 'err_y': 0.061539837234603566, 'beta_meas': 0.7995689339426754, 'measurable': True, 'beta_used': 0.7995689339426754, 'nfold': 5}


### B, scheme time5: fitted b = 0.37 +- 0.02 (x 0.34 +- 0.02, y 0.38 +- 0.02); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [16, 8, 0.1, 1.406]; dir_phiS:1: [16, 8, 0.1, 1.417]; dir_phiS:2: [16, 8, 1.0, 1.418]; dir_phiS:3: [16, 8, 0.01, 1.397]; dir_phiS:4: [12, 8, 1.0, 1.402]; both_phiS:0: [16, 4, 10.0, 1.404]; both_phiS:1: [16, 8, 10.0, 1.412]; both_phiS:2: [16, 4, 10.0, 1.409]; both_phiS:3: [16, 8, 0.1, 1.396]; both_phiS:4: [16, 8, 10.0, 1.393]


**B time5 X view** (n = 14230)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.961 +- 0.008 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.089 +- 0.009 | +0.128 +- 0.008 | -13.4 % |
| direct phi + S | 0.675 +- 0.006 | -0.285 +- 0.008 | +29.7 % |
| direct phi (S = 0) | 0.709 +- 0.006 | -0.251 +- 0.007 | +26.2 % |
| ref-free: mesh only | 0.907 +- 0.008 | -0.053 +- 0.006 | +5.6 % |
| ref-free tilt, b = 0.5 | 0.854 +- 0.008 | -0.106 +- 0.007 | +11.1 % |
| ref-free tilt, b fitted | 0.853 +- 0.007 | -0.108 +- 0.007 | +11.2 % |
| ref-free tilt + v, b = 0.5 | 0.826 +- 0.009 | -0.134 +- 0.007 | +14.0 % |
| ref-free tilt + v, b fitted | 0.822 +- 0.008 | -0.139 +- 0.007 | +14.5 % |
| ref-free (tilt+v) then direct phi + S | 0.673 +- 0.006 | -0.288 +- 0.008 | +30.0 % |
| ref-free tilt then direct phi (S = 0) | 0.710 +- 0.006 | -0.251 +- 0.007 | +26.1 % |

**B time5 Y view** (n = 14230)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.966 +- 0.008 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.174 +- 0.010 | +0.207 +- 0.012 | -21.4 % |
| direct phi + S | 0.735 +- 0.006 | -0.231 +- 0.008 | +23.9 % |
| direct phi (S = 0) | 0.740 +- 0.007 | -0.226 +- 0.008 | +23.4 % |
| ref-free: mesh only | 0.949 +- 0.008 | -0.017 +- 0.005 | +1.8 % |
| ref-free tilt, b = 0.5 | 0.883 +- 0.008 | -0.084 +- 0.007 | +8.7 % |
| ref-free tilt, b fitted | 0.881 +- 0.007 | -0.086 +- 0.007 | +8.9 % |
| ref-free tilt + v, b = 0.5 | 0.868 +- 0.007 | -0.098 +- 0.007 | +10.2 % |
| ref-free tilt + v, b fitted | 0.864 +- 0.007 | -0.102 +- 0.007 | +10.6 % |
| ref-free (tilt+v) then direct phi + S | 0.729 +- 0.006 | -0.238 +- 0.008 | +24.6 % |
| ref-free tilt then direct phi (S = 0) | 0.740 +- 0.007 | -0.227 +- 0.008 | +23.5 % |

### B, scheme loro: fitted b = 0.37 +- 0.00 (x 0.34 +- 0.01, y 0.38 +- 0.01); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [12, 4, 0.01, 1.442]; dir_phiS:1: [16, 8, 1.0, 1.408]; dir_phiS:2: [16, 8, 0.01, 1.405]; both_phiS:0: [12, 4, 1.0, 1.425]; both_phiS:1: [16, 4, 10.0, 1.407]; both_phiS:2: [12, 8, 0.01, 1.403]


**B loro X view** (n = 14230)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.961 +- 0.008 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.089 +- 0.009 | +0.128 +- 0.008 | -13.4 % |
| direct phi + S | 0.697 +- 0.007 | -0.264 +- 0.007 | +27.5 % |
| direct phi (S = 0) | 0.724 +- 0.007 | -0.237 +- 0.008 | +24.6 % |
| ref-free: mesh only | 0.914 +- 0.008 | -0.047 +- 0.006 | +4.9 % |
| ref-free tilt, b = 0.5 | 0.860 +- 0.008 | -0.101 +- 0.007 | +10.5 % |
| ref-free tilt, b fitted | 0.859 +- 0.008 | -0.101 +- 0.007 | +10.5 % |
| ref-free tilt + v, b = 0.5 | 0.836 +- 0.008 | -0.125 +- 0.007 | +13.0 % |
| ref-free tilt + v, b fitted | 0.839 +- 0.008 | -0.121 +- 0.007 | +12.6 % |
| ref-free (tilt+v) then direct phi + S | 0.688 +- 0.006 | -0.272 +- 0.007 | +28.3 % |
| ref-free tilt then direct phi (S = 0) | 0.726 +- 0.006 | -0.235 +- 0.008 | +24.5 % |

**B loro Y view** (n = 14230)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.966 +- 0.008 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.174 +- 0.010 | +0.207 +- 0.012 | -21.4 % |
| direct phi + S | 0.742 +- 0.006 | -0.224 +- 0.007 | +23.2 % |
| direct phi (S = 0) | 0.747 +- 0.007 | -0.219 +- 0.008 | +22.7 % |
| ref-free: mesh only | 0.952 +- 0.008 | -0.014 +- 0.005 | +1.5 % |
| ref-free tilt, b = 0.5 | 0.882 +- 0.007 | -0.085 +- 0.007 | +8.8 % |
| ref-free tilt, b fitted | 0.884 +- 0.008 | -0.082 +- 0.007 | +8.5 % |
| ref-free tilt + v, b = 0.5 | 0.869 +- 0.008 | -0.097 +- 0.007 | +10.0 % |
| ref-free tilt + v, b fitted | 0.869 +- 0.008 | -0.097 +- 0.006 | +10.1 % |
| ref-free (tilt+v) then direct phi + S | 0.737 +- 0.007 | -0.230 +- 0.008 | +23.8 % |
| ref-free tilt then direct phi (S = 0) | 0.749 +- 0.007 | -0.218 +- 0.008 | +22.5 % |


## Mounting A625: A (det3), rows ['det3:d625'], N = 6279

beta: {'slope': -0.5794293724567934, 'err': 0.014304119333702308, 'slope_x': -0.6006099948204684, 'err_x': 0.009938885256895489, 'slope_y': -0.5583616459138907, 'err_y': 0.027649470754729913, 'beta_meas': 1.3777219199580615, 'measurable': True, 'beta_used': 1.3777219199580615, 'nfold': 5}


### A625, scheme time5: fitted b = 0.22 +- 0.01 (x 0.20 +- 0.01, y 0.22 +- 0.01); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [9, 8, 0.01, 1.349]; dir_phiS:1: [12, 8, 0.01, 1.345]; dir_phiS:2: [12, 8, 0.01, 1.339]; dir_phiS:3: [12, 8, 0.01, 1.35]; dir_phiS:4: [9, 8, 0.01, 1.354]; both_phiS:0: [12, 8, 10.0, 1.334]; both_phiS:1: [16, 4, 10.0, 1.331]; both_phiS:2: [9, 4, 1.0, 1.339]; both_phiS:3: [16, 4, 10.0, 1.343]; both_phiS:4: [12, 8, 10.0, 1.345]


**A625 time5 X view** (n = 6279)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.876 +- 0.011 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.065 +- 0.011 | +0.189 +- 0.010 | -21.5 % |
| direct phi + S | 0.681 +- 0.010 | -0.195 +- 0.011 | +22.3 % |
| direct phi (S = 0) | 0.742 +- 0.010 | -0.134 +- 0.010 | +15.3 % |
| ref-free: mesh only | 0.871 +- 0.010 | -0.006 +- 0.007 | +0.6 % |
| ref-free tilt, b = 0.5 | 0.825 +- 0.010 | -0.051 +- 0.010 | +5.9 % |
| ref-free tilt, b fitted | 0.829 +- 0.011 | -0.047 +- 0.009 | +5.4 % |
| ref-free tilt + v, b = 0.5 | 0.763 +- 0.011 | -0.113 +- 0.010 | +12.9 % |
| ref-free tilt + v, b fitted | 0.767 +- 0.010 | -0.109 +- 0.009 | +12.4 % |
| ref-free (tilt+v) then direct phi + S | 0.685 +- 0.009 | -0.192 +- 0.010 | +21.9 % |
| ref-free tilt then direct phi (S = 0) | 0.746 +- 0.009 | -0.131 +- 0.010 | +14.9 % |

**A625 time5 Y view** (n = 6279)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.030 +- 0.017 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.305 +- 0.023 | +0.275 +- 0.013 | -26.7 % |
| direct phi + S | 0.666 +- 0.008 | -0.364 +- 0.017 | +35.3 % |
| direct phi (S = 0) | 0.705 +- 0.010 | -0.325 +- 0.017 | +31.6 % |
| ref-free: mesh only | 0.890 +- 0.013 | -0.140 +- 0.012 | +13.6 % |
| ref-free tilt, b = 0.5 | 0.789 +- 0.011 | -0.241 +- 0.016 | +23.4 % |
| ref-free tilt, b fitted | 0.795 +- 0.011 | -0.235 +- 0.016 | +22.8 % |
| ref-free tilt + v, b = 0.5 | 0.799 +- 0.010 | -0.231 +- 0.017 | +22.4 % |
| ref-free tilt + v, b fitted | 0.788 +- 0.012 | -0.242 +- 0.016 | +23.5 % |
| ref-free (tilt+v) then direct phi + S | 0.659 +- 0.009 | -0.370 +- 0.017 | +36.0 % |
| ref-free tilt then direct phi (S = 0) | 0.703 +- 0.010 | -0.326 +- 0.017 | +31.7 % |


## Mounting A628: A (det3), rows ['det3:sat', 'det3:p2'], N = 18720

beta: {'slope': -0.4834243538929169, 'err': 0.021296107520189088, 'slope_x': -0.5444772196819652, 'err_x': 0.020835148326086288, 'slope_y': -0.37965006101321386, 'err_y': 0.024840342388754745, 'beta_meas': 0.9358249029663039, 'measurable': True, 'beta_used': 0.9358249029663039, 'nfold': 5}


### A628, scheme time5: fitted b = 0.46 +- 0.00 (x 0.37 +- 0.01, y 0.51 +- 0.01); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [16, 8, 0.01, 1.297]; dir_phiS:1: [16, 8, 0.01, 1.287]; dir_phiS:2: [16, 8, 0.01, 1.286]; dir_phiS:3: [16, 8, 0.1, 1.285]; dir_phiS:4: [16, 8, 0.1, 1.289]; both_phiS:0: [16, 8, 0.1, 1.3]; both_phiS:1: [16, 8, 0.01, 1.288]; both_phiS:2: [16, 4, 1.0, 1.287]; both_phiS:3: [16, 4, 1.0, 1.289]; both_phiS:4: [16, 8, 0.1, 1.292]


**A628 time5 X view** (n = 18720)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.923 +- 0.006 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.042 +- 0.007 | +0.119 +- 0.006 | -12.9 % |
| direct phi + S | 0.688 +- 0.005 | -0.235 +- 0.006 | +25.5 % |
| direct phi (S = 0) | 0.725 +- 0.006 | -0.198 +- 0.007 | +21.5 % |
| ref-free: mesh only | 0.913 +- 0.006 | -0.010 +- 0.005 | +1.1 % |
| ref-free tilt, b = 0.5 | 0.852 +- 0.007 | -0.071 +- 0.006 | +7.7 % |
| ref-free tilt, b fitted | 0.853 +- 0.007 | -0.070 +- 0.006 | +7.6 % |
| ref-free tilt + v, b = 0.5 | 0.817 +- 0.007 | -0.106 +- 0.006 | +11.5 % |
| ref-free tilt + v, b fitted | 0.819 +- 0.007 | -0.104 +- 0.006 | +11.2 % |
| ref-free (tilt+v) then direct phi + S | 0.689 +- 0.005 | -0.234 +- 0.006 | +25.4 % |
| ref-free tilt then direct phi (S = 0) | 0.731 +- 0.006 | -0.193 +- 0.007 | +20.9 % |

**A628 time5 Y view** (n = 18720)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.805 +- 0.007 | +0.000 +- 0.000 | -0.0 % |
| constants only | 0.934 +- 0.007 | +0.128 +- 0.006 | -16.0 % |
| direct phi + S | 0.599 +- 0.005 | -0.206 +- 0.007 | +25.6 % |
| direct phi (S = 0) | 0.621 +- 0.005 | -0.184 +- 0.006 | +22.9 % |
| ref-free: mesh only | 0.849 +- 0.007 | +0.043 +- 0.006 | -5.4 % |
| ref-free tilt, b = 0.5 | 0.746 +- 0.006 | -0.059 +- 0.005 | +7.4 % |
| ref-free tilt, b fitted | 0.745 +- 0.006 | -0.060 +- 0.005 | +7.4 % |
| ref-free tilt + v, b = 0.5 | 0.729 +- 0.006 | -0.077 +- 0.005 | +9.5 % |
| ref-free tilt + v, b fitted | 0.728 +- 0.006 | -0.078 +- 0.005 | +9.6 % |
| ref-free (tilt+v) then direct phi + S | 0.602 +- 0.005 | -0.204 +- 0.007 | +25.3 % |
| ref-free tilt then direct phi (S = 0) | 0.624 +- 0.005 | -0.181 +- 0.006 | +22.5 % |

### A628, scheme loro: fitted b = 0.46 +- 0.01 (x 0.38 +- 0.01, y 0.51 +- 0.00); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [16, 8, 0.01, 1.3]; dir_phiS:1: [12, 8, 0.01, 1.279]; both_phiS:0: [16, 8, 0.01, 1.3]; both_phiS:1: [12, 8, 0.01, 1.279]


**A628 loro X view** (n = 18720)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.923 +- 0.006 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.042 +- 0.007 | +0.119 +- 0.006 | -12.9 % |
| direct phi + S | 0.696 +- 0.005 | -0.227 +- 0.006 | +24.6 % |
| direct phi (S = 0) | 0.728 +- 0.006 | -0.195 +- 0.007 | +21.1 % |
| ref-free: mesh only | 0.912 +- 0.006 | -0.011 +- 0.004 | +1.2 % |
| ref-free tilt, b = 0.5 | 0.851 +- 0.007 | -0.072 +- 0.007 | +7.8 % |
| ref-free tilt, b fitted | 0.852 +- 0.007 | -0.072 +- 0.006 | +7.8 % |
| ref-free tilt + v, b = 0.5 | 0.820 +- 0.006 | -0.103 +- 0.005 | +11.2 % |
| ref-free tilt + v, b fitted | 0.821 +- 0.006 | -0.102 +- 0.006 | +11.0 % |
| ref-free (tilt+v) then direct phi + S | 0.695 +- 0.005 | -0.228 +- 0.006 | +24.7 % |
| ref-free tilt then direct phi (S = 0) | 0.731 +- 0.005 | -0.192 +- 0.006 | +20.8 % |

**A628 loro Y view** (n = 18720)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.805 +- 0.007 | +0.000 +- 0.000 | -0.0 % |
| constants only | 0.934 +- 0.007 | +0.128 +- 0.006 | -16.0 % |
| direct phi + S | 0.610 +- 0.005 | -0.195 +- 0.007 | +24.3 % |
| direct phi (S = 0) | 0.626 +- 0.005 | -0.179 +- 0.006 | +22.2 % |
| ref-free: mesh only | 0.848 +- 0.006 | +0.042 +- 0.006 | -5.3 % |
| ref-free tilt, b = 0.5 | 0.748 +- 0.006 | -0.058 +- 0.005 | +7.1 % |
| ref-free tilt, b fitted | 0.746 +- 0.006 | -0.059 +- 0.005 | +7.4 % |
| ref-free tilt + v, b = 0.5 | 0.730 +- 0.006 | -0.075 +- 0.005 | +9.4 % |
| ref-free tilt + v, b fitted | 0.729 +- 0.006 | -0.077 +- 0.005 | +9.5 % |
| ref-free (tilt+v) then direct phi + S | 0.613 +- 0.005 | -0.192 +- 0.006 | +23.9 % |
| ref-free tilt then direct phi (S = 0) | 0.630 +- 0.005 | -0.175 +- 0.007 | +21.8 % |


## Mounting E623: E (det4), rows ['det4:623'], N = 1610

beta: {'slope': -0.6145054310949777, 'err': 0.09169762808458481, 'slope_x': -0.620410561156441, 'err_x': 0.12675796761799948, 'slope_y': -0.64638014332991, 'err_y': 0.0480625313076307, 'beta_meas': 1.5940702688508661, 'measurable': True, 'beta_used': 1.5940702688508661, 'nfold': 5}


### E623, scheme time5: fitted b = 0.17 +- 0.05 (x 0.43 +- 0.20, y 0.14 +- 0.03); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [6, 4, 1.0, 3.721]; dir_phiS:1: [6, 4, 0.1, 3.698]; dir_phiS:2: [6, 4, 1.0, 3.622]; dir_phiS:3: [6, 4, 1.0, 3.77]; dir_phiS:4: [6, 4, 0.1, 3.745]; both_phiS:0: [6, 0, 1.0, 3.718]; both_phiS:1: [6, 4, 0.01, 3.659]; both_phiS:2: [9, 0, 1.0, 3.618]; both_phiS:3: [9, 0, 10.0, 3.715]; both_phiS:4: [9, 0, 1.0, 3.73]


**E623 time5 X view** (n = 1610)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 2.405 +- 0.075 | +0.000 +- 0.000 | -0.0 % |
| constants only | 2.617 +- 0.078 | +0.212 +- 0.048 | -8.8 % |
| direct phi + S | 2.354 +- 0.077 | -0.051 +- 0.047 | +2.1 % |
| direct phi (S = 0) | 2.406 +- 0.071 | +0.001 +- 0.053 | -0.1 % |
| ref-free: mesh only | 2.523 +- 0.085 | +0.118 +- 0.049 | -4.9 % |
| ref-free tilt, b = 0.5 | 2.501 +- 0.077 | +0.096 +- 0.043 | -4.0 % |
| ref-free tilt, b fitted | 2.509 +- 0.079 | +0.104 +- 0.046 | -4.3 % |
| ref-free tilt + v, b = 0.5 | 2.407 +- 0.074 | +0.002 +- 0.040 | -0.1 % |
| ref-free tilt + v, b fitted | 2.424 +- 0.079 | +0.019 +- 0.042 | -0.8 % |
| ref-free (tilt+v) then direct phi + S | 2.346 +- 0.085 | -0.059 +- 0.052 | +2.4 % |
| ref-free tilt then direct phi (S = 0) | 2.400 +- 0.072 | -0.004 +- 0.053 | +0.2 % |

**E623 time5 Y view** (n = 1610)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.627 +- 0.068 | +0.000 +- 0.000 | -0.0 % |
| constants only | 2.150 +- 0.066 | +0.523 +- 0.056 | -32.1 % |
| direct phi + S | 1.377 +- 0.051 | -0.251 +- 0.048 | +15.4 % |
| direct phi (S = 0) | 1.471 +- 0.060 | -0.156 +- 0.051 | +9.6 % |
| ref-free: mesh only | 1.529 +- 0.053 | -0.099 +- 0.042 | +6.1 % |
| ref-free tilt, b = 0.5 | 1.499 +- 0.069 | -0.128 +- 0.050 | +7.9 % |
| ref-free tilt, b fitted | 1.485 +- 0.063 | -0.142 +- 0.043 | +8.7 % |
| ref-free tilt + v, b = 0.5 | 1.459 +- 0.060 | -0.168 +- 0.046 | +10.3 % |
| ref-free tilt + v, b fitted | 1.457 +- 0.069 | -0.170 +- 0.045 | +10.5 % |
| ref-free (tilt+v) then direct phi + S | 1.395 +- 0.050 | -0.233 +- 0.047 | +14.3 % |
| ref-free tilt then direct phi (S = 0) | 1.475 +- 0.060 | -0.152 +- 0.050 | +9.3 % |


## Mounting E624: E (det4), rows ['det4:624'], N = 3128

beta: {'slope': None, 'measurable': False, 'beta_used': 0.68, 'nfold': 0}


### E624, scheme time5: fitted b = 0.09 +- 0.05 (x 0.27 +- 0.07, y 0.06 +- 0.03); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [9, 0, 10.0, 3.412]; dir_phiS:1: [9, 0, 1.0, 3.414]; dir_phiS:2: [9, 0, 10.0, 3.359]; dir_phiS:3: [6, 0, 10.0, 3.421]; dir_phiS:4: [9, 0, 0.1, 3.422]; both_phiS:0: [9, 0, 10.0, 3.31]; both_phiS:1: [12, 0, 10.0, 3.297]; both_phiS:2: [9, 0, 10.0, 3.257]; both_phiS:3: [6, 0, 10.0, 3.343]; both_phiS:4: [9, 0, 10.0, 3.332]


**E624 time5 X view** (n = 3128)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 2.141 +- 0.057 | +0.000 +- 0.000 | -0.0 % |
| constants only | 2.284 +- 0.052 | +0.144 +- 0.034 | -6.7 % |
| direct phi + S | 2.120 +- 0.048 | -0.021 +- 0.038 | +1.0 % |
| direct phi (S = 0) | 2.120 +- 0.048 | -0.021 +- 0.038 | +1.0 % |
| ref-free: mesh only | 2.183 +- 0.049 | +0.042 +- 0.038 | -2.0 % |
| ref-free tilt, b = 0.5 | 2.158 +- 0.050 | +0.018 +- 0.041 | -0.8 % |
| ref-free tilt, b fitted | 2.178 +- 0.048 | +0.037 +- 0.038 | -1.7 % |
| ref-free tilt + v, b = 0.5 | 2.117 +- 0.053 | -0.024 +- 0.039 | +1.1 % |
| ref-free tilt + v, b fitted | 2.121 +- 0.052 | -0.020 +- 0.038 | +0.9 % |
| ref-free (tilt+v) then direct phi + S | 2.054 +- 0.043 | -0.087 +- 0.035 | +4.0 % |
| ref-free tilt then direct phi (S = 0) | 2.112 +- 0.049 | -0.028 +- 0.037 | +1.3 % |

**E624 time5 Y view** (n = 3128)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.436 +- 0.040 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.840 +- 0.040 | +0.404 +- 0.039 | -28.1 % |
| direct phi + S | 1.324 +- 0.042 | -0.112 +- 0.031 | +7.8 % |
| direct phi (S = 0) | 1.324 +- 0.042 | -0.112 +- 0.031 | +7.8 % |
| ref-free: mesh only | 1.391 +- 0.036 | -0.045 +- 0.026 | +3.1 % |
| ref-free tilt, b = 0.5 | 1.489 +- 0.047 | +0.053 +- 0.036 | -3.7 % |
| ref-free tilt, b fitted | 1.396 +- 0.040 | -0.041 +- 0.029 | +2.8 % |
| ref-free tilt + v, b = 0.5 | 1.447 +- 0.055 | +0.011 +- 0.041 | -0.7 % |
| ref-free tilt + v, b fitted | 1.372 +- 0.036 | -0.065 +- 0.026 | +4.5 % |
| ref-free (tilt+v) then direct phi + S | 1.278 +- 0.039 | -0.158 +- 0.029 | +11.0 % |
| ref-free tilt then direct phi (S = 0) | 1.320 +- 0.043 | -0.117 +- 0.031 | +8.1 % |


## Mounting C: C (det6), rows ['det6:long', 'det6:longer', 'det6:short'], N = 7985

beta: {'slope': -0.8463768580878037, 'err': 0.07893716149928304, 'slope_x': -0.8052648799993485, 'err_x': 0.06729803836153624, 'slope_y': -0.8851577904294817, 'err_y': 0.09778589530487802, 'beta_meas': 5.509435932325565, 'measurable': False, 'beta_used': 0.68, 'nfold': 5}


### C, scheme time5: fitted b = 0.18 +- 0.04 (x 0.09 +- 0.04, y 0.36 +- 0.04); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [12, 8, 0.1, 1.998]; dir_phiS:1: [9, 8, 0.01, 2.134]; dir_phiS:2: [12, 4, 0.1, 2.248]; dir_phiS:3: [12, 4, 0.1, 2.233]; dir_phiS:4: [12, 8, 0.1, 2.209]; both_phiS:0: [12, 8, 1.0, 1.988]; both_phiS:1: [9, 4, 0.01, 2.124]; both_phiS:2: [12, 8, 1.0, 2.235]; both_phiS:3: [12, 4, 0.01, 2.237]; both_phiS:4: [12, 8, 1.0, 2.197]


**C time5 X view** (n = 7985)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.183 +- 0.016 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.727 +- 0.017 | +0.544 +- 0.015 | -46.0 % |
| direct phi + S | 0.932 +- 0.010 | -0.251 +- 0.013 | +21.2 % |
| direct phi (S = 0) | 1.049 +- 0.014 | -0.134 +- 0.016 | +11.4 % |
| ref-free: mesh only | 1.188 +- 0.015 | +0.005 +- 0.012 | -0.4 % |
| ref-free tilt, b = 0.5 | 1.163 +- 0.014 | -0.020 +- 0.015 | +1.7 % |
| ref-free tilt, b fitted | 1.148 +- 0.014 | -0.035 +- 0.013 | +3.0 % |
| ref-free tilt + v, b = 0.5 | 1.086 +- 0.013 | -0.097 +- 0.016 | +8.2 % |
| ref-free tilt + v, b fitted | 1.051 +- 0.012 | -0.132 +- 0.012 | +11.2 % |
| ref-free (tilt+v) then direct phi + S | 0.925 +- 0.012 | -0.258 +- 0.013 | +21.8 % |
| ref-free tilt then direct phi (S = 0) | 1.053 +- 0.014 | -0.130 +- 0.016 | +11.0 % |

**C time5 Y view** (n = 7985)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.480 +- 0.020 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.850 +- 0.020 | +0.370 +- 0.019 | -25.0 % |
| direct phi + S | 1.228 +- 0.014 | -0.252 +- 0.016 | +17.0 % |
| direct phi (S = 0) | 1.341 +- 0.016 | -0.139 +- 0.015 | +9.4 % |
| ref-free: mesh only | 1.453 +- 0.017 | -0.027 +- 0.014 | +1.8 % |
| ref-free tilt, b = 0.5 | 1.384 +- 0.016 | -0.095 +- 0.016 | +6.4 % |
| ref-free tilt, b fitted | 1.413 +- 0.016 | -0.067 +- 0.015 | +4.5 % |
| ref-free tilt + v, b = 0.5 | 1.317 +- 0.017 | -0.163 +- 0.015 | +11.0 % |
| ref-free tilt + v, b fitted | 1.343 +- 0.017 | -0.136 +- 0.014 | +9.2 % |
| ref-free (tilt+v) then direct phi + S | 1.223 +- 0.017 | -0.257 +- 0.017 | +17.3 % |
| ref-free tilt then direct phi (S = 0) | 1.347 +- 0.016 | -0.132 +- 0.015 | +8.9 % |

### C, scheme loro: fitted b = 0.16 +- 0.08 (x 0.07 +- 0.08, y 0.34 +- 0.08); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [9, 4, 0.01, 2.589]; dir_phiS:1: [12, 8, 0.01, 2.045]; dir_phiS:2: [9, 8, 0.01, 2.077]; both_phiS:0: [9, 4, 0.01, 2.582]; both_phiS:1: [16, 8, 1.0, 2.045]; both_phiS:2: [12, 4, 0.1, 2.07]


**C loro X view** (n = 7985)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.183 +- 0.016 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.727 +- 0.017 | +0.544 +- 0.015 | -46.0 % |
| direct phi + S | 1.037 +- 0.013 | -0.146 +- 0.014 | +12.4 % |
| direct phi (S = 0) | 1.130 +- 0.014 | -0.053 +- 0.014 | +4.5 % |
| ref-free: mesh only | 1.219 +- 0.014 | +0.036 +- 0.010 | -3.0 % |
| ref-free tilt, b = 0.5 | 1.194 +- 0.014 | +0.011 +- 0.015 | -0.9 % |
| ref-free tilt, b fitted | 1.196 +- 0.014 | +0.013 +- 0.012 | -1.1 % |
| ref-free tilt + v, b = 0.5 | 1.129 +- 0.012 | -0.054 +- 0.015 | +4.6 % |
| ref-free tilt + v, b fitted | 1.115 +- 0.013 | -0.068 +- 0.011 | +5.7 % |
| ref-free (tilt+v) then direct phi + S | 1.031 +- 0.012 | -0.152 +- 0.013 | +12.9 % |
| ref-free tilt then direct phi (S = 0) | 1.133 +- 0.013 | -0.050 +- 0.014 | +4.2 % |

**C loro Y view** (n = 7985)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.480 +- 0.020 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.850 +- 0.020 | +0.370 +- 0.019 | -25.0 % |
| direct phi + S | 1.285 +- 0.017 | -0.194 +- 0.017 | +13.1 % |
| direct phi (S = 0) | 1.388 +- 0.018 | -0.092 +- 0.015 | +6.2 % |
| ref-free: mesh only | 1.483 +- 0.017 | +0.003 +- 0.013 | -0.2 % |
| ref-free tilt, b = 0.5 | 1.405 +- 0.017 | -0.074 +- 0.015 | +5.0 % |
| ref-free tilt, b fitted | 1.457 +- 0.018 | -0.022 +- 0.013 | +1.5 % |
| ref-free tilt + v, b = 0.5 | 1.356 +- 0.018 | -0.123 +- 0.015 | +8.3 % |
| ref-free tilt + v, b fitted | 1.400 +- 0.016 | -0.080 +- 0.013 | +5.4 % |
| ref-free (tilt+v) then direct phi + S | 1.286 +- 0.016 | -0.193 +- 0.016 | +13.1 % |
| ref-free tilt then direct phi (S = 0) | 1.388 +- 0.018 | -0.091 +- 0.015 | +6.2 % |


## Mounting D: D (det7), rows ['det7:long', 'det7:longer', 'det7:short'], N = 5137

beta: {'slope': -0.7197591239373533, 'err': 0.30462285460300337, 'slope_x': -0.6744048163016488, 'err_x': 0.31465154692288905, 'slope_y': -0.7740130204486475, 'err_y': 0.28705045170157134, 'beta_meas': 2.5683588134960518, 'measurable': False, 'beta_used': 0.68, 'nfold': 5}


### D, scheme time5: fitted b = 0.47 +- 0.02 (x 0.50 +- 0.04, y 0.46 +- 0.02); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [12, 8, 10.0, 2.279]; dir_phiS:1: [9, 8, 0.01, 2.293]; dir_phiS:2: [9, 8, 0.1, 2.275]; dir_phiS:3: [9, 8, 0.1, 2.305]; dir_phiS:4: [12, 8, 0.1, 2.264]; both_phiS:0: [12, 4, 10.0, 2.243]; both_phiS:1: [9, 4, 0.1, 2.263]; both_phiS:2: [9, 4, 1.0, 2.252]; both_phiS:3: [9, 8, 1.0, 2.285]; both_phiS:4: [16, 0, 10.0, 2.238]


**D time5 X view** (n = 5137)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.617 +- 0.033 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.629 +- 0.032 | +0.012 +- 0.021 | -0.7 % |
| direct phi + S | 1.463 +- 0.029 | -0.154 +- 0.023 | +9.5 % |
| direct phi (S = 0) | 1.476 +- 0.034 | -0.141 +- 0.022 | +8.7 % |
| ref-free: mesh only | 1.577 +- 0.031 | -0.040 +- 0.017 | +2.5 % |
| ref-free tilt, b = 0.5 | 1.520 +- 0.030 | -0.097 +- 0.020 | +6.0 % |
| ref-free tilt, b fitted | 1.519 +- 0.031 | -0.098 +- 0.020 | +6.1 % |
| ref-free tilt + v, b = 0.5 | 1.504 +- 0.033 | -0.112 +- 0.022 | +7.0 % |
| ref-free tilt + v, b fitted | 1.506 +- 0.033 | -0.111 +- 0.021 | +6.9 % |
| ref-free (tilt+v) then direct phi + S | 1.433 +- 0.030 | -0.184 +- 0.022 | +11.4 % |
| ref-free tilt then direct phi (S = 0) | 1.474 +- 0.034 | -0.142 +- 0.023 | +8.8 % |

**D time5 Y view** (n = 5137)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.028 +- 0.015 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.287 +- 0.017 | +0.259 +- 0.017 | -25.2 % |
| direct phi + S | 0.829 +- 0.013 | -0.199 +- 0.015 | +19.3 % |
| direct phi (S = 0) | 0.846 +- 0.013 | -0.182 +- 0.015 | +17.7 % |
| ref-free: mesh only | 1.038 +- 0.016 | +0.011 +- 0.013 | -1.0 % |
| ref-free tilt, b = 0.5 | 0.930 +- 0.015 | -0.097 +- 0.015 | +9.5 % |
| ref-free tilt, b fitted | 0.928 +- 0.017 | -0.099 +- 0.016 | +9.7 % |
| ref-free tilt + v, b = 0.5 | 0.900 +- 0.014 | -0.128 +- 0.014 | +12.4 % |
| ref-free tilt + v, b fitted | 0.896 +- 0.016 | -0.132 +- 0.015 | +12.8 % |
| ref-free (tilt+v) then direct phi + S | 0.820 +- 0.012 | -0.208 +- 0.014 | +20.2 % |
| ref-free tilt then direct phi (S = 0) | 0.844 +- 0.013 | -0.183 +- 0.014 | +17.8 % |

**D time5 X view (events x >= 200 mm only)** (n = 1934)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.968 +- 0.023 | +0.000 +- 0.000 | -0.0 % |
| constants only | 0.920 +- 0.023 | -0.047 +- 0.017 | +4.9 % |
| direct phi + S | 0.802 +- 0.019 | -0.166 +- 0.019 | +17.1 % |
| direct phi (S = 0) | 0.802 +- 0.022 | -0.165 +- 0.020 | +17.1 % |
| ref-free: mesh only | 0.939 +- 0.020 | -0.029 +- 0.017 | +3.0 % |
| ref-free tilt, b = 0.5 | 0.878 +- 0.020 | -0.090 +- 0.018 | +9.3 % |
| ref-free tilt, b fitted | 0.875 +- 0.021 | -0.093 +- 0.017 | +9.6 % |
| ref-free tilt + v, b = 0.5 | 0.863 +- 0.018 | -0.105 +- 0.017 | +10.8 % |
| ref-free tilt + v, b fitted | 0.860 +- 0.019 | -0.108 +- 0.017 | +11.2 % |
| ref-free (tilt+v) then direct phi + S | 0.787 +- 0.020 | -0.180 +- 0.019 | +18.6 % |
| ref-free tilt then direct phi (S = 0) | 0.797 +- 0.021 | -0.170 +- 0.020 | +17.6 % |

### D, scheme loro: fitted b = 0.47 +- 0.02 (x 0.49 +- 0.05, y 0.46 +- 0.00); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [9, 4, 1.0, 2.31]; dir_phiS:1: [9, 8, 0.1, 2.263]; dir_phiS:2: [12, 4, 10.0, 2.282]; both_phiS:0: [9, 4, 1.0, 2.275]; both_phiS:1: [9, 8, 1.0, 2.24]; both_phiS:2: [12, 4, 10.0, 2.251]


**D loro X view** (n = 5137)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.617 +- 0.033 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.629 +- 0.032 | +0.012 +- 0.021 | -0.7 % |
| direct phi + S | 1.471 +- 0.031 | -0.146 +- 0.022 | +9.0 % |
| direct phi (S = 0) | 1.472 +- 0.032 | -0.145 +- 0.021 | +8.9 % |
| ref-free: mesh only | 1.573 +- 0.035 | -0.043 +- 0.018 | +2.7 % |
| ref-free tilt, b = 0.5 | 1.522 +- 0.034 | -0.095 +- 0.021 | +5.9 % |
| ref-free tilt, b fitted | 1.522 +- 0.035 | -0.095 +- 0.022 | +5.8 % |
| ref-free tilt + v, b = 0.5 | 1.503 +- 0.032 | -0.114 +- 0.021 | +7.1 % |
| ref-free tilt + v, b fitted | 1.506 +- 0.032 | -0.111 +- 0.021 | +6.9 % |
| ref-free (tilt+v) then direct phi + S | 1.469 +- 0.033 | -0.148 +- 0.022 | +9.2 % |
| ref-free tilt then direct phi (S = 0) | 1.471 +- 0.031 | -0.145 +- 0.021 | +9.0 % |

**D loro Y view** (n = 5137)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.028 +- 0.015 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.287 +- 0.017 | +0.259 +- 0.017 | -25.2 % |
| direct phi + S | 0.850 +- 0.012 | -0.178 +- 0.014 | +17.3 % |
| direct phi (S = 0) | 0.867 +- 0.013 | -0.160 +- 0.014 | +15.6 % |
| ref-free: mesh only | 1.045 +- 0.016 | +0.017 +- 0.013 | -1.7 % |
| ref-free tilt, b = 0.5 | 0.933 +- 0.015 | -0.094 +- 0.015 | +9.2 % |
| ref-free tilt, b fitted | 0.935 +- 0.016 | -0.093 +- 0.015 | +9.1 % |
| ref-free tilt + v, b = 0.5 | 0.911 +- 0.013 | -0.117 +- 0.014 | +11.4 % |
| ref-free tilt + v, b fitted | 0.909 +- 0.014 | -0.119 +- 0.014 | +11.6 % |
| ref-free (tilt+v) then direct phi + S | 0.830 +- 0.013 | -0.198 +- 0.014 | +19.2 % |
| ref-free tilt then direct phi (S = 0) | 0.866 +- 0.014 | -0.162 +- 0.014 | +15.8 % |

**D loro X view (events x >= 200 mm only)** (n = 1934)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.968 +- 0.023 | +0.000 +- 0.000 | -0.0 % |
| constants only | 0.920 +- 0.023 | -0.047 +- 0.017 | +4.9 % |
| direct phi + S | 0.826 +- 0.023 | -0.142 +- 0.020 | +14.7 % |
| direct phi (S = 0) | 0.824 +- 0.020 | -0.143 +- 0.020 | +14.8 % |
| ref-free: mesh only | 0.938 +- 0.021 | -0.030 +- 0.017 | +3.1 % |
| ref-free tilt, b = 0.5 | 0.883 +- 0.021 | -0.084 +- 0.017 | +8.7 % |
| ref-free tilt, b fitted | 0.886 +- 0.021 | -0.081 +- 0.017 | +8.4 % |
| ref-free tilt + v, b = 0.5 | 0.864 +- 0.020 | -0.104 +- 0.017 | +10.7 % |
| ref-free tilt + v, b fitted | 0.862 +- 0.021 | -0.106 +- 0.017 | +10.9 % |
| ref-free (tilt+v) then direct phi + S | 0.818 +- 0.020 | -0.150 +- 0.019 | +15.5 % |
| ref-free tilt then direct phi (S = 0) | 0.821 +- 0.019 | -0.147 +- 0.019 | +15.2 % |


## Mounting Dx200: D (det7), x>=200 mm only, rows ['det7:long', 'det7:longer', 'det7:short'], N = 1934

beta: {'slope': None, 'measurable': False, 'beta_used': 0.68, 'nfold': 0}


### Dx200, scheme time5: fitted b = 0.31 +- 0.01 (x 0.54 +- 0.07, y 0.27 +- 0.02); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [9, 0, 1.0, 1.668]; dir_phiS:1: [9, 0, 0.01, 1.666]; dir_phiS:2: [9, 0, 0.1, 1.67]; dir_phiS:3: [9, 0, 0.01, 1.676]; dir_phiS:4: [9, 0, 0.01, 1.636]; both_phiS:0: [9, 0, 1.0, 1.679]; both_phiS:1: [9, 0, 1.0, 1.656]; both_phiS:2: [9, 0, 0.01, 1.661]; both_phiS:3: [9, 0, 0.1, 1.66]; both_phiS:4: [9, 0, 0.1, 1.632]


**Dx200 time5 X view** (n = 1934)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.968 +- 0.023 | +0.000 +- 0.000 | -0.0 % |
| constants only | 0.927 +- 0.023 | -0.040 +- 0.018 | +4.2 % |
| direct phi + S | 0.796 +- 0.018 | -0.171 +- 0.019 | +17.7 % |
| direct phi (S = 0) | 0.796 +- 0.018 | -0.171 +- 0.019 | +17.7 % |
| ref-free: mesh only | 0.933 +- 0.021 | -0.034 +- 0.018 | +3.6 % |
| ref-free tilt, b = 0.5 | 0.899 +- 0.017 | -0.069 +- 0.016 | +7.1 % |
| ref-free tilt, b fitted | 0.904 +- 0.021 | -0.064 +- 0.017 | +6.6 % |
| ref-free tilt + v, b = 0.5 | 0.878 +- 0.021 | -0.089 +- 0.017 | +9.2 % |
| ref-free tilt + v, b fitted | 0.885 +- 0.026 | -0.082 +- 0.020 | +8.5 % |
| ref-free (tilt+v) then direct phi + S | 0.781 +- 0.021 | -0.187 +- 0.021 | +19.3 % |
| ref-free tilt then direct phi (S = 0) | 0.797 +- 0.018 | -0.171 +- 0.019 | +17.7 % |

**Dx200 time5 Y view** (n = 1934)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.124 +- 0.030 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.350 +- 0.037 | +0.226 +- 0.033 | -20.2 % |
| direct phi + S | 0.869 +- 0.024 | -0.254 +- 0.028 | +22.6 % |
| direct phi (S = 0) | 0.869 +- 0.024 | -0.254 +- 0.028 | +22.6 % |
| ref-free: mesh only | 1.090 +- 0.022 | -0.033 +- 0.023 | +3.0 % |
| ref-free tilt, b = 0.5 | 1.058 +- 0.027 | -0.065 +- 0.031 | +5.8 % |
| ref-free tilt, b fitted | 1.030 +- 0.024 | -0.093 +- 0.027 | +8.3 % |
| ref-free tilt + v, b = 0.5 | 1.051 +- 0.025 | -0.073 +- 0.031 | +6.5 % |
| ref-free tilt + v, b fitted | 1.018 +- 0.026 | -0.106 +- 0.029 | +9.4 % |
| ref-free (tilt+v) then direct phi + S | 0.875 +- 0.023 | -0.249 +- 0.028 | +22.1 % |
| ref-free tilt then direct phi (S = 0) | 0.867 +- 0.024 | -0.256 +- 0.028 | +22.8 % |

### Dx200, scheme loro: fitted b = 0.32 +- 0.02 (x 0.55 +- 0.10, y 0.28 +- 0.02); selections (K, K_S, lambda, inner CV sum):
  dir_phiS:0: [6, 0, 0.01, 1.698]; dir_phiS:1: [9, 0, 0.1, 1.672]; dir_phiS:2: [9, 0, 1.0, 1.691]; both_phiS:0: [6, 0, 0.01, 1.671]; both_phiS:1: [9, 0, 1.0, 1.657]; both_phiS:2: [9, 0, 1.0, 1.691]


**Dx200 loro X view** (n = 1934)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 0.968 +- 0.023 | +0.000 +- 0.000 | -0.0 % |
| constants only | 0.927 +- 0.023 | -0.040 +- 0.018 | +4.2 % |
| direct phi + S | 0.840 +- 0.026 | -0.128 +- 0.023 | +13.2 % |
| direct phi (S = 0) | 0.840 +- 0.026 | -0.128 +- 0.023 | +13.2 % |
| ref-free: mesh only | 0.919 +- 0.022 | -0.049 +- 0.019 | +5.0 % |
| ref-free tilt, b = 0.5 | 0.898 +- 0.019 | -0.070 +- 0.017 | +7.2 % |
| ref-free tilt, b fitted | 0.900 +- 0.021 | -0.068 +- 0.017 | +7.0 % |
| ref-free tilt + v, b = 0.5 | 0.888 +- 0.020 | -0.080 +- 0.018 | +8.2 % |
| ref-free tilt + v, b fitted | 0.886 +- 0.023 | -0.082 +- 0.019 | +8.4 % |
| ref-free (tilt+v) then direct phi + S | 0.831 +- 0.023 | -0.137 +- 0.022 | +14.1 % |
| ref-free tilt then direct phi (S = 0) | 0.835 +- 0.025 | -0.133 +- 0.021 | +13.7 % |

**Dx200 loro Y view** (n = 1934)

| arm | s68 (deg) | difference to pass 1 | gain |
|---|---|---|---|
| pass 1 | 1.124 +- 0.030 | +0.000 +- 0.000 | -0.0 % |
| constants only | 1.350 +- 0.037 | +0.226 +- 0.033 | -20.2 % |
| direct phi + S | 0.904 +- 0.022 | -0.219 +- 0.025 | +19.5 % |
| direct phi (S = 0) | 0.904 +- 0.022 | -0.219 +- 0.025 | +19.5 % |
| ref-free: mesh only | 1.098 +- 0.023 | -0.026 +- 0.023 | +2.3 % |
| ref-free tilt, b = 0.5 | 1.053 +- 0.028 | -0.070 +- 0.032 | +6.2 % |
| ref-free tilt, b fitted | 1.034 +- 0.025 | -0.089 +- 0.027 | +7.9 % |
| ref-free tilt + v, b = 0.5 | 1.046 +- 0.026 | -0.078 +- 0.031 | +6.9 % |
| ref-free tilt + v, b fitted | 1.021 +- 0.026 | -0.102 +- 0.028 | +9.1 % |
| ref-free (tilt+v) then direct phi + S | 0.886 +- 0.021 | -0.238 +- 0.026 | +21.2 % |
| ref-free tilt then direct phi (S = 0) | 0.912 +- 0.024 | -0.212 +- 0.025 | +18.8 % |
