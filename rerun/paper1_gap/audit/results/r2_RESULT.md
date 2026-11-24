# Audit R2: does the n_TOF drift-time map predict the n_TOF angle bias? (2026-11-24)

Scripts (new, `rerun/paper1_gap/audit/`): `r2_lib.py` (selection, P-spline field), `r2_01_extract.py` (per-track table from the pass-1 analysis view), `r2_02_explore.py`,
`r2_03_maps.py` (fields and stability), `r2_04_compare.py` (cell-by-cell against the in-situ map), `r2_05_direct.py` (held-out sigma on the A-C lines),
`r2_06_cells.py` (fine-part test, gap-map summary, figures 1-2), `r2_07_through.py` (through-going muons inside beam runs), `r2_08_bd.py` (D on the B-D lines), `r2_09_tables.py` (tables, figure 3).
Products here: `tr_cosmic.parquet`, `tr_late.parquet`, `tr_through.parquet`, `fields.pkl`, `stability.json`, `compare.json`, `cells_fine.csv`, `gapmap_summary.json`,
`direct_{late,cosmic}.csv`, `bd_direct_{late,cosmic}.csv`, `tables.md` (all sigma tables), figures `fig1_gapmaps.png`, `fig2_tilt_maps.png`, `fig3_sigma.png`.
Nothing was written under `ntof_pass1/` or into the repo's stage 3.

## Verdict

1. **The timing columns exist locally and the n_TOF drift-time field is clean.** `x_q_u90`, `y_q_u90`, `x_q_uend`, `x_t0`, `x_w`, charge are in every stage-3 table (`stage3_vt/`, 543 sub-runs). The field of ln T of each chamber
   is reproducible between run parities (r = 0.92-0.998), early against late campaign (0.95-0.999), X-view against Y-view quantile (0.90-0.999), in three time-since-flash bins of the late beam (0.99+),
   and with the amplitude walk and the angle dependence removed (the maps keep their shape in |tan| bands, r = 0.72-0.998). Statistical noise is 0.04-0.10 mm in gap units. The structure is 0.5-1.7 mm rms
   (A 0.5-0.7, B 0.8-0.9, C 1.3-1.4, D 1.6-1.7; peak to peak 2.5-6.4 mm), larger than on the bench for B, C and D.
2. **The prediction has the right sign and about the right size, but it is a partial predictor, and it is not the same for the two track samples.** With the late-beam field (independent of the cosmic lines) the held-out single-view sigma of the A-C lines
   falls, against the same model with one global tilt and scale, by (timing alone, bench dish at full size k = 1, b nominal 0.5):
   A x 7.5 %, A y 2.4 %, C x 2.2 %, C y 2.8 %, and by 7.5 / 5.8 / 3.4 / 6.2 % with b and k fitted on the other run parity. The in-situ map gives 14.8 / 11.4 / 16.0 / 15.4 %.
   So the timing prediction recovers about half of the in-situ gain for A (A x 50-66 %, A y 21-50 %) and 15-40 % for C. With the cosmic-track field the numbers are lower (A x 5 %, A y -5 to +3 %, C x 1-2 %, C y 1-4 %).
   Fitted b (expected 0.5; b' below is b / 0.5): between 0.3 and 1.9 depending on view, k and sample (table below); the event-level fit with k and b free gives b = 0.73 (A x), 0.61 (A y), 0.59 (C x), 0.80 (C y) with the late field,
   0.58 / 0.33 / 0.34 / 0.90 with the cosmic field. The fit errors from the cells (0.2-0.5 in b') are the realistic ones, the event-level errors (0.01-0.1) are far too small because k and b are degenerate with the smooth part and the cells are correlated.
3. **The two track samples disagree about A.** The gradient field of A y from beam-off muons and from late beam tracks has correlation 0.04 (A x 0.74, C x 0.66, C y 0.95), although each is internally stable and the noise ceiling is 0.98-0.99.
   I could not find the cause. It is not the beam timing (late-beam field is the same in 20-30, 30-50 and 50+ ms, r = 0.99+), the cuts (relaxed cosmic cuts give the same field, r = 0.98) or the angle (tan bands). The most likely reason is a population
   difference (beam electrons from the capsule against through-going muons: stopping and scattering electrons shorten the charge-time quantile and the beam tracks come from a point). A through-going-muon sample inside the beam runs
   (3,015 A tracks) is too small to decide (the slope of its ln T on the late field is 0.78, on the cosmic field 1.33). The A result therefore depends on which field is used; for C the two fields agree on the gradient of y (0.95) and give the same answer.
4. **Chamber D gets a correction it does not have, in one view.** On the 2,436 / 2,420 clean B-D lines (held-out by event parity, about 1,200 test lines per view) the timing prediction with the bench dish of det7 (k = 1) and nominal b lowers the sigma of the D z view
   (D x strips) from 35.9 +- 1.2 to 31.4 +- 1.3 (late field) / 31.1 +- 1.3 (cosmic field), i.e. by 12-13 %, 68-73 % of the 100 mm in-situ B-D map (18.1 %), with fitted b' = 0.87-1.01 (b = 0.44-0.50). The D y view gains 0.1-2.8 % only (b' = 0.14-0.31).
   B has no angle in pass 1 (positions only), so there is nothing to correct for B; its gap map is in figure 1 and the table below.
5. **Limits.** The gain is significant at about 3 sigma or more for A x, A y (late field, fitted), C y (fitted), D z and not for C x or D y; none of the models comes close to the in-situ map for C; for A the timing prediction becomes as good as or better than the in-situ map only
   together with the bench fine map. Beta at n_TOF is not known to me (no Magboltz grid for the n_TOF gas and fields was found), so the amplitude is quoted as the fitted coefficient c in tan per unit gradient of ln T and as b' = c / 8.39 mm (nominal beta = 0.68, g0 = 28.2 mm). If b were exactly 0.5 the
   fitted b' = 1.2-1.6 of A x, C x, C y would mean beta = 1.68 / b' - 1 = 0.05-0.4.

## 1. Data and selection (step 1)

* Source: `$WORK_DATA/ntof_pass1/stage3_vt/tracks_run_*_*.parquet` through `sept26_prelim_analysis.pass1_tracks.read` (analysis view; read only). Columns used: `x_p0, y_p0, x_tan_theta, y_tan_theta, x_q_u90, y_q_u90, x_q_u50, x_q_uend, x_q_sum, x_n_strips, chi2dof_*, drift_railed, n_cand_*, x_quality_ok, censored_window, t_since_flash_ns`.
  T = mean of `x_q_u90` and `y_q_u90` (ns). No data had to be fetched from EOS.
* Two samples: **beam-off** (registry class cosmic, `cosmic_filter()`; A 47,388 / B 43,319 / C 14,160 / D 16,296 tracks after the cuts) and **late beam** (`t_since_flash >= 20 ms`; A 833,294 / B 108,877 / C 359,075 / D 76,044; fits use at most 250,000 random tracks).
* Cuts: `gated` (true |tan| < 0.5), not `censored_window`, `x_quality_ok` and `y_quality_ok`, chi2/dof < 3 in both views, >= 6 strips per view, one candidate per view, not `drift_railed`, finite q_u90, 50 < q_sum < 1e6. Runs 79 and 81 are dropped for A (connector 8 dead). Keeping the railed tracks changes the field by r = 0.996 (A) and 0.997 (C).
  (The noise-floor change of 23 July lies before run_79: the whole pass-1 sample is on the noisy side, so nothing is pooled across it.)
* ln T model: P-spline (tensor cubic, 10 x 10 knots over strip coordinates -2.05..2.05 in 100 mm, second-difference penalty 0.1, robust 3.5 sigma clip) of the mid-gap position (U, W) = ((x_p0 + 14 tan_x - 200)/100, ...), fitted jointly with
  ln q_sum (walk), tan_x, tan_y, tan_x^2, tan_y^2, tan_x tan_y (unpenalised). Walk slope 0.00-0.06; odd tan terms ~0; quadratic terms 0.26 / 0.24 (A, beam-off) and 0.28 / 0.40 (A, late): ln T rises by 0.11-0.22 per unit |tan| (the bench had none), so this correction is needed.
  Per-track scatter of ln T: MAD 0.10-0.15.

## 2. Gap maps (step 2)  (figure `fig1_gapmaps.png`, local frame x_local = -100 U - 0.71, y_local = 100 W + 0.71 mm)

delta g = g0/(1+beta) delta ln T with g0 = 28.2 mm, beta = 0.68 (nominal; scale by 1.68/(1+beta) for another beta).


| chamber sample | n tracks | sd of delta g (mm) | peak-to-peak (mm) | plane tilt x, y (mm per 100 mm) | curvature xx, yy, xy (mm at 100 mm) | sd after quadratic (mm) | noise (mm) |
|---|---|---|---|---|---|---|---|
| A late | 833,294 | 0.67 | 3.5 | 0.19, 0.01 | 0.06, -0.54, -0.04 | 0.35 | 0.05 |
| A cosmic | 47,388 | 0.53 | 2.5 | 0.19, -0.07 | 0.07, -0.35, -0.06 | 0.34 | 0.04 |
| B late | 108,877 | 0.82 | 3.3 | 0.04, 0.53 | 0.11, -0.07, -0.06 | 0.58 | 0.07 |
| B cosmic | 43,319 | 0.90 | 5.3 | 0.07, 0.43 | 0.12, -0.25, -0.07 | 0.73 | 0.10 |
| C late | 359,075 | 1.41 | 5.4 | 0.58, -0.73 | 0.45, -0.25, -0.08 | 0.89 | 0.04 |
| C cosmic | 14,160 | 1.26 | 5.0 | 0.41, -0.70 | 0.40, -0.25, -0.19 | 0.80 | 0.07 |
| D late | 76,044 | 1.56 | 5.9 | 1.03, -0.07 | 0.68, -0.55, 0.22 | 0.71 | 0.05 |
| D cosmic | 16,296 | 1.66 | 6.4 | 0.98, -0.16 | 0.79, -0.67, 0.14 | 0.81 | 0.09 |

Features (figure 1): A has two bumps (about +0.5 mm at (-130, -20) and (+80, +40) mm) on a gap that is 1 mm smaller towards the top and bottom edges; the late and beam-off maps agree on the bumps (r = 0.88 over the field). B has a gap above the mean (+1 mm and more) over most of the plate, except a 100 mm hole near (0, -75) mm, in both samples. C has a larger gap in the lower half and along the left edge and a smaller gap (1 mm or more) in the upper right, with a diagonal ridge between them; the plane tilt is 0.6 / -0.7 mm per 100 mm.
D has a larger gap on the left, a smaller gap on the right and at the bottom centre, a plane tilt of 1.0 mm per 100 mm along x and a strong curvature. Both samples show the same features for each chamber (cosmic against late: r = 0.88 A, 0.91 B, 0.98 C; see `stability.json`).

Stability of the fields (correlation inside |U|, |W| < 1.8; `stability.json`):

| chamber sample | run parity halves | early / late campaign | no tan terms | K = 8, 12 | lambda 0.03, 0.3 | X-view / Y-view T | beam-off against late |
|---|---|---|---|---|---|---|---|
| A late | 0.996 | 0.999 | 0.91 | 0.88, 0.98 | 0.98, 0.91 | 0.98, 0.995 (each against both) | 0.88 |
| A beam-off | 0.97 | 0.99 | 1.00 | 0.96, 0.98 | 0.98, 0.98 | 0.98, 0.99 | |
| B late | 0.92 | 0.95 | 0.91 | 0.97, 0.99 | 0.98, 0.98 | 0.90, 0.94 | 0.91 |
| C late | 0.998 | 0.998 | 1.00 | 0.96, 0.99 | 0.98, 0.97 | 1.00, 1.00 | 0.98 |
| C beam-off | n/a (few tracks) | 0.99 | 1.00 | 0.99, 0.99 | 0.99, 0.99 | 1.00, 1.00 | |

|tan| bands (late A, against the full map): |tan| < 0.15 0.81, 0.15-0.3 0.93, > 0.3 0.99; C: 0.97, 0.99, 1.00. The features do not change sign in any band.

## 3. Predicted tilt against the in-situ tilt map, cell by cell (step 3)  (figure `fig2_tilt_maps.png`)

Mapping (agent G's, with the same signs): the in-situ map `angle_map_r2.json` is a(x_local, y_local) = global slope of the chamber's unmapped v(t) track minus the joined A-C slope, cells of 50 mm; a_ntof = SGN * a_strip with SGN = +1 (x view, A and C), -1 (A y), +1 (C y);
cell centre -> strip coordinates U = (-x_local - 0.71)/100, W = (y_local - 0.71)/100. The fit tilt is tan_fit - tan_true = tau, tau = grad h + (1/2) grad g (strip frame, gradient with respect to the strip coordinate). Pass 1 already subtracted the bench block, so the total tilt is
T_n = SGN a_poly + a_in (a_poly of the bench bundle `bench_bundles/A, C`, the block pass 1 applied). Model: T_n = c0 + k SGN (dh_p/d strip)_bench + b' SGN (0.5 g0/(1+beta)) d ln T/d strip; k = 1 the bench dish is fully present, k = 0 absent. 64 cells (all have >= 80 lines), gradient averaged over 5 x 5 points per cell, weights = lines per cell.
Errors: bootstrap over cells and over 20 bootstrap fields (they ignore the correlation between neighbouring cells, so they are lower limits).

| chamber view, field | r(a_in, G) all 64 cells | k (free) | b' (free) | k = 0: b' | k = 1: b' | r of a_in against (kH + G - block), k = 1, b nominal |
|---|---|---|---|---|---|---|

| A x, cosmic | -0.12 ± 0.17 | -0.63 ± 0.24 | 0.72 ± 0.47 | 0.46 ± 0.55 | 0.04 ± 0.72 | -0.24 ± 0.15 |
| A x, late | 0.27 ± 0.16 | -0.51 ± 0.25 | 1.22 ± 0.39 | 1.25 ± 0.39 | 1.30 ± 0.51 | 0.00 ± 0.17 |
| A y, cosmic | -0.08 ± 0.16 | 0.14 ± 0.07 | -0.04 ± 0.26 | -0.05 ± 0.24 | 0.03 ± 0.59 | -0.56 ± 0.08 |
| A y, late | 0.68 ± 0.06 | 0.43 ± 0.15 | 0.96 ± 0.39 | -0.06 ± 0.19 | 2.27 ± 0.20 | -0.28 ± 0.13 |
| C x, cosmic | 0.00 ± 0.09 | 0.91 ± 0.12 | 0.32 ± 0.18 | 0.80 ± 0.38 | 0.27 ± 0.15 | 0.16 ± 0.11 |
| C x, late | 0.26 ± 0.15 | 0.75 ± 0.12 | 0.84 ± 0.25 | 1.74 ± 0.27 | 0.56 ± 0.21 | 0.36 ± 0.15 |
| C y, cosmic | 0.19 ± 0.13 | 0.98 ± 0.09 | 1.06 ± 0.22 | 0.37 ± 0.41 | 1.07 ± 0.22 | 0.34 ± 0.14 |
| C y, late | 0.24 ± 0.13 | 1.01 ± 0.09 | 0.96 ± 0.17 | 0.31 ± 0.27 | 0.95 ± 0.15 | 0.49 ± 0.09 |

(b' = 1 is the nominal prediction b = 0.5 with beta = 0.68. r_ain_G is the correlation of the full in-situ residual map with the nominal timing tilt alone; the cells are dominated by the edge cells of the in-situ map (up to 0.04 in tan), which the smooth timing field cannot reach.)

**Fine part only** (plane removed from the in-situ map and from the predicted tilt, so the dish and any k drop out; `cells_fine.csv`): correlation r and slope b' of the in-situ fine map on the nominal timing tilt.

| chamber view | late field r, b' | beam-off field r, b' | late field, run-parity halves b' | late field, early/late campaign b' |
|---|---|---|---|---|
| A x | 0.51 ± 0.11, 1.52 ± 0.33 | 0.31 ± 0.12, 1.02 ± 0.45 | 1.60, 1.47 | 1.64, 1.62 |
| A y | 0.40 ± 0.12, 1.09 ± 0.35 | -0.01 ± 0.13, -0.02 ± 0.23 | 1.12, 1.07 | 1.16, 1.10 |
| C x | 0.31 ± 0.10, 0.67 ± 0.23 | 0.10 ± 0.09, 0.18 ± 0.17 | 0.66, 0.60 | 0.69, 0.60 |
| C y | 0.23 ± 0.12, 0.33 ± 0.17 | 0.20 ± 0.14, 0.35 ± 0.24 | 0.29, 0.38 | 0.30, 0.34 |

Reading. With the late-beam field the fine in-situ map correlates with the timing prediction at r = 0.51 +- 0.11 (A x), 0.40 +- 0.12 (A y), 0.31 +- 0.10 (C x) and 0.24 +- 0.13 (C y), and the slope b' is 1.5 +- 0.3, 1.1 +- 0.35, 0.67 +- 0.23, 0.33 +- 0.17, i.e. b = 0.76, 0.54, 0.34, 0.16 against the expected 0.5.
A x and A y are consistent with b = 0.5 (1.5 and 0.2 sigma from it), C x and C y fall short (1.4 and 3.9 sigma below b' = 1, a statement that depends on the nominal beta). With the beam-off field the same numbers are 0.31 / 1.0, -0.01 / -0.02, 0.10 / 0.18, 0.20 / 0.35: only A x is positive and none is significant.
The correlations are far below the noise ceilings of the in-situ map (0.97 A x, 0.83 A y, 0.97 C x, 0.90 C y between run-parity halves, agent G), so the timing tilt explains at most about a quarter (r^2 = 0.06-0.26) of the fine structure of the in-situ map.

## 4. Held-out sigma on the A-C cosmic lines (step 4)

Same protocol as agent G (`g_direct.py`): 31,387 A-C lines, train on one run parity, test on the other and average, sigma = MAD of (corrected slope - joined slope) for |j| < 0.45 (the "all" column; the clean-line sigma behaves the same, see `direct_*.csv`), bootstrap errors of the sigma.
All models carry one tilt and one scale (the "global" column). Gains are against global and their errors are about 2 percentage points (quadrature of the two sigma errors, conservative: the models are paired). "tim" = the timing prediction (late-beam field: independent of the cosmic lines),
nominal = b' 1, fit = b' (and k) fitted on the training parity at event level, "+bench" = plus G's bench fine map with one free amplitude. k = 0 means no dish at n_TOF, k = 1 the full bench dish. Figure `fig3_sigma.png`.


### A-C lines, field from late tracks (sigma_all x 1000, MAD of corrected - joined slope, |j|<0.45, parity-averaged held-out; error = bootstrap)

| chamber view | global | in-situ | tim nom k=0 | tim nom k=1 | tim fit k=0 | tim fit k=1 | tim fit k,b | +bench nom k=0 | +bench nom k=1 | +bench fit k,b |
|---|---|---|---|---|---|---|---|---|---|---|
| A x | 28.7 ± 0.4 | 24.5 ± 0.3 (+14.8 %) | 26.9 ± 0.4 (+6.5 %) | 26.6 ± 0.5 (+7.5 %) | 26.5 ± 0.4 (+7.7 %) | 25.9 ± 0.4 (+9.8 %) | 26.6 ± 0.4 (+7.5 %) | 23.7 ± 0.3 (+17.6 %) | 24.5 ± 0.4 (+14.6 %) | 24.0 ± 0.4 (+16.5 %) |
| A y | 26.0 ± 0.4 | 23.1 ± 0.4 (+11.4 %) | 26.3 ± 0.5 (-1.0 %) | 25.4 ± 0.4 (+2.4 %) | 25.6 ± 0.4 (+1.8 %) | 24.8 ± 0.4 (+4.9 %) | 24.5 ± 0.4 (+5.8 %) | 25.6 ± 0.4 (+1.9 %) | 24.7 ± 0.4 (+5.0 %) | 23.8 ± 0.4 (+8.5 %) |
| C x | 32.4 ± 0.5 | 27.2 ± 0.4 (+16.0 %) | 33.9 ± 0.5 (-4.6 %) | 31.7 ± 0.4 (+2.2 %) | 32.5 ± 0.5 (-0.4 %) | 31.5 ± 0.5 (+2.7 %) | 31.3 ± 0.4 (+3.4 %) | 34.4 ± 0.5 (-6.3 %) | 31.6 ± 0.4 (+2.6 %) | 31.3 ± 0.5 (+3.3 %) |
| C y | 37.1 ± 0.5 | 31.4 ± 0.5 (+15.4 %) | 43.1 ± 0.6 (-16.1 %) | 36.1 ± 0.5 (+2.8 %) | 42.8 ± 0.6 (-15.4 %) | 35.3 ± 0.5 (+4.8 %) | 34.8 ± 0.5 (+6.2 %) | 43.0 ± 0.6 (-15.9 %) | 36.0 ± 0.5 (+3.0 %) | 34.7 ± 0.5 (+6.4 %) |

Share of the in-situ gain recovered (gain_model / gain_insitu, %):

| chamber view | tim nom k=0 | tim nom k=1 | tim fit k=1 | tim fit k,b | +bench nom k=0 | +bench fit k,b |
|---|---|---|---|---|---|---|
| A x | 44 | 51 | 66 | 50 | 118 | 111 |
| A y | -9 | 21 | 43 | 50 | 16 | 74 |
| C x | -29 | 14 | 17 | 21 | -40 | 20 |
| C y | -104 | 18 | 31 | 40 | -103 | 41 |

Event-level fitted coefficients, late field (b' = fitted / nominal, nominal = 0.5 G0/(1+0.68); so b = 0.5 b'; k = bench dish amplitude); mean of the two parities, difference in brackets:

| chamber view | model | k | b' | b = 0.5 b' |
|---|---|---|---|---|
| A x | tim_fit_k0 | 0.00 (0.00) | 1.47 (0.04) | 0.73 |
| A x | tim_fit_k1 | 1.00 (0.00) | 1.50 (0.07) | 0.75 |
| A x | tim_fit_kb | -0.02 (0.18) | 1.47 (0.04) | 0.73 |
| A y | tim_fit_k0 | 0.00 (0.00) | 0.29 (0.07) | 0.14 |
| A y | tim_fit_k1 | 1.00 (0.00) | 1.93 (0.07) | 0.97 |
| A y | tim_fit_kb | 0.56 (0.06) | 1.22 (0.18) | 0.61 |
| C x | tim_fit_k0 | 0.00 (0.00) | 1.92 (0.01) | 0.96 |
| C x | tim_fit_k1 | 1.00 (0.00) | 0.93 (0.05) | 0.47 |
| C x | tim_fit_kb | 0.73 (0.04) | 1.19 (0.07) | 0.59 |
| C y | tim_fit_k0 | 0.00 (0.00) | 1.18 (0.08) | 0.59 |
| C y | tim_fit_k1 | 1.00 (0.00) | 1.54 (0.10) | 0.77 |
| C y | tim_fit_kb | 1.19 (0.07) | 1.61 (0.09) | 0.80 |

### A-C lines, field from cosmic tracks (sigma_all x 1000, MAD of corrected - joined slope, |j|<0.45, parity-averaged held-out; error = bootstrap)

| chamber view | global | in-situ | tim nom k=0 | tim nom k=1 | tim fit k=0 | tim fit k=1 | tim fit k,b | +bench nom k=0 | +bench nom k=1 | +bench fit k,b |
|---|---|---|---|---|---|---|---|---|---|---|
| A x | 28.7 ± 0.4 | 24.5 ± 0.3 (+14.8 %) | 27.4 ± 0.4 (+4.5 %) | 27.3 ± 0.4 (+5.1 %) | 27.3 ± 0.4 (+5.1 %) | 27.1 ± 0.4 (+5.6 %) | 27.6 ± 0.4 (+4.1 %) | 23.8 ± 0.4 (+17.3 %) | 25.0 ± 0.4 (+13.1 %) | 24.3 ± 0.4 (+15.5 %) |
| A y | 26.0 ± 0.4 | 23.1 ± 0.4 (+11.4 %) | 25.9 ± 0.4 (+0.5 %) | 27.3 ± 0.4 (-4.9 %) | 25.5 ± 0.4 (+2.0 %) | 27.4 ± 0.4 (-5.2 %) | 25.2 ± 0.4 (+3.2 %) | 25.1 ± 0.3 (+3.6 %) | 26.8 ± 0.4 (-2.8 %) | 24.2 ± 0.3 (+7.0 %) |
| C x | 32.4 ± 0.5 | 27.2 ± 0.4 (+16.0 %) | 34.4 ± 0.5 (-6.3 %) | 32.0 ± 0.5 (+1.1 %) | 33.8 ± 0.5 (-4.4 %) | 31.9 ± 0.5 (+1.4 %) | 31.8 ± 0.4 (+1.7 %) | 35.3 ± 0.5 (-9.0 %) | 32.2 ± 0.5 (+0.6 %) | 31.9 ± 0.5 (+1.6 %) |
| C y | 37.1 ± 0.5 | 31.4 ± 0.5 (+15.4 %) | 43.3 ± 0.5 (-16.6 %) | 36.8 ± 0.5 (+0.9 %) | 43.0 ± 0.7 (-15.9 %) | 35.8 ± 0.5 (+3.6 %) | 35.5 ± 0.5 (+4.4 %) | 43.2 ± 0.6 (-16.5 %) | 36.7 ± 0.5 (+1.1 %) | 35.1 ± 0.5 (+5.3 %) |

Share of the in-situ gain recovered (gain_model / gain_insitu, %):

| chamber view | tim nom k=0 | tim nom k=1 | tim fit k=1 | tim fit k,b | +bench nom k=0 | +bench fit k,b |
|---|---|---|---|---|---|---|
| A x | 30 | 34 | 38 | 28 | 117 | 105 |
| A y | 4 | -43 | -46 | 28 | 32 | 61 |
| C x | -39 | 7 | 9 | 11 | -56 | 10 |
| C y | -108 | 6 | 23 | 28 | -107 | 35 |

Event-level fitted coefficients, cosmic field (b' = fitted / nominal, nominal = 0.5 G0/(1+0.68); so b = 0.5 b'; k = bench dish amplitude); mean of the two parities, difference in brackets:

| chamber view | model | k | b' | b = 0.5 b' |
|---|---|---|---|---|
| A x | tim_fit_k0 | 0.00 (0.00) | 1.17 (0.16) | 0.59 |
| A x | tim_fit_k1 | 1.00 (0.00) | 1.09 (0.25) | 0.55 |
| A x | tim_fit_kb | -0.13 (0.19) | 1.17 (0.18) | 0.58 |
| A y | tim_fit_k0 | 0.00 (0.00) | 0.58 (0.04) | 0.29 |
| A y | tim_fit_k1 | 1.00 (0.00) | 0.94 (0.04) | 0.47 |
| A y | tim_fit_kb | 0.23 (0.01) | 0.65 (0.06) | 0.33 |
| C x | tim_fit_k0 | 0.00 (0.00) | 1.46 (0.02) | 0.73 |
| C x | tim_fit_k1 | 1.00 (0.00) | 0.62 (0.04) | 0.31 |
| C x | tim_fit_kb | 0.90 (0.04) | 0.67 (0.03) | 0.34 |
| C y | tim_fit_k0 | 0.00 (0.00) | 1.39 (0.14) | 0.69 |
| C y | tim_fit_k1 | 1.00 (0.00) | 1.74 (0.15) | 0.87 |
| C y | tim_fit_kb | 1.15 (0.06) | 1.80 (0.14) | 0.90 |

### Chamber D on B-D lines, field from late tracks (sigma x 1000, event-parity held-out, ~1,200 test lines per view)

| D view | global | in-situ 100 mm map | tim nom k=0 | tim nom k=1 | tim fit k=1 | tim fit k,b |
|---|---|---|---|---|---|---|
| z | 35.9 ± 1.2 | 29.4 ± 1.1 (+18.1 %) | 34.0 ± 1.4 (+5.3 %) | 31.4 ± 1.3 (+12.3 %) | 31.5 ± 1.3 (+12.2 %) | 31.5 ± 1.2 (+12.2 %) |
| y | 34.7 ± 1.3 | 28.6 ± 1.1 (+17.6 %) | 40.1 ± 1.5 (-15.3 %) | 34.7 ± 1.2 (+0.1 %) | 34.1 ± 1.1 (+1.8 %) | 34.2 ± 1.1 (+1.6 %) |

D fitted coefficients (mean of parities): z: k_fit_kb 1.09, b' 0.97 (k=1 fixed: b' 1.01); y: k_fit_kb 0.91, b' 0.14 (k=1 fixed: b' 0.19)

### Chamber D on B-D lines, field from cosmic tracks (sigma x 1000, event-parity held-out, ~1,200 test lines per view)

| D view | global | in-situ 100 mm map | tim nom k=0 | tim nom k=1 | tim fit k=1 | tim fit k,b |
|---|---|---|---|---|---|---|
| z | 35.9 ± 1.2 | 29.4 ± 1.1 (+18.1 %) | 33.4 ± 1.3 (+7.0 %) | 31.1 ± 1.3 (+13.2 %) | 31.3 ± 1.3 (+12.6 %) | 31.1 ± 1.3 (+13.2 %) |
| y | 34.7 ± 1.3 | 28.6 ± 1.1 (+17.6 %) | 41.2 ± 1.7 (-18.5 %) | 34.7 ± 1.2 (+0.1 %) | 33.8 ± 1.1 (+2.8 %) | 34.0 ± 1.1 (+2.1 %) |

D fitted coefficients (mean of parities): z: k_fit_kb 1.05, b' 0.87 (k=1 fixed: b' 0.89); y: k_fit_kb 0.99, b' 0.29 (k=1 fixed: b' 0.31)

## 5. B and D (step 5)

* **B** has no pass-1 angle, so no line residual and no correction exists. Its gap map is the largest and flattest of the four (figure 1); a map for B's positions is not what this test addresses.
* **D**: truth = B position and D position (`bd_lines.parquet`, 2,793 lines; 2,713 matched to D strip positions). Slope in the line frame: z (D x strips) = -0.93 x_tan_theta, y = -0.94 y_tan_theta (checked on run_145 and run_149 tables), so SGN_D = -1 for both views. The model, the dish h and the block use det7's bench bundle.
  Result in the tables above: the z view recovers 12-13 % against 18 % for the 100 mm B-D map, the y view nothing (0-3 %). In the y view the dish term (k = 1 against k = 0: sigma 34.7 against 40.1) matters, the gap term adds 0-3 %.
* D is contaminated (31 % of late tracks miss by more than 60 mm after the censored cut, README `rerun/ntof/improve` finding 15) and its T field uses only 76,044 (late) / 16,296 (beam-off) strict tracks. The test lines are only 1,200 per view (sigma error 3.5 %), and the parity is by event, not by run.

## 6. What the result does not rule out, and what I could not do

* **Absolute amplitude and beta.** No Magboltz table for the n_TOF gas/field was found in `rerun/paper1_gas/` (its fits are for the bench gas), so beta = 0.68 and g0 = 28.2 mm are the bench values. The fitted amplitude is therefore a statement about the product b g0/(1+beta); b' = 1 means the nominal value.
* **k and b are partly degenerate.** The dish k (bench mesh slope kept at n_TOF) and the gap term both contain smooth quadratic structure. The fine-part test removes it but has fewer degrees of freedom. The sign of k for A x (about zero or negative) agrees with agent G (the dish is mostly gone for A on the vertical board); C wants k = 0.75-1.2.
* **Sample dependence for A** (verdict item 3). The beam-off and late-beam samples give different A gradient fields, so the answer for A is not fixed by this test alone. Chamber C and D do not show this.
* **Particle population.** The late-beam tracks are electrons from the capsule, the beam-off ones are muons. A through-going-muon sample inside the beam runs is too small (3,015 A and 1,518 C tracks) to decide which field is right for muons.
* **Only q_u90 was used** (q_u50 gives the same maps, r = 0.96-0.99). The chain's own bias of T with |tan| (0.11-0.22 per unit |tan|) is removed with quadratic terms; the beam late sample has angle and position correlated, so a non-quadratic angle dependence of T could leak into the position map. The |tan| band maps do not show it for |tan| > 0.15.
* **The gain tests are partly circular for the in-situ map**: it is fitted on the same lines (held-out by run parity, and errors 2 points); the timing field is fitted on tracks that are independent of the lines (late) or that contain them (beam-off, T is not their angle).
* No new data, run or hardware check is implied; every statement uses the existing pass-1 tables.
