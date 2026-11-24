# R4: does the shape of chamber A's drift gap depend on the drift voltage, or on the mounting? (audit tag r4, 2026-11-24)

Scripts (`rerun/paper1_gap/audit/`): `r4_01_fetch.py`, `r4_01b_fetch_other.py` (EOS copies, adler32 checked), `r4_lib.py` (selection, covariates, quadratic fit),
`r4_02_survey.py`, `r4_03_fits.py` (per sub-run fit), `r4_04_alt.py` (q_u50, per-view q_u90, window-end cut), `r4_05_inject.py` (window-cut emulation),
`r4_06_check_template.py` (selection check, golden template), `r4_07_analysis.py` (all per-unit numbers and time blocks), `r4_08_summary.py` (beta(V), fits against V^2 and V, comparisons),
`r4_09_figs.py`, `r4_10_tables.py`, `r4_11_covcheck.py`.
Outputs here: `analysis.json`, `summary.json`, `fits_main.json`, `fits_alt.json`, `inject.json`, `selcheck_template.json`, `tables.md` (all tables, including the 19 mesh-scan sub-runs), `eos/` (event tables and bundles).
Figures: `fig1_coef_vs_drift.png` (coefficients against drift voltage, with 6-25 and golden marked), `fig2_maps_quadratic.png`, `fig2b_maps_spline.png` (maps side by side), `fig3_mesh_scan.png`, `fig4_time_blocks.png`.

## Verdict

**Mounting, mainly. The drift voltage changes the map only below about 700 V, not as V^2, and not toward the 6-25 shape. Whether that low-voltage change is the foil or the measurement is undecided.**

1. **The 6-25 shape is not reproduced by any drift voltage in the golden mounting.** The 6-27 sub-run at 500 V (same mounting as golden, N = 534) has gww = -0.022 +- 0.013, guw = +0.002 +- 0.006, guu = -0.023 +- 0.015 (ln q_u90 per (100 mm)^2). The 6-25 dataset at the same nominal 500 V has gww = +0.049 +- 0.005, guw = -0.041 +- 0.002, guu = +0.052 +- 0.004. The differences are 5.0 sigma (gww), 6.3 sigma (guw) and 4.9 sigma (guu). The twist (guw) of 6-25 is absent at every voltage of 6-27 (|guw| <= 0.012, 0 within errors at 500 and 300 V).
2. **A third mounting gives a third shape at the golden voltage.** The 6-22 overnight run of chamber A (drift 1000 V, N = 12,103) has gww = 0.000 +- 0.005 (-0.003 to +0.019 depending on the covariates), guu = +0.008 +- 0.004, guw = -0.003 +- 0.002, and almost no golden ridge (amplitude 0.12 +- 0.06). Against golden run 1 at the same nominal 1000 V the difference in gww is +0.103 +- 0.006 (16 sigma). The voltage is the same, the shape is not. (Its mean q_u90 is 1201 ns against 615 ns, so its gas or effective field differs. The curvature sign and the absence of the ridge do not depend on that.)
3. **From 700 to 1100 V the shape is constant, which a V^2 pull would not give.** Gap curvature gww in mm per (100 mm)^2 (beta(V) measured from the mean q_u90 of the scan): -1.65 (700 V), -1.68 (900), -1.55 (golden run 1, 1000), -1.48 (1100); errors 0.15-0.25. Fit against V^2 over these four points: slope +0.24 +- 0.32 mm per (V/1 kV)^2 (chi2 0.2 for 2 dof). The electrostatic pressure step from 700 to 1100 V (0.127 Pa) is as large as the step from 500 to 1000 V (0.133 Pa). Over 700 to 1100 V gww changes by +0.17 +- 0.29 mm, over 500 to 1000 V by -1.29 +- 0.17 mm (i.e. 500 V is the flat one). A linear foil response to an equal pressure step would give the same change in both. Using the 6-25 point as the "low voltage" anchor, the V^2 line from 6-25 (500 V) to golden (1000 V) misses the 700 V point by 9.4 sigma, the 900 V point by 3.6 sigma and the 1100 V point by 3.1 sigma (`summary.json`, `V2_line_6-25_to_golden_gww`).
4. **Below 700 V the map does change in the golden mounting, reversibly.** At 500 V gww = -0.022 +- 0.013 ln (-0.26 +- 0.15 mm) where a fixed gap shape predicts -0.133 +- 0.005 (beta from the five-point fit; -0.18 if beta is extrapolated from the 700-1100 V points): 8 sigma or more. The golden ridge amplitude falls to 0.24 +- 0.18 (500 V) and 0.29 +- 0.22 (300 V) against 1.0-1.4 at 700-1100 V and 1.3 expected at 500 V. Fifty minutes later, at 1000 V (golden run 1), the golden shape is back (gww -0.103 +- 0.004). So it is not a drift in time. It looks like a flattening toward a map with a weakly positive curvature (gww +0.029 +- 0.020 at 300 V). It is not 6-25's map.
5. **Caveat on 4.** At 500 and 300 V the column ends are close to the end of the readout window (q_uend median 1500 ns at 500 V, window end 1740 ns), so q_u90 may not be a clean column duration. Emulating the window by scaling golden-shape data up to the 500 V level and capping it (hard cap 1700 ns, and a pessimistic soft cap from 1300 ns) changes gww by at most 0.012 (`inject.json`), far below the 0.11 needed. The window-independent q_u50 at 500 V gives gww = -0.012 +- 0.038 where its golden value scaled by the same factor predicts about -0.19 to -0.25 (4.5 sigma or more). These tests do not rule out a field-dependent response of q_u90 itself at low field (beta and diffusion change). That cannot be separated from a foil motion with these data.
6. **Electrostatic estimate (step 5 of the task).** Foil weight per area: 50 um kapton 0.70 Pa, plus copper 0.1 um to 9 um: total 0.71 to 1.49 Pa. Electrostatic pull epsilon0 (V/d)^2 / 2 with d = 5 mm: 0.044 Pa (500 V), 0.087 (700), 0.177 (1000), 0.214 (1100). At 1000 V the pull is 12 to 25 % of the weight, so it can reduce a sag but cannot lift the foil above flat. Direction: the window is above the cathode, so the pull lifts the foil and lengthens the gap at the centre, which is the sign of the golden bowl and of the 500 to 1000 V change. The size does not fit: 0.133 Pa produced a curvature change of 1.3 mm per (100 mm)^2, so a linear response would put the weight (0.7 to 1.5 Pa) at 7 to 15 mm per (100 mm)^2 of sag, 4 to 9 times the whole golden curvature, and the equal step 700 to 1100 V should have given 1.2 mm and gave 0.17 +- 0.29. A linear electrostatic response is therefore excluded as the explanation of the data. A threshold or a contact effect (foil resting on a support at low pull) is not excluded.

## Step 1: what exists on EOS (`.../Analysis_pass1/<run>/<subrun>/mx17_3/wft/`)

- `mx17_det3_saturday_scan_6-27-26` (golden mounting): 25 sub-runs. Drift scan at mesh 490 V: 100, 300, 500, 700, 900, 1100 V (acquired in the order 1100, 900, 700, 500, 300, 100 between 21:16 and 22:33). Mesh scans at 1000 V drift: `hv_scan` 425-525 V (step 10) and `hv_scan2` 460-520 V (step 10, 15 minutes each). The long run (490 V / 1000 V, golden run 1) follows at 22:49. Each short sub-run has an `events.parquet` of 700-800 rows (no gate table); the long run also has `bow_gate_table.parquet`. No `pairing_cost/` directory exists in these trees (events tables only). Each sub-run has `calib_bundle/bundle.json`; I fetched the bundles of the long runs (the sub-run bundles are not needed for q_u90).
- `mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check` (golden run 2, 22,300 rows, 12 acquisition files), `mx17_det3_day_6-25-26/long_run` (8,877 rows, 6 files, one sub-run only: **no other voltage exists on 6-25**).
- The 100 V sub-run has no reconstructed track at all (`n_tracks = 0` in all 754 events), so it is empty for this study.
- Other chambers' drift scans: none. `rerun/pass_registry.csv` lists drift scans only for det3 on 6-27. Other non-1000 V rows are single-voltage runs (det3/det4 6-23 at 600 V, det6/det7 at 700 V). The 6-23 det3 tables have `n_tracks = 0` in every event (chamber not reconstructed), so they could not be used. I additionally fetched the 6-22 runs (chamber A, 1000 V) as a third mounting.

## Step 2: selection

Chamber-only: `n_tracks == 1`, `x_ok`, `y_ok`, `x_quality_ok`, `y_quality_ok`, no `spark`, position `x_p0 + 14 x_tan_theta`, `y_p0 + 14 y_tan_theta` inside the active area minus 30 mm. q_u90 = mean of `x_q_u90` and `y_q_u90`. The events table has the same columns as the gate table minus the telescope ones.
N per sub-run: drift scan 534 (500 V), 551 (700), 513 (900), 462 (1100), 448 (300); mesh-scan sub-runs 224-506; golden run 1: 4,667; run 2: 14,090; 6-25: 6,304; 6-22: 12,103.

**Check on golden run 1** (gate-table selection with the telescope, N = 4,654, against the events-table selection, N = 4,667; 4,646 common): shape coefficients are the same within 0.001 (gww -0.1031 +- 0.0045 against -0.1027 +- 0.0045; guu +0.0267 against +0.0256; guw +0.0033 against +0.0031). The K = 12 spline maps (R1's P-spline) agree with r = 0.998 (total) and 0.991 (fine part), rms difference 0.0036 in ln q (rms of the maps 0.057). The telescope is not needed.

## Step 3: method

Per sub-run, one robust fit (3.5 MAD clip) of
`ln q_u90 = c0 + gu U + gw W + guu U^2/2 + gww W^2/2 + guw U W + a (ln Q - median) + b_x |tan_x| + b_y |tan_y|`
with U = (x - 200)/100 mm, W = (y - 200)/100 mm, Q the mean strip charge sum (amplitude walk), tan the pass-1 angles. Errors: bootstrap over events (200 replicates; 100 for time blocks). The covariates matter little (`r4_11_covcheck.py`): going from no covariates to walk plus |tan| moves gww by -0.017 (6-25), -0.003 (golden 1), -0.004 (500 V), -0.004 (700 V) and +0.003 (6-22, but +0.019 with the walk term alone). Walk slopes are <= 0.06; |tan| coefficients are -0.1 to -0.4 for 6-25 and 6-22, ~0 for golden. Dropping all events with a saturated strip leaves gww = +0.043 +- 0.024 (6-25, N = 839) and -0.105 +- 0.003 (run 2, N = 9,988), the same as with them.
"Golden ridge amplitude": the K = 12 spline map of golden run 2 (independent events), its quadratic part removed, is used as a template. A sub-run is fitted with the quadratic free plus the template times an amplitude a. a = 1 means the sub-run carries the golden fine structure (the diagonal ridge, `fig2b_maps_spline.png`) at the golden strength in ln units. For a fixed gap shape a rises at lower voltage by (1 + beta(V)) / (1 + beta(1000 V)). Golden run 1 against the run 2 template: 1.14 +- 0.06 (the two golden runs agree on the fine map, r = 0.94, slope 0.89).
Conversion to mm: delta g = 28.2 mm x delta ln q / (1 + beta). Reference beta = 0.68. beta(V) measured from the mean q_u90 of the scan (quadratic in ln V through 500, 700, 900, 1000, 1100 V): 1.43 (500 V), 1.16 (700), 0.96 (900), 0.87 (1000), 0.80 (1100). Without the 500 V point (window) it is 2.05 (500, extrapolated), 1.41 (700), 0.94 (900), 0.74 (1000), 0.56 (1100). beta(V) is therefore uncertain by about 0.1-0.15; all conclusions rest on signs and on ln units first. For 6-25 beta = 1.28 (R1's assumption), for 6-22 0.68.

### Main table (ln q_u90 per (100 mm)^2; full table with mesh scans in `tables.md`)

| sub-run | mesh V | drift V | N | mean q_u90 (ns) | guu | gww | guw | golden-ridge amp. | guu / gww / guw in mm, beta 0.68 | in mm, beta(V) |
|---|---|---|---|---|---|---|---|---|---|---|
| 6-27 drift scan 300 V | 490 | 300 | 448 | 1374 | -0.023 +- 0.015 | +0.029 +- 0.020 | +0.002 +- 0.008 | +0.29 +- 0.22 | -0.39 / +0.49 / +0.04 | -0.27 / +0.34 / +0.03 |
| 6-27 drift scan 500 V | 490 | 500 | 534 | 1372 | -0.023 +- 0.015 | -0.022 +- 0.013 | +0.002 +- 0.006 | +0.24 +- 0.18 | -0.38 / -0.38 / +0.03 | -0.26 / -0.26 / +0.02 |
| 6-27 drift scan 700 V | 490 | 700 | 551 | 905 | +0.033 +- 0.012 | -0.127 +- 0.013 | -0.012 +- 0.006 | +1.44 +- 0.16 | +0.55 / -2.12 / -0.20 | +0.43 / -1.65 / -0.16 |
| 6-27 drift scan 900 V | 490 | 900 | 513 | 674 | +0.016 +- 0.014 | -0.117 +- 0.014 | +0.000 +- 0.007 | +1.32 +- 0.18 | +0.27 / -1.96 / +0.00 | +0.23 / -1.68 / +0.00 |
| 6-27 drift scan 1100 V | 490 | 1100 | 462 | 580 | +0.031 +- 0.011 | -0.095 +- 0.015 | -0.007 +- 0.006 | +1.01 +- 0.17 | +0.53 / -1.59 / -0.11 | +0.49 / -1.48 / -0.11 |
| 6-27 long run (golden run 1) | 490 | 1000 | 4667 | 615 | +0.027 +- 0.004 | -0.103 +- 0.004 | +0.003 +- 0.002 | +1.14 +- 0.06 | +0.45 / -1.73 / +0.05 | +0.40 / -1.55 / +0.05 |
| 6-28 p2 overnight (golden run 2) | 490 | 1000 | 14090 | 606 | +0.026 +- 0.002 | -0.097 +- 0.003 | +0.000 +- 0.001 | +1.21 +- 0.03 | +0.44 / -1.62 / +0.01 | +0.39 / -1.46 / +0.01 |
| 6-25 day (bottom slot) | 495 | 500 | 6304 | 1056 | +0.052 +- 0.004 | +0.049 +- 0.005 | -0.041 +- 0.002 | +0.45 +- 0.07 | +0.88 / +0.82 / -0.68 | +0.65 / +0.60 / -0.50 (beta 1.28) |
| 6-22 overnight long run | ? | 1000 | 12103 | 1201 | +0.008 +- 0.004 | -0.000 +- 0.005 | -0.003 +- 0.002 | +0.12 +- 0.06 | +0.13 / -0.01 / -0.06 | same (beta 0.68) |

(gu and gw, the plane terms, are +0.03 and -0.03 at golden voltages and about 0 at 300 and 500 V; the first one is +0.031 +- 0.002 on 6-25 too. They are in `tables.md`.) The 6-25 minus golden difference in gww here is +0.152 +- 0.007 and in guw -0.044 +- 0.003, against +0.136 and -0.045 in R1 (R1 fitted the quadratic to the spline map of the telescope-matched selection; the sign and size agree).

## Step 4

**(a) Drift scan in the golden mounting.** gww, guw against V^2 and against V, four points 700, 900, 1000, 1100 V (`vfits_700_1100` in `summary.json`):

| coefficient | model | slope per (V/kV)^2 or per (V/kV) | chi2 / 2 dof | extrapolation to 500 V | observed 500 V | deviation |
|---|---|---|---|---|---|---|
| gww (ln) | V^2 | +0.047 +- 0.024 | 0.1 | -0.139 +- 0.017 | -0.022 +- 0.013 | 5.4 sigma |
| gww (ln) | V | +0.083 +- 0.041 | 0.2 | -0.145 +- 0.020 | -0.022 +- 0.013 | 5.1 sigma |
| gww (mm) | V^2 | +0.24 +- 0.32 | 0.2 | -1.74 +- 0.23 | -0.26 +- 0.15 | 5.3 sigma |
| guw (ln) | V^2 | +0.017 +- 0.012 | 4.7 | -0.011 +- 0.008 | +0.002 +- 0.006 | 1.2 sigma |

gww and guw do not move monotonically from the golden values toward the 6-25 values as the voltage falls. Over 700-1100 V the magnitude of gww in ln units falls slightly with rising voltage (-0.127, -0.117, -0.103, -0.095), as beta predicts for a fixed gap, and in mm it is flat. At 500 V gww jumps to -0.022 and at 300 V to +0.029 +- 0.020 (q_u90 at 300 V is partly cut by the window: 2.7 % of events within 10 ns of its end), so the sign change expected from the 6-25 comparison is approached only at 300 V, with a 1.5 sigma significance. beta cannot cause a sign change. 500 V of 6-27 against 6-25 (direct, same nominal drift voltage):

| coefficient | 6-27, 500 V | 6-25 | difference | sigma |
|---|---|---|---|---|
| guu | -0.023 +- 0.015 | +0.052 +- 0.004 | +0.075 | 4.9 |
| gww | -0.022 +- 0.013 | +0.049 +- 0.005 | +0.071 | 5.0 |
| guw | +0.002 +- 0.006 | -0.041 +- 0.002 | -0.042 | -6.3 |
| gu | +0.003 +- 0.005 | +0.031 +- 0.002 | +0.028 | 4.9 |
| gw | -0.019 +- 0.006 | -0.014 +- 0.002 | +0.005 | 0.9 |
| golden ridge amplitude | 0.24 +- 0.18 | 0.45 +- 0.07 | +0.21 | 1.1 |

The 6-27 500 V map is neither the golden one nor the 6-25 one. It shares with 6-25 only the reduced ridge amplitude (0.24 +- 0.18 and 0.45 +- 0.07; an amplitude of 1.3 would be expected from a fixed gap at 500 V in the golden gas). 6-27 at 500 V against golden run 1: gww +0.081 +- 0.014 (5.8 sigma), guu -0.049 (3.2 sigma), guw 0.

**(b) Mesh-voltage scan at 1000 V drift, 19 sub-runs, mesh 425-525 V** (`fig3_mesh_scan.png`; saturated-event fraction 0 to 99 %, mean q_u90 609 to 663 ns). The shape is constant: weighted means guu = +0.024 +- 0.003, gww = -0.0995 +- 0.0040, guw = +0.0001 +- 0.0017 against golden run 1 (+0.027, -0.103, +0.003). chi2 against a constant: 18.6, 18.5, 20.5 for 17 dof. Slopes per 100 V of mesh: guu +0.002 +- 0.013, gww -0.012 +- 0.017, guw +0.003 +- 0.007; ridge amplitude 1.21 +- 0.05, chi2 11/17, slope -0.16 +- 0.20 per 100 V. The control passes: gain changes q_u90 by 8 % over the scan and not the shape.

**(c) Diagonal ridge.** Present at 700 V (amplitude 1.44 +- 0.16), 900 V (1.32 +- 0.18), 1000 V (1.14 +- 0.06 and 1.21 +- 0.03, golden runs), 1100 V (1.01 +- 0.17) and in all mesh-scan sub-runs (mean 1.21 +- 0.05). Absent at 500 V (0.24 +- 0.18) and 300 V (0.29 +- 0.22), and nearly absent in 6-22 (0.12 +- 0.06). It is partly present in 6-25 (0.45 +- 0.07), though with another quadratic shape. The ridge amplitudes at 700-1100 V follow the fixed-shape expectation (dashed line in `fig1`) within errors (a falls from 1.4 to 1.0 as beta falls); the three points below 700 V and the other mountings do not.

**(d) 6-25 sub-runs.** The only 6-25 sub-run is `long_run` (one voltage, 500 V drift, mesh 495 V). Its six acquisition files give a constant shape (see (e)). There is no voltage information from 6-25.

**(e) Stability in time** (`fig4_time_blocks.png`, one block per acquisition file, ~1,200 events, errors 0.009-0.012 on gww):
- 6-27 long run (4 blocks) against 6-28 overnight (12 blocks), one day apart: gww -0.103 +- 0.004 against -0.097 +- 0.003 (difference 0.006 +- 0.005), guu +0.027 against +0.026, guw +0.003 +- 0.002 against +0.000 +- 0.001 (1.2 sigma). The fine map agrees (r = 0.94, slope 0.89).
- Inside the 6-28 run (12 blocks, ~11 hours): gww mean -0.0975, chi2 7.9/11; guu 9.7/11; guw 10.4/11. No trend.
- Inside the 6-27 long run (4 blocks): gww chi2 1.8/3; guw 6.2/3 (blocks +0.008, +0.007, -0.006, +0.005; errors 0.003-0.004, a ~0.01 scatter in the twist).
- Inside the 6-25 run (5 blocks used): gww +0.0486, chi2 1.6/4; guw -0.040, chi2 2.2/4; guu 1.2/4. Stable over its ~6 hours.
The shape is stable within every dataset, so the differences between datasets are not slow drifts.

## What the result does not rule out
- A foil that responds to the drift voltage only below ~700 V (threshold or contact), or a field-dependent response of q_u90 at low field. These look the same in this data. The window cap (1740 ns) comes close to the column end at 300-500 V, and beta(V) at 500 V is not measured independently of the cap (a 5-point fit gives 1.43, an extrapolation 2.05).
- That the mounting difference is not only the mounting: 6-22, 6-25 and 6-27 differ in date, gas (mean q_u90 of 1201, 1056 and 615 ns at nominal 1000/500/1000 V) and possibly other things. "Mounting" here means everything that changed between the datasets other than the drift voltage inside 6-27.
- A voltage effect on the 6-25 shape that works through another mechanism than the one seen in 6-27: the 6-27 scan covers 300-1100 V only in one mounting. The 6-25 mounting has no scan.
- Statistics: the short sub-runs have N = 450-550, so each coefficient has an error 3 to 5 times that of the long runs (gww +-0.013 to 0.015), and amplitudes of the ridge carry +-0.15 to 0.2.
- No new data can be taken, so the low-voltage behaviour cannot be probed further.
