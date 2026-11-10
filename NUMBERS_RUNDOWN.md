# Paper I: numbers that disagree between places, run down to their sources (2026-11-10)

Reading and small recomputations only; nothing in the repo was changed except this file. `.tex` references are to
`papers/paper1/sections/` unless stated. Chambers: A=det3, B=det2, C=det6, D=det7, E=det4. Run 1 = `sat_det3`,
Run 2 = `p2_det3` (registry key `c26__mx17_det3_p2_det1_overnight_6-27-26`, subrun `long_run_p2_det1_sanity_check`).
"Sheet" = `rerun/PAPER1_NUMBERS_2026-10-31.md`.

---

## 1. Run 2 drift velocity ("35.8 against 37.2 um/ns")

- **Paper:** `03_setups.tex:49` "The one difference is the drift velocity, 35.8 against 37.2 um/ns, as the gas dried through the night".
  `06_gas.tex:11` "timing velocity 17.7--37.2 um/ns at 177--390 V/cm" (37.2 there is the top of the scan, 1100 V on the cathode).
- **Source:** the two numbers are two different estimators, and 37.2 for Run 2 has nothing to do with the 37.2 of the scan.
  - 35.8 = Run 1's *timing* velocity (head-on stack, gap/T): `rerun/calib/physics_inputs.json` key `det3.v_timing_um_ns` = 35.8;
    `rerun/paper1_res/results/rows.csv` (`sat_det3`) `v_drift` = 35.8.
  - 37.19 = Run 2's *registry* velocity, the reference-ladder velocity with kw folded in (`vrefit --fold-kw`):
    `rows.csv` (`c26__mx17_det3_p2_det1_...`) `v_drift` = 37.1948 (kw_pass 1.0089). It is 35.8 x Run 2's mean kw, not a timing velocity.
  - `rerun/paper1_timeseries/make_report.py:244-252` says exactly this: "Those are two different estimators ... Recommendation for
    the text: quote the change on one estimator (about +1.8 % ladder, +0.5 % column), not 35.8 against 37.2."
- **On one estimator** (`rerun/paper1_timeseries/results/timeseries.json`):
  - reference ladder, whole rows, unanchored: 36.77 (Run 1) and 37.42 (Run 2), +1.8 %.
  - ladder anchored to Run 1's timing v (k = 35.8/36.77 = 0.9735): **35.80 and 36.43 um/ns** (`v/anchored_whole`). Run 1 flat (35.75 to 35.85 in the line fit); Run 2 rises inside the run, 36.24 to 36.64 (`trends/v/per_run/run2/line_start, line_end`; mean 36.44). This is what the time-series figure shows (35.8; 36.2-36.6).
  - column duration (no reference), scaled to Run 1 = 35.8: Run 2 mean 35.99, +0.5 %.
  - timing velocity gap/T of each row (`rerun/paper1_timing/results/golden.csv`, column `v`): 36.03 (Run 1), 36.56 (Run 2).
  - ladder steps up by about 0.3 um/ns (0.9 %) across the 23 min between the runs while the column does not: source cannot say whether that step is gas or a difference between the two rows' reconstructions (Run 2 uses Run 1's bundle with v folded).
- **Field:** 1000 V is 333 V/cm on a 30 mm gap (registry `E_drift_V_cm`) and 355 V/cm on the 28.2 mm gap that the 177-390 V/cm labels of the scan use. The scan's timing velocity at about 355 V/cm is about 36 um/ns, i.e. Run 1's value; the 37.2 at 390 V/cm is 1100 V.
- **Use in the paper:** "The drift velocity (reference ladder, anchored to the timing velocity of Run 1) is 35.8 um/ns in Run 1 and 36.4 um/ns in Run 2 (rising from 36.2 to 36.6 during the night, +1.8 %), as the water falls from 0.99 to 0.94 %." Never "37.2" for Run 2. Add "at 1000 V drift (355 V/cm)" or "(333 V/cm on 30 mm)" consistently with `06_gas`.

## 2. Spread of f: "0.116" (D) against core sigma of ln(Q_X/Q_Y) = 0.109

- **Paper:** `02_design.tex:12` and `05_efficiency.tex:9` "event-to-event spread 0.014--0.032 (D 0.116, its X readout fault)"; caption `05_efficiency.tex:15` "core sigma of it ... D 0.109".
- **Source:** `rerun/paper1_charge/make_report.py:93-100`, `results/summary.json` key `golden[]`. Two different statistics:
  - "spread" in f = `f_s68` = half the 16-84 percentile range of f = Q_X/(Q_X+Q_Y) (not a Gaussian sigma).
  - caption sigma = `xy_ln_core_sigma` = 1.4826 x MAD of ln(Q_X/Q_Y) (robust core, ignores tails).
  - They differ for D because 26 % of D's unsaturated events lie more than 0.5 in ln away from the median (`xy_tail_frac` 0.2601); the percentile range includes them, the MAD does not. For the other chambers the tail is 0.05-9 % and the two agree after converting (df = f(1-f) dL, about 0.25 dL).
- **One consistent set** (golden rows, unsaturated tracks; n = unsaturated tracks in the caption):

  | chamber | f median | s68 of f (percentile) | core sigma of ln(Q_X/Q_Y) | core sigma in f (= 0.25 x) | tail fraction |
  |---|---|---|---|---|---|
  | A | 0.485 | 0.014 | 0.058 | 0.014 | 0.0005 |
  | B | 0.513 | 0.016 | 0.067 | 0.017 | 0.005 |
  | C | 0.504 | 0.032 | 0.114 | 0.028 | 0.086 |
  | D | 0.485 | 0.116 | 0.109 | 0.027 | 0.260 |
  | E | 0.509 | 0.026 | 0.097 | 0.024 | 0.033 |

- **Use in the paper:** say "core spread of f 0.014 (A) to 0.028 (C, D); on D a tail (26 % of tracks beyond e^0.5 in the charge ratio, from its X readout fault) raises the 68 % half-range to 0.116". Do not call 0.116 a sigma. Median f in the caption (A 0.485, B 0.513, E 0.509, C 0.504, D 0.485) matches. Run-by-run caveat: D's 2459 events are only 14 % unsaturated (`rerun/paper1_charge/README.md`), C's f_sat is 0.92, so C and D have 346-452 tracks.

## 3. Active area: "399.4 x 359.9 mm^2", bands 18.0 and 18.7, x edge 398.6

- **Paper:** `02_design.tex:27-29,38`, `08c_edge.tex:8` ("x = 0 and 398.6 mm"), `08c` caption (18.0 and 18.7 dotted).
- **Source:** `common/mx17_active_area.py`, `ntof_active_area/ACTIVE_AREA_2026-08-11.md:140-155`, `docs/detector_spec/spec.py:147,155,225-227`.
  - Nominal (CAD): 512 x 0.78 = **399.36 mm** metallised window (`spec.py:155`) -> "399.4". The strip *coordinates* run 0 to **398.58 mm** (`STRIP_MAX_MM`, `mx17_m1_map.csv`: 511 x 0.78, the start of the last strip) -> "398.6" is the position of the last strip, not a different width. 399.36 - 398.58 = one pitch (0.78).
  - X: no passivation, sharp edges 0 / 398.6, measured live at strip 0 and 511 (chamber B, run_79).
  - Y bands **18.0 / 18.7 mm are chamber A only** (June cosmic bench, 50 % efficiency points, `PASSIVATION_Y_LOW/HIGH_MM`). Per chamber (`TRUE_ACTIVE_BY_DET`): B 19.7/19.8, C 17.9/17.9, D 20.4/19.3, E 18.4/20.0. Chamber A's own Y span is 398.58 - 18.0 - 18.7 = **361.9 mm** (not 362.7: that subtracts from 399.4, which is not the coordinate of the last strip).
  - **359.9 +- 1.8 mm is a combination, not A's value**: the mean of seven measurements (3 beam, run_79: A 19.5-376.7 = 357.2, B 18.7-379.1 = 360.4, C 20.3-379.1 = 358.8; 4 June: B 359.1, C 362.8, D 358.9, E 360.2), midpoint 199.1 mm against the strip-plane centre 199.3 (`ACTIVE_AREA_2026-08-11.md:151`). It is the value used for the Geant `mm_size_v` (`SIM_ACTIVE_V_CM` = 36.0). `spec.py:227` writes it "399.4 mm in X x 359.9 +- 1.8 mm in Y".
- **Use in the paper:** "metallised area 399.4 x 399.4 mm^2 (512 x 0.78 mm); no passivation in x; in y the strip ends are passivated over about 19 mm on each side (chamber A 18.0 and 18.7 mm, all chambers 17.9-20.4 mm), leaving an efficient length of 359.9 +- 1.8 mm over all chambers and both methods (A alone: 361.9 mm from the June 50 % points, 357.2 mm in the beam)". Quote the x extent as 0--398.6 mm for strip positions, or 399.4 for the width, but say which. Do not subtract 18.0 and 18.7 from 399.4.

## 4. Dead time after a discharge: "below 1.2 ms" against "fitted below 0.1 s"

- **Paper:** `04_operation.tex:36` "for chamber A it is below 1.2 ms (95 % CL, 0.2 +- 0.6 ms)"; `A2_discharge_recovery.tex:4-5` "the integrated dead time per discharge is fitted below 0.1 s"; caption A: 0.2 +- 0.6 ms.
- **Source:** `rerun/paper1_ops/README.md` Verdict 3: "Efficiency against the time since the previous discharge is flat from 1 ms to 1 s. The lost time per discharge, integrated below 0.1 s" -> det3 `0.2 +- 0.6 ms`, 95 % UL 1.2 ms (B -1.7 +- 1.3 / UL 2.1; C 0.8 +- 1.1 / 2.6; D 0.4 +- 1.1 / 2.1; E -2.4 +- 9.4 / 15.5 ms).
- **Why they differ:** "0.1 s" is the *integration window* (the efficiency deficit is summed over delays 0-0.1 s after the discharge), not the result. The result is 0.2 +- 0.6 ms, UL 1.2 ms. The appendix sentence reads as if the bound were 0.1 s.
- **Use:** A2: "the dead time per discharge, integrating the efficiency deficit over the first 0.1 s after a discharge, is 0.2 +- 0.6 ms for chamber A (95 % upper limit 1.2 ms; B-E: upper limits 2.1, 2.6, 2.1, 15.5 ms)". Note for 04: 1.2 ms is the *upper limit* of the integral, and the first bin (0-3 ms, 53 tracks) is 1.07 +- 0.04 of the late value. Chamber E's limit (15.5 ms) is not "none" in the same sense; keep it in the caption.

## 5. Dark current: "no measurable ohmic part" against "a leakage floor of about 0.2 uA"

- **Paper:** `04_operation.tex:13` "with no measurable ohmic part"; caption `04_operation.tex:22` "before the gain takes off the current is a leakage floor of about 0.2 uA (0.17 uA at 425 V on the scan night, 0.2--0.3 on the other nights)".
- **Source:** `rerun/paper1_ops/currents/README.md` and `results/currents.json` key `det3_6-27_fit`: model I = b + A 2^((V-490)/V2), fitted by `hv_currents.py:126` to the 18 hv_scan points: **b = 0.012 +- 0.038 uA** ("b = 0.01 +- 0.04 uA"), A = 1.049 +- 0.046 uA, V2 = 13.32 +- 0.32 V.
- **What the fit actually says (recomputed here from `results/currents.csv`):** the constant is consistent with zero, but the single exponential does not describe the foot of the scan. Measured I at 425/435/445/455 V = 0.167/0.172/0.176/0.181 uA; the fit gives 0.049/0.073/0.115/0.184 uA (it undershoots by 0.12, 0.10, 0.06 uA and overshoots by 0.05-0.15 uA at 460-480 V); rms residual of the fit 0.073 uA. The foot is flat at 0.17-0.18 uA (+8 % for +7 % in V from 425 to 455 V: consistent with an ohmic leak, about 0.4 uA per kV, 2.5 GOhm; an ohmic term V/R instead of a constant gives the same fit quality, rms 0.073, and b = 0.010 +- 0.043 uA). So the fit result and the 0.2 uA floor are not in conflict about the high-V data, but "no measurable ohmic part" is not supported by the points below 460 V; it is only that the fit does not need one to describe the points above 470 V. Chamber B on 6-22 shows a 1.9 uA flat floor below 480 V ("an ohmic leak appeared", README table).
- **Use:** "Above about 465 V the current doubles every 13.3 +- 0.3 V, I = 1.05 +- 0.05 uA x 2^((V-490 V)/13.3 V), i.e. I proportional to G^1.25. Below 460 V it is flat at 0.17-0.18 uA on the scan night (0.2-0.3 uA on other nights), about 15 % of the current at 490 V; the fit with a constant term gives 0.01 +- 0.04 uA and does not follow this foot." Drop "no measurable ohmic part" or say "the exponential needs no additional constant".

## 6. Efficiency map: "92.6 % (pool of 30 rows, 70 470 rays)" against Runs 1+2 and 92.3 %

- **Paper:** `05_efficiency.tex:37,48` (caption: "A 92.6 % (pool of 30 rows, 70 470 rays), B 90.3 (14 rows), C 74.4 (13), D 68.0 (21), E 40.9 (golden run only)"); text `05_efficiency.tex:33` and `A1_loss_categories.tex:10`: interior 92.3 % (92.29), 26 542 tracks of Runs 1+2; `03_setups.tex:46`: 29 398 reference tracks (full active area).
- **Source:** `rerun/paper1_figs/results/efficiency_map.json` (rows, rays, golden_eff_ig), pools in `rerun/effbudget/pass1/results/map_pools.json`, `rerun/effbudget/pass1/fine_maps.py` docstring:
  - **A "row"** = one registry sub-run (`rerun/pass_registry.csv`): a single HV step of a scan or a long run. A "plateau pool" = every row of that chamber at the golden drift field whose interior efficiency is within 3 points of the golden row's, so the map is **not** the long runs.
  - Pool of A (30 rows, 70 470 rays): 6-22 det3 rows (13: long_run, short_run, 450-500 V), 6-27 scan rows (15: 425-505 V), `sat_det3` (Run 1), the 6-27 p2 long run (Run 2), and one 6-22 test row. So Runs 1+2 are 2 of the 30 rows; the map contains many scan points at other voltages.
  - The **92.6 %** is `sat_det3` = Run 1 alone, pass-1 table (6 454 rays; `rerun/effbudget/PASS1_RESULT_2026-10-31.md`: 92.6 +- 0.3). Run 2 alone: 92.17 +- 0.19 (20 088 rays; `rerun/paper1_ops/figures/hv_scan.csv`). The **92.3 %** is Runs 1+2 on the two-track re-pass tables (`Analysis_pass1_twotrack`; `A1_loss_categories.tex:10`, 26 542 interior rays). 29 398 is the full-area count. Three different samples, three numbers.
  - **B** pool: 14 rows of the 6-22 night (450-500 V, long_run, o22 calibration row), golden `o22_long_det2` 90.3 % (3 395 rays); the quoted 6-22 `long_run` row reads 90.9 % (17 568 rays).
  - **C, D** pools are scan points, not the long run: C 13 rows (6-26 scan 410-485 V, `g_det6_long` at 495 V, the overnight longer_run); D 21 rows (scan 400-495 V, long run 495 V, longer_run, short_run, and the overnight steps where D sat at 480-495 V while C was at 505-520 V). Selection "within 3 points of the golden row's interior efficiency" (74.4 / 68.0 %) brings in both lower-efficiency low-HV points and higher-efficiency optimum points, so the map is a mixture of operating voltages. The caption's 74.4 and 68.0 are the golden long run (495 V) only.
  - E: one row (`g_det4`).
- **Efficiency at the long-run HV and at the scan optimum** (interior_good, R = 5 mm, `rerun/paper1_ops/README.md` Verdict 1, `figures/hv_scan.csv`):

  | chamber | long run (HV resist / drift) | eff at long-run HV | scan optimum | eff there | plateau (within 1 pt) |
  |---|---|---|---|---|---|
  | A | 490 / 1000 V | 92.3 % (Runs 1+2); Run 1 92.6, Run 2 92.2 | 470 V | 93.8 % (690 rays) | 460-490 V |
  | B | 495 / 1000 V | 90.3 % (o22 row); 90.9 % (quoted 6-22 long run) | 480 V | 92.2 % | 475-490 V |
  | C | 495 / 700 V | 74.4 % | 480 V | 85.8 % (338 rays) | 480 V only |
  | D | 495 / 700 V | 68.0 % (X channels 256-511) | 430 V | 89.8 % (147 rays) | 430-435 V |
  | E | 495 / 600 V (bench) | 40.9 % (71.3 % in live bands) | n/a, no scan | n/a | n/a |

  Optima are +-5 V (150-700 rays per scan point). Unverified: whether the 89.8 % of D at 430 V uses the same half-X (ch 256-511) restriction as the 68.0 %; the scan rows 66.1 % at 495 V (scan) against 68.0 % (long run) suggest it does, but I did not check the producer. Tracks lost to discharges at the long-run HV: A 0.9-1.2 %, B 1.5-1.9 %, C 16.8 %, D 24.7 % (`PAPER1_NUMBERS` row 4 / `paper1_ops` README).
- **Use:** caption "(a) chamber A ... plateau pool of 30 rows (70 470 reference rays; Runs 1 and 2 are two of the rows)"; quote 92.3 % (Runs 1+2) in the text and drop the 92.6 % from the caption, or write "golden-run interior efficiency (pass-1 tables): A 92.6 (Run 1) ...". For C and D state that the map pools scan points near the plateau and the numbers are the 495 V long run.

## 7. Angular resolution: A 0.98 vs 0.99; B 1.03/1.08 vs 1.05/1.13; decile range 0.82-0.94

- **Paper:** `02_design.tex:32` "A 0.98/0.90, B 1.03/1.08"; `08_tracking.tex:7-8` "A 0.99/0.90; B 1.05/1.13"; `03_setups.tex:47-48` Run 1/Run 2 "0.975 / 0.991 in X and 0.896 / 0.900 in Y"; `05_efficiency.tex:64`.
- **Source:** `rerun/paper1_res/results/angle_resolution.json` (Runs 1+2, 03_angles interior_good, pass arm), `rows.csv`, `PASS1_RESOLUTION_2026-10-31.md`.
  - A: Run 1 (`sat_det3`) X 0.9748 / Y 0.8960 (6 198 / 6 182 tracks); Run 2 X 0.9906 / Y 0.8999 (19 224 / 19 177); **Runs 1+2 summed X 0.9870 / Y 0.8986** (25 422 / 25 359). `02_design` 0.98/0.90 = Run 1 alone (0.975 rounds to 0.98); `08_tracking` 0.99/0.90 = Runs 1+2 summed (its caption says "Runs 1 and 2 summed"). Run 2 alone is also 0.99/0.90, which is why the two look alike. (`PASS1_RESOLUTION` table quotes 0.991/0.901 = the Run 2 row, "6-27 sanity long_run".)
  - B has two golden rows: the quoted 6-22 `long_run` (16 746 tracks) X 1.0448 / Y 1.1254 -> **1.05/1.13** (`08_tracking`); the calibration row `o22_long_det2` (3 198 tracks) X 1.0253 / Y 1.0835 -> **1.03/1.08** (`02_design`). `PASS1_RESOLUTION` headline = the quoted row.
  - C 1.256/1.506, D 1.232/1.461, E 2.412/2.076 (pass arm); the v(t) arm is lower by up to 0.063 deg (C 1.211/1.454, D 1.225/1.449, E Y 2.013).
- **Use:** A 0.99/0.90 (Runs 1+2) and B 1.05/1.13 (quoted row, 16 746 tracks) in both sections; in `02_design` replace 0.98/0.90 and 1.03/1.08 accordingly. In `03_setups` keep 0.975/0.991 and 0.896/0.900 as Run 1/Run 2.
- **Decile range (0.82-0.94 X, 0.72-0.87 Y):** producer **NOT FOUND** (searched `rerun/paper1_ops/`, `paper1_res/`, `paper1_timing/`, `paper1_floor/`, the figure manifest `papers/paper1/figures.toml`; the numbers are not in any results file or script; `hv_metrics.py:17` has only the claim "Per track, the angle residual does not depend on the track's charge"). The selection is in the sentence itself (`05_efficiency.tex:63-64`): pooled scan points below 510 V (so 425-505 V, 1000 V drift) and **unsaturated tracks only**. The headline 0.99/0.90 is all tracks at 490 V, of which 38 % have a strip at the ADC rail (`timeseries`: `f_sat` 38.0 %), and saturated tracks are the poorer ones. Per-HV s68 of all tracks over the same scan is 0.91-1.03 X / 0.80-0.96 Y (`05_efficiency.tex:54-55`, `rows.csv`), which is already above the decile range. This is an inference from the selection, not a recomputation: to state it in the paper, the decile numbers need a producer script in the repo.

## 8. Position residual core at the mesh, chamber A

- **Paper:** `03_setups.tex:48` 0.50/0.51 mm (X); `05_efficiency.tex:56` 0.43-0.56 mm; `A4_efficiency_definition.tex:41` 0.55 +- 0.02 / 0.53 +- 0.03 mm; `08_tracking.tex:13` 430-620 um.
- **Source and estimator of each:**
  - **0.50 / 0.51 mm (X)** = Run 1 / Run 2 `pos_mesh_core_pass` X in `rerun/paper1_res/results/rows.csv`: 0.5036 / 0.5088 (Y: 0.4726 / 0.4890). Estimator of `paper1_res/res_rows.py:73`: Gaussian fitted to the histogram of the residual in the fixed window median +- 2 s68, per view, M3 pointing *included*, spark veto, |r| < 15 mm.
  - **0.43-0.56 mm** = the range of the same `pos_mesh_core_pass` over the 6-27 HV-scan rows 425-510 V (about 660 tracks each): X 0.456-0.561, Y 0.434-0.565 (recomputed from `rows.csv`; max X at 485 V, min Y at 470 V). Statistical scatter of small rows, not a trend.
  - **0.55 / 0.53 mm (Gaussian core)** = Gaussian fit to the 0.25 mm histogram of the mesh residual within |r| < 1.5 mm, Poisson weights, Runs 1+2 summed, all effbudget rays (`rerun/paper1_timeseries/timeseries.py:96-97, 24`; results: Runs 1+2 X 0.547 / Y 0.531, per run X 0.543/0.543, Y 0.514/0.531). Reference pointing and scattering included. Narrower window and different binning than the previous estimator, hence 0.04 mm larger in X.
  - **430-620 um** = "bench quoted" residual core range over chambers/rows, `rerun/paper1_floor/README.md:57` (and "W 430-620 um" at `:119`); it is not chamber A's range. Pointing-subtracted per-chamber cores (`PASS1_RESOLUTION` Verdict 2): B 456/444, A 447/443, E 663/526, C 348/382, D 434/488 um (that is sigma_DUT, which is lower than the raw core).
- **Use:** one estimator for chamber A: "the Gaussian core of the residual at the mesh (|r| < 1.5 mm), 0.55 (X) and 0.53 mm (Y) on Runs 1+2". In `03_setups` replace the per-run 0.50/0.51 by the same estimator per run, or by the sentence "0.54/0.54 mm (X)" (timeseries per-run X 0.543/0.543); the 0.43-0.56 mm HV-scan range is from the other (window) estimator and should be labelled so, or recomputed with the |r| < 1.5 mm fit. In `08_tracking` "430-620 um" should read "the core of the residual for the five chambers' bench rows (0.35-0.66 mm after pointing, chamber A 0.55/0.53 mm)" or be dropped.
- Also note `08_tracking.tex:13-17` and the figure caption give A chamber term 128 +- 33 (X) / 113 +- 36 (Y) um and B 163 +- 29 / 176 +- 21; the sheet and `paper1_floor/README.md:57` say A 126/116 +- 35 (older numbers). The 08_tracking values are the current ones (not re-derived here).

## 9. SPS gas figure: band "bench golden rows, chambers A-D, 0.42-1.00 %"

- **Paper:** `06_gas.tex:49` caption "Grey band in (a): the water of the bench golden runs, chambers A--D, 0.42--1.00 %"; text `06_gas.tex:15` "B 0.69, A 0.97, E 0.42, C 1.00, D 0.48".
- **Source:** `rerun/paper1_spsfigs/attach_fig.py:63,66,83`: `bw = [B[d]['water'] for d in ('det2','det3','det4','det6','det7')]` (all five), `axhspan(min(bw), max(bw))`, label `'bench golden rows, A-D: %.2f-%.2f %%'`. Values from `rerun/paper1_gas/results/summary.json` `golden_water`: det2 (B) 0.689, det3 (A) 0.967, det4 (E) 0.423, det6 (C) 1.00, det7 (D) 0.481. So the band 0.42-1.00 % **includes E**; the label "A-D" is wrong.
- **Use:** either label "bench golden rows, chambers A-E, 0.42-1.00 %" (the code as it is; E's own bench value is a natural comparison for E at SPS), or restrict to A-D and change to 0.48-1.00 % in `attach_fig.py`. Recommend A-E. Caveat: E's 0.42 % uses its input gap 30.48 mm; at the geometric 28.6 mm the DET4GAP note (`rerun/det4_gap/README.md`, sheet row "det4 drift gap") says E's bench water moves up. `rerun/paper1_gas/README.md:23`: det4 0.42 [0.40, 0.45], gap-error band 0.39-0.45; B 0.64-0.74, C 0.91-1.09, D 0.38-0.58; A (6-27 drift scan, 28.19 mm) 0.97 [0.95, 0.99].

## 10. Discharge definitions and rates

- **Paper:** `04_operation.tex:29` (hv_scan caption) "a discharge is a chip with >= 8 strips over 3000 ADC", "discharge loss ... below 1 % up to 490 V"; `A2_discharge_recovery.tex:2`; `A1_loss_categories.tex:32` ("113 strips over threshold ... beyond the 50-strip veto"); `05_efficiency.tex:90,94,198` discharges 0.5 / 0.47 %.
- **Source:** `rerun/paper1_ops/spark_timeline.py:13-17,83-85` and `results/spark_rows.csv`:
  - **veto** = trigger with more than 50 strips (`wft.seed.SPARK_VETO_HITS`) over the per-plane relative significance floor: the reconstruction's spark veto, and the loss budget's "spark" class.
  - **discharge** = a vetoed trigger with at least one DREAM chip of >= 8 strips (`CHIP_MIN`) above 3000 ADC (`DIS_ADC`) (`band_census.py` test). **burst** = vetoed and not a discharge. So discharge is a subset of the veto.
- **Rates for chamber A at 490 V** (`spark_rows.csv`, `hv_scan.csv`): per *trigger* (all triggers, 4 Hz): veto 5.1-8.2 %, discharges (`spark_per_trig`) 3.7-4.2 % (long run 4.2 %, p2 run 4.1 %, scan row 3.7 %), bursts 1.4-4.0 %. Per *interior reference track* (`f_in_discharge`, `eff_vs_hv_discharge.csv`): 0.85-0.95 % (scan 490 V 0.85 %, Run 1 0.95 %, Run 2 1.15 %). The HV-scan figure's "a few per cent at 490 V" is the first (right axis, per trigger), the "below 1 %" the second (tracks). Per second: 0.04 x 4 Hz = 0.17/s, as the sheet says.
  - Why per-track is 4x lower than per-trigger: `n_rays / n_trig` is 20 % for Run 1 (6 454 / 32 793), so only a fifth of triggers have an interior reference ray, and a trigger carrying a discharge is evidently less likely to have one; the sources do not say why. Not verified further.
  - **0.50 % / 0.47 %** are a third quantity, the A1 "discharge" class: first-matching class of a lost interior track (interior_good Run 1: 0.50 %; the 0.47 % is the single-track full-area figure behind `fig:lossbudget`, Runs 1+2, `05_efficiency.tex:198`; sheet "decision H": the whole spark class 1.56 % splits into 0.88 % chip-test discharge + 0.68 % burst, and the A1 "discharge" 0.50 % is 0.37 true + 0.12 burst). The chip-test discharges (0.88-0.95 %) include events the A1 classes put in "shower"/"muon + other" (0.40, 0.67 %), because those are vetoed too; the 0.50 % is only the subset that the display audit labels as a real discharge (113 strips over threshold, 220-230 at SNR > 6).
- **Use:** one sentence in the HV-scan caption: "Discharges per trigger (right axis) are triggers with a chip of >= 8 strips above 3000 ADC; 4 % of all triggers at 490 V, but only 0.9 % of the reference tracks are lost to them, because ..." and in `05_efficiency` "discharges (0.47 %)" should say "discharges of the strict class (the display-audited subset of 0.9 % of tracks in vetoed events with a saturated chip)". Make the three names explicit: veto (> 50 strips), discharge (chip test, subset of veto), A1 class "discharge" (subset again).

## 11. "same scaling as the noise bands (G^1.4-1.6, appendix A)"

- **Paper:** `04_operation.tex:16`. Appendix A (`A1_loss_categories.tex:29-32`) has no such statement (only "noise band ... 0.22 %").
- **Source:** `rerun/bands/README.md:24` (Result, "Fixed-patch bands (det3, det7)"): "Their rate rises x1.8-1.9 per 10 V of resist HV (gain^1.4-1.6) and is flat in the drift field"; data `rerun/bands/hvscan.py`, `hvscan.json`; published at dylan-neff.web.cern.ch/notes/mx17-noise-bands.html. It applies to the fixed-patch bands only (A and D); whole-plane bands (B, C, E) are flat in HV. `rerun/paper1_ops/currents/README.md` repeats it.
- **Inconsistency to fix at the same time:** the exponents differ: current G^1.25 (`det3_6-27_fit.excess_gain_power` 1.254), bands G^1.4-1.6, discharges G^2-2.7 (D G^1.4; `paper1_ops` README). "The same scaling" is too strong.
- **Use:** "a gain-driven dark current (I proportional to G^1.25), a family with the noise bands of the fixed-patch type (rate proportional to G^1.4-1.6, flat in the drift field; chambers A and D) and the discharges (G^2-2.7)". Cite the bands note, or add one sentence on the bands to appendix A, or drop the cross-reference `\cref{app:losses}`.

## 12. Operating points "480/480/440 V"

- **Paper:** `04_operation.tex:6` (placeholder).
- **Source:** `rerun/paper1_ops/README.md` Verdict 1 and `figures/hv_scan.csv`; story file line 23 "optima 480 / 470 / 480 / 430 V (det2/3/6/7)". The "480/480/440" naming is unresolved in the README (open question 1).
- **Scan optimum (best interior efficiency), plateau, long run:**

  | chamber | scan optimum | efficiency | plateau within 1 pt | long run resist / drift | long run efficiency |
  |---|---|---|---|---|---|
  | A | 470 V | 93.8 % | 460-490 V | 490 V / 1000 V (Runs 1, 2) | 92.3 % |
  | B | 480 V | 92.2 % | 475-490 V | 495 V / 1000 V | 90.3-90.9 % |
  | C | 480 V | 85.8 % | 480 V | 495 V / 700 V | 74.4 % |
  | D | 430 V | 89.8 % | 430-435 V | 495 V / 700 V | 68.0 % |
  | E | n/a (no scan) | n/a | n/a | 495 V / 600 V | 40.9 % |

  Long runs are 10 V (A), 15 V (B), 15 V (C), 65 V (D) above the scan optimum (`03_setups.tex:44-45` "10--65 V").
  Chambers A and B differ by 10 V, so "480/480/440" does not match any single reading; the data say 470-480 / 480 / 480 / 430 (D) with D the one with the sharp edge (best 430 V, 440 V already 82 %, scan rows short: 118-341 rays).
- **Use:** replace the placeholder by the table: "the scans put the efficiency optimum at 470 (A), 480 (B), 480 (C) and 430 V (D) (+-5 V); the golden runs were 10-65 V above". Note the placeholder's 440 V does not match D (430-435 V plateau; 440 V is a low point, 82.2 +- 3.5 %).

## 13. SPS "angle scale at a known angle" and "charge footprint at normal incidence"

- **Paper:** `03_setups.tex:23-25` promises both; no section of Paper I shows or quotes them (grep for `known_angle`, `kernel_forms`, `footprint`, `0.981` in `papers/paper1/sections/*.tex` finds only that sentence).
- **Known angle:** `rerun/paper1_spsfigs/README.md:8,23,49-60`, `results/known_angle.json`, figure `figures/known_angle.pdf` (assigned to Paper II section 8, not Paper I): 25.4 deg mount (the config said 25.64; the measured yaw is 25.4 deg), inclined view with tan rescaled to the column-duration v: **0.981 +- 0.03 of tan 25.4 deg** (13 rows >= 180 V/cm, 28.8 mm gap; 0.988 at the geometric gap 28.6 mm); with the registry v 0.927 +- 0.04 (6 rows >= 190 V/cm), falling to about 0.5 at 110 V/cm. The 13 rows read tan 0.467-0.496 against 0.475. The 15.465 deg run61 rows have no column v and their stage angle is unmeasured. Only |tan| (SPS tan sign not determinable).
- **Footprint at normal incidence:** `rerun/paper1_spsfigs/README.md:10,21` (figure `kernel_forms.pdf`, Paper II section 2) and `rerun/sps_audit/README.md` section 5 (`results/footprint.csv`): run 71 RAW (no zero suppression), 243 V/cm: sharing kernel +-1/lead **X 0.377, Y 0.390**; +-2/+-1 **X 0.19, Y 0.26**; field-independent from 96 to 243 V/cm (so not diffusion); X +-1 asymmetry -0.07 to -0.085, Y <= 0.01. In `sps_audit` terms (cluster of strips above 50 ADC): 2.25 (X) / 3.02 (Y) strips, lead strip 0.65 / 0.58 of the cluster, neighbour 0.31 (X) / 0.34 (Y) of the lead (different definition: per-event ratio over a 50 ADC cluster; the 0.377/0.390 is the mean peak-in-+-240 ns ratio for lead 400-3000 ADC). On ZS rows X reads 0.50-0.55 against Y 0.38-0.43, which is the threshold, not the chamber.
- **Use:** either add a short subsection "SPS cross-checks" with the two numbers (known-angle ratio 0.981 +- 0.03; kernel +-1/lead 0.38 X / 0.39 Y at normal incidence, RAW) and the figures `known_angle` / `kernel_forms`, or delete the clause. If kept, state the two caveats (column-duration v, not registry v; RAW run 71, not the zero-suppressed rows). Not shown in Paper I currently, so the promise is unfulfilled.

## 14. Stale numbers in the abstract note, `papers/paper1/main.tex:49-51`

| text | now | source |
|---|---|---|
| X/Y charge balance 0.49/0.53 | A 0.485, B 0.513 (f = Q_X/(Q_X+Q_Y), waveform charge, unsaturated; 0.49/0.51 rounded) | `rerun/paper1_charge/results/summary.json`; sheet row 1 (0.53 was the old hits proxy; the "0.53" was chamber B's old 0.531) |
| no dead time after a spark | holds: A 0.2 +- 0.6 ms, 95 % UL 1.2 ms; B-E ULs 2.1 / 2.6 / 2.1 / 15.5 ms | `rerun/paper1_ops/README.md` Verdict 3 |
| intrinsic resolution 176 um (0.30 x pitch) in the SPS beam | **chamber E, working region R (X 150-160 x Y 222-246 mm, run 63 flat700): X 138 um (p_mid; telescope-f range 112-153, stat +-7), Y 120 um (p0; 89-136, stat +-9)**; model-free upper bounds X <= 153 +- 6, Y <= 145 +- 6 um. 120-138 um = 0.15-0.18 x the 0.78 mm pitch. The 176 um was run 53 Y (hits); 190/209 um were whole-spot run 63 values. 0.30 x pitch was the back uRWELL's resolution (176/780 = 0.23). Decided by Dylan 2026-11-01. | `rerun/sps_region/README.md`; `rerun/paper1_spsfigs/results/figs_summary.json` (`sps_resolution`); sheet decisions A+B; `08_tracking.tex:21-23` caption already says 120-138 |
| cosmic angular resolution 1.15-2.51 deg | **0.90-2.41 deg** over all five chambers and both views (pass arm; s68, no clip: A 0.99/0.90, B 1.05/1.13, C 1.26/1.51, D 1.23/1.46, E 2.41/2.08); chambers A-D only 0.90-1.51 deg; the v(t) arm gives 0.90-2.41 as well (E Y 2.01). The sheet's 0.90-2.41 uses A's 0.901 (Run 2 row) / 0.899 (Runs 1+2), i.e. "0.90". | `rerun/paper1_res/PASS1_RESOLUTION_2026-10-31.md` Verdict 1; sheet row "cosmic angular resolution" |

## 15. Five-chamber summary table (draft)

Sources: HV `03_setups.tex` table / `rerun/pass_registry.csv`; efficiency `rerun/effbudget/PASS1_RESULT_2026-10-31.md`, `05_efficiency.tex:33`, `A1_loss_categories.tex:10`; angles `rerun/paper1_res/results/rows.csv` and `angle_resolution.json` (pass arm, no clip, interior_good); position `08_tracking.tex:13-17` and `rerun/sps_region/README.md`; time `rerun/paper1_timing/results/golden.csv`, `07_timing.tex:11`; dish `rerun/model_audit/followups/fb_bow_maps/results_final/summary_paper.json` (the 2-D fit the text of `08b_bow.tex:19-20` quotes; not the pass-1 per-view numbers of the sheet); water `rerun/paper1_gas/README.md:19-23`; current `rerun/paper1_ops/currents/README.md`; f `rerun/paper1_charge/results/summary.json`.

| | A (det3) | B (det2) | C (det6) | D (det7) | E (det4) |
|---|---|---|---|---|---|
| long-run HV resist / drift | 490 V / 1000 V | 495 V / 1000 V | 495 V / 700 V | 495 V / 700 V | 495 V / 600 V (bench) |
| efficiency, interior, standard definition (R = 5 mm, production pick) | 92.3 % (Runs 1+2; Run 1 92.6, Run 2 92.2) | 90.3 % (o22 row; quoted row 90.9) | 74.4 % | 68.0 % (X ch 256-511; 62.9 % all X) | 40.9 % (71.3 % in live bands) |
| efficiency at scan optimum (HV) | 93.8 % (470 V) | 92.2 % (480 V) | 85.8 % (480 V) | 89.8 % (430 V) | n/a |
| angular resolution s68 X / Y (deg) | 0.99 / 0.90 (Runs 1+2: 0.987/0.899) | 1.05 / 1.13 (quoted row; o22 row 1.03/1.08) | 1.26 / 1.51 (in-sample) | 1.23 / 1.46 (in-sample) | 2.41 / 2.08 (in-sample) |
| intrinsic position resolution X / Y (um) | 128 +- 33 / 113 +- 36 (bench, reference subtracted; quoted ~120 +- 60) | 163 +- 29 / 176 +- 21 (bench) | n/a | n/a | 138 / 120 (SPS, region R; upper bounds 153 / 145) |
| event time vs trigger; uncorrelated (t_X-t_Y)/sqrt2 (ns) | 41.6, 40.3 (Run 1, 2); 22.4, 22.2 | 41.7; 27.6 | 78.1; 52.5 | 45.6; 25.5 (433 triggers) | n/a (bench 6-24 carries a 40-60 ns trigger term; SPS not measurable) |
| dish depth at 150 mm (mm) | 1.31 +- 0.22 | 1.87 +- 0.24 | 1.76 +- 0.18 | 1.68 +- 0.23 | 1.69 +- 0.22 |
| water fraction of the long run (%) | 0.97 [0.95, 0.99] (drift scan; Run 1 0.99, Run 2 0.94-0.96 by ladder) | 0.69 [0.63, 0.75] | 1.00 [0.96, 1.04] | 0.48 [0.45, 0.51] | 0.42 [0.40, 0.45] (at 30.48 mm gap) |
| dark current at 490 V (uA) | 0.29-1.14 night to night (1.05 on the scan night) | 1.0-2.4 | 0.45 (0.78 at 495 V) | 6.0-6.4 | 0.58 (6-23); 0.12 at 495 V on 6-24 |
| charge fraction f = Q_X/(Q_X+Q_Y), median (spread) | 0.485 (core 0.014) | 0.513 (0.017) | 0.504 (0.028) | 0.485 (0.027 core; 0.116 s68) | 0.509 (0.024) |

Notes on the table:
- Uncertain cells: D's efficiency at 430 V (half-X restriction unverified, 147 rays); C and D time rows come from 366 and 433 triggers; E's bench event time in `golden.csv` is 110.9 ns with a 41.7 ns trigger term and is deliberately not quoted (sheet "Stated limitations"). A's chamber-term position (128/113) is the `08_tracking.tex` value; the sheet has 126/116 +- 35.
- Dish depths: the sheet's pass-1 1-D per-view numbers (A 1.24, B 1.98, C 1.50, D 1.46, E 1.90 mean of X and Y; per view 0.90-2.45 mm) are a different estimator; the text of `08b_bow.tex` and the figure `bow_profiles` use the 2-D fit above.
- Water of A is for the drift scan at one water for 500-1100 V; the golden runs of the other chambers are single-field values at the row's field (B 321, C 250, D 238, E 197 V/cm).
- C's f is from a 92 %-saturated sample (f_sat 0.92) and D's from 14 % unsaturated events; C's and D's charge quantities beyond f are not measurable (`paper1_charge` README).
