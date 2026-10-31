# Paper I numbers on pass 1 (2026-10-31)

Every number that `papers/paper1/*.tex` quotes or keeps a placeholder for, set against the from-scratch pass
(`Analysis_pass1`, code `6a0bb1f`; SPS `3ac0e79`, re-run rows `c36322ec`/`48244778`: `rerun/sps_junk`), with its source. Then the INFN–IN2P3 abstract and the MPGD
proceedings. **WRONG** = the text's number no longer holds; **CHOICE** = it depends on a decision listed at the end;
**OK** = it holds; **NOT RE-DERIVED** = nothing on pass 1 checks it. The .tex files were not edited.

**Counts, by table row (a row can carry several numbers). Paper I: 31 WRONG (3 of them also hang on a choice),
3 CHOICE, 17 OK, 12 NOT RE-DERIVED, 2 new numbers with no text yet. INFN–IN2P3 abstract: 2 WRONG, 1 CHOICE, 2 OK.
MPGD proceedings: 7 WRONG, 2 CHOICE, 4 OK, 1 not checked.** Two of the WRONG rows are wrong independently of pass 1:
"176 µm = 0.30 × pitch" (it is 0.23 × the 0.78 mm pitch; 0.30 × pitch is the back uRWELL's resolution), in the
Paper I abstract, §8 and the INFN source note; and the MPGD "M3 pointing ~0.4 mm" (the known units trap).

Sources (main, 2026-10-31):

| short | file | commit |
|---|---|---|
| EFF | `rerun/effbudget/PASS1_RESULT_2026-10-31.md`, `rerun/effbudget/pass1/results/a1.json` | 87bb8b92 |
| CHG | `rerun/paper1_charge/README.md` | 5a199061 |
| RES | `rerun/paper1_res/PASS1_RESOLUTION_2026-10-31.md`, `results/summary.json`, `results/bow/summary.json` | 0cef1f00 |
| SPS | `rerun/sps_audit/README.md` (final SPS pass) | 322d8574 |
| VT | `rerun/VT_PLAN_2026-10-30.md`, `rerun/vt_all/CROSSCHECK_2026-10-31.md` | 514f7aac, 5ceffe40 |
| S4 | `rerun/calib/S4_PHYSICS_INPUTS_2026-10-30.md` (incl. the follow-up and re-check sections) | 14fd76b2, fd307195, 14ef0a88 |
| BOW | `rerun/calib/bow/results_pass1/GATE_RESULT_2026-10-30.md`, `bow_finalize.json` | 404672c7 |
| ABL | `rerun/calib/DET4Y_ABLATION_2026-10-30.md` | 4ee73a3e |
| FD | `rerun/model_audit/followups/fd_floor_realism/REPORT.md` § Pass-1 update (toy re-run on the pass-1 bundles) | c144af21, toy re-run in the commit after it |

## Paper I

### Abstract (`main.tex`)

| text | pass 1 | status | source |
|---|---|---|---|
| X/Y charge balance 0.49/0.53 | 0.485 (det3) / 0.513 (det2) → "0.49/0.51" | **WRONG** (det2) | CHG |
| intrinsic resolution 176 µm in the SPS beam | run 63 Y 190 ± 10 (p0) / X 209 ± 11 (p_mid); run 53 Y 176 ± 10 (hits) | **CHOICE** (A, B) | SPS §2 |
| "(0.30 × pitch)" | 176 µm / 0.78 mm = **0.23 × pitch**; 0.30 × pitch is the back **uRWELL's** resolution (0.301 × its pitch) | **WRONG** (independent of pass 1) | SPS, sps-spatial-resolution memory |
| cosmic angular resolution 1.15–2.51° | 0.90–2.41° (s68, interior_good, golden rows, both views; v(t) arm 0.90–2.41) | **WRONG** + **CHOICE** (C, D) | RES §Verdict 1 |
| no dead time after a spark | not re-measured on pass 1 (spark_deadtime figure stale) | NOT RE-DERIVED | FIGURE_PROVENANCE |

### §2 Design

| text | pass 1 | status | source |
|---|---|---|---|
| efficient area 399.4 × 359.9 mm² | not a pass product (active-area study, `common/mx17_active_area.py`) | NOT RE-DERIVED | — |
| ~19 mm dead band at each Y end | the fiducial uses 8 mm from the passivated Y edges (`common/mx17_fiducial.py`); the 19 mm band itself is not re-measured | NOT RE-DERIVED | EFF |

### §4 Operation

| text | pass 1 | status | source |
|---|---|---|---|
| chosen points 480/480/440 V | the golden (quoted) rows run at 490 V (det3) and 495 V (det2, det4, det6, det7) | **WRONG** (or the text means something else: check) | `rerun/pass_registry.csv` |
| spark rate Poisson, muon-induced 4–6×, edges | not re-derived; pass 1 adds: det6/det7 lose 17 % / 24 % of interior tracks to whole-plane discharges at 495 V, rising steeply with HV | NOT RE-DERIVED (+ new number) | EFF |
| gain curves (no number in text) | ×2 every 16.6 (det2), 16.7 (det3), 19.1 (det6), 17.9 V (det7); curved; det4 has no bench gain curve | new | CHG |

### §5 Charge balance and efficiency

| text | pass 1 | status | source |
|---|---|---|---|
| f median 0.487 (det3) | 0.485 | OK (rounds the same) | CHG |
| f median 0.531 (det2) | 0.513 | **WRONG** | CHG |
| σ68 = 0.07 | s68 0.014–0.032 (det7 0.116) | **WRONG** (it was the hits proxy) | CHG |
| "three charge proxies agree" | only the waveform charge was measured on pass 1 | NOT RE-DERIVED | CHG |
| efficiency within 5 mm, det3 93 % | 92.6 ± 0.3 % (interior_good) | OK (rounds to 93) | EFF |
| det2 92 % | 90.3 ± 0.5 % | **WRONG** | EFF |
| det4 42 % | 40.9 ± 0.5 % (71.3 % in the live bands) | **WRONG** | EFF |
| det6 75 % | 74.4 ± 0.5 % | OK (rounds to 74; within 1 pt) | EFF |
| det7 57 % | 68.0 ± 0.7 % (X on ch 256–511); 62.9 % with all X | **WRONG** | EFF |

### §6 Gas

| text | pass 1 | status | source |
|---|---|---|---|
| det3 drift scan, six fields 35–382 V/cm | seven fields, 33–367 V/cm (drift 100–1100 V over 30 mm) | **WRONG** | RES §3 (gas_vE) |
| v = 2.5–37.8 µm/ns | 2.55–38.8 µm/ns, angle-scale v (v_drift × kw, not a timing v; ~5 % above Magboltz everywhere); timing v of det3 at 333 V/cm 36.03 ± 0.34 | **WRONG** (and say which v) | RES, S4 re-check |
| 0.95 % water, ≲ 10 ppm O₂ (bench) | not re-derived on pass 1 (the SPS/bench gas fit is a separate session) | NOT RE-DERIVED | — |
| beam run_71: 1.51 % water, 163 ppm O₂; run_63 230 → 147 ppm | not re-derived (gas-fit session) | NOT RE-DERIVED | — |
| gap topography, bench det3 spread 1.5 mm | det3 gap 28.19 ± 0.19 mm (interior), input 28.08 ± 0.19 holds; the spread is not re-measured | OK (mean) / NOT RE-DERIVED (spread) | S4 re-check |

### §7 Timing

| text | pass 1 | status | source |
|---|---|---|---|
| hits value 33 ns, not quoted | still not quoted; at SPS the time resolution is not measurable (t0_X − t0_Y 180–390 ns; leading edge 130–210 ns X−Y, ~400 ns vs trigger): a stated limitation | OK (as a non-quote) | SPS §4 |

### §8 Tracking

| text | pass 1 | status | source |
|---|---|---|---|
| σθ = 1.15–2.51° | 0.90–2.41°: det2 1.045/1.125, det3 0.991/0.901, det4 2.412/2.076, det6 1.256/1.506, det7 1.232/1.461 (pass); v(t) arm det4 Y 2.013, det6 1.211/1.454, det7 1.225/1.449 | **WRONG** + **CHOICE** (C, D) | RES |
| (G1, if quoted) det4 Y 2.35° | 2.348 (pass) / 2.290 (v(t) arm), all events; the −0.25° vs the audit is the template, not the bow | **CHOICE** (C) | ABL, VT |
| X asymmetry a_Q det3 +0.165 ± 0.019, copy 0.13×, 110 ns; det2/det4/det7/det6 +0.08/+0.08/+0.19/−0.02; K = −0.77 ± 0.13; ≤ 44 / ≤ 116 µm | not re-derived on pass 1 (`wft.xasym` ± systematic has no script yet, PASS_PLAN §4) | NOT RE-DERIVED (7 numbers) | — |
| intrinsic 176 ± 10 µm | see the abstract row | **CHOICE** (A, B) | SPS |
| = 0.30 × pitch | 0.23 × pitch for 176 µm (0.24 for 190, 0.27 for 209); 0.30 is the telescope | **WRONG** | SPS |
| bench 0.6–0.7 mm = M3 pointing ⊕ ~1.1 mrad MS | σ vs M3 with the pointing subtracted 348–663 µm per view (det2 456/444, det3 447/443, det4 663/526, det6 348/382, det7 434/488), MS still in → upper bounds; the 1.1 mrad arithmetic is not redone | **WRONG** (range) | RES §Verdict 2 |

### §8b Bow

The estimator is unchanged (fb_bow_maps), re-run on the pass-1 golden tables (`bow_pass1.py`). The v(t) arm is what the
paper figures show.

| text | pass 1 (v(t) arm; pass arm where different) | status | source |
|---|---|---|---|
| dish 1.5–2.3 mm below a point 150 mm out | per view 0.90–2.45 mm (½ h″ · 150 mm²); per-chamber mean of X and Y 1.24–1.98 mm | **WRONG** | RES bow |
| det2 1.8 ± 0.2 mm | X 1.52, Y 2.45, mean 1.98 | **WRONG** | RES bow |
| det3 1.5 ± 0.2 | X 0.90, Y 1.57, mean 1.24 | **WRONG** | RES bow |
| det4 1.9 ± 0.2 | X 1.69, Y 2.10, mean 1.90 | OK | RES bow |
| det6 1.8 ± 0.2 | X 0.95, Y 2.04, mean 1.50 | **WRONG** | RES bow |
| det7 2.3 ± 0.4 | X 1.22, Y 1.71, mean 1.46 | **WRONG** | RES bow |
| curvature 1.3–2.1 × 10⁻⁴ mm⁻¹ | per view 0.80–2.17 × 10⁻⁴ (X 0.80–1.50, Y 1.40–2.17) | **WRONG** | RES bow |
| w0 ramp ±0.6 µm/ns over 300 mm, all ten views | ±0.41–0.95 µm/ns (X ±0.41–0.64, Y ±0.54–0.95); rises on all ten views | **WRONG** (size) | RES bow |
| r_f = 0.4–0.9 on nine of ten views | 6/10 in 0.4–0.9; 8/10 in the physical ½–1 range; det3 X 1.11 and det6 X 1.91 outside | **WRONG** | RES bow |
| the fitted slope has the sign of the M3 slope on every chamber | 245/247 (row, view) pairs; the two exceptions are the excluded det2 6-17 | OK | CHG §sign |
| M3 z up; slots 702 / 232 mm | unchanged (external facts) | OK | CHG §sign |
| stability: 15/19 within 2σ; det3 6-25 0.9 ± 0.1 vs mean 1.5; 94 % of the gain transfers | not re-run as such; the pass-1 gate: one block per chamber improves **every** row incl. det3 6-25 (−0.17/−0.23°); strict arm worse on other-day rows | NOT RE-DERIVED (the gate replaces it) | BOW |
| angle improvement median 16 % (1–30 % per view) | median 15.7 %, 3.6–28.9 % (28 row-views, pass arm vs no bow) | **WRONG** (range; median OK) | BOW |
| final settings: Y improves 13–25 %; −8 % compression on det6 X removed | golden rows Y 13.9–25.2 %, X 3.6–28.9 % (det6 X 28.9) | OK (Y); compression not re-measured | BOW |
| h tanθ improves position by 4–15 % | 1.7–10.6 %, median 4.2 % | **WRONG** | BOW |
| plate estimate 0.1–0.25 / 0.3–0.7 / ~1 mm | analytic, unchanged | OK | — |

### §9 det4 stripes

| text | pass 1 | status | source |
|---|---|---|---|
| 62 % of det4's area does not amplify | the A1 class "dead / low-gain stripes" is 34 % of interior M3 tracks (interior_good, g_det4); the 62 % area fraction is from the SPS assessment, not re-derived; the dead-stripe map is the pre-pass one | NOT RE-DERIVED (34 % is a different quantity) | EFF |

### Appendix A1 (det3, sat_det3, interior_good, n = 6 454; text: 6 451, 91.18 % detected)

| class | text % | pass 1 % | status |
|---|---|---|---|
| detected | 91.18 | **92.64** | **WRONG** |
| n M3 tracks | 6 451 | 6 454 | **WRONG** (trivial) |
| noise band | 1.43 | 0.22 | **WRONG** (band censoring is in the chain) |
| second particle, M3 one fitted | 1.01 | 1.02 | OK |
| second particle, other fitted | 1.26 | 1.43 | **WRONG** |
| merged in projection | 0.57 | 0.57 (multi:m3_not_found) | OK |
| spark: discharge / muon + other / shower | 0.50 / 0.40 / 0.67 | 0.50 / 0.40 / 0.67 | OK |
| δ-ray on the track | 0.93 | 0.90 | within 0.03 (OK) |
| saturated δ blob | 0.81 | 0.84 | within 0.03 (OK) |
| fit off | 0.34 | 0.23 | **WRONG** |
| head-on, never seeded | 0.26 | 0.02 (fit_failed, head-on) | **WRONG** (the dead-seed bridge / head-on handling) |
| position tail | 0.22 | 0.20 | OK |
| edge of the other coordinate | 0.19 | 0.15 | **WRONG** (small) |
| second deposit, not particle-like | 0.17 | 0.19 | OK |
| other | 0.08 | 0.05 (cut 0.02, charge elsewhere 0.02, fit_failed 0.02) | OK |
| satblob recovery 71.1 → 73.7 %; 93 % | not re-run on pass 1 (satstudy) | NOT RE-DERIVED |
| det2/det4/det6/det7 columns | available (EFF a1.json: det2/det3 ≤ 1.7 % per class; det4 stripes 34 %; det6/det7 discharges 17 / 24 %) | new (decision E) |

(Each A1 class is one row in the counts; "WRONG (small)" and "WRONG (trivial)" count as WRONG.)

## INFN–IN2P3 abstract (`papers/infn_in2p3_2026_abstract/abstract.md`, v2, not submitted)

| text | pass 1 | status |
|---|---|---|
| angular resolution 1.2° | det3 headline 0.99 / 0.90° (s68, interior_good) → "1.0°" or "0.9–1.0°" | **WRONG** |
| 93 % efficiency | 92.6 % (det3 golden), 92.2 % (det3 quoted golden row) | OK |
| spatial resolution 180 µm | run 63: Y 190 ± 10 (p0), X 209 ± 11 (p_mid); run 53 Y 176 ± 10 | **CHOICE** (A, B) |
| source note "176 µm, 0.30 × pitch" | 0.23 × pitch | **WRONG** |
| angles "about 4° too steep" from hit times | from RECONSTRUCTION_BASIS (hits), unaffected by pass 1 | OK |

## MPGD2026 proceedings (`papers/mpgd26_proceedings/main.tex`, deadline 17 Jan 2027, not submitted)

| line | text | pass 1 | status |
|---|---|---|---|
| 29–31, 244 | best chamber 93 %, ~1.2° | 92.6 % OK; angle 0.99/0.90° on det3 → ~1.0° | 93 % OK, **1.2° WRONG** |
| 31 | 176 µm in a beam | run-63/run-53 choice | **CHOICE** |
| 122–123 | M3 pointing ~0.4 mm at the chamber | 0.21–0.28 mm per axis (0.40 was the deconvolved DUT value: the known units trap) | **WRONG** |
| 126 | 93.1 % of 21 953 muons within 5 mm | det3 quoted golden row 92.2 ± 0.2 % of 20 088 (interior); sat_det3 92.6 % of 6 454 | **WRONG** |
| 127–130 | rest: 4.2 % far, 2.6 % discharge, 0.2 % no x–y, 3 muons no signal | A1 on pass 1: discharge-type sparks 0.50 % + shower 0.67 % + muon+other 0.40 %; far classes ~2.2 %; the old split is not the pass's | **WRONG** (4 numbers; one entry) |
| 130–131 | other chambers 92 % down to 42 % | 90.3 % (det2) down to 40.9 % (det4); det6 74.4, det7 68.0 | **WRONG** |
| 134 | σ68 = 1.19 (x) / 1.23° (y), bias < 0.1° | det3 0.991 / 0.901°; det2 bias 0.11° X | **WRONG** |
| 135 | ≈ 1° above ~5° | det3 flat within 0.15° to 20° at ~0.9–1.0° | OK |
| 137–138 | plane-to-plane timing 33 ns, 1.1 mm | hits-based; Paper I does not quote it | **WRONG** for consistency with Paper I (remove or label hits) |
| 140 | 176 ± 35 µm at normal incidence | see the choice; the ±35 has no source in the final SPS pass (±10 stat, ±40 µm telescope systematic) | **CHOICE** |
| 88–89 | first neighbour ~5 % / ~15 % | not a pass-1 product (the kernel is the pass calibration's; footprint at SPS: neighbour/lead 0.31 X / 0.34 Y, RAW) | not checked |
| 91–92 | ladder compressed 20–30 %, ~4° too steep | RECONSTRUCTION_BASIS | OK |
| 121 | bench drift field 333 V/cm, Ar/iC4H10 95/5 | 1000 V over 30 mm | OK |
| 181–182 | 78.6 %, 41.8e6 events, 17.9 TB | n_TOF end-of-run report, not pass 1 | OK |

## Decisions for Dylan

- **A. SPS run:** run 63 flat700 (CF4; telescope calibrated in the same run; recommended by the final SPS pass) or
  run 53 (the original 176 ± 10, Ar/CO2, Y only has all three zones).
- **B. SPS position estimator, per view:** fit p0, p_mid (p0 at 7.5 mm), charge centroid, or hits. On run 63 the best
  is p_mid in X (209) and p0 in Y (190); on run 53 the centroid/hits (155/176 Y). Up to 1.6× between estimators.
- **C. v(t) arm for the bench angles:** quote the pass table or the v(t)-corrected table. It moves det4 Y −0.06°, det6
  −0.05°, det7 −0.01°; it is always lower.
- **D. s68 convention:** 03_angles' (no ±5° clip; the headline above) or FD/G1-style with a ±5° clip. They differ by
  up to 0.75° on det4 X (2.41 against 1.66°) and 0.1° on det6 Y; ≤ 0.06° on det2/det3. Whatever is chosen, the FD
  budget only adds up in the clipped one.
- **E. A1:** a column per chamber (EFF has them) or det3 only.
- **F. Bow numbers:** per view or per-chamber mean; the pass-1 X/Y depths differ by up to 1.1 mm (det6), so a single
  per-chamber number hides it.

## Stated limitations to carry into the text

- Bench positions are upper bounds: multiple scattering between the last M3 plane and the chamber is in the widths.
- det4, det6, det7 headline angles are in-sample (their only golden row is the calibration row).
- det6's drift velocity over 6-26 is known only to a factor ~2 (ladder vs column duration); its depths carry it.
- det4 has no bench gain curve (6-23 night flat; degraded M3 matching or a gain drift); its gain comes from SPS.
- Charge: no absolute gain (fit ADC units); the 3550 rule misses 37 % (det4) / 46 % (det6) of truly clipped samples,
  which count as unsaturated and read low (K4, locked).
- SPS: tan sign not determinable (only |tan| quoted); the time resolution is not measured; the telescope term
  (f 0.25–0.34) is a ±40 µm systematic on det4; four CO2-epoch ZS rows carry 2–8 % junk Y fits (positions-only rows;
  being cleaned).
- **FD budget:** the toy was re-run on the pass-1 bundles (1500 events per angle, the data's angle mix, each row's own r(t)).
  54–88 % of the X variance is still unexplained. Interior_good, calibration rows, v(t) arm, FD convention: det2 0.72 of
  0.98°, det3 0.68 of 0.93°, det4 1.52 of 1.62°, det6 0.93 of 1.13°, det7 0.98 of 1.12°. All toy terms together are
  0.38–0.46°; the toy floor is 0.31–0.34°.
- The efficiency loss classes were tuned on det3 and not checked by eye on the other chambers; the det4 stripe map is
  pre-pass.
