# Cross-section consistency audit for Paper I (2026-11-01)

Read-only on every product. Inputs: the section READMEs (`rerun/paper1_ops`, `paper1_figs`, `paper1_gas`, `paper1_timing`,
`paper1_floor`, `paper1_res`, `paper1_charge`, `effbudget`, `vt_all`, `calib/S4_*`, `sps_audit`, `sps_junk`) and the numbers
sheet `rerun/PAPER1_NUMBERS_2026-10-31.md`.

One new measurement: the M3-free gain proxy (`rerun/paper1_ops/hits_gain.py`, unchanged) on the fixed-HV rows of 6-23 and
6-22 (`condor/`, cluster 4605383; JSONs in `results/`, copy on EOS `cosmic_bench/x17rerun/consistency/hits_gain/`).
The 6-22 det3 `longer_run` has no pass-1 table, so it has no point.

## 1. The 6-23 / 6-24 night: two causes, not one

**Verdict.** Four sessions found the det3+det4 night of 6-23 odd. Their findings come from two separate effects:
- **(a) The chambers were not yet at working gain on 6-23, and recovered through the night.** Measured.
- **(b) A trigger-timing term on 6-23 and 6-24.** Measured; its cause is not identified.

The flat HV scan is (a). The common column shift is (b). The degraded M3 of 6-23 was a software bug, already fixed
by the v2 rays (LESSONS, 2026-06-24), and is not a third effect.

### (a) Gain and efficiency recovering in time (established)

**At a fixed 495 V, both chambers' charge and efficiency rise through the night.** These are M3-free and fit-free.
- p50 = the median lead-strip amplitude of MIP-like triggers.
- MIP/trig = the share of triggers that are MIP-like in both views.
- eff = interior efficiency within 5 mm (`rerun/paper1_ops/results/tradeoff.csv`).

| row (start) | det3 p50 | det3 MIP/trig | det3 eff | det4 p50 | det4 MIP/trig | det4 eff |
|---|---|---|---|---|---|---|
| 6-23 short_run (19:29) | 258 | 0.04 | 0.06 % | 163 | 0.20 | 3.9 % |
| 6-23 longer_run (20:29) | 423 | 0.16 | 3.2 % | 361 | 0.34 | 10.7 % |
| 6-23 scan, 495 V (00:39) | 619 | — | 21.3 % | 746 | — | 18.7 % |
| 6-23 long_run (03:16) | 1031 | 0.40 | 35.1 % | 1176 | 0.26 | 28.5 % |
| 6-24 g_det4 (12:22) | — | — | — | 1028 | — | 40.9 % |
| 6-22 det3, same 495 V and HV channel (19:18) | p25 2700, f_sat 0.62 | 0.86 | 92.9 % | | | |

**What the table shows:**
- **On 6-23 det3 reads about 1/10–1/17 of its 6-22 charge** at the same HV on the same channel (`hv_channels` identical in
  both `run_config.json`). Pass-1 fitted q_sum p25 is 650–750 against 11 500–13 100.
- **Within `long_run` it is still rising,** by +15 % over the four quarters. g_det4's charge also rises +14 % over its row
  (6-24), with f_sat flat at 0.21–0.22.

**Not the drift field.** 6-23 and 6-24 ran the drift at 600 V (200 V/cm). det3 at 500 V on 6-25 is at 89.6 %, and on the
6-27 drift scan at 500 V it is at 93.3 %.

**Why the scan reads flat.** The scan ran 22:30 → 02:51 in descending HV.
- **The time recovery is part of it.** Over the scan window the fixed-HV gain rises ×1.65 (det3) and ×2.6 (det4). That
  cancels about 20 % (det3) and 35 % (det4) of the ln-gain a normal 0.45 per 10 V slope predicts over 60 V.
- **The proxy itself is the rest.** Only 4–40 % of the triggers are MIP-like in both views, against 86 % on 6-22, so the
  proxy's p50 is mostly not chamber signal. The scan cannot be inverted to a gain curve. The 6-23 exclusion stands, now for
  this measured reason.

**Inferred, not established: a gas contaminant from the 6-22/6-23 chamber changes, flushing out over about a day.**
- det4 was installed on 6-23. Its `run_config` note says "Bulked by Stephan June 22 / 24 (?)".
- The pre-pass hits gas survey found det3 "water-saturated (> 3 % H₂O) until ~6-24, ~1 % from 6-25"
  (`mx_june_cosmic_qa/DET3_WEEKEND_ANALYSIS.md`). That fits det3's broken micro-TPC angle on 6-23 (v 1.1–3.5).
- The same signature, gain and efficiency recovering over hours after a new install, is measured on det6 on 6-26:
  gain ×10–25, attachment clearing in ~4 h (`rerun/paper1_gas/README.md`).
- There is no gas record, so the species and the line order are unknown.

### (b) The trigger term (established effect, unidentified cause)

- **Size:** cov(start, end) of the column, which is +σ_trig² for a common trigger shift:

  | night | rows | cov(start, end) |
  |---|---|---|
  | 6-23, det3 | 4 | +3 916 ns² |
  | 6-23, det4 | 14 | +4 356 ns² |
  | 6-24 (g_det4) | 1 | +3 903 ns² |
  | 6-22 | — | −93 to +518 |
  | 6-25 | — | −67 |
  | 6-26 | — | −1211 / −248 |
  | 6-27 | — | −110 |
  | 6-27 drift scan, 300–1100 V | — | −2 588 … +158 |

  (`rerun/paper1_timing/results/rows*.csv`)
- **It is not (a):** it persists on 6-24, when det4's gain and efficiency are back to normal.
- **Attachment doesn't produce it:** det6's attachment-heavy early 6-26 rows read negative.
- **It is not the field:** low-field rows elsewhere read ≤ 0.
- **The only DAQ-config difference unique to 6-23 and 6-24** (`run_config.json`, all five nights compared):
  - no fresh pedestal/threshold run (`do_pedestal_threshold_run False`, pedestals "latest" from disk);
  - the pre-6-26 config, which has no `trigger_feu`, `included_feus` or `set_feus_from_detectors`. These appear from 6-26.

  6-22 took its own pedestal run; 6-26 and 6-27 used disk pedestals but with the new keys.
- That this configuration caused it is an inference. The bench paddle and discriminator settings are not recorded.

### Consequences for numbers that use g_det4 (6-24) or the 6-23 rows

| quantity | consequence |
|---|---|
| det4 bench gain curve | none on the bench, for the measured reason (a) above; det4's gain comes from SPS |
| det4 bench timing (111 ns) | contains the 40–60 ns trigger term (b): not quotable (as §7 already says) |
| det4 bench position (663 µm X / 526 Y, upper bounds) | carries tan · v · δt from (b) (slope 6.3 ± 0.6 µm/ns measured, as predicted); its size in the width is not established. Quote det4's resolution from SPS only. |
| det4 angle (2.41 / 2.08°, G1 2.35°) | a common time shift moves p0, not the slope: the angle is unaffected by (b). The +14 % in-row charge rise from (a) is within the amplitude spread every row has. No change. |
| det4 calibration on g_det4 (template, v, kw, bow, σ0) | v: averaging over the −9.6 % in-row drift moves T by −0.9 ns (S4), negligible. Template: the DREAM stacks are not pulse-aligned, so a 40–60 ns trigger jitter widens the template in quadrature: ≲ 2–3 % of a ~300 ns pulse (estimate, not measured). Flag; the det4 ablation's −0.233° template effect is the pass vs audit template on the same events, so it is unaffected. |
| det4 efficiency 40.9 % (g_det4) | g_det4 is ≥ 9 h after 6-23's recovery, at the plateau of the 6-23 → 6-24 trend (28.5 → 40.9 %); the in-row charge still rises +14 %. State it; no change. |
| §6 det4 water "0.70 → 0.21 % through 6-23" | the stack inversion has no attachment term, and 6-23 was not attachment-free. The 0.70 is the first third of short_run (T 1032 ± 451 ns). Quote the night only with the caveat, or drop it; the 6-24 value (0.42 %) is clean. |
| v(t) det4 6-23 (+7.6 %) | the ladder (+8.3 %, attachment-blind) and the column (+6.3 %) agree, so the v drift is real; its gas reading carries the caveat above. |
| det3 6-23 rows | already skeptical; now explained (a). |

## 2. Number-level consistency

| quantity | values (source) | consistent? | explanation / fix |
|---|---|---|---|
| det2 edge T (edge zone − interior) | −86 ± 33 ns (S4 re-check, head-on stack, mask) vs −4 ns (FIGS, column duration, all triggers) | **yes (definitional), closed 11-01** | the two "edge zones" are different bands of a non-monotonic T(d): S4's out30 is the M3 reference < 30 mm from ANY side, and 40 % of it sits in the outer 0–5 mm where the column runs off the strips (column T ≈ 584 ns), 69 % nearest a y side. Weighting the edge study's own column-duration profile T(d) with the S4 sample's distance distribution predicts −78.5 ns, against S4's −86 ± 33. `rerun/paper1_spsfigs/det2_edge_check.py`, `results/det2_edge_check.json`. det3 (+40 vs +26) and det4 (+60 vs +78) agree as before. Quote edge T only with its band definition. |
| golden vs optimum efficiency | EFF golden 90.3 / 92.6 / 40.9 / 74.4 / 68.0 vs OPS optima 92.2 / 93.8 / – / 85.8 / 89.8 | yes | different HV: the golden rows ran 10–65 V above the optima (decision G). OPS reproduces EFF on the same rows (o22 90.31, sat_det3 92.64, g_det4 40.92, g_det6 74.43, g_det7 68.04). |
| det3 headline efficiency | 92.6 (sat_det3, 6 454 tracks) vs 92.2 (6-27 sanity row, 20 088) | yes | two golden rows; pick one per text (the sheet's INFN row lists both). |
| det6/det7 discharge losses | OPS f_in_discharge 16.77 / 24.69 % vs A1 split 16.73 / 24.18 %; OPS loss_spark 18.27 / 25.97 vs A1 one-class 18.13 / 24.95 | yes (definitional) | OPS counts tracks in a discharge/vetoed trigger; A1 assigns each lost track to one class. Quote A1's in the loss table, OPS's for "tracks in a discharge trigger". |
| bench water | §6 golden rows det2 0.69, det4 0.42, det6 1.00, det7 0.48 %; 6-26 night curves det6 1.47 → 0.76, det7 0.68 → 0.35; det3 0.97 % (drift scan) | yes | golden rows fall on their night's curve at their time (det6 golden 06:00–10:20 between 1.47 and 0.76). Pre-pass hits survey (det6/det7 1.19/0.73, det4 0.76–0.83) is superseded. |
| v per chamber | S4 re-check (interior stack) det2 38.25, det3 36.03, det4 32.70, det6 24.78; §6 stack on the same rows identical; ladder/timing ratio det6 0.970, det7 0.984 | yes | S4's first-pass det4 31.34 is superseded by 32.70 (masks + interior); no sheet row quotes 31.34. |
| g_det4 in-row v drift | −9.6 % (ladder), −8.5 % (column) vs §6 stack thirds 33.1 / 32.2 / 33.2 ± 1.4–3.0 µm/ns | not contradictory | the thirds' errors (±4–9 %) cannot see a 9 % slope. Unresolved, as §6 says. |
| angle s68 | 03_angles (no clip): det3 0.991 / 0.901°, det4 X 2.41° vs FD/G1 (±5° clip): det3 0.93°, det4 X 1.66° | yes (definitional) | decision D. |
| det4 dead fraction | 62 % (old §9 text, hits) vs 52 % of the interior x range below half the X plateau (FIGS) vs 34 % of interior M3 tracks lost to "stripes" (A1) vs 41 % two-view detection, 71.3 % in live bands (EFF) | yes, once defined | tracks in sub-half-plateau regions are partly still detected at low gain (the stripes are a gain continuum, FIGS) or lost to other classes. Sheet §9 row amended with the reconciliation. |
| bench vs SPS position | bench chamber term near normal det2 163/176 ± 25, det3 126/116 ± 35 µm (± ~50 from P) vs SPS det4 run63 X 209 ± 11 / Y 190 ± 10 µm, run53 Y 176 ± 10 | yes | different chambers (det4 is the fleet's weakest) and both carry a reference systematic (P ±50 µm bench; telescope f ±40 µm SPS). |
| det4 drift gap (added 2026-11-01) | SPS gas fit 27.7 ± 0.1 stat (quoted 27.9, ± 0.9 shaper) vs bench input 30.48 ± 0.44 (`physics_inputs.json`, X-view `gap_extent`) vs geometric 28.6 ± 0.5 (bench Y view, no reference; SPS rot25 plateau at the ZS floor 28.1–28.9, a lower bound) | **resolved: the gap is ~28.6** | `rerun/det4_gap/README.md`. The gas fit's gap is within one profile step; its water at 28.8 mm is 0.10–0.19 % lower, inside its gap term. The bench input is 6 % long, on det4 only: its X view crosses the dead stripes (working explanation). Moves det4's timing v (−6 %), the field labels and the §6 det4 water; not the angles. Not changed in `physics_inputs.json` (decision for Dylan). |
| timing HV dependence | sheet "det3 102 → 41 ns (425 → 500 V)" vs §7 report "83 → 21 ns" | yes | the first is the event time, the second the per-view X−Y/√2 part (`rerun/paper1_timing/README.md` l. 45, 49). |
| SPS Y junk | sheet limitation "four CO₂-epoch ZS rows carry 2–8 % junk Y fits (being cleaned)" vs sps_junk: all 30 re-run rows ≤ 0.55 %, run71 RAW re-run | **stale → fixed** | sheet limitation updated. |
| A1 class audit / det4 stripe map | sheet limitation "not checked by eye on other chambers; stripe map pre-pass" vs FIGS: audited (rules hold), stripe map re-derived on pass 1 (bench vs SPS agree 82–85 %) | **stale → fixed** | sheet limitation updated. |
| det4 no bench gain curve | sheet "(6-23 night flat; degraded M3 matching or a gain drift)" vs OPS (M3-free proxy flat too) and §1 above | **stale → fixed** | sheet limitation updated to the measured reason. |

**Counts:** 13 quantities checked: 10 consistent (6 of them only once the definitions are stated; det2's edge T closed
11-01, see its row), 1 not contradictory (g_det4's in-row v), 3 stale sheet entries fixed. No product is wrong. The §6 det4 6-23 water
needs a caveat (attachment-free assumption) and the §5 edge-T row a det2 caveat; both are added.

## Still to fold in (running when this was written)

- **The SPS gas fit** (`rerun/sps_gasfit`). The bench-vs-SPS gas comparison (§6 item 3) is provisional until it lands.
  *Landed (4127f9ea).* Its det4 gap (27.7–27.9) was checked geometrically on 2026-11-01: det4's gap is 28.6 ± 0.5 mm
  (`rerun/det4_gap/README.md`, row in the §2 table).
- **The residual bootstrap of the angle floor** (`rerun/paper1_floor/BOOTSTRAP_*`).
- **Provenance is stale for four figures.** `papers/FIGURE_PROVENANCE_2026-10-31.md` still lists five figures as
  "pending-sps":
  - `sps_resolution`, `kernel_forms`, `angled_kernel` and `known_angle` can now be built, since the SPS final pass and
    sps_junk are done;
  - `attachment` waits for the SPS gas fit.

  Nobody has built them yet.
