# n_TOF model checks before the bundles freeze (2026-11-04)

Plan §3 P3 / §6. Bundles v1k_A, v1k_C, v1_D (t0 free, sat 3550, K 20); bench = pass-1 golden bundles (det3 sat_det3,
det6 g_det6_long, det7 g_det7_long). Report: `report.html`. Data: `/media/ucla/work/ntof_pass1/model_checks/`.

| check | verdict | bundle value |
|---|---|---|
| 1 Template | holds on A and C; on D the time shape holds, the peak misfit is footprint width; systematic 0.1° / 0.15 mm | keep the bench X template |
| 2 Saturation | keep 3550; 3700 is worse where it matters | `sat_adc` 3550; the real fix is the per-sample rail |
| 3 Dp | A measured on the stack; C and D move to Magboltz 90/10 | A 0.0245 ± 0.0014 (was 0.023), C 0.0136 ± 0.0008 (was 0.018), D 0.0157 ± 0.0009 (was 0.0225) |

## 1 Template (template_resid.py, template_toy.py)
- Unsaturated single-track X planes refitted with the pass model; residual (data − model)/model peak on the strip under
  the track, against the bench head-on cache of the same chamber with the same code.
- A max |Δ| 0.017 (n 161 vs 355), C 0.018 (n 93 vs 81) over k = −1…+6. Post-column residual ≤ 0.02 of peak; q_late
  1–2 % vs 1–3 % bench. Template tail 0.14 of peak at +300 ns, then an undershoot: little room for a gas-dependent ion tail.
- D: +0.15 at the peak (bench +0.02), 0.046 for k ≥ +1: lateral (footprint), not the time shape; +0.11 with Magboltz Dp.
  D beam planes are dominated by oscillating channels 448–451 (see tests/d_b) and cannot test the template.
- Toy: ±5 % tail ≤ 0.02°; ±3 % time stretch ≤ 0.08° / 0.15 mm; Dp wrong by 13–28 % ≤ 0.15° (D), ≤ 0.08° (A, C).

## 2 Saturation (sat_census.py, sat_compare.py)
- The rail after pedestal + CM subtraction is 4095 − ped − CM: beam blocks with a big signal have CM ≈ −290 (rail ≈ 4080),
  cosmics CM +20…+190 (rail ≈ 3650–3750).
- Clipped samples missed at 3550 / 3700: beam x A/C/D 3.3/3.4/18.6 % vs 5.9/5.1/33.4 %; beam y 11.9/12.4/11.4 % vs
  18.6/19.0/22.2 %; cosmics x 31.3/6.4/27.1 % vs 52.2/31.8/50.2 %.
- Real samples censored 3550 / 3700 (beam x A/C/D): 8057/13014/8809 vs 6033/10106/6127; 3700 buys back only ~25 %.
- Reco with both on every event with a sample ≥ 3500 (6 % cosmics, 2 % beam): 3–11 % of A/C planes and 19–27 % of D
  planes move, nearly all already saturated with χ²/dof 3–10× median. A–C line unchanged (7 events). Pointing < 15 mm:
  A 2/10 vs 2/12, C 4/31 vs 4/31, D 30/94 vs 27/96.
- Proposed (not applied): carry per-sample rails in `ntof_tracking/wft_beam.py::_windows_for_tag` so `SAT_RAIL` can be
  tested at n_TOF (bit-identical with the switch off). D (det7) is where it would count.

## 3 Dp (mb_9010.py, dp_stack.py, dp_scan.py)
- The old values are the bench 95/5 hypers copied by ntof_bundle.py (`rerun/calib/physics_inputs.json`); det6's 0.018
  was never a measurement.
- Magboltz 90/10 at 720.8 Torr, fields 249/250/238 V/cm, water fraction chosen to reproduce the in-situ v (A 0.24 %,
  C 0.67 %, D 0.49 %): D_T 320/256/281 µm/√cm → Dp 0.0199/0.0136/0.0157.
- Stack estimator closes on bench det3 (0.0229 vs 0.023).
- A: stack 0.0244 ± 0.0009 (trigger-aligned, n 149), 0.0250 ± 0.0019 (threshold, n 37); χ²-scan 0.0232 (bench 0.0223);
  the det3 ×1.2 excess over Magboltz seen on the bench too → 0.0245 ± 0.0014.
- C: window-limited stack (σ0 at its bound); χ²-scan 0.0137 (bench det6 0.0119), Magboltz-consistent → 0.0136.
- D: 25 clean planes, flat χ²; bench det7 matched Magboltz (1.03) → 0.0157.
- Re-measure kw on the A–C line after the change.

## What this does not rule out
- A template change visible only > ~500 ns after the column end (outside the 20-sample fit anyway).
- The oscillating-channel and second-deposit rules used here are cuts, not the P2 channel census.
- "A–C line does not move" under the sat rule rests on 7 events plus only saturated planes moving.
- C and D Dp assume water is the only contaminant and carry the gap errors (D ± 3.8 %).
