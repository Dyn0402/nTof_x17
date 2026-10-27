# ws_dead_charge — F12 (dead lists across datasets) and F15 (x–y charge balance), 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Commit 2117912.)

## F15 — not dead strips, not missing normalisation
- Gain cancels in qx/qy (shared amplification). Per-connector X/Y gains (split-half): equal to ±2 % (≤ ±5 %); applying
  them moves the width ≤ 0.004; ln(qx/qy) map in 25 mm cells flat to 0.02–0.03. No equalisation key added (no effect);
  the bundle's per-channel gain slot exists.
- Censoring restores dead-strip charge on average (median +0.01…+0.06, det7 −0.09) but censored windows are 1.3–3× wider.
- Width cascade (rσ of ln qx/qy):

| | all fitted | M3, both views | no censored strip | unsaturated | + connector gains |
|---|---|---|---|---|---|
| det3 | 0.060 | 0.056 | 0.055 | 0.051 | 0.050 |
| det2 | 0.075 | 0.068 | 0.065 | 0.055 | 0.054 |
| det4 | 0.126 | 0.098 | 0.087 | 0.078 | 0.079 |
| det6 | 0.196 | 0.174 | 0.165 | 0.104 | 0.105 |
| det7 | 0.322 | 0.193 | 0.125 | 0.069 | 0.068 |

  Drivers: wrong x–y pairing (det7 0.32 → 0.19 on M3 both-view), saturation (16 % det3, 30 % det2, 69 % det6/det7),
  censored windows. Residual on det6/det7 tracks the X–Y χ²/dof difference (r 0.44–0.46) → model side (F4/F18 family).
- For H: charge pairing needs a width per event class (saturated / censored), not one number per chamber.

## F12 — dead lists in strip position across 17 bench rows, 3 SPS flat sets, n_TOF run_55/79/145
Methods: census (M3/uRWELL crossing), new reference-free **flank** test (silent strip between two > 300 ADC
neighbours), **silent** test (< 5 % of neighbourhood occupancy). Two fault classes, consistent across datasets:
- **Chamber-side** (follows the chamber across FEUs): det6 X 0.8–18 mm alternate, 150–186, 342–350; det7 X conn 0–3
  (reappears at n_TOF D on FEU 1 → not bench seating); det6/det7 Y 40.6–43.7; det2 X 7.0, 53; det4 X dead stripes
  (confirmed at SPS).
- **Electronics-side** (follows an FEU channel): bench FEU 6 ch 269 (det2/4/7); FEU 4 ch 50 (det3, det6, n_TOF A);
  FEU 7 ch 307/398/400.

| chamber view | chamber-side | electronics | single-dataset | golden mask holds | chamber-side missing from mask |
|---|---|---|---|---|---|
| det2 X | 5 | 3 | 280 (ovn_det2, thin) | 7 (4 chamber) | 1 |
| det3 X | 0 | 6 | 65 (n_TOF A conn 8) | 3, all FEU-7 electronics | 0 |
| det4 X | 16 | 3 | 66 | 77 (16 chamber, 58 g_det4-only) | 0 |
| det6 X | 36 | 1 | 13 | 24 (21 chamber) | 15 (177–185 mm block) |
| det7 X | 65 | 16 | 62 | 114 (47 chamber) | 18 |
| det7 Y | 8 | 1 | 65 | 11 | 4 (143.5–148.2, 256.6) |

(Y views of det2/3/6 in `f12_table.md`.) det3's bench mask is entirely FEU-7 electronics — transferred to n_TOF A
(read by FEU 3) it would censor 3 live strips and miss FEU 4 ch 50. The census over-calls at low crossings (det4 X: 35
of 54 "dead" fire normally beside big neighbours; ~16 expected by chance at ~30 crossings/channel). Time changes:
n_TOF A-x conn 8 silent through run_79, live by run_145; o23_long_det4 X silent 300–400 mm. **n_TOF bundles carry no
dead/hot list.** Proposed rule (policy for H): chamber-side faults by position + each dataset's own electronics faults
from its own hits.

## SPS remap (fixed)
H4: det4 on FEU 03, connectors 3–6 only; X = ch 0–255 (bench 128–383), Y = ch 256–511 (bench 128–383), so the bench
mask by channel number censored wrong strips. New `CalibrationBundle.remap_channels` (`wft/calib.py`, test
`wft/tests/test_channel_remap.py`); H4 mask measured at H4 with uRWELL: `rerun/channel_layer/det4_sps_h4.json` (X 18,
Y 1 = FEU 03 ch 510; stripe-inclusive 52-strip X list as mode `h4_stripes`). Bench-by-position vs H4: 12/13 in-beam X
agree counting stripe strips. `sps_mask.py` modes h4 (default) / h4_stripes / bench / none; wired into `sps_reco.py
--mask`, `heldout_arm.py --sps-mask`, `make_sps_dag.py`.

## Open / risks
- **SPS gate with the H4 mask not run** (28 % of run63 X windows contain a masked strip with samples; 40 % stripe list).
- Golden bench masks unchanged: over-mask on census stats (det4), under-mask chamber-side (det6/det7) → needs its own gate.
- FEU-ID identity bench↔n_TOF assumed (FEU 4 ch 50 supports, FEU 3 ch 79 not / repaired). Thin n_TOF statistics.
- `Mx17StripMap.map_hit` connector bug untouched (no-op for 1:1 rows; H4 uses `det4_sps_map`).

Figures: `figures/f12_dead_map.png`, `f15_cascade.png`, `f15_maps.png`.
