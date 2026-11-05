# n_TOF chambers D and B before the bundles freeze (2026-11-04)

Plan §6 "remaining". Report: `report.html`. Data: `/media/ucla/work/ntof_pass1/d_b/` (line_kw/measure_D_v1.csv,
wall_kw/summary_D_v1.json, dbeam/stack_fits.csv, dbeam/d_beam_reco.csv, bd_cosmic/{eff,bpos,dv_lines}.csv).

## 1 D on beam: the box-fit failure is a beam-gated oscillation (dbeam.py, ringing.py)
- Not window, latency, trigger timing or a column offset: on D's own trigger (its wall + plastic, `hw_D`) the beam box is
  where A's and C's are (start −45/−60 ns, T 1005 ± 17 / 1031 ± 13 ns x/y). The withdrawn "D window offset" stays withdrawn.
- Only 34 % of D's clean beam clusters are on D's own trigger (A 68 %, C 80 %); the rest stack flat with a 6-sample pattern.
- A ~360 ns (2.8 MHz) oscillation puts 3–4 in-phase humps on tens of strips: 82–90 % of D beam triggers, every trigger
  from ~10 ms after the flash to 80 ms; 5 % on beam-off cosmics. A and C show it only 10–30 ms after the flash. 66–102 D
  channels have noise > 25 ADC (A 6–11). 16/32-channel CM or odd/even splits do not remove it.
- D own-trigger beam fits after 10 ms: median χ²/dof 24/25 (A 1.5/1.4), t0 width 135/143 ns (A 58).
- Not truncation: own-trigger columns end ~960 ns of 1200; 7–10 % run past (as C); but they read +9/+18 % longer on beam
  than cosmics (A, C +3–5 %).
- **Limitation:** D's v is not measurable on beam and D's beam angles are limited by the oscillation.

## 2 kw_D without an A–C line (centroids.py, line_kw.py, wall_kw.py, wall_check_A.py)
- Truth: cosmic lines through another chamber's cluster centroid (positions only). Validated on A–C (cos_0001, v1k): ≈ 1 %
  from ac_line.py's ratio.
- B–D lines, 12 run_149 sub-runs: fitted/true 0.880 ± 0.014 (x, n 94), 0.873 ± 0.010 (y, n 106); flat over quarters of D
  and B, same on both tan signs (multiplicative).
- A–D lines ≥ 250 mm: 0.87–0.90; shorter A–D lines read 0.71–0.83 (placement of A's partly visible steep track).
- C–D lines impossible: D x dead (137 channels) beyond local x +40 mm.
- D's wall (x): direct s = 1.05 (0.97–1.11); old 1.525 on the new fits → 0.908, 0.94 after the wall's +3.6 % on A.
  Spread → the ±0.03 systematic. Plastic L/R gap not used (0.69 on A where the line says 1).
- D timing v vs angle scale differ by 12 %: lines give kw·v ≈ 27.3 µm/ns; D's timing v depends on the muon sample
  (34/33 B–D lines, 29/27 A–D, 31.7/30.2 all). For A and C the two agree to ~1 %.
- **Proposed D bundle:** v 31.0 (timing, as A/C), kw 0.880 (x) / 0.881 (y), ± 0.03 syst; cosmics only.

## 3 B (bd_cosmic.py)
- Confirmed: no field-shaping ring, no angle, no measurable v (column not a box).
- Cosmic efficiency 9 ± 4 % on D-predicted crossings (n 65), 22 ± 9 % on A-predicted (n 31); C 59 % on A-predicted.
- B's v moves no position: same events at v 42.6 and 30.0 give identical p0 when the same candidate wins; through the
  bow term p0 − tan·h only (median 0.06 mm, 95th pct 0.65–1.23 mm for 30 % in v).
- B's v changes B's efficiency through the raw-tan plausibility gate: winner changes on 1 % (cosmics) / 11–14 % (beam)
  of planes; paired cosmic tracks 118 (v 42.6) vs 155 (v 30). A limitation on B efficiency, not positions.
- **Proposed B bundle:** v 42.6 (Magboltz prior), angles NaN downstream; no (or a very wide) tan_max so a meaningless
  angle does not gate candidates; optionally drop the bow h term.

## Proposed, not applied
- `wft_beam` seeding/fit: flag strips with ≥ 3 evenly spaced humps (240–480 ns apart), or subtract a ~360 ns sinusoid
  per channel; the only route seen to usable D beam angles. Untested.

## Not reviewed
- 12 of 87 cosmic sub-runs; the ringing detector is a heuristic; A–D partial-track placement validated only where it
  barely matters; B field-cage distortion unmeasured (central vs outer B agree within 0.02).
- A's oscillation decays over 10–30 ms after the flash, like A's beam wall-scale transient (0.77 → 0.92): not followed up.

## 4 D t0 pinned vs free (2026-11-05; `dpin_run.sh`, `dpin_score.py`; plan §6 11-05)
- D cosmic latency (run_149): x +18.3 ± 3.4, y +20.0 ± 2.2 ns. D beam latency not measurable free (oscillation).
- Beam run_145 `_000` (1,500): pointing < 15 mm 68 (free) → 91 (pin, cosmic + A/C beam offset) → 99 (pin, cosmic
  latency); same-track DCA 68 % 15.8 → 13.1 mm; χ²/dof unchanged. Own-trigger beam tracks sit near the cosmic latency.
- Cosmic B–D lines: kw closes free (0.983 / 1.009); under the pin 0.970 / 0.990 → kw_D 0.854 / 0.872.
- Proposed: D pinned with its cosmic latency on both trigger classes, kw 0.854 / 0.872.
