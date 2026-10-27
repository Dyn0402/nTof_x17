# ws_v_w0 — F8 (real ladder-v uncertainty) and F9 (what w0 is), 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.)

## F8 — the ±0.07 µm/ns was statistics only
Three distinct numbers:
1. **Error on the run-mean v**: ±0.16–0.27 µm/ns (0.4–0.7 %) det2/det3; ±0.4–1.2 det4/det6/det7. Includes time-block
   bootstrap, estimator and |tan|-window choice, M3 angle scale (~0.25 %, irreducible). det7 X dominated by trimmed-mean
   vs median (skewed).
2. **Local spread around it** (what matters when v is applied elsewhere): 1.7–2.1 % det2/det3, 5–8 % det4/det6/det7.
   Position: quadrant v differs (det3 X 35.6 → 37.5). Time: v drifts within runs (det4 −9…−10 % over its day run;
   det6 +12 %, det7 +4 % over the shared night); the independent column-duration route moves the same way → real gas
   change.
3. **±tan**: the two sides agree to ≤ 1 % (det6 Y 2.7 %). The apparent 2–7 % came from a biased head-on offset in the
   ladder estimator in `fleet_tables.py` (≤ 0.15 µm/ns effect on global v) — still to fix.

**det3 X vs Y**: event-paired +2.13 ± 0.39 % — real, stable in time, varies by quadrant (−0.2 … +3.7 %), and changes
sign between chambers (det2 0.0, det4 −3.0, det6 +1.9, det7 +0.6 %) → not strip-layer geometry. A per-view angle-scale
error of the size of the width–tilt trade (F21; Y carries more modelled width in every bundle). kw absorbs it on the
bench; elsewhere a 2–3 % per-view systematic.

## F9 — w0 = chamber tilt against M3
The fit puts each drift slice's charge at p0 + w·(t − t0); w0 is the w for a track M3 calls perpendicular, i.e. an
angle offset α = w0/v. A tilt also gives a position residual ∝ tan × position (mesh height varies across the chamber),
measured from positions alone. The two agree: r = 0.81 over 16 chamber-views; rms offset 0.34° → 0.18° after
subtracting the position-route tilt; det6 Y −0.86° vs −0.89°; det3 in the bottom slot (6-25) +0.31/−0.62° vs
+0.25/−0.46° — the offset follows the mounting.
**Sign**: every Y offset negative (−0.27 … −0.89°), X small: the bench slots are not level along M3 x (~0.3° top slot,
0.6–0.9° bottom). Sign fixed by M3's z axis pointing up (upper plane at 1302 mm), not by self-consistency.
**det3 field dependence was an artefact** of `set_w0kw.py` folding (kw − 1)·v·tan into w0 when kw ≠ 1 (500 V rows
read with the 1000 V v). Fitted as a line the offset is constant: −0.28 / −0.24 / −0.28° at 1000 / 700 / 500 V.

## Open / risks
- **Bench w0 must not transfer to n_TOF** (mounting, not chamber). `refit_B_beam.py` says the n_TOF bundles are bench
  transfers incl. w0/kw → −0.3…−0.9° Y bias if used. `insitu_calib.py` re-measures w0 at n_TOF; which w0 the stage-3
  products used is NOT checked.
- det3 500/700 V rows rest on ~700-event dry runs; position-route tilt has its own ≤ 0.3° systematic.
- Not yet changed: `set_w0kw.py` (fit w0 as a line), `fleet_tables.py` head-on offset.

Scripts: `build_table.py`, `v_uncertainty.py`, `budget.py`, `w0_study.py`, `diagram.py`; data in
`~/x17/cosmic_bench/model_audit/ws_v_w0/`; figures in `figures/`. Commit ae971c3.
