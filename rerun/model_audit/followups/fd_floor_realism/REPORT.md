# FD — FOLLOWUPS item 10: the angle floor, the 60 ns jump, and why we don't reach the floor, 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Commit 7b6c290; site 6f4768d.) Note: https://dylan-neff.web.cern.ch/notes/mx17-angle-floor.html

## (a) The 60 ns jump — mechanism confirmed, two corrections
Perfect-model toy on all five chambers (v 24–39 µm/ns), 20/30/60/90 ns bins, with/without clustering; det3 also at pitch
0.5 / 1.2 mm and footprint 0.10 / 0.45 mm. The fit spreads each depth bin's charge evenly along its track piece
L = v·Δt·tanθ; real charge in a bin is a few lumps; once L is long enough the strips see the lumps, the model cannot
place them, the slope absorbs it → random error + shallow bias.
- **The scale is the single-electron footprint, not the pitch:** σ_eff² = σ0² + Dp²·u_mid + pitch²/12 ≈ (0.52–0.68 mm)².
  Onset at L ≈ 1.55 σ_eff (±12 % over 19 runs) vs L ≈ 1.14 pitch (±16 %). At 0.5 mm pitch L/pitch moves 1.14 → 1.66
  while L/σ_eff stays (1.58 → 1.56); a 0.45 mm footprint delays the onset 25° → 28°. On MX17 σ_eff ≈ 0.7 pitch, so
  "about one strip" is numerically right.
- **Clustering doubles it, doesn't create it:** without cluster-size fluctuations the excess halves (det3 60 ns, 45°:
  0.68° instead of 1.61°); single electrons with gain fluctuations are lumpy too.
- 20 ns bins: flat to 45° on every chamber.

## (b) The realism ladder (generated events only get the imperfection; every rung fitted by `wft.reco.fit_plane` on the
promoted bundle; 150 tracks at 3/9/16°, five chambers, X view)

| X view (°) | det2 | det3 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| floor (perfect model) | 0.34 | 0.34 | 0.32 | 0.32 | 0.31 |
| production fit, ideal events | 0.36 | 0.34 | 0.41 | 0.32 | 0.43 |
| + every measured imperfection | 0.41 | 0.41 | 0.76 | 0.55 | 0.64 |
| χ²/dof toy / data | 21 / 118 | 11 / 62 | 3 / 18 | 40 / 444 | 25 / 117 |
| position-dependent tilt (2D, 40 mm cells, cross-validated) | 0.66 | 0.64 | 0.46 | 1.09 | 0.43 |
| M3 reference (bound) | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| **unexplained** | **0.77** | **0.72** | **1.38** | **1.26** | **1.36** |
| measured | 1.10 | 1.06 | 1.65 | 1.76 | 1.57 |

Ladder reaches 31–46 % of the measured resolution; data χ²/dof is 5–11× the toy's.

Ranked (each alone on top of the production fit, quadrature): 1. unexplained 0.72–1.38° (47–75 % of the variance);
2. 2D tilt 0.43–1.09° (the 1D own-axis angle_map captures ⅓–½ of it; det6/det7, same triggers, share part of their errors:
r 0.15 X, 0.31 Y); 3. strip gain spread 0.55° det4, ≤ 0.19° elsewhere; 4. v drift over the run 0.41° det4, 0.40° det6,
0.19° det3; 5. unmasked dead strips + CM 0.42° det7, 0.21° det6; 6. footprint shape (0.25 core + measured tails) 0.34°
det7, ≤ 0.06° elsewhere; 7. production fit itself (blind start, per-event σ0) 0.29° det7, 0.25° det4; 8. one-sided X copy
0.11–0.22°; 9. pulse-shape mismatch ≤ 0.21° (upper estimate); 10. M3 ≤ 0.15°, det3 extra diffusion 0.08°.

**The unexplained part is not a waveform misfit:** in data the resolution is flat across the bottom four χ²/dof
quintiles (det3 X 0.99–1.06°) and flat in charge. Four unmeasured waveform candidates at generous sizes on det3 give
≤ 0.17° (10 ns channel skew 0.17°, ±5 % pulse width 0.08°, per-event footprint 0.20–0.45 mm 0.00°, FC's σ0 0.30 0.00° det3 /
0.27° det7) and never push toy χ²/dof above ~15. A χ²-blind angle error that partly follows position → geometry (local
field or mounting tilt finer than 40 mm) more than the waveform model. The data χ² excess is real but a separate problem.
FC's one-σ0 change is resolution-neutral in the toy.

## What would most improve the measured resolution (reconstruction only)
1. 2D (x, y) tilt map per chamber from M3, cross-validated, in the angle conversion: −0.43…−1.09° in quadrature (bench
   geometry — never to n_TOF without an in-situ check). (= FB's bow block, finer cells worth testing.)
2. 20 ns or interpolated depth grid: removes the steep-track jump (det3 +1.6° at 45° with 60 ns) and shallow bias; main
   term for n_TOF pairs at 30–45°.
3. Bundle content: strip gain map (det4 −0.55°), v(t) (det4, det6 −0.4°), complete dead masks (det7 −0.4°).
4. Drop the per-event σ0 choice (decided): −0.25…−0.29° det4/det7.

## Limits / risks
X view only (Y along-strip spread not simulated); bench gas only; pulse-shape and hypothesis sizes are not measurements;
the tilt estimate is limited by 40 mm cells (cell noise ~0.3°); M3 term a bound. No pulse shape matched the measured
residual pattern once the depth profile is free — sized on isolated pulses, labelled an upper estimate.

## Files
`scan_a.py`, `onset.py`, `ladder.py`, `measured_chi2.py`, `data_split.py`, `tilt_map_xval.py`, `budget.py`, `diagram.py`,
`make_floor_deck.py`, `condor/`, `results/*.json`, `deck/mx17-angle-floor.html`. Condor cluster 4583195 (286 jobs) →
`/eos/user/d/dneff/x17rerun/followups/fd/`, mirror `~/x17/cosmic_bench/model_audit/fd/`.
