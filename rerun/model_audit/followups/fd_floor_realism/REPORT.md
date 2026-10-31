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
| M3 reference (bound; GB 10-28: 0.05, ≤ 0.11 measured) | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| **unexplained** | **0.77** | **0.72** | **1.38** | **1.26** | **1.36** |
| measured | 1.10 | 1.06 | 1.65 | 1.76 | 1.57 |

Ladder reaches 31–46 % of the measured resolution; data χ²/dof is 5–11× the toy's.

Ranked (each alone on top of the production fit, quadrature): 1. unexplained 0.72–1.38° (47–75 % of the variance);
2. 2D tilt 0.43–1.09° (the 1D own-axis angle_map captures ⅓–½ of it; det6/det7, same triggers, share part of their errors:
r 0.15 X, 0.31 Y); 3. strip gain spread 0.55° det4, ≤ 0.19° elsewhere [**withdrawn 10-28 by GA**: the σ 0.31 input was the stripe structure, measured strip-to-strip gain is 0.02–0.04 and a measured map changes s68 by 0 ± 0.03°; see `followups/ga_gainmap/REPORT.md`]; 4. v drift over the run 0.41° det4, 0.40° det6,
0.19° det3; 5. unmasked dead strips + CM 0.42° det7, 0.21° det6; 6. footprint shape (0.25 core + measured tails) 0.34°
det7, ≤ 0.06° elsewhere; 7. production fit itself (blind start, per-event σ0) 0.29° det7, 0.25° det4; 8. one-sided X copy
0.11–0.22°; 9. pulse-shape mismatch ≤ 0.21° (upper estimate); 10. M3 ≤ 0.15° (GB 10-28: 0.05°, ≤ 0.11° measured), det3 extra diffusion 0.08°.

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
3. Bundle content: ~~strip gain map (det4 −0.55°)~~ (withdrawn, GA), v(t) (det4, det6 −0.4°), complete dead masks (det7 −0.4°).
4. Drop the per-event σ0 choice (decided): −0.25…−0.29° det4/det7.

## Limits / risks
X view only (Y along-strip spread not simulated); bench gas only; pulse-shape and hypothesis sizes are not measurements;
the tilt estimate is limited by 40 mm cells (cell noise ~0.3°); M3 term a bound. No pulse shape matched the measured
residual pattern once the depth profile is free — sized on isolated pulses, labelled an upper estimate.

## Files
`scan_a.py`, `onset.py`, `ladder.py`, `measured_chi2.py`, `data_split.py`, `tilt_map_xval.py`, `budget.py`, `diagram.py`,
`make_floor_deck.py`, `condor/`, `results/*.json`, `deck/mx17-angle-floor.html`. Condor cluster 4583195 (286 jobs) →
`/eos/user/d/dneff/x17rerun/followups/fd/`, mirror `~/x17/cosmic_bench/model_audit/fd/`.

## Pass-1 update, 2026-10-31 (`fd_pass1.py` on lxplus → `results/fd_pass1.json`; `budget_pass1.py` → `results/budget_pass1.json`)

**Headline: with GA (gain row withdrawn), GB (M3 0.05°), the bow block in the pass and v(t) measured, the unexplained
term is still 54–90 % of the variance: X view, interior_good, calibration rows, v(t) arm: det2 0.72 of 0.98°, det3 0.72
of 0.93°, det4 1.54 of 1.62°, det6 0.97 of 1.13°, det7 0.89 of 1.12°.** It is unchanged on det2/det3 (0.72 then and now)
and grows on det4 (1.38 → 1.54°), because GA took its 0.55° gain term away. The bow block removed most of the
position term (det6 1.09 → 0.45°, det2/det3 0.64–0.66 → 0.47–0.54°), but 0.26–0.54° of cross-validated 2D tilt is
still there after it. The gap to the toy has not closed. What is unexplained is not a waveform misfit (10-27 finding, not
re-tested).

| X view, interior_good, calib row (°) | det2 | det3 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| floor L0 (toy, audit bundle) | 0.34 | 0.34 | 0.32 | 0.32 | 0.31 |
| production fit q(L1, L0) (toy) | 0.12 | 0.00 | 0.25 | 0.06 | 0.29 |
| L2 footprint / L3 X copy / L5 pulse (toy) | 0.04 / 0.14 / 0.10 | 0.00 / 0.16 / 0.13 | 0.00 / 0.11 / 0.00 | 0.06 / 0.00 / 0.13 | 0.34 / 0.22 / 0.21 |
| L6 det3 diffusion / L7 dead + CM (toy) | 0 / 0 | 0.08 / 0.11 | 0 / 0.16 | 0 / 0.21 | 0 / 0.42 † |
| ~~L4 strip gain map~~ (GA: withdrawn) | 0 (was 0.11) | 0 (0.05) | 0 (0.55) | 0 (0.13) | 0 (0.19) |
| L8 v drift: pass arm, measured q(pass, vt) ‡ | 0.00 | 0.01 | 0.35 | 0.29 | 0.14 |
| L8 v drift: v(t) arm | 0.00 | 0.01 | 0 | 0 | 0 |
| all toy terms, pass / v(t) | 0.39 / 0.39 | 0.36 / 0.36 | 0.49 / 0.34 | 0.46 / 0.36 | 0.63 / 0.61 |
| 2D tilt left after the bow, q(slope, 2D-40 mm xval), pass / v(t) | 0.54 / 0.54 | 0.47 / 0.47 | 0.36 / 0.36 | 0.42 / 0.45 | 0.26 / 0.28 |
| M3 (GB) | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| **unexplained, pass / v(t)** | **0.72 / 0.72** | **0.72 / 0.72** | **1.54 / 1.54** | **0.99 / 0.97** | **0.90 / 0.89** |
| measured s68 (FD convention), pass / v(t) | 0.98 / 0.98 | 0.93 / 0.93 | 1.66 / 1.62 | 1.17 / 1.13 | 1.13 / 1.12 |
| measured, 10-27 audit tables (full sample) | 1.10 | 1.06 | 1.65 | 1.76 | 1.57 |

- **Rows:** calibration golden rows (o22_long_det2, sat_det3, g_det4, g_det6_long, g_det7_long); the headline rows
  (det2 6-22 long_run, det3 6-27 sanity) agree to ≤ 0.02° in every term (`budget_pass1.json`, role `headline`).
  Full-sample (FD's 10-27 sample) versions of every row are there too.
- ‡ **The toy's v-drift rung over-predicts.** Rescaled from its 2.9 % rms to each row's measured r(t) rms, it gives
  0.41 / 0.71 / 0.03° (det4 / det6 / det7). The correction actually removes 0.35 / 0.29 / 0.14° on the same events, so
  where a v(t) map exists the measured value replaces the toy. Rows below the v(t) gate (det2 o22, det3 sat) keep the
  toy rescaled to |dv|/√12.
- † **det7 L7 is the audit's dead-strip/CM rung.** interior_good already excludes det7 X ch < 256 and the pass carries the
  census mask, so 0.42° is an over-estimate for this sample. Moving it to 0 raises the det7 unexplained to 0.98°.
- **Y view** (no toy exists; measured / tilt left / rest incl. model, v(t) arm): det2 1.02 / 0.48 / 0.90, det3
  0.85 / 0.43 / 0.73, det4 1.22 / 0.40 / 1.16, det6 1.35 / 0.45 / 1.27, det7 1.29 / 0.50 / 1.19°.
- **Convention: this is FD's s68** (|d − median| 68th percentile over |d| ≤ 5°). The Paper I headline (03_angles,
  `rerun/paper1_res`) has no ±5° clip and reads higher where tails are large: det4 X 2.41 against 1.66°, det6 Y 1.51
  against 1.40°, det2/det3 within 0.06°. The paper must say which; the budget only adds up in the clipped one.
- **Not re-run:** the toy rungs (floor, production fit, L2/L3/L5/L6/L7) are the 10-27 toys on the AUDIT bundles. The pass
  bundles differ (the template moved det4 Y by −0.23°, DET4Y_ABLATION), so the toy terms can move. Re-running
  ladder.py on the pass bundles is a 286-job condor campaign (~1 h wall clock on the 10-27 run); not done tonight.
