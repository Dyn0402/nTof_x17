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

## Pass-1 update, 2026-10-31 (toy RE-RUN on the pass-1 bundles: `toy_inputs_pass1.py` + `fd_pass1.py` on lxplus, condor `condor/make_jobs_pass1.py`, `budget_pass1.py` → `results/budget_pass1.json`)

**Headline: re-running the toy on the pass-1 bundles changes almost nothing. The unexplained term is still 54–88 % of the
X-view variance: interior_good, calibration rows, v(t) arm: det2 0.72 of 0.98°, det3 0.68 of 0.93°, det4 1.52 of
1.62°, det6 0.93 of 1.13°, det7 0.98 of 1.12°.** All toy terms together are 0.38–0.46° (v(t) arm) against a measured
0.93–1.62°. The two fixes land as expected:
- **det7's dead-strip term** on interior_good is 0.07° (unresolved); the audit toy had 0.42°.
- **The v-drift rung** now runs on each row's own r(t) and the data's angle mix. It agrees with the measured correction on
  det4 (0.29 against 0.35°) and det7 (0.12 against 0.14°).
- **On det6 the toy still reads more** (0.41 against 0.29°). That is the part of det6's drift the linear v(t) model does not
  remove, so it stays in the v(t) arm (below).

Against GA/GB/bow (first version of this section, toy on the audit bundles): det4 grows 1.38 → 1.52° because GA took its gain
term away; the bow block removed most of the position term (det6 1.09 → 0.45°), but 0.26–0.54° of cross-validated 2D tilt
is still there after it. What is unexplained is not a waveform misfit (10-27 finding, not re-tested on pass 1).

| X view, interior_good, calib row (°) | det2 | det3 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| floor L0 (toy, pass bundle) | 0.33 | 0.34 | 0.31 | 0.32 | 0.32 |
| production fit q(L1, L0) | 0.00 | 0.00 | 0.18 | 0.10 | 0.00 |
| L2 footprint / L3 X copy / L5 pulse | 0.08 / 0.08 / 0.13 | 0.07 / 0.14 / 0.14 | 0.15 / 0.10 / 0.13 | 0.08 / 0.00 / 0.09 | 0.20 / 0.19 / 0.13 |
| L6 det3 diffusion / L7 dead + CM | 0 / 0.07 | 0.07 / 0.11 | 0 / 0.10 | 0 / 0.05 | 0 / 0.07 † |
| ~~L4 strip gain map~~ (GA: withdrawn; the toy's value) | 0 (0.12) | 0 (0.10) | 0 (0.65) | 0 (0.14) | 0 (0.14) |
| L8 v drift: toy on the row's r(t), data angle mix | 0.04 ⁿ | 0.08 ⁿ | 0.29 | 0.41 | 0.12 |
| L8 v drift: measured q(pass, vt) on the same events | no map | no map | 0.35 | 0.29 | 0.14 |
| L8 in the budget, pass / v(t) arm ‡ | 0 / 0 | 0 / 0 | 0.35 / 0 | 0.41 / 0.29 | 0.14 / 0 |
| all toy terms in quadrature, pass / v(t) | 0.38 / 0.38 | 0.42 / 0.42 | 0.56 / 0.44 | 0.55 / 0.46 | 0.47 / 0.45 |
| 2D tilt left after the bow, q(slope, 2D-40 mm xval), pass / v(t) | 0.54 / 0.54 | 0.47 / 0.47 | 0.36 / 0.36 | 0.42 / 0.45 | 0.26 / 0.28 |
| M3 (GB) | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| **unexplained, pass / v(t)** | **0.72 / 0.72** | **0.68 / 0.68** | **1.52 / 1.52** | **0.94 / 0.93** | **0.99 / 0.98** |
| unexplained with the toy on the audit bundles (c144af21), v(t) | 0.72 | 0.72 | 1.54 | 0.97 | 0.89 (0.98 with L7 = 0) |
| measured s68 (FD convention), pass / v(t) | 0.98 / 0.98 | 0.93 / 0.93 | 1.66 / 1.62 | 1.17 / 1.13 | 1.13 / 1.12 |
| measured, 10-27 audit tables (full sample) | 1.10 | 1.06 | 1.65 | 1.76 | 1.57 |

**What the re-run changed** (`pass1_basis.py`, switched on by `FD_BASIS=pass`; without it the toy is the 10-27 one):
- **Bundles:** the five pass-1 calibration bundles: template, sigma_p0 0.30, dead mask, t0 prior and kernel. The toy's ideal
  footprint follows the 0.30, so L1 stays ideal. q50 is each row's median x_q_sum (`results/toy_inputs_pass1.json`).
- **Dead strips the pass mask still misses:**
  - det2 ch 69 and det3 ch 39 (cloud_basics 39);
  - none on det4, which over-masks;
  - none on det6: the pass mask's ch 208–224 is the 177–185 mm block that F12 found missing from the audit mask;
  - † none on det7 for interior_good: it keeps X ch 256–511 only, and every chamber-side det7 fault is below ch 256.
  For the full sample det7 gets F12's 18 misses, at random below ch 256: L7 = 0.26° there (`ladder_pass1_full_n1500_det7`).
- **Placement:** tracks in chip blocks 1–6 (det7 4–6), an approximation of interior_good. The full-sample budgets use the
  same toy, except det7's L7.
- **v drift:** v_true = v · v_rel, with v_rel drawn from the row's own per-event r(t) (the v(t) map; a uniform |dv| on rows
  below the gate), instead of a Gaussian of the FOLLOWUPS-1 rms.
- **Angle mix:** each toy s68 is weighted to the row's |θ_ref| mix instead of pooling 3 / 9 / 16° equally. The data are
  48–57 % below 6°, 36–40 % at 6–12.5° and 7–17 % above.
- **Statistics:** 1500 events per angle point, 10× the 10-27 run (288 jobs, `ladder_pass1_*_n1500_*`).
  - At 150 events a single term was resolvable only above ~0.2°; that run is kept as `ladder_pass1_ig_<det>.json`.
  - At 1500 the bootstrap error on a resolved term is 0.01–0.03° (`term_err`, `term_sig`).
  - The terms flagged unresolved here (≤ 0.08°) are consistent with 0.
- **toy_all** is now the quadrature sum of the terms. The 10-27 form (the cumulative rung with L4/L8 swapped) agrees within
  0.02°, except on det4: its withdrawn gain-map rung (0.65°) does not come out in quadrature (`toy_all_cumswap` 0.31 against 0.56).

‡ **The v-drift gap, explained.** The 10-27 rung over-predicted what the correction removes (0.71 against 0.29° on det6)
for three reasons:
1. it pooled the angle points equally, while the drift cost grows with tan θ and the data sit at small angles (the angle mix
   alone takes det6's toy term from 0.57 to 0.41°);
2. it drew a Gaussian of the FOLLOWUPS-1 rms instead of the row's actual r(t);
3. it was rescaled linearly from a different rms.

With the row's r(t) and the data's angle mix, the toy matches the measurement on det4 (0.29 against 0.35°) and det7
(0.12 against 0.14°). On det6 it still reads 0.41 against 0.29°:
- **Reading:** the linear r(t) removes only part of det6's drift; the cross-check found the line misses det6's knee (χ² 82/38).
- **Treatment:** the remainder, q(0.41, 0.29) = 0.29°, stays in the v(t) arm.
- **Caveat:** det6's drift amplitude is itself uncertain by about ×2 (ladder +17.9 % against column duration +9 %,
  `rerun/vt_all/CROSSCHECK_2026-10-31.md`). With the column's amplitude the toy would read ~0.2°, so the det6 residual sits
  inside that uncertainty.

ⁿ **Rows with no v(t) map** (det2 o22 and det3 sat, |dv| 1.2 % and 0.5 %): the toy term is unresolved, so 0 in the budget.

- **Rows:** the calibration golden rows (o22_long_det2, sat_det3, g_det4, g_det6_long, g_det7_long). The headline rows
  (det2 6-22 long_run, det3 6-27 sanity) agree within 0.03°: unexplained 0.70 / 0.69°, v(t) arm (`budget_pass1.json`,
  role `headline`).
- **Full sample (FD's 10-27 sample), v(t) arm:** unexplained det2 0.77, det3 0.74, det4 1.54, det6 1.03, det7 1.34°.
  det7's full sample includes ch < 256: measured 1.55°, L7 0.26°.
- **The toy's floor barely moved:** 0.31–0.34°, as on the audit bundles, so Paper II §5's physics floor stands.
- **Y view** (no toy exists; measured / tilt left / rest incl. model, v(t) arm): det2 1.02 / 0.48 / 0.90, det3
  0.85 / 0.43 / 0.73, det4 1.22 / 0.40 / 1.16, det6 1.35 / 0.45 / 1.27, det7 1.29 / 0.50 / 1.19°.
- **Convention: this is FD's s68** (|d − median| 68th percentile over |d| ≤ 5°). The Paper I headline (03_angles,
  `rerun/paper1_res`) has no ±5° clip and reads higher where tails are large: det4 X 2.41 against 1.66°, det6 Y 1.51
  against 1.40°, det2/det3 within 0.06°. The paper must say which; the budget only adds up in the clipped one.
- **Open:**
  - the toy is X-only;
  - the placement approximates interior_good by chip block;
  - the toy runs HEAD's wft, whose `fit_plane` is identical to the pass code 6a0bb1f (only `tan_bow`/`tan_of` and `compat`
    changed since);
  - the unexplained 0.7–1.5° is still unattributed: it is not a waveform misfit (10-27 finding, not re-tested on pass 1).
- **Condor and outputs:**
  - `~/x17rerun/fd_toys_pass1/`: 334 jobs at 150 events (scan_a included); `.../n1500/`: cluster 4602789, 288 jobs.
  - Code archive `/eos/user/d/dneff/x17rerun/fd_toys_pass1/code_fdt_ba4728ed`.
  - Outputs `/eos/user/d/dneff/x17rerun/fd_toys_pass1/{scan_a_ig,ladder_ig,ladder_full,ladder_ig_n1500,ladder_full_n1500}`.
  - `results/scan_a_pass1_ig.json` is the depth-bin scan on the pass bundles; it is not part of the budget.
