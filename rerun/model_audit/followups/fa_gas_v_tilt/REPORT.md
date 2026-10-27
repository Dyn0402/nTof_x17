# FA — FOLLOWUPS items 1 (gas v(t), v(x,y)), 2 (det3 X/Y), 3 (tilt per run), 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Commits 2c11c09, 0bed8c8. No wft changes. Condor: `/eos/user/d/dneff/x17rerun/followups/fa/Analysis/`, local
`~/x17/cosmic_bench/model_audit/followups/fa/`.)

## Answers
1. Gas drift over each run is measured from every trigger with no M3 (column duration), v(x,y) separated from the bow
   by fitting them together; correcting both makes the angle scale clearly more uniform in time and across the chamber.
2. det3's X/Y difference is **not** a v difference: a per-view angle-scale error of the fit; ~half is the per-view σ0.
3. Rigid tilt is constant within a mounting and changes on remount; the bow is static per chamber.

## Method: column duration T (`colsum.py`)
Per trigger and view, sum the main cluster ±2 strips: angle drops out, the sum is a box T = gap/v ⊗ template, fit T. No
hit times. Clock: DREAM timestamp × 10 ns (checked on 15-min scan rows: 15.006 min). ~½ of per-event T on the fit's 10 ns
grid → block/cell statistics use a 20 % trimmed mean. Saturated tracks read a different absolute T (+1 % det2, +7 %
det7, +16 % det4) but the same trend → normalised per class (det6/det7 saturate 80–90 %). v enters after the fit
(bundle v ±10 % → bit-identical w on 120 det4 events) → v(t), v(x,y) are pure angle-conversion changes, table-level gate
exact. det3 drift scan from T alone: 30 mm/T = 18.8, 26.6, 34.3, 37.4, 38.9 µm/ns at 500…1100 V → β = dln v/dln E = 0.87.

## Item 1
**v(t)**, block size at the Allan-deviation minimum: det2 8.5 min (floor 0.34 %, range over run 2.2 %); det3 14 min
(0.25 %, 1.8 %); det4 43 min (1.0 %, 7.2 % falling); det6 14 min (0.72 %, 10.5 % rising); det7 49 min (0.50 %, 1.9 %).
Ladder cross-check (8 blocks) agrees in direction but the angle scale moves faster than 1/T on changing-gas chambers
(slope 1.45 det4, 1.68 det6, 1.09 det7); on det6 the excess follows the gain (+38 % over the night, r = 0.78) →
brightness-dependent fit scale, not v.

**v(x,y) vs bow**, per 100 mm cell fit w = v·tan + a + g·(P − Pc) (slope = v; intercept and gradient = bow tilt), plus T
per cell. A pure bow changes v through the field: dln v = −β/(1+β)·dln T = −0.46·dln T; pure gas: −1·dln T. Measured
k: det3 −0.45 ± 0.02 (bow only), det2 −0.27 ± 0.03 (v nearly flat), det7 −0.35 ± 0.02 (bow-like; T map has holes),
det4 −0.75 ± 0.03 (bow + gas), det6 −0.77 ± 0.02 (gas). Ladder v(x,y) rms 2.3 / 1.6 / 4.9 / 6.3 / 9.0 %. X and Y see the
same v map (r up to 0.96) → gas, not view. Gap map dln T + dln v (rms 3–10 %) = an independent column route for FB; raw T
maps (rms 10–13 %) contaminated by edge truncation / dead regions, used only through k.

**Gate** (maps from even eids, metrics on odd, paired bootstrap; each conversion re-measures its own kw, w0), X / Y:
| chamber | v(t): Δ time spread of scale | + v(x,y): Δ cell spread | + v(x,y) + w0(x,y): Δ s68 |
|---|---|---|---|
| det2 | +0.31 / −0.03 pp | −0.52 / −0.50 pp | −0.04 / −0.26° |
| det3 | +0.30 / 0.00 pp | −1.01 / −1.04 pp | −0.13 / −0.22° |
| det4 | −1.46 / −1.77 pp (3.2→1.7, 3.8→2.1 %) | −1.89 / −2.05 pp | −0.19 / −0.40° |
| det6 | −2.42 / −1.98 pp (4.5→2.0, 4.5→2.5 %) | −5.89 / −6.28 pp (8.5→2.6 %) | −0.65 / −0.61° |
| det7 | −0.33 / −0.45 pp | −1.30 / −2.25 pp | −0.06 / −0.33° |
Most of the s68 gain is the w0(x,y) bow-tilt map (FB). M3-free form v(x,y) = T(x,y)^k matches the ladder map on det3,
det4, det6; fails det7 Y (+0.7 pp, T holes). On stable rows v(t) adds ~1 % non-v wander (method floor).

## Item 2 — det3 X/Y +1.7 % is a per-view tan scale
Column duration per view (no angle; same events, unsaturated): T_X/T_Y − 1 = −0.63 ± 0.06 % det3; det2 −0.19, det4 +0.14,
det6 −1.0, det7 +0.34 % — does not track the ladder ratios (det3 +1.70 ± 0.25, det4 −3.5, det6 +1.5 %); ≤ 0.6 % of
det3's could be timing. Survives the bow-tilt map (+1.60 ± 0.22). Pinned t0 enlarges it (+2.36 vs +1.67, 6729 events).
One common σ0 (0.29, no per-event choice) halves it: +0.77 ± 0.24 (+1.20 with pinned t0); det6 +1.36 → +0.37; det4 the
other way (−3.66 → −5.51) → different cause there. Quadrants +2.3/+2.6/+0.8/+0.8 % (one sign; 3×3 χ² 59/8). Per run:
sat +1.70, 6-27/28 night +1.66, 6-25 bottom slot +1.37 (follows the chamber). 18 mesh rows 425–525 V: +1.78 ± 0.18 %
(χ² 22/17), slope +0.19 ± 0.07 %/10 V (mild gain dependence).

## Item 3 — tilt per run (47 rows, `results/tilt_table.md`)
Mesh height as a quadric about the centre: slope = rigid tilt, curvature = bow; angle route and position route on the
same events. Angle route broken on det2 6-17 and det3 6-22/6-23 bottom-slot rows (ladder 0–15 µm/ns; FB saw the same) →
position route only there.
| chamber, mounting | X tilt ° angle / position | Y tilt ° angle / position |
|---|---|---|
| det2 6-17 (top) | — / −0.32 ± 0.11 | — / +0.01 ± 0.17 |
| det2 6-22 (top, 3 rows) | −0.38 / −0.13 | −0.36 / −0.21 |
| det3 6-22, test, 6-23 (bottom, 5) | — / +0.17…+0.29 | — / −0.54…−0.66 |
| det3 6-25 (bottom) | +0.25 / +0.33 | −0.46 / −0.58 |
| det3 6-27 sat (top, 23) | −0.09 / −0.04 | −0.16 / −0.24 |
| det3 6-27/28 night (top) | −0.08 / +0.03 | −0.17 / −0.23 |
| det4 6-23 (top, 3) | −0.38 / −0.25 | −0.52 / −0.64 |
| det4 6-24 (top) | −0.18 / −0.06 | −0.20 / −0.17 |
| det6 6-26 (bottom) | +0.12 / +0.10 | −0.72 / −0.75 |
| det7 6-26 (top) | −0.02 / −0.51 (det7 X card fault) | −0.25 / −0.31 |
Constant within a mounting (det3 23 sat rows χ² 27/22 X, 15/22 Y; drift field and mesh HV don't move it). Changes on
remount (det3 slot move X +0.3°, Y −0.4°; det2 6-17 → 6-22; det4 0.3–0.5° Y between 6-23 and 6-24, same slot → re-seated).
Slot dominates: Y −0.5…−0.8° bottom, −0.2…−0.36° top. Bow curvature positive on every row, both routes, ~25–125 /km, no
sign change on remount.

## Spec for FE
1. `v_time` per run `{clock: dream_ts_10ns, acq_order, t_h[], v_rel[], block_min}`, v_rel = T̄/T_block, piecewise-linear;
   the reco event record must carry the DREAM `timestamp` + acquisition tag (today only `ftst`). Measured by `colsum.py`
   (any dataset incl. n_TOF).
2. `v_map` per chamber × run condition, common to both views: 4×4 cells of 100 mm `{e, v_rel, err}`, bilinear, empty
   cells → median; optional M3-free `{T_map 50 mm, k}` with v_rel = T_rel^k (k: det3 −0.45, det4 −0.75, det6 −0.77, det2
   −0.27, det7 −0.35).
3. tan = (w·1e3 − w0_run − A_bow(U,W)) / (kw_run·v_drift·v_rel(t)·v_rel(x,y)), mid-gap position, both views, one
   iteration. **v_map replaces FB's linear scale term (1 + su·U + sw·W)** — don't apply both; keep FB's A(U,W).
   `u_window()` uses v·v_rel(t).
4. Re-measure w0, kw per run after bow + v corrections (rigid tilt changes on every remount).
5. Gate: re-finalise golden rows, `heldout_compare.py`; pass = time/cell spreads drop as above and det2/det3 X s68 not worse.
6. Never transfer v_map or w0 to n_TOF; re-measure v(t) and the T map there with `colsum.py`.

## Figures (`figures/`)
fa_vt (v(t) column vs ladder), fa_allan (block size), fa_maps (T, ladder v, gap maps), fa_twomap (v(x,y) vs bow field
prediction; det3 β = 0.87), fa_gate, fa_xy_scale (X/Y ratio vs T_X/T_Y, per run, per HV row), fa_tilt_rows, fa_bow_rows.

## Risks
v(t) under-corrects det4/det6 (scale moves 1.1–1.7× faster than 1/T; det6 excess follows gain, unresolved); v(t) floor
~1 % non-v wander in T, slightly worsens X on stable rows; T maps indicative only; `colsum.py` covers only acquisitions in
the M3 directory (det3 golden 2.35 h); broken angle route on early rows not chased; det6 pinned-t0 arm (4583275.19) not in
the X/Y numbers; common-σ0 arm used 0.29 not 0.30; the 1-D bow reference in fa_gate is in-sample (overstated).
