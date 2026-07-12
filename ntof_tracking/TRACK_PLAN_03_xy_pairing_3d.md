# TRACK_PLAN_03 — X/Y pairing and 3D segments per chamber

**Turn PLAN_02's per-plane segments into 3D track segments per chamber. The
charge-balance result (bench PLAN_38) is the new ingredient that makes this
robust in multi-track windows. Deliverable: the 3D segment table — the
handoff product every physics analysis consumes.**

## Pairing

For each (window, detector): candidates X = {x-plane segments}, Y = {y-plane
segments}. Use `microtpc_lib.pair_planes(x_cands, y_cands, f_med, f_s68)`:

- **time IoU** of the [t0, t0+duration] spans, min 0.20 (May track_explorer
  convention — two projections of one particle share the drift-time span);
- **charge balance**: f = qX/(qX+qY) is narrow (σ68 ≈ 0.07) and flat in
  position and angle (bench PLAN_38: det3 med 0.487, det2 0.531 — per-chamber
  assembly constant). A candidate pair with |f−f_med|/σ68 > ~3 is two
  DIFFERENT particles even if time-compatible. Weights in `pair_planes` are
  starting values — tune on multi-track windows by eye first, then against
  PLAN_04 inter-chamber confirmation.
- f_med/f_s68 for det6/det7 (C/D) are placeholders in bench_constants:
  **measure them in situ** from unambiguous windows (exactly 1 X and 1 Y
  candidate, IoU>0.5): one histogram per detector per HV setting. Log to the
  calibration store (PLAN_05).

Ambiguities: greedy by combined score is implemented; if two-pair windows
show systematic mispairing (checked via PLAN_04 cross-chamber consistency),
upgrade to exhaustive assignment over ≤3×3 candidates — trivial combinatorics.

## 3D segment construction

For an accepted (x, y) pair on detector d:
- **position**: (pos_anchor_x, pos_anchor_y) at the chamber's mesh plane
  z_d (chamber positions from PLAN_04 geometry) — anchors are mesh-referenced
  by construction (earliest strip = mesh end of the column).
- **direction**: unit vector from (tanθx, tanθy) in the DETECTOR frame:
  v ∝ (tanθx, tanθy, 1) with the chamber's drift-axis orientation applied
  (sign: which side the mesh is on — per-chamber install fact, PLAN_04).
- **time**: t0 = min(t0_x, t0_y) — earliest strip = mesh-arrival ≈ particle
  crossing time (+drift≈0 at the mesh); absolute per PLAN_07. Bench: the two
  planes time the same electrons (inter-plane bias −1.3 ns, σ_Δt/√2 = 33 ns).
- **charge**: qX, qY, f — carried for PID-adjacent uses and QA.

## Output schema (3D segments)

`run, subrun, eventId, det, t0_ns, x_mm, y_mm, z_mm, tanx, tany, q_x, q_y,
f_balance, iou, n_strips_x, n_strips_y, flags_x, flags_y, pair_ambiguity`

(one row per 3D segment; keep unpaired plane segments in a side table with a
reason code — single-plane inefficiency is physics-relevant bookkeeping,
bench hit-eff ~93 % per plane means ~7 %/plane unpaired even when perfect).

## Acceptance

- On unambiguous windows, pairing reproduces the trivial assignment 100 %.
- f distributions per detector: narrow (σ68 ~ 0.07), stable vs HV and time;
  compare A/B against bench 0.487/0.531 — a shift >~0.03 means gain topology
  changed (report, don't silently adopt).
- Inter-plane Δt0 distribution: centred near 0, width tens of ns (bench 47 ns
  for the pair → 33 ns per plane). A tail = mispairings; correlate with IoU
  and f-pull to tune weights.
- Two-pair windows: fraction where BOTH pairs beat the f-pull cut; compare
  pairing choice against PLAN_04 cross-chamber linking on the subset that
  crosses two chambers (the truth for pairing).

## Gotchas

- σ68=0.07 on f is the SINGLE-track width; delta rays/overlaps broaden it —
  use the pull as a soft score, hard-cut only at ≥4.
- Saturated clusters distort f (amplitude is saturation-corrected, but
  correction quality differs per plane) — bench Δf(sat−clean) was −0.007
  (det3) to −0.037 (det2); flag saturated pairs.
- det C shares FEU numbers (7/8) with bench det3 — identity comes from the
  July map, not FEU numbers.
