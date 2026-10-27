# ws_unify — F22 (+ F7, F10, F11, F13): one reconstruction, 2026-10-27

(Written by the coordinating session from the workstream's final message. Commits 03ed6c4, 7731777, 5fb571d, 8587c43,
4e5f56a. Gate code `/eos/user/d/dneff/x17rerun/code_ccb272e`; gate bundles `Analysis_audit/<row>/wft/calib_bundle_xyp`.)

## Design
`wft` no longer reads `WFT_*` environment variables: the algorithm = code constants + the calibration bundle. Study
variants go through one hook, `wft/diag.py` (old `WFT_*` names still accepted there; a default value is a no-op, any
other value is recorded as an override in each table's meta and in the STAGES logs). `rerun/stage_job.py` refuses a
production stage with any override unless `--diag`. Tests 95/95 incl. `wft/tests/test_diag.py`.

## Switch dispositions
| switch | now |
|---|---|
| PIVOT_START, COL_GUARD, DEAD_SEED | always on (first two were already pass settings; dead seed already default) |
| band veto (DROP_FLAT_HITS), head-on seeds + rescue | always on (fleet gate pass) |
| RC t0 interpolation, saturated-sample re-solve | always on (correctness) |
| two-track fit | always on; minimum separation 1.2 mm → 4 pitches (3.11 mm) |
| x–y t0 coincidence window | bundle: 2.5 × measured rσ(t0x − t0y); fallback 120 ns |
| head-on charge-match tolerance | bundle's own charge balance; fallback det3 |
| per-event angle error floor | bundle `tan_floor` (written by `set_w0kw.py`, ws_floor); fallback 0.018 |
| ZS samples | measured zeros (ws_sps); front-end-dropped RAW samples marked missing by the reader |
| MODEL_FRAC, TWO_SIDED, CM_MASK, local floor / split-gap / hot-bridge, ZS_BOUND, two-track knobs, PAIR_SELECT, TAN_MAX | fixed constants; study override only |
| WFT_U_FIXED | deleted |

**F7**: neither hard-coded nor loosened ad hoc: window = (0.30, 1.60) × 30 mm / v_bundle (det3 250–1330 ns, det6
375–2000 ns). Upper edge 1.34 → 1.60 drift times because the old one rejected 7–17 % of M3-confirmed det2/det3 singles.

## Gate3: unified vs production (8000 M3 muons / chamber; `figures/gate_delta.png`)
| | det2 | det3 | det4 | det6 | det7 |
|---|---|---|---|---|---|
| detected (5 mm) | +1.12 pt | +1.15 | +3.70 | +1.85 | +5.45 |
| far | −0.12 pp | −0.19 | −0.06 | +0.13 | −0.02 |
| s68 on production's own tracks X / Y | −0.005 / −0.019° | −0.014 / +0.002 | −0.019 / −0.040 | −0.032 / −0.010 | −0.007 / −0.010 |
| overall s68 X / Y | +0.008 / −0.023 | −0.001 / −0.006 | +0.064 / +0.180 | −0.005 / −0.037 | +0.120 / −0.023 |
Existing tracks improve slightly on every view; overall s68 rises on det4 and det7 X only from the newly found
(lower-quality) head-on / det7 X card-1 tracks. Refactor check (all new pieces off vs production): 96.2–99.8 %
identical; the rest is the F7 window (largest on det6/det4).

## det6 two-track splits
The 417 were the brightest single muons with the worst χ² (p94–p99), refitted as two parallel tracks ~2 mm apart sharing
charge ~40/60 — 4.1 % of clean singles on det6 Y. Real two-muon overlays spread to 16 mm (`figures/split_sep.png`).
With the 4-pitch minimum, false splits on clean singles are 0.05–0.72 % on every chamber/view (det6 Y highest), inside
the 1 % bound (`figures/split_rate.png`).

## F10 / F11 / F13
F10: `rerun/promote_phys.py` fills empty conditions at promotion (`--fill-conditions` in place); not yet applied to
det7's promoted bundle (lxplus; re-promotion fills it). F11: `set_w0kw.py --table` (+ new `--pairing-only`). F13: none.

## Other workstreams — status in the single path (all bundle content, not switches)
- D (t0): latency pin enters via the bundle's t0 values; the window re-measures under a pinned bundle. Column-time
  (t0 + u50) comparison NOT implemented.
- A / B (σ0, bow): A's angle map read from the bundle. Fleet-fixed σ0 + dropping the per-event σ0 choice = a
  finalize bundle edit — NOT done, NOT gated.
- G: per-event-class charge-pairing widths NOT implemented (one width per chamber).
- E: 20 ns charge grid untested.
- F: "no bench w0 in n_TOF bundles" is a transfer rule; nothing enforces it.

## Risks
- `wft_beam` (n_TOF) inherits the new defaults (two-track fit with 4-pitch minimum, saturated re-solve, continuous t0,
  column guard, pivot start); pair efficiency at small separation NOT re-measured. Band veto / head-on rescue act only
  in the bench driver.
- Gate bundles' t0 offsets / pairing widths were measured on the production table (before the saturated change) →
  re-finalize (`set_w0kw.py --table`) and promote under the unified algorithm.
- SPS changes not gated (and F16/F20 SPS numbers used the 256-ADC-offset caches, ws_sps).
- `make_gate_dag.py` writes a `run_gate.sh` taking the row key as an extra argument; AFS copy is the new one.
