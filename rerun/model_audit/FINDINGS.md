# Model audit (B4), night of 2026-10-27 — findings ledger

Scope (Dylan, 00:20): independent audit of the physical RC waveform model and its performance on the June bench
(five golden rows) and SPS det4; four figures of merit (angle resolution vs M3/uRWELL, detection efficiency / loss
budget, drift velocity / depth scale, position resolution + x-y pairing); production path + the prototype switches as
candidates; fix whatever is found; do everything on SPS incl. B3.  Tools: `rerun/model_audit/` (this directory).
Inputs: `Analysis_rr1/<golden>/wft/final_physmask/` tables (code 082ce25, PIVOT_START=1, COL_GUARD 0.01) and the
promoted `calib_bundle`s, pulled to `~/x17/cosmic_bench/model_audit/<det>/`.  `fleet_tables.py` → `fleet_tables.md`.

Status key: **FIXED** (on main), **ARM** (held-out / gate arm running), **SYST** (write down as systematic),
**LIMIT** (cannot be answered from the data), **OPEN**.

## F1 · The t0 prior is on for det2/det3/det4 and off for det6/det7 — SYST (inert)
Bundles carry `t0_prior_sigma` 5.0 (det2/3/4) / 0.0 (det6/7); the finalize tables confirm (meta `t0_prior.sigma`).
RERUN_PLAN W5 says "no prior, the phys bundles carry sigma = 0".  C18/C19: the penalty ((Δt0)/5)² is invisible next
to χ² ~1e4–1e6, so w is unchanged; its one visible effect is the t0 curvature error (F2).  Policy: set 0 everywhere
(or make the prior χ²/dof-scaled, see F6).

## F2 · t0_err is NaN on 27–31 % of det6/det7 fits; the RC kernel is a 5 ns staircase in t0 — FIXED (opt-in) + ARM
`build_matrix_rc` snaps t0 to the 5 ns tensor grid (the delay kernel interpolates), so the ±2 ns curvature is zero
whenever the two probes fall in one step; the prior's quadratic term hid it on det2/3/4 (0.1–0.3 % NaN).
`WFT_RC_T0_INTERP=1` interpolates the design matrix linearly between the two neighbouring tensors: on 40 det3 events
χ² falls on 84 % of fits (median −17), w unchanged (rσ 0.006 µm/ns), t0 continuous.  Held-out arm `t0i` on all five.

## F3 · Per-event angle errors are a constant 1° floor, not errors — SYST
`tan_err = hypot(curvature, FLOOR_TAN 0.018)`: 86–100 % of fits sit at the floor on every chamber; pull widths are
1.0–1.3 (det2/3) but 1.45–1.9 (det4/6/7), i.e. the floor is det3's resolution.  Downstream weights (n_TOF vertexing)
should use the measured per-chamber resolution, or the floor should be per chamber.  `p0_err` likewise (0.33 floor).

## F4 · The angle response is still nonlinear: implied v falls with |tan| on every chamber and both views — OPEN/ARM
Implied v (median w/tan_ref) from the |tan| 0.08–0.14 bin to the 0.28–0.45 bin: det2 X 39.7→38.0, Y 40.5→36.5;
det3 X 37.7→36.0, Y 36.6→35.6; det4 X 34.2→30.7, Y 35.7→31.3; det6 X 27.3→23.3, Y 28.1→24.9; det7 X 40.1→37.4,
Y 39.9→36.4 (`angle_response.json`).  Local slope dtan_fit/dtan_ref at |tan| 0.3–0.45: det3 X 0.95, Y 0.81; det4 X
0.84; signed bias at tan +0.28…+0.45: det3 X −0.37°, det4 X −0.48°, det6 X −1.26°, Y −1.54°, det7 X −0.18°; at
tan −0.45…−0.28 det3 X +0.07°, det4 X +0.10°, det6 X +0.67° — one-sided in X on det3/det4/det6 (the X asymmetry's
sign), symmetric on det7.  Saturated events read 1–5 % shallower in Y (det3 Y 0.955 vs 1.005).  A single kw cannot
absorb this.  Arms: `ts` (two-sided start: basin or model?), `asymp/asymn/asymp07` (one-sided delayed copy in the RC
kernel, new opt-in keys `rc_asym_x`, `rc_asym_tau_x`), `s0x20`, `sat`.

## F5 · The x–y t0 coincidence gate fails on 15–44 % of genuine single tracks; dt_xy is stale — FIXED (tooling)
Residual (t0x − t0y) − dt_xy[ftst class]: median −48 (det2) / −60 ns (det3), rσ 69–144 ns, outside the 120 ns
tolerance 20 / 24 / 15 / 34 / 44 % (det2/3/4/6/7).  Re-measured per ftst class the two classes sit exactly one depth
bin (60 ns) apart (det3: 0 / −60; old 5.4 / 30.2 — the old values are the old kernel's t0 definition); det4's dt_xy
was EMPTY (default −18.8 for every event).  With the new offsets 12 / 13 / 14 / 30 / 40 % still fall outside 120 ns:
the per-plane t0 itself is uncertain by ~50 ns (det2/3) to ~100 ns (det6/7).  Consequence: `n_tracks` = 0 on 40–80 %
of single-track events (det3 2843/7093), the multi-track gate, the pairing swap, head-on rescue and two-track
corroboration all key on it; the beam seeder inherits it.  Fix: `set_w0kw.py --table` (finalize) now re-measures
dt_xy and writes the `xy_pairing` block (lq, u50, u90, t0 medians + robust widths); the gate tolerance should follow
the measured width (2.5 rσ keeps 92–97 %).  t0 + u50 is tighter on det2/det3 (rσ 44–48 ns) but not on det4/6/7.

## F6 · The t0 prior as written cannot act — SYST (design)
An unscaled ((t0 − t0p)/σ)² against χ² with χ²/dof 60–700 is a 1e−3 perturbation.  Either scale the prior by the
fit's χ²/dof (equivalently use `WFT_MODEL_FRAC` so χ²/dof ~ 1) or pin t0 (arm `t0fix`: t0 fixed at the trigger
prediction; tests whether depth/coincidence improve at no angle cost).

## F7 · Column-duration plausibility window hard-coded for v = 36.6 — FIXED
`U_MIN/U_MAX = 250/1100 ns` flagged 7 / 14 % (det3 X/Y), 17 % (det2 Y), 21–23 % (det4), 24–42 % (det7), 38–60 %
(det6) of M3-confirmed single tracks "implausible"; the ranking key (plausible, dchi2) then prefers any noise candidate
whose q_uend falls inside, and the multi-track gate requires plausibility.  Now `u_window(cal)` = (0.30, 1.60) × 30 mm/v
(`WFT_U_FIXED=1` restores the constants); recorded in `meta.selection.u_window_ns`.  Affects the full pass (many
candidates per plane), the vrefit rows (folded v 15.8 → T 1900 ns) and the beam.

## F8 · The ladder (reco angle scale) differs between X and Y on det3 by 2.2 % (7σ) — SYST
Robust ladder v over 0.10 < |tan| < 0.45: det3 X 36.47 ± 0.07 / Y 35.69 ± 0.07 (timing 35.8); det4 31.4 / 32.3
(33.0); det6 24.8 / 24.3 (24.17); det7 37.0 / 36.7 (36.8); det2 38.46 / 38.44 (39.3).  kw absorbs it on the bench
(det3 kw_x 1.021 / kw_y 0.996); off the bench (n_TOF) the X scale is a +2 % systematic unless kw is transferred.

## F9 · w0_y is negative on every chamber (−0.16…−0.40 µm/ns = −0.26…−0.95°), w0_x ≈ 0 except det2 — SYST
Not a telescope tilt (the angle differs by chamber; on det3 it changes sign with drift field: 1000 V −0.164, 700 V
−0.099, 500 V +0.101 µm/ns, dry-run w0kw.json) — a field-dependent Y-kernel skew, calibrated per v group by the vrefit
nodes.  Fine for the pass; a stated systematic wherever w0 cannot be measured.

## F10 · Promoted det7 bundle has empty `conditions` — FIXED? (record)
`check_conditions` cannot check it.  Harmless now; fill at promotion.

## F11 · det4 `dt_xy` empty; `xy_pairing` empty on all five — covered by F5 tooling.

## F12 · SPS: the bench det4 dead list is in bench channel numbers — B3 requirement
FEU 03 at H4 has a different cabling (inverted connectors); a bundle `dead` list is applied by channel number
(`prep_plane`), so the bench mask would censor the wrong strips.  SPS bundles need the mask remapped by position (or
dropped: `heldout_arm.py --no-dead`).  ZS data need exactly-zero samples censored: `WFT_ZS_CENSOR=1` (new, opt-in).

## F13 · The K = 24 depth grid truncates ~10 % of det6/det7 columns — ARM (k30)
`q_uend` p90 = 1440 ns = K × 60 on det6 (both views) and det7 (both views); det6 Y p50 1200.  C5 set K = 24 for
1000 V rows and 30 for ≤ 500 V, but det6/det7 ran at 700 V (243 V/cm) and det6 drifts at 24 µm/ns (T ≈ 1240 ns before
diffusion and RC spread).  A column ending on the last bin is either truncated or the NNLS parking charge there
(COL_GUARD keeps columns with weighted norm ≥ 1 % of the largest).  Held-out arm `k30` measures the angle effect.

## F14 · Position resolution of the finalize alignment (M3 residual, |r| < 10 mm; reference-limited at ~0.40 mm) — record
det2 0.504 / 0.508, det3 0.522 / 0.582, det4 0.560 / 0.734, det6 0.513 / 0.508, det7 0.643 / 0.762 mm (X / Y).  det3's
Y and det4/det7's Y are 10–30 % worse than X; det6 the best in Y despite the worst angles (position at the mesh is a
different quantity from the slope).  No pointing-estimator problem found here.

## F15 · x–y charge balance is chamber-specific: ln(qx/qy) rσ det3 0.058, det2 0.073, det4 0.11, det6 0.19, det7 0.28 — SYST
The pairing-by-charge rule (satstudy: 96–100 % on det3) rests on det3's 0.054–0.058 width.  On det7 (X card 1 shorts +
pickup) and det6 the width is 3–5× larger: charge pairing is weak there, and any beam chamber must have its own width
measured (now written to the bundle's `xy_pairing` by `set_w0kw.py --table`).  det7's median ln(qx/qy) is −0.10 (X
reads 10 % less charge than Y).
