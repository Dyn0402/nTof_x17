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

## F6 · The t0 prior as written cannot act; pinning t0 at the stored trigger prediction moves the ladder — SYST (design)
An unscaled ((t0 − t0p)/σ)² against χ² with χ²/dof 60–700 is a 1e−3 perturbation.  Either scale the prior by the
fit's χ²/dof (equivalently use `WFT_MODEL_FRAC` so χ²/dof ~ 1) or pin t0.  Arm `t0fix` (t0 pinned at the bundle's
`t0_abs` per ftst class): the x–y t0 width collapses (134 → 14 ns det7, 91 → 35 det3) but the ladder moves +2…+5 %
(det2 38.7→39.4, det4 33.1→34.6, det7 36.5→37.6/38.5) and χ²/dof rises 10–30 %: the stored `t0_abs` is another
kernel's t0 definition (stale like dt_xy, F5).  Resolution unchanged within errors except det4 X (+0.09 ± 0.05).
A phys-kernel `t0_abs` (re-measured by the finalize step) would be the depth/coincidence lever; not an angle lever.

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

## F13 · The K = 24 depth grid and the det6/det7 columns ending on the last bin — CLOSED (null)
`q_uend` p90 = 1440 ns = K × 60 on det6 (both views) and det7 (both views); det6 Y p50 1200.  C5 set K = 24 for
1000 V rows and 30 for ≤ 500 V, but det6/det7 ran at 700 V (243 V/cm) and det6 drifts at 24 µm/ns (T ≈ 1240 ns before
diffusion and RC spread).  A column ending on the last bin is either truncated or the NNLS parking charge there
(COL_GUARD keeps columns with weighted norm ≥ 1 % of the largest).  Held-out arm `k30` (K = 30) on all five chambers:
ladder, s68, far and χ² unchanged to ±0.01° — the grid is not a lever; a column ending on the last bin is NNLS parking.

## F14 · Position resolution of the finalize alignment (M3 residual, |r| < 10 mm; reference-limited at ~0.40 mm) — record
det2 0.504 / 0.508, det3 0.522 / 0.582, det4 0.560 / 0.734, det6 0.513 / 0.508, det7 0.643 / 0.762 mm (X / Y).  det3's
Y and det4/det7's Y are 10–30 % worse than X; det6 the best in Y despite the worst angles (position at the mesh is a
different quantity from the slope).  No pointing-estimator problem found here.

## F15 · x–y charge balance is chamber-specific: ln(qx/qy) rσ det3 0.058, det2 0.073, det4 0.11, det6 0.19, det7 0.28 — SYST
The pairing-by-charge rule (satstudy: 96–100 % on det3) rests on det3's 0.054–0.058 width.  On det7 (X card 1 shorts +
pickup) and det6 the width is 3–5× larger: charge pairing is weak there, and any beam chamber must have its own width
measured (now written to the bundle's `xy_pairing` by `set_w0kw.py --table`).  det7's median ln(qx/qy) is −0.10 (X
reads 10 % less charge than Y).

## F16 · SPS known-angle test: the fitted slope agrees with the fit's own column duration to 4 % at 235 V/cm; the 10 % deficit against the span-route v is a reference problem, and below 200 V/cm the column overruns the depth grid — REFRAMED (see addendum)
run_63 operating_00, 25.64° mount, Ar/CF4/iso, 64 samples, self-seeded, bench det4 bundle (dead list dropped, v = span
v(E), K covering the column), no uRWELL (`sps_build_cache.py`, `heldout_arm.py`, 1000 events per plateau).  The
inclined view is Y (X carries the known tilt, −1.9° at v 14).  |w_y| / tan 25.64° against the span-route v:
235 V/cm 12.62 vs 14.0 (0.90); 165 V/cm 8.53 vs 11.6 (0.74); 148 V/cm 7.59 vs 11.6 (0.65) — θ(v_span) = −23.4°,
−19.4°, −17.4°; per-event rσ 1.4–1.7°, 11–16 % beyond 35°; χ²/dof 30–65.  At the operating field the angle scale
is 10 % low with no reference involved (mount angle and the column duration are both in the data), at the low fields
more.  With the ZS samples censored (`sps2`, the right treatment: χ²/dof 31 → 135 shows how cheap the zeros were) the
ratio is unchanged — 0.892 / 0.868 / 0.840 / 0.798 / 0.727 / 0.646 at 235 / 217 / 200 / 182 / 165 / 148 V/cm — so ZS
truncation is not the cause; the deficit grows smoothly with the drift time (2.1 → 2.5 µs), i.e. the lateral motion
per unit drift time is under-estimated the longer the column.  The low-field span v is an upper bound (truncated
span), so part of the low-field deficit may be the reference; the 235 V/cm point is anchored on run_71's end lobe.
t0 interpolation changes nothing here either.  The bench kw_y (0.977) does not transfer: on the beam the steep-track compression of F4 (5–20 % at |tan|
0.3–0.45) continues to 10 % at tan 0.48.  This is the one meaningful improvement left in the angle response.

## F17 · A fractional model-error term (WFT_MODEL_FRAC 0.03) does not buy resolution — CLOSED (null / slightly worse)
det3 held-out, t0 interpolation + MODEL_FRAC 0.03: χ²/dof 41 → 10 (X), 58 → 14 (Y) — the term absorbs the bright-sample
template mismatch as designed — but s68 unchanged within ±0.03°, far +1.2 / +2.0 pp, x–y t0 width unchanged (91 → 99 ns).
The production weighting (noise only) stays.  Other chambers in `heldout2` (mf03, mf05) for the record.

## F18 · The per-event σ0 profile trades footprint width for tilt: fits that chose the wide σ0 read 6–12 % shallower — OPEN (B1 input), ARM (noprof)
Held-out base fits (`heldout/<det>_base.npz`, column sigma_used), |tan_ref| 0.10–0.25, median tan_fit/tan_ref for the
fits that kept the narrow footprint vs those whose lowest χ² was the wide one (0.42 / 0.50 mm): det3 X 1.008 vs 0.974
(13 % wide), Y 1.007 vs 0.882 (5 %); det4 X 1.017 vs 0.938 (23 %), Y 1.000 vs 0.990 (14 %); det6 X 1.028 vs 0.927 (13 %),
Y 1.024 vs 0.917 (18 %); det7 X 1.038 vs 0.876 (26 %), Y 1.033 vs 0.896 (36 %).  A wider prompt footprint explains part
of an inclined track's lateral extent without tilt, and the lowest-χ² choice takes it.  The single kw then averages
two populations (narrow 1.01–1.04, wide 0.88–0.94): the mixture is part of the compression of F4, and the profile's
wide fraction grows with charge, so bright/steep tracks are the most compressed.  C21 adopted the profile for the
head-on far tail on det4 (−3.9 pp) and for saturated events; the trade is measured by the `noprof` arm (held-out, all
five; SPS `sps3`).  B1 should choose σ0 from a measured quantity (saturation, charge) rather than by χ², or fit the
width at fixed tilt.  `steep_probe.py`: at the M3 angle the model's χ² is higher than at the fit in 94–95 % of det3's
steep tracks (Δχ²/dof +2.5…+3.1), so the compression is the model's preference, not an optimiser miss (5–6 %).

## F19 · Held-out fleet verdicts of the environment arms (1320 events × 5 chambers, event-paired; `heldout_compare_env.md`)
- `t0i` (RC t0 interpolation): s68 within ±0.02° everywhere (det3 X −0.023 ± 0.013), far ±0.4 pp, ladder ≤ 0.5 %, χ²/dof
  −1…−4 %, t0_err defined: adopt as a correctness change (continuous t0), not an angle lever.
- `mf03` / `mf05` (fractional model error): χ²/dof falls to a quarter but far rises +1…+12 pp and s68 worsens by
  0.1–0.25° on det6/det7: reject (the bright-sample weighting is what pins the track).
- `sat` (saturated samples re-solved at the clip): s68 ±0.015°, far −0.1…−0.4 pp on det3/det4/det7, χ² −2…−10 %:
  harmless small gain, optional.
- `ts` (two-sided start): det4 X far −1.2 ± 0.4 pp, otherwise null; steep-bin biases unchanged — the optimiser is not
  the compression (F4), consistent with `steep_probe.py`.
- `k30`, `t0fix`: F13, F6.

## F20 · B3 dry run: `sps_reco.py` runs end to end on condor; ZS data are a poor input for the waveform fit — DONE / LIMIT
`run63_flat__flat700` (flat mount, 243 V/cm, ZS 4σ, 64 samples, 1500 events, 25 min incl. the EOS read): both views
fit on 100 % of events; Y head-on median +0.01° (rσ 1.15°), X −1.25° (the known det4 X tilt), position vs own centroid
rσ 0.6 mm.  But: far > 5° 25 % (X) / 35 % (Y), χ²/dof 100–120, x–y t0 width 312 ns, ln(qx/qy) median −0.33 (det4's X
gain stripes) with rσ 0.45, and only 5–6 strips per window carry samples — the zero suppression removes the faint
neighbour and tail samples the fit needs for the angle and t0 of a head-on column.  So on SPS the ZS runs give the
median angle scale (F16) and little else; the RAW run_71 (flat, 700/450/275 V) is the clean head-on set.  The pass
registry's K gets a +10-bin margin (uend sat on the grid end at K = 40).  `run63_rot25__d675` still running.

F18 addendum (`steep_probe.py` base, held-out |tan_ref| ≥ 0.25, 100–120 events per chamber, p0 re-scanned at the M3
angle): the model prefers the fitted angle in 93–96 % (det3), 83–90 % (det4), 90–91 % (det7) of steep tracks — the
optimiser-miss rate is 5–7 % on det3 but 10–17 % on det4/det7, which is what the two-sided start recovers (det4 X far
−1.2 pp).  Median w_fit/w_ref on these samples: det3 0.982 / 0.988, det4 1.037 / 1.057 (against the bench kw of 0.953 /
0.977: the held-out sample scatters ±0.15 in 26–31 % of events), det7 0.982 / 0.943 (X / Y).

F16 addendum (ZS-censored `sps2` base, all plateaus): |w_y|/tan against the fit's OWN column, v_col = 28.8 mm / uend:
235 V/cm 12.48 vs 12.00 (**1.04**), 217 11.45 vs 11.43 (1.00), 200 10.42 vs 10.67 (0.98); below that uend sits on the
grid end (2880 ns = K·60 at K = 48) so the column is truncated and the ratio (0.92 → 0.75) is an artefact of the
depth grid, not physics — the registry K margin is raised to +15 bins (K ≈ 58 at 148 V/cm, inside the 3.84 µs window).
The external span-route v is 14 % above v_col at EVERY plateau (v_col / v_span 0.857–0.866): a constant factor, i.e.
the run_71 end-lobe anchor (14.0 µm/ns at 233 V/cm) or the 28.8 mm gap, not the model.  None of the model terms
moves the ratio (dp ×0.7 / ×1.3: 0.901 / 0.882; noprof 0.895; rcd ±0.003; asym 0; t0i 0.890).  Conclusion: at 25.6°
the model is self-consistent to +4 % (the same size as det4's bench steep-bin effect); the open question for the beam
is the absolute v (anchor / gap), which this audit cannot settle from the SPS data alone (LIMIT: the span route and
the column both need an independent gap).

## F21 · Held-out model-term arms (1320 events × 5 chambers; `heldout_compare_cli.md`) — the width–tilt trade
- `noprof` (σ0 profile off): ladder +0.4…+1.2 % on every chamber (the wide-footprint fits were compressing), s68
  unchanged (±0.03°), far +0.5…+2.4 pp (det4 +2.3/+2.4, the head-on / saturated rescue C21 adopted it for); the steep
  biases do not move beyond ±0.2°.  The profile is ~1 % of scale, not the 5–20 % steep-bin compression.
- `dp_lo` / `dp_hi` (Dp × 0.7 / × 1.3): ladder +1.1…+2.2 % / −1.2…−2.7 %; far +0.2…+2.3 / −0.2…−1.8 pp; s68 ±0.07.
  `rcd_hi` (rc_D_y × 1.5): Y ladder −1.0…−2.6 %, s68 incl +0.07…+0.10 (det3, det6), χ² up.  Every extra unit of modelled
  lateral width (footprint, diffusion, RC) lowers the fitted slope: the width–tilt degeneracy is the lever behind F4,
  and the data prefer LESS width at depth than the model carries on the steep tracks.  A term that narrows the deep
  footprint for inclined tracks (or a per-event Dp) is the B1 candidate; not tested tonight.
- `asymp` / `asymn` / `asymp07` (X one-sided copy ±0.13 / +0.07): det7 X s68 −0.065 ± 0.036 (+0.13) / −0.039 ± 0.037
  (+0.07), det4 X far −1.7 ± 0.5 pp (+0.13; −0.8 with +0.07), det6 X s68 −0.03…−0.06 (either sign, χ² up), det2/det3
  null or worse; steep biases move ±0.4–0.8° in opposite directions for the two signs.  A per-chamber amplitude
  (aQ: det3 +0.17, det7 +0.19, det2/det4 +0.08, det6 −0.02) is required before this is a gain; at most 0.05° on det7.
- `s0x20` (σ0_x 0.20): null everywhere (±0.03°).

## F22 · Fleet gate of the prototype switches (8000 M3-matched events per golden row, interior X+Y at 5 mm; `gate_compare.md`) — B2
| arm | Δ detected (det2 / det3 / det4 / det6 / det7) [pt] | Δ far [pp] | Δ s68 x / y [°] | note |
|---|---|---|---|---|
| band veto (`DROP_FLAT_HITS=1`) | +0.71 / +0.82 / +0.02 / +0.35 / +1.07 | −0.03 / −0.08 / −0.11 / +0.18 / −0.03 | ≤ 0.005 except det7 X +0.09 | rejected_cut falls 0.3–0.7 pt; det7 X s68 rises from the added card-1 tracks |
| head-on rescue (`HEADON_SIG=20`, `HEADON_RESCUE=1`) | +0.38 / +0.26 / **+3.15** / +1.44 / **+3.67** | ±0.05, det6 +0.27 | det4 +0.08 / +0.21, det7 X +0.09 | fit_failed → 0; the rescued tracks have poorer angles (low SNR), not a degradation of the others |
| two-track fit (`TWO_TRACK_FIT=1`) | 0.00 / +0.03 / +0.02 / −0.04 / +0.01 | −0.03 / −0.12 / −0.12 / **+0.26** / −0.01 | ≤ 0.01, det6 Y +0.026 | splits 47 / 76 / 155 / **417** / 100 — det6 accepts 5 % splits (noisy chamber): the false-split bound must be re-measured on det6 before production |
| all three | +1.12 / +1.15 / +3.20 / +1.77 / (det7: gate2 node RECO_det7_all still running at 03:50; the row in gate_compare.md is the old env-less copy = prod) | −0.09 / −0.15 / −0.07 / +0.25 / — | det4 +0.09 / +0.22 | additive on det2–det6 |
Resolution of the pre-existing tracks is untouched by the band veto and the two-track fit; the head-on rescue adds
1.4–3.7 pt of detection on the weak chambers at the price of a poorer-angle population.  det3's +1.15 pt reproduces the
satstudy numbers (band 96.9 %, head-on 17/17).  Verdict for B2: band veto + head-on rescue pass the fleet gate; the
two-track fit needs det6's split rate understood (417 splits) before it is on by default.

F20 addendum: `run63_rot25__d675` through the B3 stage (1500 events, K 40, 64 samples, ZS censored): Y |w|/tan 12.50 vs
28.8/uend 12.00 (+4 %, θ = −26.6° at the column v), X −1.8° (the tilt), far > 10° 25 % (Y) / 18 % (X), χ²/dof 105–135 —
the same numbers as the audit harness, so the stage and the harness agree.  Both SPS rows of the dry run are at
`/eos/user/d/dneff/x17rerun/model_audit/sps_pass/`.  AFS quota at the end of the night: 88 % (the ten `~/x17rerun/code_*`
archives, 5.6 GB, are the bulk; `lxstore offload` them before the pass).
