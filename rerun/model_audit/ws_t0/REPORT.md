# ws_t0 — F1 / F2 / F5 / F6: t0 is one DAQ latency on the bench, 2026-10-27

(Written by the coordinating session from the workstream's final message. Commit aad6500.)

## F1 — one latency for the whole bench
All golden rows: same DREAM config (`CosmicTb_MX17.cfg`, 32 × 60 ns), scintillator-coincidence trigger. `ftst` = trigger
phase vs each FEU's 60 ns clock; every plane shows −10 ns/class (measured −7.7…−12.0). So mesh-charge arrival
t0 = C − 10 ns·ftst. C measured fit-free (earliest 30 % pulse crossing, phase-corrected), X / Y [ns]:
det2 300/309, det3 319/327, det4 334/339, det6 323/327, det7 304/304. No X–Y FEU offset (column mid-time −2…+6 ns).
The old per-detector bundle values were medians of the free fit's t0 (jumps by 60 ns bins; det3 X 290/265/243/255/229/264
across ftst — not even a −10 ns ladder); the ±5 ns prior was invisible next to χ² ~ 1e4.

## F6 — t0 is a floating degree of freedom
Bin k sits at time t0 + u_k and position p0 + w·u_k: shifting t0 and p0 together along the track draws the same line.
Only diffusion growth with depth and "no charge before the mesh" hold t0 → it wanders ±53–140 ns/event, with side
minima one 60 ns bin away; reported t0_err 1–2 ns is meaningless; p0 slides by w·Δt0 → position cost on inclined tracks.

Pin at C − 10·ftst (held-out 1320 tracks/chamber, through the bundle, vs free t0 + t0i, per-arm z refit):
mesh position vs M3 (X / Y) det2 −4/−15 %, det3 −11/−13 %, det4 −31/−20 %, det6 −12/−12 %, det7 −17/−20 % (all
significant except det2 X; M3's ~0.40 mm still included); angle s68 within ±0.03°; det4 far −1.9/−2.0 pp, others
±0.5; χ²/dof +4–20 % (model error the free t0 absorbed). C scan (600 ev × 17 values): angle scale +5 %/100 ns of C; at
C ≈ 300–330 it matches the timing v on all ten planes (−4…+3 %) with no per-chamber tuning → explains F6's +2…+5 %
(the old stored t0s were too late).

## F2 — adopt continuous t0 (`WFT_RC_T0_INTERP`): +1.5–5 % wall time; t0_err undefined 22–33 % → 0 %; angles unchanged.

## F5 — x–y coincidence
Free t0: fake x–y median −19…−55 ns, width 55–126 ns. Column mid-time (t0 + u50) centres at 0, no table needed. Once t0
is pinned, a t0 gate checks nothing → gate on column time; tolerance per chamber at p98 (free-t0 numbers: det2 206,
det3 164, det4 733, det6 529, det7 753 ns — re-measure under the pinned bundle).

## Recommendation to H
1. Pin t0 on scintillator-triggered data (bench; SPS if its trigger is the scintillator) — `set_w0kw.py
   --latency-cache <cache.pkl> --pin-t0`; planes without ftst → free t0; never pin on n_TOF (trigger ≠ particle).
2. Replace ftst-keyed `dt_xy` with the zero-offset column-time comparison.  3. Keep `WFT_RC_T0_INTERP` on.
4. Re-measure w0/kw after pinning.

## Risks
C within ±20 ns moves the angle scale ~±1 % (kw absorbs on the bench); χ² prefers C 0–40 ns earlier, position 0–20 ns
later. Scale-vs-v check relies on each bundle's v (few %). `measure_latency` selects with the M3 angle (drop the cut
without M3). SPS ZS leading edge not checked. A χ²/dof-scaled prior not tested as an alternative.

Code: `wft/calibrate.py` (measure_latency, t0_abs_from_latency, T0_PIN_SIGMA), `rerun/w12/set_w0kw.py`
(--latency-cache, --pin-t0); analysis + DAGs in this dir; data `~/x17/cosmic_bench/model_audit/ws_t0/`,
`/eos/user/d/dneff/x17rerun/model_audit/ws_t0/`. Figures (gitignored, local): figures/t0_*.png.
