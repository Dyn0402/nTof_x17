# ws_spsv — F16: the SPS known-angle test and the absolute drift velocity, 2026-10-27

(Written by the coordinating session from the workstream's final message; the workstream could not write this file.
Commit 23338ad.)

## Answer
**The old "14 %" was a setup artefact.** The old fits sized their depth grid from the span-route v; the fitted column
end `q_uend` sat exactly on the grid end at every plateau (2400, 2520, 2700, 2820, 2880, 2880 ns), so
v_span / (28.8 mm / q_uend) = 1.155–1.167 by arithmetic. The "+4 % self-consistency" compared the slope with the grid
end — not a measurement either. Those fits also used the 256-ADC-offset caches (ws_sps).

**Column duration, model-free.** Summed over strips the signal is a box of length T = gap / v convolved with the
electronics template; angle terms cancel. At 243 V/cm: head-on RAW run_71 1970 ns, head-on ZS run_63 2050, 25.64° ZS
run_62 2040 (Y) / 2130 (X) → **T = 2.05 ± 0.07 µs**. Flat box reads 2–5 % shorter than box × charge gradient; charge falls
~50 % along the column (late charge lost).

**Absolute v (det4 at SPS, gap / T)** [µm/ns, 28.8 mm (30 mm)]: 243 V/cm 14.1 (14.7); 235: 13.5 (14.1); 217: 12.7
(13.3); 200: 11.2 (11.6); 174–182: 9.6–10.4 (10.0–10.8). Below ~170 V/cm the column overruns the 3.84 µs window
(bounds). **v(243 V/cm) = 14.1 +0.6 −1.1 µm/ns** (box estimator, gap 26.5–30 mm, run-to-run) ≈ 1.4–1.6 % water vs
Magboltz for the SPS gas. HV readback = set; whether mesh voltage adds to the drift field is not in any record, so
fields are HV / 28.8 mm.

**Span route retired.** Its run_71 "end lobe" is the column's falling edge (~2650 ns) through the shaper undershoot;
its "233 V/cm" label is 700 V / 30 mm while the registry uses 28.8 mm, so the auto-anchor attached the 700 V v to the
675 V plateau; and it is built on combined_hits times plus a fitted shaping constant (not a reconstruction basis). It
reads 9–13 % high at 182–200 V/cm.

## What remains (LIMIT)
Fitted slope |w|/tan ≈ **8 % below 28.8 mm / T at every field 174–243 V/cm** (8 plateaus, 2 runs, both ZS treatments).
Slope × duration gives the gap with no v: |w|·T/tan = 26.2–27.0 mm (25.0–26.4 flat box).
- Not the model widths: Dp 0.020 → 0.010 / 0.006 and σ0_y = 0.32 with no profile move the slope ≤ 0.7 % (χ²/dof
  7.3 → 5.9); censored vs measured zeros ≤ 0.4 %.
- Local gap can be neither confirmed nor ruled out: det4's bench gap reading is unusable (25.5 mm overall, charge
  quartiles 24.4–33.5 mm); its ±3 mm bow allows ~26.5 mm at the beam spot.
- Against a pure gap: run_61 at the same field reads slope 11.38 at 15.465° and 12.67 at 25.64° — a gap error cannot do
  that (fit compression with angle, a ~1.5° error in the 15° mount, or the 32-sample truncation; not separable).

Angle scale: with v = 28.8 mm / T, 25.64° reads 8 % low in tan (23.8°). With the slope as v it closes by construction at
25.64° but carries ~10 % uncertainty to other angles (15° point). **Truth: −8 % ± 3 %, owed to the local gap or
angle-dependent fit compression; existing data cannot split them.** The old +4 % is withdrawn.

## Figures (`figures/`, gitignored, main checkout)
f16_concepts (side view, column box, ladder), f16_grid_artefact, f16_column_243, f16_v_of_E, f16_gap_eff.

## Code / data
`ws_spsv/`: qfit.py, build_raw71_cache.py, boxfit.py, edges.py, zsstack.py, analyse.py, dp_arms.py, make_figures.py,
fig_concepts.py; results/*.json. No wft behaviour changed. Valid EOS outputs: `model_audit/ws_spsv/{qfixed,qdp,fixed}/`;
the top-level `q_*.npz` and `cache/` there were built on offset caches — **invalid**. Four fixed caches rebuilt here
(run_62, run_63 flat, run_61 op25 and rot15) with main's fixed `sps_build_cache.py`; run_63 25.64° ones are ws_sps's.

## Risks
Box estimator assumes the bench det4 X template at H4 (same DREAM registers; not checked further); ~11 % of inclined
events truncated by the ±16-strip window (strip sum incomplete, not corrected); the 15° vs 25.64° slope difference is
unexplained and the mount angle unchecked; run_71 RAW caches use a per-strip pre-window baseline without CM correction.
`/eos/user/d/dneff/x17rerun/code_ws_spsv` may be partly deleted (background rm not confirmed; nothing uses it).
