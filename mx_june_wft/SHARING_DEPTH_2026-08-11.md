# SHARING_DEPTH_2026-08-11 — model-free X/Y sharing-mechanism measurement (T1.2 first leg)

**Script:** `09_xy_sharing_depth.py` (bench_cache_ftst.pkl, no fits, no model).
**Method:** near-vertical tracks (|tan_ref| < 0.05, central 300–3550 ADC),
central and ±1/±2 neighbour waveforms averaged after aligning each event to the
central strip's sub-sample half-max crossing. `centered` selection = left/right
neighbour peaks symmetric to < 25 % (track hit the middle of the strip), which
removes the impact-position mixture; it barely changes the curves, so the
mixture was not dominating. n = 945/1318 (all), 333/607 (centered), X/Y.

## What the curves say (det3, 490 V / 1000 V)

| observable | X | Y | reading |
|---|---|---|---|
| ±1 peak / central peak | 0.38 | 0.42 | comparable at ±1 |
| ±1 peak delay vs central | ~+70 ns | ~+90 ns, broader | both dispersed, Y more |
| **±2 peak / central** | **0.06** | **0.15** | the mechanism split: cascaded (twice-dispersed) transport is ~2.5× stronger on Y |
| ±2 peak time | ~+350 ns | ~+430 ns | both slow |
| past the column end (t > 1000 ns) | ±1, ±2 → ~0, tracking the central's small (−2 %) undershoot | ±2 stays clearly **positive** while the central undershoots −10 %; ±1 in between | the RC keeps discharging into Y neighbours after the direct signal ends — the model-free RC signature, nearly absent on X |
| central undershoot | ~−2 % | **~−10 %** | reproduces the T14 "deep Y undershoot is detector-side" number independently |

## Verdict for F6 / T1.2

- The **asymmetry is real and has the predicted sign**: Y carries the strong,
  slow, cascaded transport; the post-column-end behaviour separates the
  mechanisms without any model.
- But the strong form "X cannot have resistive sharing at all" is **too strong
  for det3 data**: X's ±1 copy is also delayed (~+70 ns) and X's ±2 is small
  but dispersed, not prompt. A weak slow channel exists on X — candidates:
  inter-strip capacitance, imperfect groove isolation, or the readout chain.
  Pure prompt diffusion cannot produce a delayed copy, so a depth-dependent
  `c1_X` should REPLACE only part of X's kernel, not all of it.
- Model hook is in place: `hyper['cX']` scales the discrete kernel on X only
  (default 1.0 = today's behaviour), so the T1.2 arms are `hyper_patch`
  variants: `{cX: 0}` + Dp/sigma_p0 refit vs production, judged on
  implied-velocity flatness.

## Caveats

- The ±1/central ratio at early/late times blows up numerically where the
  central is small; curves are cut at central > 0.02.
- The alignment reference is the central strip's half-max, so all times are
  relative to the central rise, not to t0; peak *delays* are unaffected.
- Averages mix track depths uniformly (cosmics): the depth-resolved version
  (ratio vs drift time *within* the column) is the next leg, and is what the
  simulation's c1_X(z) ×45 prediction actually addresses.
