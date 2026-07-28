# The reconstruction basis: waveforms, not hits

**Decided 2026-07-28. This is the canonical statement; other documents link
here rather than restating it.**

## The rule

> **Never reconstruct position, angle or drift depth from `combined_hits` times.**
> A per-strip hit time is an *aggregate* of that strip's own charge and delayed,
> dispersed copies of its neighbours' charge. It is not a drift-time measurement
> and no threshold, CFD or matched-filter refinement makes it one.
>
> Hits remain legitimate for two things: **finding candidates** (which strips and
> which events to look at) and **QA/monitoring** (rates, amplitudes, occupancy,
> detection efficiency). Everything geometric comes from the waveforms via
> `wft/`.

## Why — the measurement

The MX17 strips are resistive and capacitively coupled. Each strip's waveform is
its own charge plus **~29 % of each ±1 neighbour delayed by τ ≈ 47 ns** and
~5 % of each ±2 neighbour at 2τ (det3 numbers; kY = 1.375 stronger on Y, and
det4's kernel differs — the kernel is per detector).

Consequently:

- a strip near the **mesh** (its charge arrives early) also receives *late*
  copies from neighbours further up the gap → its single hit time is pulled
  **later**;
- a strip near the **cathode** (charge arrives late) receives *early* copies from
  neighbours nearer the mesh → its time is pulled **earlier**.

Both ends squeeze inward: the time ladder is compressed by **20–30 %**, which at
a fixed drift velocity reads as a track **~4° too steep**, and every strip lands
at the wrong depth. Measured on 600 det3 muons (`THREADING_DISPLAYS_2026-07-28.md`),
charge-weighted median |cluster − M3 reference line| against drift depth:

| depth [mm] | 3 | 9 | 15 | 21 | 26.5 |
|---|---|---|---|---|---|
| production (hits), common frame | 0.45 | 0.50 | 0.61 | 0.67 | 0.63 |
| production (hits), own t0 and v — what the old 3-D displays draw | 0.43 | 0.63 | 0.88 | 1.07 | 1.17 |
| **waveform-first** | **0.39** | **0.44** | **0.49** | **0.50** | **0.55** |

Both agree at the mesh, at the ~0.4 mm M3 pointing floor. The hits-based cluster
then walks away from the true track as it ascends the gap; the waveform-first
one does not. The comparison is not circular: the displayed cluster comes from a
line-free 2-D deconvolution and a sub-pitch free ladder, never from the fitted
track.

**This is estimator-independent.** Production rising-edge, leading-edge 20 %,
CFD and full matched-filter all show the same compression
(`WAVEFORM_FIRST_THREADING.md` §3) — because all of them read a *mixed*
waveform. The mixing has to be modelled forward or removed, not thresholded
away.

## What replaces it

The forward-model reconstruction (`WAVEFORM_FIRST_THREADING.md`, packaged in
`wft/`): the charge arriving in each 60 ns slice of drift, folded through the
measured per-plane impulse response and the sharing kernel, generates the whole
(strip × sample) picture, and that whole picture is fitted per event. The
neighbours' delayed copies become part of the model instead of contamination.

Validated: per-event angle σ ≈ 1.0–1.1° — at the measured diffusion/charge-
granularity **physics floor**, not a fit limitation (toy closure, §12) — against
1.58/1.53° for the SOTA hybrid and 4.97/5.74° for the raw ladder on the same
events; mesh σ ≈ 0.58 mm; ~70 % of medium-angle events thread < 1 mm over the
full gap. Portable: det2 and det4 both calibrate and reconstruct with their own
kernels.

## Consequences already known

- **v_drift**: the hits ladder implies v = 47–50 µm/ns and the gap estimators
  read 31.5–34; the forward fit gives **36.6 ± 0.3 ± 0.9** at 1000 V, which
  matches Magboltz Ar/iso 95/5 + 0.8 % H₂O. Old v numbers derived from either
  ladder or gap carry this bias.
- **Drift gap**: gap-based estimators inherit −3…−7 % from assuming 29 mm; the
  real column is 27.9 mm on det3 (cathode tilt), 30.5 mm on det2, 28.8 mm on
  det4.
- **Angles and depth-resolved anything** from the old chain are compressed.
- **Positions at the mesh and efficiency are much less affected** — the July-25
  significance-floor fix left det3 at 93.1 % within 5 mm with core σ 0.45 mm, and
  that still holds. Efficiency is a property of the analyzer's trigger, not of
  the fit.

## Migration status

Updated in the same commit as each ported analysis — a number is only trustworthy
if you know which basis produced it.

| analysis | basis | where |
|---|---|---|
| Threading / cluster displays | **waveform** | `waveform_first_threading/37_threading_displays.py` |
| Per-event tracking (angles, positions) | **waveform** | `wft/` (R&D validated; packaging in progress) |
| v(E), gas fit | **waveform** | `waveform_first_threading/` scripts 19–21 |
| Gap / column maps | **waveform** | `waveform_first_threading/` scripts 29–35 |
| Alignment | hits (port pending) | `mx_june_cosmic_qa/03_alignment_and_tpc.py` |
| Efficiency maps and breakdown | hits — legitimate (detection is an analyzer property) | `mx_june_cosmic_qa/08`, `09`, `12` |
| Residuals / resolution maps | hits (port pending) | `mx_june_cosmic_qa/03`, `12` |
| Angular resolution, hybrid tracking | hits (port pending — **superseded in accuracy**) | `mx_june_cosmic_qa/34`, `36` |
| Time resolution | hits + waveform (port pending) | `mx_june_cosmic_qa/42` |
| Sparks, charge balance, fringe field | hits — QA-level, unaffected | `mx_june_cosmic_qa/38`–`40` |

## Reading order for the evidence

1. `mx_june_cosmic_qa/waveform_first_threading/THREADING_DISPLAYS_2026-07-28.md`
   — the displays and the 600-event census (this document's table)
2. `mx_june_cosmic_qa/waveform_first_threading/WAVEFORM_FIRST_THREADING.md`
   — the full study: §3 estimator-independence, §12 physics floor, §15 the
   recommended reconstruction, §17–21 v and gap consequences
3. `mx_june_cosmic_qa/REFERENCE_TRACK_THREADING_REPORT.md` — the earlier
   systematics report that posed the question
