# The liquid scintillators: what is actually wrong, and what can be done

**2026-07-29, overnight.** Follows the three failed attempts to replace the
liquid pulse-shape templates (v3_shapes, v5_liqshort, v9_liqaug).

Everything here is measured from raw stream1 waveforms of run 224572, on
isolated late-time (1-18 ms) pulses, with the tools in this directory:
`psd_from_raw.py`, `local_fit.py`, `tail_basis.py`, `misfit_controls.py`,
`is_it_photon_statistics.py`.

---

## The short version

The framing we started from -- *"the pulses are too fast for the processor to
handle"* -- is not what the data says, and neither was my own earlier guess
that the liquids carry two pulse classes. Three things are true instead:

1. **The liquids are a pileup problem, not a template problem.** Only
   **8-24 %** of liquid pulses are isolated. Every template we built, and every
   test that said it was good, used that isolated minority.
2. **A measured template really is better on isolated pulses** -- 3-4x better
   than the shipped pair -- and it still made the processed output worse,
   because a longer, more faithful template overlaps its neighbours more in a
   population that is mostly overlapping.
3. **Single-pulse fit quality is floored by photon statistics.** The residual
   scales as sqrt(amplitude), not with amplitude, so it is shot noise in the
   slow component and no template can absorb it.

So the honest answer to "can a better template fix the liquids" is **no**, and
we now know why rather than just observing it three times.

---

## 1. There are not two pulse classes

A liquid scintillator discriminates neutrons from gammas by the size of its
slow component, so the obvious hypothesis was that the liquids carry two
shapes and that one averaged template fits neither -- which would neatly
explain why averaging failed while it worked for the walls.

It is wrong. Above ~3000 ADC the tail/total ratio is a **single tight band at
0.21**, not a bimodal distribution (`liq_psd.png`). The low-tail population
visible at small amplitude is noise: at amplitude 500 the tail sits at a few
ADC counts, below the baseline RMS.

Splitting that unimodal distribution at its own percentiles does produce two
different-looking median shapes (`liq_shapes_by_psd.png`), which is exactly the
trap -- it looks like evidence for two classes and is not.

## 2. The misfit is photon statistics

Fit a single measured template to isolated pulses, amplitude free, and look at
how the residual scales with pulse size. Systematic shape variation would give
a residual proportional to amplitude; photon counting gives sqrt(amplitude).

LIQD, over a factor 25 in amplitude:

| amp | resid RMS [ADC] | resid/peak | **resid/sqrt(A)** |
|---|---|---|---|
| 1 900 | 26.6 | 0.0140 | **0.61** |
| 3 700 | 37.5 | 0.0103 | **0.62** |
| 5 100 | 45.4 | 0.0090 | **0.64** |
| 7 500 | 55.0 | 0.0077 | **0.65** |
| 14 000 | 74.0 | 0.0061 | **0.67** |

`resid/sqrt(A)` is flat to 10 %; `resid/peak` falls by a factor 2.3. LIQA gives
the same picture (0.73 → 0.92). **The slow component is a countable number of
photoelectrons and fluctuates irreducibly.**

That is why binning the template basis by tail fraction bought almost nothing:
1 shape → 8 shapes moved LIQA from 70.8 to 63.1 and LIQD from 23.5 to 20.1
(held-out scoring). There is no shape left to learn.

Controls, to be sure the residual is not my fitter: adding a free baseline
changes chi2 by 0.5 %, adding a slope by 3 %, adding a free width by 10 %. The
assumed noise is at most 1.8x too small. None of it closes a factor 20-70.

**Saturation is separate and real.** The top amplitude bin of both detectors
breaks the sqrt scaling hard -- `resid/sqrt(A)` jumps to 3.11 (LIQA) and 2.18
(LIQD) at ~31 000 ADC, where the pulse reaches the rail. Those pulses have a
genuinely different shape and should be flagged, not fitted.

## 3. The template result that looked good, and why it did not transfer

On isolated pulses, scored on a held-out half:

| basis | LIQA chi2 p50 | LIQD chi2 p50 |
|---|---|---|
| shipped pair (LIQA_Signal_7 + LIQB_Signal_0) | 224.3 | 80.9 |
| one measured median template | **70.8** | **23.5** |
| measured, truncated to 60 ns | 75.3 | -- |
| 8 shapes binned by tail fraction | 63.1 | 20.1 |

A single measured template is 3-4x better than what is shipped. And in the PSA
the same idea made things worse every time. The resolution is in §4: this test
used the 8-24 % of pulses that are isolated.

For the record, the measured template is also the more faithful one: its
area/amplitude is 8.7 against 9.0 for real pulses, while the shipped LIQA
template gives 7.5 -- it is 17 % short because it stops before the slow
component ends. That means the ~30 % amplitude drop we saw when we shipped
measured templates was **mostly a change of amplitude definition, not a
degradation** -- the shipped templates under-count area and therefore
over-estimate amplitude. Worth knowing before anyone compares liquid
amplitudes across processings.

## 4. The liquids are a pileup problem

Fraction of raw liquid blocks containing a pulse that survives isolation cuts
(nothing above 10 % of peak before it, monotone decay after):

| | LIQA | LIQB | LIQC | LIQD |
|---|---|---|---|---|
| isolated / blocks | 1014 / 6965 | 812 / 10033 | 136 / 1418 | 1250 / 5175 |
| **isolated** | **15 %** | **8 %** | **10 %** | **24 %** |

Three quarters to nine tenths of the liquid signal is piled up. That is the
regime the configuration has to handle, it is what Riccardo meant by "the
liquids are still kinda bad", and it is not addressed by any template.

It also explains the direction of the failures: a longer template overlaps more
neighbours, so the fitter merges or rejects more -- consistent with the 15-30 %
hit loss we measured each time.

## 5. What to do instead -- variant `v12_liqpileup`

Keep the shipped templates. Change two things:

- **`STEP SIZE` 2/4 → 1/3.** The finest derivative window available, for a 6 ns
  FWHM pulse at 1 GS/s. The guide's first practical advice is that reducing
  STEP SIZE resolves pileup, and `v7_step` (which moved LIQ to 2/3) was the
  only change so far that raised liquid yield at all, by 3-6 %.
- **`SIGNAL WIDTH HIGH THR.` 5000 → 5000/30.** This enables the fast/slow area
  split. **`afast` and `aslow` are currently 0.0 % filled**: the PSA's
  pulse-shape-discrimination observable has never been switched on for these
  detectors, and PSD is the entire reason one runs a liquid scintillator. The
  boundary at 30 ns sits past the prompt peak and inside the slow component,
  which the raw pulses show running to ~150 ns.

The second is worth having regardless of how the first performs: it costs
nothing and it is the difference between having n/gamma separation in the
output and not having it.

## 6. If none of that is enough

The floor established in §2 is physical, so a custom processing would not beat
the PSA on single-pulse fit quality either. What custom processing *would* buy
is pileup handling -- simultaneous multi-pulse fitting over a window rather
than pulse-by-pulse recognition and subtraction. That is a real gain and it is
not expressible in the UserInput.

If we want to go that way, the raw waveforms are what we need, and the request
should be scoped: **stream1 for the runs we care about**, which is ~2.7 GB per
70 s chunk and ~150 files per run. We already have the reader
(`nTof_x17_DAQ/stream1_monitor/ntof_raw.py`) and the extraction tooling, so the
only missing ingredient is access.

Before asking for that, `v12_liqpileup` should be graded -- it costs one
processing round and tests the actual hypothesis.
