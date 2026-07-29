# Revised PSA UserInput for the X17 EAR2 2026 campaign

**From the X17 / DREAM group (Dylan Neff), 2026-07-28, revised 2026-07-29.**
Proposed replacement for `UserInput_2026_EAR2_X17.h` (R. Mucciola, 2026-07-17).

Contact: dneff@cern.ch. Full analysis, tooling and the comparison report are in
the group repository under `ntof_processing/`.

---

## What is in this directory

```
UserInput_2026_EAR2_X17_v12.h     the proposed UserInput
pulse_shapes/                     every template it references (24 new + 2 shipped)
comparison_report.pdf             the measurements behind each change
adc_wrap_as_recorded.png          the ADC wrap of §8b(b), exactly as stored
adc_wrap_examples.png             the same pulses with our reconstruction on top
```

The UserInput is the one we ran to produce every number in this document; it is
byte-identical to the variant our own analysis has now adopted as production.

**Before running**, rewrite the `PULSE SHAPE ADDRESS` column to the absolute
path of `pulse_shapes/` on your system -- the file ships with bare filenames.
One line does it:

```bash
sed -i "s#\(X17_[A-Za-z0-9_]*\.txt\)#$PWD/pulse_shapes/\1#g" UserInput_2026_EAR2_X17_v12.h
```

Then check that each detector row still declares as many shapes as it lists
addresses: **3 for each wall and each plastic, 2 for each liquid** (the four
liquids share the two shipped templates between them, so `pulse_shapes/` holds
26 distinct files for 32 references). `PKUP` and `SILI` fit no shapes.

---

## Why: the two defects we found

Both were diagnosed against run 224572 and confirmed in the raw stream1
waveforms. Section numbers refer to the comparison report.

### 1. The plastic γ-flash is mis-identified in 37-85 % of bunches

`PSS*` uses `G-FLASH THRESHOLD = 50.` with no lower time limit. The plastic
flash saturates the digitiser -- it drives the signal from a baseline of
~30 800 past 0 -- so 50 channels is ~0.2 % of the feature it is meant to find,
and the first noise excursion in the 11.6 µs before the flash wins instead.

Measured fraction of bunches whose stored `tflash` is more than 150 ns from the
tree's own mode:

| WALA | WALB | WALC | WALD | PSSA | PSSB | PSSC | PSSD | LIQA-D | PKUP |
|---|---|---|---|---|---|---|---|---|---|
| 1.7 % | 1.1 % | 0.3 % | 0.0 % | **84.5 %** | **65.4 %** | **36.8 %** | **80.6 %** | 0.0 % | 0.0 % |

Because every hit's physics time is `tof − tflash`, each failed (tree, bunch)
has all of its hits shifted, by up to 11.6 µs. Every parasitic pulse is
affected plus an arm-dependent share of the dedicated ones.

**Fix**: `G-FLASH THRESHOLD = 2000/1e4` -- 2000 channels, and do not look
before 10 µs. The flash sits at 11.63 µs in every bunch by hardware. The
plateau is wide: 500, 2000 and 5000 give identical results.

### 2. The SiPM walls time the divert gate, not the flash

The wall signal is blanked by a ~1 µs protection gate around the flash, so the
walls never see it directly. What the digitiser records is the gate-closing
transient at 11.24 µs, then a clamped copy of the real flash leaking through
at 11.60 µs, then the gate-opening transient at 12.3 µs.

With `G-FLASH THRESHOLD = 500.` and no time limit, which of the first two wins
depends on their relative amplitude -- and **WALB's gate transient is the
weakest of the four**, so WALB usually lands on the flash while WALA/C/D land
on the gate. That bistability is the origin of the arm-dependent offsets seen
in coincidences:

```
prompt wall-plastic coincidence peak, official:  PSSA -362  PSSB +20  PSSC -333  PSSD -336 ns
                                                 LIQA -373  LIQB +10  LIQC -350  LIQD -348 ns
```

Arm B reads ~0 only because WALB was already timing the flash.

**Fix**: `G-FLASH THRESHOLD = 250/11400` on the walls -- at least 250 channels,
never before 11.4 µs, which is after the gate transient is over and before the
flash. The threshold must stay ≤400: at 600 the weakest channels miss the leak
and fall through to the gate-opening transient.

After both fixes, on the same run: flash mis-identification **0.0 % on 12 of 13
trees** (PSSD 0.2 %), and the coincidence offsets become
**−3.5 / +0.6 / −3.0 / −5.9 ns** for the plastics.

*(Independently, we calibrated the absolute flash time on the seven runs of the
campaign taken with the divert disabled: `t_flash = tof_PKUP + C`, C ≈ −1719 ns
per wall channel, reproducible to 0.5 ns. We are happy to share that too.)*

---

## The other changes, and the evidence for each

### 3. `AREA/AMP HIGH` was eliminating real pulses on PSS and LIQ

Measured on isolated late-time pulses from the raw waveforms:

| family | area/amp p1 … p99 | shipped cut | fraction removed |
|---|---|---|---|
| WAL | 42 … 110 | 10 … 200 | 0 % |
| PSS | 1.3 … 29 | 2 … 20 | **~25 %** |
| LIQ | 1.7 … 14 | 2 … 10 | **~19 %** |

The wall cut sits a factor 2 above the observed maximum, which is what a safe
elimination window looks like; the plastic and liquid cuts sit inside the bulk
of their own distributions. Widened to `1 … 60`. The guide's own advice is that
elimination should be loose and that final rejection belongs in the analysis.

### 4. Plastic amplitude threshold 100 → 50

The walls use 50; there was no reason for the plastics to be twice as strict.
(We tested 25 as well: it adds 15-25 % more plastic hits but changes nothing
downstream, because the hardware discriminator sits far above it. 50 is kept.)

### 5. Measured pulse-shape templates for the walls

The shipped wall templates are each a **single raw pulse, 314 ns long**, while
the wall pulse is still at 4-6 % of peak 200 ns after the maximum and 0.5 % at
500 ns. A template that ends inside the tail biases every fitted amplitude and
weakens the pileup deconvolution.

Replaced with median averages of 387-1327 clean isolated pulses per tree per
amplitude regime, 720-861 ns long. Fit quality improves on all four walls:

| | WALA | WALB | WALC | WALD |
|---|---|---|---|---|
| chi2 p50, shipped | 0.896 | 1.227 | 1.133 | 1.184 |
| chi2 p50, measured | **0.851** | **1.002** | **1.002** | **1.062** |

The tail is mildly amplitude-dependent (5.8 % at 200 ns in the lowest bin vs
4.1 % in the highest), so the three-shape machinery is kept and now carries
three genuinely distinct shapes rather than three arbitrary single pulses
shared across all four walls.

### 6. Pulse-shape fitting enabled for the plastics

`PSS` used `AMPLITUDE OPTION = 1` (parabolic top), i.e. no deconvolution, while
being the highest-rate tree in the file. Switched to option 2 with measured
101 ns templates.

This gave the single largest downstream gain, and by an instructive route: it
produces **fewer** plastic hits (0.72-0.99 of the previous count at every
amplitude cut) but **better-timed** ones, merging pileup fragments back into
one correctly-timed pulse. Our DREAM coincidence matcher goes from 95.3 % to
96.3 % efficient at the same 0.5 % false rate, and from 93.5 % to 95.0 % in the
hardest 1-3 ms bin (252 bunches; an earlier 100-bunch sample gave 95.2 → 96.4
and 93.4 → 95.5, so the gain is stable). On the full DREAM reference pair —
2061 bunches, 213 k events, two disjoint hours — this configuration gives
**95.7 % / 0.5 %**, see "What we verified" below.

### 7. `SIGNAL WIDTH LOW THR.` 10 → 4 ns on PSS

Plastic pulses are 13 ns FWHM, so a 10 ns floor sits on top of the width of a
pileup-*truncated* plastic pulse -- exactly the pulses the shape fit should be
recovering. Improves plastic fit chi2 by 3-13 %.

### 8. The liquids: `STEP SIZE` 2/4 → 1/3, and the fast/slow split enabled

The liquids were the hardest part and we spent a night of raw-waveform work on
them. Three things came out of it that are worth passing on whatever you make
of the configuration change.

**They are a pileup problem, not a template problem.** Only **8-24 % of liquid
pulses are isolated** (LIQA 1014 of 6965 raw blocks, LIQB 812/10033,
LIQC 136/1418, LIQD 1250/5175). That is why replacing the templates kept
failing: a longer, more faithful template overlaps more neighbours in a
population that is mostly overlapping.

**Single-pulse fit quality is floored by photon statistics -- on three of the
four.** Fitting one measured template to isolated pulses, the residual scales as
sqrt(amplitude) rather than amplitude: LIQD gives residual/sqrt(A) = 0.61, 0.62,
0.64, 0.65, 0.67, 0.70 across a factor 9 in amplitude, LIQA is flat to 41 % and
LIQC to 14 %, while residual/peak falls 2.8-3.4x. The slow component is a
countable number of photoelectrons and fluctuates irreducibly, so no template
basis can absorb it: giving the fit one template per amplitude octile instead of
one buys 2-3 % on a held-out half.

**LIQB is the exception.** Its residual/sqrt(A) runs 0.62 → 1.59 and
residual/peak stops falling, which is the systematic-shape signature rather than
the shot-noise one, and an amplitude-binned basis does cut its residual by 24 %
held-out (27 % in the top amplitude quartile). Looking at the shapes directly,
LIQB's *small* pulses are the unusual ones -- narrower and nearly tail-free
(tail/total 0.136 against 0.188 for LIQD) -- while its large pulses look like
every other liquid. We have not chased it, because the reason measured templates
were rejected was never fit quality (a measured template was already 3-4x better
on isolated pulses and still made the processed output worse). But if you care
about LIQB specifically, that is where to look.

**What did work**: `STEP SIZE` 2/4 → 1/3, the finest derivative window, for a
6 ns FWHM pulse at 1 GS/s. Yield **+14 to +21 %** on all four liquids with fit
quality neutral-to-better (LIQD chi2 −8.5 %), and the pileup flag rate up ~50 %,
i.e. it is genuinely separating overlapping pulses.

We also set `SIGNAL WIDTH HIGH THR. = 5000/30` to enable the fast/slow area
split, because **`afast` and `aslow` are 0.0 % filled in the current
processing** -- the PSA's pulse-shape-discrimination observable has never been
switched on for these detectors, and PSD is the entire reason one runs a liquid
scintillator.

**That did not work, and the reason is worth your attention.** With the
boundary set, `afast` fills for 100 % of hits but `aslow` stays ~0. `aslow` is
integrated "from the boundary up to the end of the pulse", and with
`EXPAND PULSES = 0` the liquid pulse boundary closes 20-40 ns after the peak
while the slow component runs to ~150 ns -- so it falls outside the
reconstructed pulse. We tried `EXPAND PULSES = 1` with a 150 ns suggested width
to fix that (variant v13) and it backfired: `aslow` was still 0 at the median,
and the expansion merged pulses in the pileup-dominated population, costing
17-28 % of the hits and raising chi2 by 14-47 %. So we reverted it.

**Consequence beyond PSD**: the reported liquid `area` has been missing its slow
component all along, in this and every previous processing. Anything calibrated
on liquid `area` is affected. If there is a way to capture it pulse-by-pulse
that we have not found, we would very much like to know.

We do not think this is a shortcoming of the PSA. Measured on the raw
waveforms, **67-76 % of liquid pulses have another pulse inside their own
150 ns tail** (median hit spacing 24-30 ns against a 6 ns FWHM pulse), so there
is almost never a clean window in which to integrate one pulse's slow
component. We also checked whether custom processing would find more pulses
than the PSA does: an iterative matched-filter deconvolution on the raw
waveforms recovers only **0.67x** the PSA's hits, so the recognition is not
what is limiting the liquids either. This looks like a rate limitation rather
than a software one.

### 8b. Three warnings about the output, added after a final round of tests

These came out of a pre-ship review on 2026-07-29. The first two are properties
of the PSA and the DAQ rather than of this UserInput, so **they apply to the
official processing as well** and you may want them independently of anything
else here.

**(a) `satuflag` is not a reliable saturation flag.** Counting hits whose `amp`
exceeds the channel's own baseline -- i.e. physically impossible amplitudes --
in run 224572:

| tree | hits | baseline | `amp` > baseline | `satuflag` set | largest `amp` |
|---|---|---|---|---|---|
| LIQA | 3 369 621 | 31 220 | 2 006 (0.060 %) | 791 | 3 234 415 |
| LIQD | 2 305 162 | 31 108 | 659 (0.029 %) | 382 | 763 834 |
| WALB | 1 844 839 | 34 003 | 1 069 (0.058 %) | **0** | 43 427 |
| WALD | 1 995 146 | 34 492 | 809 (0.041 %) | **0** | 42 773 |
| PSSA | 5 582 034 | 30 841 | 491 (0.009 %) | 203 | **243 257 568** |
| PSSC | 9 043 427 | 30 712 | 519 (0.006 %) | 198 | **320 356 608** |

`satuflag` is **never set on any of the four walls**, and on the liquids it
catches only a third to a half. The affected fraction is tiny, but a single hit
with `amp` = 2.4e8 will destroy any sum, mean or calibration it enters.
**Recommend: cut on `amp` above the largest amplitude the range can represent**,
rather than relying on `satuflag`. That limit is the headroom from the baseline
to whichever end the pulses run toward — the baseline itself for the
negative-going plastics and liquids (~30 700-31 200), and `65535 − baseline` for
the positive-going walls (~31 000-31 800). Conveniently the two come out nearly
equal, so **~31 000 is a single safe ceiling for every tree**. (The table above
used the plain baseline as the rail, which is the same number to within 3 % on
the walls; the counts do not change materially.)

**(b) The ADC wraps at the end of its range; it does not clip.** stream1 samples
are unsigned 16-bit and every channel sits well inside that range, so the largest
measurable amplitude is the distance from the baseline to whichever end the
pulses run toward. A larger pulse wraps modulo 65536 instead of being clamped.

**Which end depends on the polarity, and the two families differ:** the plastics
and liquids are negative-going on a baseline of 30 700-31 200, so a too-big pulse
runs below 0 and reappears near 65 535; the walls and PKUP are positive-going on
a baseline of 33 700-34 500, so theirs runs past 65 535 and reappears near 0.
(We measured polarity on in-range blocks, 91-100 % consistent per detector, and
it agrees with the sign of your own shipped templates.)

```
LIQA  ... 32767 32767 32767 32768 63712  4641 15598 27611 32160 ...
                                  ^^^^^ should have been below zero

WALA  ... 65243 65336 65428    58   100   229   446 ...
                             ^^^^ should have been above 65535
```

A threshold test therefore cannot identify a wrap on its own — on a wall, a
sample above 60 000 is an ordinary large pulse. What identifies one either way is
the **discontinuity**: a step of more than 20 000 ADC between adjacent samples,
which no real pulse produces at 1 GS/s.

There is no flat top, so a clipping test does not find it; the reported `amp` is
whatever the last un-wrapped sample on the rising edge happened to be, so it is
randomly *under*-reported; and the fitted shape sees a full-scale positive spike
one sample after the peak. Frequency, as a fraction of zero-suppressed blocks
containing at least one wrap: liquids 0.09-0.67 %, plastics 0.03-0.04 %, walls
0.05-0.09 %, SILI 0.84 %. **For the walls and plastics every single occurrence
is inside the γ-flash** (0 of 178 and 0 of 41 respectively fall outside it),
where saturation is expected; the liquids are the only detectors where it
happens at physics times.

`adc_wrap_as_recorded.png` in this directory shows six liquid ones, all in the
1-21 ms physics window, as stored: the trace sits at the ~31 200 baseline, dives
toward 0, and **1-2 samples** (never more than 3) appear up at 63-65 000 before
it resumes its normal fall. Undoing the wrap recovers a pulse of 31 340-36 656
ADC against a 31 200 ceiling — only 0.4 to 17 % over range, median 6 % — which
is why nothing about these pulses looks anomalous except that one sample.

**(c) `afast` is not an n/γ discriminant.** Now that the boundary is set,
`afast` fills for 100 % of hits and `aslow` is **always zero** -- see §8. What
`afast` does deliver is weaker than we first hoped. On isolated late-time pulses
above amp 3000, `(area − afast)/area` has a median of 0.055-0.124 against 0.113
measured on the raw waveforms at the same 30 ns split, so it is roughly right
*in aggregate*. But its per-hit spread (p16-p84) is 0.14-0.31 against a physical
band of 0.033-0.044 -- **4 to 9 times too wide** -- and it falls by a factor two
from small to large pulses, which a real pulse-shape variable does not do. Use
it in aggregate if at all; do not cut on it per pulse. Independently, the raw
tail/total distribution is a single band rather than bimodal, so there is no
n/γ separation to find in this data in the first place.

### Left alone deliberately

`PKUP` (0 % flash failures -- it is the natural absolute-time anchor), `SILI`,
all wall elimination windows, all baseline parameters, and the liquid pulse
shapes -- see §8 for why the liquid templates are best left as shipped.

---

## What we verified

Everything below is on run 224572 unless stated, with our own laptop-side
`tflash` repair **disabled**, so it tests the processing alone. The matcher
numbers are now over the **full DREAM reference pair**: both sub-runs of
`run_79` (`stat090_0000` and `stat090_0001`), 2061 bunches on disjoint ranges
146-1157 and 1165-2213, 213 k events, two independent hours of data. They agree
to 0.0 points with each other, so this is not a sample fluctuation. The
"official" column is the official file **with our laptop-side repair applied**,
i.e. the best that processing can do; on its own stored `tflash` it gives
12.2 %.

| | official (+ our repair) | this UserInput |
|---|---|---|
| flash mis-identification | PSS 37-85 % | 0.0 % on 12 of 13 trees |
| per-arm coincidence offset, no repair | −362 / +20 / −333 / −336 ns | +2.5 / +1.5 / +0.5 / −3.0 ns |
| DREAM matcher efficiency | 92.4 % | **95.7 %** |
| … false-match rate | 0.5 % | 0.5 % |
| … in the 1-3 ms bin | 89.3 % / 1.3 % | **94.7 %** / 1.6 % |
| … in the 40-80 ms bin | 87.4 % | **95.2 %** |
| … wall leg alone | 98.4 % | 98.4 % |
| … cost of requiring a plastic | 5.9 % | **2.7 %** |
| wall timing resolution (top↔bottom) | — | 6.65 ns, unchanged |
| wall↔plastic coincidence width | — | 6.41 ns, unchanged |
| MIP peak width (FWHM/peak) | — | 1.22, unchanged |
| liquid yield | — | **+14 to +21 %** |
| liquid fit chi2 (LIQD) | 1.768 | **1.617** |

**One caveat on the liquid yield.** The +14-21 % is solid as a count, and the
fit quality moves the right way (chi2 p50 neutral, chi2 p90 clearly better --
LIQA 31.4 → 25.3, LIQD 21.6 → 16.2 against the official file). What we could not
finish is the per-hit check that each extra hit is a real pulse: **95 % of the
gain is resolved shoulders on existing pulses** rather than pulses missed
outright, and verifying those individually needs a raw-waveform comparison we
could not make trustworthy. Counting rather than matching, v12 reports 0.96 of
the pulses the raw data resolves against 0.77 for the baseline, i.e. it
approaches that ceiling without crossing it — supportive, not conclusive.

**One question would probably close it.** Stacking the stream1 trace on every
reported hit, the raw pulse peak sits **19–26 ns after `peak_tof`**, by a
constant that differs per detector (LIQA +26, LIQB +27, LIQD +19 ns). We
understand the branch definitions — Žugec's guide gives `tof` as the 30 %
constant-fraction arrival and `peak_tof` as the peak moment, and the two differ
by 1.3 ns in our files exactly as they should. So this is not a definition
mismatch. It is not a sampling-rate mismatch either: the lag is constant from
1.7 to 7.0 ms of time of flight. **Is there a per-channel offset between the
`start` index of an ACQC block in stream1 and the sample origin the PSA times
against?** If so, we would like the numbers; if not, we are misreading the raw
format and would like to know that too.

The false-match rate at 1-3 ms roughly doubles. That is the cost of recovering
the plastic hits and we accept it deliberately -- the candidate rate rises from
~935 to ~1042 per bunch. If a different analysis needs early-time purity
instead, tightening `AREA/AMP HIGH` back towards 20 trades it back.

## An independent cross-check: the Micromegas and the liquids

Added 2026-07-29. Everything above grades the n_TOF file against itself or
against DREAM *timing*. This one asks whether the events the matcher selects
are physically the right ones, using a detector that knows nothing about the
n_TOF processing: the DREAM Micromegas chambers, one per arm.

On 31 432 non-flash DREAM events, 96.3 % match a thresholded wall+plastic
single in at least one arm and 95.8 % in exactly one. For the exclusively
matched events, the fraction with a Micromegas cluster (≥2 strips in both
planes) per chamber:

```
  matched to      chA     chB     chC     chD      n
    arm A only   81.7%   57.4%   26.5%   64.7%   6526
    arm B only   38.2%   77.5%   33.0%   68.8%   8237
    arm C only   36.8%   59.8%   76.8%   67.3%   7987
    arm D only   33.7%   57.6%   28.7%   85.6%   7358
```

The diagonal is enhanced in every row over chamber-dependent occupancy floors,
and a large-pulse tier sharpens it further (arm C: 21.6 % on chamber C against
~4 % off-arm). So the wall+plastic coincidences this processing reports really
are particles crossing the corresponding arm.

Two things follow that matter for the campaign:

- **The residual inefficiency is ours, not the file's.** Events the matcher
  misses have a Micromegas cluster 96.4 % of the time against 96.5 % for the
  events it finds — statistically identical. The misses are not fake DREAM
  triggers; they are analysis-side, so no further UserInput change is indicated
  by them.
- **The liquid time base is externally confirmed.** Repeating the exercise on
  the `LIQ*` trees, same-arm liquid hits show a 5-7× excess over an accidental
  floor (a +100 µs shifted control) at a stable **−5 to −25 ns** residual, with
  every off-arm pair at the floor. The ~350 ns per-tree offsets present in the
  official file are gone. This is the first end-to-end test of the liquid leg
  of this configuration against an outside detector, and it passes.

## One operational note

The merge step of `RunProcessing.sh` could not produce a merged file for any of
our EAR2 runs: the merge job ships the per-file partials through condor file
transfer and dies on `max total download bytes exceeded (max=1024 MB, this
file=1662 MB)`. All 16 processing jobs succeed every time; only the merge node
fails, on all runs we tried. We worked around it by reading the partials in
`completed/` directly. You may want to look at that independently of anything
here.
