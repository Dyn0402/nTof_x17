# The gamma-flash time base, and how to plug in pre-calibrated numbers

**2026-07-28. Follow-up to `HANDOFF_2026-07-28_ntof_processing.md`.**
Everything marked **[measured]** was extracted this session from raw stream1
waveforms of run 224572 (7 chunks spanning the whole run: bunches 161-163,
398-400, 798-800, 1198-1200, 1598-1600, 1998-2000, 2398-2400; 851 channel-bunch
flash blocks, ~30 k late-time pulses). **[inferred]** is a reading of the data
or the PSA guide that has not been confirmed against a reprocessed file.

---

## 0. The short version

You said the SiPM signal is intentionally diverted for ~1 us around the flash,
so the measured wall flash time cannot be trusted. The raw waveforms confirm
that and add one thing you may not have expected:

> **The walls DO record the real gamma flash.** It leaks through the closed
> protection gate as a small (+450 .. +1650 ADC), clamped, positive pulse at
> ~11.60 us -- coincident, to under 10 ns, with the flash seen by the plastics
> and the liquids. What the current processing times instead is the
> **gate-closing transient at 11.24 us**, i.e. ~370 ns *before* the flash.

So the ~350 ns "cross-detector flash-feature inconsistency" that broke the
DREAM matching is **the divert gate's lead time**, and it is fixable in the
UserInput without any calibration constant: point the flash finder past the
gate transient (a lower time limit of 11.4 us) and it lands on the real flash
on all four walls.

Pre-calibrated numbers are still needed, but for a much smaller residual (tens
of ns, not hundreds), and there is exactly one place to plug them in: the PSA
has **no** facility for a fixed or externally supplied flash time, so it must
be an offline per-tree constant. That hook now exists
(`ntof_processing/flash_calibration.json`, section 5).

---

## 1. What the wall waveform actually does  [measured]

Per bunch, on all 32 wall channels, in this order:

| t (us) | what | signed amplitude | width |
|---|---|---|---|
| 11.222 - 11.259 | **gate closes**: negative, flat-bottomed (clamped) step | -860 .. -2300 | 30-50 ns |
| ~11.28 - 11.30 | its positive overshoot | ~ +1000 | ~30 ns |
| 11.598 - 11.605 onset, peak 11.61-11.66 | **the gamma flash, leaking through the closed gate** | +451 .. +1655 | ~60 ns |
| 12.28 - 12.44 | **gate opens**: rail-to-rail transient, exceeds the ADC range and wraps | > +31000 | ~700 ns |
| 12.4 -> ~40 | baseline recovery | +21500 @13us, +3800 @15, +1300 @20, +830 @25, +400 @30, +100 @36 | |

Divert window = 11.24 -> 12.25-12.44 us = **1.01-1.20 us**, matching the ~1 us
you described.

Cross-check that the 11.60 us feature really is the flash: the plastics and the
liquids -- which are **not** diverted -- start their (saturating, negative)
flash at 11.598-11.610 us in the same bunches. Two independent detector types
agreeing with the wall feature to <10 ns, in every bunch, is not a coincidence.
PKUP sits at 13.32-13.34 us on its own cable.

Signal polarity, measured from ~30 k isolated late-time pulses and confirmed
independently by the sign of the shipped pulse-shape templates:
**walls positive, plastics and liquids negative, PKUP positive.**

### Why the current processing gets it wrong, per arm

`G-FLASH THRESHOLD = 500.` with no lower time limit takes the first positive
excursion above 500 channels. Two candidates clear 500: the gate overshoot at
~11.28 and the flash leak at ~11.61. Which one wins is decided by amplitude,
and **WALB's gate blip is the weakest of the four walls** (860-1378 vs
1298-2300 on A/C/D) -- so WALB usually falls through to the flash while A/C/D
latch onto the gate. That is precisely the pattern seen in the official file:

```
modal tflash   WALA/C/D 11245-11275 ns      WALB 11615-11645 ns
coincidence    PSSA -362   PSSB +20   PSSC -333   PSSD -336   (vs the walls)
offsets        LIQA -373   LIQB +10   LIQC -350   LIQD -348
```

Arm B ~ 0 because WALB was already timing the flash. **The bistability is the
bug, not a hardware difference between arms.**

---

## 2. What the fix is, and what it leaves behind

`userinputs/v1_flash/` sets `G-FLASH THRESHOLD = 250/11400` on the walls: at
least 250 channels, and never look before 11.4 us. The gate transient is over
by 11.32 us, so only the flash leak qualifies.

Validated locally with `ntof_processing/flash_finder_emulator.py`, which
implements G-FLASH OPTION=0 as documented and reproduces the official file's
failure pattern before reproducing the fix (851 channel-bunches):  **[measured]**

```
                          median tflash (ns)      p2-p98 spread    not found
current   WALA/C/D              11268-11279        378-1021 ns        0 %
          WALB                  11452 (bimodal)        386 ns         0 %
          PSSA-D                  122-152            565-629 ns       0 %   <- the bug
proposed  WALA-D                11604-11607          25-35 ns         0 %
          PSSA-D                11627-11633          10-19 ns         0 %
          LIQA-D                11597-11606           7-10 ns         0 %
```

Robustness plateau **[measured]**: on the walls any threshold in 150-400 works
and any time limit past the gate works; **at 600 the spread explodes to ~660 ns**
because the weakest channels miss the leak and fall through to the gate-release
transient -- so 250 sits in the middle of a plateau bounded above at ~400. On
the plastics the plateau is 500-5000 (identical results); 2000 is the midpoint.

What is left over after the config fix is a per-tree constant of order
**25-30 ns** (emulator: walls 11605, liquids 11601, plastics 11630 -- the
plastics read later because their flash saturates, so constant-fraction timing
lands further up a much steeper edge). That is the number the pre-calibration
has to supply, and it is ~13x smaller than what we live with today.

### The divert also costs the walls the first few us after the flash

The gate-release transient saturates and then decays for tens of us: the wall
baseline is still +400 channels at 30 us, i.e. 8x the amplitude threshold.
Wall hits are unusable from ~11.6 us to ~13-15 us and degraded to ~40 us.
At the 19.5 m EAR2 flight path that is E_n above roughly 1 keV. **Irrelevant
for X17** (ms-scale times) but it must be stated in any flux or E_n work.
`TIME LIMIT = 40000` in the wall row is well matched to this recovery and
should stay.

---

## 3. Can the PSA be given a fixed flash time? No.

Checked against `PSA_Guide_20240704.pdf` in full. Every flash mechanism
locates the flash in the waveform:

- `G-FLASH OPTION=0` first pulse crossing `G-FLASH THRESHOLD`
- `G-FLASH OPTION=1` first pulse going into saturation (+ optional MIN SATURATION)
- `G-FLASH OPTION=2` oscillatory treatment
- modifiers: `threshold/time_limit`, `MIN WIDTH`, `option/constant_fraction`,
  `G-FLASH WINDOW`

There is no "flash time" input, no per-detector time offset column, and no way
to point one detector's flash at another's. The only knobs are *which feature*
and *what fraction of it*. **[measured, from the guide]**

Consequence: pre-calibrated numbers are applied **downstream of the
processing**, when `t_since_flash = tof - tflash` is formed.

---

## 4. What the pre-calibration measurement has to deliver

Define, per tree (and possibly per channel):

```
    offset[tree] = t_true_flash_arrival_at_detector  -  tflash_reported_by_PSA
    t_since_flash = tof - (tflash_stored + offset[tree])
```

For the walls `offset` absorbs three things:

1. **divert-path vs direct-path delay.** The leak traverses the protection
   branch; physics pulses traverse the normal branch. Any propagation
   difference between the two shows up here, one-for-one. This is the term
   that *cannot* be got from beam data alone, because the flash only ever
   arrives with the gate closed.
2. **constant-fraction bias**: the PSA times the clamped, ~60 ns-wide leak at
   CF x amplitude; it times a physics hit on a much faster SiPM edge. Different
   slopes, different offsets from true arrival.
3. **per-channel electronics/cable spread.** The gate-close onset already
   varies 11.222-11.259 us across the 32 channels -- ~40 ns of channel-to-
   channel structure that may or may not be common to the physics path. **[measured]**

Two things we can already say about the shape of the answer:

- **A constant is enough; no intensity dependence.** Splitting the sample into
  dedicated (PKUP flash ~26000) and parasitic (~14000) bunches, the flash time
  shifts by only **-7 .. +5 ns** on every tree. The reason is visible in the
  data: the wall leak amplitude is nearly intensity-independent (706 vs 675,
  917 vs 862, 819 vs 742, 749 vs 662 on A/B/C/D), i.e. it is set by the
  protection clamp, not by the flash size. A clamped feature is a *good*
  timing fiducial. **[measured]**
- **Absolute accuracy barely matters for X17 physics.** At 19.5 m,
  dE/E = 2 dt/t: a 400 ns error is 8e-4 at t = 1 ms and 8 % at t = 10 us.
  What broke the DREAM matching was the *relative* wall<->plastic alignment,
  not the absolute one.

### How it could be measured (for you to choose)

| # | method | gives | cost |
|---|---|---|---|
| a | **pulser injection** into all 32 wall channels + plastics + PKUP, taken twice: gate inhibited and gate active | term 1 and 3 directly, per channel; the cleanest | bench / beam-off, needs the gate to be inhibitable |
| b | **one beam run with the divert disabled** (reduced intensity, or a dummy load on the SiPM bias) | terms 1+2+3 together, exactly as the physics sees them | beam time, and the protection exists for a reason |
| c | **in-situ prompt wall-plastic coincidences** (cosmics crossing both) | all *relative* offsets, free, every run -- this is what `tflash_repair._coinc_offsets` already does | none |
| d | **PKUP + the 19.5 m flight path** as absolute anchor | the absolute zero, given the PKUP cable delay | needs the cable delay |

Recommended: **(c) always, for the relative alignment, + (a) once, for the
absolute wall term.** (c) is already implemented and validated; (a) is the only
one that isolates the divert-path delay, and it is a beam-off measurement.
(b) is the gold standard but I would not risk the SiPMs for it.

---

## 5. Where the numbers get plugged in (implemented)

`ntof_processing/flash_calibration.json` + `flash_calibration.py`:

```python
from ntof_processing.flash_calibration import offsets, describe
offsets(224572)     # {'WALA': 0.0, ..., 'PSSA': -362.3, ...}
describe(224572)    # provenance, status, validity range
```

Entries carry `valid_from` / `valid_to` run ranges, a `reference_tree`, a
`status` (`provisional` | `calibrated`) and a `method` string, so a
data-derived stopgap can never be mistaken for a measured constant.

Consumed by the existing repair:

```python
tflash_repair.corrected_tflash(run, offsets_source='calib')   # use the JSON
tflash_repair.corrected_tflash(run, offsets_source='fit')     # in-situ, default
tflash_repair.corrected_tflash(run, offsets_source='none')
```

Two entries exist today: the current provisional set (measured on the *broken*
official processing -- relative only, do not use for absolute ToF), and a
clearly-marked template to fill in once a run has been reprocessed.

### Order of operations

1. reprocess with `userinputs/v1_flash` -> the walls time the real flash;
   the 350 ns arm-dependent offsets should collapse to tens of ns
2. `validate_reprocessing.py` measures the residual per-tree offsets -- paste
   them into the JSON as a new `provisional` entry, reference `PKUP`
3. do the pre-calibration measurement (4a) -> replace the wall numbers, flip
   `status` to `calibrated`, keep `PKUP` as the reference
4. only then is absolute E_n above ~100 keV meaningful

---

## 6. Open questions

1. Is the ~40 ns channel-to-channel spread of the gate-close onset **[measured]**
   also present in the physics path? If yes, `offset` has to be per channel,
   not per tree, and the calibration in 4a must be read out per channel.
   Testable offline: compare per-channel wall/plastic coincidence peaks.
2. The gate lead time (~370 ns before the flash) is presumably a setting on
   whatever generates the gate. If it is ever retuned, any calibration anchored
   on the gate transient silently breaks -- another reason to time the leak,
   which cannot drift.
3. Does the leak's clamp level depend on SiPM bias / temperature? If it does,
   the CF bias (term 2) drifts with it. The dedicated-vs-parasitic result
   (-7..+5 ns) suggests the sensitivity is weak.
