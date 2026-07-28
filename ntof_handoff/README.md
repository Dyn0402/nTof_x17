# Revised PSA UserInput for the X17 EAR2 2026 campaign

**From the X17 / DREAM group (Dylan Neff), 2026-07-28.**
Proposed replacement for `UserInput_2026_EAR2_X17.h` (R. Mucciola, 2026-07-17).

Contact: dneff@cern.ch. Full analysis, tooling and the comparison report are in
the group repository under `ntof_processing/`.

---

## What is in this directory

```
UserInput_2026_EAR2_X17_v11.h     the proposed UserInput
pulse_shapes/                     every template it references (24 new + 2 shipped)
comparison_report.pdf             the measurements behind each change
```

**Before running**, rewrite the `PULSE SHAPE ADDRESS` column to the absolute
path of `pulse_shapes/` on your system -- the file ships with bare filenames.
One line does it:

```bash
sed -i "s#\(X17_[A-Za-z0-9_]*\.txt\)#$PWD/pulse_shapes/\1#g" UserInput_2026_EAR2_X17_v11.h
```

Then check that each detector row still declares as many shapes as it lists
addresses (LIQC has 2, everything else with fitting has 3).

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
one correctly-timed pulse. Our DREAM coincidence matcher goes from 95.2 % to
96.4 % efficient at the same 0.6 % false rate, and from 93.4 % to 95.5 % in the
hardest 1-3 ms bin.

### 7. `SIGNAL WIDTH LOW THR.` 10 → 4 ns on PSS

Plastic pulses are 13 ns FWHM, so a 10 ns floor sits on top of the width of a
pileup-*truncated* plastic pulse -- exactly the pulses the shape fit should be
recovering. Improves plastic fit chi2 by 3-13 %.

### Left alone deliberately

`PKUP` (0 % flash failures -- it is the natural absolute-time anchor), `SILI`,
all wall elimination windows, all baseline parameters, and the liquid pulse
shapes. We tried replacing the liquid templates three times (551 ns, 81 ns,
per-detector averages) and every attempt was worse: chi2 more than doubled and
amplitudes fell ~30 %. The shipped liquid pair spans FWHM 1 ns and 7 ns -- a
near-delta plus a normal pulse -- and we suspect that spread is doing real work
that a set of similar averaged shapes cannot.

---

## What we verified

Everything below is on run 224572 unless stated, with our own laptop-side
`tflash` repair **disabled**, so it tests the processing alone.

| | official | this UserInput |
|---|---|---|
| flash mis-identification | PSS 37-85 % | 0.0 % on 12 of 13 trees |
| per-arm coincidence offset | −362 / +20 / −333 / −336 ns | +1.5 / +2.0 / +1.0 / −2.0 ns |
| DREAM matcher efficiency | 93.7 % (needs our repair) | **96.4 %** |
| … false-match rate | 0.5 % | 0.6 % |
| … in the 1-3 ms bin | 89.9 % / 1.3 % | **95.5 %** / 2.6 % |
| wall timing resolution (top↔bottom) | — | 6.65 ns, unchanged |
| wall↔plastic coincidence width | — | 6.41 ns, unchanged |
| MIP peak width (FWHM/peak) | — | 1.22, unchanged |

The false-match rate at 1-3 ms roughly doubles. That is the cost of recovering
the plastic hits and we accept it deliberately -- the candidate rate rises from
~935 to ~1042 per bunch. If a different analysis needs early-time purity
instead, tightening `AREA/AMP HIGH` back towards 20 trades it back.

## One operational note

The merge step of `RunProcessing.sh` could not produce a merged file for any of
our EAR2 runs: the merge job ships the per-file partials through condor file
transfer and dies on `max total download bytes exceeded (max=1024 MB, this
file=1662 MB)`. All 16 processing jobs succeed every time; only the merge node
fails, on all runs we tried. We worked around it by reading the partials in
`completed/` directly. You may want to look at that independently of anything
here.
