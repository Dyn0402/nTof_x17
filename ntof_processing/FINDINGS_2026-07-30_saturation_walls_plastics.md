# Where the walls and plastics saturate, and whether a post-processing flag is worth it

**2026-07-30.** Follow-on to `FINDINGS_2026-07-29_signed_decoding.md`, which
established that stream1 samples are signed `int16`, the usable amplitude is
~63 800 counts, and `satuflag` is reliable on the liquids but **never set on any
wall** (their saturation is a negative undershoot, outside any found pulse) nor
on PSSD.

The question here: **can we see where the walls and plastics saturate from the
amplitude distribution, and is there a hard ceiling below the readout limit that
a post-processing flag could catch?** Answer: **yes for the walls, no for the
plastics** — and at physics times the affected population is essentially empty,
so the flag is hygiene rather than physics.

Tools: `liq_study/wal_pss_saturation.py` (spectra + width-vs-amplitude figure),
`liq_study/sat_curve.py` (the tables below), `liq_study/amp_ceiling_census.py`
(the whole-run census). All on v12/224572 partials 5, 10, 15 unless stated.

---

## 1. The method, and why it is trustworthy

An unsaturated pulse has an **amplitude-independent width**. So plot median
`fwhm` against `amp` and look for the departure from the plateau.

The liquids are the control: `satuflag` there is verified per pulse, so we know
where the truth is. And the method reproduces it exactly —

```
LIQA, physics time, median fwhm [ns] (n)
  8-12k  12-16k  16-20k  20-25k  25-30k  30-34.6k  34.6-40k  40-45k  45-50k  50-55k  55-60k  60-63.8k   >63.8k
   6.1     6.2     6.2     6.2     6.2      6.2       6.2      6.1     6.1     6.1     6.1      6.0      7.3
  (8481)  (3870)  (2098)  (1505)  (849)    (545)     (434)    (305)   (207)   (180)   (136)    (98)     (399)
```

Flat to within 0.1 ns over a factor 8 in amplitude, right up to the ADC ceiling,
then it breaks — which is exactly where `satuflag` starts firing. LIQB and LIQD
do the same (6.1 / 6.4 ns flat, break above 63.8 k). Two things follow: the
liquids have no front-end compression at all, and **the width test is calibrated**.

One control worth stating, because it kills the obvious objection: inside the
γ-flash the **liquids stay at 6.0-6.7 ns** all the way to the ceiling. So a
width departure inside the flash is not a generic flash artifact — if it were,
the liquids would show it too.

## 2. The walls: a hard ceiling at ~44 000, well below the readout limit

`amp` spectra terminate abruptly, and at the same place on all four channels
(whole-run census, `amp_ceiling_census.py`):

| | WALA | WALB | WALC | WALD |
|---|---|---|---|---|
| largest `amp` in the run | 43 220 | 44 915 | 44 152 | 43 972 |
| hits above the 63 800 ADC ceiling | **0** | **0** | **0** | **0** |
| `satuflag` set | **0** | **0** | **0** | **0** |

**Nothing on a wall ever reaches the ADC ceiling** — the distribution stops dead
at ~69 % of it, and the *measured* peak `amp_0` stops at 30 884-33 570, i.e.
**48-52 % of the ADC range** on all four channels.

**What sets that limit is the flash/divert step, not compression of particle
pulses.** `pss_over_ceiling_waveforms.py --ceiling 34600` on WALB shows it: during
the flash and divert the trace steps from its −31 770 baseline up to **ADC zero**
and sits there for hundreds of nanoseconds. The PSA reconstructs that step as a
129-266 ns wide "pulse" of measured height 29 697-30 326 and fits it at
38 615-42 441, with χ² of 3.3 × 10⁴ to 7.8 × 10⁵ and `pileup1/pileup2` both set.
Those are exactly the hits above the cut. It matches the raw census maximum
excursions of 32 888-34 635 counts in `FINDINGS_2026-07-29_signed_decoding.md`
§2 — that number is this artifact, not an amplifier limit measured on particles.

**At physics times the walls do not saturate at all.** Over segment 8 (~105 k
physics-time hits per channel) the largest physics-time `amp_0` is 16 102 /
19 175 / 22 739 / 18 978, `amp` agrees with `amp_0` to better than 1 %, and the
largest one fits with χ² = 17 at the normal 72 ns width. So the wall flag is a
**hygiene cut against flash artifacts** — worth having because a 42 000-count
artifact entering a sum is indistinguishable from a real large pulse — and not a
recovery of any physics.

The width test shows the same population, in the flash where these hits live —
read it as "the things the PSA reports up there are not pulses of the normal
shape", not as amplifier compression:

```
median fwhm [ns] (n), FLASH region
ch      plateau   8-12k  12-16k  16-20k  20-25k  25-30k  30-34.6k  34.6-40k  40-45k
WALA      71.4     72.9    74.7    55.5   105.7   131.0    111.0     133.0    214.5
WALB      71.9     72.4    73.8    54.7   100.0   130.0     78.9     131.0    176.7
WALC      74.2     74.6    75.8    68.7   105.0   133.0    108.0     135.0    188.0
WALD      75.4     75.5    76.5    69.8    96.0   125.0    122.0     147.0    201.7
```

The plateau is ~72-76 ns and holds to 16 k; by 20-25 k the median width has
**doubled**, and by 40-45 k it is 2.5-3x the plateau. (The dip at 16-20 k is
reproducible on all four channels and we have not explained it; the fit
presumably latches onto a narrower core before it broadens. It is a departure
from the plateau either way.)

**At physics times there is nothing to flag.** Wall amplitudes stop below 25 k
and the width is flat across the whole range:

```
median fwhm [ns] (n), physics time
WALA   75.0 plateau | 75.1 (5993)  74.3 (775)  73.3 (83)  70.0 (6)  — (1)  then zero
WALB   73.0         | 73.2 (8464)  72.3 (920)  72.0 (73)  69.2 (5)  — (1)  then zero
```

Whole-run physics-time maxima are 26 263 / 35 992 / 30 450 / 35 596 with p99.99
at 13 500-14 100. So the entire saturating wall population is inside the
γ-flash, where the walls are diverted and unusable anyway.

## 3. The plastics: no sub-readout ceiling — they clip at the rail

The plastics behave the opposite way. Their spectra run **past** the ADC ceiling,
with the tell-tale pile-up in the last bins before it (2 000-ADC bins, all times,
3 partials):

```
        44k  46k  48k  50k  52k  54k  56k  58k  60k  62k  64k  66k  68k
PSSA      4    3    2    2    4    5    5   66   26   64   14   28   20
PSSB     45   10   17   20   33   52   61  144  231  195   26    0    0
PSSD     63  214  290  139   40   39   68  167  196  148   41    0    0
```

and the whole-run census finds 211-6 417 hits per channel above 63 800, with
largest `amp` of 2.7 × 10⁸ (PSSA) and 3.9 × 10⁸ (PSSC).

### 3.1 `amp > 63 800` on a plastic is not a saturation test

**Correction to the first version of this file**, which read the low `satuflag`
coverage of the over-ceiling plastic hits (45-47 % on PSSA/PSSC, 0 % on PSSD) as
more of the `AnalyseSaturation` blind spot. It is not. Using `amp_0`, the PSA's
own **pre-fit measured peak**, the over-ceiling population splits in two and the
flag is right about both halves (partial 5):

| tree | `amp > 63 800` | `satuflag`=1 | `satuflag`=0 | `amp_0` p50 of the unflagged | `amp_0` max | `amp`/`amp_0` p50 |
|---|---|---|---|---|---|---|
| PSSA | 390 | 177 | 213 | 58 003 | 60 065 | 40.7 |
| PSSB | 183 | 169 | 14 | 44 446 | 44 732 | 79.6 |
| PSSC | 427 | 198 | 229 | 5 639 | 62 462 | 22.3 |
| PSSD | 23 | **0** | 23 | 44 326 | 44 680 | **1.45** |

The unflagged ones **never reached the rail** — their measured peaks sit at
58-62 k against a rail at 63 568, i.e. within ~1-5 k of it but short. `satuflag`
correctly does not fire; what is wrong is the *fit*, which overshoots the
measured peak by 1.45x (PSSD) to 80x (PSSB). So the honest reading is:

- **`satuflag`** = "this pulse touched the rail". Reliable.
- **`amp > 63 800`** = "the fit returned something the hardware cannot produce".
  A fit-quality flag, useful for hygiene, not a saturation flag.

### 3.2 Why PSSD never sets the flag: it is analogue-limited, like the walls

The measured peak tells each channel's real ceiling. Excluding the impossible
`amp_0` values (>100 k, a handful of flash-core hits where even the pre-fit
maximum is garbage):

| | measured `amp_0` max | ADC rail (baseline + 32 768) | limit |
|---|---|---|---|
| WALA / WALB / WALC / WALD | 33 570 / 31 466 / 31 060 / 30 884 | 64 168 | **analogue, 48-52 % of range** |
| PSSA / PSSB / PSSC | 63 632 / 63 540 / 64 062 | 63 568 | ADC rail |
| **PSSD** | **44 806** | 63 568 | **analogue, 70 % of range** |
| LIQA / LIQB / LIQC / LIQD | 68 220 / 91 876 / 63 907 / 63 899 | 63 968 | ADC rail |

**PSSD is a plastic that behaves like a wall**: its front end limits at 70 % of
the ADC range, so it never reaches the rail, so `satuflag` can never fire on it —
and all 23 of its over-ceiling hits are fit overshoot on unclipped pulses
(`amp`/`amp_0` = 1.45). The walls are the same story at 48-52 %. The liquids pile
up exactly at their rail, which is why the flag works there.

Their width creeps up gradually rather than at a knee — PSSC physics-time
15.7 ns plateau → 17.4 (16-20 k) → 18.1 → 19.6 → 20.9 → 22.4 → 23.6 (40-45 k) —
i.e. mild compression from ~16 k, with tens of hits involved. Physics-time
maxima are 42 289 / 54 753 / 51 793 / 35 357.

## 3.3 What an over-ceiling plastic hit actually looks like

`pss_over_ceiling_waveforms.py` (figure `pss_over_ceiling_PSSC.png`), PSSC
segment 8, the five over-ceiling hits in the two bunches the raw chunk holds.
Every one of them is in the γ-flash at 11.64-11.73 µs, and they all look the
same:

- the trace sits at its +31 250 baseline, then **plunges through zero to the
  negative rail in 10-20 ns** — a swing of ~64 000 counts, the entire signed
  range;
- it stays at or just above the rail for ~65 ns (66 samples at −32 768 in the
  deepest, or a turning point at −31 642, ~130 counts short, in the others);
- it recovers over 200-600 ns back up to +22 000…+29 000, with individual pulses
  riding on the recovery, and the block carries 14 000-29 000 PSA hits in total.

So nothing "runs past the rail" in the data — the *fit* does. The two failure
modes of §3.1 are both visible: the deepest one touches the rail, is flagged, and
is fitted at `amp` = 7.7 × 10⁷ against a measured `amp_0` of 8.9 × 10⁷ (both
impossible — in the flash core even the pre-fit maximum is garbage); the others
stop just short of the rail, are correctly unflagged, and are fitted at 72 000-
88 000 against a measured 60 900-62 300, a 15-40 % overshoot on a genuine
near-full-range pulse.

## 4. Recommended post-processing flags

**Walls — worth adding, and only a post-processing flag can do it:**

```
WAL saturated  :=  amp > 34 600          # the front-end limit, from the raw traces
WAL extrapolated := amp > 40 000         # if you want only the certainly-bad ones
```

No rail test can find these, because the pulse never reaches the rail in its own
direction. Expect it to fire only inside the γ-flash.

**Plastics and liquids — the ADC ceiling plus the flag, as already recommended
in the handoff:**

```
saturated := satuflag  OR  amp > 63 800
```

`satuflag` alone misses 8.9-15 % of over-ceiling liquid hits and 8.5-55 % of
plastic ones (100 % on PSSD); the `amp` cut alone misses the ~4 000 hits per
liquid tree that are flagged with an extrapolated `amp` back inside the range.

**A width-based flag is possible but not recommended as a first line.** The
plateau is sharp enough to use (`fwhm` > 1.3 × the per-channel plateau of
72-77 ns on the walls, 13-16 ns on the plastics, 6.1-6.4 ns on the liquids), but
`fwhm` also responds to pileup, so it would need `pileup1`/`pileup2` gating and
would cost real hits. The amplitude cuts above are cleaner and are enough.

## 4.1 `area` is proportional to `amp` by construction — and the guide says so

Observed first as a puzzle: `area/amp` is a per-tree constant (7.55 on LIQA,
90.91 on WALB, 17.31 on PSSC), so `area` adds nothing to `amp` and cannot be used
to reconstruct a clipped amplitude. The PSA guide explains it, under **"Finding
the amplitude and area"**:

> AMPLITUDE OPTION=0 implies the search for the highest point of the pulse.
> AMPLITUDE OPTION=1 activates the parabolic fitting to the top of the pulse,
> while **AMPLITUDE OPTION=2 activates the predefined Pulse Shape fitting. In
> case of the last option, Pulse Shapes need to be provided and both the final
> amplitude and area will be determined from the fitted pulse.** Otherwise, the
> area under the pulse is calculated by the simple pulse integration.

Our UserInput runs AMPLITUDE OPTION=2 on the walls and plastics (and the liquids
use the shipped shapes), so both numbers come from the same scaled template and
`area = amp x integral(shape)` exactly. Confirmed to the hit: `area/amp` takes
exactly one value per `pulseshape`, and the counts match — LIQA's 2 282 953 hits
with `pulseshape == 1` are precisely the 2 282 953 hits with `area/amp` = 2.7543,
the remainder (`pulseshape == 0`) giving 7.5457. PSSC shows three values
(15.3948 / 17.3062 / 18.8980) for its three shapes.

**The measured quantities are `amp_0` and `area_0`.** The guide, under "Pulse
elimination": "Even if AMPLITUDE OPTION≠0 is selected, **amplitudes and areas are
first determined by the simplest procedures — search for the maximum and
integration**", and the fast/slow note is explicit that "the integrated area,
`area_0`, (not the fitted area, `area`) is considered in the afast/aslow
calculation". Practically:

| want | use | not |
|---|---|---|
| a real integral | `area_0` | `area` (spread 0.05 vs 0.46 against `amp`) |
| an amplitude that cannot be extrapolated | `amp_0` | `amp` |
| pulse-shape-fit quality | `chi2`, or `amp`/`amp_0` | — |

`amp`/`amp_0` is in fact the cleanest saturation diagnostic in the file: ~1.0 for
clean pulses, 1.24-1.30 for the wall flash artifacts, 1.45 for PSSD's overshoots
and 22-80 for genuinely clipped plastics.

## 5. What this does *not* change

Nothing in the shipped UserInput. Both flags are analysis-side, applied when
reading the trees, and the affected population at physics times is zero on the
walls and tens of hits per three partials on the plastics. The reason to add
them is hygiene — one hit at `amp` = 3.9 × 10⁸ ruins any sum it enters — not
recovered physics.
