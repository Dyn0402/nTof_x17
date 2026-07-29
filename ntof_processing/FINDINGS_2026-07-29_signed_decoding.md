# stream1 samples are signed int16 — what that overturns, and where the PSA
# saturation flag actually stands

**2026-07-29, evening.** Written after going into the n_TOF `detector` (PSA)
source to find the saturation flag. The investigation ended somewhere other than
it started: the flag is fine on the liquids, and three findings recorded earlier
today were artifacts of reading the raw samples with the wrong sign.

Supersedes, on the points listed below: `FINDINGS_2026-07-29_pre_ship_tests.md`
(NEW 1, NEW 2/T8, NEW 3), `liq_study/FINDINGS_liquids.md` (warning §1),
`liq_study/adc_range_census.py`, `liq_study/adc_wrap_examples.py`.

---

## Headline

| claim recorded earlier today | status now |
|---|---|
| the ADC **wraps** under-range instead of clipping | **wrong** — artifact of a `<u2` decode; there is no wrap |
| the largest measurable amplitude is the baseline (~31 000) | **wrong** — it is ~63 800, about twice that |
| `satuflag` catches only a third to a half of saturated liquid hits | **wrong** — it catches them, 119/123 verified per pulse |
| `satuflag` is never set on any wall | **right, and now explained** — and it is not a rail-test problem |
| PSA `tof` and the raw sample index do not align per hit | **wrong** — they align exactly, once a 259-sample offset is applied |

---

## 1. The decoding

`nTof_x17_DAQ/stream1_monitor/ntof_raw.py:163` reads ACQC payload samples as
`<u2`. They are **`int16_t`** — that is how ntoflib reads them
(`ntoflib/include/ReaderStructACQC.h:41`, `std::vector<int16_t> data`), and the
DAQ settings written into our own output files say the same thing:

| detector group | `fullScalemV` | `baselineOffsetmV` | `zeroSuppThrSign` | measured baseline (int16) |
|---|---|---|---|---|
| LIQ, PSS, SILI | ~2004 | **+950** | 0 (negative-going) | LIQA +31 222, PSSA +30 830, SILI +26 346 |
| WAL, PKUP | ~2004 | **−950** | 1 (positive-going) | WALA −31 407, WALB −31 406, PKUP −26 664 |

±32 768 codes span ±1002 mV, so ±950 mV is ±31 070 codes — the measured
baselines agree to better than 1 % on every channel. Each channel is offset to
~95 % of the way toward one rail, leaving ~52 mV of headroom and ~1950 mV of
swing in the pulse direction.

Decoded signed, every trace is continuous. What the unsigned decode showed as a
pulse "running through 0 and reappearing at 65 535" is a pulse crossing zero.
The cards are S014 (ADQ14); ntoflib's `getNbBits()` hard-codes 16 for that type
(`ReaderStructMODH.cpp:665-690`, with a `return 14` commented out just below),
and the data agree with 16 — sample values populate all residues mod 4 and both
−32768 and +32767 occur — so the rails really are ±32768.

**Consequence for every earlier raw-waveform number:** the usable amplitude is
**~63 800 ADC**, not ~31 000. Pulses previously called "0.4–17 % too big for the
range" are ordinary half-scale pulses.

## 2. What saturation actually looks like

Figures in `liq_study/`: `sat_examples_liq.png` (individual pulses),
`sat_population_liq.png` (run lengths, times, flat-top vs depth),
`sat_clip_or_wrap.png` (clip versus rail-to-rail flip).

Census over three 430 MB chunks (segments 8, 20, 40; ~640 k blocks),
`saturation_examples.py`:

| det | blocks | baseline | deepest excursion | clipped blocks | at physics time |
|---|---|---|---|---|---|
| LIQA | 7 241 | +31 220 | 64 003 | 14 | **4** |
| LIQB / LIQC / LIQD | 9 688 / 2 024 / 5 591 | +31 1xx | ~63 900 | 8 / 8 / 8 | 0 |
| PSSA / PSSB / PSSC | 35 027 / 30 726 / 39 175 | +30 8xx | ~63 600 | 8 / 7 / 7 | **0** |
| PSSD | 42 526 | +30 888 | 44 830 | 0 | 0 |
| WALA–D | 53 k–180 k | −31 4xx | 32 888–34 635 | 56–65 | **0** |
| SILI | 3 907 | +26 368 | 59 786 | 26 | 0 |

- **Plastics, SiPM walls and SILI saturate only in the gamma flash.** The
  liquids are the only detectors that clip at physics times, and only LIQA does
  so with any regularity.
- **The walls never reach their own rail in the pulse direction** — largest
  excursion ~34 600 counts ≈ 1 060 mV of a 2 004 mV range, i.e. the wall front
  end limits at about half of ADC full scale. What clips on a wall is the
  negative *undershoot* during flash recovery, from a baseline only ~1 360
  counts above the rail.
- **A physics-time liquid clip is 2–5 samples wide**; a flash clip is 70–130.
- **The liquids show no front-end compression.** Flat-top width at the peak
  stays 1–3 ns for every pulse depth from 20 000 ADC up to ~64 000, and only
  jumps to 50–130 ns once the rail is reached. So for the liquids, ADC clipping
  is the whole of the saturation story, and a rail test is the right instrument.

### Clip, or wrap?

`saturation_clip_or_wrap.py`, 123 clipped runs over seven chunks: **there is no
arithmetic wrap anywhere.** Saturated samples are always exactly at a rail code.
The deepest flash saturations on LIQA (3 of 13 runs) and LIQB (2 of 6) show a
**rail-to-rail flip** — the output jumps from −32768 to exactly +32767 for 3–7
samples and back:

```
LIQB  [-32768, -32768, -32768, -32768, -32768, -32768, 32767, 32767, 32767, -32768, ...]
```

A true wrap would store `true + 65536`, i.e. arbitrary values below the positive
rail, and the sequence would keep descending from the top. Only rail codes
occur, with at most one transition sample as the ADC crosses. Flips happen only
in the flash, only on LIQA/LIQB, never at physics time.

## 3. Does `satuflag` fire? Yes.

Where the flag comes from, in the `detector` package (identical on `master` and
`dev_rz`):

| step | location |
|---|---|
| rails | `PSAdetector.cc:776-777` — `sign * modh.getMin/MaxDataValue()` |
| the test | `PSA_Functions.cc:2793-2806`, `AnalyseSaturation` — exact equality with either rail, over samples `[lower[i], upper[i]]` of each found pulse |
| flash path | `PSA_Functions.cc:104-163` → `PSAdetector.cc:2132` |
| to the tree | `pulse.saturation` → `detector.cc:221` → branch `satuflag` |

`SignalY_type` is `double` (`PSAdetector.h:17`), so the `+32768` that a
positive-polarity detector produces is exact — there is no signed-16-bit
overflow in the comparison, and both rails are tested for every detector.

**Verification** (`dump_clips.py` → `verify_satuflag.py`): every genuine clipped
liquid run in seven raw chunks, matched to the reprocessed trees by
`segment` + `BunchNumber` + time.

- **119 of 123 clipped runs have a flagged hit within 100 ns**
- **7 of 7 physics-time clips**, including 2-sample ones
- median |Δt| **3.2 ns**, p90 11.4 ns
- accidental rate: 1.6–4.2 flagged hits per bunch over a 20 ms window, so
  P(chance match in ±100 ns) ≈ 4 × 10⁻⁵ — these are per-pulse identifications

The 4 non-matches are flash-region runs that merge into a single flash pulse.

**T8's "catches only a third to a half" compared against the artifact rail** (the
baseline, from the unsigned decode). Against the real ceiling the flagged hits
sit at amp ≥ 64 000, exactly where they should.

## 4. The raw ↔ PSA time base: a constant 259 samples

`time_base_offset.py`, isolated large late pulses matched to their PSA hit:

| detector | pulses matched | Δt = tof − raw peak index | spread (p84−p16) |
|---|---|---|---|
| LIQA | 135 / 135 | **−258.7 ns** | 1.1 ns |
| LIQD | 85 / 85 | **−258.9 ns** | 1.2 ns |

A constant, not a scatter. The block `start` counter in our parser is the
zero-suppression **trigger** sample, while the payload begins ~259 samples
earlier (the pre-samples), so `start + j` over-counts by exactly that. The flash
block starts at 0 and carries no pre-samples — which is why flash pulses match
with no offset.

**This closes NEW 3 of the pre-ship findings.** Per-hit raw-to-tree matching is
reliable; it needed the offset, not a caveat. Use

```
tof = start + j - (259 if start > 0 else 0)
```

Note the earlier "+19 to +26 ns" figure was measured a different way and is
superseded.

## 5. What is genuinely not flagged

Three real gaps remain, none of which is "the rail test is too strict":

1. **The walls, structurally.** Their saturation is an undershoot — opposite to
   the pulse direction — so it never lands inside a detected pulse window, and
   `AnalyseSaturation` only scans `[lower[i], upper[i]]`. Concretely, WALA
   segment 8 bunch 161: 6 569 hits, 111 in the flash with amp up to 25 533 and
   χ² up to 1.3 × 10⁵, **zero flagged**, while that channel's raw trace clips for
   21–27 samples at a time across the same microseconds (checked in the raw:
   the trace walks into the rail and back out, it is not zero-suppression fill).
2. **Only the clipping pulse is flagged, not its victims.** In the flash windows
   above there are 74–117 hits and 1–5 flagged; everything riding on the
   distorted recovery is reported clean.
3. **`amp` on a flagged hit is not a measurement** — 66 000 to 832 000 against a
   63 800 ceiling, because the fit extrapolates through the excluded samples.
   Cut flagged hits; do not try to correct them.

A fourth, latent: **the zero-suppression fill value is 0x8000, the same code as
the negative rail** (LIQA: 17 fill runs against 14 genuine clips in three
chunks). Fill and clip are distinguishable only by context — a clip is
approached, a fill is not. The PSA does not make that distinction; it has not
bitten us because fill samples rarely fall inside a found pulse window, but any
change to the rail test has to handle it.

Note also that a *tolerance band* around the rails, the obvious first fix, is
poorly suited to this hardware: the clip code is exactly `±rail`, and the liquid
baseline sits only 1 547 counts below the positive rail, so a wide band would
flag ordinary baseline noise. If anything goes upstream it should be the
window/polarity blind spot (1) and severity information (`n_sat`), not a
tolerance.

## 6. Tools added in `liq_study/`

| script | what it does |
|---|---|
| `saturation_examples.py` | census + example figures, signed decode, fill-aware |
| `saturation_clip_or_wrap.py` | clip vs rail-to-rail flip, per detector |
| `dump_clips.py` | ns-precision list of clipped runs from one raw chunk |
| `verify_satuflag.py` | per-pulse match of clipped runs to `satuflag` in the trees |
| `check_satuflag.py` | the same at ±1 µs, from the block table (kept: it is what first raised the question) |
| `plot_clip_with_hits.py` | raw block with PSA hits overlaid, flagged ones in red |
| `time_base_offset.py` | measures the raw ↔ `tof` offset on a control population |

Inputs: raw chunks in `/media/dylan/data/x17/ntof_raw_224572/head_*.bin`,
reprocessed output in `/media/dylan/data/x17/ntof_reproc/v12_liqpileup/`
(part `NNNN` covers raw segments `10*(NNNN-1)` … `10*NNNN-1`, which is how the
`segment`/`BunchNumber` branches locate a raw block).

## 7. Open items

- **`ntof_raw.py:163` still decodes `<u2`.** Everything above was done by
  reinterpreting in the analysis scripts; the shared parser in the
  `nTof_x17_DAQ` repo has not been changed, and should be. Every consumer of it
  is affected.
- `ntof_handoff/README.md` was to carry a `satuflag` warning and a wrap warning.
  The wrap warning should be dropped; the `satuflag` warning should be rewritten
  to say the flag is reliable on the liquids, absent on the walls, and that
  flagged hits must be cut rather than used.
- Whether the wall/undershoot blind spot is worth a merge request upstream is
  still open — it affects flash-time hits only. Forks are cloned at
  `/afs/cern.ch/user/d/dneff/ntof_src/{ntof-detector,ntof-raw-2-root}`
  (`dev_rz` and `master`, `upstream` remotes wired, Kerberos over port 8443).
