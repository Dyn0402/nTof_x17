# Pre-ship tests: results

> **CORRECTION, 2026-07-29 evening.** Three results below are wrong, all from
> one cause: the raw samples are **signed** int16, and the tooling read them as
> unsigned. **NEW 1 (the ADC wrap) does not exist**, **NEW 2 / T8 (`satuflag`
> unusable) is wrong** — the flag is verified on 119/123 clipped liquid runs —
> and **NEW 3 (tof vs raw index) is solved**: a constant 259-sample offset.
> The T1-T6 results and the ship decision are unaffected.
> See `FINDINGS_2026-07-29_signed_decoding.md` before acting on anything here.

**2026-07-29.** Running `PRE_SHIP_TESTS.md` against the candidate final
UserInput `v12_liqpileup`, on run 224572 partials 0001+0002 (bunches 1-397) and
seven raw stream1 chunks.

Read `PRE_SHIP_TESTS.md` first -- it states each test and its decision rule.
This file records what came back, including two things nobody was looking for.

---

## Headline

| test | result | verdict |
|---|---|---|
| T1 matcher on a larger sample | **v12 96.3 % / 0.5 %, v4 95.3 % / 0.5 %** over 252 bunches (was 100) -- gap preserved | **confirmed, go** |
| T2 chi2 ordering | wall templates still win on 3.8 M hits/tree (was ~200 k); liquid chi2 p90 better | **passes** |
| T3 sideband robustness | every metric moves <= 2.4 % over sideband position and +-50 % width | **passes** |
| T4 new liquid hits real? | proxies ambiguous; population budget favourable but bracket-dependent; **per-hit raw matching could not be established** | **not closed** |
| T5 does `afast` carry PSD? | median roughly right, but per-hit spread 4-9x the physical band and drifts 2x with amplitude | **keep, document hard** |
| T6 photon floor on B/C | holds on A/C/D; **LIQB genuinely violates it** | qualified |
| T8 saturation census | 0.006-0.06 % of hits above what was taken for the rail | ~~`satuflag` is not usable~~ **RETRACTED — wrong rail; flag verified good** |
| walls/plastics v11 vs v12 | hit counts bit-identical in all eight trees | confirmed |
| NEW | ADC **wrap-around** at the end of range, not clipping | **RETRACTED — no wrap exists** |
| NEW | PSA `tof` and raw sample index do not align per hit | **RESOLVED — constant 259-sample offset** |

Nothing here changes the wall or plastic configuration, and nothing found is a
reason to hold the UserInput.

### Recommendation

**Ship v12 as it stands**, and add three warnings to `ntof_handoff/README.md`
(`satuflag`, `aslow`/`afast`, the wrap-around). The case:

- the go/no-go test (T1) is green on 2.5x the sample with the gap intact, and
  the two confirmatory tests that could have forced a re-grade (T2, T3) both
  pass on much larger samples than before;
- T5 does not justify a reprocess. `afast` is approximately right in aggregate
  and useless per pulse; a documentation fix costs nothing and a new variant
  costs a full reprocess-and-grade cycle;
- T4 is genuinely open, but it is open on the *evidence*, not against it. Every
  measurement that does exist -- rate profile, chi2 p50 neutral and p90 better,
  hit count still under the raw resolvable-pulse ceiling -- points the same way,
  and the reason it is not closed is a raw/PSA alignment question that is not
  specific to v12 and that one answer from n_TOF would probably resolve.

**Do not** ship the liquid yield as a confirmed number. Say +14-21 % with the
caveat that 95 % of the gain is resolved shoulders on existing pulses and that
the individual-hit check remains open.

---

## NEW 1. The ADC wraps at the end of its range; it does not clip

Found while checking T6. The report says the largest liquid pulses "reach the
~31 000 ADC rail". They reach a rail, but what happens there is not clipping,
and the difference matters.

The samples are unsigned 16-bit, and every channel sits on a baseline well
inside that range, so the largest measurable amplitude is the distance from the
baseline to whichever end the pulses run toward. A pulse bigger than that
**wraps** modulo 65536 -- it reappears at the opposite end -- rather than being
clamped.

**Which end depends on the polarity, and it is not the same for all of them.**
Measured on in-range blocks (`liq_study/adc_range_census.py`, polarity column;
consistent with §0 of `FINDINGS_2026-07-28_psa_optimization.md` and with the
sign of the shipped templates):

| | PSS, LIQ | WAL, PKUP |
|---|---|---|
| polarity | **negative**-going, 100 % of blocks | **positive**-going, 91-100 % |
| baseline | 30 700-31 200 | 33 700-38 900 |
| a too-big pulse runs | below 0 | past 65 535 |
| and reappears near | 65 535 | 0 |

An earlier version of this section said all of them were negative-going. They
are not, and it matters twice over: the wall wrap is an *over*-range wrap, and a
"sample above 60 000" test — which is right for a liquid — flags ordinary large
pulses on a wall. The polarity-independent test is the **discontinuity**: a step
of more than 20 000 ADC between adjacent samples, which no real pulse produces
at 1 GS/s. That is what the census now uses.

For a liquid, from the raw stream:

```
LIQA  [ ... 32767 32767 32768 63712  4641 15598 27611 32160 ... ]
                              ^^^^^ a sample that should have been below 0
```

Consequences:

- a flat-top test does not catch it -- there is no flat top;
- the reported `amp` is whatever the last un-wrapped sample on the rising edge
  happened to be, so it is **randomly under-reported**, not clipped to a
  constant. That is why the wrapped population in the isolated-pulse sample
  spans amp 23 777-31 210 rather than piling up at one value;
- the fitted shape sees a full-scale positive spike one or two samples after the
  peak, which no template matches.

Census over two raw chunks (`liq_study/adc_range_census.py`, step test), as a
fraction of zero-suppressed blocks containing at least one wrap:

| | LIQA | LIQB | LIQC | LIQD | PSSA-D | WALA-D | SILI |
|---|---|---|---|---|---|---|---|
| blocks with a wrap | 0.67 % | 0.09 % | 0.39 % | 0.33 % | 0.03-0.04 % | 0.05-0.09 % | 0.84 % |
| of those, late-time | 27/37 | 1/7 | 0/6 | 7/14 | **0/41** | **0/178** | 1/25 |

**The walls and plastics are affected only during the flash**, where saturation
is expected and already understood. The liquids are the only detectors where it
happens at physics times, and even there it is a sub-percent effect.

### What they look like

`liq_study/adc_wrap_examples.py` draws the late-time ones. Three figures:
`adc_wrap_as_recorded.png` is the honest one — the stored samples, nothing
subtracted and nothing undone, on a 0…65 535 axis; `adc_wrap_examples.png`
overlays our reconstruction of the true pulse; `adc_wrap_summary.png` is the
population. Over the same two raw chunks, 21 of 13 165 late-time liquid blocks
wrap: 17 LIQA, 3 LIQD, 1 LIQB, spread over 1.0-21.1 ms of time of flight, i.e.
ordinary physics pulses far from the flash.

As recorded, a wrapped liquid pulse is unmistakable: the trace sits flat at the
~31 200 baseline, dives toward 0, and one or two samples appear up at 63-65 000
before it resumes its normal fall.

Every panel is labelled with its provenance — file segment, bunch and trigger
number — and the script prints the same for all of them, so any block can be
pulled up again:

```
det    seg   bunch      trig   t [ms]  nwrap  true peak
LIQA    20     398    182348    3.398      1      32414
LIQA    20     399    182357    1.096      2      32315
...
LIQA    40     799    184651    6.248      2      36656
LIQD    40     800    184654    5.074      1      32474
```

They are not concentrated in one bad bunch or one bad segment: 21 blocks over
6 bunches in 2 segments, LIQA 17 / LIQD 3 / LIQB 1, spread across the whole
time-of-flight window. That is the signature of ordinary large pulses
occasionally exceeding the range, not of a periodic or bunch-correlated
artefact.

In pulse-height coordinates (baseline − sample, so pulses point up) the
recorded trace rises normally, plunges to about −34 000 for **one or two
samples** at the peak — three at most, never a plateau — and then continues
down the falling edge as if nothing happened. Undoing the wrap
(`sample − 65536`) recovers a clean pulse of 31 340-36 656 ADC against a
ceiling of ~31 200, i.e. these pulses are only **0.4-17 % (median 6 %) too big
for the range**. Nothing else about them is unusual, which is exactly why the
effect is easy to miss: it is one sample in a hundred-nanosecond pulse, on
0.16 % of late liquid blocks, and it moves `amp` *down* rather than pinning it
at a rail.

Detection recipe, in order of preference: on the raw samples, an adjacent-sample
step above 20 000 ADC — polarity-independent, so it works on walls too. For
liquids and plastics only, `rows.min() < -0.5` on peak-normalised pulses is
equivalent and cheaper. On the reconstructed output there is no clean signature
for the under-reported case, only for the fit-corrupted one — see NEW 2.

## NEW 2. `satuflag` does not flag saturation (this is T8)

Counting hits whose reported `amp` exceeds the per-channel baseline -- i.e.
physically impossible amplitudes -- in v12, partial 0001:

| tree | hits | rail | amp > rail | `satuflag` set | max `amp` |
|---|---|---|---|---|---|
| LIQA | 3 369 621 | 31 220 | 2 006 (0.060 %) | 791 (0.023 %) | 3 234 415 |
| LIQB | 3 728 129 | 31 156 | 593 (0.016 %) | 456 | 3 649 367 |
| LIQC | 983 979 | 31 122 | 346 (0.035 %) | 346 | 2 175 592 |
| LIQD | 2 305 162 | 31 108 | 659 (0.029 %) | 382 | 763 834 |
| WALA | 1 894 802 | 34 148 | 224 (0.012 %) | **0** | 42 645 |
| WALB | 1 844 839 | 34 003 | 1 069 (0.058 %) | **0** | 43 427 |
| WALC | 2 046 320 | 34 294 | 577 (0.028 %) | **0** | 43 872 |
| WALD | 1 995 146 | 34 492 | 809 (0.041 %) | **0** | 42 773 |
| PSSA | 5 582 034 | 30 841 | 491 (0.009 %) | 203 | **243 257 568** |
| PSSC | 9 043 427 | 30 712 | 519 (0.006 %) | 198 | **320 356 608** |
| PSSD | 8 002 519 | 30 925 | 526 (0.007 %) | **0** | 65 542 |

`satuflag` is **never set on any wall**, and catches only a third to a half of
the over-rail liquid hits. The affected fraction is tiny, so this does not
threaten the processing -- but a single hit with `amp` = 2.4e8 entering a sum,
an average or a calibration is catastrophic, and nothing in the output warns
about it.

**For the handoff:** tell users to cut on `amp` above the per-channel baseline
(~31 000 liquids and plastics, ~34 100-34 500 walls) and not to rely on
`satuflag`. This applies to the official processing too -- it is a property of
the PSA and the DAQ, not of our UserInput.

## NEW 3. PSA `tof` and the raw sample index do not align per hit

This blocked the per-hit half of T4 and should be known before anyone else
tries it.

**Updated later the same day, after actually reading the PSA guide.** The guide
we already have (`~/x17/ntof_processing/PSA_Guide_20240704.pdf`, "Timing
properties") defines the branches, and I should have checked it first:

- `tof` is a **30 % constant-fraction arrival time**, not the peak;
- `peak_tof` is the **peak moment** -- first highest point, parabola vertex, or
  fitted-Pulse-Shape peak, depending on `AMPLITUDE OPTION`. It exists in the
  tree and I had not looked;
- the guide gives the conversion between them, `arrival = peak - dt`.

That is internally consistent with the data: `peak_tof - tof` is **1.3 ns**
median (p16-p84 0.6-2.6 ns), exactly what a 30 % crossing on a 6 ns FWHM pulse
should give.

**It does not explain the offset.** The raw pulse peak sits **+26 ns after
`peak_tof` on LIQA and +19 ns on LIQD** -- the definition accounts for ~1 ns of a
~20-28 ns discrepancy. So the question is sharper now, not answered: *why does
the reconstructed time sit ~20-28 ns before the raw sample-index peak, by a
per-detector constant?* It is not a sampling-rate mismatch (the lag is constant
across absolute times from 1.7 to 7.0 ms) and there is no per-detector time
offset in the UserInput. Leading guess: the ACQC block `start` in stream1 and the
sample origin the PSA uses differ by a per-channel amount.

Also checked: the `waveform` branch exists in every tree but is **empty**
(length 0 for every hit), so there is no shortcut around stream1.

What is established:

- the bunch identification is certain. Raw bunch 161 scores 20.1 % of its large
  isolated peaks against PSA `BunchNumber` 161, against a 1.5 % background over
  all 197 candidate bunches (median score 2/334);
- there is a stable per-detector lag between `tof` and the raw pulse peak:
  **+28 ns LIQA, +29 ns LIQB, +21 ns LIQD**, measured two independent ways
  (stacking the raw trace on every hit, and matching large isolated raw peaks
  back to the nearest hit). For raw peaks above 4 000 ADC the lag is tight,
  p16-p84 = 27-30 ns on LIQA.

What does not work:

- only ~20 % of unambiguous large raw pulses have a PSA hit at that lag;
- the matched hit's `amp` is ~2 % of the raw peak height for large pulses;
- individual overlays show hits on stretches of flat baseline. Around one clean
  20 290 ADC pulse with 140 ns of quiet in front of it, the PSA reports eight
  hits, only one of which is at the pulse.

I could not resolve this. It is **not** a v12 problem -- four of those eight
hits are in v11 as well -- and it does not affect any count-based result, since
those only need a hit to land in the right ~1000 ns block. But **no per-hit
raw-vs-PSA classification from this repo should be believed until it is
understood**, and that includes anything built on `deconv_vs_psa.py`'s matching.

---

## T1 -- the headline holds on 2.5x the sample: GO

The 96.4 % rested on 100 bunches of one DREAM sub-run, which `REVIEW.md` Section
4.1 called the thinnest sample under the biggest claim. Re-run on all 252
bunches that partials 0001+0002 and `run_79 / stat090_0000` have in common
(146-397), grading v4 and v12 on exactly the same bunches:

| | 100 bunches (before) | **252 bunches** |
|---|---|---|
| v4 singles matcher | 95.2 % / 0.6 % | **95.3 % / 0.5 %** |
| v12 singles matcher | 96.4 % / 0.6 % | **96.3 % / 0.5 %** |
| gap | +1.2 pp | **+1.0 pp** |
| v4, hardest 1-3 ms bin | 93.4 % | 93.5 % |
| v12, hardest 1-3 ms bin | 95.5 % | 95.0 % |
| wall-only (both) | 98.9 % | 98.7 % |
| plastic leg costs, v12 | 2.5 % | 2.4 % |

Every number reproduces within 0.5 pp and **the gap survives**, so the v8-family
win was not a fluctuation of the small sample. `match_window` is 99.9 % and the
measured per-arm time-base offsets are +2.0/+1.5/+0.5/-3.0 ns, i.e. still
consistent with zero on a correctly processed file.

This was the go/no-go. It is green.

## T2 -- fit chi2 on a 20x larger sample: ORDERING HOLDS

The chi2 comparison had used one partial (0016, ~20 bunches, ~200 k hits per
tree). Re-run on partials 0001+0002 against the official merged file, capped at
3 M entries per tree per file:

| tree | official chi2 p50 | v4 = v12 | change | chi2 p90 official -> v12 |
|---|---|---|---|---|
| WALA | 0.903 | **0.852** | -5.6 % | 4.17 -> 4.08 |
| WALB | 1.223 | **0.996** | -18.6 % | 6.67 -> 5.35 |
| WALC | 1.121 | **0.990** | -11.7 % | 5.52 -> 4.90 |
| WALD | 1.181 | **1.059** | -10.3 % | 5.70 -> 5.40 |

The wall templates still win on every arm, on 3.7-4.1 M hits per tree instead of
~200 k, with amplitudes 2-4 % higher and 24-38 % more hits. The scorecard's
"WAL chi2 p50 0.85-1.06" reproduces exactly. **T2 passes.**

Liquids, v12 against official: chi2 p50 is neutral (+0.1 %, +0.1 %, +1.9 %,
-7.1 % on A/B/C/D) while chi2 **p90 is clearly better** (31.4 -> 25.3,
37.5 -> 35.2, 19.5 -> 15.8, 21.6 -> 16.2). The extra hits are not being bought
with a worse tail of bad fits -- if anything that tail shrinks.

Plastics: official and v4 report `no shape fit` (AMPLITUDE OPTION != 2); v12
fits, at chi2 p50 1.21-1.34. The 41-48 % lower `amp` is the known
amplitude-definition change, not a degradation.

**Two caveats.** (i) `REVIEW.md` Section 4.2 says the chi2 differences were
"large (2x) so the ordering is probably safe". For the *walls* the real margin
is 5.6-18.6 %, not 2x -- the 2x was the liquid template failure. The ordering is
safe because the sample is 3.8 M hits, not because the margin is big.
(ii) the entry cap reads the official file's *first* 3 M entries per tree while
the candidates supply up to 6 M, so the two are not drawn from identical bunch
ranges. chi2 p50 is not expected to move much across a run, but the comparison
is not exactly like-for-like.

## T3 -- sideband robustness: PASSES, with one correction

Moving the off-time sideband and resizing it, on v12 partial 0001:

| sideband start / width | T1 sigma | T2 sigma | A1 peak | FWHM/peak | A2 resid | flatness | T3 walk |
|---|---|---|---|---|---|---|---|
| 300 / 20 (default) | 6.65 | 6.41 | 1080 | 1.22 | 0.362 | +9.2 % | 1.48 |
| 200 / 20 | 6.65 | 6.41 | 1080 | 1.22 | 0.362 | +9.2 % | 1.48 |
| 450 / 20 | 6.65 | 6.41 | 1080 | 1.22 | 0.362 | +9.2 % | 1.48 |
| 300 / 10 | 6.65 | 6.41 | 1054 | 1.23 | 0.368 | +8.9 % | 0.37 |
| 300 / 30 | 6.65 | 6.41 | 1081 | 1.22 | 0.366 | +9.3 % | 1.57 |

Largest excursion of any quoted number is the MIP peak at -2.4 %, well inside
the 10-20 % the test asked for. The report numbers stand.

**Correction to what T3 was testing.** The sideband position only enters A1 and
A2, which select on it. T1 and T2 sigma do *not* use it -- `peak_width`
estimates its own flat accidental level from the outer thirds of each dt
distribution -- which is why they do not move at all. So this does not test the
"38.8 -> 6.46 ns" subtraction; that would need `peak_width`'s own span and fit
window varied. What it does establish is that the MIP and linearity numbers are
not artefacts of where the sideband was put.

**`T3 walk` is not comparable across settings** -- it runs 0.37 to 1.57 ns
purely because the prompt window bounds it. It is a valid guard *between
variants at fixed settings*, which is how it has been used, and should not be
quoted as an absolute.

## T4 -- are the extra liquid hits real? NOT CLOSED

The yield claim reproduces exactly: +21.3 / +16.3 / +14.2 / +19.5 % on
LIQA/B/C/D, chi2 neutral-to-better, walls and plastics untouched.

**Something not previously reported:** the net gain is a difference of two
larger numbers. v12 also *loses* hits v11 had:

| | LIQA | LIQB | LIQC | LIQD |
|---|---|---|---|---|
| v12-only (new) | 205 432 (23.7 %) | 168 210 | 24 071 | 115 562 |
| v11-only (lost) | 59 268 (**8.4 %**) | 44 420 (6.5 %) | 6 975 | 26 779 (6.1 %) |

### The cheap proxies come out ambiguous

- **(ii) time of flight: passes.** The new hits follow the pre-existing rate
  profile to 4-7 % in every decade bin. But this proxy has little power here,
  because 95 % of the new hits are splits of existing pulses and therefore
  inherit the profile of the pulses they were split off.
- **(i) amplitude: fails.** New hits crowd the amp-50 elimination cut 3-4x more
  than pre-existing ones (19-22 % within 10 ADC of the cut, against 5-6 %).
  That is the fake-population signature -- and also exactly what a genuinely
  resolved shoulder on a big pulse's tail looks like.
- **(iii) split vs recovery:** 94-96 % split, 4-6 % recovery, and 53-60 % of
  the splits are within 10 ns of the pre-existing hit, which is under two pulse
  FWHM.

### The population budget

Since the per-hit test could not be made to work (NEW 3), the same question was
asked without needing per-hit alignment: over exactly the same zero-suppressed
samples, how many pulses does the raw data resolve, and how many does each
processing report (`liq_study/raw_pulse_budget.py`)?

| raw "resolvable pulse" definition | raw count | v4 / v11 | v12 |
|---|---|---|---|
| local max > 5 sigma, dominating +-3 ns | 18 043 | 0.77 | **0.96** |
| local max > 5 sigma, dominating +-1 ns | 27 041 | 0.51 | 0.64 |
| local max > 10 sigma, dominating +-3 ns | 11 527 | 1.21 | 1.51 |

Read the middle-strictness row: v12 moves the yield from 77 % to 96 % of the
number of pulses the raw data resolves, without crossing it -- recovery, not
invention. But the answer is **bracket-dependent**: on the strictest definition
v12 reports 1.5x the raw count, and a count taken at the UserInput's own amp-50
threshold is noise-dominated and gives 0.09. There is no threshold-independent
statement here.

**Verdict: not closed.** The evidence leans towards the new hits being real
(rate profile, chi2 neutral-to-better, budget below the ceiling at the natural
threshold), but T4's own decision rule -- ">= 85 % in classes (a)+(b)" -- was
not evaluated, because the per-hit classification it calls for could not be
made trustworthy.

## T5 -- the 5000/30 boundary: KEEP, but it is not a PSD variable

`afast` fills 100 %; `aslow` stays at 0 %, as expected, because the slow
component lies outside the reconstructed pulse boundary. `afast` is **not
degenerate** -- it never equals `area` -- so the "obviously drop it" branch of
the decision rule does not apply either.

**First, a correction to the target.** The 0.21 tail band in
`FINDINGS_liquids.md` is the fraction of the pulse beyond **12 ns**. `afast`
splits at **30 ns**, so 0.21 is the wrong number to compare against and makes
the PSA look twice as bad as it is. Re-measured on the same isolated raw pulses
at a 30 ns split:

| tree | raw p16 | raw p50 | raw p84 | width |
|---|---|---|---|---|
| LIQA | 0.101 | 0.119 | 0.136 | 0.035 |
| LIQB | 0.083 | 0.108 | 0.124 | 0.041 |
| LIQC | 0.095 | 0.115 | 0.139 | 0.044 |
| LIQD | 0.096 | 0.113 | 0.129 | 0.033 |

Now the PSA, on isolated (no neighbour within 200 ns) late-time hits, amp>3000:

| tree | n | p16 | p50 | p84 | width | small vs large pulses |
|---|---|---|---|---|---|---|
| LIQA | 20 002 | 0.021 | 0.097 | 0.243 | **0.222** | 0.136 -> 0.069 |
| LIQB | 14 059 | 0.018 | 0.124 | 0.325 | **0.307** | 0.155 -> 0.100 |
| LIQC | 1 313 | 0.019 | 0.085 | 0.178 | **0.159** | 0.095 -> 0.074 |
| LIQD | 12 336 | -0.003 | 0.055 | 0.139 | **0.142** | 0.081 -> 0.039 |

**The median is approximately right** -- LIQA 0.097 against 0.119, LIQB 0.124
against 0.108, LIQC 0.085 against 0.115, LIQD 0.055 against 0.113. So the split
is doing roughly the right thing *in aggregate*.

**The width is 4-9x too large.** The physical band is 0.033-0.044 wide; what
comes out is 0.14-0.31. The per-hit value is therefore dominated by
reconstruction scatter, not by the pulse's tail fraction. It also **falls by a
factor two with amplitude**, which a real PSD variable does not do.

**Verdict: keep the boundary, and document it hard.** It costs nothing, the
aggregate number is usable, and dropping it would cost a full
reprocess-and-grade cycle -- there is no existing variant with the `STEP SIZE`
change but *without* the boundary, so "v11 + STEP SIZE 1/3 only" has never been
run and it cannot be assumed that removing the boundary leaves the other
branches untouched. But `ntof_handoff/` must say, in the strongest terms:

- `aslow` is **always zero**. It is not a measurement.
- `afast` is meaningful only in aggregate, and only for isolated pulses.
- `(area - afast)/area` is **not an n/gamma discriminant**. Its per-pulse
  resolution is 4-9x the physical spread and it drifts with amplitude.
- and, independently: there is no n/gamma separation to find in this data
  anyway -- the raw tail/total distribution is a single band, not bimodal.

## T6 -- the photon-statistics floor: qualified, and LIQB is different

Two corrections to the existing statement.

**First, the "saturation breaks the scaling at ~31 000 ADC" line was measuring
the wrap.** With under-range pulses removed (`rows.min() < -0.5`), the top
amplitude bin behaves:

| tree | old top-bin resid/sqrt(A) | with wraps removed |
|---|---|---|
| LIQA | 3.11 | 0.95 |
| LIQD | 2.18 | 0.70 |

**Second, extended to all four detectors** (LIQB/LIQC were never measured; the
npz does carry them, and the six-bin split was silently dropping LIQC because
every bin fell under the minimum count):

| tree | pulses | amp span | resid/sqrt(A) spread | resid/peak falls |
|---|---|---|---|---|
| LIQA | 878 | 19x | +41 % | 3.4x |
| **LIQB** | 573 | 17x | **+158 %** | 2.0x |
| LIQC | 116 | 2x | +14 % | 1.3x |
| LIQD | 1143 | 9x | +16 % | 2.8x |

LIQD is flat, LIQA nearly so, LIQC is flat but only spans a factor 2 so it is a
weak test. **LIQB is not flat**: `resid/sqrt(A)` runs 0.62 -> 1.59 and
`resid/peak` stops falling at 0.012, which is the systematic-shape signature.

Is that a shape a template could absorb? Measured, not argued
(`liq_study/amp_binned_basis.py`): fit with one template, then with one template
per amplitude octile, always scored on a held-out half.

| tree | 1 template | 8 by amplitude | gain | gain in the top quartile |
|---|---|---|---|---|
| LIQA | 82.6 | 79.7 | -3.4 % | -5.6 % |
| **LIQB** | 97.2 | 73.4 | **-24.5 %** | **-27.4 %** |
| LIQD | 52.4 | 51.2 | -2.3 % | -5.5 % |

So there really is an amplitude-dependent shape in LIQB. Looking at it directly,
LIQB's *small* pulses are the odd ones: they are narrower and nearly tail-free
(tail/total 0.136 against 0.188 for LIQD), while LIQB's large pulses have the
same shape as every other liquid. LIQB is also the detector with the most
blocks and the fewest isolated pulses.

**This does not reopen the shipped configuration.** v12 keeps the *shipped*
templates, and the reason measured templates were rejected (v3/v5/v9) was never
fit quality -- a measured template was already 3-4x better on isolated pulses
and still made the processed output worse, because a longer template overlaps
more neighbours. A 24 % residual improvement on isolated LIQB pulses is the same
kind of evidence that did not transfer three times.

What it does change is the report: the sentence "the residual is photon
statistics, so no template can help" is true of LIQA/LIQC/LIQD and **not** of
LIQB, and should say so.

## Walls and plastics

v11 and v12 hit counts are identical in all eight trees (WALA 1 894 802,
PSSC 9 043 427, ...), confirming the claim that the wall and plastic
configuration is untouched, so every wall/plastic conclusion carries to v12.
