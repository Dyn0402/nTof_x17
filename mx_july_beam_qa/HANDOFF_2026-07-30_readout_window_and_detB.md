# Handoff: the run_79 readout window, and what is wrong with detector B

**Written 2026-07-30. Audience: someone picking this up cold.** Everything
marked **[measured]** was executed in the session that wrote this and is
reproducible from the commands quoted; **[inferred]** is a reading of the
numbers that has *not* been confirmed. The distinction matters most in §4,
where most of the interesting statements are still inferred.

**Two questions drove this work.** (1) Is run_79's 20-sample readout window
long enough for waveform-first tracking (`ntof_tracking/TRACK_PLAN_08_…`)?
(2) Is there dead pre-signal baseline at the start that could be traded for
tail by changing the DREAM latency?

**The answers.** The window is fine for detector A, marginal for C and D, and
**too short for B — but B is the only chamber that needs a longer window, and
B's field is missing its grading hardware, not slow gas.** Extending
`n_samples` costs rate, so **that decision should wait until B's degrador is
fixed or its removal is confirmed permanent** (§4.3d: it was pulled
deliberately, pre-campaign, for an unresolved hardware problem — this is not a
mystery any more, but it also isn't free to reverse). There is no dead pre-roll
to reclaim; the frame is if anything ~2 samples too early already, and the
missing leading edge costs measurable angle bias.

---

## 1. What was measured, and with what

All tools are new and live in `mx_june_wft/bench/` (they sit next to the June
bench harness because they compare bench and beam; the beam-side ones are
listed in `TRACK_PLAN_08` §4):

| tool | what it does |
|---|---|
| `framing_compare.py` | where the drift column sits in the readout frame, bench vs beam, per detector and plane; writes `framing.json` + per-cluster parquet |
| `beam_window_loss.py` | how much *charge* the window cuts, measured on beam waveforms with a beam-measured pulse template |
| `run_bench.py --crop START:N` | the window ablation: crops cached bench windows to emulate a shorter readout |
| `window_ablation.sh` | the window-length scan |
| `latency_scan.sh` | the frame-position scan at fixed window length |
| `make_bundle_variant.py` | derive a bundle (new v, new angle constants, new window) from another |
| `summarize_scans.py` | collect the result jsons into a table |

Column selection throughout is the production seed (per-plane relative
significance floor, 12 mm gap clustering, largest cluster, ≥5 strips) plus a
**clean micro-TPC column** cut: |rank corr(strip position, peak sample)| > 0.7,
5–25 strips, peak > 300 ADC, largest pulse per channel.

> **Trap, found the hard way.** At beam a channel can carry **several hits in
> one event** (pileup, and ringing after a saturated event). Indexing hits by
> channel therefore pulls a later secondary pulse in as the column's "deep
> edge" and inflates every deep-edge number — the first pass of this analysis
> reported 21–49 % truncation everywhere because of it. Deduplicate to the
> largest pulse per channel. Any beam analysis that maps channel → hit needs
> the same guard.

## 2. The window, measured  [measured]

### 2a. Where the column sits

`framing_compare.py`, one hits file per side. `onset`/`edge` are the earliest /
latest peak sample in a column, `ceiling` is the fraction of columns whose
deepest strip peaks in the **last** sample bin.

| | window | onset p5 | edge p50 | span p50 | at ceiling |
|---|---|---|---|---|---|
| bench det3 X / Y | 32 | 6.9 / 7.0 | 18.9 / 19.5 | 10.7 / 11.2 | 0.07 / 0.29 % |
| run_79 **A** (det3) X / Y | 20 | 1.4 / 1.4 | 12.7 / 12.9 | 9.3 / 9.5 | 4.5 / 3.3 % |
| run_79 **B** (det2) X / Y | 20 | 0.5 / 0.7 | 18.1 / 19.0 | 12.6 / 13.9 | **45.7 / 55.7 %** |
| run_79 **C** (det6) X / Y | 20 | 1.3 / 1.2 | 16.3 / 16.4 | 12.9 / 12.7 | 13.3 / 16.1 % |
| run_79 **D** (det7) X / Y | 20 | 1.4 / 1.3 | 14.7 / 14.8 | — / 11.2 | 8.6 / 11.6 % |

Column length, estimated from the columns with the **earliest onset** (maximum
room, so the estimate is not biased by the truncation it is trying to measure):

| | A | B | C | D |
|---|---|---|---|---|
| column length [samples] | 10.4–11.1 | **14.3–17.0** | 13.5–14.1 | 13.0–14.1 |
| ≈ [ns] | ~650 | **≥950** | ~830 | ~820 |
| still cut with maximum room | 0–3 % | **26–37 %** | 6–7 % | 5–21 % |

### 2b. How much charge is actually lost

`beam_window_loss.py`, 3 file-tags of `stat090_0000`, on the beam waveforms —
so the gas and the drift field are included by construction, no bench
extrapolation.

| | charge in last sample | clipped-pulse loss | still live at end |
|---|---|---|---|
| A x/y | 1.2 / 0.9 % | **2.3 / 2.5 %** | 7 / 7 % |
| B x/y | 5.0 / 4.9 % | **10.7 / 10.5 %** | **56 / 59 %** |
| C x/y | 2.6 / 2.1 % | **4.8 / 3.9 %** | 39 / 30 % |
| D x/y | 3.4 / 2.2 % | 23.5 / 13.4 %¹ | 38 / 27 % |

¹ contaminated — see §4.5; treat D as "between C and B" until the isochronous
population is cut.

`clipped-pulse loss` is charge missing from pulses that *did* peak in-window,
computed from a template measured on run_79 itself. **Charge arriving so late
that its pulse never peaks in-window is invisible to any in-window method and
is not included** — the ceiling fractions bound its incidence.

The stacked cluster-charge profiles (`% of in-window total` per sample) are the
clearest single piece of evidence:

```
A x  1.5 3.0 4.6 5.9 6.8 7.3 7.5 7.7 7.9 7.9 7.8 7.5 6.7 5.5 4.2 2.9 1.9 1.4 1.1 0.9   <- rises, plateaus, FALLS
C x  1.4 2.5 3.7 4.6 5.3 5.8 6.0 6.2 6.4 6.6 6.6 6.4 6.3 6.2 6.0 5.5 4.9 4.1 3.2 2.3   <- falls, but only just
D x  3.2 4.1 5.1 5.6 5.6 5.4 5.1 5.3 6.0 6.5 6.3 5.4 4.5 4.4 5.2 6.1 6.1 4.9 3.2 1.9   <- bimodal (§4.5)
B x  2.4 3.5 3.9 3.6 3.7 4.4 5.1 5.4 5.2 5.2 5.6 6.1 6.0 5.6 5.1 5.5 6.3 6.5 5.9 5.0   <- NEVER falls
```

**B's column is still fully live when the window closes.** A's is complete,
C's and D's are complete with the tail trimmed.

### 2c. What the truncation costs the reconstruction

Bench det3 windows cropped to the measured run_79 framing (start +6) and
scanned in length; identical 1 200 events at every point, production
configuration, `calib_bundle_lp2`, scored against M3.

| kept | within 5 mm | core σ [mm] | σθ X | σθ Y | compression \|tan\|>0.14 X / Y |
|---|---|---|---|---|---|
| 32 (full) | 94.80 | 0.473 | 1.06 | 1.11 | −0.08 / −0.10° |
| 26 | 94.80 | 0.478 | 1.10 | 1.18 | −0.23 / −0.33° |
| 24 | 94.80 | 0.471 | 1.10 | 1.18 | −0.23 / −0.33° |
| 22 | 94.63 | 0.469 | 1.10 | 1.21 | −0.23 / −0.35° |
| **20** | 94.80 | 0.465 | 1.12 | 1.22 | −0.26 / −0.41° |
| 18 | 94.80 | 0.466 | 1.10 | 1.26 | −0.31 / −0.48° |
| 16 | 94.63 | 0.486 | 1.09 | 1.32 | −0.41 / −0.65° |
| 14 | 94.46 | 0.468 | 1.18 | 1.36 | −0.55 / −0.82° |

**Position is untouched at every length** — within-5 mm and core σ are flat
from 32 down to 14 samples. The entire cost is in angles, and mostly as a
*bias* (compression of inclined tracks), which does not average away.

The bench and the beam have different column lengths (bench 10.7 samples at
333 V/cm in wet 95/5), so a bench point maps to a chamber by its **tail
margin**, not by its sample count:

| chamber | measured margin [smp] | ≈ bench point | expected σθ Y / compression Y |
|---|---|---|---|
| A | 6.2 | **n = 20** | 1.22° / −0.41° |
| D | 4.2 | n ≈ 18 | 1.26° / −0.48° |
| C | 2.6 | n ≈ 16–17 | 1.32° / −0.65° |
| B | ~0 | n ≤ 14, off the end of the scan | ≥1.36° / ≤−0.82° |

## 3. Latency: there is no room at the front  [measured]

**The pulse rise, measured on run_79 waveforms** (bright strips peaking
mid-window, peak-aligned median): 20 % of peak at **−2.0 samples**, 5 % at
**≈ −3**; the fall reaches baseline by **+7**. A single pulse therefore
occupies ~10–11 samples, peak−3 → peak+7.

**Where the first peak sits:** earliest peak in a column at sample **0.5–1.4
(p5)**, **~3 (p50)**. Subtract the 3-sample rise: the leading edge of the
prompt, near-mesh charge is at or outside the window start for a large fraction
of columns. `trunc_left` fires on 23 % of all hits.

**The stacked profiles confirm it**: sample 0 already carries 1.5–3.2 % of the
in-window charge — 19 % (A_x), 21 % (C_x), 37 % (B_x), 49 % (D_x) of that
column's plateau. The window opens on live signal, not on baseline.

**And the front samples are not dead weight for the fit.** In the ablation,
`n = 26` removes *only* pre-signal samples and keeps the entire tail, yet
compression goes from −0.08/−0.10° to −0.23/−0.33° and σθ Y from 1.11 to
1.18°. Losing the leading edge costs about as much as losing 6 samples of
tail — presumably because the rise is what constrains t0, the mesh arrival
time, and t0 trades against the slope.

So the frame is **~2 samples too early**, not too late. `latency` has nothing
to give; any extra tail has to come from `n_samples`.

**And moving the frame the other way is worth real angle quality, for free.**
`latency_scan.sh` scans the frame position at fixed n = 20 (full table in
`mx_june_wft/WINDOW_ABLATION_2026-07-30.md` §2d):

| framing | σθ X | σθ Y | compression X / Y | core σ [mm] |
|---|---|---|---|---|
| full 32-sample window | 1.06 | 1.11 | −0.08 / −0.10° | 0.473 |
| start 3 | 1.05 | 1.14 | −0.19 / −0.28° | 0.478 |
| **start 4 — the optimum** | **1.05** | 1.17 | **−0.18 / −0.18°** | 0.465 |
| start 5 | 1.07 | 1.20 | −0.21 / −0.21° | 0.465 |
| **start 6 = run_79 as recorded** | 1.12 | 1.22 | −0.26 / −0.41° | 0.465 |
| start 7 | 1.18 | 1.37 | −0.31 / −0.48° | 0.524 |
| start 8 | 1.29 | 1.45 | −0.40 / −0.68° | 0.546 |

**Putting the signal 2–3 samples later than run_79 does halves the compression
bias (−0.41 → −0.18/−0.28° on Y) and recovers σθ Y 1.22 → 1.14–1.17, at zero
readout cost** — while keeping *two fewer* tail samples, the same lesson as
the n = 26 point above. In DAQ terms: **latency 27 → 29–30**.

The trade is chamber-dependent, because moving later spends tail margin (§2c):
A has 6.2 samples and can afford it; C has 2.6 and cannot. The start 7/8 rows
show what running out of margin looks like — and note it damages **position**
too (core σ 0.465 → 0.546 mm), which nothing else in these scans did. That is
the regime B is already in.

**Untested corner, probably the real optimum**: start 4 *with* n = 24–26 —
raise the latency and the sample count together. Neither scan covers it
(`window_ablation.sh` fixes start at 6, `latency_scan.sh` fixes n at 20); two
runs would settle it.

Second-order but real: with zero pre-signal samples in the window, **per-event
baseline determination is impossible**. Both the analyzer and `wft` lean
entirely on pedestal runs and multi-event medians. Cutting the front further
would make that worse. And the frame position also moves the **gamma flash**,
which run_79's G&D delay deliberately places at sample ~5.

## 4. Detector B — the thing to solve before spending rate  [mostly inferred]

**The decision at stake.** Only B needs a window longer than ~24 samples.
Going 20 → 26 samples is ~30 % more payload per event and a corresponding hit
to trigger rate (**the exact cost is not measured here — get it from the
run_77/78 scans or a bench measurement before quoting it**). Spending that on
one chamber is only justified if B is *genuinely* a slow chamber. The evidence
below says B is more likely **broken**, in which case fixing it is free and
`n_samples` can stay at 20–24.

### 4.1 What B does that the others do not  [measured]

| | A | **B** | C | D |
|---|---|---|---|---|
| column length [smp] | 10.8 | **≥15.6** | 13.8 | 13.6 |
| stacked profile at sample 19 | 0.9 % | **5.0 %, no fall** | 2.3 % | 1.9 % |
| clipped-pulse charge loss | 2.3 % | **10.7 %** | 4.8 % | 23 %¹ |
| clean columns per hits file (x/y) | 446/518 | **223/253** | 572/578 | 1010/483 |
| median hits/plane/event (x/y) | 19/24 | **7/10** | 3/27 | 44/38 |
| **drift current** [µA] (700 V) | 0.18 | **0.00** | 0.18 | 0.18 |
| **resist current** [µA] | 2.1 | **5.2** | 0.01 | 1.0 |
| bench-calibrated v [µm/ns]² | 36.6 | **39.94 — the fastest** | 26.7 | 36.6 |
| bench drift gap² | 27.9 mm, dished | **30.5 mm, flat (the control)** | not mapped | not mapped |
| position in the gas chain | 1st | **2nd** | 3rd | 4th |

² **Caveat added 2026-07-30 (§4.3d):** degrador presence varied run-to-run at
the bench and was not tracked. These two B/det2 numbers assume that bench run
had the divider connected — not yet confirmed. Treat as provisional until the
bench degrador audit (§4.3d) is done.

### 4.2 The gas cannot explain it  [inferred, but the logic is tight]

The four chambers are **daisy-chained in series, A → B → C → D → exhaust**
(`DRIFT_WINDOW_HANDOFF.md` §0). Water content therefore increases
monotonically downstream, and water is what slows the drift. Observed column
lengths are **A 10.8 < D 13.6 ≈ C 13.8 < B ≥15.6** — B, the *second* chamber,
is slower than the two downstream of it. That ordering is impossible for a
shared series line.

Either the plumbing is not the documented A→B→C→D (**check this first, it is
free**), or B's slowness is not the gas.

### 4.3 The HV monitor is the strongest lead  [measured, interpretation inferred]

Both run_79 sub-runs, 3 500+ samples each, stable throughout:

```
drift  9:0 (A) 699.75 V  0.18 uA      9:1 (B) 699.75 V  0.00 uA
       9:2 (C) 699.75 V  0.18 uA      9:3 (D) 700.00 V  0.18 uA
resist 5:1 (A) 539.75 V  2.11 uA      5:2 (B) 539.50 V  5.17 uA
       5:3 (C) 524.50 V  0.01 uA      5:4 (D) 519.50 V  0.97 uA
```

B's drift channel sits at the right voltage and draws **exactly zero** current
where its three siblings all draw an identical 0.18 µA, and B's resistive
channel draws **the most current in the fleet** — 2.5× A and 500× C.

A drift electrode at the correct potential drawing no current at all is what an
**open/disconnected cathode circuit** looks like: the supply sees no load, the
electrode floats or charges only through the gas, and the field in the gap is
whatever leakage establishes — lower than nominal, plausibly non-uniform, and
therefore **slower drift**. That would explain B's long column with no gas
anomaly at all.

**Caveats, which is why this is inferred, not measured**: 0.18 µA repeated
identically on three channels smells like a monitor offset/quantisation floor
rather than a real current, and a channel reading 0.00 could equally be a
per-channel calibration difference in the readback. This must be checked at
the hardware level, not argued from the CSV.

### 4.3b The degrador hardware, and why B is the exception  [2026-07-30 addendum, reasoning inferred]

New information changes §4.3 from "an inferred, unexplained floating cathode"
to "a named, specific piece of missing hardware with a testable consequence."

**The hardware.** Each chamber's 30 mm drift gap is bounded at its perimeter by
a **3-step copper-ring degrador PCB** — a resistive field cage, standard
practice for keeping the boundary field parallel to the bulk rather than
fringing outward. Wiring: **top ring tied directly to the drift/mesh HV**,
then **~1 GΩ to the middle ring**, **~1 GΩ to the bottom ring**, **~1 GΩ from
the bottom ring to ground** — three equal steps that approximate the linear
potential gradient the bulk field already has. **Detector B is the fleet's one
exception: no connection between the drift copper and ground at all.**

**This explains the drift-current reading without needing a broken HV feed.**
An intact 3×~1 GΩ divider at ~700 V draws a DC bleeder current
I = V/R ≈ 700 V / 3 GΩ ≈ **0.23 µA** — which is what A, C and D actually read
(0.18 µA; implied R ≈ 3.9 GΩ, i.e. ~1.3 GΩ/ring, ordinary tolerance for
GΩ-class resistors). **B reads 0.00 µA because there is no resistor path to
draw current through, full stop.** That is the expected Ohm's-law signature of
"no divider," not evidence that B's own drift HV feed is disconnected — B's
drift channel still reports a correct, stable 699.75 V, which is what an
intact, regulated supply driving a purely capacitive (floating-guard) load
looks like. **So B's cathode is very likely correctly biased; what's missing
is the field *grading*, not the field itself.**

**Why that would produce exactly B's symptoms.** With the divider gone, the
middle and bottom rings float. A floating conductor at a field boundary does
not reproduce the graded solution the divider is there to enforce — it
distorts the local field (edge enhancement/de-enhancement) instead of gently
grading it. A properly graded chamber's fringe effect is Laplace-shaped and
short-range (e-folding ~10 mm for a 30 mm gap, symmetric, dead by ~45 mm —
exactly what `GAP_STUDY_2026-07-30.md`'s fringe cross-check measured). An
ungraded boundary has no reason to respect that decay length; the distorted
zone plausibly extends much further into the nominally active area, which
would show up exactly as observed: far fewer "clean," linear drift columns
(223/253 vs A's 446/518), roughly half the hits/plane, and a column that keeps
filling instead of falling because a large fraction of charge is following
distorted, longer paths rather than a clean vertical drift.

**Open tension this raised — resolved in §4.3d below.** The June bench cosmic
data used det2 (**= B**) as the flat, full-gap **control**
(`GAP_STUDY_2026-07-30.md`): 30.5 mm, edge fringe completely normal —
Laplace-shaped, symmetric, decayed to flat by ~45 mm. That is what a *working*
divider looks like, on this same physical chamber, so at face value it seemed
to contradict "B has no divider." It doesn't; see §4.3d.

### 4.3c What the HV monitor time series adds  [measured, 2026-07-30 evening]

The medians in §4.3 hide two things that the full 3 500-sample-per-sub-run time
series shows, and both support §4.3b.

**The 0.18 µA is a real, resolved, rock-stable current — not a monitor floor.**
The readback quantum is **0.02 µA** (B's own excursions take the values 0.00,
0.02, 0.06, 0.18, 0.24, 0.36, 0.38 — all multiples of 0.02). So 0.18 µA is
*nine* counts, comfortably resolved, and A, C and D each hold it at **exactly
one unique value across all 3 570 samples** of an hour-long sub-run. A current
that never moves by even one LSB over an hour is what a fixed resistor chain
looks like; a gas-borne or rate-dependent current would wander. Implied
R = 700 V / 0.18 µA ≈ **3.9 GΩ**, i.e. ~1.3 GΩ per step for a 3-step divider —
§4.3b's arithmetic, confirmed from the other side. **B reads 0.00, i.e. below
0.01 µA: there is no resistor chain.** This measurement was the one §4.3b asked
for, and it lands in favour of the missing-divider explanation.

**B's drift channel is also intermittently unstable, and the others are not.**
In 0.1–0.3 % of samples (10 events in `stat090_0000`, 4 in `stat090_0001`) B's
drift voltage **sags from 699.75 V to 683–684 V**, a ~16 V drop, and its
current simultaneously jumps to 0.15–0.38 µA:

```
              when V normal        when V sags (n=10 / 3570)
B drift I       0.000 uA             0.154 uA mean
```

A, C and D show **no** such excursions at all. Discrete, self-clearing
current bursts with a voltage sag are discharges — and a floating conductor
that has nowhere to bleed its accumulated charge is exactly the thing that
charges up and discharges periodically. This is independent evidence for the
floating-guard-ring picture, and it also means B's field is not merely
mis-graded but **time-varying**.

Worth noting for §4.3b's open tension: an intermittent connection (a divider
that came loose in the move) would look precisely like this — mostly open,
occasionally arcing over.

### 4.4 The run_55 cross-check: attempted, and it does NOT discriminate
  [measured — negative result, do not repeat]

The idea was that run_55 (2026-07-18, **B at 800 V**, 32-sample window) would
show whether B's column responds to drift voltage. **It does not work, for
three independent reasons. Do not spend more time on it.**

1. **The lever arm is too small.** In clean Ar/iso 90/10 the drift velocity is
   near the peak of v(E) between 600 and 800 V — Magboltz gives 40.5 / 42.6 /
   44.1 µm/ns at 600 / 700 / 800 V — so the expected column-length change over
   that range is only ~4–5 %, below the precision of a median span from a few
   tens of columns. The control confirms it: **A** shows no significant change
   either (span p50 6.7 at 600 V vs 6.9 at 700 V) and A is a healthy chamber.
   Only a *wet* chamber would respond strongly, because wet gas sits on the
   steep part of v(E) — so "B didn't respond" argues weakly against the water
   hypothesis and says nothing about the field.
2. **Analyzer mismatch** (the trap boxed below): run_55 predates the 2026-07-24
   analyzer, has no `significance` branch, and needs an absolute amplitude cut
   to be comparable at all.
3. **Statistics.** Even pooling **nine** sub-runs of run_55's resist scan
   (drift is common to all of them; only gain differs) B yields **9–14** clean
   gap-crossing columns per plane, because the busy/flash veto rejects 73–83 %
   of B's planes there even at a 200-strip threshold. That is not a measurement.

What it did give, at a common 300 ADC cut, is a consistency check: B's span is
**1.5–1.7× A's in both runs** — 11.6–14.7 samples against A's 6.7–7.8 — so the
anomaly is not specific to run_79 and is not created by the 20-sample window.

Everything below the box is retained because the reasoning is still correct;
only the conclusion "decisive" is withdrawn.

### 4.4a run_58 IS the decisive test, and B fails it  [measured, 2026-07-30 evening]

The run_55 cross-check (§4.4) failed for lack of lever arm and statistics.
**run_58 has both**: a 2-D drift × resist scan, **drift 700 → 200 V in 9
points**, 64 samples × 60 ns (3.84 µs) — a window explicitly "sized to contain
the full drift column across the whole drift sweep" — same Ar/iso 90/10 gas,
zero-suppress off, 76 sub-runs on EOS. It was run 2026-07-19/20, a week before
run_79, and its existing analysis (`ntof_july_analysis/run58_scan/`) was only
ever reported for detector A.

Processed on lxplus/HTCondor (76 jobs, 21 s each; see `lxplus/README.md`).
Median column length [60 ns samples] vs drift voltage, ± the median's standard
error, with the quality cut loosened to `span > 3` only so that B and D have
usable statistics:

```
det        200V        300V        400V        500V        600V        700V
Ax    27.1+-1.2   18.8+-1.0   15.2+-1.1   13.9+-1.2   12.3+-1.1   11.5+-1.2
Cx    33.6+-1.3   29.3+-0.8   21.1+-0.6   16.3+-0.6   13.2+-0.6   12.1+-0.5
Bx    16.4+-1.8   18.2+-1.8   18.4+-1.9   13.0+-2.1   17.4+-1.9   24.1+-1.3
Dx    19.3+-0.8   19.2+-0.8   19.1+-0.7   18.8+-0.8   19.3+-0.8   19.0+-0.8
```

**A and C do exactly what a drift field must do**: the column lengthens
monotonically as the field falls, by a factor **2.4 (A)** and **2.8 (C)** over
200–700 V, with the two selections (clean-ladder and loose) agreeing to ~10 %.

**B does not respond at all.** Flat within errors from 200 to 600 V — where A
more than doubles — and if anything *shorter* at low field, which is
unphysical. Under the strict ladder cut the ratio comes out **0.63×**, i.e.
backwards. There is no monotonic trend under either selection.

**This is the measurement §4.3/§4.3b needed.** B's drift column is
uncorrelated with B's drift supply, across a 3.5× range of set voltage, in a
window that cannot truncate it. Together with **zero bleeder current at all
nine voltages** (§4.4d) the conclusion is that **B's drift field is not set by
its power supply** — the supply reads its set point and delivers nothing to the
gap.

Two supporting observations from the same pass:

* **B is flooded**: 77 % of B's X planes exceed the 200-strip busy veto,
  against 29 % (A), 51 % (C), 45 % (D). And its clean-column yield is an order
  of magnitude below A's and C's at every drift point (6–18 per point against
  60–290). A chamber without a defined drift field sprays charge instead of
  forming columns — this is the same collapse in clean-column yield seen in
  run_79 (§4.1).
* **det D is a second, separate question.** D's column is **dead flat at
  19.0–19.5 samples at every drift voltage** under the loose cut (errors
  ±0.8), and erratic under the strict one. That is not the behaviour of a
  healthy chamber either, and it is a *different* pattern from B's. D also
  carries the wide isochronous population (§4.5) and the edge-localised
  anomaly (§4.4b). **Do not treat D as a clean chamber until this is
  understood** — but note A and C are unambiguously healthy, and they are
  enough for the tracking programme.

### 4.4d The bleeder current, measured across the whole drift sweep  [measured]

run_58's per-sub-run `hv_monitor.csv` (378 kB each) settles what §4.3c could
only suggest from a single voltage. Drift current vs drift set point:

| drift set | 200 | 300 | 400 | 500 | 600 | 700 V |
|---|---|---|---|---|---|---|
| A [µA] | 0.00 | 0.04 | 0.08 | 0.10 | 0.14 | 0.18 |
| C | 0.00 | 0.04 | 0.08 | 0.10 | 0.14 | 0.18 |
| D | 0.00 | 0.04 | 0.06 | 0.10 | 0.14 | 0.18 |
| **B** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |

A, C and D lie on a straight line through the origin. Fitted slopes give
**R = 2.98 GΩ (A), 2.98 (C), 2.7–2.8 (D)** — i.e. **3 × ~1 GΩ**, the 3-ring
degrador divider, measured rather than assumed. **B draws zero at every
voltage**, not merely below the least count at one. This also dates the
condition: B had no divider on **2026-07-19**, a week before run_79.

### 4.4b Edge localisation: B's anomaly is NOT at the edge  [measured]

The sharp prediction of floating guard rings is an **edge-localised** effect:
a missing perimeter divider distorts the boundary field, and for a 30 mm gap
that distortion should decay into the active area on the scale of the gap
(~30 mm; the bench fringe measurement found it dead by ~45 mm). Column length
binned by position across the plane, run_79, 3 file-tags, clean columns only:

```
                0-40    40-80   80-160  160-240 240-320  320-360  360-400 mm
A x span p50     8.3     9.3      9.7     9.3     10.1     9.8       -
A y              9.1     9.4      9.6     9.9     10.0     9.5      9.9
B x             13.1    13.9     13.6    12.0     13.5    14.3     12.7
B y             14.5    12.3     14.5    15.0     13.8    14.6     15.2
C x             11.6    12.4     12.0    13.1     14.0    13.7     12.5
C y             10.8    12.6     13.8    12.9     12.2    13.0     10.0
D x               -     12.2     11.8    11.4     13.9    12.8      6.6
D y              8.8    11.1     12.1    12.1     11.6    10.3      7.1
```

**B is long everywhere** — 12.0–15.2 samples in every bin, edge and centre
alike, against A's flat 8.3–10.1. There is no rim structure and no decay
inward. The whole drift volume is affected.

**The method is not blind to edge effects — it finds one on D.** D's X plane
collapses to 6.6 samples (and Y to 7.1) in the outermost 360–400 mm bin while
sitting at 11–14 everywhere else. That is where D's wide isochronous
population lives (§4.5), and it *is* edge-localised. So the flat B profile is
a real null, not a lack of sensitivity.

**What this does to §4.3b.** A missing *perimeter* divider, on its own, does
not predict a uniform slowdown across a 400 mm face — boundary effects decay
on the gap scale. Either the degrador stack does more than grade the rim, or
something else is also wrong. The question to put to the hardware people is
specific: **does the degrador divider define the drift cathode's own
potential, or only the guard rings?** If the cathode plane is fed through the
stack, a missing divider leaves the cathode floating at whatever leakage sets
— and that *would* slow the entire volume uniformly, draw no measurable
current, and discharge occasionally, which is the complete set of symptoms.

**A quantitative prediction that follows, if so** [inferred]: B's column is
~1.4× A's at the same gap, so v_B ≈ v_A / 1.4 ≈ 30 µm/ns. On the clean
Ar/iso 90/10 curve (42.6 µm/ns at 233 V/cm, 37.3 at 166 V/cm) that requires
E well below ~100 V/cm, i.e. an **effective cathode potential of roughly
300 V or less against the 700 V set point**. That is a testable number: if
someone can measure the actual potential on B's drift electrode, this
predicts it is less than half of nominal.

**run_55 (2026-07-18) is on the laptop**: same detectors, same gas, **drift
B = 800 V** (against run_79's 700 V), **32 samples** at latency 35 — so B's
column is *fully contained* there and can be measured without truncation bias.

* If B's column at 800 V is much shorter than at 700 V, scaling roughly as
  v(E) does, then B's drift field is real and B is simply a slow (wet) chamber
  → the gas/plumbing story needs rescuing, and the `n_samples` decision is
  live.
* **If B's column barely responds to a 100 V change in drift voltage, the field
  is not reaching the gap** → §4.3 is confirmed, B is broken, and no readout
  change is warranted.

Note run_55's A is at 600 V and B/C/D at 800 V, so it also gives a second point
on A and a cross-check of the whole v(E) picture — and because all four
chambers are in the *same* run, the within-run comparison is free of any
analyzer or threshold difference. **If B at 800 V is still slower than A at
600 V, that is decisive on its own.**

> **Trap when comparing runs (hit on the first attempt).** run_55 was processed
> with the **pre-2026-07-24 analyzer**: its `combined_hits` has no
> `significance`, `trunc_left` or `trunc_right` branch. Two consequences.
> (a) The production relative-significance floor is a no-op there, so run_55
> carries far more low-amplitude hits; the first pass tripped the
> busy/flash veto on 30–94 % of planes and left 1–25 clean columns per
> detector — unusable. (b) Any span or occupancy number compared *between*
> run_55 and run_79 is threshold-dependent and therefore meaningless without a
> common cut. `framing_compare.py --amp-min <ADC>` applies an absolute
> amplitude cut to both sides; the comparison in this section must be run
> with it (300 ADC was used) and with `--busy-strips 200`.
>
> First-pass numbers, for the record, at run_55's native threshold: A at 600 V
> gives span p50 12.6 samples against run_79 A at 700 V giving 9.4 — the right
> direction for a lower drift field, but *much* larger than Magboltz's ~5 %
> for 600 → 700 V in clean 90/10, so part of that difference is the threshold
> change, not the field. This is exactly why the common cut is needed.

### 4.4c Direct check: does B produce reasonable tracks at all?  [measured, 2026-07-30]

Asked directly: forget the HV monitor and the hypothesis, look at the actual
waveforms. Using the cached clean-column tables from `framing_compare.py`
(`ladder > 0.7`, 5–25 strips, amp > 300 ADC, largest pulse per channel — the
same "real inclined column, not the ringing-block artifact" cut used
throughout §2/§4) on run_79 `stat090_0000`, file 0:

**Yes — B produces genuinely real, physically sensible tracks. Far fewer of
them, and stretched over more of the window, but not garbage.**

| | A | B | C | D |
|---|---|---|---|---|
| both-planes-clean rate (% of events) | 4.9 | **1.0** | 8.4 | 2.9 |
| clean-X ladder (rank corr), median | 0.983 | **0.900** | 0.993 | – |
| clean-Y ladder, median | 0.976 | **0.908** | 0.983 | – |
| X–Y span correlation, both-clean events | 0.540 | **0.617** | 0.572 | 0.688 |
| span median, both-clean [smp] | 9.6 | **14.0** | 13.1 | 12.3 |
| both-clean events landing at the window ceiling | 0.0 % | **3.1 %** | 0.2 % | 0.0 % |

Reading it plane by plane:

* **The rate is the real deficit**: B's both-planes-clean fraction is 1.0 % of
  events against A's 4.9 % and C's 8.4 % — roughly a fifth to an eighth as
  many usable gap-crossing columns, matching §4.1's clean-column counts.
* **What survives is not marginal.** B's ladder correlation (0.90/0.91) sits
  well clear of the 0.7 cut — comparable to, if somewhat below, A and C's
  ~0.98–0.99 — and its X–Y span correlation (0.617) is at least as good as
  A's (0.540) or C's (0.572): when B does produce a clean column, its two
  independently-fit planes agree with each other about as well as they do in
  the healthy chambers. That is not the signature of noise passing a cut by
  chance.
* **Visual event displays confirm it.** Three of B's cleanest events (5947,
  7131, 5837 — plotted in `figures/detB_check/detB_track_check.png` alongside
  a representative A event) show smooth, monotonic strip-position-vs-sample
  ladders in both planes, structurally indistinguishable from A's example —
  just stretched over most of the 20-sample window (starts ~3–4, ends ~19–20)
  instead of A's tighter ~4–14.
* **The span ratio lands almost exactly on §4.4b's prediction.** 14.0/9.6 =
  1.46, matching "B's column is ~1.4× A's" independently.
* **Window truncation mostly kills candidacy rather than lingering as a
  truncated-but-clean track**: only 3.1 % of B's *clean* events sit at the
  sample-19/20 ceiling, far below the 46–56 % "at ceiling" fraction over the
  *whole* population reported in §2a. Most of B's truncated columns fail the
  clean cut outright (motion cut off mid-ladder breaks the rank correlation)
  rather than surviving as a shortened-but-valid track.
* **Spatial distribution is unremarkable** — B's clean-track positions span
  the full 0–380 mm active width with no single-bin spike, i.e. no sign these
  are an artifact concentrated on one hot region/channel block.

**This is a small, useful update to §4.4b/§4.3b's framing.** The claim that "B
is broken" needs qualifying: B's cathode is producing real, well-formed drift
columns — just at much reduced efficiency and much longer transit time,
uniformly across the face. That is exactly what §4.4b's refined hypothesis
predicts (a globally reduced field from a floating *cathode*, not merely
floating guard rings): a weaker field lengthens the drift path for every
track, and — via extra transverse diffusion over the longer transit — plausibly
smears enough columns out of the "clean, 5–25 strip, |corr|>0.7" band to
explain the collapsed clean-rate too, without needing a separate efficiency
mechanism. **Net: B is a real, working chamber currently running at a
much-reduced, apparently uniform field — not a dead one.**

### 4.3d Resolved: the removal was deliberate, and it isn't a fixed fleet property  [confirmed, user, 2026-07-30 evening]

**§4.3b's tension is resolved, not by data but by history: B's degrador was
physically removed on purpose, just before the nTOF campaign — there was a
problem with it and no time to fix or reinstall it before shipping.** That is
why the June bench data (det2 as the flat, full-gap, Laplace-fringed control)
and the July beam data (B as the outlier) don't agree: they're not describing
the same hardware state. Nothing was quietly lost in the move; it was taken
out deliberately, under time pressure, and this is the first record of that
decision.

**This also plausibly explains §4.3c's intermittent discharges.** "Removed"
most likely means the *resistor* divider was disconnected, not that the
copper ring PCBs themselves were pulled out of the chamber. Bare rings with no
bleeder path are still there, still floating, and — per §4.3b — floating
conductors near a strong field are exactly what charges up and periodically
arcs over when nothing drains them. That is a clean read of the 683–684 V
sags + 0.15–0.38 µA current bursts seen on 0.1–0.3 % of samples, and on no
other chamber: not a loose/intermittent connection, but a permanently-floating
ring occasionally discharging.

**New complication: degrador presence is not a fixed fleet property.** At the
cosmic bench, chambers sometimes ran *with* the ring divider connected and
sometimes without — not only B, and not tracked systematically. So any bench
number in §4.1's table that leans on a chamber's field being properly graded —
**det2/B's 30.5 mm flat "control" status in the gap study, and its 39.94 µm/ns
"fastest" bench v** chief among them — is only as trustworthy as that specific
run's actual degrador state, which is not yet known.

**Action item, not yet done:** audit `hv_monitor.csv` drift-channel current
per bench run/detector using the §4.3b/§4.3c signature (rock-stable ~0.15–0.25
µA at ~700 V = connected, ~0.00 = not) to build a per-run degrador-connected
table across the bench dataset, starting with whichever run(s)
`GAP_STUDY_2026-07-30.md` drew its det2 control from. This is nontrivial
because the bench HV channel↔detector mapping is **not fixed run to run** — it
was hand-calibrated per run in `mx_june_cosmic_qa/30_fleet_gas_survey.py`
(e.g. its "0:7 = top drift" comment, which flips depending on which two
chambers and which physical stacking order a given bench run used).

**Gas plumbing, confirmed:** at nTOF (this deployment) the four chambers share
one line, **A → B → C → D → exhaust**, as already used in §4.2. The bench
paired only two chambers at a time (det2_det3, det3_det4, det6_det7, ...) and
is not known to share this same topology — don't project the nTOF chain onto
bench runs.

**Net effect on the open decision:** B is very likely not a broken chamber in
the sense of "no field" — the mesh is almost certainly correctly biased. It is
missing its field-grading hardware, on purpose, for now, and that is
sufficient on its own to explain the long/never-falling column, the collapsed
clean-column yield, and the intermittent discharges, with no gas or plumbing
anomaly required. Fixing it is a known, scoped hardware task (reinstall or
repair the divider), not a mystery — but it is *not* free in the way §4's
framing assumed ("if B is broken, fixing it is free"): whatever problem forced
its removal has to be solved first.

### 4.5 Other hypotheses, ranked, with their tests

| # | hypothesis | test | cost |
|---|---|---|---|
| 1 | ~~Open/floating cathode on B~~ → **CONFIRMED: no degrador divider, removed on purpose pre-campaign** (§4.3b–d) | done — user-confirmed hardware history; remaining test is the bench per-run degrador audit (§4.3d) | free (audit) / hardware (fix) |
| 2 | Plumbing is not A→B→C→D | **not needed** — nTOF chain confirmed A→B→C→D (§4.3d); B's anomaly is explained by #1 without touching the gas story | closed |
| 3 | **Per-FEU frame offset** — B's onset p5 is 0.5–0.7 against A's 1.4, i.e. B's signal sits ~1 sample earlier, which alone pushes its tail out | the **gamma flash is a common fiducial**: it reaches all four chambers simultaneously, so compare the flash peak sample across FEUs on run_79 flash events. Decisive and cheap | ~1 h |
| 4 | **B's response is intrinsically more dispersive** — its bench bundle has `sigma_s` 71.3 ns against det3's 12.1, so its shared copies are far more smeared; a longer per-strip response inflates the apparent column | compare the per-detector single-pulse template on run_79 (`beam_window_loss.py` already measures one per plane — save and plot them). **Partly excluded already**: B's span excess is ~3–4 samples at *every* cluster width (5 → 12 strips), so it is not a pure width/spreading effect | ~1 h |
| 5 | **Different particle population in arm B** — B has the lowest occupancy (7/10 hits per plane) and the fewest clean columns, so its surviving sample may not be comparable | event displays; per-arm trigger rate; cross-check against the arm assignment in `ntof_dream_merge` | ~2 h |
| 6 | **Charging-up of the resistive layer at beam rate** (B draws 5.2 µA) | column length vs time within run_79 and vs beam intensity | ~1 h |

Related and worth resolving in the same pass: **det D's X plane is 55 %
wide isochronous deposits** — median 17 strips all peaking within 0.3 samples,
which are not gap-crossing columns and which contaminate D's numbers above
(the bimodal profile in §2b). The beam seeder for `wft` will need a minimum
time-span requirement, not just a strip count.

### 4.6 History says B has always been the gate

`DRIFT_WINDOW_ANALYSIS.md` (2026-07-19) already found B's deep edge piling at
the sample-31 ceiling in the 32-sample cosmic run, could not decide between
noise and real late drift, and named B as **the** knob that separated "trim 3
samples" from "trim 8". Eleven days later, with a ladder-quality cut that
rejects the ringing artifact that analysis worried about, **B still rides the
ceiling** — so it is drift-like, not the artifact. This is a chronic,
unresolved condition, not a run_79 accident.

## 5. Recommendations

1. **Do not change `n_samples` yet.** Settle §4 first: run_55 (§4.4), the gas
   line, the flash fiducial. If B is broken, 20–24 samples is the right
   readout and no rate is spent.
2. **If B turns out to be genuinely slow**, the sizing is: A needs n ≈ 23,
   C/D n ≈ 26, B n ≈ 29–30 for the complete pulse footprint (column + 3-sample
   rise + 7-sample fall + baseline). The reconstruction saturates earlier than
   that — bench parity is reached by n ≈ 24–26 — so **n = 24 buys almost all of
   the available angle quality for A/C/D at 20 % more payload**, and the last
   4–6 samples are B's alone.
3. **A latency change is free angle quality — take it.** `latency 27 → 29–30`
   halves the compression bias on A at no readout cost (§3). Do it together
   with any `n_samples` increase, and remember it also moves the gamma flash,
   which run_79's G&D delay places at sample ~5. Do NOT apply it alone to a
   chamber with less than ~4 samples of tail margin (C, D, B).
4. ~~**Read `latency_scan.sh` before touching latency.**~~ Done — see §3. The frame looks ~2
   samples too early; if the scan confirms it, that is free angle-bias recovery
   at fixed readout cost. Remember it also moves the gamma flash.
4. **Quote the rate cost from a measurement**, not from the payload ratio.
5. Everything in §2–3 feeds `ntof_tracking/TRACK_PLAN_08_waveform_first_run79.md`
   §4; the migration table in `RECONSTRUCTION_BASIS.md` should record the
   window ablation once §4 is closed.

## 6. Reproduce

```bash
cd /home/dylan/PycharmProjects/nTof_x17
V=.venv/bin/python
# where the column sits, bench vs beam (writes framing.json + parquet)
$V mx_june_wft/bench/framing_compare.py
$V mx_june_wft/bench/framing_compare.py --beam-run run_55 \
     --beam-subrun scintd_r530_dr800dA600_c00_006 --out mx_june_wft/bench/framing_run55.json
# how much charge the window cuts, on beam waveforms
$V mx_june_wft/bench/beam_window_loss.py --dets A,B,C,D --tags 000,001,002
# what it costs the fit (bench, M3 truth)
bash mx_june_wft/bench/window_ablation.sh 3 1200
bash mx_june_wft/bench/latency_scan.sh    2 1200
$V mx_june_wft/bench/summarize_scans.py w --ref full32
# HV
python - <<'PY'
import pandas as pd, glob
for f in sorted(glob.glob('/media/dylan/data/x17/beam_july/runs/*/*/hv_monitor.csv')):
    d = pd.read_csv(f)
    print(f, {c: round(d[c].median(), 3) for c in d.columns if 'imon' in c and c[:2] in ('9:', '5:')})
PY
```

Data: DREAM `/media/dylan/data/x17/beam_july/runs/run_79/` (both real sub-runs,
`decoded_root` present, 12 GB each) and `.../run_55/`. Bench caches and bundles
under `/home/dylan/x17/cosmic_bench/Analysis/…/wft/`.
