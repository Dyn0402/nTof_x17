# Slimming n_TOF onto the DREAM triggers — feasibility

**Verdict: it works, it is cheap, it is built, and it is proven on condor.**
Two passes over each n_TOF run with the DREAM↔n_TOF clock fitted from scratch in
between, then every scintillator hit within **±150 ns** of the fully corrected
prediction — plus the same width at +100 µs as an accidental control — written
beside the DREAM sub-run on EOS. About **33 MB per DREAM sub-run**, **~8–9 GB**
for the whole campaign: 0.3 % of the DREAM data it accompanies, 0.1 % of the
processed n_TOF it replaces.

It reproduces, **from the slim alone**, both the published match (95.89 % at
0.046 % accidental, against 95.84 % / 0.049 %) and the published liquid same-arm
coincidence — on two disjoint hours, each against its own published numbers.

Measured on DREAM run_79 against n_TOF 224572. §§1–3 are the fallback case (what
a slim would need if it had to run *before* the clock fit); §4 is the design that
was actually built; §§6–7 audit the source and say what can run today; §8 is the
pipeline and its validation. The calibration authority remains
`../ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`; nothing here changes it.

**Read this first if you are picking the work up:** §8 (what exists) and §9
(what is left).

---

## 0a. Update, 2026-08-09 — three things this document got wrong

**The campaign has since run. Its report is
[`SLIM_CAMPAIGN_2026-08-09.md`](SLIM_CAMPAIGN_2026-08-09.md), which supersedes
the sizing and coverage numbers below: 119 of 202 segments slimmed, covering 116
of 170 DREAM sub-runs, at a ±1 µs window and 7.1 GB. The remaining 54 sub-runs
turned out to contain a real but ~6 µs wide association at −0.982 ms, which
cannot support a ±25 ns match.**

### The original two


Read this before the rest. Both were found by running the pipeline on a pair
other than the one it was developed on, which is the only reason they surfaced.

### The clock fit did not generalise, and passed validation by luck

`fit_global` started from a hard-coded seed (`K = 1.1e-4`, `T0 = -250 ns`) and
selects candidates within ±250 ns of wherever it is currently looking. That is
right for run_79 (fitted `T0 = -252.6 ns`) and wrong for run_77 (`T0 = +109.5`).
Result on the first pair the campaign touched: **7 of 9 segments failed**, and
the 2 that survived started from 312 candidates against a hard floor of 200 —
the difference between success and failure was how much of the trigger-latency
tail happened to fall inside the seed window.

Fixed by `clockfit.bootstrap`: histogram every candidate within ±50 µs, take the
peak, require S/N ≥ 6 over the accidental floor beside it, and only then iterate.
On the reference pair it lands ±10 ns from the old seed at S/N ~1850 and every
published constant reproduces to 4 decimals; run_570 went 0/3 → 3/3, run_571
1/9 → 7/11 with the remaining two being 2–3 minute slivers that genuinely cannot
be fitted.

**The general lesson, which is the one worth keeping:** a fit that is seeded near
its answer validates beautifully and tells you nothing about whether it works.
The validation set was one pair, and one pair cannot exercise a per-pair constant.

### ±150 ns was too narrow, and the reason was mis-measured

The window check inherited from `validate.py` asks whether the kept `dt`
distribution is still *rising* at the window edge. A coincidence peak WIDER than
the window falls away slowly and passes it: it returned 0.94 ("flat, window is
wide enough") on the reference while 23 % of the plastic yield was being cut.

Measured properly (`slim_study/pss_tail_probe.py`, ±10 µs on 150 bunches), the
plastic tail is real, one-sided late by 22×, smooth with no discrete echoes, and
extends to microseconds. **The window is now ±1 µs**, which captures 93 % of the
PSS excess within ±2 µs at 2.24× the hits — ~72 MB/segment, ~14 GB for the
campaign instead of ~6 GB.

The liquids do **not** need it: their apparent tail is symmetric (3,701 early
against 2,224 late), i.e. subtraction noise. An earlier reading of the integral
scan called it real; that was wrong.

### What now guards against a repeat

`slim_pipeline/clock_qa.py` (13 absolute checks per segment),
`dashboard/make_clock_dashboard.py` (robust-z against the fleet — the layer that
catches a segment which passes every absolute check but sits 380 ns from its
peers) and `tests/test_clock_qa.py` (19 injected defects, asserting each check
fires). See `slim_pipeline/README.md`.

### The third: coverage was assumed, not measured

This document's §7 counted how much beam time n_TOF had *processed*. It did not
occur to me to ask how much of that would actually yield a coincidence, and the
answer is 77 %, not 100 %. Availability of input is not coverage of output, and
only running the thing revealed the difference.

Numbers below that predate this section — the ±150 ns window, ~33 MB per sub-run,
~8–9 GB total — are superseded by the ±1 µs figures in
[`SLIM_CAMPAIGN_2026-08-09.md`](SLIM_CAMPAIGN_2026-08-09.md).

---

## 0. Headline numbers

| | |
|---|---|
| accept window, after the per-bunch clock fit | **±25 ns** (confirmed, unchanged) |
| **slim window, on the fully-fitted clock** | **±150 ns** + a +100 µs control |
| slim size | **~33 MB per DREAM sub-run**, ~300 B/trigger, 8.5 hits/trigger |
| … against DREAM `decoded_root` (12 GB/sub-run) | **~0.28 %** |
| … against the n_TOF source it replaces (30 GB) | **~0.11 %** |
| whole campaign (282 beam sub-runs) | **~8–9 GB** |
| cost to produce | **~10 min per DREAM sub-run**, one core, 3.2 GB RSS |
| processable today | **133 h of 253 h (52 %)** — the rest waits on n_TOF |

**Built, validated and run on condor, 2026-08-08.** `slim_pipeline/`; three
segments of run_79 × 224572 produced on a worker in 28 min. §8.

---

## 1. Before the clock is fitted, ±50 and ±100 ns are too tight

The ±25 ns accept window is real, but it **only exists after the per-bunch
clock fit** (`δa_b`, `δk_b`), and that fit cannot run before the slim — it is
fitted on the matched sample, which is what the slim is supposed to produce.
So the slim has to survive on the global map (`K`, `T0`, per-arm offsets)
alone, and there the DREAM timestamp clock's ~1 ppm bunch-to-bunch drift is
still in the residual, growing in proportion to time since flash.

Measured envelope of the arm-corrected residual under the global map alone
(`slim_study/window_envelope.py`, 213 420 triggers):

| t since flash | n | 68 % hw | 99 % | 99.9 % | ±50 ns | ±100 ns | ±200 ns | ±500 ns |
|---|---|---|---|---|---|---|---|---|
| 1–3 ms | 17 481 | 9.4 | 27.9 | 1485 | 99.71 % | 99.71 % | 99.72 % | 99.76 % |
| 3–10 ms | 46 202 | 11.2 | 33.2 | 1135 | 99.73 % | 99.75 % | 99.77 % | 99.81 % |
| 10–20 ms | 53 524 | 16.0 | 47.6 | 63.4 | 99.30 % | 99.96 % | 99.96 % | 99.97 % |
| 20–40 ms | 51 423 | 25.2 | 83.7 | 105.6 | 89.37 % | 99.84 % | 99.99 % | 99.99 % |
| **> 40 ms** | 35 783 | **44.2** | **150.9** | **185.7** | **72.02 %** | **91.04 %** | **99.95 %** | **99.99 %** |
| ALL | 205 112 | 18.7 | 115.9 | 200.6 | 92.18 % | 98.30 % | 99.90 % | 99.92 % |

(The 1–1.8 µs entries in the 99.9 % column at early times are triggers with no
real partner, where "nearest candidate" picks up an accidental — not a window
failure. The plateau in the coverage columns is the honest read.)

At > 40 ms a ±100 ns window throws away 9 % of the signal. That is where 17 %
of the triggers live.

## 2. Closure: on the provisional clock, ±250 ns is free

Slim the candidate list at ±W on the **arm-agnostic** global prediction, then
run the real downstream chain on what survives — per-bunch fit (cross-validated,
`core = 200 ns`) and the ±25 ns accept — and compare to the unslimmed result
(`slim_study/slim_closure.py`):

| slim W | efficiency @ ±25 ns | accidental | purity | Δ vs no slim |
|---|---|---|---|---|
| ±100 ns | 94.2283 % | 0.0487 % | 99.9483 % | **−1.612 pts** |
| ±150 ns | 95.6260 % | 0.0487 % | 99.9490 % | −0.215 pts |
| ±200 ns | 95.8312 % | 0.0487 % | 99.9491 % | −0.009 pts |
| **±250 ns** | **95.8406 %** | **0.0487 %** | **99.9492 %** | **0.000** |
| ±300 ns | 95.8406 % | 0.0487 % | 99.9492 % | 0.000 |
| ±500 ns | 95.8406 % | 0.0487 % | 99.9492 % | 0.000 |
| no slim | 95.8406 % | 0.0487 % | 99.9492 % | — |

±250 ns reproduces the unslimmed number to four decimal places and reproduces
the published 95.84 % / 0.049 %. **The accidental rate is identical at every
window** — the slim costs no purity, only (at small W) efficiency. ±250 ns is
where the curve has been flat for 50 ns; it is the right place to sit, and it
leaves the `core = 200 ns` the per-bunch fit uses entirely inside the file.

The arm-agnostic centring matters: the per-arm offsets span −16.8 to +7.5 ns and
the slim cannot know which arm fired, so that 24 ns spread is absorbed by the
window rather than applied. Negligible against 250 ns.

## 3. What it costs, in hits and in bytes (provisional clock)

Hits surviving the window, run 224572, all twelve scintillator trees, centred
on the **global map only** (`slim_study/window_yield.py`, output in
`window_yield_{narrow,wide}.json`). §4 has the same scan on the fitted clock:

| window | hits kept | fraction of run | 1/N | per trigger |
|---|---|---|---|---|
| ±25 ns | 610 418 | 8.35e−4 | 1/1197 | 2.86 |
| ±50 ns | 892 806 | 1.22e−3 | 1/818 | 4.18 |
| ±100 ns | 1 304 811 | 1.79e−3 | 1/560 | 6.11 |
| **±250 ns** | **1 941 289** | **2.66e−3** | **1/376** | **9.10** |
| ±1 µs | 3 541 363 | 4.85e−3 | 1/206 | 16.59 |
| ±2 µs | 5 216 936 | 7.14e−3 | 1/140 | 24.44 |
| ±20 µs | 32 900 869 | 4.50e−2 | 1/22 | 154.16 |

Per family at ±250 ns: **~2.9 wall hits, ~5.9 plastic, ~0.32 liquid** per
trigger. That is the firing segment's two bar ends plus neighbours, the plastic
partner, and the liquid coincidence rate the physics is after.

Two more reductions sit on top and are already in the "fraction of run" column:
the DREAM pair covers 2061 of 3018 bunches, i.e. **68.5 %** of the run's hits
are even in play.

### Bytes on disk, measured not estimated

`slim_study/slim_prototype.py` writes a real ROOT file for 200 bunches
(20 934 triggers, 189 594 hits) in two layouts:

| layout | contents | B/hit | B/trigger | run_79 pair |
|---|---|---|---|---|
| **full** | `eventId, det, detn, tof (f8), dt_ns (f4), amp, area_0, amp_0, fwhm, risetime, chi2, satuflag, pileup1, pulseshape` | 32.7 | **296** | **63 MB** |
| lean | `dt_ns` as **int16** instead of `tof` (`tof` is quantised to 1 ns — max quantisation residual 0.500 ns, i.e. exact), `detn` uint8, flags bit-packed, `risetime`/`pileup1` dropped | 20.7 | 187 | 40 MB |

The full layout is worth its 1.6× — it keeps the absolute `tof` (so the file
stands alone if the calibration is ever re-derived) and every branch that
carries per-hit information. Recommend **full**.

For scale, the source is ~41 B/hit compressed (30 GB / 730.7 M hits), so the
slim's per-hit cost is comparable; all the saving is in *which* hits.

### Relative to DREAM

| | size | slim as % |
|---|---|---|
| DREAM run_79 pair, `decoded_root` | 24 GB | 0.26 % |
| DREAM run_79 pair, `combined_hits_root` | 2.4 GB | 2.6 % |
| n_TOF source covering run_79 (91 partials, 224572-224579) | ~190 GB | 0.033 % |

(±250 ns single-stage, i.e. the fallback case of §§1-3. The two-stage product
of §4 at ±150 ns is ~33 MB per DREAM sub-run.)

**Campaign:** 280 `stat*` beam sub-runs exist on EOS across runs 1–156; the DAQ
alone currently holds 1.6 TB in 106 of them (~15 GB/sub-run), so the campaign is
a few TB of DREAM and, reprocessed, order 10 TB of n_TOF. At run_79's ~105 k
triggers per sub-run that is ~29 M triggers → **~8.7 GB for the entire campaign
single-stage, ~7 GB with the §4 scheme.** It fits on a laptop.

## 4. The clock is fully fixed before the final cut

**The final slim is cut on the completely calibrated clock**, global map *and*
per-bunch (`δa_b`, `δk_b`). §§1–3 measured what a slim would need if it had to
run *before* the fit; it does not have to, so those sections are the fallback
case and this one is the design.

```
  per (DREAM sub-run x n_TOF run) SEGMENT, on lxplus/condor:

  [0] join      bunch_join: eventId -> BunchNumber, t_since_flash, is_flash
        │       PKUP + index only; beam record, independent of the PSA
        ▼
  [1] pass 1    wall top/bottom offsets, then the N1081B SINGLES emulation
        │       -> candidate list          reads WAL + PSS (78 % of the hits)
        ▼
  [2] fit       K, T0, per-arm offsets; then per-bunch (da_b, dk_b)
        │       CLOCK NOW FULLY FIXED -- residual flat at 6 ns.  Seconds.
        ▼
  [3] pass 2    every scintillator hit within +-150 ns of the CORRECTED
        │       prediction, plus the same width at +100 us as the accidental
        ▼       control            reads all 12 trees
  [4] write     <run>/<subrun>/ntof_hits/ on EOS
```

**No hit is ever cut on a provisional clock.** An earlier draft of this document
proposed a ±2 µs intermediate buffer written during pass 1 so pass 2 could read
170 MB instead of re-reading the source. It was only ever an I/O optimisation,
never a compromise on the calibration, and it has been **dropped**: two straight
passes are simpler, easier to test, and the measured cost is 385 s per sub-run
on a single core. If EOS I/O turns out to hurt at campaign scale, the buffer
slots into pass 1 without changing anything downstream.

### The final window, on the fixed clock

Once the per-bunch fit is applied the residual is flat at 6 ns over the whole
80 ms, so the final window is a physics choice, not a calibration one. Measured
(`window_yield.py --perbunch`), against the same run:

| final window | hits | 1/N of run | per trigger | +100 µs control | signal+control |
|---|---|---|---|---|---|
| ±25 ns (the accept) | 738 660 | 1/989 | 3.46 | 39 115 (5.3 %) | 25 MB |
| **±100 ns** | **1 335 963** | **1/547** | **6.26** | **157 151 (11.8 %)** | **49 MB** |
| ±250 ns | 1 943 404 | 1/376 | 9.11 | 392 211 (20.2 %) | 76 MB |

**±150 ns — and a retraction.** This section first said ±100 ns, then ±250 on
the grounds that ±100 clipped the liquid diagonal. **The ±250 claim was wrong.**
The raw `sig` number does move with the window, but so does the control, and the
background-subtracted signal does not: 0.135/0.119/0.012/0.075 at ±100 against
0.136/0.119/0.013/0.075 at ±250, i.e. identical to 0.001 per event. What ±250
recovered was accidental floor inside `liq_coincidence`'s deliberately wide
±100 ns integration window, which cancels in the subtraction.

Measured containment of the background-subtracted excess, run_79/stat090_0000:

| | ±25 ns | ±50 | ±100 | ±150 | ±250 |
|---|---|---|---|---|---|
| WAL | 94.1 % | 98.9 % | 99.1 % | 99.6 % | 100 % |
| LIQ | 55.3 % | 76.3 % | 88.1 % | 92.4 % | 100 % |
| PSS | 22.9 % | 36.0 % | 65.3 % | 80.5 % | 100 % |

So the coincidence really is narrow, as expected from the timing budget: n_TOF
internal alignment is ≤ 1 ns liquid-to-wall and a few ns per channel, and the
DREAM match is 6 ns after the per-bunch fit. The wall — which *is* the trigger —
is 94 % contained within the ±25 ns accept.

±150 ns is chosen for two reasons that are **not** the coincidence width: it
holds 92 % of the liquid excess rather than 88 % (the late tail a peak-centred
metric never integrates but a total yield would), and it leaves ~6× the accept
window of flat sideband so a segment with a bad clock is visible in the file.
For the second, note the primary alarm is not the window at all: `qa.json`
carries the efficiency and `events.residual_ns` the nearest-candidate residual
out to the 400 ns fit search, so a shifted clock shows up whatever the slim
width. It is one constant, `config.SLIM_NS`, and re-slimming a sub-run is
~6 minutes.

**New and unexplained: the plastics have a long LATE tail.** sig/ctl per 25 ns
bin is flat at 1.03 for dt < −150 ns but still **4.2** at dt = +238 ns. One-sided,
so not a control rate-gradient artifact. Most likely the PSS shape fitting
(101 ns templates) splitting long pulses, or afterpulsing. Truncated at any
sensible window; does not touch the trigger match or the liquid physics. Worth
one look from the source.

Note the per-bunch correction is doing real work here: at ±25 ns it keeps
738 660 hits against 610 418 on the global map, **21 % more**, because the
window is finally centred. Above ~150 ns the two converge — that is the
accidental floor, which no centring changes.

### Carry the accidental control, or you can never re-measure the background

The 0.049 % accidental rate in `DREAM_NTOF_CALIBRATION.md` is **not** measured
from a local sideband. It is the identical match with the DREAM time shifted by
**+100 µs** (`study_common.SHIFT_NS`), because the local rate structure varies
too much across the 80 ms for a neighbouring-window sideband to be a fair
control. A slim that keeps only ±W around the prediction can therefore never
reproduce its own background.

So write a second window of the same half-width at `t_pred + 100 µs`, with a
`is_control` flag. It costs **11.8 %** more file at ±100 ns (measured, not
modelled: `window_yield.py --perbunch --shift-ns 100000`) and it makes the slim
self-contained. In-window S/B at ±100 ns is 7.5.

### What `measure_tb_offsets` is (it is not a background estimate)

It is part of the **trigger emulation**, not the background. Each wall bar
segment is read out at both ends; the N1081B discriminates the analogue *sum*
of the two ends, so `fast_singles` has to pair a top hit with a bottom hit, and
that pairing needs the fixed time offset between the two ends of each segment.
`measure_tb_offsets` measures it in situ as the modal `t_top − t_bottom` per
segment, using hits late in the bunch where the rate is low.

It matters because the offsets are **per processing, not per cabling**: on the
official file they are ±32–39 ns with a −77.5 ns outlier, on v12 they are within
±5.5 ns. The structure was the old flash-finder's leading-edge timing. Reusing a
stored table on a reprocessed file pairs the bar ends around a 38 ns offset that
is no longer there and loses most genuine pairs. It runs in seconds.

It runs in step [3], on the buffer — genuine bar-end pairs are within
`TB_MAX_NS` of each other so both ends are inside the same ±2 µs window. That
is an argument, not a measurement; see §7.

### What to put in the file

Per hit: `eventId`, `det` (tree code), `detn`, `tof`, `dt_ns` (to the final
per-bunch-corrected prediction), `amp`, `amp_0`, `area_0`, `fwhm`, `risetime`,
`chi2`, `satuflag`, `pileup1`, `pulseshape`, `is_control`.

`area` is deliberately absent — it is `amp × integral(shape)` by construction and
carries nothing `amp` does not (`ntof_io` docstring). Apply
`ntof_io.saturated()` semantics downstream, not at slim time: keep the flags,
cut later.

Alongside, per DREAM event: predicted n_TOF time, `BunchNumber`, matched arm,
`δa_b`/`δk_b`, `is_flash`, and a matched/unmatched flag — so "no n_TOF partner"
is distinguishable from "not written". Per bunch: `index`, `PKUP`, `tflash` per
tree, beam intensity, per-tree total hit counts, and the fitted tb offsets. Per
file: the n_TOF processing name and the UserInput hash (§6), and the calibration
constants. All of that is kilobytes and it is the difference between a slim and
a lossy slim.

**Flash triggers are tagged, not slimmed.** `is_flash` on the event record, no
n_TOF hits written. The source stays on EOS if they are ever wanted.

## 5. What the slim gives up, and the cheap insurance

Everything outside a DREAM trigger window is gone. That costs:

- un-triggered n_TOF singles rates and QA;
- any liquid-leg or flux analysis not conditioned on a DREAM trigger;
- redoing the trigger emulation with different thresholds (a lower plastic
  threshold pulls in hits that were never written).

The per-bunch count/summary trees above cover the first, cheaply. The third is
the one to think about: `v6_lowthr` showed 15–25 % more plastic and 29–47 % more
liquid hits below the discriminator threshold, and those are in the ±150 ns
window and would be kept — so a *threshold* re-study is safe. A change to the
*time* model is not.

The insurance is that **the source is not deleted** — the reprocessed runs stay
in `/eos/experiment/ntof/processing/official/done/`, and re-slimming a segment
costs 385 s. That is cheaper than any buffer and it is why the ±2 µs buffer was
dropped (§4).

## 6. The source: what n_TOF has actually processed

**Audited 2026-08-08 against `/eos/experiment/ntof/processing/official/done/`,
not taken on trust. Three things to know.**

### (a) They are running our UserInput — but it is called `v4`

Every X17 file in `done/` carries `UserInput_2026_EAR2_X17_v4.h` in its
`history` TObjString, and its header comment is ours verbatim: *"X17 EAR2 2026 —
variant v12_liqpileup = v11_pssfit_width + LIQ\* STEP SIZE → 1/3"*. The
parameter table was diffed against `ntof_handoff/UserInput_2026_EAR2_X17_v12.h`
and is **identical on all 14 detector rows, including all 26 pulse-shape
filenames**.

So the content is right and the name is a trap: `v4` is n_TOF's own version
counter, and it collides with **our** `v4_walshapes`, which is a different and
superseded thing. Never identify a file by that string — check the variant
comment or diff the table.

### (b) The reprocessing is incomplete, and stalled

`done/` holds **325 X17 runs spanning 224300–224687**, every one written
2026-08-05 to 08-07, with nothing since 08-07 19:56. Against the run range:

- **66 runs are missing inside 224300–224687** — 31 still have raw stream1 on
  the EOS disk, 35 do not;
- the campaign starts at **224269** (07-02) — 224269–224299 are absent;
- runs after **224687** are absent, and the measurement is still going.

The list is **derived, not transcribed** — `slim_study/make_handoff.py` takes
every run in `[224300, 224687]` absent from `done/` and splits it on whether raw
stream1 survives. An earlier hand-copied version of it here said 63/33/30 and
was two runs short (224405 and 224535), which made the handoff's own arithmetic
disagree with itself. Read the current list from
`NTOF_REPROCESSING_REQUEST_2026-08-08.md` or `missing_runs_2026-08-08.csv`.

They are not in any other queue.

**And there is a clue as to why** (`slim_study/why_skipped.py`). Of the 135
in-range runs whose stream1 is still staged, the skipped ones differ from the
processed ones in exactly one visible way — size:

| raw TB | runs | skipped | rate |
|---|---|---|---|
| 0.00–0.35 | 63 | **0** | **0 %** |
| 0.35–0.45 | 13 | 5 | 38 % |
| 0.45–0.55 | 36 | 12 | 33 % |
| 0.55–0.65 | 14 | 9 | 64 % |
| 0.65–0.75 | 6 | 3 | 50 % |
| 0.75–1.00 | 3 | 1 | 33 % |

**Below 0.35 TB nothing was ever skipped, 0 of 63.** Above it 30 of 72 were, and
the rate keeps climbing (31 % in the lower half of that group, 53 % in the
upper). Ruled out: directory structure (identical — `stream0` + `stream1`, every
file `.finished`), an output-size cap (it would have to sit below 21 GB and 42
processed runs exceed that), and position in the run range (the gaps are
scattered). The non-deterministic, size-rising shape is a **resource ceiling a
large job sometimes misses and sometimes makes** — wall clock, memory or scratch
— not a rule that rejects a run outright.

The control that separates the two mechanisms: of the 11 runs missing from
*after* 224687, three are **below** 0.35 TB, a band in which the in-range
mechanism never skipped anything. Those are missing because the pass stopped.

### (c) Do not silently fill the gaps from our own `prod_v11`

Three of the nine runs covering DREAM run_79 — **224573, 224576, 224577** — are
in the missing list, and we *do* have them under
`/eos/experiment/ntof/data/x17/reproc/prod_v11/`. But their history says
`variant v11_pssfit_width`, **not v12**. Verified.

Diffing our v11 against v12: they are identical on WAL, PSS, SILI and PKUP and
differ **only** on the four LIQ rows — `STEP SIZE` 2/4 → 1/3 and
`SIGNAL WIDTH HIGH` 5000 → 5000/30. So mixing them across one DREAM run is:

- **safe** for the wall and plastic legs, i.e. for the trigger match itself;
- **not safe** for the liquids, which is a **14–21 % yield step** between runs —
  landing squarely in the liquid same-arm coincidence measurement.

Coverage of DREAM run_79 as it stands:

| n_TOF run | official `done/` (v12) | our `reproc/` |
|---|---|---|
| 224572 | ✓ | ✓ v12_liqpileup |
| **224573** | ✗ | v11 only |
| 224574 | ✓ | v11 |
| 224575 | ✓ | v11 |
| **224576** | ✗ | v11 only |
| **224577** | ✗ | v11 only |
| 224578 | ✓ | v11 |
| 224579 | ✓ | v11 |
| 224580 | ✓ | ✗ |

Either wait for n_TOF to finish, or reprocess those three ourselves on v12. The
slim must record the processing identity per file so this can never be mixed
silently.

## 7. What can be processed today: 52 % of the beam time

`slim_study/coverage_map.py` joins the reprocessed-run windows (`index` tree
Date/Time), the raw stream1 windows (what could still be reprocessed) and the
DREAM sub-run windows. Output kept as `coverage_map_2026-08-08.txt`.

The physics campaign is **DREAM runs 77-156**, 2026-07-26 to 08-08, **282
`stat090_*` beam sub-runs**, 23 440 decoded files. (Runs 1-76 are HV scans, mesh
tests and commissioning with descriptive sub-run names — no `stat090_*`, nothing
to slim.)

| | beam time | share |
|---|---|---|
| **PROCESSED** — a run in `done/` covers it | **133 h** | **52 %** |
| NOT PROCESSED | 120 h | 48 % |

**Process the covered half now, backfill the rest.** Nothing is at risk: the
staged stream1 on disk is temporary, but the raw is archived to tape, so every
missing run can still be processed — the disk state only decides whether a tape
recall is needed first.

### Do NOT gate on whole runs — gate per bunch

n_TOF run boundaries fall *inside* DREAM runs, so only **3** DREAM runs are 100 %
covered, and even at sub-run granularity only 97 of 282 (31 % of files) are
wholly inside a reprocessed run. Both numbers badly understate what is usable,
because a sub-run that straddles one v12 run and one missing run is still
processable for the half that is covered.

The natural unit is the **(DREAM sub-run x n_TOF run) segment**: a DREAM event
is slimmable iff its bunch belongs to a reprocessed n_TOF run. That recovers the
full 52 % with no chunk boundaries, and backfilling later just adds events to
existing files. It also matches the calibration's own granularity — `K`, `T0`
and the per-arm offsets are fitted per (DREAM run, n_TOF processing) pair
anyway, so a segment needs its own fit regardless. The `K` lever arm is time
*since flash* (0-80 ms), not wall-clock, so a few hundred bunches per segment is
ample.

If clean whole-sub-run units are wanted instead, the largest contiguous READY
stretches are 5.6 h (run_77-79, 07-26 14:57-20:34, 608 files), then 3-4 h blocks
in run_106, run_116, run_118, run_132.

### The 41 n_TOF runs to ask for

The list is generated, never typed — `slim_study/make_handoff.py`, exported to
`missing_runs_2026-08-08.csv` and to the request in
`NTOF_REPROCESSING_REQUEST_2026-08-08.md`. 30 sit inside 224300–224687 and 11
after it. 39 still have stream1 staged on disk; **224649 and 224650 need a tape
recall**.

Those last two are worth a note on method. They have neither a processed file
nor staged stream1, so they had no measurable time window at all and fell
straight out of the first version of this analysis. Bracketing them between
their nearest measurable neighbours by run number — n_TOF run numbers are
strictly time-ordered — puts them squarely in beam time on 02 August. Without
that step the request would have been 39 runs and quietly short.

The other 38 in-range gaps are not requested because they were live while DREAM
was not, so they block nothing for us.

### One timezone trap, since it bit this analysis

The n_TOF `index` tree's `Date`/`Time` are **local (UTC+2)**, not UTC.
`ntof_io._index_epoch` builds an epoch from them as if they were UTC — harmless
where it is used (relative matching inside one run) and a flat +7200 s error the
moment it is compared to anything else. Measured against the raw mtimes, which
are true UTC: `raw_start - index_start` is a flat **-7127 s** over the 109 runs
that have both. Before correcting it this analysis reported 13.2 % of beam time
as lost; after, 1.6 %.

## 8. Built and validated

`slim_pipeline/` runs the §4 chain for one segment. Written and tested
2026-08-08 on **run_79/stat090_0000 x n_TOF 224572** — a sub-run wholly inside a
v12 run, with both sides staged locally and published numbers to check against.

```bash
python slim_pipeline/run_segment.py run_79 stat090_0000 224572 \
    --ntof-source /media/dylan/data/x17/ntof_reproc/v12_liqpileup --out <dir>
python slim_pipeline/validate.py <dir>/.../ntof_hits_*.root
```

106 127 events, 1 151 054 hits, **40 MB, 385 s** on one core (112 s candidates,
246 s the hit pass). All four validation checks pass:

| check | slim | measured without this pipeline |
|---|---|---|
| K | 1.106350e−4 | 1.103724e−4 (fitted on both sub-runs) |
| T0 | −252.60 ns | −253.64 ns |
| arm offsets A/B/C/D | −17.06 / +7.79 / +1.86 / −1.01 | −16.81 / +7.55 / +1.62 / −0.83 |
| efficiency @ ±25 ns | 95.8864 % (cross-val 95.8493 %) | 95.84 % |
| accidental | 0.0457 % | 0.049 % |
| per-bunch δa / δk RMS | 6.55 ns / 0.92 ppm | 6.5–6.8 ns / 0.92–0.96 ppm |
| **liquid same-arm diagonal** | **0.163 / 0.150 / 0.018 / 0.093** | **0.165 / 0.151 / 0.018 / 0.094** |

Nothing in the pipeline reads a stored calibration constant — `K`, `T0` and the
per-arm offsets are re-fitted from the segment's own candidates every time, and
they land on the published values to 0.24 % and 0.25 ns.

The liquid row is the check that matters. It is a physics result recomputed
**from the slim alone**, and it is what caught the window being too narrow (§4).

### Output

`<eos>/july_beam/runs/<run>/<subrun>/ntof_hits/`, EOS only — one directory per
DREAM sub-run, no overlap between sub-runs (the DAQ log shows a ~13 s gap
between them, so the 1.2 s bunch structure cannot straddle a boundary).

| | |
|---|---|
| `ntof_hits_<run>_<subrun>_<ntofrun>.root` | trees `hits`, `events`, `bunches` |
| `calibration.json` | K, T0, arm offsets, tb offsets, thresholds, fit trace |
| `qa.json` | efficiency, accidental, purity, counts, runtime |
| `provenance.json` | n_TOF files, processing name, det codes, branches |

`events` carries **every** DREAM trigger — flash, unmatched and matched — with
`is_flash`, `matched`, `residual_ns`, `arm`, `da_ns`, `dk`, `corr_ns`, so "no
n_TOF partner" is never confused with "not written". Flash triggers are tagged
and get no hits.

### On condor

`slim_pipeline/lxplus/` — one job per n_TOF run, every overlapping DREAM
sub-run inside it. **Run for real on 224572**: xrdcp 30 GB in 133 s, three
segments, **28.0 min wall, peak RSS 3.16 GB**, one core. `segments.py` enumerates
**206 ready segments over 60 n_TOF runs** for the campaign.

The condor result also closes segment independence: `stat090_0001` and
`stat090_0002` had never been run, and 0001 reproduces **its own** published
diagonal (0.162/0.145/0.016/0.090 against 0.164/0.146/0.016/0.092). The clock is
genuinely refitted per segment — K = 1.106350e−4 on 0000 against 1.101174e−4 on
0001 — and `stat090_0000` gives an efficiency identical to four decimals whether
it ran locally or on a worker.

### Still to do before the campaign runs

| | |
|---|---|
| condor packaging | **done and proven** |
| segment enumeration | **done** — `slim_pipeline/segments.py`, 206 ready segments |
| a second validated segment | **done** — `stat090_0001`, disjoint hour, own published diagonal |
| `measure_tb_offsets` cross-check | **closed** — it runs in pass 1 on the source, and its offsets (±0.5–6.5 ns) match the study's |
| submit the 60 ready runs and publish to EOS | not done |
| the unexplained plastic late tail (§4) | not chased |

## 9. Open

1. **The 41 missing n_TOF runs.** The request is written and published:
   `NTOF_REPROCESSING_REQUEST_2026-08-08.md`, and as a page at
   <https://dylan-neff.web.cern.ch/notes/ntof-reprocessing-request.html>.
   Ask n_TOF on Monday, including *why* the in-range runs were skipped — the
   size correlation in §6b is an association, not a diagnosis.
2. **Run the campaign.** 206 ready segments over 60 n_TOF runs
   (`slim_pipeline/segments.py`), ~10 min per sub-run. Nothing blocks this but
   the decision to start.
3. **Publish the three validated outputs to EOS** — they are production quality
   and sit in `~/x17slim/out` on lxplus, unpublished.
4. **The plastic late tail** (§4): sig/ctl is flat at 1.03 for dt < −150 ns but
   still 4.2 at +238 ns, one-sided, so not a control artifact. Probably PSS
   shape fitting splitting long pulses. Does not touch the trigger match or the
   liquid physics; worth one look from the source before anyone quotes a
   plastic hit yield.

Settled: ROOT output; EOS only; one directory per DREAM sub-run; flash triggers
tagged and not slimmed; slim window **±150 ns** on the fully-corrected clock;
+100 µs accidental control carried; no intermediate buffer; job granularity =
one n_TOF run.

### Two mistakes this study made, since they are the instructive part

- **±250 ns was recommended on a wrong reading.** The liquid diagonal appeared
  to be clipped at ±100 ns, but the number that moved was the raw `sig`, which
  contains accidental floor inside `liq_coincidence`'s ±100 ns integration
  window. The background-subtracted signal is identical at ±100 and ±250. The
  control window was in the file precisely to allow that subtraction and was not
  used. `validate.py` now reports `sig − ctl` alongside `sig` for this reason.
- **Two hand-transcribed run lists were wrong** — the raw-gone gaps (two runs
  short) and, separately, the whole class of runs with no measurable window
  (which hid 224649/224650 from the request). Both are now derived in code.
  Nothing in this analysis should be a typed list of run numbers.

## 10. Reproduce

```bash
cd ntof_processing/slim_study
python window_envelope.py                                  # §1, seconds
python window_yield.py                                     # §3 narrow, ~4 min
python window_yield.py --out window_yield_wide.json \
       --windows 250 1000 2000 5000 20000                  # §3 wide, ~4 min
python window_yield.py --perbunch --out window_yield_final.json \
       --windows 25 50 75 100 150 250                      # §4 final clock
python window_yield.py --perbunch --shift-ns 100000 \
       --out window_yield_control.json --windows 25 100 250 # §4 control
python slim_closure.py                                     # §2, ~3 min
python slim_prototype.py                                   # §3 bytes, ~1 min
python coverage_map.py [--verbose]                         # §7, instant
```

`coverage_map.py` reads cached listings under `coverage_inputs/`; the commands
that regenerate them are in its `refresh_inputs()` docstring. Re-run it after
n_TOF processes anything.

They read `v12_liqpileup` through `study_common.use_variant()` and take the
calibration from `ntof_dream_merge/calibration.py`. Nothing here re-derives the
time map; if the pair changes, re-fit first (`DREAM_NTOF_CALIBRATION.md` §6).
