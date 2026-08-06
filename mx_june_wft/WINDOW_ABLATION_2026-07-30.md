# Phase 0 of PLAN_08: what the n_TOF readout costs the waveform fit

Answers the three questions `ntof_tracking/TRACK_PLAN_08_waveform_first_run79.md`
§4 poses, **on bench data with M3 truth**, before any beam reconstruction is
built: (1) what does run_79's 20-sample readout window cost, (2) does a
calibration bundle transfer to data it was not fitted on, (3) which calibration
constants actually have to be re-measured in situ.

Tools (all new, all in `bench/`): `framing_compare.py`, `window_ablation.sh`,
`sensitivity_scan.sh`, `transfer_ablation.sh`, `make_bundle_variant.py`,
`summarize_scans.py`; `run_bench.py` gained `--crop START:N` and `--k-bins`.

---

## 1. Framing: where the drift column sits in each readout window  [done]

`framing_compare.py`, one hits file per side, production seed clustering
(relative significance floor, 12 mm gap, largest cluster, ≥5 strips), largest
pulse per channel. "Clean" additionally requires a real micro-TPC column:
|rank corr(strip position, peak sample)| > 0.7, 5–25 strips, peak > 300 ADC.

`onset` / `edge` are the earliest / latest peak sample in the cluster; `span`
is the column's drift time in samples; `ceiling` is the fraction of clusters
whose deepest strip peaks in the **last** sample bin, i.e. whose column is cut
off by the end of the window.

| | window | onset p5 | edge p50 / p95 | span p50 | **at ceiling** | clean clusters/file |
|---|---|---|---|---|---|---|
| bench det3 X (`sat_det3`) | 32 | 6.9 | 18.9 / 21.1 | 10.7 | **0.07 %** | 5 804 |
| bench det3 Y | 32 | 7.0 | 19.5 / 22.1 | 11.2 | **0.29 %** | 5 893 |
| run_79 A (det3) X | 20 | 1.4 | 12.7 / 17.6 | 9.3 | **4.5 %** | 446 |
| run_79 A Y | 20 | 1.4 | 12.9 / 17.6 | 9.5 | **3.3 %** | 518 |
| run_79 B (det2) X | 20 | 0.5 | 18.1 / 19.7 | 12.6 | **45.7 %** | 223 |
| run_79 B Y | 20 | 0.7 | 19.0 / 19.8 | 13.9 | **55.7 %** | 253 |
| run_79 C (det6) X | 20 | 1.3 | 16.3 / 19.2 | 12.9 | **13.3 %** | 572 |
| run_79 C Y | 20 | 1.2 | 16.4 / 19.3 | 12.7 | **16.1 %** | 578 |
| run_79 D (det7) X | 20 | 1.4 | 14.7 / 19.0 | (see below) | **8.6 %** | 1 010 |
| run_79 D Y | 20 | 1.3 | 14.8 / 19.3 | 11.2 | **11.6 %** | 483 |

### What this says

* **The crop that emulates run_79 on the bench is `start +6, keep 20`** — the
  bench prompt sits at sample 6.4–7.0 and the beam prompt at 0.2–1.4, and the
  offset comes out +5.5…+6.4 on all four chambers, both planes, raw or clean.
  Measured, not derived: run_79 changed both the DREAM `latency` (35 → 27) *and*
  the M4.D1 trigger G&D delay, so the frame shift is not the latency difference.
* **run_79 truncates the drift column, by a lot on some chambers.** The bench
  loses 0.1–0.3 % of columns to the end of its window; run_79 loses **3–4 % on
  A, 9–12 % on D, 13–16 % on C and 46–56 % on B**. B is the binding chamber, as
  `mx_july_beam_qa/DRIFT_WINDOW_ANALYSIS.md` predicted on 07-19 — and its
  ceiling pile-up survives the ladder cut, so it is drift, not the ringing
  artifact that analysis warned about.
* **The column length tracks the gas, and A is the fast one.** Span p50: A 9.4
  samples (≈ 565 ns) against the bench's 10.7–11.2 (≈ 660 ns) — i.e. the dry
  90/10 gas at 233 V/cm drifts *faster* than the wet 95/5 bench gas at 333 V/cm,
  as Magboltz says it should. C 12.8 and B 12.6–13.9 are slower, and B's is a
  lower bound because half of its columns are cut. Do not turn these into
  v numbers: threshold losses cut both ends of a hits-level span (the July
  analysis used a 0.89 recorded-column factor), and the truncated chambers bias
  low. The reference-free column-endpoint method (PLAN_08 §6.3) is the way to
  measure v.
* **The prompt end is tight too.** onset p5 = 0.5–1.4 means the near-mesh charge
  arrives within the first sample or two; there is essentially no pre-signal
  baseline in run_79. The fit's t0 prior must sit near sample 0, not the
  bench's ~8.

### Two things found while measuring

* **A channel can carry several hits per event at beam** (pileup, and ringing
  after a saturated event). Indexing hits by channel therefore pulled a later
  secondary pulse in as the column's "deep edge" and inflated every deep-edge
  number in the first pass. `cluster_stats` now keeps the largest pulse per
  channel. Any beam analysis that maps channel → hit needs the same guard.
* **det D's X plane is full of wide isochronous deposits**: 55 % of its
  otherwise-clean clusters have a span < 3 samples with a median of 17 strips
  (≈ 13 mm of strips all peaking at once). Those are not gap-crossing columns
  and the forward model would fit them as junk — the beam seeder needs a
  minimum-span requirement, not just a strip count.

### Scope, as a side effect

Clean columns in **both** planes of the same event, per hits file (1/13 of a
sub-run): A 224, B 80, C 308, D 176 → scaled over 13 files × 2 sub-runs,
**≈ 5 800 (A), 2 100 (B), 8 000 (C), 4 600 (D)** reconstructable events. That is
plenty for both calibration and the merge, and it is a conservative floor: the
ladder cut deliberately rejects near-vertical tracks, which are the most useful
events for measuring the template and the sharing kernel.

---

## 2. Window-length scan  [done]

det3 `sat_det3` cache, `calib_bundle_lp2`, production configuration
(`MODEL_FRAC=0.03`, coarse pre-scan), the same 1 200 events at every point,
crop start +6 as measured above. `full32` reproduces the production numbers
(93.54 % / 0.460 mm / 1.08–1.11° on all 7 093 events).

| kept | within 5 mm | core σ [mm] | σθ X | σθ Y | compression \|tan\|>0.14 X / Y | v-spread X / Y |
|---|---|---|---|---|---|---|
| 32 (full) | 94.80 | 0.473 | 1.06 | 1.11 | −0.08 / −0.10° | 2.1 / 3.7 |
| 26 | 94.80 | 0.478 | 1.10 | 1.18 | −0.23 / −0.33° | 2.3 / 3.2 |
| 24 | 94.80 | 0.471 | 1.10 | 1.18 | −0.23 / −0.33° | 2.2 / 3.4 |
| 22 | 94.63 | 0.469 | 1.10 | 1.21 | −0.23 / −0.35° | 2.2 / 3.7 |
| **20** (run_79) | 94.80 | 0.465 | 1.12 | 1.22 | −0.26 / −0.41° | 2.3 / 4.1 |
| 18 | 94.80 | 0.466 | 1.10 | 1.26 | −0.31 / −0.48° | 2.3 / 4.3 |
| 16 | 94.63 | 0.486 | 1.09 | 1.32 | −0.41 / −0.65° | 2.3 / 4.5 |
| 14 | 94.46 | 0.468 | 1.18 | 1.36 | −0.55 / −0.82° | 2.4 / 4.2 |
| 20, K = 15 | 94.80 | 0.467 | 1.12 | 1.21 | — | 2.3 / 4.1 |
| 20, K = 12 | 94.80 | 0.485 | 1.11 | 1.27 | — | 2.1 / 4.0 |

**Position is untouched at every length.** within-5 mm and core σ are flat from
32 down to 14 samples; the fit finds the track, it just measures its angle less
well. The whole cost is in angles and mostly as a **bias** — the compression of
inclined tracks that a short window produces does not average away, unlike σ.

**Shortening the charge basis does not help.** K = 15 (matched to the shorter
window) is identical to the default K = 18, and K = 12 is worse. The model
already handles a truncated window correctly — what is lost is information, not
basis coverage. Do not "fix" a short window by shrinking K.

**The leading edge is not dead weight.** `n = 26` removes *only* pre-signal
samples and keeps the entire tail, and it still costs −0.08 → −0.23 (X) and
−0.10 → −0.33° (Y) of compression, plus σθ Y 1.11 → 1.18. That is as much
damage as the following six samples of *tail* (26 → 20). The rise is what
constrains t0, and t0 trades against the slope. See
`mx_july_beam_qa/HANDOFF_2026-07-30_readout_window_and_detB.md` §3 — at n_TOF
the leading edge is already partly outside the window.

### Mapping a bench point to a beam chamber

Bench and beam have different column lengths, so a bench sample count does not
transfer; the **tail margin** does. Margin = (last sample) − (median column
edge), measured in each frame:

| chamber | margin [smp] | ≈ bench point | expected σθ Y / compression Y |
|---|---|---|---|
| A | 6.2 | **n = 20** | 1.22° / −0.41° |
| D | 4.2 | n ≈ 18 | 1.26° / −0.48° |
| C | 2.6 | n ≈ 16–17 | 1.32° / −0.65° |
| B | ~0 | n ≤ 14, off the end of the scan | ≥1.36° / ≤−0.82° |

### 2d. Frame position at fixed window length  [done]

`latency_scan.sh`: same 1 200 events, same n = 20, only the crop start moves.
A *smaller* start puts the signal *later* in the frame — at n_TOF that is a
**higher** DREAM latency (latency 35 → 27 moved the signal 8 samples earlier,
so ≈ 1 sample per unit).

| framing | within 5 mm | core σ [mm] | σθ X | σθ Y | compression X / Y |
|---|---|---|---|---|---|
| full 32 | 94.80 | 0.473 | 1.06 | 1.11 | −0.08 / −0.10° |
| start 3 | 94.80 | 0.478 | 1.05 | 1.14 | −0.19 / −0.28° |
| **start 4** | 94.71 | 0.465 | **1.05** | 1.17 | **−0.18 / −0.18** |
| start 5 | 94.71 | 0.465 | 1.07 | 1.20 | −0.21 / −0.21° |
| **start 6 = run_79** | 94.80 | 0.465 | 1.12 | 1.22 | −0.26 / −0.41° |
| start 7 | 94.71 | 0.524 | 1.18 | 1.37 | −0.31 / −0.48° |
| start 8 | 94.71 | 0.546 | 1.29 | 1.45 | −0.40 / −0.68° |

**The optimum is start 3–4, two to three samples later than run_79 runs.** It
recovers σθ Y 1.22 → 1.14–1.17 and roughly halves the compression bias
(−0.41 → −0.18/−0.28° on Y) **at zero readout cost** — and it does so while
keeping *two fewer* tail samples, which is the same lesson as §2's n = 26
point: the leading edge buys more than the tail.

Going the other way is punished hard, and the punishment shows up in
**position** as well as angle (core σ 0.465 → 0.546 mm by start 8) — that is
the regime where the column runs off the end, i.e. what B is already in.

**Translated to the DAQ**: at n = 20, **latency 27 → 29–30**. But the trade is
chamber-dependent, because moving the signal later spends tail margin
(§2c): A has 6.2 samples of margin and can afford it, C has 2.6 and cannot,
and the start 7/8 rows show what running out looks like.

**Untested and probably optimal**: start 4 *with* n = 24–26, i.e. raise the
latency **and** the sample count together. Neither scan covers that corner —
`window_ablation.sh` fixed start at 6 and `latency_scan.sh` fixed n at 20.
Two runs would settle it.

## 3. Constant sensitivity — which constants need an in-situ measurement  [done]

`sensitivity_scan.sh`: one-at-a-time ±25 % perturbation of every model hyper of
`calib_bundle_lp2`, 800 events, scored against M3. Base: σθ 1.09 / 1.21°,
core σ 0.470 mm, within-5 mm 94.12 %.

| constant | σθ X, −25 % / +25 % | σθ Y, −25 % / +25 % | verdict |
|---|---|---|---|
| `c1` (±1-strip sharing) | +0.3 / −0.9 % | +0.2 / +0.6 % | **insensitive** |
| `c2` (±2-strip sharing) | −1.0 / +0.8 % | −1.4 / +0.2 % | **insensitive** |
| `sigma_s` (copy smearing) | −0.1 / +0.1 % | +1.2 / +1.6 % | **insensitive** |
| `kY` (Y sharing amplitude) | 0.0 / 0.0 % | −3.8 / +0.6 % | mild, Y only |
| `tau_s` (sharing timescale) | +0.6 / +0.6 % | **−8.6 / +5.4 %** | **sensitive, Y** |
| `kTauY` (Y/X timescale ratio) | 0.0 / 0.0 % | **−8.6 / +5.4 %** | **sensitive, Y** |
| `sigma_p0` (initial cloud size) | **+6.6** / −0.4 % | +2.0 / **−11.4 %** | **sensitive** |
| `Dp` (transverse diffusion) | −1.2 / −0.3 % | +3.4 / **−4.8 %** | **sensitive, Y** |

Position is essentially untouched by all of them (core σ 0.464–0.500 mm,
within-5 mm 93.86–94.25 %) — the same pattern as the window ablation: these
constants buy angles, not positions.

**What this means for run_79.** The constants that matter are exactly the ones
physics says change: the **diffusion pair** (`sigma_p0`, `Dp`) is gas- and
field-dependent and *must* be re-measured in situ, and the **sharing timescale**
(`tau_s`, `kTauY`) is sensitive enough that it has to be verified rather than
assumed. The sharing *amplitudes* (`c1`, `c2`, `kY`) and the copy smearing
(`sigma_s`) can be carried from the bench — a 25 % error in any of them costs
under 2 % of σθ, well inside the statistical error. That shrinks the in-situ
calibration to a 3–4 parameter problem, which is what makes a reference-free
fit tractable (PLAN_08 §6.5).

**Caveat worth chasing separately**: several perturbations *improve* σθ Y over
the bench value — `sigma_p0` +25 % gives 1.07° against 1.21°, `tau_s` and
`kTauY` −25 % give 1.11°. The statistical error here is ~±0.03°, so these are
real. The bundle was fitted by χ², not by angle resolution, so this is not a
contradiction — but it says the det3 lp2 bundle is not at the σθ optimum and a
targeted refit could buy ~10 % on Y.

## 4. Bundle transfer — the kernel does NOT travel for free  [done]

`transfer_ablation.sh`. Kernel and template come from one calibration, the
angle constants (`w0`, `kw`) always from the target run, via
`make_bundle_variant.py` — so this isolates the kernel, not the angle mapping.

| target | calibration used | within 5 mm | core σ | σθ X | σθ Y |
|---|---|---|---|---|---|
| det3 6-27 (`sat_det3`) | **own** | 94.12 | 0.470 | **1.09** | **1.21** |
| det3 6-27 | det3 **6-22** kernel | 94.37 | 0.502 | 1.33 (+22 %) | 1.47 (+21 %) |
| det2 6-22 | **own** | 93.80 | 0.408 | **1.17** | **1.53** |
| det2 6-22 | **det3** kernel | 93.93 | 0.419 | 1.79 (+53 %) | 2.19 (+43 %) |

**Position transfers perfectly; angles do not.** Within-5 mm and core σ are
unchanged in both directions (they even improve marginally), while σθ degrades
**21–22 % for the same detector across two runs five days apart**, and
**43–53 % across detectors**.

So for run_79: a bench bundle is good enough to reconstruct **positions** at
n_TOF today, and **not** good enough for angles. The kernel has to be verified
or re-measured in situ before any angle is quoted — which is what PLAN_08 §6.4
asks for, now with a number attached.

*Caveat on the same-detector row*: the 6-22 det3 bundle carries `kw` = 0.37/0.40
where a good calibration gives ~1.0, i.e. that calibration is itself suspect, so
22 % is an upper bound on same-detector transfer loss. The reverse direction
could not be measured — the 6-22 det3 benchmark cache scores σθ 6–7° under
*every* bundle including its own, so that cache (or its M3 alignment) is broken
and is not a usable target. **Do not use `g_det3`'s bench cache.**

### 4a. v_drift is a pure angle scale — confirmed  [measured]

Reconstructing `sat_det3` with the same bundle at v ±10 %:

| v | within 5 mm | core σ | median r | σθ X | σθ Y |
|---|---|---|---|---|---|
| 36.6 (own) | 94.12 | 0.470 | 0.732 | 1.09 | 1.21 |
| +10 % (40.26) | 94.12 | 0.470 | 0.732 | 1.44 | 1.35 |
| −10 % (32.94) | 94.12 | 0.470 | 0.732 | 1.69 | 1.90 |

**Position is bit-identical** — v genuinely never enters the fit, exactly as the
code says (it appears only in `tan = (w·10³ − w0)/(kw·v)`). A v error is
therefore correctable after the fact without re-reconstructing.

But note it does **not** leave σθ alone: a scale error on tan turns the spread
of true angles into extra residual, so 10 % of v costs 12–57 % of σθ. "Pure
scale, correctable" is right; "harmless" is not. v still has to be measured to
~1 % before angles are quoted, or corrected afterwards from the same data.
