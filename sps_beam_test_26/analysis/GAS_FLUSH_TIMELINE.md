# Gas flushing at H4 — the timeline, the time constant, and what was ever in the chamber

**2026-08-05.** Every gas statement in this campaign so far has been about what
was *flowed*; this note is about what was *in the chamber*. The flow rate was
**~2 ln/h** (operator recollection, Dylan, 2026-08-05 — no flowmeter log
exists), against a chamber gas volume of ~4.6 L. That ratio drives everything:
the mixture exchange takes half a day, and the water level never goes away at
all.

## 0. The numbers

- Active gas volume: 399.36 mm × 399.36 mm × 28.8 mm drift gap ≈ **4.6 L**
  (plus manifold/dead volume, not in any drawing we hold — treat 4.6 L as a
  floor).
- Flow **Q ≈ 2 ln/h** → ideal-mixing exchange constant
  **τ = V/Q ≈ 2.3 h**; real chambers (dead pockets, laminar short-circuits)
  run 1–2× slower, so carry **τ = 2.3–4.6 h**.
- Old-gas fraction after time t: `f = exp(−t/τ)`. One volume ≈ 2.3 h;
  95 % exchanged ≈ 7–14 h; 99 % ≈ 11–21 h.

## 1. Gas events, now dated by the TAX stopper

The H4 accesses are dated to the second by `XTAX_022_023:POSITION_MEAS`
(`tax_windows.py`; the backfilled CSVs are now staged locally under
`records/beam/backfill_nxcals/`, pulled from mx17-daq 2026-08-05).

| event | TAX window | gas action |
|---|---|---|
| install access | Fri 07-31 **21:22 → 22:37** | det4 installed; **Ar/CO₂/iso 95/3/2 flush starts ≈ 22:00 ± 30 min** (chamber starts full of air) |
| rotation to 25.64° | Sat 08-01 **16:05 → 16:35** | none |
| gas-change access | Sat 08-01 **20:24:07 → 21:09:59** | **switch to Ar/CF₄/iso 88/10/2 ≈ 20:45 ± 20 min** |
| rotation to 15.465° | Sun 08-02 **11:03 → 11:22** | none |
| rotation back to 25.64° | Sun 08-02 **14:20 → 15:24** | none |
| SPS off (not an access) | Sun 08-02 19:02 → 21:14 | none (linac down; run_61 stopped 18:56) |
| rotation to flat | Mon 08-03 **00:40 → 01:01** | none |

Two run-timeline questions this record closes as a by-product:

1. **run_59 `detE_long` (started 20:00:54) lost its beam at 20:24:07.**
   Sub-run 00 (20:02–20:32) caught ~22 min of beam; `_01` is 100× smaller
   because it started into the access. Not a DAQ fault. (`RUN_TIMELINE.md`
   §4 open item 2 — answered.)
2. **run_60's collapse from `overnight_15` (04:54) is the SPS, not the
   detector**: the spill record shows FTARGET extractions stop at ~04:50
   08-02 and return only ~08:30 (zero intensity until then, nominal
   ~1370 e10/cycle only from ~09:30). Everything from `overnight_15` to
   `overnight_23` is essentially beamless. The gas-transition dataset
   therefore ends at 04:50.

## 2. What was in the chamber, run by run

Times below are hours since the relevant gas start (flush start 07-31 ~22:00;
switch 08-01 ~20:45). "old-gas fraction" is `exp(−t/τ)` quoted for
τ = 2.3 h | 4.6 h.

| run | when | h since event | in the chamber |
|---|---|---|---|
| run_53 (first data) | Sat 12:59 | 15.0 after flush start | residual **air 0.1 % / 4 %** in CO₂ mix — O₂ attachment cannot be excluded for the *early* Saturday points |
| run_55 flat drift scan | Sat 14:25 | 16.4 | air 0.1 % / 3 % |
| run_56 m70V (kernel #1) | Sat 15:47 | 17.8 | air ≤ 2 %; effectively the CO₂ mixture |
| run_57 resist ladder 25° | Sat 16:55–18:03 | 19–20 | CO₂ mixture |
| run_57/58 drift ladder 25° | Sat 18:14–19:45 | 20–22 | CO₂ mixture |
| run_59 detE_long | Sat 20:02–20:32 | 22 | CO₂ mixture (last CO₂ dataset; beam dies 20:24) |
| **run_60 overnight** | Sat 21:20 → Sun 04:50 (beam) | **0.6 → 8.1 after switch** | **CO₂-mix fraction 77 % → 3–17 %: the transition itself** |
| run_61 15° half | Sun 12:13–14:00 | 15.5–17.3 | CO₂ residue 0.1 % / 2–3 % |
| run_61 25° half | Sun 16:29–18:56 | 19.7–22.2 | CO₂ residue ≤ 1 % even at τ = 4.6 h |
| run_62, run_63 rot25 | Sun evening → Mon 00:40 | 26–28 | CF₄ mixture |
| run_63 flat, run_71 RAW | Mon 01:00–05:52 | 28.3–33.1 | CF₄ mixture (12–14 τ) |

Consequences for existing conclusions:

- **The kernel gas-transfer claim is safe.** run_56 (CO₂ side) sat at ≥ 8
  volumes of CO₂-mix flushing; run_63/71 (CF₄ side) at ≥ 12 volumes of
  CF₄-mix. Neither is transition gas. What both *do* carry is the water floor
  (below).
- **run_61's 15° vs 25° efficiency gap (36.7 % vs 32.7 %,
  `RUN_TIMELINE.md` §4 item 5) has a candidate gas term**: at τ = 4.6 h the
  15° half still carries ~2–3 % CO₂-mix, the 25° half < 1 %. A percent-level
  quencher change moves the gain at fixed voltage. Not attributable yet —
  angle, pedestal set and time all still differ — but the flush model now
  puts a number on the gas axis, and the run_60 fit (§3) will pin τ.
- **The 175 V operating-point shift vs the bench**
  (`DET4_EFFICIENCY_H4_2026-08-01.md` §4): candidate 2 ("gassed for hours,
  not days") is *quantified* — at run_53 the chamber had seen 15 h ≈ 3–7
  volumes from air. Ideal mixing says the air was gone; a factor-2 mixing lag
  says up to ~4 % air (≈ 0.8 % O₂) remained for the run_53 column
  specifically. It cannot explain the shift persisting into run_56/57
  (< 2 % air even pessimistically), so the *gas identity* (candidate 1: the
  H4 mixtures are 95/3/2 and 88/10/2, not the bench's Ar/iso 95/5) remains
  the leading explanation for the bulk of the 175 V.

## 3. The water floor — why "never fully flushed" is right anyway

Mixture exchange is a transient; **water is a steady state.** The chamber
walls (FR4, glue, O-rings) outgas water continuously, and the equilibrium
water fraction is `R_outgas / Q` — it does not decay with flushing time, it
scales inversely with *flow*. At 2 ln/h this hardware family sits at
percent-level water: the July n_TOF chambers measured **~0.8 % H₂O** on the
same style of line at comparable flow, and this campaign's own drift
velocity — **13–15 µm/ns at 233 V/cm, 4× below dry Magboltz**
(`RAW_RUN71_REANALYSIS_2026-08-04.md`) — is the signature of percent-level
water in Ar/CF₄/iso.

So the campaign's gas state, in one sentence: **the labelled mixtures are
correct to ~1 % from a few hours after each change, but every dataset, both
gases, is wet at the ~1 % level, and the wetness is a property of the 2 ln/h
flow, not of the flushing duration.** The "open item" on the slow drift
velocity is therefore expected physics for this line, not an anomaly; what is
genuinely open is only the *quantitative* check (Magboltz with H₂O admixture
reproducing 14 µm/ns at 233 V/cm).

Corollary for the June bench comparison: the bench ran the same chamber dry
enough for v ≈ 34–40 µm/ns. Any H4↔bench comparison of anything
drift-time-derived (v, tilt angles, micro-TPC scales) crosses a 3–4× velocity
step that is *gas-system*, not detector.

## 4a. MEASURED (2026-08-05, `flush_run60.py`): lag 1.7 h, τ = 3.5 h

The design below was executed the same day. run_60's 13 beam-on sub-runs,
anchored by run_59 (CO₂ side, same drift HV, threshold-robust span estimator)
and run_63 `operating_03` (fully-exchanged CF₄, +28.5 h):

| | value |
|---|---|
| transport lag (line volume at 2 ln/h) | **1.72 ± 0.23 h** (≈ 3.4 L of line/manifold upstream) |
| exchange constant τ | **3.49 ± 0.57 h** = 1.5 × ideal V/Q — healthy mixing |
| 95 % exchanged | **+12 h after the switch** |
| hit-time span (∝ 1/v_drift) | 2340 → 1996 ns → **v(CF₄-mix)/v(CO₂-mix) = 1.17** at 243 V/cm |
| gain at fixed HV (drift 700.5/resist 649.75) | Y 399 → 322 (−19 %), X 387 → 319 (−18 %) — the CF₄ mixture's quencher load |
| in-time hits/event | 8.8 → 8.8 (flat — null check) |

Notes: the plain exponential first fitted 12–15 h taus — an artefact of
fitting a *lagged* transient without the lag; the anchors and the lag term
fix it. The gain observable is NOT comparable across runs (each run has its
own ZS σ); within run_60 (fixed 5σ) it is, and its A0→Ainf uses the span's
(lag, τ). run_59's raw gain (181 at 3σ, resist **669.8 V** — not 649.75, its
HV differed) is therefore not the gain anchor; only its span (2355 ns) is
used.

Consequences for §2's model rows: the measured (lag, τ) says residual
old-mix was **~2 % in run_61's 15° half (+15.5 h), < 1 % from the 25° half
onward, and fully negligible for run_62/63/71**. The kernel gas-transfer
claims stand. And the §2 run_53 "residual air" row should be read with the
same lag: first beam data came 15 h ≈ lag + 4 τ after the CO₂ flush start,
i.e. ~2–4 % residual air is the pessimistic edge, not the centre.

## 4b. run_60 flush measurement — original design

24 × 30 min sub-runs starting 21:20, i.e. 0.6 h → 8.1 h (beam-usable,
`overnight_00`–`overnight_14`) after the switch. Old-gas fraction over that
span: 77 % → 3.5 % (τ = 2.3 h) or 88 % → 17 % (τ = 4.6 h). The two
hypotheses separate by ×5 in the tail — easily resolvable.

Per sub-run observables (det4 FEU3, decoded with the patched decoder;
uRWELL reference from banco's own FEU1 combined_hits, which are ZS-mode and
therefore trustworthy):

1. **Gain proxy**: median leading-strip peak amplitude (fixed cuts). CO₂ 3 %
   → CF₄ 10 % at fixed HV is a large gain step; the approach to plateau
   measures τ directly.
2. **Drift-time ladder span**: the hit-time distribution width tracks
   v_drift; CO₂-mix and CF₄-mix at the same field differ measurably, and the
   water fraction rides along.
3. **Occupancy / hits per event** (threshold-crossing rate) as a cross-check
   with no amplitude calibration at all.
4. **In-band efficiency vs time** against the uRWELL, run61-style, as the
   integral consequence.

Fit `A(t) = A_∞ + (A_0 − A_∞) · exp(−t/τ)` per observable. Agreement of τ
across observables = the flush constant of this chamber on this line; then
§2's percent-residue rows stop being model and become measurement.

**Caveat for interpretation:** resist/drift HV were held fixed through
run_60 (confirm from the staged `hv_monitor.csv` per sub-run before fitting —
an HV step would alias into the gas curve).

## 5. Data status for this analysis

- run_60 FEU3 fdf + FEU1 combined_hits + per-sub-run HV: staging in progress
  (2026-08-05, `pull_wave1.sh`).
- run_59 (last CO₂ dataset, 64 samples, high statistics): staging in the same
  wave — it is the CO₂-side anchor `A_0` for §4's fit, 40 minutes before the
  switch.
- run_55 (flat CO₂ drift scan, 700–400 V, ~5 GB): the *flat* drift lever
  the CF₄ analysis lacks, and the CO₂-era v_drift measurement — queued for
  wave 2.
- TAX + spill CSVs: staged under `records/beam/backfill_nxcals/` (07-31 →
  08-05).
