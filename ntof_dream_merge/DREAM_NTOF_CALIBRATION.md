# The DREAM ↔ n_TOF time calibration

**This is the authoritative document for matching DREAM triggers to n_TOF
coincidences. Every other description of the match in this repository is
retired and points here.** Last re-derived 2026-07-30.

Slides: `match_study/latex/dream_ntof_matching_slides.pdf`.
Tooling and reproduce recipe: `match_study/README.md`.
Constants, machine-readable: `../../nTof_x17_DAQ/calibrations/dream_ntof/`.

Measured on the **complete reference pair** — DREAM run_79 `stat090_0000` +
`stat090_0001`, 2061 bunches, **213 420** non-flash triggers — against n_TOF run
224572 processed with **`v12_liqpileup`**, read on the file's own stored
`tflash`.

---

## 0. The result in one table

| at the accept window \|r\| < 25 ns | wall AND plastic | wall only |
|---|---|---|
| efficiency | **95.84 %** | 98.30 % |
| accidental match rate | **0.049 %** | 1.04 % |
| purity of the matched sample | 99.998 % | 99.982 % |
| ≥ 2 candidates in the window | 2.46 % | 5.36 % |
| ≥ 2 *arms* in the window | **0.15 %** | 3.04 % |

Match resolution, cross-validated: **68 % half-width 6 ns, flat from 1 ms to
80 ms.** Accidental rates are measured (the identical match with the DREAM time
shifted +100 µs; the −100 µs control gives 0.062 %), never modelled.

Coverage, accidental-subtracted, in a window wide enough that the window itself
costs nothing:

| | fraction of the 213 420 triggers |
|---|---|
| matched to a wall coincidence (wall leg alone) | 98.59 % |
| matched to a wall **AND** plastic SINGLES | **96.00 %** |
| plastic partner present, given a wall match | 97.38 % |
| plastic leg costs | 2.58 % |
| wall leg costs | 1.41 % |

---

## 1. The calibration chain

Align n_TOF to itself **first**, using only n_TOF information, then map DREAM
onto the already-consistent n_TOF time base. Steps 1–2 cannot be contaminated by
anything on the DREAM side, and step 2 is an independent check on step 1.

```
 ┌───────────────── inside n_TOF ─────────────────┐ ┌──── DREAM onto n_TOF ────┐

 [1] absolute            [2] relative              [3] global map      [4] per bunch
 t_flash(tree)           prompt coincidences       t_n = t_D(1+K)      + δa_b + δk_b·t_D
   − t_flash(PKUP)       of hits at t > 100 µs       + T0 + a_arm      from that bunch's
 per bunch, per tree                                                    own ~107 triggers
        │                        │                        │                    │
        ▼                        ▼                        ▼                    ▼
 all 12 trees on one     confirms it: WAL/PSS      K, T0 and the       residual FLAT at
 time base; the beam     2.3 ns RMS per channel,   per-arm offsets;    6 ns over the whole
 pickup needs no         LIQ−WAL < 1 ns            residual centred    80 ms
 common particle         ⇒ no offsets applied      but fans out        ⇒ accept ±25 ns
```

Step 1 is the only estimator that can compare *different* detectors — it
references every tree to the beam pickup, which needs no common particle — so it
is what puts arm A and arm D on one clock. Step 2 uses real particles and is
therefore the independent confirmation, not the primary measurement.

The trigger being matched is the N1081B **sector SINGLES**, rebuilt from the hit
trees exactly as the hardware formed it: the 428F analogue *sum* of the two bar
ends over the wall threshold, ORed over the four bar segments, ANDed with a
plastic bar over its threshold inside the 20 ns logic pulse.

---

## 2. The constants

### 2a. Offline — the DREAM → n_TOF time map

```
t_nTOF = t_DREAM · (1 + K + δk_b) + T0 + a_arm + δa_b
accept  |t_candidate − t_nTOF| < 25 ns          (one band, no satellite)
```

| symbol | value (run_79 ↔ 224572 / v12_liqpileup) | what it is |
|---|---|---|
| `K` | **1.103724e−4** | DREAM clock rate error against n_TOF |
| `T0` | **−253.64 ns** | fixed offset |
| `a_A` | **−16.81 ns** | arm A trigger-path delay |
| `a_B` | **+7.55 ns** | arm B |
| `a_C` | **+1.62 ns** | arm C |
| `a_D` | **−0.83 ns** | arm D |
| `δa_b` | fitted per bunch, RMS 6.5–6.8 ns | residual per-burst offset |
| `δk_b` | fitted per bunch, RMS 0.92–0.96 ppm | residual per-burst rate |
| window | **±25 ns** | accept half-width |

Fit quality: the global map is fitted robustly (per-time-bin median, then a
straight line, three iterations); the per-bin scatter about the line is 0.4 ns
and the two sub-runs agree to ±1 ns. The per-arm offsets reproduce between the
two independent hours to ≤ 2.6 ns. All 2061 bunches get a per-bunch fit.

### 2b. Offline — n_TOF internal

| | value on v12_liqpileup / 224572 | action |
|---|---|---|
| liquid vs wall | −0.8 … +0.2 ns | **no offset** — the liquid leg joins as it is |
| wall vs plastic, per arm (peak) | A −6.8, B −3.8, C −6.3, D −8.8 ns (σ ≈ 11 ns) | inside the 20 ns logic pulse; no offset applied |
| wall vs plastic, per channel | RMS 2.3 ns, range −7.7 … +1.9 ns | no offset |
| plastic bar spread | RMS 1.8 ns | no offset |
| wall top − bottom, per segment | within ±6 ns (peak) | **measure per file**, do not reuse a table |
| wall flash vs PKUP | −1719.3 / −1719.6 / −1721.3 / −1723.3 ns (A/B/C/D) | 4.0 ns spread |
| liquid flash vs PKUP | −1708.1 / −1710.5 / −1695.7 / −1701.0 ns | reproduces the divert-off calibration to **0.1–0.5 ns** |
| plastic flash vs PKUP | −1685.4 / −1690.2 / −1681.9 / −1682.0 ns | 31–50 ns from the divert-off constants — **take PSS per run** |
| laptop `tflash_repair` | would shift LIQC/D by 15 ns, add 25 ns RMS on PSSC | **off** on any reprocessed file |

The liquid flash times agreeing with `ntof_processing/flash_timing/` to half a
nanosecond is two independent measurements — one on seven divert-off runs, one
on this run's own data — confirming each other.

### 2c. DAQ — as operated, not fitted

These are hardware state read back from each sub-run's `n1081b_config.json`.
They are **not** something this analysis derives; they are an *input*, required
to emulate the trigger, and they are what a shifter would change.

| arm | wall threshold | plastic threshold |
|---|---|---|
| A | 25 mV | 118 mV |
| B | 35 mV | 139 mV |
| C | 34 mV | 157 mV |
| D | 36 mV | 134 mV |

---

## 3. What transfers, and what does not

| quantity | scope | re-derive when |
|---|---|---|
| DAQ thresholds (§2c) | hardware state | anyone changes a discriminator; read per sub-run regardless |
| n_TOF internal alignment (§2b) | **per processing** | any reprocessing, any UserInput change |
| wall top/bottom offsets | **per processing** | always — see below |
| `K`, `T0`, per-arm offsets | **per (DREAM run, n_TOF processing) pair** | every run pair |
| `δa_b`, `δk_b` | per bunch | always, by construction |
| accept window ±25 ns | the *method*, not a constant | stable so far; re-check the knee when re-fitting |

Three concrete transfer failures, all measured:

- **`K` and `T0` fitted on the official processing do not describe v12.** They
  leave a −45 ns offset and a 1.35 % rate error.
- **The wall top/bottom "cable offsets" are a reconstruction artifact.** Same
  bunches (1007–1156), same estimator: on the official file they are ±32–39 ns
  with one −77.5 ns outlier; on v12 they are within ±5.5 ns. The structure was
  the old flash-finder / leading-edge timing, removed by the wall shape fitting
  of `v4_walshapes`. Reusing the stored table on a reprocessed file would pair
  the bar ends around a 38 ns offset that is no longer there and lose most
  genuine pairs. `fast_singles.measure_tb_offsets` measures it in seconds.
- **The plastic flash constants do not transport between runs**, exactly as
  `ntof_processing/flash_timing/README.md` warns. The liquids do.

---

## 4. Why the window is ±25 ns and not wider

The ±150 ns window this merge was originally built with was never a resolution.
The DREAM timestamp clock drifts ~1 ppm from bunch to bunch, which smears the
residual **in proportion to the time since the flash** — 9 ns at 1 ms, 37 ns at
40–80 ms. A width proportional to elapsed time is a rate error, not a timing
resolution, and it is removable. Fitted per bunch and cross-validated:

| t since flash | before the per-bunch fit | after |
|---|---|---|
| 1–3 ms | 9.1 ns | 6.8 ns |
| 3–10 ms | 10.7 | 6.6 |
| 10–20 ms | 14.4 | 6.2 |
| 20–40 ms | 21.1 | 5.8 |
| 40–80 ms | 36.6 | 6.0 |

`δk` is *structured* in bunch number — neighbouring bursts drift together — so
it is a real oscillator drift, not fit noise (§5 quantifies that).

**The criterion** for the window is the tightest half-width still within 0.5 %
(relative) of the efficiency plateau. It lands at 23.6 ns on both legs and in
every time bin (28 ns in the single hardest bin, wall-only at 1–3 ms) → quote
**±25 ns**. Note the objective is *not* efficiency × purity: purity saturates
above 99.9 % across the whole scan, so that product is degenerate and picks
absurdly wide windows. See `match_study/scripts/recommend_window.py`.

Per time since flash, wall AND plastic at ±25 ns:

| t since flash | efficiency | accidental | purity |
|---|---|---|---|
| 1–3 ms | 94.41 % | 0.147 % | 99.991 % |
| 3–10 ms | 94.69 % | 0.128 % | 99.993 % |
| 10–20 ms | 95.75 % | 0.022 % | 99.999 % |
| 20–40 ms | 96.80 % | 0.002 % | 99.9999 % |
| 40–80 ms | 96.88 % | 0.005 % | 99.9998 % |

Going wider does not pay: ±150 ns buys +0.17 points of efficiency for 7× the
accidental background and nearly double the two-arm ambiguity (0.15 → 0.28 %).
Going tighter is expensive: ±15 ns costs 3.0 points.

**There is no satellite band.** The `[+250, +450] ns` band in the original
window was a delayed wall lobe of the *old* pulse reconstruction; the plastics
never had it. On v12 it adds 0.00 points of efficiency and 0.21 points of
background. Do not carry it.

---

## 5. The per-bunch fit does not manufacture its own matches

The clock is fitted on triggers that were matched, and then used to match. A
two-parameter fit that chased noise would pull the prediction towards whatever
candidate happened to be nearest, narrowing the residual and raising the
efficiency for free. Five tests, none of which requires believing the fit
(`match_study/scripts/bias_check.py`, figure `fig_bias.pdf`):

1. **Statistics.** Median **107** matched triggers per bunch, minimum 53, for 2
   parameters. A fit absorbs ~2/N ≈ 2 % of the variance, i.e. 1 % of a width.
   The propagated uncertainty of the correction itself is **0.73 ns**, against a
   6 ns residual.
2. **Split-half.** Fit every bunch twice, on its odd and on its even triggers —
   two independent estimates of the same quantity, so `var(k_odd − k_even)` is
   fit noise and `cov(k_odd, k_even)` is the real drift. Result: **ρ = +0.996**
   against |ρ| < 0.09 on 200 label shuffles, splitting into **0.92 ppm of real
   drift against 0.06 ppm of fit noise**. The drift is real.
3. **In-sample vs cross-validated.** 6.5 ns vs 6.8 ns at 1–3 ms, 5.7 vs 6.0 at
   40–80 ms. That 3–5 % gap *is* the overfitting, measured rather than argued —
   and every number in this document is the cross-validated one (parameters from
   the odd-numbered triggers of a bunch, applied to *all* the even-numbered
   ones, matched or not).
4. **Wide-window invariance — the decisive one.** In a window far wider than the
   whole drift envelope the correction cannot change *who* is matched, only
   where inside the window they land. Efficiency at ±500 ns: **96.0405 % before,
   96.0405 % after**. Identical to five decimal places. The fit concentrates
   matches; it does not create them.
5. **The parameters are bunch-specific.** Give each bunch a *different* bunch's
   fitted (δa, δk) — same numbers, same distribution, wrong bunch — and the
   efficiency at ±25 ns falls to 65 %, *below* the 74.6 % of no correction at
   all. Generic noise could not do that.

The complementary test — fit the clock on the accidental stream and see what it
invents — **cannot be run**, and that is itself the answer: a bunch has 0.4–0.5
accidental candidates within ±200 ns of its predictions, against the 20 the fit
requires. The population the clock is fitted on is 99.95 % real coincidences.

Consistently, applying the correction leaves the measured accidental rate where
it was (0.049 % → 0.049 % at ±25 ns, unchanged to < 0.003 points on either
sub-run).

---

## 6. Re-deriving this for a new run pair

From `match_study/scripts/` (they import `study_common` by name, so `cd` there):

```bash
python build_candidates.py <subrun> --chunk 250   # ~7 min per sub-run; the only slow step
python fit_timebase.py                            # K, T0, per-arm offsets
python fit_perbunch.py                            # δa_b, δk_b, cross-validated
python window_scan.py --timebase perbunch         # efficiency / accidental / ambiguity
python recommend_window.py                        # confirm the knee is still ~24 ns
python align_survey.py --nb 250                   # the n_TOF internal alignment
python bias_check.py                              # the five tests of §5
python make_figures.py && cd ../latex && make
```

`validate_fast.py` proves `ntof_dream_merge/fast_singles.py` reproduces
`dream_trigger.singles_candidates` bit for bit — run it if either changes.
`fast_singles` exists because the original is O(N_hits × N_bunches) and cannot
run on 2061 bunches.

### Getting the constants into code

```python
from ntof_dream_merge.calibration import load
cal = load()                                  # refuses a different run pair
t_pred = cal.predict(t_dream, arm)            # + (δa_b, δk_b) if you have them
ok = abs(t_cand - t_pred) < cal.window_ns     # cal.bands for band-shaped code
```

`calibration.py` reads the export in `nTof_x17_DAQ/calibrations/dream_ntof/`,
falling back to `match_study/data/timebase.json`. It refuses by default to hand
you a calibration for a run pair it was not fitted on — a silent mismatch there
is a 1.35 % rate error, 45 ns at 3 ms and 1 µs at 80 ms, which no window catches
and nothing else flags.

**Five scripts still carry the pre-calibration constants inline** and are
flagged in place with a `STALE CONSTANTS` banner: `eval_singles_matcher.py`,
`fake_trigger_study.py`, `plot_plastic_amplitude.py`,
`mm_activity_crosscheck.py` and `../ntof_processing/dream_regression.py`. They
are left that way so the numbers they have already published stay reproducible.
**Anything re-run for physics must take its calibration from `calibration.py`**
— `mm_activity_crosscheck.py` in particular, whose arm assignment is exactly
what §7 depends on.

### Traps that will bite

- **`ntof_io`'s caches are keyed by run number only.** An official and a
  reprocessed run 224572 sharing a cache directory silently mix. Use
  `study_common.use_variant()`, which gives the candidate its own
  `ntof_io.variant_cache()` fingerprinted on the file set (`ntof_processing/
  REVIEW.md` §5).
- **`repair_tflash` defaults to True in `ntof_io`.** It is built for the broken
  *official* flash finding. `fast_singles.REPAIR_TFLASH = False`; keep it that
  way for any reprocessed file, and be aware that a stray True invalidates any
  comparison against the published v12 numbers.
- **Never `hadd` an n_TOF run.** Read the partials —
  `ntof_io.ntof_paths()` chains `run<run>.parts/run<run>_NNNN.root`.
- **Re-fit, do not transport.** §3.
- **Cross-validate anything fitted on matched events.** §5.

---

## 6b. Who uses this in production

`../ntof_processing/slim_pipeline/` re-fits everything in §2a **per (DREAM
sub-run × n_TOF run) segment** from that segment's own candidates — it reads no
constant from this document, by design, since none of them transfer. It lands on
the values here to 0.24 % in `K` and 0.25 ns in the per-arm offsets, which is
the best available cross-check that both the recipe and this table are right.
The slim it writes carries its own `calibration.json` per segment.

## 7. For the Micromegas integration

The MM cross-check keys on *which arm* fired. At ±25 ns the two-arm ambiguity is
0.15 % (against 0.28 % at ±150 ns) and the accidental contamination of the
matched sample is 0.049 %, so both of the things that blur an arm-vs-chamber
correlation are an order of magnitude down.

A 6 ns match residual also means the n_TOF wall time is usable as an **absolute**
time reference for the MM drift window — at the 30–40 ns level of the old window
it was not. And the liquid leg is already on the wall time base to under a
nanosecond (§2b), so it joins with no further calibration.

Reconstruct MM geometry from waveforms via `wft/`, never from `combined_hits`
times — see the repository `CLAUDE.md` and `RECONSTRUCTION_BASIS.md`.

---

## 8. Provenance

| | |
|---|---|
| DREAM | run_79, `stat090_0000` (bunches 146–1157) + `stat090_0001` (1165–2213) |
| n_TOF | run 224572, `v12_liqpileup`, 16 partials, never merged |
| local copy | `/media/dylan/data/x17/ntof_reproc/v12_liqpileup/` |
| EOS | `/eos/experiment/ntof/data/x17/reproc/v12_liqpileup/completed/224572/` |
| triggers | 213 420 non-flash, 2061 bunches |
| stored `tflash` | used as-is, repair off |
| date | 2026-07-30 |

Superseded documents are in `archive/` here and in
`../ntof_processing/archive/`; each names this file as its successor.
