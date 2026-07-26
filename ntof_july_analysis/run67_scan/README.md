# run_67 analysis — plastic-threshold × drift/resist HV track-efficiency scan

run_67 (2026-07-22/23) is a PS + SINGLES beam run: a 2-D drift × resist HV scan
repeated at three GEANT plastic-discriminator thresholds (1.41 / 1.13 / 0.90 MIP,
tags `m141` / `m113` / `m090`), mesh ON throughout (B/D are the in-run no-mesh
control). Config: `nTof_x17_DAQ/run_config_hv_mesh_thresh_scan.py`.

Sibling of `run64_scan` / `run61_scan` — same trigger recipe, same reco chain,
same deliverables — but with **two structural differences** (see `scan_lib.py`):

1. **A plastic-threshold axis** (`mip` column, 141/113/90). Every deliverable is
   produced per threshold.
2. **No deadtime comb.** run_58/61/64 accepted in a rigid comb (~5/13.5/27… ms)
   and binned by tooth (early/mid/late). run_67 is a SINGLES trigger inside the
   N93B ~1–81 ms gate with FEU watermark Hwm 2, so the recorded events form a
   **continuous, broad** time-since-flash distribution. Time windows are
   therefore **defined by hand** (`scan_lib.WINDOW_SETS`): a `broad` set
   (1–10 / 10–30 / 30–80 ms) and a `fine` set (nine bins, 1→80 ms). Windows are
   applied downstream from the cache, so retuning them needs no re-reco.

Also: det D resist = A/B/C setpoint (**no −10 V offset**, unlike run_64).

## Pipeline

```
process.py            # per-event reco cache (SLOW, ~8 min/sub-run; run --jobs 2
                      #   on this 8 GB box — 4 jobs OOMs)
run_all.py            # everything below, in order:
  feu_presence.py     #   per-event FEU readout/liveness flags (dropout guard)
  flash_timing.py     #   PART 1
  analyze_tracks.py   #   PART 2 (all dets) + writes per_cell_stats_{broad,fine}.csv
  detA_2d.py          #   PART 2 (Det A raw numbers — the preferred view)
  compare_thresholds.py #  PART 2 synthesis
  slide_plots.py      #   PART 3 (boxcar vs time-since-flash) — uses slide.py
```

Outputs → `/mnt/data/x17/beam_july/analysis/July_HV_Scan/run67_scan/`.

## Deliverables

**PART 1 — `flash_timing/`**: time-since-flash distribution per threshold,
overlaid on the reweighted GEANT in-gate IPC production spectrum (analogue of
`flash_comb/ipc_vs_runs`). `ipc_vs_thresholds.png` (all three overlaid),
`ipc_vs_m{mip}.png` (one each), `timing_summary.csv`. HV-pooled — HV does not
shape the arrival-time distribution.

**PART 2 — track efficiency in drift/resist HV space, per threshold:**
- `tracks/yield_vs_hv_m{mip}.png` — P(3D pair) vs resist & drift, per broad window.
- `tracks/recovery_vs_dt_m{mip}.png` — efficiency + blind-fraction vs fine dt window.
- `detA/detA_2d_raw_{set}_m{mip}.png` — **Det A raw 2-D efficiency, p±err AND n
  per cell, best cell starred** (the operator's preferred plot). One panel per
  window; `{set}` ∈ {broad, fine}.
- `detA/detA_profiles_{set}_m{mip}.png` — 1-D slices with error bars, no smoothing.
- `compare/eff_throughput_vs_threshold_{set}.png` — efficiency AND throughput
  (= eff × events/spill = good tracks/spill) vs threshold, per window, per det.
- `compare/best_points_{set}.csv`, `compare/recommendation.md`.

**PART 3 — `slide/`: sliding-window (boxcar) efficiency vs time since flash.**
`analyze_tracks`/`detA_2d` bin dt into a few hand-picked windows, which is fine
for a 2-D HV map but smears the post-flash recovery — the structure lives in the
first ~10 ms. PART 3 replaces the hand binning **along dt only** with a boxcar:
every event within ±W/2 of a centre, centres stepped along the gate.

- **W = 6 ms, step 1 ms, linear** (operator's choice, 2026-07-25); override with
  `--width` / `--step`.
- **Drift is kept as a full 4th axis, never pooled** (operator's choice).
- The dt acceptance is **measured, not assumed**: `slide.gate_edges` finds the
  real 1→76 ms edge (the nominal gate says 81 ms, and boxes hanging off the true
  edge lose denominator in a way that looks like a physics feature).
- ***Adjacent points are correlated*** — 1 ms step under a 6 ms box means
  consecutive points share ~83 % of their events. Anything narrower than ~6 ms
  in dt is the smoothing kernel, not physics. The bands are per-point binomial
  errors and are **not** independent.

Figures: `slide/recovery/` (eff **and** blindness vs dt, 4 det panels, curve per
resist — the core plot), `slide/thresh/` (three plastic thresholds overlaid),
`slide/drift/` (three drifts overlaid), `slide/maps/` (eff heat-map over
dt × resist), `slide/slices/` (eff vs resist at fixed dt), plus
`slide/slide_curves.csv` — the tidy table behind every figure.

### Beam-pulse intensity (5th axis, `intensity.py` + `slide/intensity/`)

Each event inherits its PS pulse intensity from `ntof_july_analysis/
pulse_match.py` (cluster trigger times at a 0.5 s gap → one cluster per pulse,
fit the clock offset against the beam_watcher log). run_67 is flash-anchored,
one flash per pulse, so a cluster IS a burst IS a beam pulse. This attaches
**downstream of the reco cache**, so adding it needs no re-reco.

July pulses are strongly bimodal (~410e10 and ~850e10), so the split is not a
free parameter: `E10_SPLIT = 600e10`, the same constant as
`track_rate_hv_time_intensity/intensity_split.py` — keep them equal, the two
analyses are meant to be comparable. Measured match_frac 1.000.

- **The bands are lopsided: run-wide LOW ≈ 20.5 %, HIGH ≈ 79.4 %** (unmatched
  0.03 %), and the fraction varies strongly per sub-run — one m090 sub-run is
  12 % LOW. The intensity build therefore defaults to **2× the boxcar width**
  (`--int-width`); check `n` in `slide_curves_intensity.csv` before believing a
  LOW-vs-HIGH difference in any single cell.
- Unmatched events get `iband=''` and are **dropped explicitly**, never folded
  into a band.
- `slide/intensity/intensity_ratio_vs_dt.png` is the physics question: a
  HIGH/LOW ratio flat at 1 means per-trigger efficiency does not care how much
  beam arrived; a dip at small dt means the chamber is less efficient after a
  bigger pulse — recovery scaling with delivered flux, not just with time.

**The pulse-match cache must be rebuilt after a reprocessing.** The maps cached
2026-07-24 10:20 predate the re-decode: the clock fit and every common event's
intensity are identical, but the new map covers 14 355 events where the old one
had 13 638 — the reprocessing gave hits to ~700 previously-empty events, and
those are exactly the marginal low-amplitude ones. Reusing the stale map would
silently drop ~5 % of events, biased against the newly recovered signal. Rebuild
with `pulse_match.match_subrun(..., rebuild=True)`.

**Old-vs-new — `reproc_compare/`** (standalone, NOT in `run_all.py`, since it
needs the backup cache to still exist). Quantifies what the small-pulse
reprocessing bought, against `cache/run_67_preReprocess_20260723/`. A large
efficiency jump is equally consistent with "we now find the tracks we were
missing" and "we now accept noise", so the verdict rests on **segment quality**,
not yield: genuine recovery raises strips/tspan/r² and the X/Y pair fraction
while LOWERING median amplitude (the recovered hits are the small ones); noise
contamination would push r² and tspan down and could not raise the pair
fraction at all, since two orthogonal planes do not agree by accident. It
excludes any sub-run whose new-cache parquet is older than its `combined_hits`
(i.e. not actually re-reco'd yet) — otherwise a half-finished `--force` run
compares files to themselves and reports "no change".

First result (Det A, `m090On_dr500_r530_056`): P(3D pair) 0.008 → 0.119 (14×),
n_strips 7 → 9, r² 0.85 → 0.92, tspan 408 → 650 ns, a_max 238 → 172,
in-pair fraction 0.16 → 0.63. Det B (bad M1) stayed flat — the noise control.

## Metric

Efficiency = **P(3D x/y pair) per recorded trigger** (noise-robust; the X/Y
coincidence kills the common-mode fake tracks that inflate single-plane yield on
the noisy B/C/D M1 cards). **Det A (clean M1) is the reference.** Yields are
RELATIVE (single-arm events dominate → no absolute normalization). Denominator =
events a detector was READ OUT for (`feu_presence.readout_*`), never its
produced-hits flag — post-flash blindness is the inefficiency being measured and
must stay in the denominator; it is reported separately as `blind_frac`.

## Re-reco on the reprocessed hits (2026-07-25)

All 65 sub-runs were re-decoded on 2026-07-24 with a lower pulse-finding
threshold ("find smaller pulses"), which made the 2026-07-23 cache stale — it is
preserved at `cache/run_67_preReprocess_20260723/` for old-vs-new comparison.
The reco chain itself applies **no amplitude cut** (it takes every hit and
relies on the noise-band/isolation filters plus the geometric track-quality
cuts), so the extra small pulses flow through with no tuning change. Three code
defects surfaced and were fixed in the process:

1. **`process.py --force` did not force.** It rebuilt the *task list* but still
   called `build_subrun_tables` without `force`, so every worker reloaded the
   stale cache and "completed" in 0 s. Fixed, and `--force` completion is now
   tracked by what this invocation actually rewrote (under `--force`, an
   existing file is not evidence of freshness, so an OOM restart would
   otherwise have skipped every not-yet-redone sub-run).
2. **`segments.robust_line_fit` weighted by raw amplitude**, and amplitudes can
   be ≤ 0 (baseline undershoot, ~0.1 % of hits). `sqrt` of those is NaN,
   `polyfit` then returns NaN, and the caller's finite-check discarded the
   **whole cluster** — one undershooting strip silently destroyed an otherwise
   good track segment. Non-positive amplitudes now get zero weight. This bit
   hardest on exactly the low-amplitude hits the reprocessing recovered.
3. **~3–4 % of hits carry a diverged pulse-time fit** (`sample` = ±5×10⁶ on a
   32-sample waveform, `time` out to ±0.3 s). This is *not* new — it was 2.9 %
   before the reprocessing and 3.7 % after — but it is destructive: `noise.
   _band_intervals` bins a plane's time range at 30 ns, so one hit at 3×10⁸ ns
   turns a ~70-bin histogram into a 2-million-bin one. It crashed the re-reco
   outright (`IndexError`, noise.py:58) on the m090 block. Now cut at load
   (`io.drop_unphysical`, a sanity bound on `sample` that is independent of a
   run's window length, and it reports the fraction dropped), with the band
   finder hardened as well. **Worth checking on the other reprocessed runs —
   the same hits are in all of them.**

## Data state (2026-07-23)

Complete blocks: drift {600, 700, 500} × resist {550…520, 5 V steps} × mip
{141,113,90} = 63 sub-runs; drift 400 is a 2-sub-run fragment. ~164 spills and
~10 k reco'd events per sub-run at 1.41 MIP (more at lower thresholds); every
fine window holds ≳300 events, so per-cell binomial errors are usable.
