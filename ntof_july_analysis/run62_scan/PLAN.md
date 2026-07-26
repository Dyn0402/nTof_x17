# run_62 singles-only 2-D drift × resist scan — tracking / drift / optimization

**Run:** `run_62` (2026-07-21 06:06 → 08:16, finished normally), from
`run_config.json`: RAW **singles-ONLY** trigger — `trigger_mode.py scint
--singles` with the **PS / γ-flash leg removed** (M4.D = OR(lemo1 = C-out) only).
**Ar/Iso 90/10, ³He target, neutrons, no Pb filter.** Full readout
(zero-suppress OFF), **64 smp × 60 ns = 3.84 µs**, IPD 90, latency 33.

This is the sibling of `run58_scan`, same reco chain and same deliverables. It
is a separate package (rather than a flag on the run_58 one) so run_58's
published numbers stay frozen; the differences are documented inline in
`scan_lib.py` and summarised below.

## Scan grid (subruns `sng_dr{drift}_r{resist}_{seq}`) — TRUNCATED at 3 h
- drift **700 V**: resist A/B/C 560 → 520 V (−5 V, **9 pts, complete**)
- drift **300 V**: resist A/B/C 560 → 545 V (−5 V, **4 pts**; the last,
  `sng_dr300_r545_012`, ran only 1.5 min before the run ended — 63 events)
- det D resist held **10 V below** the A/B/C setpoint throughout.

Two consequences: the **drift axis is a 2-point comparison, not a scan** (run_58
owns the 7-point 700→200 V drift curve), and the **resist window sits 20 V below
run_58's** (580→540), overlapping only on 560…540 V. run_62's value-add is
therefore the resist axis extended down to 520 V, plus the time axis below.

## What changed vs run_58 — the time axis
run_58 triggered on the γ-flash (PS leg), so the burst leader **was** the flash
and tooth 0 was spent on it. run_62 has no flash trigger, so:

- the burst leader is a **physics single**; `dt_ms` = time since the spill start
  ≈ time since the flash (tooth 0 opens within ~150 µs of the proton pulse);
- the comb has **6 teeth** at 0, 13.6, 27.2, 40.8, 54.4, 68 ms — 5 events in
  tooth 0 (rested-SCA buffer depth at n=64), ~2 in each later tooth
  (measured: `flash_comb/run62_*_spillcomb.json`);
- probe classes are keyed on the **tooth index**, not hand-placed ms windows:
  `early` = tooth 0 (at the flash), `mid` = tooth 1, `late` = teeth ≥ 2
  (≥ 27 ms, fully recovered) — the efficiency probe;
- **leaders and saturated events are reconstructed and kept.** run_58 dropped
  them (its leader was the flash). Here post-flash blindness is the thing being
  measured, so those events stay in the efficiency DENOMINATOR; only
  pathological pile-up (> 60 k hits, ~1 %, essentially all in the last tooth) is
  zero-filled and flagged `reco_skipped`;
- the burst-quality gate is `spill_ok` (tooth 0 complete, ≥ 4 of the expected 5
  events) in place of run_58's `flash_ok` (leader saturated by the flash).

## Deliverables (same as run_58 unless noted)

### A1 `analyze_tracks.py` — yield vs time-since-flash and HV, per detector
`time_recovery.png` (3-class ladder, comparable to run_58), **`tooth_recovery.png`
(new: the full 6-tooth recovery curve, run_62 only)**, `yield_vs_hv.png`,
`gain_vs_hv.png`, `liveness.png`, **`vs_run58.png` (new: the shared drift-700
resist curve overlaid on run_58's — cross-check that the operating point did not
move)**, `per_cell_stats.csv`.

### A2 `analyze_drift.py` — v_drift, effective gap, efficiency vs drift & resist
Same estimators (T_max = P95 of micro-TPC track `tspan_ns`; v = 30 mm / T_max;
D_eff against the dry Garfield 90/10 curve). With 2 drift points this is a
**check** against run_58's curve, not a standalone measurement.

### A3 `optimize.py` — best operating point per detector
Kernel-smoothed efficiency surface + 1σ plateau + FOM = eff × live_frac. The
85 V kernel is far narrower than the 400 V drift spacing, so the two drift
columns smooth **independently**: read the output as "best resist at drift 700"
with a 300 V sanity column. `suggest_setpoint` no longer interpolates the drift
onto the v-saturation knee (run_58 could, because 500 V was a scanned point) —
it reports a measured drift and defers the drift choice to run_58.

## Layout
- `scan_lib.py` — subrun parse, spill/tooth model, reco caching, drift spectra
- `process.py`  — parallel cache builder (`--jobs N`; machine has 6 cores and is
  usually also running DAQ + decoding, so 3 is the safe default)
- `analyze_tracks.py` / `analyze_drift.py` / `optimize.py`
Output → `<ANALYSIS_DIR>/July_HV_Scan/run62_scan/`.
