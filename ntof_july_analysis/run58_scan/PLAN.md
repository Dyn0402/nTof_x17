# run_58 singles 2-D drift × resist scan — tracking / drift / optimization plan

**Run:** `run_58` (2026-07-19 23:59 →), confirmed from `run_config.json`:
RAW **singles + PS** trigger, deliberately deadtime-limited, gated by the 30 ms
N93B beam window. **Ar/Iso 90/10, ³He target, neutrons, no Pb filter.** Full
readout (zero-suppress OFF), **64 smp × 60 ns = 3.84 µs**, IPD 90, latency 33,
window sized to contain the full drift column down to 150 V, first arrival
~sample 3.

**2-D scan grid** (subruns `sngPS_dr{drift}_r{resist}_{seq}`):
- drift OUTER (all 4 dets): 700 → 150 V  (on disk so far: 700,600,500,450,400,300,200)
- resist INNER A/B/C: 580 → 540 V (−5 V, 9 pts); **det D held 10 V lower**
- ⇒ 7 × 9 = 63 subruns, ~1800 ev/subrun at high drift, still acquiring (150 V pending).

**Timing model (singles):** each beam pulse opens a 30 ms gate; ~8–9 singles
are accepted per gate in a deadtime comb. Burst leader = first accept (≈ γ-flash,
confirmed by n_big saturated hits). `dt_ms` = time since leader = **time since
flash** — the recovery/saturation axis. Unlike the doubles comb this is
continuous, so bin dt_ms finely rather than using rigid early/mid/late classes.

## Reused machinery
`ntof_tracking.reco` (io, noise, segments, pairing, geometry) — the full
noise-band flag → per-plane cluster+robust-fit → X/Y 3-D micro-TPC pairing →
global-geometry chain. Caching/aggregation pattern lifted from
`ntof_july_analysis/hv_track_scan` (the doubles run_53/55 scan).

## Deliverables

### A1 — Track yield vs time-since-flash and HV, per detector  (the "last resist scan" repeat, now 2-D)
- Cache per subrun: `events.parquet` (per-event burst/dt + per-det n_trkseg,
  n_pair, charges, clean-strip counts, busy flag) and `segs.parquet` (per track
  segment). Driver: `scan_lib.build_subrun_tables`, parallel `process.py`.
- Plots: P(track|trigger) vs dt_ms (fine bins) per det, one curve per resist,
  faceted by drift; and the collapsed **yield vs HV** showing the expected
  **local maximum** (too-high HV → longer post-flash saturation; too-low → inefficient).
- Liveness proxy (median raw hits/event) and gain (segment q_sum) vs HV & dt.

### A2 — Drift analysis: v_drift, efficiency, effective gap vs drift & resist HV
- In the caching pass also accumulate per-subrun, per-det **clean-hit time
  histograms** and **track-segment tspan / anchored-duration** distributions.
- **v_drift(drift HV):** T_max = trailing(cathode) − leading(anode/t0) edge of
  the clean-hit time spectrum; primary model-independent estimate
  `v_drift = 30 mm / T_max`. Cross-check with the upper edge of track tspan.
- **t0_daq (DAQ latency):** the leading edge (≈ sample 3 / 180 ns), ~HV-independent.
- **Effective drift gap:** fit T_max(E) against a 90/10 gas v(E) curve (one free
  parameter = gap). ⚠ the v(E) curve baked into `geometry.DriftModel` is bench
  **95/5** — source a **90/10 Garfield/bench curve** for the gap fit & gain-matching;
  until then report v(gap=30) as primary and the gap fit as provisional.
- **Efficiency vs drift & resist:** relative track-finding yield per trigger
  (fully-recovered late-dt window) as a 2-D surface; low drift field → charge
  loss / shrinking active gap, high → OK until saturation. (Absolute efficiency
  needs an external reference — note as a follow-up via inter-chamber pointing.)

### A3 — Best drift/resist operating point per detector
- 2-D figure-of-merit map over (drift, resist) per det combining: recovered
  track yield (efficiency), post-flash live-window fraction, and drift quality
  (v_drift stability / full effective gap). Report the argmax per detector with
  uncertainties. D treated on its own −10 V resist axis.

## File layout  `ntof_july_analysis/run58_scan/`
- `scan_lib.py`   — subrun parse, burst/dt model, reco caching, drift-spectrum accumulation
- `process.py`    — parallel cache builder (safe to re-run as the scan grows)
- `analyze_tracks.py`  — A1 figures + per-(drift,resist,det) stats CSV
- `analyze_drift.py`   — A2 v_drift / t0 / effective-gap / efficiency
- `optimize.py`        — A3 2-D FOM maps + best-point table
Output → `<ANALYSIS_DIR>/July_HV_Scan/run58_scan/`.
