# TRACK_PLAN_05 — the in-situ calibration loop (M3's replacement, closed)

**Once PLAN_04 links exist, every calibration the bench did against M3 can be
redone in situ against inter-chamber lines. This plan defines the loop, its
storage, and the monitoring that keeps tracking honest across a changing
campaign (HV scans, gas changes, mesh configs). Deliverable:
`calibration_store/` (one JSON per (run-era, detector)) + `monitor.py`.**

## Calibration quantities and how each is obtained

| quantity | day-1 source (no links yet) | converged source | bench recipe it mirrors |
|---|---|---|---|
| v_drift(E) | Garfield table for the run's gas (`garfield_sim/results/*_CERN_450m.json`) at E = drift_HV / gap | extent-slope/T_sat with (a) frozen_rs regressed angles (validated <1 % at 700–1100 V) and (b) link angles; require (a)≈(b) | scripts 21/35 |
| T_sat, recorded column | measure directly (median duration of inclined candidates); column = v·T_sat | same, per HV point | 21 |
| angle model (regression + sign) | frozen bench model + restandardize | RETRAIN `train_tan_regression` on link-angle truth (exact same code path as `validate_transfer_bench.train_model_for`, with tan_ref ← link angle); priority det6/det7 (C/D — self-training gained ~15 % on bench) | 34/35 |
| angle calibration b (plateau bias) | none (regression is self-calibrated) | median(|tan_det|−|tan_link|) on the plateau, per plane — apply additively in tan space | 28 |
| f_balance (X/Y pairing) | bench A/B values; C/D placeholders | histogram from unambiguous windows per detector per HV | PLAN_38 |
| position σ (for link weights) | bench 0.73–0.94 mm | link residuals, with reference-covariance deconvolution (use the per-link covariance PLAN_04 stores; method = paper PLAN_37) | 36 + PLAN_37 |
| timing σ_t | bench 33 ns | inter-plane Δt0/√2 per detector (telescope-free — works immediately); walk slope vs charge asym | PLAN_42 |
| edge fiducial | 25 mm (bench) | efficiency + angle-residual vs edge distance from link-crossings | 32 |
| attachment λ (gas health) | — | amplitude vs drift depth (z = v·t) of link-tracks; per detector | 17/19 |

## The loop

```
day 1:   Garfield v + frozen_rs models → PLAN_02/03 segments → PLAN_04 links
day 2+:  links → retrain models, fit v in-situ, measure b, f, σ, λ
         → regenerate calibration_store → REPROCESS segments with new
         constants → links improve → iterate (2 passes have always sufficed
         on the bench; stop when Δσ68 < 5 %).
```
Every quantity is stored per **calibration era** = (run range, detector, HV
setting, gas). The segments/tracks tables carry `calib_era_id` so any physics
plot can be traced to its constants.

## Monitoring (runs continuously on the DAQ, cheap)

Per new subrun, on hits only (no waveforms): flash rate, candidate rate,
per-plane occupancy vs dead-strip map, THR-relative noise, f_balance medians,
v_sig from frozen_rs angles (the gas monitor — bench-validated; a >3 % drift
of v_sig at fixed HV = gas change: alert), T_sat, spark fraction per
detector. One row per subrun into `calibration_store/monitor.csv` + a
last-24-h PNG. Bench PLAN_39 says sparks need no rate-dependent correction —
monitor the fraction only as a health metric.

## Acceptance

- Closure: after one loop iteration, retrained-model σ68 on held-out links ≤
  frozen_rs σ68 (it can only improve; if it degrades, the link sample is
  contaminated — tighten link χ²).
- v in-situ vs Garfield: within ~5 % (gas composition uncertainty); LARGER
  disagreement = wrong gas assumption or field error — escalate, don't
  average.
- det6/det7 retrained models reach ≤ their bench self-trained numbers
  (2.6–2.9°) on cosmic links.

## Gotchas

- Regression truth from 2-point links has its own error (PLAN_04 stores the
  covariance): TRAIN on links with small reference error (large Δz, both
  segments high-quality) and always deconvolve when QUOTING resolutions.
- Window truncation: at low drift fields or slow gases, T_sat within ~2× of
  the window end biases v (bench 500 V: +19 %) — flag those HV points; the
  fix is the readout window, not analysis.
- Never mix calibration eras across a gas change (bench det3 wet-week lesson:
  same detector, different week, 3× different v).
