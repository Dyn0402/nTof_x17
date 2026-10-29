# F7 · the 14 channels the GA gain map added to the masks (pre-flight, 2026-10-29)

PASS_PLAN F7 / REPROCESSING_TODO B2. GA (10-28, `model_audit/followups/ga_gainmap`) added 14 channels to the golden-row
masks from its in-fit gain map: det4 X 94 125 196 376 400 435 439 455 472 498 → dead, det4 X 274 → hot, det3 X 62 →
hot, det3 Y 41 / 510 → dead. Dylan asked for each to be double-checked on the hits, as the other masks were, with a waveform
display, and for any that fail to be dropped. Done on the rr1 inputs of the golden rows (sat_det3, g_det4), first 4
acquisitions (43k / 52k events on det4 X).

## Tests
- `ga_check.py`: **flank** (both ±1 neighbours > 300 ADC: fired fraction > 50 ADC, the `flank_census` rule) and the
  **two-sided ratio** A_c / mean(A_L, A_R) on those events, on rr1_cm0 hits and on fa80c06 waveform peaks
  (pedestal + plain block CM, as `wft.io`). Divided by the local median over every judged channel within ±32 strips → q.
  Plus a waveform display of the channel and ±2 neighbours on six flank events.
- `ga_mirror.py`: **mirror ratio** for channels with a dead or rarely bright neighbour. With a live neighbour n as the
  cluster peak, Σ A_c / Σ A_m, where m is n's other neighbour → q per side. Noisy: healthy controls read 0.24–1.36 on one side.
- `ga_display1.py`: one-sided displays for 94 / 400 / 435 / 439.

Rule: **healthy → dropped from the mask**. Confirmed weak, or inconclusive → **kept censored**. Censoring a healthy
strip costs a little information (the fit treats its charge as unknown). A weak strip fitted as live biases the fit.

## Result
| chamber view ch | GA | flank fired | q two-sided (hits / wf) | q mirror (wf, −/+) | display | verdict |
|---|---|---|---|---|---|---|
| det3 X 62 | hot | 1.00 | 1.11 / 1.06 | – / 0.82 | normal | **healthy: hot dropped** |
| det3 Y 41 | dead | 1.00 | 0.98 / 0.98 | 1.00 / 1.34 | normal | **healthy: dropped** |
| det3 Y 510 | dead | 0.57 | 0.00 / 0.04 | 0.04 / 0.23 | flat beside bright neighbours; −4000 ADC excursions | **dead: kept** |
| det4 X 94 | dead | (5 events) | – | 0.42 / – | (1 event) | weak (occupancy 0.2× median): **kept** |
| det4 X 125 | dead | 0.96 | 0.72 / 0.91 | 0.79 / 0.43 | normal | **healthy: dropped** |
| det4 X 196 | dead | 1.00 | 1.01 / 1.00 | 0.65 / 0.91 | normal | **healthy: dropped** |
| det4 X 376 | dead | 0.98 | 0.86 / 0.86 | 0.53 / 1.12 | normal | **healthy: dropped** |
| det4 X 400 | dead | (13 events) | – | 0.21 / 1.29 | ~50 ADC where the mirror strip gets 200–500 | weak: **kept** |
| det4 X 435 | dead | 1.00 (20) | – / 1.01 | 0.49 / – | ~40 ADC beside a 0.5–1k peak | weak: **kept** |
| det4 X 439 | dead | 1.00 (20) | 0.59 / 0.80 | 0.20 / 0.77 | mostly coherent-noise events; one normal | inconclusive: **kept** |
| det4 X 455 | dead | 1.00 | 0.77 / 0.77 | 0.90 / 0.82 | normal | **healthy: dropped** |
| det4 X 472 | dead | 0.99 | 0.55 / 0.59 | 0.65 / 0.62 | near 0 beside 1–4k neighbours | weak: **kept** |
| det4 X 498 | dead | 1.00 | 0.95 / 0.95 | 0.62 / 0.69 | normal | **healthy: dropped** |
| det4 X 274 | hot | 0.93 | 0.97 / 1.01 | 0.73 / – | normal | **healthy: hot dropped** (not a short partner of 271–273) |

**8 dropped, 6 kept.** The masks were written by `apply_f7.py`
(`model_audit/followups/fe_unify2/masks/bench_det{3,4}.json`: `ga_added.f7_dropped` / `f7_kept`). det3 X and det4 X now
carry no hot channel.

Inputs and results: `results/ga_check.json`, `results/ga_mirror.json`, `results/figures/ga_*.png` (EOS copy
`/eos/user/d/dneff/x17rerun/precheck_ga/`).
