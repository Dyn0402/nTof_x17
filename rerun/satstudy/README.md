# det3 loss categories and the saturated-strip treatment (2026-10-26)

Follow-up of the efficiency budget (`rerun/effbudget`) and the C22 outlier classes (`rerun/outliers`): what each
interior loss category of det3 looks like in the waveforms, whether a reference-free change to the reconstruction
recovers it, and what the forward fit does with saturated strips (Dylan 10-26). Note:
`~/x17/cosmic_bench/satstudy/deck/det3-loss-categories.html`, published as notes/mx17-det3-loss-categories.

Golden row `sat_det3` (6-27 Saturday scan long run, 490 V), production `final_phys` table and phys bundle
(`bundle_det3_phys/`, copied from EOS `Analysis_rr1/<row>/mx17_3/wft/w12/it1/phys/bundle`).

| step | script | output (`~/x17/cosmic_bench/satstudy/`) |
|---|---|---|
| waveforms of saturated events (≥ 3 saturated hits in a view) + controls | `satfit.py extract` | `wf_det3.pkl` |
| refit under five censoring rules (prod / low / both / tail / strip), 350 saturated views + 150 controls | `WFT_PIVOT_START=1 satfit.py fit` | `fits_det3.parquet` |
| where the ADC rail sits, plateau sag, bottom rail | `railstats.py` | `railstats.json` |
| waveforms of every budget category | `examples.py` | `wf_examples.pkl` |
| the "spark" rays split into discharge / muon+other / shower | `sparks.py` | `sparks_det3.parquet` |
| production reco with the spark veto off / MIN_STRIPS = 2 (env below) | `reco_variant.py` | `reco_nospark/`, `reco_min2/`, `reco_min3/` |
| numbers, then the note | `summarize.py`, `make_deck.py` | `summary.json`, `deck/` |

`reco_variant.py` needs the finalize environment:
`WFT_PIVOT_START=1 WFT_COL_GUARD=0.01 X17_BENCH_ROOT=~/x17/cosmic_bench/outliers/stage`.
`satfit.py fit` reproduces the production table only with `WFT_PIVOT_START=1`.

Findings (numbers in `summary.json`):
- The ADC rail is at 3650–3900 per channel (4095 − pedestal − CM), not at `SAT = 3550`. CM moves a clipped plateau by
  ~70 ADC (90 %: 250), and 3.9 % of plateau samples dip under 3550 and are fitted as exact data. 58 % of clipped strips
  then swing to the BOTTOM rail (max − min ≈ 4100 ADC), and those samples are fitted as data too.
- Most saturated failures are a faint MIP line plus one compact saturated blob: a short δ-electron stopping. The fit
  tilts towards the blob and reads too steep in ~90 % of saturated failures.
- "fit failed" = x never seeded: a 2-strip cluster whose third strip is at 7–9 % of the brightest, under the 10 %
  relative floor (MIN_STRIPS = 3). Not a fit failure.
- "spark" (> 50 hits) = 1/4 discharges, the rest a clean muon plus something else, or a shower at the crossing.
- MULTI: in 64 % of wrong views the M3 particle IS a candidate; production picked a brighter, separate particle.
