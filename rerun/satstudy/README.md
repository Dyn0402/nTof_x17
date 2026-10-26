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
| time extent of every production candidate (particle vs noise band); multi-track accounting + x↔y pairing | `multitrack.py feat`, `multitrack.py count` | `cand_feat_det3.parquet`, `multitrack.json` |
| head-on rule (`reco_variant.py --headon-sig 20`) on the failures + 400 controls, and on 742 empty-chamber triggers (M3 > 20 mm outside det3) for MIN_STRIPS 2 / 3 / head-on | `reco_variant.py`, `headon.py` | `reco_ho20/`, `reco_out_{min2,min3,ho20}/`, `headon.json` |
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
- "Found, cut" (2.1 %, n_dropped > 2) = 69 second-particle events (1.07 %: the M3 particle was picked, a separate MIP
  in the view) + 67 edge noise bands (1.04 %: low clusters lasting the whole 32-sample window on x < 6 mm / y > 355 mm).
  A particle lasts ≤ 24 samples (drift 14 + pulse); not counting the bands recovers all 67.
- Multi-track credit (Dylan 10-26): single-particle events 95.1 % of the interior, 96.9 % detected (band-blind cut);
  multi-particle 2.9 %, M3 track found in both views in 80 %; all interior 91.2 → 94.7 % with any-candidate credit.
  x↔y pairing of two tracks by charge alone: 96 % right (98 % + depth profile); production's beam-tuned gate is not
  usable on the bench (pairs 62 % of clean singles).
- Head-ons: fit_failed is |tan θx| < 0.03 (82 %). Plain MIN_STRIPS = 2 adds 12 fake view-fits on 742 empty-chamber
  triggers; the head-on rule (2-strip cluster admitted if both strips ≥ 20σ) recovers 17/17, 0/400 controls move, 0
  fakes (bit-identical to production there). Head-ons sit at 55–320σ on the 2nd strip, fakes ≤ 9σ.
