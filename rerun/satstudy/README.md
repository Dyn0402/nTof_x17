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
| **noise bands**: per-trigger baselines of every channel (golden row, all triggers; other chambers, 3 000 / file), then rates, x–y coincidence, null tests, in-window decay, cost | `bands.py scan [KEY [MAX]]`, `bands.py report` | `bands_scan[_KEY].npz`, `bands.json` |
| noise bands, follow-up (corruption test on the raw .fdf, where, lifetime, vs discharges, HV scans, n_TOF) | `../bands/` (README there) | `~/x17/cosmic_bench/satstudy/bands/`, `../bands/report.html` |
| band veto in wft (`WFT_DROP_FLAT_HITS=1`) on all 252 interior band events + 400 controls | `reco_variant.py` (env) | `reco_band_flat/` |
| multi-particle sub-classes + fit quality vs singles | `multitrack.py sub` | `multi_sub.parquet`, `multi_sub.json` |
| x↔y pairing: data-driven toy from real singles; the bench overlay with truth on both tracks | `pairing.py toy`, `overlay.py build/score` | `pairing_toy.json`, `overlay/`, `overlay.json` |
| head-on at low gain / doubled noise: injection into real empty-chamber noise, prototype rules (2-strip, 1-strip, column finder, charge check) | `headon2.py extract/scan` | `headon2_wf.pkl`, `headon2.json` |
| head-on rules in wft (`WFT_HEADON_SIG=20 WFT_HEADON_RESCUE=1`): real failures + controls + empty triggers; end-to-end injection | `reco_variant.py` (env), `headon3.py extract/run/score` | `reco_ho_wft/`, `reco_out_ho_wft/`, `headon3.json` |
| joint two-track fit on the multi-particle events (`WFT_TWO_TRACK_FIT=1`) | `reco_variant.py` (env) | `reco_tt/` |
| **saturated blobs** (10-26 late): per-channel rails from the pedestals + block CM; production fit under per-channel top rail / bottom rail censored / both; own optimiser track-only and track + one compact deposit (ΔBIC); the same with the track pinned to M3. Fits on lxplus condor (`~/x17rerun/satblob_code`, EOS `/eos/user/d/dneff/satblob`) | `satblob.py rails / pack / pack_pinned / merge / score`, `satblob_core.py`, `make_satblob_report.py` | `rails_det3.pkl`, `satblob_{track,blob,pinned}.parquet`, `satblob.json`, `satblob_report/` (notes/mx17-det3-satblob) |
| Paper I appendix A1: one display per recat class (PDF) + the class table | `appendix_figs.py` (after `summarize.py`) | `appendix/loss_*.pdf`, `appendix/loss_table.json` → `papers/paper1/figures.toml` |
| numbers (incl. the re-categorised budget, summarize.recat, and the multi-particle flow, summarize.multi_flow), then the note | `summarize.py`, `make_deck.py` | `summary.json`, `recat_det3.parquet`, `deck/` |
| **multi-particle follow-up** (`rerun/multiparticle/`): second particles (timing, closest approach, vertex height, charge, M3 planes), a rule to pick the trigger track, the two-track benchmark (overlay arms off / joint fit), report | `legs.py`, `m3match.py`, `choose.py`, `overlay_arm.py off\|tt` (flock), `bench.py`, `make_report.py` | `multiparticle/` (report/report.html; published notes/mx17-det3-multiparticle) |

Memory: one reconstruction at a time (`--jobs 2`); a reco job peaks at ~3 GB. Never wait on a job with `pgrep -f` /
`pkill -f` on a pattern your own shell contains (it matches itself) — wait on the output file.

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

10-26 pm round (Dylan: bands, MULTI, pairing for the paper, head-ons):
- **Noise bands** = whole-window baseline offsets (±50–3000 ADC, both signs, flat to τ ≫ 100 µs) on a contiguous patch
  of strips, in x and y together (77–95 %), 2–19 % of triggers depending on the chamber. They follow the CHAMBER: det3
  shows the same chips (x 0, y 5–7 = the corner x < 50, y > 250 mm) on FEUs 7/8 and 3/4, five days apart. Null: time in
  run, previous trigger / discharge, SCA cell, any 10–32 ms period, HV currents (|r| ≤ 0.19), common-mode coupling.
  Mechanism not identified (stated limitation). Cost on det3: detection 92.6 → 57 % in the 3.9 % of events with a band.
  `WFT_DROP_FLAT_HITS=1` (hits truncated both sides: 84 % of band hits, 0.7 % of others): 145 → 196 of 252, controls
  400/400 (1 view moved).
- **Multi-particle** (≥ 2 particle clusters in a view, 187 = 2.9 %): M3 one fitted 69, other fitted but M3 found 81,
  M3 not found 37 (30 are projection overlaps: two tracks within the 12 mm seed gap in one view). The M3 particle's fit
  matches single-track resolution (σ 0.61/0.56 mm vs 0.59/0.54; angle 1.06/1.29° vs 1.02/1.00°).
- **Pairing** by charge: toy from 5 882 real singles 95.8 % (q) / 97.6 % (q + depth profile); real 110 events 96.4 /
  98.2 %. Failures only when the two charges agree within ~2σ(lq) = 10 %.
- **Head-on** in wft (opt-in): `WFT_HEADON_SIG=20` (2-strip seeds, rescue-only) + `WFT_HEADON_RESCUE=1` (cross-view
  column finder at the other view's t0, fitted, kept only if ln(q_x/q_y) matches clean tracks within 3 × hypot(0.054,
  σ_Q/Q)). Real data: 17/17, controls 400/400 unmoved, 0 fits added on 742 empty triggers. Injection (headon2): x, noise
  ×2, gain ×0.05: production 44 % → 84 % found, fires ≤ 0.1 % of empty planes; without the charge check 92 % but 2 %.

10-26 late (satblob.py, report notes/mx17-det3-satblob): on the 350 saturated views + 150 controls, agreement with M3
71.1 % (prod) → rail 71.7 (+5/−3 views), bottom rail censored 70.9 (+2/−3), both 72.3 (+6/−2); multi-start track-only
73.4 (+8/−0); track + compact blob 73.4 / 73.7 (both rails, +15/−6); + 5 % model error 70.0. Controls 96 % in every
arm. ΔBIC > 0 adds a blob to 89 % of controls (model error) and no threshold separates. Pinned to M3 the blob model fits
worse in 93 % of failures (×3.6 vs ×1.7 on agreeing views): the deposit is a short δ track (5–10 strips, several
hundred ns), not a point. Raw clip is exactly 4095; 1 176 rail samples < 3550 (CM-pulled plateau) are fitted as data
by production, 521 samples ≥ 3550 are below their rail; bottom rail in 282/350 windows (5 792 samples).
Follow-up (2026-10-26 night): the x↔y pairing simulation study lives in `rerun/pairing_sim/` (PLAN.md §7, report
notes/mx17-xy-pairing-sim): data gates on all five chambers, the MX17_Geant response sim against them (fails σ ln(qx/qy)),
the simulated overlay.
