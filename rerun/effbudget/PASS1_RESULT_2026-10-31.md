# Paper I efficiency, loss budget and A1 on pass 1 (2026-10-31)

PASS_PLAN_2026-10-28 §4 efficiency / loss items on the pass-1 bench tables (`Analysis_pass1`, read only): the loss
budget and the A1 fractions, the det7 X ch 256–511 cut, the interior ≥ 30 mm fiducial, the fine (8 mm) efficiency
maps, the trigger-track rule as a selection, and the noise-band census. Code `rerun/effbudget/pass1/` (deployed
`code_e383a27b` → `fd127cd9` for the budgets → `89000668` for one row); report `rerun/effbudget/pass1/report/report.html`
(figures are git-ignored: full copy with figures on EOS `/eos/user/d/dneff/x17rerun/paper1_eff/report/`).

## Verdict

**Golden-row detection efficiency, interior_good, R = 5 mm** (both views fitted, raw-frame |r| ≤ 5 mm from the M3
ray, ≤ 2 competing strips per view; interior = ≥ 30 mm from X edges and ≥ 8 mm from the passivated Y edges; det7: X
footprint on ch 256–511):

| chamber | golden row | HV resist / drift | n | **eff** | R = 3 mm | x view | y view | rule | credit | interior (all X) | edge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| det2 | o22_long_det2 | 495 / 1000 | 3 395 | **90.3 ± 0.5** | 89.0 | 92.5 | 91.8 | 90.3 | 93.9 | 90.3 | 87.5 |
| det3 | sat_det3 | 490 / 1000 | 6 454 | **92.6 ± 0.3** | 90.9 | 93.8 | 93.8 | 92.7 | 95.2 | 92.6 | 85.1 |
| det4 | g_det4 | 495 / 600 | 11 482 | **40.9 ± 0.5** (live bands: 71.3) | 39.2 | 43.7 | 44.7 | 40.9 | 44.4 | 40.9 | 44.4 |
| det6 | g_det6_long | 495 / 700 | 8 358 | **74.4 ± 0.5** | 73.6 | 76.9 | 77.4 | 74.4 | 77.7 | 74.4 | 83.5 |
| det7 | g_det7_long | 495 / 700 | 3 920 | **68.0 ± 0.7** | 66.4 | 71.6 | 69.6 | 68.1 | 71.6 | 62.9 | 29.1 |

The golden_quoted rows agree: det2 6-22 long_run 90.9 ± 0.2 % (17 568), det3 6-27 p2 long_run 92.2 ± 0.2 % (20 088).

- **What limits each chamber (A1, interior_good):** det2/det3 lose ≤ 1.7 % to any one class, the largest being
  second particles (multi). det4: dead / low-gain stripes (34 % of the interior). **det6 and det7: discharges — 17 %
  and 24 % of interior tracks are spark-vetoed events whose waveforms are a whole-plane discharge** (> 150 strips at
  SNR 6 or chip-wide ringing). This is the dominant det6/det7 loss at 495 V and grows steeply with HV (figure).
- **det3 A1 moves on the pass, as expected:** 91.18 → 92.64 % detected. Noise bands 1.43 → 0.22 % (band censoring now
  in the chain), head-on 0.26 → 0.02 %; spark sub-classes, multi and δ classes are unchanged within 0.1–0.2 pt.
- **Trigger-track rule as a selection: do not adopt.** Picking the most vertical candidate per view is right more
  often than production on det2/det3 multi-particle views (det3 x 111 vs 90 of 170, y 115 vs 87 of 166) but wrong
  more often on det4/det6/det7 (det4 x 144 vs 226 of 254), and the event-level change is −0.04…+0.05 pt on every golden
  row: the multi-particle views mostly fail the competing-cluster cut anyway. Multi-track credit (uses M3, not a
  selection) would add 2.6–3.6 pt.
- **Noise-band census (all 156 rows):** fraction of triggers with a band, median per chamber: det2 5.8 %, det3 3.1 %,
  det4 3.3 %, det6 3.6 %, det7 15.8 %; golden rows det2 7.1, det3 4.1, det4 3.3, det6 6.6, det7 21.7 %. Rates rise with
  resist HV on det3/det6/det7 (det7 fixed-patch bands, as the 10-26 study); det4 is flat.
- **Fine maps:** the 8 mm disk resolves det4's ~12 mm stripes, which a 25 mm kernel smears into a uniform ~40–75 % band
  (`report/figures/effmap_det4.png`); det7 shows the dead x < 60 mm region, a dip at x ≈ 192 mm and the low band at
  y ≈ 100–170, x > 200 mm.
- **Angles on interior_good (03_angles `--fiducial interior_good`, all rows):** golden s68 x/y det2 1.03/1.08°,
  det3 0.98/0.90°, det4 2.41/2.08°, det6 1.26/1.51°, det7 1.23/1.46°. **These depend on v(t)** for g_det4, g_det6_long,
  g_det7_long (VT_PLAN_2026-10-30): quote only after the correction. The efficiencies do not depend on v.

## Bugs / inconsistencies found

1. **budget.py: event ids repeated across acquisitions were counted as losses** (`not_seeded`) instead of leaving the
   denominator. One pass row has them (det3 6-22 test short_run, skeptical: 96 table rows, 16 interior rays). Fixed
   (`n_ambiguous_id`), row re-run.
2. **budget.py read the C22 classes implicitly from `~/x17/cosmic_bench/outliers`** (pre-pass classifications on the old
   table). Now explicit (`--classified`); the pass budgets carry none, A1 rebuilds C22 on the pass tables.
3. **Skeptical det3 6-23 rows: efficiency is biased by their broken angle scale.** Their folded v (1.7–3.5 µm/ns,
   known, PASS1_STATUS) puts 5–12 % of right-position fits at |tan| > 3, which `compat.apply_fit_sanity` turns into
   fit_failed. "Skeptical = positions/efficiency only" therefore does not hold for the efficiency of these rows as the
   table counts it; the budget now carries a position-only column (`posonly_*`, no sanity cut). Not a pass-product
   bug; a stated caveat for anyone quoting them.
4. The candidates sidecar's `plausible` flag is False on ~16 % of bench production winners (beam-tuned gate, known
   from multitrack.py); any downstream selection on candidates must not use it on the bench.

No inconsistency found in the golden or consistency rows: the production-pick closure (sidecar rank 0 through the
same detection rule = the budget's `detected`) is exact on every row after fix 1.

## What was run

| step | where | output |
|---|---|---|
| budget (156 rows), trigger-rule / credit / position-only arms | condor `~/x17rerun/paper1_eff/bud2` (+`bud3`) | EOS `paper1_eff/budget/<rel>/` |
| 03_angles `--fiducial interior_good` (156) | condor `paper1_eff` | `paper1_eff/angles/<rel>/` |
| band census, ≤ 4000 triggers/file (156); all triggers on the 7 golden rows | condor `paper1_eff`, `bfull` | `paper1_eff/bands/`, `bands_full/` |
| A1 waveform side on the 7 golden rows (C22 classes, sparks, candidate time extent) | condor `paper1_eff` | `paper1_eff/a1/<rel>/` |
| collect, a1_recat, fine_maps, make_report | lxplus (lock venv) / local | `rerun/effbudget/pass1/results/`, `report/` |

Shared fiducial definition: `common/mx17_fiducial.py` (budget refactored onto it, bit-identical regions on 10⁶ random
rays); `03_angles.py --fiducial {all,active,interior,good,interior_good}` (default `all`: the pass outputs are unchanged).
04_maps was not changed: its maps come from the 02 box; the paper maps should come from `fine_maps.py`, whose
denominator is the budget's.

## What this does not rule out

- far:POS_TAIL (bent M3 tracks, C22) is counted as a chamber loss; the M3 reference is not perfect.
- det4's dead-band map (interior_live) is the pre-pass C22 live-ratio map (hardware, same raw events), not re-derived.
- C22 classes are rebuilt without the M3 cluster trees: only the POS_TAIL / M3_ELSEWHERE *explanation* is missing.
- The discharge class uses the 10-26 sparks.py rule; a large negative band offset could in principle meet its "100
  strips below −10 σ" arm. On det7 golden only 5 % of the spark-vetoed events carry a band (20 % of all triggers), so
  bands are not driving it.
- The row-level band census is sampled (~30 % of a long run's triggers); rates, not per-event flags, are quoted there.
- The A1 table for det2/det4/det6/det7 is new: the class rules were tuned on det3 (10-26) and were not re-validated
  by eye on the other chambers' displays.
