# v(t) survey cross-check (2026-10-31)

Follow-up to the v(t) survey (`VT_PLAN_2026-10-30.md`, `514f7aac`). Four checks: (1) an M3-scale-free estimator on the
7 flagged bench rows; (2) what the in-row v(t) does to what was calibrated on the golden rows; (3) the missing corrected
tables; (4) a byte-compare of the SPS cache builder across vt_all's edits. Code `rerun/vt_all/crosscheck/`, results
`crosscheck/results/`, figure `crosscheck/figures/vt_crosscheck.png` (also `/eos/user/d/dneff/x17rerun/vt_check/figures/`).
Read-only on Analysis_pass1; no calibration was re-run.

## Verdicts

1. **det2 and det4: confirmed by a timing estimator.** The column duration (no M3, no track model) reproduces the
   ladder on all three rows, and the head-on stack agrees where it has the statistics (det2).
   - **g_det4 (−9.6 %) is real**, not stripes or gain: gain is flat to ±2 % per block, the column standardised against
     the row's own position × amplitude composition moves the same (−8.4 ± 1.4 %), and so does the ≥ 30 mm interior.
   - **The sign flip between 6-23 and 6-24 is real.** Same bundle and template (md5-identical): the absolute column T
     falls 663 → 613 ns through the 6-23 night (v rises), then reads 704 ns at the start of g_det4 1.3 h later (a +13 %
     step; the ladder steps 37.3 → 33.3 µm/ns the same way), then rises to ~760 ns through the day. The column is
     alignment-free, so this is not a remount/kw artefact. The step falls at the reconfiguration (det3+det4 overnight →
     det4 alone), so a gas-flow change is the plausible reading; nothing in the record names it.
   - **det6: the two estimators disagree, and det6's depth correction is not established.** g_det6_long: ladder
     +17.9 ± 2.0 % against column +9.0 ± 2.1 % (standardised +12.5 ± 1.7). longer_run: ladder +15.8 ± 1.6 % against
     column **−24.9 ± 6.4 %**, while the gain doubles within the row (×0.66 → ×1.48). The column is attachment-confounded
     (an electronegative contaminant clearing lengthens the apparent column: vt_within_rows saw it settle by ~4 h); the
     ladder may be brightness-confounded (FA: det6's excess follows the gain). The ladder is still the right correction
     for det6's *angles* (it is the angle scale against M3, whatever moves it). It is not a measurement of the drift
     velocity, so `z_end_vt` in det6's corrected tables is not a depth correction.
   - **det7:** ladder +4.7 ± 1.0 / +6.2 ± 1.3 % against column +2.5 ± 2.4 / +1.1 ± 2.8 % (standardised +3.3 / +1.9,
     saturated class +3.0 / +3.2 ± 1.0): same sign, ladder 1.5–2.5σ higher.
2. **v(t) does not need to go inside the calibration phase.** On the G1 tables, table-level v(t) improves s68 by at most
   0.06° (det4 Y 2.348 → 2.290°), moves the folded kw by ≤ 0.6 % (det4 X; det6 ≤ 1.4 % on the night model), leaks
   ≤ 0.0008 into the bow's scale terms (against 0.003–0.063), and explains −0.1 % of S4's det4 T (−0.9 of 26 ns). The
   fit's w does not depend on v, so the angle correction is exact at table level; only the reco windows use v (open
   risk below). Recommendation: keep the calibration; apply v(t) at table level and re-fold w0/kw per row on the
   corrected mapping (minutes). Doing it inside the calibration instead: det4, det6, det7 at 4–6 h each, then 17 + 25 +
   25 bench rows.
3. **Corrected tables written** for det2 6-22 long_run, det4 6-23 long_run and g_det4, in the det6/det7 location and
   format: `/eos/user/d/dneff/x17rerun/vt_correction/<run>/<subrun>/<det>/{events_vt.parquet, vt_map.json}`. One list:
   `rerun/vt_all/results/vt_corrected_rows.csv` (51 rows; copy `vt_correction/MANIFEST.csv`). The three new tables also
   carry the timing v (`v_rel_col`, `z_end_vt_col`).
4. **SPS cache builder: byte-identical** on ZS run60 ov03 (3000 events, same 3000 keys, every record equal, same hot
   channel 373; peak RSS 10.1 → 0.58 GB). RAW run71 raw275: see §4.

## 1 · Independent cross-check

Estimators, all in the same 4 equal-count time blocks per row (`crosscheck/vtc_measure.py`):

| estimator | what it reads | M3 used for |
|---|---|---|
| **ladder** | vt_bench's kw per block / row kw (an angle scale) | the angle scale itself |
| **column** | FA's colsum: strip-sum box fit T per trigger, clean unsaturated triggers, 20 % trimmed mean; v_rel = T_row / T_block | nothing |
| column, standardised | each trigger's T over the row's median in its (4 × 4 cells of 100 mm) × amplitude-tercile cell | nothing |
| column, sat / interior / head-on | saturated class; both centroids ≥ 30 mm inside; the stack's own events, X view | head-on selection only |
| **head-on stack** | S4's stack fit (x_width.measure) with the re-check recipe (mask, < 3500 ADC), bootstrap errors | head-on selection only |

Lines over the row (Δv, %; error inflated by √(χ²/dof) above 1):

| row | ladder | column | col. std | col. sat | stack | gain over row |
|---|---|---|---|---|---|---|
| det2 6-22 long_run | +4.4 ± 0.9 | +4.3 ± 1.5 | +4.6 ± 1.8 | +3.5 ± 1.3 | +6.7 ± 2.1 | flat |
| det4 6-23 long_run | +8.3 ± 1.8 | +6.3 ± 1.7 | +5.0 ± 1.6 | +1.8 ± 2.0 | +18 ± 7 | flat |
| **g_det4** | **−9.4 ± 1.8** | **−8.5 ± 1.5** | −8.4 ± 1.4 | −4.4 ± 1.9 | −0.5 ± 11.7 | flat |
| det6 6-26 longer_run | +15.8 ± 1.6 | **−24.9 ± 6.4** | −15.6 ± 5.9 | −22.9 ± 7.9 | – | ×2.2 |
| g_det6_long | +17.9 ± 2.0 | +9.0 ± 2.1 | +12.5 ± 1.7 | +10.3 ± 1.9 | – | +21 % |
| det7 6-26 longer_run | +6.2 ± 1.3 | +1.1 ± 2.8 | +1.9 ± 1.9 | +3.2 ± 1.0 | – | flat |
| g_det7_long | +4.7 ± 1.0 | +2.5 ± 2.4 | +3.3 ± 1.4 | +3.0 ± 0.8 | – | +4 % |

- **The stack** has the statistics only on det2 (1416 head-on events) and the det4 rows (≥ 1800). On g_det4 its
  per-block bootstrap error is 4–10 %, so it neither confirms nor contradicts; on det6/det7 the blocks fall below the
  fit's 120-event floor. The column on the stack's own events (X view) follows the all-angle column on det2 and
  det4 (g_det4 −11 ± 5 %, 6-23 +15 ± 8 %, det2 +7 ± 4 %), so nothing angle-dependent separates head-on from inclined
  tracks there (on det6/det7 it has 20–100 events per block: no information).
- **The saturated class responds about half as much on det4** (−4.4 vs −8.5 %, +1.8 vs +6.3 %, 1.7σ each): a box fit
  on clipped strip sums. Quote the unsaturated column.
- det2's column line has χ² 46/2: the survey's step after ~4.5 h, not a line (all estimators see it).
- **Event clock:** colsum keeps only triggers with a cluster (about half the head-on events), so the stack's events are
  placed by interpolating in the global event id, monotonic in time; checked on g_det4 against the pass table's own
  timestamps: |error| ≤ 0.8 s.

## 2 · What it does to the calibration (golden rows, table level)

`crosscheck/vtc_calib.py` → `results/calib.json`. G1 = the CAL4 table G1 scores (`calib/r3`); closure: the constant
mapping reproduces G1 exactly (det4 2.488 / 2.348°).

| row | model | s68 x, G1 → v(t) → v(t)+re-fold | s68 y | kw x / y re-folded |
|---|---|---|---|---|
| g_det4 | line | 2.488 → 2.475 → 2.494 | 2.348 → 2.290 → 2.290 | −0.57 % / −0.02 % |
| g_det6_long | line | 1.455 → 1.422 → 1.420 | 1.618 → 1.596 → 1.578 | −0.15 % / +0.46 % |
| g_det6_long | night | 1.455 → 1.435 → 1.421 | 1.618 → 1.617 → 1.582 | +0.79 % / +1.42 % |
| g_det7_long | line | 2.489 → 2.473 → 2.471 | 1.526 → 1.522 → 1.522 | −0.11 % / +0.05 % |
| g_det7_long | night | 2.489 → 2.472 → 2.472 | 1.526 → 1.521 → 1.520 | +0.14 % / +0.24 % |

- **Bow:** the slope of r(t) on (U, W) over the kw-window events is |du|, |dw| ≤ 0.0008 on every row and view, against
  the blocks' su, sw of 0.003–0.063; the track position is uncorrelated with time (|r| ≤ 0.03). The bow is not biased.
- **S4 (det4):** the two samples (hits-chain selection the pre-pass input used, ref-only sample S4 used) span the same
  time (mean 567.57 vs 567.51 h, same deciles). A pure time-average effect predicts a T ratio of 0.9991, i.e. −0.9 ns
  of the 26 ns move. The time average explains none of it (the S4 follow-up's attribution stands: dead strips and the
  edge zone).
- **Decision for Dylan:** whether to quote G1 with v(t) applied (det4 Y 2.29° in place of 2.35°). It does not need a
  re-calibration.

## 3 · Corrected tables

`crosscheck/vtc_apply.py` (line model, the same columns as vt_within_rows' tables) on the three rows the survey
flagged and vt_within_rows did not cover. Closure: with r = 1 the mapping returns the pass table's own tangent on
99.97–99.99 % of the sane fits (|tan| ≤ 3; the other ≤ 3 events per view not chased); 1–18 wild
fits per row with |tan| > 3 (apply_fit_sanity) are carried as in the pass table.

| row | v_rel (ladder) | v_rel (column) | Δθ p95 x / y |
|---|---|---|---|
| det2 6-22 long_run | 0.978–1.020 | 0.980–1.026 | 0.24 / 0.20° |
| det4 6-23 long_run | 0.957–1.033 | 0.958–1.036 | 0.50 / 0.45° |
| g_det4 | 0.960–1.056 | 0.957–1.056 | 0.65 / 0.57° |

The manifest (`vtc_manifest.py`) lists all 51 corrected rows with their model, estimator, clock and the survey's
verdict. The 48 det6/det7 rows are vt_within_rows' night model (clock 6-26); the three new ones the line model (clock
6-01). Each `vt_map.json` states its clock.

## 4 · SPS cache builder byte-compare

`crosscheck/vtc_cachecmp.py`, the builder at `15900b39` (the SPS audit's seed fix, before vt_all) against `514f7aac`
(+ `4eb667c1` CHUNK, + `b3cc38a6` close-after-read and skip-chunks-outside-open-plateaus), the rest of the code equal,
each row built exactly as `sps_reco.py` does (subruns in order, per_plateau 3000, the plateau only, half_win 16).

| row | events | keys equal | records equal | hot channels | pickled events | peak RSS old → new |
|---|---|---|---|---|---|---|
| run60 ov03 (ZS) | 3000 / 3000 | yes | 3000 / 3000 | [373] both | byte-identical | 10.1 → 0.58 GB |
| run71 raw275 (RAW) | RAW_PENDING | | | | | |

## What this does not rule out / open risks

- **det6's drift velocity over 6-26 is not known to better than a factor ~2**: the ladder and the column disagree (in
  sign on longer_run). Angles are corrected (ladder); depths on det6 6-26 carry this as a stated limitation.
- **det7:** ladder and column differ by 1.5–2.5σ; the column is the noisier.
- **The reco windows use v.** A row whose v moves ±5 % about its mean has a column 5 % longer than the window assumes at
  one end. Not tested here (it would need a reco of g_det4 with v(t) in `u_window`).
- **The stack does not independently test det4** (too few head-on events per block).
- **What changed det4's gas between 6-23 and 6-24** is not identified (no gas log).
- In-sample: the G1 deltas use the row's own v(t) on the row's own events (vt_within_rows' held-out test covers
  det6/det7 only).
