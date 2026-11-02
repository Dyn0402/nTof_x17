# X/Y pairing of two particles on the bench: mechanism and fix (2026-11-02)

**Verdict.** The bench's wrong X/Y pairings came from `wft.reco.select_tracks`, not from the pairing cost. When any
candidate of the event fails the beam-tuned `plausible` flag, the second particle's pair is never kept, so the cost
repair has nothing to work on. The one surviving pair is then chosen by summed dchi2. The fix is a second selection
mode, **`cost`**: pair first by `xy_pair_cost`, gate after. It is switched on per bundle (`xy_pairing['select']`) for
the bench only. On the 552 det3 events where M3 identifies one particle, correct pairing goes from **80.1 % to 96.7 %**.
Events with one candidate in a view do not move at all, and the events table does not move. **n_TOF is unchanged.**

**Follow-up (same day, §5 and §7).**
- All five bench chambers improve on their M3-tagged two-particle events: det2 76.5 → 96.8 %, det3 81.6 → 95.9 %,
  det4 88.8 → 94.0 %, det6 83.1 → 92.2 %, det7 72.8 → 90.1 %. None is worse; det4 gains least.
- Every pass-1 bench table now has a `cost`-mode copy on EOS, `Analysis_pass1/<rel>/wft/pairing_cost/`.
- No Paper I number moves at its quoted precision.
- The two-track split revert now counts tracks in `gate` (`_n_gated`), so the mode can no longer change a fit.

## 1 · Mechanism (confirmed on the pass-1 tables)

`select_tracks` (mode `gate`, the pass-1 behaviour) works in two steps:

1. It ranks every x–y combination on (production, coincident, both plausible, summed dchi2).
2. It keeps pair 0, and keeps a further pair only if it is coincident **and both members are plausible**.
   `_repair_pairs` swaps y partners only between two *gated* tracks.

On the bench, `plausible` is False for 6.5 % (sat) and 8.3 % (p2) of rank-0 winners. These are real cosmics: in
two-thirds of cases the column end is above the u window, and in a quarter raw |tan| ≥ 0.6. Take two particles A and B
where one candidate is implausible, say Bx:

- The plausible combinations are (Ax, Ay) and (Ax, By).
- Pair 0 is whichever has the larger dchi2. That is (Ax, By) whenever By is the brighter y.
- (Bx, Ay) is then not gated, so it is dropped, and the repair never runs.

When Ax itself is implausible, A ends up either joined to B's segment or in no track at all.

**Replay.** `select_tracks` reads only scalar candidate attributes, all of which the candidates table stores. The
det3 bundles' `xy_pairing` uses no profile feature. So the selection can be re-run exactly on the pass-1 table
(`pairing_replay.py`):

- It reproduces pass-1 `track_id` on **29 443 / 29 443** events.
- It reproduces the bow-mapped tangents on 99.95 % of events.

Mode `gate` on the 552 M3-truth events of `xy_pairing.py`:

| | correct | wrong | unpaired |
|---|---|---|---|
| all 552 | 442 (80.1 %) | 88 | 22 |
| wrong + unpaired, by gated tracks in the event | | 0: 3 · **1: 97** · 2: 10 | |

**97 of the 110 failures have a single gated track**, so the second pair was never kept. In 78 of them a candidate of
the four is implausible. In the other ~20, all four are plausible but the coincidence pattern blocks the second pair.
Example: (Ax, By) and (Bx, Ay) are coincident, (Bx, By) is not, and dchi2 picked (Ax, By), leaving (Bx, Ay)
uncoincident. The pairing cost itself (lq per class, u50, u90, tc) picks the right 2×2 assignment in **97.8 %** of
the 552.

## 2 · The fix: `select_tracks` mode `cost` (`wft/reco.py`, `_select_tracks_cost`)

- **Two or more candidates in both views:** enumerate every matching of k = min(n_x, n_y, 3) disjoint x–y pairs. Take
  the one with the fewest rescued pairs (rescue still adds tracks, never replaces one), then the least summed
  `xy_pair_cost`. At most 10 × 60 matchings.
  - The cost includes the column-time term `tc`, so a non-coincident combination is expensive but not forbidden.
  - Plausibility takes no part in deciding which x goes with which y.
- **Then gate:** each pair is gated exactly as in `gate` (coincident and both plausible), and `n_tracks` counts the
  gated pairs. Only coincident pairs are kept: gated ones first, then by dchi2. A pair that fails only plausibility is
  kept, ungated.
- **One candidate in a view:** the `gate` answer. There is no pairing to decide, only which candidate is the lone one's
  partner. That is a selection, and the cost rule gained 10 and lost 11 of these 135 events.
- **Switch:** `xy_pairing['select']` in the bundle. Absent means `gate`, so every pass-1 bundle and every n_TOF/SPS
  bundle keeps today's behaviour.
  - `wft.calib.check_xy_pairing` refuses unknown values.
  - `rerun/w12/set_w0kw.py` writes `select='cost'` by default, `--xy-select` to change it. This is the bench bundle
    writer.
  - `rerun/ntof/ntof_bundle.py` forces `'gate'` and records it in the provenance.
  - `wft.reco.PAIR_SELECT_MODE` (a worker option) overrides it for A/B runs.
- Documented in `docs/PARAMETERS.md` §4; 18 new checks in `wft/tests/test_multitrack.py`. `pytest wft/tests`:
  160 passed, 8 skipped.

Variants tried and rejected (replay, same 552):

| variant | correct | why not |
|---|---|---|
| greedy without plausibility + 2×2 repair incl. unpaired candidates | 96.6 % | 1→0 track changes on 26 events |
| matching holding `n_tracks` at the `gate` value | 95.1 % | the held count is often a ghost's |
| matching, all events (also one candidate in a view) | 96.7 % | 11 of 135 single-view events made wrong |
| **matching, ≥ 2 candidates in both views (shipped)** | **96.7 %** | |
| keep non-coincident pairs too | 97.6 % | junk pairs with `track_id` ≥ 0; the 5 extra are tc > 197 ns on the M3 particle's own views |

## 3 · Validation

**Replay on the full pass-1 tables of both det3 golden rows** (`pairing_replay.py`, `pairing_m3_all.py`):

| | `gate` (pass 1) | `cost` |
|---|---|---|
| 552 M3-truth two-particle events: correct / wrong / unpaired | 442 / 88 / 22 (80.1 %) | **534 / 12 / 6 (96.7 ± 0.8 %)** |
| transitions | | wrong→correct 76, unpaired→correct 20, correct→wrong 4, wrong→unpaired 4 |
| all 25 197 events with a unique M3 X and Y candidate: correct | 99.37 % | **99.73 %** |
| of which 24 510 with 1 × 1 candidates | 100 % | 100 % (none changed) |
| of which 135 with one candidate in one view | 87 correct | 87 (none changed) |
| events whose pairs change (of 29 443) | | 259 (0.88 %), 0 with ≤ 1 candidate per view |
| `n_tracks` changes | | 19 (0.065 %): 1→0 ×9, 1→2 ×8, 2→1 ×2; `n_tracks == 1`: 24 483 → 24 468 |
| events-table winners (p0, tan, all columns) | | unchanged: winners are re-bowed with the winner pair last |
| non-winner candidate tangents (bow re-map with the new partner) | | 231 events; median \|Δtan\| 0.003, p99 0.022, max 0.28 |

`pairing_rereco.py` re-reconstructed **753 (sat) + 1 147 (p2) events** from the staged rr1 inputs with the full
production chain, twice: once with the bundle as is and once with `select='cost'`. The events were the 552 truth
events, every event the replay changes, and 600 random others per row.

- `gate` re-reco against pass 1: `track_id` and `n_tracks` identical. p2 is bit-identical; sat fits agree to float
  noise (max |Δp0| 1e-6 mm).
- `cost` against `gate`: every candidate fit (1 840 + 3 420) and the whole events table are bit-identical, and
  `n_splits` is identical. The split revert, the one path the replay cannot follow, did not move.
- `n_tracks` changes: 4 (sat) + 15 (p2) of 1 900 events, all among the events the replay predicted.
- **M3 truth: sat 101 → 122 / 124, p2 341 → 412 / 428, together 442 → 534 / 552 (96.7 %).** This is exactly the
  replay's number.

**Efficiency and single-track numbers** (`pairing_eff_check.py`, on the cost-mode tables `pairing_replay.py --tables`
writes):

- The efficiency chain (A1, `budget.py`, `single_track.py`, `radius.py`) reads no `track_id`, `track_gated` or
  `n_tracks`. It reads only candidate p0 (never moved) and tan_theta.
- Rank-0 tangents moved: 0.
- `single_track.py`'s second-particle verdict changes on 0 of 266 + 896 events, so the skeptical single-track
  efficiency does not move.
- `budget.py`'s trigger-track rule (most vertical candidate) picks a different candidate in 0 of 14 249 (sat) and
  4 of 44 442 (p2) views: at most 0.02 pt on the p2 rule efficiency.
- `radius.py` reads the budget rays only: unchanged except those ≤ 4 views.

## 4 · What `plausible` does on n_TOF, and why n_TOF stays on `gate`

On the n_TOF pass-1 test tables (`/media/ucla/work/ntof_pass1/tanmax`, tan_max 0.6, `pairing_ntof_check.py`):

| table | candidates implausible | winners implausible | reason: raw \|tan\| ≥ tan_max / column too short | `cost` would change | 1→0 tracks |
|---|---|---|---|---|---|
| run_145 A (beam) | 37 % | 23 % | 95 % / 15 % | 182 / 1 500 events | 23 |
| run_145 C (beam) | 54 % | 33 % | 90 % / 27 % | 345 / 1 500 | 45 |
| run_149 A (cosmics) | 46 % | 44 % | 96 % / 10 % | 105 / 3 357 | 6 |
| run_149 C (cosmics) | 49 % | 47 % | 94 % / 17 % | 100 / 3 071 | 11 |

On n_TOF, `plausible` is essentially the angular acceptance: the raw-tan limit the registry sets per dataset, plus the
isochronous/flash veto. Beam events carry many such candidates, and `cost` would change 12–23 % of beam events and
cost 2–3 % of them their gated track. There is no two-particle truth on beam to show that this is right, so **n_TOF
stays on `gate`**:

- the n_TOF bundles carry no `select`;
- `ntof_bundle.py` now writes `'gate'` explicitly;
- existing stage-3 products are made from stored candidate tables, so this commit does not touch them.

## 5 · What changed for which products (done 2026-11-02, follow-up)

- **Refreshed tables on EOS, next to the originals.** Every bench row of pass 1 (157 tables: 156 registry rows plus
  `mx17_det3_quick_6-26-26/resist_530V_drift_1200V`) has:
  - `Analysis_pass1/<rel>/wft/pairing_cost/events.candidates.parquet`
  - `Analysis_pass1/<rel>/wft/pairing_cost/events.parquet`
  - `Analysis_pass1/<rel>/wft/pairing_cost/PAIRING_COST.json` (provenance: tool, code commit, the row's bundle, sha1 of
    the source tables, rows changed)

  `../events*.parquet` are untouched. Only `track_id`, `track_gated`, `tan_theta` and `theta_deg` (candidates) and
  `n_tracks` (events) differ.
  - Made with `pairing_replay.py --all-rows --tables`, each row with the bundle its `events.meta.json` names.
  - The replay reproduces pass-1 `track_id` on **215 831 / 215 831** events. A changed candidate's tangent is the
    file's value plus the shift the re-pairing makes in the replay's bow re-map, so the 0.4 % of events whose bow replay
    does not reproduce the file keep the file's values wherever the pairing does not move them.
  - No winner (rank-0) tangent moves.
  - Recorded in `rerun/PASS1_STATUS_2026-10-30.md` and `rerun/PASS1_LEDGER.md` (addendum).
  - To read them, swap `wft/` for `wft/pairing_cost/` in the two table paths. Everything else (alignment, bundles, 02–04
    outputs) stays in `wft/`.
- **Downstream products that cut on pairing or `n_tracks`** (`pairing_downstream.py`, `results/pairing_downstream.json`):
  - `rerun/paper1_timing/matched_rows.py` (the M3-matched single-track widths of all 112 rows with timing):
    - 4 rows change their sample by one event (3 lost, 1 gained, of 53 017).
    - Largest width shift: 0.25 ns in σ_evt and 0.33 ns in σ_xy, on non-golden det6/det7 rows. All are ≤ 0.2 % and well
      inside the bootstrap errors.
    - Golden rows: g_det4 2 109 → 2 110 events, σ_evt 110.92 → 110.86 ns; g_det7_long 433 → 432, σ_evt 45.58 → 45.50 ns.
      The other golden rows do not move.
  - `rerun/paper1_timing/angle.py`: the same two golden rows, ≤ 0.1 ns per bin, plus one event in g_det7_long's
    highest-|tan| bin (n 2 → 1).
  - `rerun/paper1_floor/analyze.py`: its 'topo' sample cuts `n_candidates <= 1` in the view, and `cost` equals `gate`
    there, so it cannot move. Checked: 0 events with `n_tracks` changed and ≤ 1 candidate in a view. The other ladder
    steps do not read pairing.
  - Efficiency (`pairing_eff_check.py --all-rows`, every row with budget rays):
    - The budget trigger-track rule changes its pick in 36 of 416 074 views, and its 5 mm detection in **4 events of
      the whole bench**: 1 on o22_long_det2, 1 on det3 475 V, 2 on the det3 p2 golden row.
    - The single-track second-particle verdict changes in 0 of 13 953 events.
    - A1 and radius read neither pairing nor non-winner tangents in a way that moves.
  - `rerun/calib/saturation/satgate_pairs.py` reads the error-model gate's own A/B arm tables (`<gate tree>/<row>/wft/<arm>`),
    not pass 1. Both arms were made with `gate`, so its comparison is internally consistent. Re-running it in `cost`
    needs those arm tables re-replayed, by pointing `pairing_replay.py`'s `--root` at the gate tree. Not done: it is a
    10-29 gate study, not a pass-1 or Paper I product.
  - `rerun/satstudy/multitrack.py` reads a pre-pass-1 table (`~/x17/cosmic_bench/outliers/final/.../final_phys`). Its
    `paired` metric (gated track_id within R) is "shown but not used for the credit". Its pass-1 equivalent is
    `pairing_m3_all.py`, which is what this report quotes (det3 99.46 → 99.73 %). Recipe if wanted: set `CAND` to
    `Analysis_pass1/<sat_det3 rel>/wft/pairing_cost/events.candidates.parquet` and run `multitrack.py count`.
    `feat` needs the waveforms.
- **Paper I numbers.**
  - Nothing quoted moves at the precision it is quoted (efficiencies to 0.1 pt, timing widths to ~1 ns).
  - The one statement that changes is the pairing claim: production pairs two-particle events correctly in **96 %**
    (det3 95.9 %, 1 509 events over all det3 rows; 96.7 % on the 552 golden-row events), not 80 %.
- **Bundles.** The pass-1 bench bundles on EOS were **not** edited. They describe how the pass-1 tables were made, and
  their absent `select` reads as `gate`.
  - The next bench pass gets `cost` with no action: `rerun/w12/set_w0kw.py --table` (the W0KW node of every
    `make_pass_dag.py` phase) now writes `xy_pairing['select'] = 'cost'` (`--xy-select`).
  - A reconstruction that reuses a pass-1 bundle stamps a copy:
    `rerun/effbudget/single/stamp_select.py <bundle> --out <copy>`, or `--in-place` on a working copy, never on
    `Analysis_pass1`.
  - `rerun/ntof/ntof_bundle.py` forces `gate` for n_TOF.
- **n_TOF / SPS:** nothing to re-run; see `rerun/ntof/NOTE_PAIRING_FIX_2026-11-02.md`.

## 6 · Numbers files

All in `results/`:

- `pairing_replay.json`
- `pairing_m3_all.json`
- `pairing_rereco_compare.json`
- `pairing_eff_check.json`
- `pairing_ntof_check.json`
- `pairing_downstream.json`
- `pairing_rereco_det67_m3.json` (det6/det7: the replay on their gate re-reconstruction)

Tools: `pairing_replay.py`, `pairing_m3_all.py`, `pairing_rereco.py`, `pairing_rereco_tree.py`,
`pairing_rereco_compare.py`, `pairing_eff_check.py`, `pairing_downstream.py`, `pairing_ntof_check.py`,
`stamp_select.py`.

The working tree is `$WORK_DATA/claude-jobs/pairfix/`. The refreshed tables are in `refresh/` (also on EOS); the
staged waveform inputs were deleted after use (they are on EOS).

## 7 · All five bench chambers (follow-up, 2026-11-02)

### 7.1 The M3 test on every pass-1 row

`pairing_replay.py --all-rows` was run on all 157 tables, each with its own bundle; `pairing_m3_all.py` then scored
both modes. The events are those where M3 marks exactly one X and one Y candidate (interior, budget rays). The
two-particle subset is the events with ≥ 2 candidates in both views. Results are in `results/pairing_m3_all.json`.

| chamber | rows | M3 events | all: correct `gate` → `cost` | two-particle n | `gate` correct / wrong / unpaired | `cost` correct / wrong / unpaired | net |
|---|---|---|---|---|---|---|---|
| det2 | 29 | 32 800 | 99.30 → 99.69 % | 634 | 485 / 108 / 41 (76.5 %) | 614 / 14 / 6 (**96.8 %**) | +129 |
| det3 | 63 | 80 230 | 99.46 → 99.73 % | 1 509 | 1 231 / 217 / 61 (81.6 %) | 1 447 / 47 / 15 (**95.9 %**) | +216 |
| det4 | 17 | 10 341 | 99.66 → 99.72 % | 116 | 103 / 9 / 4 (88.8 %) | 109 / 5 / 2 (**94.0 %**) | +6 |
| det6 | 24 | 19 343 | 99.44 → 99.58 % | 296 | 246 / 46 / 4 (83.1 %) | 273 / 18 / 5 (**92.2 %**) | +27 |
| det7 | 24 | 9 243 | 99.02 → 99.32 % | 162 | 118 / 32 / 12 (72.8 %) | 146 / 16 / 0 (**90.1 %**) | +28 |

- In every chamber, no event with ≤ 1 candidate in a view changes: 1×1, 147 015 events; one candidate in one view,
  2 225 events.
- Correct → wrong transitions: det2 5, det3 14, det4 3 (+1 correct → unpaired), det6 5, det7 4. Against them:
  wrong/unpaired → correct of 134, 230, 10, 32 and 32.
- Every pass-1 table: pairing changes on 0.8–1.5 % of events.
  - `n_tracks` changes on 186 of 215 831 events: 1→0 ×94, 1→2 ×40, 2→1 ×50.
  - `n_tracks == 1` goes from 155 114 to 155 030 (−0.05 %).
- **No chamber is worse.** Five rows lose one two-particle event each, all small HV-scan rows or the det4 golden row
  (66 → 65 of 72).
- **det4 is the weakest gain** (+6 of 116). Its golden row loses one (66 → 65 of 72). It also has the most `n_tracks` 1→0
  transitions per event: 27 of 17 809, where a gated ghost becomes ungated pairs. The pairing calibration does not explain
  it. Charge-balance widths (rsig of ln qx/qy: all / clean / saturated): det2 0.074 / 0.064 / 0.079, det3 0.065 / 0.058 / 0.076,
  det4 0.116 / 0.087 / 0.163, det6 0.174 / 0.094 / 0.178, det7 0.308 / 0.075 / 0.346. det6 and det7 are wider and gain more.
  det4 already paired best under `gate` (88.8 %), and its M3-tagged two-particle sample is the smallest (116).

### 7.2 Waveform re-reconstruction per chamber (golden rows)

`pairing_rereco.py`, rr1 inputs staged from EOS. The events were: those the replay changes, the two-particle events,
and 200 random others (det7: 40). Results are in `results/pairing_rereco_compare.json`.

| row | events | `gate` re-reco vs pass 1 | `cost` vs `gate` | M3 two-particle: pass 1 → `cost` |
|---|---|---|---|---|
| sat_det3 | 753 | track_id, n_tracks identical; fits to 1e-6 mm | fits, events table, splits identical | 101 → 122 / 124 |
| p2_det3 | 1 147 | bit-identical | fits, events table, splits identical | 341 → 412 / 428 |
| o22_long_det2 | 284 | bit-identical | fits, events table, splits identical | 53 → **76 / 77** |
| g_det4 | 351 | identical (1 events-table float) | **1 split reverted differently** (event 124486: Y winner 89.6 → 90.5 mm); otherwise identical | 66 → 65 / 72 (replay: same) |
| g_det6_long | 324 | bit-identical | (replay on the re-reco tables) | 80 → 89 / 93 |
| g_det7_long | 204 | bit-identical | (replay on the re-reco tables) | 28 → 36 / 39 |

**The det4 split** is the one path the table replay cannot follow, and it showed up once. The split-revert guard
("a split may not cost the event a track", `_worker_fit`) counts gated tracks through `select_tracks`, so under `cost`
the count can differ.

**Fixed in `wft/reco.py`:** `_n_gated` now always counts with mode `gate`. That makes every fit, split and winner
independent of the pairing mode by construction. `cost` then changes only `track_id`, `track_gated`, `n_tracks` and
non-winner tangents, which is exactly what the pass-1 table refresh (§5) assumes.

From there:
- det6 and det7 were re-reconstructed once, in `gate`.
- Their `cost` answer is the replay on those re-reco tables. After this fix that is the `cost` re-reco by construction.

The det4 `cost` run predates the fix. On the 351 events, the fixed code gives the `gate` fits, which matched pass 1.

## What this does not rule out

- The M3 truth tags one particle per event, on cosmics, in Ar/iso 95/5. §7 covers all five chambers over every pass-1
  row and HV, but det4's sample is small (116 two-particle events), and that is the one chamber whose golden row
  does not gain.
- The remaining 12 wrong pairings are cost failures: similar charges, or a u50 distorted in one view. 6 more events are
  unpaired because the M3 particle's own X and Y fail the column-time tolerance.
- The bench truth tags one particle per event. The second particle's pairing follows from the 2×2, and is not
  separately confirmed for events with three or more candidates.
- n_TOF two-particle pairing (IPC pairs) has the same structural weakness: an implausible member drops the pair. It is
  not fixed here because `plausible` means acceptance there. Whether IPC pairs need a beam-specific version is open.
