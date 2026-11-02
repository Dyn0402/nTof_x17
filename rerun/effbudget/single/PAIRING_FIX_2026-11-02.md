# X/Y pairing of two particles on the bench: mechanism and fix (2026-11-02)

**Verdict.** The bench's wrong X/Y pairings came from `wft.reco.select_tracks`, not from the pairing cost. When any
candidate of the event fails the beam-tuned `plausible` flag, the second particle's pair is never kept, so the cost
repair has nothing to work on. The one surviving pair is then chosen by summed dchi2. The fix is a second selection
mode, **`cost`**: pair first by `xy_pair_cost`, gate after. It is switched on per bundle (`xy_pairing['select']`) for
the bench only. On the 552 det3 events where M3 identifies one particle, correct pairing goes from **80.1 % to 96.7 %**.
Events with one candidate in a view do not move at all, and the events table does not move. **n_TOF is unchanged.**

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

- `gate` re-reco against pass 1: `track_id` and `n_tracks` identical. Fits agree to float noise (max |Δp0| 1e-6 mm).
- `cost` against `gate`: every candidate fit and the whole events table are bit-identical, and `n_splits` is
  identical. The split revert, the one path the replay cannot follow, did not move.
- M3 truth, sat: 101 → **122 / 124**. p2: see `results/pairing_rereco_compare.json`, summarised in §6.

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

## 5 · What changes for which products, and what needs re-running

- **Nothing changes until a bundle carries `select='cost'`.** The pass-1 bench tables on EOS (`Analysis_pass1`) were
  made with `gate` and stay as they are.
- **Bench products that read pairing** (`track_id`, `track_gated`, `n_tracks`):
  - `rerun/calib/saturation/satgate_pairs.py`, `rerun/satstudy/multitrack.py`;
  - `n_tracks == 1` selections in `rerun/paper1_timing/{angle,matched_rows}.py` and `rerun/paper1_floor/analyze.py`
    (−0.06 % of their sample on det3);
  - the "production 80.1 %" statement in `README.md` §2 and Paper I.
- **Bench products that do not change:** efficiency (A1, budget, single-track, radius: §3), angles, alignment, maps.
  These read only the events table and candidate p0/tan of winners.
- **Re-pass needed?** Not for efficiency, angles or single-track results. For the pairing products, the cheap route is a
  table-level refresh: `select_tracks` and the bow re-map are pure functions of the stored candidates.
  `pairing_replay.py --tables` writes the refreshed candidates table, but today only for the two det3 rows. It would
  need generalising to every bench row and the result pushed to EOS.
  - Caveat: the refresh skips the split-revert path. The re-reco subset showed it does not move, but that is not proven
    for every row.
  - The alternative is to wait for the next bench re-pass, whose `set_w0kw` bundles will carry `select='cost'`. A
    re-pass reusing the existing pass-1 bundles would need `"select": "cost"` added to each bench
    `calib_bundle/bundle.json`.
- **n_TOF / SPS:** nothing to re-run.

## 6 · Numbers files

All in `results/`:

- `pairing_replay.json`
- `pairing_m3_all.json`
- `pairing_rereco_compare.json`
- `pairing_eff_check.json`
- `pairing_ntof_check.json`

The working tree (staged inputs, re-reco outputs, cost-mode tables) is `$WORK_DATA/claude-jobs/pairfix/`. It is
regenerable from EOS and can be deleted.

## What this does not rule out

- The 552 truth events are one chamber (det3), one gas, one HV. Other bench chambers have their own `xy_pairing`
  calibration, and the same mechanism applies to them, but their gain is not measured here.
- The remaining 12 wrong pairings are cost failures: similar charges, or a u50 distorted in one view. 6 more events are
  unpaired because the M3 particle's own X and Y fail the column-time tolerance.
- The bench truth tags one particle per event. The second particle's pairing follows from the 2×2, and is not
  separately confirmed for events with three or more candidates.
- n_TOF two-particle pairing (IPC pairs) has the same structural weakness: an implausible member drops the pair. It is
  not fixed here because `plausible` means acceptance there. Whether IPC pairs need a beam-specific version is open.
