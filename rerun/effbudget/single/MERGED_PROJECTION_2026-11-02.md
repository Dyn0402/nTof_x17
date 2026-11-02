# Two particles merged in one projection: mechanism and fix (2026-11-02)

**Verdict.** On the bench, the joint two-track fit already sees most of the events where two particles overlap in one
projection, but it refuses all of them. The refusal comes from the beam-tuned `plausible` flag, which the cross-view
trigger and the acceptance guard both require, and from a column-time coincidence that a short secondary cannot pass.

A new bundle-level corroboration mode, **`xy_pairing['split_corrob'] = 'charge'`**, lets a split through when the other
view corroborates it by charge. On the 195 det3 "merged in projection" events:

| | pass 1 | charge mode |
|---|---|---|
| muon credited (within 5 mm of M3 in both views) | 41.0 % | **53.8 %** |
| muon reconstructed as one correctly paired X/Y track | 20.5 % | **36.4 %** |
| on the 53 events where the split is accepted: muon credited | 37.7 % | 84.9 % |
| on the same 53: muon correctly paired, of those credited | | 96 % |
| on the same 53: both particles reconstructed as tracks | | 98 % |

The change was checked against the fleet false-split gate on all five bench chambers. It adds **no split on any clean
single muon** (the `ws_unify` sample) and at most 0.08 % on M3-matched events of any multiplicity. No M3 muon is lost.
The det3 single-track efficiency (96.0 %) cannot move, and the re-reconstruction confirms it does not. **n_TOF stays on
`count`**, the current behaviour, through the same bundle-level switch as the pairing fix. The 4-pitch separation floor
is kept.

## 1 · Mechanism (pass 1, then re-reconstructed from the waveforms)

The waveform re-reconstruction (`merged_rereco.py`) used the production chain on the staged rr1 inputs:

- the 219 one-view events (195 merged + 24 unclear);
- every event of both det3 golden rows whose candidate counts differ between views (106 + 339);
- 600 random events per row.

Variant `p1` (the pass-1 bundles as they are) reproduces pass 1:

- `track_id`, `n_tracks` and split decisions are identical.
- p2 fits are bit-identical; sat fits agree to float noise (|Δp0| ≤ 4e-5 mm).
- 5 p2 events have different degenerate tangents (|tan| ~1e3) in the events table. That is the re-bow ordering of the
  pairing fix on main, not this change (`results/merged_rereco_repro.json`).

What stops the split on the 195 events, in pass 1 (`results/merged_projection.json`):

| | events |
|---|---|
| **no attempt on the merged view** | **127 (65 %)** |
| · the multi view has < 2 *plausible* candidates (the count trigger needs two) | 60 |
| · two plausible candidates, but not coincident in column time t0 + u50 (median 170 ns apart; tolerance 197 ns) | 45 |
| · count condition met, no attempt recorded (residual/width triggers silent, joint fit not re-fittable) | 22 |
| **attempt refused** | **68 (35 %)**, median F 543 |
| · `both_plausible` False (alone in 26) | 44 |
| · `column_shared` False | 27 |
| · `distinguishable` False (the 4-pitch floor) | 20 |
| · F below threshold | 1 |

Two beam-tuned conditions are the core of it.

1. **`plausible`** means the column end is in the u window and raw |tan| < tan_max. Bench secondaries are short (soft
   electrons) and steep (tan spread 0.38), so they fail it. The count trigger needs both of the other view's candidates
   plausible, and the acceptance needs both children plausible.
2. **The column-time coincidence.** A short secondary drifts from part of the gap only, so its t0 + u50 sits far from
   the muon's even though both come from the same trigger. The bench t0 is pinned: the two t0s agree to 0.4 ns (median).

## 2 · The change (`wft/reco.py`, two-track functions only)

The mode is `xy_pairing['split_corrob']`:

- `'count'`: absent means this. It is the pass-1, n_TOF and SPS behaviour, unchanged.
- `'charge'`: validated by `wft.calib.check_xy_pairing`. The worker option `SPLIT_CORROB_MODE` overrides it for A/B runs.

**Trigger.** `_charge_corroboration(f, …)` holds for a selected candidate `f` when all four conditions hold:

- the merged view has fewer candidates than the other view (all candidates, not just plausible ones);
- two other-view candidates share the trigger: |Δt0| ≤ tc tolerance between them and with `f`, phase-corrected;
- `f`'s charge is not above their sum by more than 4 σ of the bundle's ln(qx/qy) balance. It may be below, because a
  one-track fit of a merged window explains only part of the charge (median z = −3.7 on the 195);
- nothing uses `plausible`.

When the trigger holds, the attempt runs at F_corrob (120), with the two candidates' t0 as hints.

**Acceptance of a charge-corroborated attempt** (`accepted_charge`) needs all of:

- F ≥ 120;
- `distinguishable`: the 4-pitch floor, unchanged;
- `column_shared`: unchanged;
- **at least one plausible child** (the muon);
- each child's charge within 3 σ of the other-view candidate it is matched to, over the best of the two assignments
  (σ = balance width ⊕ 20 %);
- each matched child–candidate pair coincident in column time (tc tolerance), which is the per-particle form of the
  x–y coincidence.

What the `count` rule accepts is still accepted, so the mode only adds splits. The lost-track revert and every other path
are unchanged.

- New columns in `events.splits.parquet`: the children's p0, tan, q and plausible flag (`a_*`, `b_*`); with `'charge'`
  also `corrob_charge`, `cc_*`, `accepted_count` and `accepted_charge`.
- `two_track_ladder` honours `accepted_charge`.
- Bundle writers:
  - `rerun/w12/set_w0kw.py --split-corrob` defaults to `charge` (the bench);
  - `rerun/ntof/ntof_bundle.py` forces `count` and records it.
- Documented in `docs/PARAMETERS.md` §4.
- Tests: `wft/tests/test_split_corrob.py`, 8 tests. Full `pytest wft/tests`: 168 passed, 8 skipped.

What remains refused on the 195 under `charge`, among 101 charge-corroborated attempts:

| reason | attempts |
|---|---|
| geometric guards (`distinguishable` and/or `column_shared`) | 77 |
| no plausible child, or F < 120 | 14 |
| child charge / column-time mismatch | 10 |

Plus 33 events with no attempt (11 % do not satisfy the trigger at table level). Two levers were considered and kept:

- **`column_shared`.** Of the 34 attempts that fail only this guard, 21 have an overlap below 0.005 and the rest below
  0.05: the children share (almost) no depth bin. That is the end-to-end cut of one track the guard exists for. Relaxing it would gain
  about 12 muons.
- **`distinguishable`, the 4-pitch floor.** In the attempts that fail it, the merged candidate is usually already within
  5 mm of the muon (69 %, median 2.0 mm). The floor is not what loses these events, and nothing here shows it is wrong.

## 3 · Validation

**(a) The 195 merged events.** Table above, from `results/merged_projection.json`. The `unclear` class (24) does not
move: no split is accepted there.

**(b) Fleet false-split gate, all five chambers.** `merged_gate.py`, `results/merged_gate.json`. Variants `cost` (the
bench baseline after the pairing fix) and `charge` were re-reconstructed from the waveforms on two sets:

- every clean or M3-matched event whose pre-split candidate counts differ between views. These are the only events the
  rule can touch, so the rates are exact over the full populations.
- 500 random clean singles per chamber (control: 0 changed).

The clean-single sample is the `ws_unify` definition: interior M3 ray, production track within 3 mm in both views.

| chamber | clean singles | split rate on clean singles, x / y: cost → charge | charge-added splits on clean singles | M3-matched (any multiplicity) | charge-added splits there | muon lost (5 mm) | moved 3 → 3–5 mm |
|---|---|---|---|---|---|---|---|
| det2 | 3021 | 0.000 / 0.000 → same | **0** | 3132 | 2 (0.06 %) | 0 | 0 |
| det3 | 5869 | 0.017 / 0.017 → same | **0** | 6030 | 4 (0.07 %) | 0 | 0 |
| det4 | 4506 | 0.022 / 0.022 → same | **0** | 4924 | 4 (0.08 %) | 0 | 1 |
| det6 | 6149 | 0.000 / 0.033 → same | **0** | 6418 | 1 (0.02 %) | 0 | 1 |
| det7 | 5434 | 0.018 / 0.000 → same | **0** | 5789 | 1 (0.02 %) | 0 | 0 |

The `ws_unify` false-split rate is unchanged on every chamber and view. The splits the rule adds on matched events are
events with a second candidate in the other view. They raise `n_tracks` 0→1 (5 events) or leave it. The muon stays within 5 mm in all 12: in 4 the split-view
muon is closer to M3 than before, in 6 it is farther (2 of them move from < 3 mm to 3–5 mm).

**(c) n_TOF (run_145 beam, run_149 cosmics).** `merged_ntof_check.py` replays the trigger on the pass-1 n_TOF test
tables (`results/merged_ntof_check.json`). Events where `charge` would corroborate a split and `count` would not:

| table | share of events |
|---|---|
| run_145 A | 4.0 % |
| run_145 C | 13.3 % |
| run_149 A | 1.8 % |
| run_149 C | 3.1 % |

In nearly all of them, an other-view candidate is implausible, which on beam mostly means outside tan_max. That would be
a material change to beam tables with no two-particle truth to check it. **n_TOF stays on `count`:**

- the n_TOF bundles carry no `split_corrob`;
- `ntof_bundle.py` writes `'count'`;
- no stage-3 product changes.

IPC pairs at small opening angle have the same structural weakness (an implausible member blocks the corroboration).
Whether beam needs its own version of this rule is open.

**(d) det3 single-track efficiency.** `merged_effcheck.py`, `results/merged_effcheck.json`. Over every event the rule
can touch, outside the one-view classes:

- the M3 credit changes on 0 events (sat 553, p2 583);
- the rank-0 detection changes on 0 events.

All movement is inside "merged in projection", which the v2 single-track sample excludes. So **96.0 % is unchanged
exactly.** In the all-tracks numbers:

- credit gains 25 events (+0.10 pt over 25 588 tracks);
- rank-0 ("production") detection gains 8 and loses 2 (+0.02 pt).

## 4 · What needs re-running

- **Nothing changes until a bundle carries `split_corrob='charge'`.** The pass-1 tables on EOS (`Analysis_pass1`) were
  made with `count` and stay valid.
- **Unlike the pairing fix, this cannot be replayed from the tables.** A split needs the waveforms, so bench products
  change only at the next bench re-pass:
  - `set_w0kw` bundles will carry `charge`;
  - a re-pass reusing the pass-1 bundles needs `"split_corrob": "charge"` (and `"select": "cost"`) added to each bench
    `calib_bundle/bundle.json`.
- **What moves:**
  - two-particle counts (`n_tracks`, `track_id`, `n_splits`);
  - the "merged in projection" credit in `single_track_v2.py` and Paper I §5 (44 % there, by the A1 credit; 41.0 → 53.8 %
    by the 5 mm credit used here);
  - all-tracks efficiency, +0.1 pt;
  - the events-table winner on the split events (10 sat + 47 p2 of the 1 626 re-reconstructed).
- **What does not move:** the single-track efficiency, angles, alignment and maps.
- **n_TOF / SPS:** nothing.

## 5 · Paragraph for Paper I (§5, multi-particle events)

> When two particles cross the chamber close together in one coordinate, that projection holds a single merged cluster
> while the other shows two. The reconstruction is aware of this case. A cluster in one view that is time-coincident
> with two clusters in the other view, and whose charge is compatible with their sum, is offered a joint fit of two
> straight tracks in the same forward model. The second track is accepted only when it is individually significant,
> separated from the first by at least four strip pitches (the reach of the resistive charge sharing), when both tracks
> share the drift column, and when the two tracks' charges match the two clusters of the other view. That last
> condition uses the X/Y charge balance of a single particle, which is known to 6 %. On the det3 cosmic data, 195 events
> have a muon identified by the reference telescope and a second particle merged with it in one projection. The muon is
> recovered within 5 mm in 54 % of them (41 % without the joint fit), and correctly paired between X and Y in 36 % (21 %).
> When the joint fit is accepted (27 % of these events), the muon is found in 85 % and both particles are reconstructed
> as X–Y tracks in 98 %. The remaining losses are pairs closer than four pitches in the merged projection, or a
> secondary whose fitted column does not overlap the muon's in depth. Neither can be resolved with this readout. On
> clean single muons the procedure adds no split on any of the five chambers.

## 6 · Files

Scripts, in `rerun/effbudget/single/`:

| script | what it does |
|---|---|
| `merged_rereco.py` | variant re-reconstruction (pairing_rereco harness) |
| `merged_compare.py` | per-event scoring of the one-view events |
| `merged_gate.py` | fleet gate |
| `merged_effcheck.py` | single-track efficiency check |
| `merged_ntof_check.py` | n_TOF trigger replay |

Results, in `results/`:

- `merged_projection.json`
- `merged_gate.json`
- `merged_effcheck.json`
- `merged_ntof_check.json`
- `merged_rereco_repro.json`

`merged_compare.py` reads `results/oneview_events.parquet`, which `single_track_v2.py` writes and git ignores. Regenerate
it, or pass `--oneview`.

The working tree is `$WORK_DATA/claude-jobs/mergedproj/`. It holds the re-reco outputs and gate tables. The staged inputs
(22 GB, all on EOS) are deleted at the end of the session.

## What this does not rule out

- The truth tags one particle (the M3 muon). For the second particle, "both reconstructed" means its multi-view cluster
  is in a track whose merged-view member is not the muon's. Its merged-view position is not independently known.
- One chamber's merged events (det3, one gas, 490 V). The other chambers enter only through the false-split gate.
- The rule cannot remove a false split that also happens to match the other view's charges. The gate bounds that at
  ≤ 0.08 % of matched events, but it does not tell them apart from real pairs.
- 33 of the 195 still get no attempt, and 77 attempts are refused by the geometric guards. Those losses stay in the
  merged class.
