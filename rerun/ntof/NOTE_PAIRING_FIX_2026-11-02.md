# Note for the n_TOF re-processing: the bench X/Y pairing fix (2026-11-02)

**For n_TOF this note changes nothing today.** The n_TOF pass keeps the X/Y track selection that every n_TOF product
was made with (`select_tracks` mode `gate`). The bench switched to a new mode (`cost`), because there it pairs
two-particle events correctly in 96 % of cases instead of 80 %. Whether n_TOF should follow is an open question, and
the n_TOF pass should answer it with beam truth (§4). This note is separate from `NTOF_PASS_PLAN_2026-11-02.md`; that
plan is not edited here.

Full bench write-up: `rerun/effbudget/single/PAIRING_FIX_2026-11-02.md` (§7 covers all five bench chambers).

## 1 · What the fix is

`wft.reco.select_tracks` turns the per-view candidate fits into X–Y tracks. There are now two modes, chosen per bundle
by `xy_pairing['select']`, with `PAIR_SELECT_MODE` as an A/B override. A bundle without the key means `gate`.

- **`gate`** is the pass-1 and n_TOF behaviour, unchanged since 2026-09-14.
  - It is greedy on (coincident, both `plausible`, summed dchi2).
  - A track after the first must be coincident **and** have both members plausible.
  - The pairing cost (`xy_pair_cost`: charge balance per event class, u50, u90, column time) can only swap partners
    between two tracks that are both gated.
- **`cost`** is the bench behaviour from now on.
  - With ≥ 2 candidates in both views, it takes the X–Y matching with the least summed pairing cost. Plausibility takes
    no part in deciding which X goes with which Y.
  - Each pair is then gated exactly as in `gate`, and only coincident pairs are kept.
  - With one candidate in a view the result is identical to `gate`.

**The failure it fixes:** in `gate`, one implausible member drops the second particle's pair, so the remaining pair is
chosen by dchi2 and not by the pairing cost.

- On det3, 97 of 110 failed events had exactly one gated track.
- The pairing cost itself picks the right assignment in 97.8 % of the 552 truth events.

**Bench results** (M3 identifies one particle; events with ≥ 2 candidates per view):

| chamber | `gate` → `cost` correctly paired |
|---|---|
| det2 | 76 → 97 % |
| det3 | 82 → 96 % |
| det4 | 89 → 94 % |
| det6 | 83 → 92 % |
| det7 | 73 → 90 % |

Single-candidate events, every fit and the events table are unchanged.

## 2 · Why n_TOF stays on `gate` for now

On n_TOF, `plausible` means something different. On the bench it is mostly the column-duration window. On n_TOF it is
mostly the **angular acceptance**: the raw |tan| < tan_max of the registry, plus the short-column flash veto. Measured on
the pass-1 tan_max = 0.6 test tables (`/media/ucla/work/ntof_pass1/tanmax`, `rerun/effbudget/single/pairing_ntof_check.py`,
`results/pairing_ntof_check.json`):

| table | candidates implausible | winners implausible | reason: raw \|tan\| ≥ tan_max / column too short | `cost` would change | events losing their gated track |
|---|---|---|---|---|---|
| run_145 A (beam) | 37 % | 23 % | 95 % / 15 % | 182 / 1 500 | 23 |
| run_145 C (beam) | 54 % | 33 % | 90 % / 27 % | 345 / 1 500 | 45 |
| run_149 A (cosmics) | 46 % | 44 % | 96 % / 10 % | 105 / 3 357 | 6 |
| run_149 C (cosmics) | 49 % | 47 % | 94 % / 17 % | 100 / 3 071 | 11 |

`cost` would re-pair 12–23 % of beam events and take the gated track from 2–3 % of them. There is no two-particle truth
on n_TOF that says this is better, and stage-3 products and the A–C line work are built on `gate`.

So nothing changes for n_TOF:

- the n_TOF bundles carry no `select`;
- `rerun/ntof/ntof_bundle.py` now writes `select='gate'` explicitly (recorded in provenance as `xy_pairing_select`),
  because the bench golden bundles it copies will carry `'cost'` from the next bench pass on;
- `ntof_tracking/wft_beam.py` writes the mode into each table's meta (`multi_track.xy_select`). Check that it reads
  `gate`.

## 3 · The open question: IPC pairs losing a member to the acceptance flag

The structural weakness the bench showed is present on n_TOF too. Take an e⁺e⁻ pair in one chamber where one member
fails `plausible`:

- in one view, through a steep raw tan (the members of a wide-angle pair are exactly the steep ones);
- or through a short column.

That member's pair is not kept as a track. The pair kept first is chosen by dchi2 among the plausible combinations, so
it may join one member's X to the other member's Y: a ghost.

The `gate_one_pair_with_second_coincident` count in `results/pairing_ntof_check.json` is the number of events where
`gate` kept one pair while a second coincident X–Y combination existed among the other candidates. It is 120 / 1 500
(A) and 255 / 1 500 (C) on run_145. It is an upper bound on the exposure, not a measured loss.

For the X17 analysis this matters in two ways:

- **Lost members:** an implausible member of a real pair is never a track, so the pair is lost.
- **Ghosts:** a wrong X–Y join biases the opening angle and vertex.

Whether either happens at a relevant rate on IPC-like pairs is **not measured**.

## 4 · What the n_TOF pass should decide

1. **A beam truth test before changing anything.** The overlay bench is `sept26_prelim_analysis/intra_bench.py`: two
   clean single-track run_145 triggers of one chamber, summed, then re-seeded and re-fitted by production; truth is each
   donor's frozen fit. The simulation version is `rerun/pairing_sim/sim_overlay.py` (cond `n2` = n_TOF noise).
   - Run it with both modes: `wft.reco.PAIR_SELECT_MODE` is a worker option (`_worker_init(..., opts=...)`), or set
     `xy_pairing['select']` in a copy of the bundle.
   - Score both-tracks-found-and-correctly-paired, ghost rate and single-track loss, against separation and against
     the donors' tans, especially for pairs where one donor has raw |tan| above tan_max.
   - The overlay donors are clean singles, so they pass `plausible` by construction. The test must deliberately
     include steep donors (|tan| > tan_max) or it will not see the failure.
2. **What `plausible` should mean on beam.** Today it mixes a fit-quality cut (column duration) with the acceptance
   (tan_max).
   - The acceptance is already applied downstream as a true-angle cut at stage 3 (`build_tracks --tan-max-true`).
   - Dropping the tan part from the *pairing* decision is possible: `plausible` would keep only the column window.
     This is a smaller change than `cost`, and it would also stop the ranking from preferring a shallow noise
     candidate over a steep real one.
3. **If a change is adopted:** apply it only through the n_TOF bundle (`ntof_bundle.py`, recorded), re-run stage 3,
   and check the A–C cosmic line and the capsule pointing do not move (single tracks: they should not, since
   single-candidate views are untouched by `cost`).
4. **Existing stage-3 products** (is2_v2 and earlier) are made from stored candidate tables, so they are untouched by
   this code change whatever the n_TOF pass decides.

Contact point: the bench fix's report is the reference; the replay tool `rerun/effbudget/single/pairing_replay.py`
works on any candidates table whose bundle's `xy_pairing` uses no profile feature (true for the n_TOF bundles: same
features as the bench). It can re-score an existing n_TOF table in either mode without waveforms.
