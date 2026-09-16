# Joint two-track fit — working log

**Started 2026-09-16.** Implements **O1** of [`OCTOBER_2026.md`](OCTOBER_2026.md),
designed in [`HANDOFF_JOINT_TWO_TRACK_FIT.md`](HANDOFF_JOINT_TWO_TRACK_FIT.md).
This file is the live log: what was built, what was measured, what is next.
When the work lands, its conclusions move into a report and this stays as the trail.

**Where it stands (2026-09-16, end of session).** The fit is written, off by
default, unit-tested and measured on synthetics, on the overlay bench and on real
triggers. On the bench it recovers pairs below 12 mm for the first time
(coincident: 18 % A, 37 % C, from 0) and lifts 12–24 mm to 43 / 45 %, without
moving ≥ 24 mm; clean single muons split at ≤ 0.66 %. **Compute is not a
constraint** (Dylan, 2026-09-16), so the to-do below is about effectiveness, and
item 1 — dropping the trigger — is measured to be worth the most. The durable
record is [`../wft/TWO_TRACK_FIT_2026-09-16.md`](../wft/TWO_TRACK_FIT_2026-09-16.md);
the report is `<out>/two_track/report.html`.

## Status

| step (handoff §7.1) | state |
|---|---|
| 0 · read the code, size the problem | done |
| 1 · synthetic two-track planes, no I/O | **done** — 11 520 planes, both arms; unit tests `wft/tests/test_two_track.py` |
| 2 · bench variant `--two-track` | **done** — 3 variants, thresholds 80/30, 200/80, 300/120 |
| 3 · single-track A/B | **done** — `split-probe` (53 658 candidates, both arms, all tags) and `split-ab` (a full tag per arm through the real worker) |
| 4 · cost benchmark | done, as a record — +115 % (A) / +223 % (C) of an arm job. **Not a gate: compute is not a constraint (2026-09-16)** |
| 5 · data | not started (October, on the re-pass) |
| report | `<out>/two_track/report.html` |

## To do, in order (as of 2026-09-16, compute unconstrained)

1. **Attempt the fit on every candidate — no trigger.** Measured above to be the
   largest recoverable loss below 12 mm. Then re-measure what the trigger was
   also (incidentally) protecting: the false-split rate on real clean single
   muons (`split-probe`, `split-ab`) and the ≥ 24 mm band on the overlay bench.
   If false splits rise, fix them with the statistic and the guards, not by
   bringing the trigger back.
2. **Reconsider every candidate, not only the ones the selector chose**
   (`WFT_TWO_TRACK_ALL_CANDIDATES=1`). 58 % of the real-trigger splits at
   threshold 80 were on unselected candidates, mostly in busy events; whether any
   are real pairs is unmeasured. The bench and `split-ab` decide.
3. **Spend the optimiser budget.** Refine every start rather than the best two,
   iterate the alternating pursuit to convergence, scan every strip at the
   production w step and a wider t0, restart Nelder–Mead from the optimum. The
   synthetic fit-limited efficiency plateaus at 55–80 %; some of the remainder is
   likely missed basins. The synthetic study A/Bs this directly.
4. **Fit more than one hypothesis.** Tied *and* free t0, keeping the better (free
   showed no gain where tested, but it now costs nothing to include); a
   three-track model; a joint fit on the union of two overlapping candidate
   windows (handoff §4.5), which today are fitted separately on windows carrying
   each other's charge.
5. **A more exhaustive one-track start inside the two-track step.** On synthetic
   single tracks a missed one-track basin is what leaves structure for a spurious
   split. Do it inside `fit_plane_two` only, so production one-track answers stay
   frozen.
6. **Understand the last 0.2 % (A) / 0.9 % (C) of unsplit events that are not
   bit-identical** in `split-ab` — 1 and 3 tracks. Possibly the laptop-vs-condor
   float difference crossing the 3 mm match; unexplained.
7. **The 6–12 mm band is below the handoff's "half of pairs" target**, and A's
   12–18 mm bench number (43 %) is below split seeding's 48 %. Items 1–5 are the
   route there; re-run the report after each.
8. **Chambers B and D**, and one run on each side of the 23 July noise boundary and
   the 27 July access. Everything so far is run_145 stat090_0000, A and C.
9. **Then the campaign side**: per-run `xy_pairing`, the env vars into
   `condor/stage2_fullpass.sub`, a smoke gate that expects *more* tracks, and the
   single re-pass — `OCTOBER_2026.md` O3/O4.

**Not fixable, to be carried as inefficiency:** two parallel tracks on the same
strips, at any time offset. `acceptance.py` still does not model two-track
finding efficiency at all (O5).

**How to re-run everything** (the measurements behind every number above):

    python -m sept26_prelim_analysis.two_track_synth run --n 30 --jobs 14    # ~50 min
    python -m sept26_prelim_analysis.two_track_synth summarise
    python -m sept26_prelim_analysis.intra_bench build --variant pairing_rescue16_two_final \
        --pairing --local-mm 16 --two-track --jobs 14                          # ~25 min
    python -m sept26_prelim_analysis.intra_bench derive --variant pairing_rescue16_two_final
    python -m sept26_prelim_analysis.intra_bench compare pairing_rescue16_ranked pairing_rescue16_two_final
    python -m sept26_prelim_analysis.intra_bench split-probe --jobs 14      # ~2.5 h, no threshold
    python -m sept26_prelim_analysis.intra_bench split-ab --jobs 14 --tags 1 # ~30 min, the contract
    python -m sept26_prelim_analysis.make_two_track_report
    python -m pytest wft/tests -q

Outputs: `<out>/two_track_synth/`, `<out>/intra_bench/{split_probe,split_ab,<variant>}/`,
`<out>/two_track/report.html`. Variants on disk from this session:
`pairing_rescue16_two` (80/30, before both defects were fixed),
`pairing_rescue16_two_mm200`, `_mm300` (before the lost-track revert),
`pairing_rescue16_two_final` (the current defaults).

## Decisions taken

1. **A split replaces its parent.** It cannot be additive the way the rescue
   floor is — the children occupy the parent's strips and counting both would
   count the charge twice. The contract is instead: switch off ⇒ bit-identical;
   split not accepted ⇒ that candidate bit-identical; the parent kept in the
   candidates side table at `rank` −1 with `split_replaced`, so any split can be
   undone downstream without re-reconstructing. (Handoff §0.1, option chosen.)
2. **The statistic is the smaller of the two marginal χ² improvements**, not the
   total. χ²(without a) − χ²(both) and χ²(without b) − χ²(both), in units of the
   one-track fit's own χ²/dof. The total improvement is large whenever the
   second block absorbs anything at all, noise included; the minimum demands
   that *each* child be individually necessary.
3. **t0 is tied by default** (`WFT_TWO_TRACK_T0=tied`). Two prompt tracks from
   one vertex share a t0, and that is the hypothesis the same-chamber vertex
   test is about. A free second t0 walks into the plane χ²'s one-depth-bin
   degenerate minima: measured on synthetics, it splits *perfectly modelled
   single tracks* at fstat > 100. `free` is kept as a study option; on the one
   separable time-offset pair tested, tied scored higher.
4. **Cross-plane corroboration needs a count mismatch** — the other plane
   resolves more coincident plausible candidates than this one — not merely "the
   other plane has two". Without it the easy ≥ 24 mm case corroborated itself
   and lost 13–16 points (log below).
5. **A split that would leave the event with fewer gated tracks is reverted.**
6. **Operating point 300, or 120 with a cross-plane mismatch.** From the
   operating curve; 200/80 put C's clean-muon split rate at 1.3 %.
7. **Compute is not a constraint** (Dylan). Condor is effectively unlimited and
   a month-long re-pass is acceptable; optimise for effectiveness. Withdraws the
   handoff's §8 cost criterion, and every compute-motivated choice is now a
   to-do (above).

## Log

### 2026-09-16 — session start

Read `wft/{model,reco,seed,calib}.py`, `intra_bench.py`, and the three handoffs.
Baseline confirmed on disk: `<out>/intra_bench/` has the production baseline and
the `pairing_rescue16_ranked` variant; `run_145/stat090_0000` waveforms and the
`reco_fullpass` / `stage3_fullpass` products are staged locally.

### 2026-09-16 — step 1: the fit exists and the synthetics say what it does

**Written** (`wft/model.py`): `_solve_nnls` factored out of `chi2_plane` (numerics
unchanged, `test_model_regression` pins it), `build_matrix_two`, `chi2_plane_two`
(2K-column NNLS, same weighting and censoring), `fit_plane_two_raw` (multi-start
Nelder-Mead in 6 or 5 parameters), `two_track_separation` / `_distinguishability` /
`profile_overlap`. In `wft/reco.py`: `two_track_probe`, `two_track_triggers`,
`_scan_residual`, `_two_track_starts`, `_two_errors`, `_child_fit`, `fit_plane_two`,
`resolve_two_tracks`, and the `WFT_TWO_TRACK_FIT` switch wired through `_worker_fit`,
`candidate_rows` and `reconstruct_run`'s meta/`.splits.parquet`.

**Four things the synthetics changed about the design.** Each was a measurement,
not a preference:

1. **The one-track fit is a bad starting point on its own.** A merged window's
   one-track fit is a compromise line belonging to neither track; searching only
   for track *b* from there missed it above ~9 mm separation. Replaced by
   *alternating matching pursuit* — subtract one track's model, re-find the other
   in the residual, repeat — which costs only one-track solves. χ²/dof of the
   joint fit went from 6–14 to ~1.0 at every separation tried.
2. **The residual scan needs the p0–slope shear.** p0 is the position at the
   mesh but a zero-slope scan finds the charge *centroid*, and those differ by
   w × half the drift column — up to 10 mm at |tan| = 0.4. Without re-centring
   the scan per slope the second track was missed whenever it was the steeper
   one. (This is doc §21.1's `P0_SHEAR`, applied inside the new scan only.)
3. **Single tracks split end-to-end, not side by side.** Two straight lines fit
   *one* track's charge column by each taking half of it in depth; they are then
   far apart transversely wherever the other has no charge, so a separation
   guard passes and each child is "individually necessary". This is what made
   4 % of perfectly modelled single tracks clear the threshold. Fixed by a
   **profile-overlap guard** — the histogram intersection of the two children's
   normalised charge profiles, ≥ 0.35 — plus the same term in the optimiser's
   barrier. Real coincident tracks both cross the drift gap and overlap almost
   completely; two halves of one column do not overlap at all.
4. **Two parallel tracks on the same strips are not separable at any time
   offset.** The free charge profile absorbs the offset: a track arriving 260 ns
   later is the same data as one column running 260 ns longer, up to w × 260 ns
   of transverse slide, which is under a strip pitch at any slope we fit. The
   one-track fit of such a pair already reaches χ²/dof = 1. This is a model
   degeneracy, not a fitter weakness, and it has to be accounted for as
   inefficiency. Pinned in `test_two_track.test_parallel_co_located_pair_is_degenerate`.

**Unit tests**: `wft/tests/test_two_track.py`, 10 tests — the two-track model
contains the one-track model, the separation measures (including crossings),
recovery, label order, the degeneracy above, the false-split control, and the
three contract tests (switch off ⇒ input returned; accepted split replaces its
parent and is bookkept; refused split leaves the candidate alone). Whole suite
42 passed. `setup_function` re-installs the test bundle because `wft.model`'s
calibration is process-global and other test modules change it.

### 2026-09-16 — steps 2 and 3 wired

* `wft.reco._worker_init(bundle, pairing, opts)` — a third argument overriding the
  `WORKER_OPTS` module globals. The two-track settings are environment variables read
  at import, and a **forked** worker inherits the value, not the environment, so an
  A/B harness could not otherwise turn the fit on. Existing two-argument callers are
  unaffected.
* `intra_bench build --two-track [--two-track-f/-f-corrob/-t0/-resid-z]` — the overlay
  bench variant, writing `splits.parquet` alongside the candidates.
* `intra_bench split-probe` — **the A/B that matters**, and it is cheaper than
  re-fitting: it rebuilds each frozen production `PlaneFit` from the candidates side
  table, cuts the same window, and runs only the probe, the trigger and (if triggered)
  the joint fit **with no threshold**. So it measures, on real charge and real noise,
  (a) the trigger rate, (b) the cost per attempt, and (c) the whole `fstat`
  distribution on clean single muons — which is the population the threshold has to be
  chosen against. Candidates are matched to windows on strip count plus p0-in-window;
  ambiguous matches are dropped, not guessed.

**Also fixed while reviewing**: the guards were reading the raw NNLS charge profile,
whose late bins are unconstrained when their charge arrives after the last sample —
that is why run_145 tracks carry `q_sum` up to 1e26 and `q_uend` reads the last bin for
nearly every track. `wm.constrained_bins(t0)` now masks those out of the overlap guard,
the separation weighting and the optimiser's barrier. And the child-splice uses
identity, not dataclass equality, to find the parent in the ranked list.

### 2026-09-16 — what real triggers said, and the three changes it forced

`intra_bench split-probe` on two file tags of run_145 stat090_0000, chamber A:
**13 071 production candidates**, 1 058 of them the only gated track of a clean
single-track trigger.

| | clean single muons | everything else |
|---|---:|---:|
| candidates | 1 058 | 12 013 |
| residual trigger fires (z > 4) | 22.7 % | 31.4 % |
| width trigger fires (> 6 mm too wide) | **1.2 %** | **43.0 %** |
| any trigger | 23.4 % | 67.2 % |
| guards pass | 3.6 % | 42.7 % |
| **split at threshold 80** | **0.47 %** | 9.4 % |

Three things this changed:

1. **The residual trigger is the weaker one on real charge, not the stronger.**
   On synthetics it was decisive; on data the model is not perfect, so a bright
   single track leaves coherent residuals too — clean singles run z p90 = 5.2,
   p99 = 8.5. The *width* trigger, which the synthetics made look useless,
   separates cleanly: 1.2 % against 43 %. Threshold raised to z > 8, which drops
   attempts on clean singles from 23 % to 2.6 % and keeps 88 % of the splits.
2. **Only candidates the selector chose are reconsidered.** 75 % of candidates
   are extra clusters in already-busy events — four candidates a plane, 45
   strips, χ²/dof 13 — and they carry 58 % of the splits at threshold 80. Those
   are the H3 population this fit was never going to fix, and they were most of
   the bill. `resolve_two_tracks` now runs `select_tracks` first and attempts
   only on members of a selected pair (`WFT_TWO_TRACK_ALL_CANDIDATES=1` to
   reconsider everything).
3. **The scan looks where the residual is.** The zero-slope stage was one NNLS
   per strip over the whole window — most of the fit's price on a 60-strip
   window. It now scans only strips with coherent positive residual and one
   either side, falling back to the whole window when the residual is
   featureless (in which case there is no second track to find anyway).

**False splits.** 0.47 % of clean single muons at threshold 80 with the old
trigger, 0.28 % with z > 8 — against the handoff's ≤ 1 % criterion. The number
to beat is not noise but the single track itself; see the column-overlap guard
above.

### 2026-09-16 — the guard that was fighting itself, and the cost

**The overlap guard was the wrong instrument.** With a threshold on the profile
overlap *inside the optimiser's barrier as well as in the guard*, the fit parked
exactly on the threshold and the guard then decided genuine 15 mm pairs on the
fourth decimal (a real overlay: two tracks 14.7 mm apart, fstat 558, rejected at
overlap 0.3499). Replaced by a single measure used in both places: **the
separation weighted by the depths where BOTH children carry charge.** It is
large for two tracks side by side, large for two that cross (the crossing is one
depth bin out of eighteen), and *zero* for one track cut in half end to end,
because there is then no depth at which both children exist. The overlap stays
as a 0.05 backstop and is still recorded.

**Cost, measured on real windows of run_145 tag 000, one core:**

| | |
|---|---|
| probe (chi2, residual, triggers) | 0.59 ms per candidate |
| one joint fit | median 1.40 s, mean 1.83 s, ~820 chi2 evaluations |
| attempted, of candidates the selector chose | 50 % |
| **per selected candidate** | **0.92 s** |

That is ~0.6 core-h added to a ~1.2 core-h arm job, so **+50 %** — at the
handoff's §8 limit, and a factor 3 better than the first working version (2.6 s
per fit, every candidate). The two things that bought it: scanning only strips
with coherent positive residual instead of every strip of the window, and
reconsidering only candidates the selector chose.

### 2026-09-16 — reporting

`make_two_track_report.py` builds `<out>/two_track/report.html` from the three
products (synthetic set, `split_probe/attempts.parquet`, `intra_bench/compare.csv`),
omitting any section whose product is absent rather than inventing it. Figures:
efficiency against separation, the operating curve (efficiency against the
false-split rate on **real** clean muons, over the threshold), the statistic's
two populations, the bench bands, and a display — one merged pair with the data,
the one-track residual and the two-track residual side by side, which is the
clearest statement of what the trigger sees and what the fit fixes.

The efficiency table carries **two** numbers per band: what the fit does with the
trigger set aside, and what survives this plane's own trigger. The synthetic
study is one plane at a time, so it cannot fire the cross-plane trigger — the one
that on data catches a pair close in this view and wide in the other — and the
true efficiency is between them.

### 2026-09-16 — the production driver had to be taught about it too

`wft.reco.reconstruct_run` is the cosmic driver; the campaign runs through
`ntof_tracking/wft_beam.py`, which popped `_cand` from each worker row but knew
nothing about `_splits`. With the switch on, that list of dicts would have gone
into the events DataFrame as a column. It now writes a `.splits.parquet` sibling
next to the candidates (empty unless the fit is on) and records the whole
two-track configuration in `reco_config` of every `.meta.json` — the same rule
as everything else in this package: a product that cannot say what produced it
is a product nobody can check.

### 2026-09-16 — the overlay bench found a real defect: corroboration without a mismatch

The first bench run of `pairing_rescue16_two` (threshold 80 / 30) bought what it
was built for and broke something else:

| separation (both views) | production | pairing + rescue | **+ joint fit** |
|---|---:|---:|---:|
| < 12 mm, A / C | 0 / 0 % | 0 / 0 % | **9.6 / 20.5 %** |
| 12–24 mm | 17.9 / 17.3 % | 29.0 / 28.3 % | 28.7 / **36.0 %** |
| ≥ 24 mm | 46.5 / 38.7 % | **71.3 / 65.7 %** | 60.7 / 55.2 % ← regression |

Diagnosis, and it is unambiguous: **every one of the 341 lost tracks is in an
event where a split was accepted**, and at ≥ 24 mm 90 % of accepted splits were
"corroborated" — they took the *lower* threshold. The cause is that I
implemented only half of handoff §4's trigger 2. The condition should be a
cross-plane **count mismatch** — "the other plane has two time-coincident
plausible candidates *and this plane has one*" — and I had dropped the second
clause. So a pair that both planes already resolve, which is the easy case,
corroborated itself, got the discount, and split one of the two correct
candidates in two, destroying it (a split replaces its parent).

With the mismatch required, of the splits accepted at ≥ 24 mm only 6.7 % survive,
against 79 % of those below 12 mm — the condition discriminates exactly where it
should. `_cross_plane_mismatch` now requires it, with a unit test, and the
default thresholds go to 200 / 80 (the synthetic efficiency curve is flat to
~200–300, and the real clean-muon false-split rate there is 0.19 %).

Re-running the bench at 200/80 and 300/120 to choose between them.

### 2026-09-16 — the second defect, and the final configuration

**A split that costs the event a track.** The clean `split-ab` (no x/y pairing,
so only the joint fit differs from the frozen pass) showed 8 (A) and 25 (C)
triggers ending with *fewer* gated tracks than production had — 0.2 % and 0.8 %.
The mechanism: a split replaces its parent, and if the two children then fail to
pair with the other plane, the event loses the track it had. A fit whose job is
to find tracks must not lose one, so `_worker_fit` now reverts the whole event to
the unsplit answer whenever the split would reduce the gated-track count. Events
losing a track: 8 → **1** (A), 25 → **1** (C), with everything else unchanged.

**Also corrected a test whose premise was mine, not the data's.** I had assumed
the free-t0 mode was needed for time-offset pairs. On the one case where a pair
*is* separable that way — opposite slopes, 260 ns apart — the **tied** fit scores
higher (418 against 298). Nothing measured gives `free` an advantage, so it is
now documented as a study option rather than a fallback.

**Final configuration** (`WFT_TWO_TRACK_FIT=1`, everything else default):
threshold 300, 120 with a cross-plane count mismatch; residual trigger z > 8;
width trigger > 6 mm; tied t0; only candidates the selector chose.

| | A | C |
|---|---:|---:|
| bench, coincident, < 12 mm (was 0 / 0) | **18.3 %** | **37.2 %** |
| bench, coincident, 12–24 mm (was 37.5 / 29.0) | **42.5 %** | **45.2 %** |
| bench, coincident, ≥ 24 mm (was 73.3 / 67.8) | 72.8 % | 67.8 % |
| clean single muons split, real triggers | 0.25–0.37 % | 0–0.66 % |
| unsplit events bit-identical | 99.8 % | 99.1 % |
| events losing a track | 1 of 3 233 | 1 of 3 256 |

### 2026-09-16 — compute is not a constraint (Dylan's decision)

> Treat condor as effectively unlimited: a re-pass that takes a month is fine.
> The important part is finding an **effective** algorithm.

This withdraws the handoff's §8 "≤ +50 % core time" criterion and turns several
choices made during this session into things to undo. The biggest one is
measured already. On the synthetic set, coincident pairs with both legs
detectable, threshold 300, recovered by the fit alone against recovered *and*
triggered in this plane:

| separation | fit alone, A / C | this plane triggers, A / C | both, A / C |
|---|---:|---:|---:|
| 0–6 mm | 57 / 56 % | 25 / 26 % | **16 / 17 %** |
| 6–12 mm | 69 / 62 % | 64 / 63 % | 52 / 43 % |
| 12–18 mm | 82 / 66 % | 77 / 80 % | 72 / 59 % |
| 18–24 mm | 73 / 60 % | 79 / 82 % | 62 / 53 % |
| ≥ 24 mm | 72 / 61 % | 83 / 82 % | 63 / 52 % |

**At the closest separations the trigger throws away two thirds of what the fit
can recover** — two tracks a few mm apart leave little residual and no excess
width, which is exactly why they are hard to trigger on. The trigger existed
only to save compute. (The synthetics are one plane at a time, so the
cross-plane trigger is not in these numbers; on data it recovers some of this,
by an unmeasured amount.)

The full list of compute-motivated choices, with what each costs, is §8 of
`wft/TWO_TRACK_FIT_2026-09-16.md`.
