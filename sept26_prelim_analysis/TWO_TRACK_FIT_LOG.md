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

**Resume here (2026-09-30 evening):** [`TWO_TRACK_LIMIT_RESUME.md`](TWO_TRACK_LIMIT_RESUME.md)
has the current answer, the resumable validation chain and the lxplus plan.

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

### 2026-09-29 — the overlay bench doubled donor b's noise, and it cost 13–18 points at ≥ 24 mm

Start of the "find the real limit" work (`HANDOFF_TWO_TRACK_LIMIT.md`), §6 checks.

**Bundles, drift extent.** The bench uses production's run_145
`calib_bundle_prelim` per arm, which is the right per-detector, per-condition
product. Both pass `check_kernel_ordering`. A is r06-style (`c2_over_c1` = 0.6);
C stores c2 = 0.053 against c1 = 0.064 (ratio 0.82), with σ_s = 166 and
σ_p0 = 0.039 mm, very unlike A's. Noted, not acted on. v = 42.6 µm/ns is a
Magboltz prior, so the window depth is 18 × 60 ns × v ≈ 46 mm.

**Noise was doubled.** `payload` summed b's whole waveforms onto a's over b's
region (seed span ± 6 strips), so b's own strips always carried two triggers'
noise. The fit is not told this, and telling it does not help. The effect was
measured directly on clean single donors, with no second track, by adding an
empty trigger's waveforms on the donor's *own* region (175 donors per arm,
best configuration: pairing + rescue16 + joint fit at 300/120, tied):

| | t0 moves > 30 ns (x or y) | track found |
|---|---:|---:|
| A single / + own noise / + own noise, σ×√2 in fit | 0 / 65 / 58 % | 100 / 83 / 86 % |
| C single / + own noise / + own noise, σ×√2 in fit | 0 / 62 / 66 % | 99.4 / 77 / 77 % |

Position, slope and charge do not move. **t0 moves in steps of about one 60 ns
sample**, which is the near-degeneracy of t0 against a shift of the q(depth)
profile. The jump then breaks the x–y coincidence (± 120 ns) in `select_tracks`.
On the old bench at ≥ 24 mm, b's t0 jumped in 36–41 % of view fits against
4–5 % for a, and b was found 68–72 % against a's 84 %.

**Fix (bench only, opt-in):** `build --overlay replace`. On b's region outside
a's own region, b's waveforms *replace* a's (a's hits there are dropped). Only
the strips both regions share still carry two triggers' noise, which is
unavoidable there. The default stays `add`, so every earlier product reproduces.

Coincident pairs, both found and correctly paired, same seed and same pairs
(`pairing_rescue16_two_final` → `…_replace`):

| separation | A | C |
|---|---:|---:|
| < 12 mm | 18.3 → 19.2 % | 37.2 → 35.9 % |
| 12–24 mm | 42.5 → 47.5 % | 45.2 → 56.5 % |
| ≥ 24 mm | 72.8 → **85.0 %** | 67.8 → **86.1 %** |

Offset pairs at ≥ 24 mm: 69.4 → 93.3 % (A), 61.7 → 80.0 % (C). The a/b
asymmetry is gone (89.6 / 88.1 % A, 87.8 / 86.3 % C).

**Consequences.**
1. Every overlay-bench efficiency in this log, in `wft/TWO_TRACK_FIT_2026-09-16.md`
   and in the handoffs was pessimistic by this much at ≥ 12 mm. The < 12 mm
   numbers barely move, because there the regions overlap anyway.
2. **What is left at ≥ 24 mm is x/y pairing.** Donor-track outcomes, coincident
   pairs: found 86.9 / 88.6 %, **swapped 11.7 / 7.2 %**, candidate missing
   0.3 / 2.2 %, other 1.1 / 1.9 % (A / C). The "flat plateau" was a bench
   artefact plus swaps. Pairing (handoff §1.3) is the first target at large
   separation.
3. t0 is fragile at the one-sample level on real noise, so the ± 120 ns x–y gate
   has little margin. This matters for real events too, independently of the
   bench.
4. The synthetic plateau (72 / 60 % at ≥ 24 mm) has no doubled noise, so it is
   *not* explained by this and remains to be apportioned (step 1).

Script for the single-donor measurement: scratch `selfnoise.py`; it has not
been promoted into the bench yet.

### 2026-09-30 — the ideal-world curve: the physical limit is about one strip pitch

`two_track_limit.py` (new). It is a ladder of idealisations, all one plane
(x), tied t0, equal charge (Q = 1450 each, the synthetic median), white noise
13.3 ADC, run_145 bundles. Tracks are parallel at separation d, at tan 0 (a toy
extreme: no clean real donor has |tan| < 0.05) and tan 0.3 (representative).

**R1, the information limit (`asimov`).** A noise-free two-track window, fitted
with the best single track. The χ² left over, λ(d), is the expected Δχ² of a
perfect analysis. λ ≈ 12 at 1 mm and 160 at 2 mm for tan 0.3 (A). It falls like
~d⁴ near 0 and saturates once the tracks stop overlapping (≈ 8 mm at tan 0.3).
It is ~50× smaller at tan 0.3 than at tan 0 for the same d.

**R2, the oracle (`oracle`).** R1's windows plus noise. The fit is pure χ², with
truth starts plus a broad split set, and each Nelder-Mead is restarted.
Threshold: 99th percentile of Δχ² on 400 synthetic singles per cell (22–29;
χ²/dof on singles 0.988). 100 pairs per d. Efficiency (a pair is found when
both fitted lines are within 1 mm r.m.s. of distinct true lines):

| d (mm) | A tan 0 | A tan 0.3 | C tan 0 | C tan 0.3 |
|---|---:|---:|---:|---:|
| 0.25 | 8 % | 8 % | 39 % | 9 % |
| 0.5 | 75 % | 15 % | 65 % | 45 % |
| 0.75 | 100 % | 46 % | 98 % | 68 % |
| 1.0 | 100 % | 77 % | 99 % | 90 % |
| ≥ 1.5 | 100 % | 98–100 % | 100 % | 100 % |

**With a perfect model the limit is about one strip pitch (0.78 mm), and
above 1.5 mm nothing is lost.**

**Matching lesson.** Comparing p0 at the mesh wrongly fails correct inclined
lines: t0 slides by whole depth bins along the t0 ↔ q(depth) near-degeneracy
and drags p0 by w·Δt (seen at 3 mm = 4 bins). Matching is now the r.m.s.
distance between lines at the same absolute times (`line_dist`, `LINE_MM` = 1).
The overlay bench's `MATCH_MM` test at the mesh has the same weakness.

**C's calibration at tan 0.** With σ_p0 = 0.039 mm and Dp ≈ 0, C puts a
vertical track on one strip, with neighbours at the ~4σ sharing level. 361 of
400 synthetic vertical singles never make a ≥ 3-strip window. The bundle marks
σ_p0 and Dp as "not revalidated". Real clean tracks never go below
|tan| ≈ 0.05, so this is untested and does not matter for the data, but C's tan 0
numbers are the model's, not the chamber's.

### 2026-09-30 — production on the ideal curve's own planes: four algorithmic losses

`two_track_limit.py synthprod` runs production's one-track fit, probe,
trigger and joint fit on **the same** synthetic planes as R2 (same seeds), cut
to a production window (5σ strips + pad). Truth matching is by line
(`match`). Efficiency is "split accepted and both lines within 1 mm", in % (R2
ideal in brackets):

| d | A tan 0: prod / +scale fix / with trigger | A tan 0.3 | C tan 0.3 |
|---|---|---|---|
| 1 mm (ideal 100 / 77 / 90) | 0 / 0 / 0 | 22 / 24 / 0 | 25 / 30 / 0 |
| 1.5 mm (100 / 98 / 100) | 73 / 73 / 36 | 91 / 96 / 0 | 78 / 84 / 4 |
| 2–3 mm (100) | 2–25 / 3–32 / 0 | 95–99 / 97–100 / 0 | 68–84 / 71–85 / 0–6 |
| 4–6 mm (100) | 24–99 / 73–99 / 1–98 | 88–97 / 88–97 / 0–1 | 62–63 / 63 / 0 |
| 8–12 mm (100) | 98–100 | 87–97 / 87–97 / 0–52 | 87–92 / 87–92 / 0–63 |

False splits on 400 synthetic singles: 0–2 %, **identical with and without the
scale fix**.

The four losses, each measured:

1. **The trigger** fires on ~0 % of tan 0.3 pairs below 12 mm (and on 0 % of
   A vertical pairs at 2–4 mm).
2. **The fstat scale.** `scale = chi2_one / dof` holds the second track's
   unexplained charge, so fstat ≈ λ / (1 + λ/dof) < dof. A close vertical pair's
   window has dof ≈ 200–300 < 300 = `TWO_TRACK_F`. Median fstat at A 2–4 mm is
   213–287 when Δχ² ≈ 5 000. **Opt-in fix `WFT_TWO_TRACK_SCALE=two`** (scale
   from the two-track fit) in `wft/reco.py`; the default is unchanged and the
   tests pass. The rescue is large at tan 0, small at tan 0.3.
3. **The search.** At A vertical 2 mm, production's children form an "X"
   (tan ≈ ±0.1, t0 two bins early). It is seeded by the one-track parent, which
   is itself a slanted compromise line. On the same window, the truth-seeded
   pair has χ² lower by a median 685. Every unfound case in the test has a
   lower-χ² true solution (27/27), so this is search, not information.
   Refining all of production's own starts, with restarts: 4 → 50 % (2 mm),
   25 → 75 % (3 mm), 71 → 83–88 % (C tan 0.3, 4–6 mm). **A global grid of
   parallel line pairs** (strip step, common tan −0.4…0.4 in 0.1, tied t0 at
   parent ± 60 ns, NNLS per point, Nelder-Mead from the best 3): **96 / 92 /
   100 / 100 / 100 %**, median χ² gap to truth 0.0, 2–10 k evaluations per
   window (~5–30 s). Scratch `search_test.py`, 24 planes per cell.
4. **The distinguishability guard** (`TWO_MIN_SEP_MM` = 1.2) is a hard floor
   where the ideal is already 77–100 %.

### 2026-09-30 — real overlays: the limit on real tracks is ~2–3 mm, and production is far from it

`two_track_limit.py real` (R3/R4). Per view: clean donor pairs of one chamber,
file tag and trigger phase, coincident in this view (|Δt0| < 30 ns), overlaid
with the `replace` semantics. 40 pairs per bin of mesh separation and 600
single donors per (arm, view). Each window is 61 strips centred on the pair
and gets:
- the ideal fit on the real window;
- the ideal fit on its **perfect-model twin** (each donor refitted alone on its
  own real window, rebuilt by the forward model from that fit and its profile,
  plus white noise at each strip's pedestal σ);
- production (the bench's final configuration) on the same overlay.

Separation is quoted as the r.m.s. over the drift column, because real pairs
are not parallel: a pair 0.3 mm apart at the mesh can give Δχ² ≈ 1 500 through
its slope difference.

**The model mismatch is the real limit.** On single donors the ideal split
gains a median Δχ² of 48 (A) and 117 (C), against 8 for their twins. Children
of the bulk (< 90th percentile) stay within ~1 mm of the track, so this is
the track itself being split, not foreign charge. The tail is foreign charge
in my wide window: above the 90th percentile the donor alone fits at χ²/dof
5–134, and 30–47 % have a child > 5 mm away. Production's narrower windows
see less of it.

On well-modelled windows (donor(s) alone χ²/dof < 2: A 87 % of singles and
81 % of pairs; C 55 % / 36 %), the thresholds at 1 % false splits are:
real 274–851, twin 24–33. Pairs resolved in one view, twin / real ideal /
production:

| r.m.s. sep | A | C |
|---|---|---|
| 0.5–1 mm | 87 / 23 / 11 % | 100 / 38 / 12 % |
| 1–1.5 mm | 100 / 58 / 21 % | 100 / 78 / 44 % |
| 1.5–3 mm | 100 / 70–94 / 44–54 % | 100 / 100 / 0–68 % (n = 4–19) |
| 3–12 mm | 100 / 95–100 / 33–44 % | 100 / 98–100 / 50–58 % |
| 16–24 mm | 100 / 100 / 80 % | 100 / 100 / 80 % |

So: a perfect model resolves from ~1 mm; the real detector's mismatch
moves that to ~2–3 mm; production gets 33–58 % in between, where the ideal
fit on the same real windows gets 94–100 %. **The 40–60-point gap is
algorithmic.** C's well-modelled fraction (36–55 %) is low and unexplained;
it needs its own look before C's numbers are quoted.

**Why C's windows are so often badly modelled: charge.** The fraction of
single-donor windows with donor χ²/dof ≥ 2, by quartile of the donor's charge,
goes 17 / 34 / 55 / 82 % in C and 4 / 4 / 7 / 35 % in A. Only 6 % (C) and 14 %
(A) of those have a child > 5 mm away. Production's own χ²/dof for the same
donors is 11.3 (C) and 7.9 (A), against 2.7 and 1.6 for the well-modelled
ones. The mismatch grows with signal, as a fractional model error does
(residual ∝ ε·signal, so χ² ∝ q²). Two consequences. (1) The well-modelled
subset leans toward dim tracks. (2) A candidate fix for the split statistic,
**not yet tested**: carry σ² = noise² + (ε·model)² in the fit, with ε
calibrated on clean singles, instead of rescaling by χ²/dof afterwards. That
would make the split threshold independent of charge.

### 2026-09-30 — the fixed chain at equal false-split rate on real single tracks

First pass at `PROD_VARIANTS['fixed']` (`TWO_TRACK_SCALE=two`,
`TWO_TRACK_SEARCH=grid`, no trigger via `RESID_Z=-inf`, `MAX_TRY=99`, every
candidate) at F = 300 on the R3 windows. Real pairs improved a lot, but **real
single donors were split 5.2 % (A) and 8.8 % (C), against 1.0 / 1.2 % for
production.** Of that, 14–20 % was on bright, badly modelled donors and
2.9 / 5.0 % on well-modelled ones. Events with ≥ 2 tracks rose only
0.17 → 0.17 % (A) and 0.33 → 1.0 % (C). The synthetics could not show this:
on a perfect model χ²_two ≈ χ²_one for a single track, while on a real one the
mismatch makes χ²_two noticeably smaller, so the 'two' scale inflates fstat for
singles too.

**So compare at equal false-split rate.** `real --prod-only current_f0 /
fixed_f0` runs production at threshold 0 and records every attempt
(`r3_cands_*`, `r3_splits_*`). `two_track_limit scan` then applies any
threshold offline (the corroborated one at 0.4 F), undoing refused splits from
the recorded parents, and excludes replaced parents from the truth match (the
earlier real numbers used them; the effect is small). False splits = the
fraction of 600 single donors per view with an accepted split in either view.
Pairs = all real windows, 0–24 mm.

| | production @ 300 | fixed @ matched F |
|---|---|---|
| A: singles split / pairs resolved | 1.0 % / 36.5 % | **0.5 % / 66.0 %** (F = 1200) |
| C: singles split / pairs resolved | 1.7 % / 45.5 % | **1.2 % / 58.1 %** (F = 2400) |

At zero false splits among 600 singles, fixed reaches 58 % in A (F = 3200),
already above production at its own operating point. Per bin of r.m.s.
separation, all windows, production → fixed at matched F:

| sep | A | C |
|---|---|---|
| 2–3 mm | 50 → 66 % | 47 → 49 % |
| 3–8 mm | 34–44 → 78–85 % | 34–47 → 44–64 % |
| 8–16 mm | 32–40 → 83–87 % | 48–52 → 70–75 % |
| 16–24 mm | 71 → 83 % | 68 → 68 % |

Well-modelled windows reach 73–92 % (A) and 65–88 % (C). Bright ones stay at
42–69 % (A) and 27–70 % (C): the charge-dependent mismatch is the next target.
Still to do before any of this is proposed: the full overlay bench (event
level, x/y pairing), `split-ab` on real triggers, and the no-production-track-lost
check.

### 2026-09-30 — the 1.2 mm guard protects real singles; x/y profiles can pair tracks

**The distinguishability guard, on real windows.** Guard verdicts of every
split attempt in the fixed chain at threshold 0 (`r3_splits_fixed_f0`), by the
window's r.m.s. separation:

| | `distinguishable` fails | `column_shared` fails | `both_plausible` fails |
|---|---:|---:|---:|
| real single donors | **42 %** | 40 % | 10 % |
| pairs 0–1 mm | 58 % | 30 % | 5 % |
| pairs 1–1.5 mm | 36 % | 27 % | 3 % |
| pairs 1.5–2 mm | 8 % | 13 % | 0 % |
| pairs ≥ 2 mm | 0–2 % | 6–7 % | 2 % |

On real data the 1.2 mm guard is one of the main defences against false
splits (it stops 42 % of attempts on singles), and it costs pairs only below
~1.5 mm, at or under the real-track limit (~2–3 mm). Its synthetic "floor"
belonged to the perfect-model world. **Decision: leave it.** At most it could
buy part of the 1–1.5 mm bin (ideal 58–78 %, fixed 21–34 %), at a false-split
cost.

**x/y pairing by the shared depth profile.** Scratch `profile_pairing.py`:
clean donors of stat090_0000 (A 1 500, C 1 057). Production window and fit;
profile = the NNLS q at the fit. Coincident donor pairs (same tag and phase,
|Δt0| < 30 ns in both views; A 2 437, C 695). Decision: keep vs swap the y
partners, lower summed cost wins.

| cost | A correct | C correct |
|---|---:|---:|
| charge ratio `lq` (production) | 86.7 % | 89.2 % |
| profile, bin by bin, each in its own t0 frame | 72.1 % | 80.0 % |
| profile on a common absolute-time grid, smoothed 1 bin | 88.3 % | 92.8 % |
| `lq` + that profile term | **91.3 %** | **95.4 %** |
| held-out (weight and smoothing chosen on the other half) | **91.6 %** vs 87.4 | **92.8 %** vs 89.1 |

Bin-by-bin in each view's own t0 frame fails because the views' t0 slide by
whole bins (the same t0 ↔ q(depth) degeneracy). On an absolute-time grid the
profiles carry real pairing information: **about a third fewer wrong
pairings.** Implemented opt-in: `PlaneFit` objects now keep `_q`;
`wft.reco.profile_distance`; `xy_pair_cost` adds `weight · distance / scale`
only when the pairing calibration has a `prof` block (inert otherwise).
Calibrations: `xy_pairing_{A,C}_prof.json` (A weight 0.5, C 1.0, σ = 1 bin,
scale = median same-track distance, 0.123 / 0.141). **Caveat:** scale and weight
were chosen on stat090_0000, the bench's own sub-run (the original `lq`
calibration used 0001/0002), so this is in-sample for the bench. Recalibrate
on 0001/0002 before any real use.

### 2026-09-30 — profile pairing at event level, and the q_sum blow-up that defeats pairing

**Profile pairing on the bench, first calibration** (`final_replace_prof`: the
best configuration with only the pairing swapped; `xy_pairing_*_prof.json`
recalibrated on stat090_0001, σ = 1 bin, weight 0.5, so it is out of sample for
the bench). Held-out decisions on 0000 donor pairs: A 86.7 → 91.5 %, C
89.2 → 95.3 %. At event level, coincident ≥ 24 mm: A 85.0 → 87.2 %, C
86.1 → 86.1 %. Swapped donor tracks: A coincident 11.7 → 10.0 %, A between
13.1 → 7.5 %, **C unchanged to the decimal**. Nothing regressed and the noise
control is identical.

**Why C did not move** (scratch `repair_debug.py`, 12 swapped C overlays
replayed with `_repair_pairs` instrumented). Every one had two gated tracks and
a gate-passing swap, so the repair had its chance. The costs chose wrong for
two reasons.
1. **q_sum blow-ups.** 6 of the 12 involved a donor with fitted charge
   2.7·10⁹ … 9·10¹⁵. The charge-ratio term saturates at its cap on both options,
   and the normalised profile is dominated by the exploding bin.
2. Ordinary charge-ratio outliers (a track with x/y charge 1 863 / 376), where
   the profile term moves the costs the right way but not far enough.

**The blow-up is not rare. In stage-3 run_145/stat090_0000, q_sum > 10⁶ on
12–33 % of tracks per view in every chamber** (A 23 %, B 21–29 %, C 12–14 %,
D 28–33 %). It is known (`wft.model.constrained_bins` docstring: bins
arriving after the last sample have ~zero columns and NNLS parks arbitrary
charge there), and those tracks start late (median t0 400 ns against 80).
Positions and angles are unaffected, but **every charge-derived quantity of
those tracks is meaningless**: q_sum, q_u50/u90/uend, the x/y `lq` pairing,
and any downstream use of track charge (`build_tracks`, `candidate_filter`,
`ntof_tracking/reco/pairing`, … all read q_sum). **Flagged, not fixed, outside
pairing.**

**Pairing on constrained bins (opt-in).** `wft.reco.constrained_charge` and
`_constrained_profile` zero the unconstrained bins. New x/y feature `lqc`
(`wft.calib.XY_FEATURES`). `profile_distance` now uses the constrained profile.
Calibration `xy_pairing_{A,C}_profc.json`: features [lqc] + prof, fitted on
0001 (A σ = 1 bin, weight 2; C σ = 2, weight 1). Held-out 0000 donor pairs,
correct keep/swap:

| cost | A | C |
|---|---:|---:|
| `lq` (production) | 86.7 % | 89.2 % |
| `lqc` alone | 83.9 % | 90.1 % |
| **`lqc` + constrained profile** | **92.2 %** | **96.7 %** |

Clean donors carry few blow-ups (2–4 % of 0001 donors), so this understates the
gain where blow-ups sit. Bench variant `final_replace_profc` running.

**Constrained profile pairing on the bench** (`final_replace_profc`, calibrated
on 0001). ≥ 24 mm, efficiency / swapped donor tracks, baseline → profc:
A coincident 85.0 / 12.2 % → **90.6 / 7.2 %**; A between 81.1 / 13.9 → 85.6 / 9.4 %;
C coincident 86.1 / 7.2 → 85.6 / 7.8 %; C between 81.7 / 10.0 → 83.3 / 8.9 %.
C coincident 12–24 mm: 56.5 → 64.5 %. Offset, singles and the noise control
unchanged. **A gains clearly, C barely.** Replaying C's remaining swaps: the
tracks involved disagree with *themselves* across views (a donor with x/y
charge 2069/515, another 721/2200; own-profile distances 0.7–0.9 against a
median 0.14), so no pairing cost built on shared charge can place them. The
x/y charge ratio is flat with position (median log ratio within ± 0.05
across the chamber; only the outermost 40 mm in y fall to −0.3…−0.5, edge
truncation), so a gain map would not help. Its per-track spread is 0.48 (A) /
0.67 (C) on all gated tracks against 0.12–0.20 on clean donors. That
per-track disagreement is the floor for charge-based pairing.

### 2026-09-30 — a fractional model error (ε) does not help: dropped

`two_track_limit.py real --eps 0 0.03 0.1`: the ideal fit on the R3 windows
(x view, 15 pairs per bin plus 200 singles per chamber) with
σ² = noise² + (ε · model)² solved by reweighting (`Plane(eps=)`); ε = 0
reproduces the plain fit exactly. ε does what it was meant to do to the
**median**. Single-track Δχ² by charge quartile, A: ε = 0 → 30 / 43 / 66 / 136;
ε = 0.1 → 23 / 25 / 36 / 57 (C: 30 / 85 / 135 / 352 → 24 / 42 / 75 / 174). But
the threshold is set by the tail, and ε shrinks the pairs' Δχ² as much or more.
At a fixed false-split rate efficiency never improves:
- all windows, A, 2–3 mm: 68 / 53 / 5 % at 2 % false splits (ε = 0 / 0.03 / 0.1);
  95 / 95 / 95 % at 5 %;
- well-modelled windows: A neutral within statistics (1–2 mm: 74 / 74 / 81 %,
  27 pairs); C worse (1–2 mm: 80 / 70 / 30 %).

The mismatch of real tracks is not a constant fraction of their signal, so this
form of error model takes information away from real pairs as fast as it
removes false splits. **Not pursued.** The charge-dependent mismatch remains
the real-data limit; curing it needs a better *model* (per-track shape), not a
looser χ².

### 2026-09-30 — validation moved to lxplus condor

Chamber A's fixed-chain bench (local, event level, F = 1200), coincident:
< 12 mm 19.2 → 56.7 %; 12–24 mm 47.5 → 75.0 %; ≥ 24 mm 85.0 → 83.9 % (± 3,
unchanged). Between: 8.3 → 41.7 / 34.2 → 55.8 / 81.1 → 82.2 %. Offset:
1.1 → 12.0 / 27.4 → 34.5 / 93.3 → 93.9 %. Clean singles on the bench 100 %
found, no extra tracks; noise control identical. This used the old `lq`
pairing.

The rest of the validation went to condor (cluster 4334051, 147 jobs, all
seven tags instead of one): `sept26_prelim_analysis/condor/two_track/`. That
needed `intra_bench build --only-tag` (overlay ids identical to an unsharded
build; the noise-control empties come from a per-tag generator) and
`split-ab --only-tag --shard i/n`, with `split_ab_summary` factored out so
shards merge through the same code. Resume and merge instructions:
`TWO_TRACK_LIMIT_RESUME.md`.

**split-ab: the bit-identity drop is `--pairing`, not the fixed chain.** The
local fixed-chain split-ab (A, tag 000, F = 1200, `--pairing`) gave clean
singles split 1/272 (0.37 %, as baseline), 48 events split (as baseline),
1 event losing a track (as baseline), but unsplit events bit-identical 93.0 %
and 74 tracks not recovered in unsplit events (baseline 99.8 %, 1). The condor
control, current production with the **same** `--pairing` and F = 300, gives
93.8 % and 65 on the same tag (37–65 per tag, 93.8–96.7 % over all seven). The
recorded baseline (`intra_bench/split_ab/`) was run without `--pairing`; the
frozen full pass was reconstructed without x/y re-pairing, and split-ab's match
(x and y both within 3 mm) counts a re-paired track as "not recovered". The
fair contract comparison is fixed against current **with the same pairing**:
48 = 48 splits, 0.37 % = 0.37 % clean-single splits, 75 vs 93 tracks not
recovered overall. Condor's current A tag 000 reproduces the local baseline's
1 474 attempts and 48 splits exactly.

### 2026-10-01 — condor results: event-level benches, and the contract on real triggers

**Benches, all seven tags** (event level, both tracks found and correctly
paired, coincident; before = `pairing_rescue16_two_final_replace`):

| | < 12 mm | 12–24 mm | ≥ 24 mm |
|---|---:|---:|---:|
| A before → fixed (F = 1200) → **fixed + profc** | 19.2 → 56.7 → **57.5** | 47.5 → 75.0 → **79.2** | 85.0 → 83.9 → **89.4** |
| C before → fixed (F = 2400) → **fixed + profc** | 35.9 → 47.4 → **48.7** | 56.5 → 59.7 → **69.4** | 86.1 → 86.1 → **85.6** |

Offset pairs unchanged or slightly up; noise controls A 97–98 %, C 94 %.

**split-ab contract, fixed against current on the same triggers, both with
`--pairing`** (`two_track_scratch/contract.py` →
`intra_bench/contract_fixed_vs_current.csv`):

| | A current | A fixed | C current | C fixed |
|---|---:|---:|---:|---:|
| triggers | 22 434 | 22 434 | 22 788 | 22 788 |
| clean singles split | 6/1804 (0.33 %) | **9/1804 (0.50 %)** | 7/1057 (0.66 %) | **5/1057 (0.47 %)** |
| events losing a track | 0 | **0** | 0 | **0** |
| clean tracks lost | 1 | 0 | 3 | 3 |
| tracks not recovered | 485 | **405** | 466 | **234** |
| events split / gaining a track | 272 / 85 | 346 / 70 | 913 / 300 | 207 / 46 |

**A passes**: 0.50 % against the contract's ≤ 0.66 %; 9 vs 6 is within
fluctuation (0–2 per tag in both chains); no event loses a track; 80 more
production tracks kept. **C passes on the first 35 %** (20/56 shards). At its
matched threshold (2400) C's fixed chain accepts *fewer* splits on real
triggers than current production (80 vs 304 events; 15 vs 101 events gaining a
track), yet it resolves more pairs on the bench. Current production's C splits
(4 % of real triggers, 913 events over seven tags) deserve a look: the bench
says a lower threshold is affordable in C. The remaining 36 C shards are
running (C events are busier, ~14 h per shard).

**2026-10-01 evening — C complete (56/56 shards), C passes.** All 147 jobs of
cluster 4334051 returned; merged and the contract rerun (table above now holds
the full seven tags for both chains). C fixed: clean singles split 5/1057
(0.47 %) against current's 7/1057, no event loses a track, 232 more production
tracks kept (466 → 234 not recovered). The 3 clean tracks lost are the same
count as current production. The pattern of the partial result holds at full
statistics: at F = 2400, C's fixed chain splits 207 events (46 gaining a track)
where current splits 913 (300). Both chains are inside the contract, so the
open question is unchanged: is F = 2400 leaving real C pairs on the table, or
are current's extra C splits false? That needs an F rescan on split-ab C.
