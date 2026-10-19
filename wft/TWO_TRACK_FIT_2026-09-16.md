# A joint two-track fit for tracks that share one seed cluster — 2026-09-16

**Two tracks closer than the 12 mm seed gap are one cluster, one window and one
compromise line.** That costs ~100 % of pairs below 12 mm and 62–65 % at
12–24 mm (`intra_bench`, 2026-09-14), and small-opening-angle pairs —
conversions, low-angle IPC — are exactly the population it eats. Splitting the
seed was tried and rejected: it breaks real single tracks
([`MULTITRACK_2026-09-14.md`](MULTITRACK_2026-09-14.md) §3.2). This is the other
route — offer the window a second track and let χ² decide.

**Off by default** (`WFT_TWO_TRACK_FIT=0`); with it off the output is
production's, candidate for candidate. Nothing here is in the production chain.

> **Compute is not a constraint (Dylan, 2026-09-16).** Treat condor as
> effectively unlimited: a re-pass that takes a month is acceptable. The goal is
> the most *effective* algorithm, not a cheap one. Several choices below were made
> to save compute before this was settled — they are listed in §8 with what each
> is measured to cost in efficiency, and they are the first things to revisit.
> The cost figures in §6 are kept as a record, not as a gate.

Companions: the design and the acceptance criteria,
`sept26_prelim_analysis/HANDOFF_JOINT_TWO_TRACK_FIT.md`; the working log with
every measurement that shaped the algorithm,
`sept26_prelim_analysis/TWO_TRACK_FIT_LOG.md`; the report,
`<out>/two_track/report.html`.

---

## 1 · The model

For one plane window, with θ = (p0, w, t0) per track:

    W  ≈  M(θa)·qa  +  M(θb)·qb ,      qa, qb ≥ 0

Nothing in the forward model assumed one track except `build_matrix` being
called once — the charge profile was already a free non-negative vector, so a
second track is a second block of columns in the *same* NNLS
(`wm.chi2_plane_two`, 2K = 36 columns), with the same noise weighting,
saturation censoring and dead/hot handling; `_solve_nnls` is factored out of
`chi2_plane` so there is one copy of that, not two.

Setting qb = 0 reproduces the one-track model exactly, so χ²(two) ≤ χ²(one) at
the same θa — which is what makes Δχ² a model-selection statistic rather than a
fit artefact, and is pinned by a test.

**t0 is tied by default.** Two prompt tracks from one vertex reach the mesh
together, and that is the hypothesis the same-chamber vertex test is about. A
free second t0 walks into the plane χ²'s near-degenerate minima one depth bin
apart, and splits perfectly modelled *single* tracks. `WFT_TWO_TRACK_T0=free`
exists for accidentals and has to earn its place separately.

## 2 · Finding the second track

A merged window's one-track fit is a compromise line belonging to neither track,
so searching only for track *b* from there misses it above ~9 mm separation.
Instead, **alternating matching pursuit**: subtract one track's model, re-find
the other in the residual, repeat. It costs only one-track solves, and the joint
Nelder–Mead then has to move the tracks a little rather than find them.

Two details that turned out to matter:

* **The scan re-centres per slope.** p0 is the position at the mesh but a
  zero-slope scan finds the charge *centroid*, and those differ by w × half the
  drift column — up to 10 mm at |tan| = 0.4. Without this the second track is
  missed whenever it is the steeper one.
* **The scan looks where the residual is**, not at every strip. That was most of
  the fit's price on a wide window.

## 3 · One track or two

**The statistic is the smaller of the two marginal χ² improvements** —
χ²(without a) − χ²(both) and χ²(without b) − χ²(both) — in units of the
one-track fit's own χ²/dof. Not the total improvement: that is large whenever
the second block absorbs anything at all, noise included. The minimum demands
that *each* child be individually necessary. χ²/dof is 1.4–6.8 on clean single
tracks here and 5–12 across the full pass, so the scaling is empirical and a
Wilks threshold would split almost everything.

**The guard that matters is the separation measured at the depths where both
children carry charge.** The failure mode a threshold has to survive is not
noise: it is two straight lines fitting *one* track's column by taking half of
it each in depth. Weighted by the shared depths, the separation is large for two
tracks side by side, large for two that cross (the crossing is one depth bin out
of eighteen), and zero for one column cut in two. Guards: that separation
(combined in quadrature with any t0 difference) ≥ 1 resolvability unit, both
children plausible on the production column-duration window, and a profile
overlap backstop.

Depth bins whose charge arrives after the last sample are unconstrained — NNLS
parks arbitrary charge there, which is why run_145 tracks carry `q_sum` up to
1e26 — so every profile-based guard is masked to `wm.constrained_bins(t0)`.

## 4 · When it runs

A 6-parameter fit with multi-start costs several single fits, so as built it
runs only where the window looks merged, and only on candidates `select_tracks`
actually chose. **Both restrictions exist to save compute, and the trigger is
now measured to be the largest single loss at the smallest separations** (§8).

Triggers: the window is wider than one track of the fitted |tan| and
column duration can cover; the one-track fit leaves a large coherent positive
per-strip residual; or the other plane resolves two time-coincident plausible
candidates where this plane resolves fewer — a count *mismatch*, §7.1 — which
also lowers the threshold and supplies t0 hints.

## 5 · The contract

A split **replaces** its parent — it cannot be additive the way the rescue floor
is, because the two children occupy the parent's strips and counting both would
count the charge twice. So instead:

* with the switch off, the output is bit-identical;
* a candidate whose split is not accepted is bit-identical;
* a split that would leave the event with fewer gated tracks is reverted (§7.2);
* the parent is kept in the candidates side table at `rank` −1 with
  `split_replaced`, and the children carry `split_child` and `split_dchi2`, so a
  split can be undone downstream without re-reconstructing;
* the false-split rate on clean single muons is measured and bounded.

## 6 · What it does, measured

Three populations, in increasing realism and decreasing truth. Operating point
**300 / 120** (base / cross-plane-mismatch threshold), chosen from the curve in
`<out>/two_track/report.html`, not from one number.

### Overlay bench — two real single-track triggers of one chamber, summed

Both donors found *and correctly paired*, time-coincident pairs, on top of x/y
pairing + the rescue floor (11 520 overlays, run_145 stat090_0000):

| separation, both views | production | + pairing & rescue | **+ joint fit** |
|---|---:|---:|---:|
| **< 12 mm**, A / C | 0 / 0 % | 0 / 0 % | **18 / 37 %** |
| **12–24 mm** | 21 / 15 % | 38 / 29 % | **43 / 45 %** |
| ≥ 24 mm | 44 / 37 % | 73.3 / 67.8 % | 72.8 / 67.8 % |

Below 12 mm the two tracks share one seed cluster and nothing recovered them
before. At ≥ 24 mm the fit costs half a point on A and nothing on C — at
threshold 80 / 30 and without the two guards of §7 it cost 13 and 16 points.

### Synthetic planes — the forward model's own, with the run's noise

Both tracks recovered, coincident, both legs detectable on their own, fit-limited
(the trigger set aside; the study is one plane at a time so it cannot fire the
cross-plane trigger):

| 0–6 | 6–12 | 12–18 | 18–24 | ≥ 24 mm |
|---:|---:|---:|---:|---:|
| A 54 % | 65 % | 72 % | 69 % | 72 % |
| C 54 % | 59 % | 61 % | 58 % | 60 % |

Flat in separation, which is the point: once two tracks clear the degeneracy
floor the fit does not care how far apart they are. p0 against truth is
≲ 1 mm r.m.s. in every band.

### The contract, on real triggers

`intra_bench split-ab` re-reconstructs a whole file tag per arm through the real
worker with the switch on, and matches every production gated track to the frozen
pass (no x/y pairing, so only the joint fit differs):

| | A | C |
|---|---:|---:|
| triggers / split attempts | 3 233 / 1 474 | 3 256 / 2 498 |
| events with an accepted split | 1.5 % | 3.8 % |
| **clean single muons split** | **0.37 %** (1 of 272) | **0 %** (0 of 165) |
| **unsplit events bit-identical** | **99.8 %** | **99.1 %** |
| clean single tracks lost | **0** | **0** |
| events gaining / losing a track | 13 / **1** | 38 / **1** |
| production gated tracks replaced by children | 2.9 % | 4.6 % |

The last row is the contract's cost, not a bug: a split replaces its parent, so
in the 1.5–3.8 % of events where one is accepted the parent no longer matches.
The A/B cannot say whether the replacement is right — it shows production answers
did not move elsewhere. Whether the added tracks are real is `intra_vertex`'s
and `det_a_intra`'s question after a re-pass.

The residual 0.2 % / 0.9 % of unsplit events that are not bit-identical are 1 and
3 tracks respectively; unexplained, and small enough that they may be the known
laptop-against-condor difference crossing the 3 mm match. Worth a look before a
pass.

### Real triggers — 53 658 production candidates the selector chose

| | clean single muons | everything else |
|---|---:|---:|
| candidates (A / C) | 3 606 / 2 113 | 20 543 / 27 396 |
| any trigger fires | 1.8 / 6.7 % | 57 / 68 % |
| **split accepted** | **0.25 / 0.66 %** | 0.7 / 2.2 % |

Against the handoff's ≤ 1 % criterion on clean single tracks. The population to
beat is not noise, it is the single track itself.

### Cost — a record, not a gate

Measured under 14-way parallel load:

| | A | C |
|---|---:|---:|
| probe (chi2, residual, triggers) | 1.1 ms | 1.3 ms |
| one joint fit, median | 2.5 s | 3.4 s |
| attempted, of candidates the selector chose | 57 % | 68 % |
| added per tag-arm job | ~1.4 core-h | ~2.7 core-h |

An arm job is ~1.2 core-h today, so this is +115 % (A) to +223 % (C). The
handoff's §8 "≤ +50 %" criterion was **withdrawn on 2026-09-16**: compute is not
a constraint. (A standalone warm-cache measurement had given +50 %; under load
it is 2–3× that — worth knowing only for sizing a pass.)

## 7 · Two defects the overlay bench found, which the synthetics could not

### 7.1 Corroboration without a count mismatch

The first bench run (threshold 80 / 30) bought what it was built for **and broke
the easy case**: ≥ 24 mm went 71 → 61 % (A) and 66 → 55 % (C). Every one of the
341 lost tracks was in an event where a split was accepted, and 90 % of the
splits accepted at ≥ 24 mm were "corroborated" — they took the *lower* threshold.

The cause was half an implementation of the cross-plane condition. It has to be a
count **mismatch** — the other plane resolves two time-coincident plausible
candidates *and this plane resolves fewer* — and the second clause was missing.
So a pair both planes already resolve corroborated itself, got the discount, and
split one of the two correct candidates in two, destroying it. With the mismatch
required, of the splits accepted at ≥ 24 mm only 7 % survive, against 79 % of
those below 12 mm.

### 7.2 A split that costs the event a track

A split replaces its parent, so if the two children then fail to pair with the
other plane the event ends up with **fewer** gated tracks than it had — measured
at 0.2 % (A) and 0.8 % (C) of real triggers. A fit whose job is to find tracks
must not lose one, whatever its chi2 says, so the event now reverts to the
unsplit answer whenever the split would reduce the gated-track count. That took
events losing a track to 1 and 1, and moved the bench's ≥ 24 mm band back to the
baseline (C) or half a point below it (A).

Both of these are the same lesson: **a synthetic study cannot find that class of
bug**, because it has one plane and no selector. The overlay bench has both.

## 8 · Choices made to save compute — revisit first

Made before compute was ruled out as a constraint. Each should be A/B'd on the
overlay bench and `split-ab`, with effectiveness as the only criterion.

| choice | where | what it costs in efficiency (measured) | try instead |
|---|---|---|---|
| **per-plane trigger** (residual z > 8 or width > 6 mm) | `two_track_triggers`, `resolve_two_tracks` | **The largest loss at small separation.** Synthetic coincident pairs, both legs detectable, threshold 300: the fit alone recovers 57 / 69 / 82 % (A, 0–6 / 6–12 / 12–18 mm) but this plane's trigger fires on only 25 / 64 / 77 % of them, leaving **16 / 52 / 72 %**. C: 56 → 17 % at 0–6 mm. (The synthetics cannot fire the cross-plane trigger, so on data it is somewhat less.) | attempt on **every** candidate and let the statistic and guards decide; the false-split rate on real clean muons is then the thing to re-measure |
| **selected candidates only** | `TWO_TRACK_SELECTED_ONLY` | on real triggers 58 % of the splits at threshold 80 were on candidates the selector had not chosen — mostly busy events (4 candidates a plane, 45 strips). Whether any are real pairs is unmeasured | `WFT_TWO_TRACK_ALL_CANDIDATES=1`, then check the bench and `split-ab` |
| `TWO_TRACK_MAX_TRY = 2` per plane | `resolve_two_tracks` | not measured | no cap |
| optimiser budget: `TWO_N_REFINE = 2`, Nelder–Mead 220 + 140 iterations, `n_alt = 2` | `fit_plane_two_raw`, `_two_track_starts` | not measured; the synthetic fit-limited efficiency plateaus at 55–80 %, so some of the remainder may be missed basins | refine every start, iterate the pursuit to convergence, restart NM from the optimum |
| residual scan: only strips with coherent residual, w at 2× the production step, t0 ±60 ns | `_scan_positions`, `_scan_residual` | not measured | scan every strip, production w step, wider t0 |
| **one hypothesis at a time** — tied t0 only; never three tracks; overlapping windows fitted separately | design | free t0 showed no gain where tested; the others are untried | fit tied *and* free and keep the better; a three-track model; a joint fit on the union of overlapping windows (handoff §4.5) |
| the **one-track** starting point is production's | `fit_plane` / `_global_start` | on synthetic single tracks a missed one-track basin (~17 % of planes, doc §21) is what leaves structure for a spurious split — the long tail of the statistic on singles | a more exhaustive one-track scan *inside the two-track step only*, so production one-track answers stay frozen |

## 9 · Where the code is

| what | where |
|---|---|
| 2K NNLS, separation and overlap measures, the optimiser | `wft/model.py`: `chi2_plane_two`, `fit_plane_two_raw`, `two_track_separation`, `common_weight`, `constrained_bins` |
| probe, triggers, residual scan, starts, decision, splice | `wft/reco.py`: `two_track_probe`, `two_track_triggers`, `_scan_residual`, `_two_track_starts`, `fit_plane_two`, `resolve_two_tracks` |
| switches | `WFT_TWO_TRACK_FIT`, `_F`, `_F_CORROB`, `_T0`, `_RESID_Z`, `_WIDTH_MM`, `WFT_TWO_TRACK_ALL_CANDIDATES`; or `reco._worker_init(..., opts=)` |
| synthetic study | `sept26_prelim_analysis/two_track_synth.py` |
| real triggers, no threshold | `intra_bench split-probe` |
| overlay bench | `intra_bench build --two-track` |
| contract check on real triggers | `intra_bench split-ab` |
| report | `sept26_prelim_analysis/make_two_track_report.py` |
| tests | `wft/tests/test_two_track.py` |
| report | `<out>/two_track/report.html` |
