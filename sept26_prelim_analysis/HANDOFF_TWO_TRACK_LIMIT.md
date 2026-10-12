# HANDOFF — two tracks in one chamber: find the real limit, then reach it

**Written 2026-09-29.** For a fresh reader. Nothing in production changes; this
is a research-and-benchmark task done locally. Read `CLAUDE.md` first
(reconstruction basis: positions come from waveforms, never from hit times).

**Compute is not a constraint** (Dylan, 2026-09-16, reaffirmed today). Optimise
for *effectiveness only*. A fit that takes minutes per event is acceptable if it
finds more pairs. The only budget is your own iteration speed on the bench (§5).

---

## 0 · The ask

Same-chamber pairs (two particles in one micro-TPC, e.g. conversions, low-angle
IPC) are about a third of the two-track sample and the only single-calibration
vertex test. Production finds almost none closer than 12 mm. A joint two-track
fit exists (off by default) and gets part of the way. **The goal is to find where
the real, physical limit is, and get as close to it as possible.**

The starting claim to test, from Dylan: *there is a real physical limitation
somewhere; find it.* My reading of where it is, and where it is not, is in §1.
Treat §1 as a hypothesis to falsify, not a result. Nothing in it has been
measured except where a table says so.

## 1 · The crux (hypothesis)

### 1.1 Current efficiency is nowhere near a physical limit

Pitch is **0.78 mm** (`wft/model.py:76`). Everything below is in mm of p0.

| separation | in strips | what limits us today | physical? |
|---|---:|---|---|
| 12 mm | ~15 | seed gap `GAP_THRESHOLD_MM = 12` chains both into one window | **no**, a threshold |
| 6 mm | ~8 | joint fit's trigger (fires on 25 % of coincident pairs at 0–6 mm) | **no** |
| 1–3 mm | 1.5–4 | two blobs per depth slice are within ~2 kernel widths | **probably yes**, S/N-dependent |
| < ~1 mm | ≲ 1 | one blob per depth slice | **yes** |

Two tracks 6–12 mm apart are 8–15 strips apart: at *every* depth slice they are
two separate charge blobs, well outside the ±1-strip sharing kernel and the
diffusion width. That is not a hard inference problem. It is a search and
model-selection problem that the current code solves badly.

**The strongest evidence that the ceiling is algorithmic, not physical:** at
≥ 24 mm separation, where any method should approach 100 %, we get 72.8 % (A) /
67.8 % (C) on the overlay bench and 72 % (A) / 60 % (C) on the synthetics. The
synthetic plateau is *flat in separation*. A physical limit would fall with
separation. A flat plateau below 100 % is a fixed algorithmic loss. Known
contributors, not yet apportioned: a missed one-track basin (~17 % of synthetic
planes, `wft/TWO_TRACK_FIT_2026-09-16.md` §8 last row), x/y swaps, the plane-wide
significance floor, saturation, and donors that are themselves imperfect.
**Finding out how much each one costs is the first job** (§4, step 1).

### 1.2 Where the physical limit should sit

Per view, a track is a line in (strip, time) with a free non-negative charge
profile q(depth) over K = 18 bins of 60 ns (`model.py:79-81`). The window is
modelled as `M(θa)·qa + M(θb)·qb`, 36 non-negative columns.

* **Parallel tracks at separation d.** Every depth slice holds two blobs d apart,
  each smeared by σ² = σ_p0² + Dp²·u + (slope smear)² and by the strip pitch and
  resistive sharing. Resolvability depends on d against that width and on S/N.
  Diffusion grows with depth, so slices near the mesh (small u) are the sharpest.
* **d → 0 is a true degeneracy.** At d = 0, same slope, same t0, only qa + qb is
  identifiable. Near it, the information about d falls off as d² (the two-track
  model is singular at d = 0). Below that the physical answer is "one track, twice
  the charge", and no algorithm does better.
* **Non-parallel tracks** separate by their slope difference over the shared
  depths. Crossing tracks are at least as distinguishable as parallel ones, except
  at the crossing point itself (one depth bin in eighteen).
* **The free q profile is what makes it hard.** It is flexible enough to absorb
  structure, which is why an over-eager second track splits real single tracks,
  and why the model needs guards. Whether real charge profiles are clumpy
  (primary-ionisation clusters, δ-rays) enough to mimic a second track is the open
  question behind the 0.25–0.66 % false-split rate.

**A Fisher-information / Cramér-Rao calculation on the forward model, plus a
truth-seeded likelihood-ratio test, gives the physical curve directly:**
detection power vs separation, at the run's measured noise, per chamber.
Nobody has computed it. It is the yardstick everything else is measured against.

### 1.3 The second real limit: two views, and their pairing

A pair is only found if it is resolved **in both views and the views are paired
correctly**. Efficiency is (roughly) a product. A pair 12 mm apart in u and 300 mm
apart in v is lost as completely as one close in both (`STATUS` 2026-09-10,
0.005 per-view efficiency at 0–20 mm). Two structural points:

1. **The two views see the same charge.** Each track deposits one q(depth) that
   both x and y strips sample (up to a gain ratio and the y-side resistive
   kernel). A view that resolves two tracks tells the other view what each
   track's profile is. **A merged view should be fitted with its two profiles
   constrained by the resolved view.** This is unexploited, and it attacks the
   degeneracy of §1.2 with new information rather than better statistics. If u is
   merged and v is resolved, you already know there are two tracks and roughly how
   their charge is distributed in time.
2. **Pairing is a combinatorial ambiguity** (2 tracks → 2 pairings) broken by t0
   and by q(depth) agreement. Today `select_tracks` pairs by dchi2 rank; the
   xy_pairing fix uses charge ratio. A joint 3D fit would score all pairings on
   the shared profile.

### 1.4 What is probably *not* the limit

Fit quality once a separated track is found: robust σ(p0) against donor
≲ 0.1 mm, same strip count. Localisation is fine. Losses are in *finding and
splitting* (`MULTITRACK_2026-09-14.md`).

---

## 2 · What exists (do not rebuild)

| thing | where |
|---|---|
| 2K NNLS, separation and overlap measures, optimiser | `wft/model.py`: `chi2_plane_two`, `fit_plane_two_raw`, `two_track_separation`, `constrained_bins` |
| probe, triggers, residual scan, starts, decision, splice | `wft/reco.py`: `two_track_probe`, `two_track_triggers`, `_scan_residual`, `_two_track_starts`, `fit_plane_two`, `resolve_two_tracks` |
| switches | `WFT_TWO_TRACK_FIT`, `_F`, `_F_CORROB`, `_T0`, `_RESID_Z`, `_WIDTH_MM`, `WFT_TWO_TRACK_ALL_CANDIDATES`, or `reco._worker_init(..., opts=)` |
| overlay truth bench | `sept26_prelim_analysis/intra_bench.py` (`build`, `floor`, `derive`, `split-probe`, `split-ab`, `--variant`, `--two-track`) |
| synthetic planes, one plane at a time, no I/O | `sept26_prelim_analysis/two_track_synth.py` |
| report | `make_two_track_report.py` → `<out>/two_track/report.html` |
| tests | `wft/tests/test_two_track.py` (11), `test_model_regression` pins the one-track numerics |
| full record | `wft/TWO_TRACK_FIT_2026-09-16.md` (**read §1–§3, §7, §8**), `TWO_TRACK_FIT_LOG.md`, `HANDOFF_JOINT_TWO_TRACK_FIT.md`, `MULTITRACK_2026-09-14.md`, `OCTOBER_2026.md` |

Current numbers to beat (overlay bench, coincident pairs, both found *and*
correctly paired; production → pairing+rescue → +joint fit):

| separation | A | C |
|---|---|---|
| < 12 mm | 0 → 0 → **18 %** | 0 → 0 → **37 %** |
| 12–24 mm | 21 → 38 → **43 %** | 15 → 29 → **45 %** |
| ≥ 24 mm | 44 → 73.3 → **72.8 %** | 37 → 67.8 → **67.8 %** |

Contract on real triggers (`split-ab`): clean single muons falsely split
≤ 0.66 %, unsplit events bit-identical 99.8 % / 99.1 %, events losing a track 1 of
~3 250. Any new algorithm must be measured against the same contract.

**Already rejected, do not retry as stated:** local floor in *replace* mode
(loses 1.4–7.3 % of production tracks); seed splitting at 6/8 mm (loses 11–27 %);
free t0 as a general mode (splits perfectly modelled single tracks; tied t0 scores
higher even on an opposite-slope 260 ns-offset pair).

## 3 · Known compute-motivated handicaps (from §8 of the record)

These were chosen to save CPU and are now just losses. Each is a candidate first
experiment, and each should be A/B'd for effectiveness alone:
per-plane trigger (measured: fires on 25/64/77 % of recoverable pairs at
0–6/6–12/12–18 mm, A); selected candidates only; `TWO_TRACK_MAX_TRY = 2`;
optimiser budget (`TWO_N_REFINE = 2`, NM 220+140, `n_alt = 2`); residual scan
restricted and coarse; one hypothesis at a time (no three-track, no free-and-tied);
production's one-track start (17 % missed basins).

---

## 4 · Suggested plan

**Step 1 — the oracle ladder. This is the most important step.** On the
synthetic and overlay benches, remove one piece of ignorance at a time and record
efficiency vs separation. Each rung's gain is that stage's cost:

1. truth-seeded joint fit (start at the true θa, θb), count known → the fit's own
   ceiling. If this is far below 100 % at ≥ 24 mm, the model or noise is the
   limit, and that is worth understanding before anything else.
2. truth count, truth-seeded, real selector and pairing.
3. truth *window* (no seed clustering), fit from production starts.
4. + real seeding; + real trigger; + real one-track start; + real x/y pairing.

Also do the negative controls: truth-seeded fit on single-track donors, to get
the false-split rate at each statistic threshold with everything else perfect.
That rate is the price of every split, and sets the threshold.

**Step 2 — the physical curve.** Fisher information of the two-track model at the
measured noise, separation from 0.5 to 12 mm, parallel and non-parallel, tied t0,
per chamber (A, C), several depths of charge. Then a truth-seeded likelihood-ratio
power curve. Overlay it on step 1. The gap between the two curves is the
algorithmic headroom; the curve itself is the answer to "what is the real limit".

**Step 3 — attack the largest rung.** Ideas, roughly ordered by how much I expect
them to matter. All are hypotheses.

* **Joint xy fit with shared q(depth)** (§1.3). Fit both views' windows together,
  constraining each track's profile to agree between views up to a gain ratio.
  Resolves pairing and lets a resolved view rescue a merged one. Biggest
  structural gain I can see, and the only idea that adds information.
* **Convex dictionary search instead of seeding.** Non-negative (group-)sparse
  regression of the window on a dense dictionary of (p0, w, t0) atoms with free
  profile. No seed gap, no basin, no trigger; the answer falls out of the
  solution's support, then a local refine. Compute is free, so a dense grid is
  affordable. This removes the whole seed → trigger → start chain.
* **N = 1, 2, 3 on every window, always.** No trigger, no selected-only. Choose N
  by a statistic calibrated on the single-track donors, not on a Wilks threshold
  (χ²/dof is 1.4–6.8 on clean singles, so the scale is empirical).
* **A better one-track start inside the two-track step only**, leaving production
  frozen, to remove the missed-basin tail that makes spurious splits.
* **Iterate alternating pursuit to convergence, refine every start, restart NM
  from the optimum.** The plateau at 55–80 % may be missed basins.
* **Profile priors.** Real profiles are clumped, monotone-ish in depth, and the
  pair's two tracks are prompt (tied t0). A smoothness or sparsity prior on q
  makes the free profile less able to fake a track, which lowers the false-split
  rate at a given power.
* **Global pair hypothesis.** Use the event as a whole: the capsule position and
  the pair's two-track vertex constraint (`intra_vertex`) as a prior on where a
  second track can be. Careful: this uses physics the analysis is later meant to
  *test*, so keep any such prior out of the primary result or measure its bias.

**Step 4 — validate against the contract**, every time, before believing a gain:
overlay bench (all separations, both chambers, coincident and offset), `split-ab`
on real triggers, the unit tests, and the 0-production-tracks-lost check.

## 5 · Iterating locally

Machine: 16 cores, 62 GB. Local data is already staged (run_145
`stat090_0000` waveforms; `reco_fullpass` and `stage3_fullpass` products; output
root via `paths.out(...)`, i.e. under `~/x17`, not the repo). One joint fit ≈
2.5 s (A) / 3.4 s (C) under 14-way load; the full bench is 11 520 overlays.

Do not iterate on the full bench. Build a **fixed, stratified subset** (a few
hundred overlays per separation band, both chambers, fixed RNG seed) and a
one-command harness that prints the §4 table in minutes; run the full bench only to
confirm. Keep the synthetics for step 1 and 2, since they are one plane and no I/O.

## 6 · Things to check that I have not verified

* **Does the overlay add noise twice?** Each donor is a real trigger *including
  its own noise*; summing two doubles the noise variance. If the bench does not
  compensate, it is pessimistic by ~√2 in noise, and all the efficiencies above are
  lower bounds. `intra_bench.py` has `N_NOISE_PER_CELL`; I did not read how it is
  used. Check first, because it moves every number.
* **The donors' truth.** The "truth" is each donor's frozen single-track fit. A
  wrong donor fit gives a wrong label. How often, and does it fall on one class?
* **Which calibration bundle the bench uses.** It must be `calib_bundle_r06`
  (c2 = 0.6·c1); read c2 through `wft.calib.effective_c2`, never `hyper['c2']`. Any
  calibrated product is per detector *and* per run condition (CLAUDE.md).
* **The n_TOF noise condition.** run_145 is production, so it is on the noisy side
  of 23 July. Do not compare noise-dependent numbers with the June bench without
  saying so.
* **Drift extent.** K·DT·v ≈ 18 × 60 ns × ~40 mm/µs ≈ 43 mm (my arithmetic, not a
  quoted gap). Check `CAL.v_drift` and the real gap before quoting a
  depth-resolution number.
* **Saturation** (`SAT = 3550`) and what censoring does to two overlapping
  tracks: a bright pair may be limited by that, not by geometry.

## 7 · Rules that carry over

* Waveforms, not hits, for any position, angle or depth (`CLAUDE.md`).
* With the switches off, output is production's, candidate for candidate. A new
  mode is opt-in until Dylan decides otherwise.
* A synthetic study cannot find selector-level bugs; the overlay bench can. Both
  defects found in September (cross-plane corroboration without a count mismatch;
  a split that costs the event a track) only showed on the overlay bench.
* Write findings into `TWO_TRACK_FIT_LOG.md` as you go, with the measurement that
  justified each change. Ship a `report.html` per `CLAUDE.md` if a result is worth
  showing.
* Do not touch the production chain or start a condor re-pass. That is Dylan's
  call, after the algorithm is settled.

## 8 · Separate thread, not part of this

`pair_timing.py` / `make_pair_timing_report.py` (2026-09-29) ask why the two arms
of an *inter*-chamber pair are not prompt with each other. It is a different
question and should not be mixed into this benchmark.
