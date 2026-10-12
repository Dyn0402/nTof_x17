# HANDOFF — a joint two-track fit for tracks that share one seed cluster

**Written 2026-09-14.** The design, the case for the fit, and the checks it has
to pass. It picks up where `HANDOFF_INTRA_TWO_TRACK_RECO.md` §10 and
`wft/MULTITRACK_2026-09-14.md` stopped.

> **Implemented 2026-09-16.** The fit exists, is off by default, and has been
> measured on synthetics and on real run_145 triggers. What was built, what the
> measurements changed about the design, and what is still open: §12 below,
> `wft/TWO_TRACK_FIT_2026-09-16.md` (the record) and
> `TWO_TRACK_FIT_LOG.md` (the working log, with every measurement).

> **Branch note.** The code this refers to (`intra_bench.py`, the rescue floor,
> `select_tracks(pairing=)`, `MULTITRACK_2026-09-14.md`) is on `origin/main`
> (`8dc1479`, `0bd9f95`). `pair-vertex-imaging` was written before those commits.
> Line numbers below are for `origin/main`.

Companions:

- `wft/MULTITRACK_2026-09-14.md`: the x/y pairing and rescue floor, and the
  single-track A/B every change must pass.
- `wft/MULTITRACK_2026-08-12.md`: "tier 3", the first sketch of this fit (a 2K NNLS
  basis, 6 outer parameters, and a 1-vs-2-track model-selection penalty).
- `sept26_prelim_analysis/intra_bench.py`: the overlay truth bench.
- `CLAUDE.md` → *Reconstruction basis*. Geometry comes from waveforms. Hits may
  choose which strips to look at, never a position, angle or depth.

---

## 0 · Decisions for Dylan before starting

1. **The single-track contract has to be restated.** Today the rule is "pair 0 is
   production's, extra candidates can only *add* a track" (R4). That rule works
   for the rescue floor because rescued clusters share no strips with production
   clusters. A split cannot follow it: in every event where it acts, it *replaces*
   the one merged track with two. Proposed contract:
   - events where the two-track hypothesis is not accepted come out bit-identical;
   - on clean single tracks the false-split rate is measured and bounded (§8);
   - the parent fit is kept in the candidates sidecar with a flag, so downstream
     can undo any split.
2. **Decide whether the fit ships with the rescue floor and x/y pairing in one
   condor re-pass, or those go first.** Recommendation: one re-pass. The pass
   itself is ~8 h of condor (§10), but the chain around it is a day's work. The
   joint fit changes the same files and the same candidate tables, so two passes
   would mean validating the same product twice.

---

## 1 · Why: what is still lost after the rescue floor

On the run_145 overlay bench (two clean single tracks of one chamber, same tag and
trigger phase), after pairing + rescue:

| separation (smaller of the two views) | both tracks found and correctly paired, A | C | what loses them |
|---|---:|---:|---|
| ≥ 24 mm | 71 % | 66 % | fixed by pairing + rescue |
| 12–24 mm | 29 % | 28 % | **62–65 % merged at the 12 mm seed gap** |
| < 12 mm | ~0 | ~0 | **98–100 % merged** |

Two measurements show the merged tracks are recoverable in principle:

- **Split seeding at 6 mm, bench only:** 12–18 mm goes to 48 % (A) / 44 % (C), and
  18–24 mm to 72 % / 70 %. Once a merged pair is cut into two windows, both halves
  fit well. Split seeding was **rejected** because it breaks real single tracks
  (loses 11–27 % of production tracks; real clusters have holes). The joint fit
  should reach at least the split-seeding numbers without that damage.
- **Once found, a separated track fits like a single track** (robust σ of p0
  against its donor ≲ 0.1 mm, same strip count). So what is lost is *finding and
  splitting* the tracks, not fit quality.

On data, the in-situ two-track resolution (`det_a_intra`, STATUS 2026-09-10) is
**0 real intra-A pairs below 20 mm, and 56 below 40 mm where ~12 400 are
expected**. The per-view efficiency is 0.005 at 0–20 mm, and the loss is a
product over the two views. A pair 12 mm apart in u and 300 mm apart in v is lost
just as completely as one 12 mm apart in both. Small-opening-angle pairs
(conversions, low-angle IPC) are exactly the ones that land close together.

**What this fit will not do.** The data's 3–4× y degradation for two-track
chambers (HANDOFF_INTRA §2.1) is H3, busier events (bench §10). The joint fit
recovers tracks that are merged today. It does not make busy events clean.

---

## 2 · Why a merged window fits as one track today

- `wft/seed.py:206` `seed_candidates`: strips within `GAP_THRESHOLD_MM = 12` chain
  into one cluster, and one cluster gives one window
  (`wft/io.py:115` `extract_window`, pad 3 strips).
- `wft/reco.py:273` `fit_plane_candidates` fits each window as **one** track:
  `fit_plane` (`:210`) → `_global_start` (`:133`, ~310 NNLS-profiled χ² evaluations
  over (p0, t0) then (p0, w)) → `wft/model.py:403` `fit_plane_raw` (Nelder–Mead in
  (p0, w, t0)).
- `wft/model.py:276` `build_matrix` gives the (strip × sample, K) design matrix of
  one straight track, K = 18 depth bins of 60 ns. `chi2_plane` (`:362`) solves the
  charge profile q ≥ 0 by NNLS. A two-track window fitted this way gets one
  compromise line, with a wide window and a large χ²/dof.

Nothing in the model assumes one track except `build_matrix` being called once.
The profile is already a free non-negative vector, so **a second track is a second
block of columns in the same linear solve.**

---

## 3 · The model

For one plane window, with θ = (p0, w, t0):

    W  ≈  M(θa) · qa  +  M(θb) · qb ,     qa, qb ≥ 0 (K bins each)

- **Linear part:** NNLS on the stacked `[M(θa) | M(θb)]` (2K = 36 columns), with the
  same noise weighting, saturation censoring and dead/hot handling as `chi2_plane`.
  Reuse `prep_plane` and the censoring code as they are. Do not write a second copy.
- **Outer parameters:** (p0a, wa, t0a, p0b, wb, t0b), 6 in all. Also fit a **tied-t0**
  variant (t0a = t0b, 5 parameters). Two prompt tracks from one vertex reach the
  mesh at the same time to within a few ns. Tying t0 removes a degeneracy, and it
  is the physics hypothesis the intra control is testing. The bench's `offset`
  class (> 150 ns apart) needs the free variant. Choose between them by Δχ² with
  the same calibration as §6.
- **Label order:** require p0a ≤ p0b at the **middle of the drift column**
  (p + w·u_mid), not at the mesh. Crossing tracks swap order at the mesh but
  rarely at mid-column.
- **Kernel:** exactly the production bundle's (`calib_bundle_r06` family,
  `c2_over_c1`; read c2 with `wft.calib.effective_c2`, never `hyper['c2']`).
  The fit adds no calibration constant.

### Degeneracies to handle explicitly

| degeneracy | what it looks like | handling |
|---|---|---|
| **θa ≈ θb** | the two column blocks are identical, so the charge split between them is arbitrary and χ² equals the one-track value | no gain in χ², so model selection rejects it. Also add a minimum-separation guard (§6) so Nelder–Mead cannot collapse the tracks and report a "split" |
| **t0 basins** | each track has the known 60 ns, one-depth-bin basins (p0 slides by w·60); with two tracks that is 4 combinations | multi-start over ±60 ns per track in the free variant. Tied t0 removes most of it |
| **crossing in one view** | an X shape; order swaps with depth | mid-column ordering; test explicitly with synthetic crossings (§7.1) |
| **one track with a hole or a δ-ray** | a real single track that two tracks explain better | this is what broke split seeding. Only the §6 threshold plus the false-split rate on real single tracks can control it |
| **one track fits one track's charge, the other fits noise** | one child has a tiny, patchy profile | require each child to pass the existing plausibility cut (`U_MIN_NS ≤ q_uend ≤ U_MAX_NS`, \|tan\| < `TAN_MAX`) and a minimum `q_sum` fraction |

---

## 4 · When to try it

A 6-parameter fit with multi-start costs several single fits (§10), so it must run
only on a small, well-chosen subset of plane candidates. Any of these triggers an
attempt:

1. **Too wide for one track:** the window's strip count exceeds what one track of
   the fitted |tan| and `q_uend` can cover (transverse extent ≈ |w|·q_uend plus the
   charge spread and ±2 kernel strips), by more than a margin calibrated on clean
   single tracks.
2. **Cross-plane count mismatch:** the other plane has two time-coincident,
   plausible candidates, and this plane has one within `DT_XY_TOL_NS` of both.
3. **Cross-plane charge mismatch:** the `lq = log(q_sum_x / q_sum_y)` pairing feature
   is far out for the best pair. A merged plane carries about twice its partner's
   charge (+0.69 against a median near 0); reuse `xy_pairing`'s median and rsig.
4. **Poor fit:** χ²/dof above a per-arm quantile of clean single tracks. Use it only
   together with 1–3; on its own it selects busy events (H3).
5. **Overlapping windows:** two candidates in one plane whose padded windows overlap
   are fitted **jointly on the union window**, instead of each as one track.

**Measure the trigger rate on real triggers before anything else.** Run the triggers
alone (no fit) over run_145 stat090_0000 and the fixed noisy-side sub-run. Report
the fraction of plane candidates that trigger, per arm. That number sets the condor
cost.

---

## 5 · Initialisation

Nelder–Mead in 6-D from a bad start will fall into the collapsed θa ≈ θb basin.
Start from the one-track fit (p0, w, t0) and use waveforms only:

- **Symmetric splits:** p0a,b = p0 ∓ d/2 for d ∈ {2, 4, 7, 11} mm, with wa = wb = w and
  t0 tied; evaluate χ² on the grid and keep the best 2–3 as starts.
- **Residual-driven:** after the one-track fit, find the strip × sample residual
  (data − model), fit the residual's largest positive lobe with the existing
  one-track `_global_start` restricted to those strips, and use it as track b.
  The one-track fit shifted to absorb the rest is track a.
- **Cross-plane:** when trigger 2 fired, take t0a and t0b from the other plane's
  two candidates through `dt_xy`.

Then Nelder–Mead from each start, keep the lowest χ², and refine once more
(the two-stage pattern `fit_joint` uses, `wft/model.py:458`).

---

## 6 · One track or two

- **Statistic:** Δχ² = χ²(1 track) − χ²(2 tracks), on the same window and the same
  censoring. **χ²/dof is not 1 on this data.** Clean selected single tracks run A
  y 2.2 / x 1.4 and C y 6.8 / x 4.4, and full-pass medians are 5–12. A fixed
  Δχ² threshold from Wilks would split almost everything. Use
  `Δχ² / (χ²₁/dof)` or an empirically calibrated threshold, per arm and plane.
- **Calibration:** the bench already builds the three populations needed:
  - `single`: clean donors re-fitted alone, which gives the false-split rate;
  - `noise`: a donor plus a charge-free trigger's waveforms on the other strips,
    which gives splits caused by extra noise;
  - `overlay`: truth, for efficiency against separation.

  Scan the threshold, plot efficiency at 6–12 / 12–18 / 18–24 mm against the
  false-split rate, and choose the operating point from that curve. Do not choose
  it from one number.
- **Guards applied after the statistic:** both children plausible (§3 table),
  |p0a − p0b| at mid-column ≥ ~1.5 strip pitches, and each child `q_sum` ≥ ~15 % of
  the total. Tune these on the bench, not on data.

---

## 7 · Integration and output

- **Where:** a new `fit_plane_two` in `wft/reco.py` next to `fit_plane`, returning
  two `PlaneFit`s. The linear part belongs in `wft/model.py` (a `chi2_plane_two`
  sharing `chi2_plane`'s internals). `fit_plane_candidates` calls it on triggered
  candidates.
- **Switch:** `WFT_TWO_TRACK_FIT` (0 = off, the default), following
  `WFT_SIG_FLOOR_LOCAL_MM`. Record it, the threshold and the trigger settings in
  both `reconstruct_run`'s and `wft_beam._write_meta`'s `.meta.json`. With the
  switch off, output must be bit-identical to today's (MULTITRACK_09-14 §3.3 check).
- **Candidates:** an accepted split adds two children flagged `split_child`, with a
  shared `split_parent` index and `split_dchi2`. The parent stays in the sidecar
  flagged `split_replaced`. `select_tracks` treats the parent and its children as
  mutually exclusive, so no charge is counted twice. The children then go through
  x/y pairing (`_repair_pairs`, whose `lq`/`u50`/`u90` features come from each
  child's own profile) like any other candidates.
- **Contract:** see §0.1. Add unit tests in `wft/tests/test_multitrack.py` for
  mutual exclusivity and for the "not accepted ⇒ identical" rule.

### 7.1 Build order

1. **Synthetic, no I/O:** generate two-track planes with `wft/model.py` under the
   production bundle, with the bundle's noise; scan separation, Δtan, charge ratio,
   tied/offset t0, and crossings. Unit-test recovery, label order and no split on
   one-track synthetics. Only the logic is being tested here.
2. **Bench variant:** add `--two-track` to `intra_bench build` (the same plumbing as
   `--local-mm`/`--pairing`) and a row to `compare`. Run it on top of
   `pairing_rescue16_ranked`, so the gain is only the joint fit. Refine
   `SEP_BINS` below 12 mm if needed (0/3/6/9/12).
3. **Single-track A/B:** extend `intra_bench floor-ab` (or add `split-ab`). Re-fit
   every production-fitted trigger of run_145 stat090_0000 where any plane
   candidate triggers, and match to the frozen pass. Report: attempts, accepted
   splits, production gated tracks replaced, and clean single tracks split.
4. **Cost benchmark:** core-seconds per arm-event with and without the fit, on the
   same sub-run, with the trigger rate from §4.
5. **Data:** only after 1–4, and on the re-pass. Run
   `pair_vertex_imaging.intra_vertex --multiplicity`, then `det_a_intra`'s
   real/mixed two-track resolution map. Two-track chambers must appear below
   12 mm, and the 0–20 mm per-view efficiency must rise from 0.005.

---

## 8 · Acceptance criteria (proposed — confirm before the bench run)

On the overlay bench, A and C, on top of pairing + rescue:

- **12–24 mm:** at least split seeding's numbers (A 48 % / 72 %, C 44 % / 70 % in
  12–18 / 18–24 mm);
- **6–12 mm:** report the curve. Initial target: half of pairs found and
  correctly paired;
- **≥ 24 mm:** no regression from 71 % / 66 %;
- **split children against their donors:** report robust σ of p0 and tan per bin.
  Proposal: σ(p0) ≤ 1 mm and σ(tan) ≤ 2× the single-track value at ≥ 12 mm;
- **false splits:** ≤ 1 % of `single` and of `noise` overlays.

On real triggers (single-track A/B, run_145 stat090_0000):

- events with no accepted split are bit-identical;
- accepted splits on triggers whose frozen fit is a clean single track (the donor
  selection): ≤ 1 %. For scale, 08-12 measured 1.0–1.5 % of real single muons
  already reporting `n_tracks ≥ 2`, and many were real second particles;
- ~~the change adds ≤ 50 % to an arm job's core time (§10).~~ **Withdrawn
  2026-09-16: compute is not a constraint.**

---

## 9 · Pitfalls already paid for

- **Geometry from waveforms, never hit times** (`CLAUDE.md`). The residual-driven
  start in §5 uses waveform samples, which is allowed. `combined_hits` times are not.
- **Do not lower `GAP_THRESHOLD_MM` or split seeds** to feed this fit. Both failed
  the single-track A/B. The fit should act on the production windows.
- **The A/B compares against the frozen pass, not truth.** It shows production
  answers did not move. It does not show the added tracks are real; that is §7.1 step 5.
- **Calibration is per detector and per run condition.** The trigger thresholds,
  the Δχ² threshold and `xy_pairing` from run_145 are quiet-side-of-access,
  post-23-July-noise. Check one run on the other side of each boundary (run_67 or
  earlier for noise; run_79 is also missing A-x connector 8, channels 448–511 of
  FEU 3) before a campaign pass.
- **Chamber D:** noisy columns and a dead band. Develop on A and C; D's hot-channel
  seeding has not been combined with the rescue floor either.
- **Laptop vs condor:** one C event differs by 4 µm in p0 between the laptop and the
  condor nodes on unchanged code (MULTITRACK_09-14 §3.3). Bit-identity checks
  should be laptop against laptop.
- **`pandas.Series.to_numpy()` may be a view.** Use `copy=True` on any mask that is
  modified in place (the `det_a_intra` bug).

---

## 10 · Cost, and what the condor re-pass will take

> **Superseded as a constraint, 2026-09-16.** Dylan: treat condor as effectively
> unlimited — a re-pass that takes a month is fine. The numbers below are kept for
> sizing only; nothing in the algorithm should be traded for them.

**Measured on the 2026-09-09 full pass** (HANDOFF_FULLPASS §3–4, STATUS 2026-09-09/10):
12 932 jobs = 3 233 tags × 4 arms, `workday`, 2 CPUs, ~1.2 core-h/job, ~16 000
core-hours. Median job 32 min, p90 50, max 53; D 76 min in the smoke gate.

| step | 2026-09-09/10 |
|---|---|
| smoke gate (4 jobs, one per arm) | 21:10 decision → 22:27 pass (~1 h 15) |
| condor | 22:27 → **06:31 (8 h)** |
| pull + unpack (28 GB) | → 07:19 (48 min) |
| `merge_campaign` + `fullpass_chain` (k_arm per run, `campaign_tracks`, QA) | → 10:14 (~3 h) |
| **decision to track database** | **~13 h** |

**Estimate for a re-pass:**

- **Pairing + rescue only:** rescue changes seeds in 1 319 / 1 726 of ~22 400 fitted
  triggers per arm (6–8 %), and each adds about one plane fit. Pairing costs almost
  nothing. So **+3–8 % core time, ~8–9 h on condor, ~14 h end to end**, if the queue
  is as clear as that night (a 2026-09-07 probe of 8 jobs started in 78 s).
- **With the joint fit:** unknown until §7.1 step 4. A rough scaling: a two-track χ²
  evaluation is ~2× the columns on a wider window, so 3–6× one evaluation; with
  multi-start, one attempt is roughly 5–10× one plane fit. At a 5 % attempt rate
  that is +25–50 %, so **~10–12 h on condor**. Jobs stay far inside `workday`.
  The condor time roughly follows core-hours because the pool ran ~2 000 cores
  in parallel.
- **A and C only:** the change is validated only there. B and D could keep their
  frozen products, at ~40 % of the core-hours (~4–5 h). But the re-merged table
  would then mix reco versions per arm, so stamp it.

**Before submitting**, none of which exists yet:

- `xy_pairing` into each run's bundles. `wft_beam.make_bundle` has no merge step for
  it (MULTITRACK_09-14 §5), and pairing is per run condition, so it needs 36
  calibrations from each run's own stage-3 tracks;
- the env vars in `stage2_fullpass.sub`'s `environment` line;
- a new smoke gate. The existing one requires event counts equal to the August
  pass, and this change adds tracks by design. Gate on "production candidates
  reproduced, extra tracks counted" instead;
- archive `<out>/reco_fullpass` and `stage3_fullpass` (40 GB unpacked) or write to
  new trees. `/media/dylan/data` had 113 GB free on 2026-09-10.

---

## 11 · Where things are (`origin/main`)

| what | where |
|---|---|
| forward model, NNLS profile, one-track fit, two-plane `fit_joint` | `wft/model.py:276`, `:362`, `:403`, `:458` |
| plane fit, global start, candidate fit, track selection, x/y repair, worker | `wft/reco.py:210`, `:133`, `:273`, `:326`, `:414`, `:570` |
| seeding, gap, rescue floor | `wft/seed.py:206`, `:30`, `:108`; beam seeder `ntof_tracking/wft_beam.py:333` |
| bench: overlays, scoring, outcomes, single-track A/B | `sept26_prelim_analysis/intra_bench.py:226`, `:452`, `:538`, `:787` |
| bench reports | `<out>/intra_bench/<variant>/report.html` (`pairing_rescue16_ranked` is the baseline) |
| tests | `wft/tests/test_multitrack.py`, `wft/tests/test_seed_and_select.py` |
| condor | `sept26_prelim_analysis/condor/stage2_fullpass.sub`, `run_stage2_fullpass_wrapper.sh`, `overnight_fullpass_2026-09-09.sh` |
| after condor | `sept26_prelim_analysis/fullpass_chain_2026-09-10.sh` |


---

## 12 · Progress, 2026-09-16

**Written, measured, off by default.** Record: `wft/TWO_TRACK_FIT_2026-09-16.md`;
working log and to-do: `TWO_TRACK_FIT_LOG.md`; report: `<out>/two_track/report.html`.

On the overlay bench, on top of x/y pairing + the rescue floor, time-coincident
pairs, both found *and correctly paired*:

| separation, both views | production | + pairing & rescue | **+ joint fit** |
|---|---:|---:|---:|
| **< 12 mm**, A / C | 0 / 0 % | 0 / 0 % | **18 / 37 %** |
| **12–24 mm** | 21 / 15 % | 38 / 29 % | **43 / 45 %** |
| ≥ 24 mm | 44 / 37 % | 73.3 / 67.8 % | 72.8 / 67.8 % |

That clears §8's 12–24 mm bar on C (45 % against split seeding's 44 %) and misses
it on A (43 % against 48 %); 6–12 mm is below the "half of pairs" target.
Real clean single muons split at 0.25–0.37 % (A) / 0–0.66 % (C), inside the ≤ 1 %
criterion. Events with no accepted split are bit-identical to 99.8 % / 99.1 %.

**Four things the measurements changed about §3–§6**, each a measurement:

1. **The starts.** A merged window's one-track fit belongs to neither track;
   searching only for track *b* from it missed it above ~9 mm. Replaced by
   alternating matching pursuit (§5's residual-driven start, run both ways and
   iterated). The residual scan also needs the p0–slope shear.
2. **The statistic** is the *smaller* of the two marginal Δχ², not the total.
   §6's Δχ² is large whenever the second block absorbs anything at all.
3. **The guard** is the separation measured at the depths where *both* children
   carry charge. §3's minimum-separation guard does not catch the failure that
   actually matters — one track cut in half end to end — and §6's `q_sum`
   fraction is unusable because unconstrained depth bins carry runaway charge.
4. **t0 tied by default.** §3 proposed choosing between tied and free by Δχ²;
   free walks into the one-depth-bin degenerate minima and splits perfectly
   modelled single tracks, and nothing measured gives it an advantage.

**§8's cost criterion is withdrawn (Dylan, 2026-09-16): compute is not a
constraint** — condor is effectively unlimited and a month-long re-pass is
acceptable; the goal is the most effective algorithm. For the record, the fit as
built adds +115 % (A) / +223 % (C) to an arm job. Several choices made to keep
that down are now the first things to undo; the largest, the per-plane trigger,
is measured to discard two thirds of the recoverable pairs at 0–6 mm. The list is
`wft/TWO_TRACK_FIT_2026-09-16.md` §8 and the log's to-do.

**Two decisions of §0 answered by the work:** the contract is "a split replaces
its parent, the parent stays in the sidecar at `rank` −1, and a split that would
cost the event a track is reverted"; and the fit should ship in the same re-pass
as pairing and the rescue floor, since the bench numbers above are only reachable
on top of them.
