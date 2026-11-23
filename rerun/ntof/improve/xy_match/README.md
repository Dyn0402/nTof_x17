# The x/y matching of the n_TOF tracks (study, 2026-11-23)

Dylan asked whether the x/y matching can be fixed, for example by trying every combination of x and y candidates and
judging them with other measured quantities. This study answers that question on chambers A, C and D, on the five
sub-runs of the ringing study (run_145 stat090_0000-0002, run_104 stat090_0003, run_139 stat090_0003).

**Short answer.**
- The matching can be redone from the tables alone. A new reconstruction is not needed for it.
- A better matching rule does not make better tracks. Where an n_TOF-calibrated likelihood disagrees with pass 1,
  neither choice points at the source. These events are made of fake candidates in both views, and no pairing of
  them gives a real track.
- In real events with two tracks in one chamber, the likelihood swaps the y views in 5 % (A) and 9 % (C) of the
  late events. The swap moves the opening angle by a median of 0.1-0.2°. So the X17 observable is almost blind to the
  x/y matching.
- We therefore keep the pass-1 matching. The two useful products are a per-track matching score (a quality flag)
  and, for a pass 2, the depth profile of every candidate in the candidates table.

## How the matching works in pass 1

The reconstruction fits the x and the y view of a chamber separately (`wft.reco._worker_fit`). It writes every
fitted candidate to the stage-2 candidates table, up to five per view. Only after the fits does `select_tracks`
decide which x goes with which y. The n_TOF bundles use mode `cost`:
- With two or more candidates in both views, the matching of least summed `xy_pair_cost` is taken. The cost uses the
  charge balance ln(q_x/q_y), the column times u50 and u90, and the column-time difference tc. Its widths were
  measured on bench cosmics.
- With one candidate in a view, the partner is chosen by mode `gate`. That is the coincident and plausible
  combination with the largest summed Δχ². The charge balance takes no part in that choice.
- The time-coincidence tolerance is the bench value: 197 ns (A), 456 ns (C) and 726 ns (D).

Since every candidate is stored with its time, charge and timing quantiles, any other matching rule can be replayed
on the tables. The depth profile is not stored, so the profile feature (`profile_distance`) cannot be replayed.

## What separates a right pair from a wrong one

`build_combos.py` writes every x-y combination of every event (1.5 M combinations in 12 s). We take two samples.
- **Right pairs:** late (≥ 30 ms) events with one candidate in each view, the wall fired and both views plausible.
- **Wrong pairs:** the x of one such event with the y of the next event of the same tag. Both are timed by their own
  trigger, as two particles of one capture would be.

The separation of each feature is measured by its AUC (1 = perfect, 0.5 = none):

| feature | A | C | D | width on right pairs (A / C / D) |
|---|---|---|---|---|
| charge balance ln(q_x/q_y) | 0.91 | 0.88 | 0.87 | 0.077 / 0.126 / 0.216 |
| column-time difference tc | 0.83 | 0.80 | 0.79 | 40 / 73 / 88 ns |
| u50 difference | 0.75 | 0.75 | 0.73 | 67 / 98 / 123 ns |
| u90 difference | 0.62 | 0.69 | 0.69 | |
| t0 difference | 0.54 | 0.56 | 0.56 | |

The charge balance is the strongest feature. The bench time tolerance is 5-8 times wider than the n_TOF time
resolution, so it rejects almost nothing.

**Overlay test** (`overlay_test.py`). We take two right pairs from different events and ask whether a rule keeps
the true assignment or swaps the y views. The rule is calibrated on half of the events and tested on the other half.

| rule | A | C | D |
|---|---|---|---|
| pass-1 bench cost | 0.962 | 0.952 | 0.922 |
| n_TOF widths, same features | 0.965 | 0.960 | 0.934 |
| n_TOF likelihood (tc, lq, u90, uend) | 0.976 | 0.969 | 0.958 |
| charge balance alone | 0.926 | 0.904 | 0.915 |

So the n_TOF likelihood is 1-4 points better than the bench cost on clean tracks.

## Does the n_TOF likelihood change pass 1, and is it better?

`llr_assign.py` replays the matching with the n_TOF likelihood (the best assignment of up to three pairs,
`scipy.optimize.linear_sum_assignment`). It changes many early tracks and few late ones. Below is the fraction of
pass-1 gated tracks it would pair differently (`repair_count.py`):

| ToF (ms) | 1-3 | 3-5 | 5-8 | 8-12 | 12-16 | 16-20 | 20-30 | ≥ 30 |
|---|---|---|---|---|---|---|---|---|
| A | 0.42 | 0.37 | 0.30 | 0.15 | 0.033 | 0.008 | 0.005 | 0.004 |
| C | 0.41 | 0.38 | 0.30 | 0.16 | 0.026 | 0.007 | 0.005 | 0.006 |
| D | 0.24 | 0.36 | 0.49 | 0.51 | 0.44 | 0.39 | 0.32 | 0.064 |

We check the two kinds of change against measurements that do not take part in the matching (`validate.py`,
`validate_by_gain.py`).
- **The x view changes.** The x view alone sets where the track points in x, so we compare the pointing excess at
  the source (within 15 mm, over a flat sideband). On A at 1-8 ms the pass-1 x tracks have an excess of 0.02 and the
  likelihood's x tracks 0.00-0.01. The unchanged tracks there have 0.03-0.06, and clean late tracks 0.57. This
  stays true when we keep only the changes the likelihood is most sure of (gain of 5-10 in log-likelihood). C and D
  show the same.
- **The y view changes.** The wall measures the position along its bars from the light at its two ends. That is a
  measurement of y. On A at 5-8 ms the pass-1 y agrees with the wall (within 40 mm) in 0.30 of the changed tracks,
  and the likelihood's y in 0.12. The unchanged tracks agree in 0.44. C and D show the same.

So in the events where the two rules disagree, the pass-1 choice is a little better, and neither choice is a real
source track. These events hold fakes in both views. The ringing is coherent between the x and y views of a chamber
(correlation 0.79 on A, `rerun/ntof/ringing/README.md`), so two ringing fakes arrive at the same time with similar
charge. A pure consistency test cannot tell them from a real pair. Pass 1 also uses Δχ², which measures how
track-like each candidate is, and that information is what the likelihood lacks.

On D the changes are many at every ToF. They come from D's ~50 oscillating channels per view (ringing study, item
1), which make fakes in both views at all times. Re-pairing does not fix them.

## Two tracks in one chamber

On the five sub-runs there are 219 (A) and 162 (C) events at ≥ 12 ms with two pass-1 gated tracks and a 2x2
alternative. The likelihood swaps the y views in 10 (4.6 %) and 15 (9.3 %) of them (`two_track_angle.py`).
- The swap moves the opening angle by a median of 0.1° (A) and 0.2° (C), and 0.2° / 0.7° at the 90th percentile.
  The median opening angle of these pairs is 28°.
- The reason is geometric. The cosine of the opening angle is (1 + tx1 tx2 + ty1 ty2) / (|v1| |v2|). Exchanging ty1
  and ty2 leaves the numerator unchanged and changes only the norms, a second-order effect.
- For the same reason a vertex built from the two lines does not move. The x view fixes where the x projections
  cross, the y view fixes where the y projections cross, and neither changes under the swap.

What a swap does change is which particle is which in 3D. That matters for a per-particle quantity: the
scintillator energy of each track, or a per-track 3D direction. It does not matter for the opening angle, the
vertex or the summed energy.

## The likelihood as a quality cut

Used as a cut on the pass-1 tracks instead of for re-pairing (`llr_as_cut.py`), the likelihood behaves like the
existing `xy_balance_ok` flag. With LLR > -2:
- It keeps 97 % (A) and 96 % (C) of the late wall-confirmed tracks.
- The purity at 12-16 ms goes from 0.69 to 0.74 (A) and from 0.74 to 0.78 (C). The 8-12 ms purity goes from 0.31 to 0.38
  (A) and from 0.34 to 0.46 (C).
- On D it does not raise the purity at all.

This is the same size of gain as `xy_balance_ok` (0.82 to 0.85 at 12-16 ms, at a cost of 2-3 %), so the two are not
independent. We do not propose a further cut.

## What we recommend

1. **Keep the pass-1 matching.** A consistency-only rule is not better where it disagrees, and the disagreements are
   junk events.
2. **Give every track its matching score at table level** (a stage-3 column with the n_TOF likelihood, and the gain
   of the best alternative assignment). An analysis that needs the identity of each particle (per-particle
   scintillator energy) can then require a confident matching. Every combination is already in the candidates
   table, so this needs no reconstruction.
3. **Use the n_TOF time tolerance** (`TC_TOL_NTOF` in `quality_join.py`: A 151, C 296, D 535 ns) in place of the
   bench value, in the n_TOF bundles' `xy_pairing`, if a pass 2 is done.
4. **For a pass 2: store each candidate's depth profile** (20 bins) in the candidates table. The profile is the one
   feature that cannot be replayed. On run_145 donor pairs it raised the correct keep/swap decision from 87 to 92 % (A)
   and from 89 to 93 % (C) (`wft.reco.profile_distance`, 2026-09-30). The cost is small, and it would let any later matching
   rule use it.

## What this study does not cover

- The overlay test uses tracks from different events, which do not share strips. Real two-track events can share
  strips in one view, where the charge is split between the two fits.
- The wall's position resolution is 30-55 mm (median |residual| on late single tracks), so it is a weak test of a
  single y choice.
- The likelihood was calibrated on late tracks only. Its widths at early ToF are wider, because the ringing widens
  the charge balance.

## Files

- `build_combos.py`: every x-y combination and the mixed sample, per sub-run (`--subruns production` for all).
- Study scripts (fixed paths): `overlay_test.py`, `repair_count.py`, `llr_assign.py` (writes `ana4.pkl`),
  `validate.py`, `validate_by_gain.py`, `two_track_angle.py`, `llr_as_cut.py`.
- Outputs: `$WORK_DATA/ntof_pass1/ana/xy_match/` (`combos/`, `mixed/`, `ana4.pkl`).
