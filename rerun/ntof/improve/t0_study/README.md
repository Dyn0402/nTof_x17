# Out-of-time tracks, pile-up and accidental same-chamber pairs (study, started 2026-11-23)

Dylan asked whether two uncorrelated particles in one readout window (statistical pile-up) are reconstructed as if
they were coincident, which would bias the same-chamber pair results. This study answers that question with the data
we have: the pass-1 tracks of the rebuilt analysis view (`stage3_r2_final`, 293 production sub-runs, 8.3 M gated A/C/D
tracks) and the scintillator wall hits of the slim files, which cover ±1000 ns around each trigger.

**Short answer so far.**
- The chamber's fitted start time t0 does not follow a late-arriving particle. A track that arrives up to 700 ns late
  is fitted with a t0 only 5-40 ns late, because the fit absorbs the delay into the depth profile. An early track
  pulls t0 earlier, but only to the ±180 ns edge, where 30-45 % of such tracks rail.
- The column centre, tc = (t0 − pin) + u50, does follow the arrival time, with a slope of about 1 from −300 to +700 ns
  on A and C. Its spread on in-time tracks is ±110 ns (A) and ±155 ns (C).
- With tc, the time difference between the two tracks of a same-chamber pair measures the accidental pairs directly.
  From 25 to 75 ms, 19 ± 1 % (A) and 18 ± 2 % (C) of the pairs without a split child are accidental. The fraction falls
  with the time since the flash, as accidentals should: on A it is 25 % at 25-35 ms and 11 % at 50-75 ms.
- Taken over all pairs (30 % of them contain a split child, which is in time), that is about 13 % (A) and 12 % (C).
  The same-chamber rate fit gives 7 % (A) and 13 % (C). So C agrees, and A's accidental share appears to be about twice
  what the rate fit subtracts.
- An injection test (real in-time tracks shifted by known amounts and added to other triggers) confirms the response
  directly. On A, tc equals the true offset within 10 ns from −120 to +420 ns; t0 moves by at most 90 ns. The track
  angle changes by less than 0.5° within −120 to +300 ns, so accidental pairs keep their opening angle there.
- The injection test also found that two separated in-time tracks are both found only 76 % (A) and 71 % (C) of the
  time. Most of the loss comes from the seeder's plane-wide significance floor. A 30 mm local floor raises this to
  87 % and 84 % (roadmap 4.12).

## How the reconstruction treats the start time

Each view's fit has a start time t0. The trigger tells us when the triggering particle crossed the chamber, through a
fixed measured delay, the pin. The fit has a weak prior at the pin and a hard limit at pin ± 180 ns. A track whose fit
reaches that limit is called railed (`t0_railed`, |t0 − pin| ≥ 175 ns).

On late (≥ 25 ms) single tracks that fired their own wall, t0 − pin is −26 ns at the median with a spread of 24 ns
(A) and 35 ns (C). Railing is rare on them: 0.4-0.6 % per view on A and C (`look1.py`).

## Railing in pairs is mostly the two-track split

Railing is ten times more frequent in same-chamber pairs (13-16 % of late pairs have a railed track) than in single
tracks. The reason is the two-track split. 30 % of the late pairs contain a split child, which is one cluster that the
two-track fit divided into two tracks. These children ignore the t0 pin (roadmap 4.3), and 27-28 % of them rail. The
pairs without a child rail at 2 % per track (`look2.py`).

## The chamber's response to the arrival time, measured with the wall

The wall gives each track a time. A wall hit is the two ends of one group firing within 15 ns. Its time has a
resolution of about 6 ns under beam. For each track we take the wall hits of its own group (`wall_time.py`).
- 74 % (A), 83 % (C) and 65 % (D) of the late single tracks have an in-time hit in their own group.
- 0.4 % have no in-time hit but exactly one off-time hit. Many of these tracks still point at the source (0.25-0.42
  within 15 mm, against 0.58-0.62 for in-time tracks), so most are real particles from another capture.
- The off-time wall hits are not symmetric about the trigger. There are more after it than before it. This is
  expected, since an earlier particle would itself have fired the trigger.

For those tracks (`look4.py`, `look6.py`), the median response to an arrival offset Δ is:

| Δ (ns) | −400 to −300 | −180 to −120 | +60 to +120 | +180 to +300 | +400 to +500 | +500 to +700 |
|---|---|---|---|---|---|---|
| A: t0 − pin | −118 | −77 | −7 | +7 | +39 | +41 |
| A: tc shift | −250 | −133 | +66 | +230 | +441 | +536 |
| C: t0 − pin | −120 | −82 | −9 | +28 | +28 | +28 |
| C: tc shift | −212 | −125 | +72 | +214 | +379 | +453 |

These medians are diluted by tracks whose off-time wall hit is unrelated to them, so the true response is somewhat
steeper. The column end (t0 + u90, `look5.py`) also follows late arrivals, but it stops at the 1200 ns end of the
readout window.

## The accidental fraction of the same-chamber pairs, from timing

`look9.py` fits the |Δtc| distribution of the non-child pairs as a sum of two templates, both built from data.
- **In time:** two in-time single tracks, mixed.
- **Accidental:** one in-time track, and one track whose arrival offset is drawn from the density of off-time
  own-group wall hits. That track's tc is drawn from the off-time tracks of the nearest offset bin.

| ToF (ms) | 12-16 | 16-25 | 25-35 | 35-50 | 50-75 | 25-75 |
|---|---|---|---|---|---|---|
| A accidental fraction | 0.84 ± 0.02 | 0.38 ± 0.02 | 0.25 ± 0.02 | 0.21 ± 0.02 | 0.11 ± 0.02 | 0.19 ± 0.01 |
| C accidental fraction | 0.60 ± 0.04 | 0.32 ± 0.03 | 0.24 ± 0.03 | 0.16 ± 0.03 | 0.11 ± 0.03 | 0.18 ± 0.02 |

The fit is good from 16 ms on (χ² 13-34 for 23 degrees of freedom) and poor at 12-16 ms on A (176/23). At 12-16 ms
the pairs are mostly accidental. This agrees with finding 9 of `../README.md`, which saw a different population there.

**Cross-check of the in-time template.** Pairs whose two tracks both have an in-time hit in their own (different) wall
groups are in time. Only 6-7 % of them (A) have |Δtc| > 300 ns, against 17.5 % of all A pairs from 25 ms on
(`look8.py`). On C these numbers are 18-22 % and 23 %, since C's tc resolution is worse.

**What this means for the rate fit.** The correlated pairs per 1000 single tracks that follow from this are flat in
ToF on A: 2.26, 2.27 and 2.44 per 1000 at 25-35, 35-50 and 50-75 ms. This is what correlated pairs should do, so the
timing picture is self-consistent. However, the rate fit's accidental share for A (7 %) is about half of this one
(about 13 % of all pairs). We do not yet know which is right.

## The injection test (2026-11-23)

The wall-based numbers above are diluted, because some off-time wall hits are unrelated to the track. So we also
measured the response directly, with an injection test (`inject.py`, analysis `inject_ana.py`). The test uses one
sub-run, run_145 stat090_0000, on chambers A and C.

**How the test works.** A donor is a real late single track (ToF ≥ 25 ms). It must be the only gated track of its
chamber, have one seed cluster per view, not be railed or a split child, and have an in-time hit in its own wall group.
The last condition means that it arrived with the trigger. We cut the donor's waveforms out on its seed cluster plus two
strips on each side. We shift them by k whole samples (60 ns each, so the shift is exact) and add them to the
waveforms of a host trigger. The host is then reconstructed with the pass-1 configuration (the E2_beam bundle, minimum
3 strips). There are three kinds of host.
- An **empty** host is a late trigger with no seed in the chamber. Each of 300 donors goes into one empty host at each
  of 19 shifts, from −360 to +720 ns.
- A **pair** host is a late trigger that already holds another clean in-time single track. Each of 1500 pair hosts gets
  one donor at a random shift. This makes an accidental same-chamber pair whose true time offset we know.
- A **control** host is an empty host with nothing added. None of the 300 control hosts made a gated track, on A or C.

Three details of the method matter.
- The true offset is delta = 60 k + pin(ftst of the donor) − pin(ftst of the host). The pin moves by 10 ns per ftst
  tick.
- The donor's charge keeps the donor's x–y trigger phase. With the same k in both views, 45 % of the injected tracks
  would carry a 60 ns x–y column-time error. Since that error is exactly one sample, the y view gets its own shift
  k_y = k ± 1 where needed, so both views see the same true offset.
- On the injected strips, the analyzer hits are replaced by hits from the injected waveform (5σ, as
  `wft_beam.hits_from_waveforms`). Adding a second event's waveform also raises the noise on those strips by √2. The
  k = 0 injection measures what these two changes do by themselves. It gives an acceptance of 0.97 and a tc spread
  (16-84 %) of ±30 ns about the donor's own value on both chambers.

### Single tracks: the response to the arrival time

Each row compares the injected track with the same donor reconstructed in its own event, so the donor-to-donor spread
cancels. Δt0 and Δtc are medians of (injected − donor). dθ is the median angle between the injected and the donor
direction.

| delta (ns) | −360 | −240 | −120 | 0 | +120 | +240 | +360 | +480 | +600 | +720 |
|---|---|---|---|---|---|---|---|---|---|---|
| A acceptance | 0.80 | 0.92 | 0.95 | 0.97 | 0.97 | 0.95 | 0.92 | 0.84 | 0.81 | 0.75 |
| A Δt0, x view | −115 | −100 | −61 | −5 | +32 | +54 | +73 | +83 | +87 | +70 |
| A Δtc | −281 | −209 | −111 | −1 | +118 | +236 | +358 | +469 | +565 | +638 |
| A railed | 0.55 | 0.40 | 0.05 | 0.00 | 0.01 | 0.12 | 0.18 | 0.21 | 0.23 | 0.16 |
| A dθ (°) | 2.9 | 1.3 | 0.46 | 0.20 | 0.29 | 0.37 | 0.57 | 0.96 | 1.5 | 2.6 |
| C acceptance | 0.89 | 0.93 | 0.95 | 0.97 | 0.92 | 0.90 | 0.86 | 0.84 | 0.74 | 0.58 |
| C Δtc | −248 | −192 | −109 | −1 | +122 | +237 | +339 | +415 | +458 | +544 |
| C railed | 0.52 | 0.44 | 0.15 | 0.00 | 0.01 | 0.05 | 0.06 | 0.05 | 0.05 | 0.04 |
| C dθ (°) | 2.5 | 1.1 | 0.47 | 0.21 | 0.35 | 0.56 | 1.06 | 1.6 | 2.4 | 3.8 |

What this shows.
- **tc measures the arrival time.** On A, Δtc equals delta within 10 ns from −120 to +420 ns. Beyond +480 ns it falls
  behind (by 80 ns at +720 ns), because the end of the column leaves the 1200 ns window. On the early side, tc
  follows 75-80 % of the offset. On C, tc follows the offset to +300 ns and then falls behind faster, with a wider
  spread. So the wall-based table above was indeed diluted, and the true response is steeper, as expected.
- **t0 does not measure the arrival time.** A late track moves t0 by only about a quarter of the offset, and never by
  more than 70-90 ns (A) or 50-75 ns (C). The fit puts the rest of the delay into the depth profile.
- **The angle is preserved near the trigger.** Within −120 to +300 ns, the direction changes by less than 0.5°. It
  changes by 1° at about −240 and +420 ns (A), and by 2.5-3.8° at the edges of the range. A late track reads
  shallower (|tan| × 0.86 at +720 ns), and an early track on A reads steeper (|tan| × 1.18 at −360 ns).
- **Acceptance falls slowly.** A track 300 ns late is still found 94 % (A) and 86 % (C) of the time. So accidental
  tracks from about ±500 ns around the trigger enter the sample.

### Accidental pairs: the time difference and the opening angle

In the pair hosts, the two reconstructed tracks are matched to the donor and the host view by view, since the x–y
combination of two tracks can swap (the swap leaves Δtc unchanged, and the opening angle almost unchanged).
- The opening angle of an accidental pair is close to its true value when the offset is small. Within −120 to
  +300 ns, the median change is below 0.2° and 68 % of the pairs change by less than 0.5°. At the edges of the range,
  the median change is 1-1.5° and 68 % change by less than 2-3°. **So an event-mixed shape built from in-time tracks
  describes the accidental pairs within ±300 ns well.** Beyond ±300 ns it is less exact, but the |Δtc| < 300 ns cut
  removes most of those pairs anyway: the fraction with |Δtc| > 300 ns rises from about 0.1 (A) and 0.2 (C) at zero
  offset to 0.66 (A) and 0.51 (C) at +360 ns.

### Finding: a second track is often lost, mostly by the seeder

Two in-time tracks more than 50 mm apart should both be found about 0.97² ≈ 94 % of the time if they were independent.
Instead, both are found 76 ± 3 % of the time on A and 71 ± 3 % on C (|k| ≤ 1). Over all shifts the numbers are 66 %
and 60 %. Below 50 mm apart, the two tracks merge, as expected.

We looked at the events where one of two well-separated tracks was lost. Two causes appear.
- **The seeder does not offer the lost track's cluster in one view** (about two thirds of the cases). The cause is
  the significance floor in `wft/seed.py`: a strip is kept only if it is above 10 % of the strongest strip **in the
  whole plane** (`SIG_REL_FLOOR`, `SIG_FLOOR_LOCAL_MM = 0`). A bright track therefore pushes a fainter one anywhere in
  the plane below the 3-strip minimum. The bench intra-chamber study found the same thing in September
  (`sept26_prelim_analysis/intra_bench.py`), but the local floor never reached production.
- **Both views are fitted, but the x–y association builds only one track** (the rest). The two views are combined
  across the tracks, and the leftover combination then fails the x–y coincidence.

We tested the first cause directly. We reconstructed the same pair hosts, with the same donors and shifts, with the
floor judged against the strongest strip within 30 mm (`--floor-local-mm 30`). Both tracks are then found 87 ± 3 %
(A) and 84 ± 3 % (C) of the time at |k| ≤ 1, and 77 % and 76 % over all shifts. So the plane-wide floor costs
roughly 11-14 points of pair acceptance for two separated tracks.

What this means. The same-chamber pair yield is reduced by this loss. A comparison with Geant4 accounts for it only if
the digitised Geant4 events go through the same seeder with realistic relative amplitudes. We have not checked that
yet. The local floor's effect on single tracks (fakes, noise clusters) has not been measured either, so it is a
candidate for phase 4 (roadmap 4.12), not a decision.

### What the injection test does not cover

- The donors are clean tracks with a wall hit, so they are biased toward the wall acceptance and toward ordinary
  track energies.
- For k < 0, the samples the shift opens at the end of the window are filled with the donor's last sample. So the
  early side is approximate at the end of the window.
- Only one sub-run (run_145 stat090_0000) and chambers A and C are done.
- The accidental fraction of the data has not been redone with this response yet (next: the rate fit with
  |Δtc| < 300 ns, and A's 7 % against about 13 %).

## What this study does not cover yet

- The accidental template takes its shape from tracks selected by an off-time wall hit. That selection can bias it
  (wall acceptance, energy). The injection test above now gives the response directly; the |Δtc| template of
  `look9.py` has not been rebuilt from it yet.
- Chamber D is not done: its fakes need the `d_faint` view and its oscillating channels first.

## Possible mitigations (not decided)

- A pair-level time cut |Δtc| < 300 ns keeps 93 % (A) and 81 % (C) of in-time pairs and removes about half of the
  accidentals.
- Per-track wall time in the analysis view (from the full ±1000 ns slim range), to confirm in-time tracks where the
  wall fired.
- In a pass 2, a per-track arrival time from the stored depth profile (roadmap 4.11), the children pinned (4.3), and an
  upper column-end bound tied to the gap.

## Files

- `collect.py` writes `tracks_t0.parquet` (t0 − pin per view, per gated track).
- `wall_time.py` writes `wall_hits.parquet` (own-group wall hits within ±1000 ns).
- Study scripts: `look1.py` to `look9.py` (fixed paths, output to stdout).
- `inject.py` (the injection, about 25 min per chamber and sub-run on 7 cores) and `inject_ana.py` (matching and
  tables). Outputs: `$WORK_DATA/ntof_pass1/ana/t0_study/inject/` (`ana_prod.txt`, `ana_local30.txt` hold the tables).
- Outputs: `$WORK_DATA/ntof_pass1/ana/t0_study/`.
