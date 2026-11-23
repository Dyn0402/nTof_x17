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

## What this study does not cover yet

- The accidental template takes its shape from tracks selected by an off-time wall hit. That selection can bias it
  (wall acceptance, energy). An injection test is still to do: real late tracks shifted in time by known amounts,
  added to empty events and reconstructed. It would give the exact response and acceptance.
- The opening-angle shape of the accidental pairs has not been compared with the event-mixed shape yet. Early
  arrivals rail and read 14-33 % too steep, so the two shapes may differ.
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
- Outputs: `$WORK_DATA/ntof_pass1/ana/t0_study/`.
