# HANDOFF — recovering two tracks in one chamber (A–A, C–C)

**Written 2026-09-13.** The measurements below are campaign-wide (33 runs of the
condor full pass, stage-3 tracks). Nothing in this document has been tried in
the reconstruction yet; it is the case for doing so and a plan for how.

Companions:

- `ntof_athens_26/pair_vertex_imaging/README.md` — the analysis that found this,
  and `intra_vertex.py`, which produces every number quoted here.
- Note: *Two-track vertices in 3D, and same-chamber pairs* —
  <https://dylan-neff.web.cern.ch/notes/pair-vertex-3d-intra.html>.
- `wft/reco.py` (`fit_plane_candidates`, `select_tracks`), `wft/seed.py`
  (`GAP_THRESHOLD_MM`), `wft/tests/test_multitrack.py`.
- `CLAUDE.md` → *Reconstruction basis*: geometry comes from the waveforms, never
  from hit times. Hits are for seeding only. Everything below respects that.

---

## 1 · Why this matters

Same-chamber pairs are about a third of the two-track sample (A–A, C–C and
D–D together were 19,707 of ~62,700 real pairs at the published 30 mm cut).
They are the only pairs where both legs are measured by one chamber on one
calibration, which makes them the natural place to ask whether two tracks come
from **one vertex** — and they are where the small-opening-angle backgrounds
(conversions, low-angle IPC) live.

That question cannot be answered today. The analysis tried three
event-mixed-controlled tests (§3) and all three are blind — not because nothing
is there, but because **a track that shares its chamber with a second track is
badly measured**, and **two tracks close together are not reconstructed as two
at all**. Dylan expected the x–y pairing ambiguity and cluster merging to make
these events hard; the data confirm both are plausible and show the damage is
larger and more general than local merging alone.

The goal: **two-track events that reconstruct as well as single-track events,
without moving single-track events at all.** It likely needs a dedicated
multi-track reconstruction, developed and validated on a sample with known
truth.

---

## 2 · The symptoms, measured

### 2.1 A second track in the chamber degrades both

Tracks gated, angle-calibrated, slope-measured (|tan| ≥ 0.08) and not in a noisy
column in **both** views, pointing within 30 mm of the measured capsule position
in the transverse plane. "Tracks in the chamber" counts every gated track of that
chamber in the trigger, before those cuts.

| chamber | tracks in chamber | tracks | y at capsule depth, robust σ | x at capsule depth, robust σ | y strips | x strips | χ²/dof y | χ²/dof x | > 1 y candidate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 759,847 | **41 mm** | 16 mm | 15 | 16 | 2.2 | 1.4 | 7 % |
| A | 2 | 25,292 | **139 mm** | 23 mm | 35 | 41 | 12.8 | 11.8 | 100 % |
| A | 3+ | 5,689 | 168 mm | 24 mm | 38 | 46 | 15.3 | 13.4 | 100 % |
| C | 1 | 465,421 | **48 mm** | 18 mm | 17 | 17 | 6.8 | 4.4 | 17 % |
| C | 2 | 44,813 | **170 mm** | 24 mm | 51 | 53 | 15.1 | 17.8 | 100 % |
| C | 3+ | 8,687 | 191 mm | 25 mm | 50 | 54 | 14.7 | 18.1 | 100 % |

`intra_multiplicity.csv`. y degrades 3–4×, x about 1.4×; the fits in both views
take 2–3× the strips at 6–10× the χ²/dof.

### 2.2 It gets WORSE as the two tracks move apart — so it is not charge overlap

For chambers with exactly two tracks, against the two tracks' separation on the
strip plane (`intra_twotrack_separation.csv`):

| chamber | view | separation | tracks | robust σ at capsule | median strips |
|---|---|---|---:|---:|---:|
| A | y | 16–24 mm | 120 | **79 mm** | **19** |
| A | y | 24–40 mm | 1,387 | 142 mm | 28 |
| A | y | 40–80 mm | 5,229 | 146 mm | 35 |
| A | y | 80–400 mm | 18,539 | 137 mm | 35 |
| C | y | 16–24 mm | 84 | **83 mm** | **19** |
| C | y | 24–40 mm | 1,345 | 135 mm | 30 |
| C | y | 40–80 mm | 10,041 | 172 mm | 46 |
| C | y | 80–400 mm | 33,334 | 172 mm | 57 |

(Single track in the chamber: A 41 mm and 15 strips, C 48 mm and 17 strips.)

**The trend is the opposite of what overlapping charge would give.** The closest
resolvable pairs (16–24 mm, ~100 tracks per chamber) are the least damaged, with
near-single-track strip counts; tracks 80–400 mm apart are the most damaged, with
fits 2–3× as wide as one track's. Two tracks that far apart share no charge on the
strip plane. Whatever widens those fits scales with the distance between the tracks
— which is what a candidate window spanning both, or a plane fit built from the
wrong combination of candidates, would do. The x view shows the same strip-count
trend (A 16 → 43, C 17 → 58) with a much milder resolution cost.

### 2.3 Close tracks are not two tracks

Of the 25,292 selected tracks in two-track A chambers, **1** has its partner within
12 mm of it in y on the strip plane (4 within 12 mm in x); for C, **2** of 44,813
(2 in x). Below 16 mm in y: 17 in A, 9 in C. These count tracks passing the §2.1
selection, per view, from `intra_twotrack_separation.csv`.
Two tracks from one vertex at a small opening angle land close together on the
strip plane — and the reconstruction turns those into **one** track. They are
missing from the pair sample, not merely mis-measured in it. (Consistent with
§4.1: hits closer than 12 mm join one seed cluster.)

### 2.4 What that does to the vertex tests

(`intra_summary.csv`, both legs cleaned in both views, clones removed — there
were none.)

| | A–A real | A–A event-mixed | C–C real | C–C event-mixed |
|---|---:|---:|---:|---:|
| pairs | 1,462 | 1,462 | 3,389 | 3,390 |
| Δx at capsule depth, robust σ | 27.7 mm | 27.0 mm | 30.3 mm | 28.9 mm |
| Δy at capsule depth, robust σ | 152 mm | 168 mm | 230 mm | 231 mm |
| x-view and y-view depths agree to 30 mm | 9.5 % | 7.4 % | 6.5 % | 6.4 % |
| Spearman ρ(depth from x, depth from y) | −0.02 | −0.01 | +0.01 | −0.04 |

For a common vertex Δy should be ~√2 × a single track's y resolution ≈ 57 mm in A;
event-mixed pairs come out at 168 mm because the legs themselves are at ~100 mm
(A) and ~160 mm (C). The Δx test is insensitive by construction (the capsule is
about as wide as the x pointing blur), and the depth-agreement test is poorly
conditioned for tracks this nearly parallel (~190 mm depth error at a 0.15 angle
difference). One weak hint: A–A with both legs within 10 mm of the capsule in x has
|Δy| < 40 mm in 31 % of 179 real pairs against 22 % of 173 mixed — ~2σ.

---

## 3 · How `wft` builds tracks today (read from the code, 2026-09-13)

### 3.1 Seeds — `wft/seed.py`

Hits of one plane are clustered with a spatial gap of `GAP_THRESHOLD_MM = 12.0`:
hits closer than that join one cluster, and each cluster seeds one candidate
window (`wft.io.extract_window` pads outward). **Two tracks within ~12 mm in a
plane produce one window.**

### 3.2 Per-plane candidates — `wft/reco.py:fit_plane_candidates`

Every candidate window is fitted **independently, as one track**
(`fit_plane`), and ranked by `(plausible, dchi2)` — plausibility is the
column-duration window and |tan| < `TAN_MAX`; `dchi2` is the χ² improvement over
no track. Nothing in a plane's fit knows another candidate exists.

### 3.3 Pairing x with y — `wft/reco.py:select_tracks`

Every (x candidate, y candidate) combination gets the key
`(coincident, plaus, dchi2_x + dchi2_y)`, where `coincident` is
`|t0x − t0y − dt_xy| ≤ DT_XY_TOL_NS = 120 ns`. Combinations are sorted on the key
and taken greedily, disjoint in x and in y; pair 0 is always kept, and every
further pair must be coincident and have both members plausible; at most
`MAX_TRACKS = 3`.

**Consequence for two tracks from one vertex.** Both tracks arrive at essentially
the same time, so both assignments `(x1,y1)(x2,y2)` and `(x1,y2)(x2,y1)` are
coincident and plausible, and the key reduces to `dchi2_x + dchi2_y`. The greedy
step then takes the **strongest x with the strongest y** and pairs the two weaker
ones — a *rank-by-fit-improvement* matching that does not look at geometry at all.
Whenever the two tracks' x-plane and y-plane rankings disagree (different path
lengths, charge sharing, one track near a dead or noisy region in one plane), the
assignment is wrong. `test_multitrack.py` covers the selector with idealised fits
whose two tracks are 500 ns apart, which is exactly the case where this does not
arise; there is no test with time-degenerate candidates.

---

## 4 · Hypotheses, and what separates them

| | hypothesis | predicts | separated by |
|---|---|---|---|
| **H1** | wrong x↔y assignment between time-coincident candidates (§3.3) | y (and 3D direction) wrong for both tracks; x per plane fine; independent of separation | truth overlay: assignment correctness vs. the two tracks' relative `dchi2` ranking |
| **H2a** | merging below the 12 mm seed gap (§3.1) | no two-track chambers below ~12 mm | observed (§2.3); truth overlay gives the efficiency curve |
| **H2b** | a candidate window, or the plane fit, spanning or mixing both tracks (§3.2) | wide fits and bad χ² in both views, **growing with separation** | the trend in §2.2 fits this; truth overlay with window bookkeeping (window extent vs. both donors' positions) |
| **H3** | two-track events are intrinsically messier (showers, δ-rays, noise bursts) — not a reconstruction failure | degradation persists when two *clean* single-track events are overlaid; shows in cluster shapes before any fit | truth overlay: if overlays of clean events reconstruct well, it is H3 |

§2.3 shows H2a is real. §2.2's trend — damage growing with separation — is the
signature of H2b or H1, not of charge overlap; it is also what H3 could give if widely
separated two-track events are a messier population (showers spread wide). H1, H2b and
H3 are untested, and only a truth sample separates them.

---

## 5 · The test bench: a waveform-overlay truth sample

Build two-track events whose answer is known, from data, in the same chamber and
conditions as the real ones:

1. **Pick donor events**: clean single-track events of one chamber (one gated
   track, slope measured and not noisy in both views, pointing at the capsule),
   same run and sub-run so pedestals, noise and calibration bundle match.
2. **Overlay** two donors' decoded waveforms (`<run>/<subrun>/decoded_root`),
   pedestal-subtracted, sample by sample. Two choices to make and record:
   - **noise:** summing doubles the noise; either accept it and compare against
     single donors with injected extra noise, or add only the second donor's
     signal region above threshold;
   - **timing:** shift the second donor so the two tracks are time-coincident
     (the common-vertex case) or at a random offset (the accidental case).
3. **Choose the pairs to scan the space that matters**: separation on the strip
   plane in x and in y (0–400 mm, dense below 30 mm), relative charge, and the
   two donors' `dchi2` ranks in each plane (to probe H1 directly).
4. **Re-run seeding and reconstruction** on the overlays with the production
   bundle (`calib_bundle_r06` family; see `CLAUDE.md` on c2 < c1).
5. **Truth** for each output track is its donor's single-track reconstruction.

Metrics, per separation bin and per chamber:

- two-track finding efficiency (0, 1 or 2 tracks found);
- x↔y assignment correctness;
- per-view p0 and tan residuals against the donors;
- y and x at the capsule's depth, robust σ, against single donors;
- strips and χ²/dof, to confirm the overlays reproduce §2.1 before any fix.

**The bench is only valid if the unmodified reconstruction reproduces §2 on it.**
If overlays of clean events reconstruct well, the data's degradation is H3 and the
reconstruction is not the lever.

A faster first pass that needs no waveform I/O: synthesise two-track planes with
the forward model (`wft/model.py`) under the production bundle. It tests the logic,
but only the overlay tests it against real noise.

---

## 6 · Reconstruction options to develop and A/B

- **R1 — assign x to y by more than time.** Both planes see the same drifting
  electrons, so a track's x and y clusters share more than t0: their
  charge-arrival profiles (`q_u50`, `q_u90`, `q_uend`), their summed charge
  (x/y charge sharing is correlated per avalanche), and their implied depth
  extent. Score both 2×2 assignments on those and choose the consistent one,
  instead of rank-matching on `dchi2` (§3.3). Cheapest; addresses H1 alone.
- **R2 — joint two-track fit in a plane.** When a plane has two candidates whose
  windows overlap, or one candidate whose charge profile is too wide for one track,
  fit a two-track forward model to the window jointly rather than two one-track
  fits. Addresses H2.
- **R3 — seeding.** Split a seed cluster whose hits show two arrival-time
  structures, or lower `GAP_THRESHOLD_MM` only where a cluster is wider than a
  single track can be. Addresses §2.3. Must not fragment single tracks — see §8.
- **R4 — keep the single-track contract.** `test_multitrack.py` pins pair 0 to
  `select_pair`'s choice so multi-track output cannot move the single-track answer.
  Every option above must keep that, and add tests for the time-degenerate case.

---

## 7 · Acceptance criteria

On the overlay bench:

- two-track finding efficiency above 90 % for separations ≥ 20 mm, with the curve
  below that reported rather than hidden;
- x↔y assignment correct in ≥ 95 % of found pairs;
- y at the capsule's depth for two-track events within **1.2×** the single-track
  value (A ~41 mm, C ~48 mm); x within 1.1×.

On data:

- `python -m pair_vertex_imaging.intra_vertex --multiplicity` shows the two-track
  rows of §2.1 approaching the one-track rows, and two-track chambers appearing
  below 12 mm separation;
- **single-track events unchanged**: gated-track counts and single-track resolution
  on a fixed run_145 sub-run identical to the frozen pass (the D hot-channel
  wildcard failed exactly this check on its first tuning — `STATUS.md`,
  2026-09-08);
- then the three tests of §2.4 (`python -m pair_vertex_imaging.intra_vertex`) are
  re-run unchanged, and are sensitive: event-mixed Δy approaches √2 × single-track.

---

## 8 · Pitfalls already paid for

- **Geometry from waveforms, never hit times** (`CLAUDE.md`). Hits may pick
  candidates; they may not set a position, angle or depth.
- **A seed change can quietly lose single tracks.** The D noisy-channel wildcard
  lost 16,152 good fits out of 33,393 events on its first tuning. A/B every seeding
  or window change on single-track gated counts first.
- **Chamber D** has noisy columns (a third of its tracks) and a dead band; develop on
  A and C, where the problem is the reconstruction and not the hardware.
- **Calibration bundles** must have c2 < c1 (`wft.calib.check_kernel_ordering`).
- **`x_slope_reliable` / `y_slope_reliable`** are `|tan| ≥ 0.08` and gate nothing
  in the current chain; use them explicitly in any evaluation, as the analysis did.

---

## 9 · Where things are

| what | where |
|---|---|
| the analysis | `ntof_athens_26/pair_vertex_imaging/intra_vertex.py` |
| the numbers above | `<out>/pair_vertex/intra_multiplicity.csv`, `intra_twotrack_separation.csv`, `intra_summary.csv`, `pairs_intra.parquet` |
| the figures | `ntof_athens_26/pair_vertex_imaging/figures/intra_multiplicity.png`, `intra_test1–3.png` |
| reproduce | `X17_ROOT=D:/x17 python -m pair_vertex_imaging.intra_vertex --jobs 8` then `--multiplicity` (from `ntof_athens_26/`) |
| the reconstruction | `wft/seed.py`, `wft/reco.py`, `wft/model.py`, `wft/tests/test_multitrack.py` |
| stage-3 tracks | `<out>/stage3_fullpass/tracks_<run>_<subrun>.parquet` |
