# Detector-A double-track search (runs 58 / 61 / 62 / 63)

**Goal.** Find Det-A events with **more than one track** — both *separated
pile-up* (two independent particles) and *vertex-opening / crossing pairs*
(the e⁺e⁻-from-target topology). Started 2026-07-21.

## Why the frozen reco isn't enough

`ntof_tracking.reco.segments.find_segments` fits **one** robust line per
connected-component cluster. That already separates two spatially disjoint
tracks (different components → different segments), but **two tracks that cross
or share strips near a vertex merge into one cluster and are fit as a single
line** (or dropped as a blob). We also found the frozen `classify()` is too
permissive for this job: horizontal afterpulse rows and vertical isochronous
columns get counted as "track" segments, so the cached `n_trkseg` counts
**cannot** be used to define doubles (they over-count badly — e.g. run_58
ev349/ev1067 read as 2×2 in the cache but carry no real diagonal track).

## What this package adds

- `dtrack_lib.py` — the double-track finder, layered on the frozen reco
  primitives (read-only): per plane, cluster (connected components) → **split
  each cluster into multiple lines by sequential weighted RANSAC** → keep only
  track-grade lines → drop **cospatial over-splits** with a distinctness test
  (two lines are different tracks only if they separate by ≥ 14 mm somewhere
  over their *union* time span — offset OR fanning; a single deposit fit by two
  near-parallel lines stays close everywhere and is rejected). Then pair the
  X/Y lines into 3-D micro-TPC segments (bench time-IoU + charge balance).
  **A Det-A double = ≥ 2 distinct track-grade lines in BOTH planes** (the
  chosen definition). RANSAC is reseeded per event from the eventId, so results
  are reproducible and order-independent.
- `scan.py` — runs the finder over whole runs. **Low-memory Det-A-only loader**
  (streams the combined-hits tree in batches, keeps only FEUs 3/4 → peak
  ~0.6 GB vs 1.4 GB for the full 4-detector load — this machine has ~15 GB and
  little free). Skips the γ-flash leader / saturated flash-pile-up, applies a
  cheap pre-filter (≥ 10 clean hits per plane, not a full-plane discharge), and
  **tags** (does not cut) moderately-busy events so genuine busy multi-prong
  events like ev977 survive. Caches per sub-run: `<subrun>_ev.parquet`
  (per-event features) and `<subrun>_cand.pkl` (full line + hit-dump detail for
  every double).
- `analyze.py` — aggregates the caches, computes **global-frame pointing** per
  track (DCA to the beam axis, `beam_y`) and the **inter-track 3-D closest
  approach** (the vertex test) from the cached pairs, ranks candidates, writes
  `census.txt` + `candidates.csv`, and renders `gallery/` event displays.
- `probe.py` — single-event display for interactive checking:
  `probe.py <run> <subrun> <eventId>`.

## Ranking / topology

`score` favours: ≥ 2 confirmed 3-D X/Y pairs, straight lines (high r²), low
occupancy (not busy), post-flash recovered window (dt > 8 ms), both tracks
pointing back toward the beamline, and **clear spatial separation** (a 90 mm
gap is unambiguous; a ~15 mm gap near the distinctness floor is marginal /
possible over-split). Topology tag from the 3-D geometry: `vertex` (tracks meet
within ~30 mm), `separated`, `+beam` when both point back near the beam axis.

⚠ **Global geometry is provisional**: depth uses the uncalibrated DAQ t0
(~450 ns) + bench 95/5 v_drift, so absolute pointing/vertex distances carry
systematic error. They **rank and flag**; they do not yet measure.

## Reproduce

```bash
# scan (jobs to taste; ~0.6 GB/worker, watch RAM on this box)
.venv/bin/python ntof_july_analysis/detA_doubletrack/scan.py process \
    run_63 run_62 run_58 run_61 --jobs 4
# aggregate + gallery
.venv/bin/python ntof_july_analysis/detA_doubletrack/analyze.py \
    run_58 run_61 run_62 run_63 --top 40
```
Outputs → `<ANALYSIS>/July_HV_Scan/detA_doubletrack/` (`census.txt`,
`candidates.csv`, `gallery/`, `probe/`).

## Calibration examples (run_58 sngPS_dr400_r545_034)

- **ev1242** — clean **separated** double: two tracks ~95 mm apart in both
  planes, time-consistent → 2 pairs. The unambiguous kind.
- **ev977** — busy **multi-prong fan**: 4 diagonals fanning from a common
  region in y (dt = 27.9 ms, recovered window). Real structure, needs eyeball.
- **ev1718** — **rejected**: one compact deposit that RANSAC first over-split
  into two cospatial lines; the union-span distinctness test correctly collapses
  it to a single track.

## RESULTS (2026-07-21, full scan of runs 58/61/62/63)

158 sub-runs, **192 254 reco'd events** (post pre-filter), 1 h 53 m at jobs=4.

| run | trigger | reco ev | doubles | rate |
|---|---|---|---|---|
| run_58 | singles+PS | 39 992 | 117 | 0.29 % |
| run_61 | singles+PS | 106 136 | 217 | 0.20 % |
| run_62 | singles only | 20 222 | 68 | 0.34 % |
| run_63 | **doubles**+PS | 25 904 | 64 | 0.25 % |

**367 double-track candidates** after the no-extrapolation distinctness
re-filter (which removed 99 fragmented-single-track false positives):
33 non-busy, 215 with ≥ 2 confirmed 3-D X/Y pairs, 195 well-separated
(≥ 40 mm), 120 tagged `vertex+beam`, 95 `separated+beam`.

**GOLDEN sample — 5 events** (not busy, ≥ 2 pairs, separation ≥ 40 mm):

| event | sep | 3-D trk DCA | topo | dt |
|---|---|---|---|---|
| run_58 `sngPS_dr300_r580_036` ev1054 | 193 mm | 62 mm | separated+beam | 27 ms |
| run_63 `dblPS_dr600_r560_000` ev578 | 63 mm | 70 mm | separated+beam | 57 ms |
| run_58 `sngPS_dr450_r545_061` ev397 | 54 mm | 61 mm | separated+beam | 29 ms |
| run_63 `dblPS_dr600_r555_001` ev4092 | 46 mm | 15 mm | vertex+beam | 31 ms |
| run_58 `sngPS_dr700_r560_004` ev196 | 45 mm | 39 mm | separated+beam | 28 ms |

Visually confirmed: **ev1054** is the cleanest — two well-separated tracks in
both planes (y-plane n=29, r² = 0.99) in a nearly noise-free recovered-window
event. **ev397** is the best *coincident* pair: two tracks ~45–50 mm apart in
both planes with **overlapping time ranges** (simultaneous particles, not
stacked in depth). **ev4092** is the most vertex-like (3-D DCA 15 mm).

### Honest caveats
- **Doubles are rare (~0.2–0.3 % of reco'd events) and mostly busy** — real
  multi-track events are inherently high-occupancy, so only 33/367 are clean.
  Treat the busy 334 as needing eyeball, not as a measured sample.
- **No true vertex (e⁺e⁻) event is established.** The `vertex` tag only means
  the two 3-D lines pass close; with the uncalibrated drift t0/v_drift the
  absolute geometry is provisional. Nothing here is a confirmed pair-production
  vertex — the well-separated candidates look like two independent tracks
  (in-window pile-up), which is the expected dominant source.
- **The rate is not an efficiency.** The pre-filter (≥ 10 clean hits/plane)
  and the flash/leader veto both bias the denominator.
- Known false-positive class, now suppressed but worth re-checking on new data:
  a single track running along the drift direction, chopped by a coherent noise
  band, whose fragments fake a pair (run_61 ev2522 was the archetype).
