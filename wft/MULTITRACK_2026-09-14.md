# Two tracks in one chamber: x/y pairing and the rescue floor — 2026-09-14

**Two opt-in changes recover same-chamber two-track events without moving a single
production track.** On the run_145 overlay truth bench, both tracks of a pair ≥ 24 mm
apart come back correctly paired in **71 % (A) / 66 % (C)**, against 47 % / 39 % with
production reconstruction. Re-fitting every production trigger of run_145 stat090_0000
whose seeds the change touches loses **no** production track. With both options off
(the default) the output is the production output, candidate for candidate.

Nothing here is switched on in the production chain yet. How to do that is §5.

Companions: the problem statement and plan, `sept26_prelim_analysis/HANDOFF_INTRA_TWO_TRACK_RECO.md`
(§10 is the progress record); the bench, `sept26_prelim_analysis/intra_bench.py`; the earlier
multi-track work this builds on, `MULTITRACK_2026-08-12.md`.

---

## 1 · What the bench showed was wrong

The bench sums the waveforms of two clean single-track triggers of one chamber, file
tag and trigger phase (the second donor only on its own signal strips), merges their
hits, and runs the unmodified seeder and fit. Truth for each track is its donor's frozen
single-track fit; the harness reproduces the frozen pass bit-for-bit.

| failure | where | size in production reco |
|---|---|---|
| **x/y swaps** | `select_tracks` pairs time-degenerate candidates by summed dchi2, so the strongest x goes with the strongest y | 15–26 % of tracks ≥ 24 mm apart within 150 ns; ~75 % when the two planes rank the tracks differently |
| **plane-wide significance floor** | `apply_significance_floor` keeps strips ≥ 10 % of the brightest strip *of the plane*, so a brighter partner anywhere pushes a fainter track under the seeder's strip minimum | ~16 % of tracks ≥ 24 mm apart never seeded, independent of separation |
| **merging at the 12 mm seed gap** | `seed_candidates` | 98–100 % of tracks < 12 mm apart, 62–65 % at 12–24 mm |

Once a separated track *is* found, its plane fit is the single-track fit (robust σ of p0
against its donor ≲ 0.1 mm, same strip count). The handoff's H2b — a window or fit
spanning both tracks — is not what degrades real two-track events.

## 2 · What changed

### 2.1 x/y pairing by what the planes share — `wft.reco.select_tracks(pairing=)`

After the unchanged greedy selection, for every two **gated** tracks whose swapped
combinations also pass the gate (time-coincident within `DT_XY_TOL_NS` and both
plausible), the y partners are swapped if that lowers

    cost = Σ_features min(((f_xy − median) / rsig)², 25)

over the chosen x-minus-y features of each track: `lq` = log(q_sum_x / q_sum_y),
`u50`/`u90` = difference of the charge-arrival quantiles, `t0` = t0_x − t0_y − dt_xy.

- The calibration is a bundle field, `CalibrationBundle.xy_pairing` =
  `{features, median, rsig, provenance}`, validated by `wft.calib.check_xy_pairing` at
  load and save. `wft.reco._worker_init(bundle, pairing_path)` overrides it for A/B runs.
- Features chosen per chamber on the bench: **A `['lq']`, C `['lq', 'u50', 'u90']`**.
  Median and width from clean single tracks of stat090_0001–0002 (A 8 300, C 6 024
  tracks), so the bench sub-run is not calibrated on itself.
- **Contract:** only y members move. Every x member, the track order, every gate decision
  and every event with a single gated track are unchanged; time-separated tracks can never
  be re-paired (their swapped combinations fail the gate).

### 2.2 Rescue floor — `wft.seed`, `ntof_tracking.wft_beam.seeds_from_hits_beam`

`WFT_SIG_FLOOR_LOCAL_MM` (or `local_mm=`) > 0 turns on a second, local floor: each strip
is compared with the brightest strip within ± that many mm instead of the whole plane.
In **`rescue` mode (the default mode)**:

1. seeds are formed from the plane-wide floor exactly as in production;
2. clusters found under the local floor whose position range overlaps **no** plane-wide
   cluster (vetoed ones included) are appended as extra candidates, up to
   `n_candidates`, flagged `Seed.rescued`;
3. `select_tracks` ranks any combination using a rescued candidate below every
   production combination (`fit._rescued`, carried into the candidates side table as
   `rescued`). Pair 0 is therefore always production's choice, and a rescued candidate
   can only add a further track.

`replace` mode (the local floor instead of the plane-wide one) is kept for study and
**must not** be used in production — see §3.2.

### 2.3 Split seeding — `WFT_SPLIT_GAP_MM` / `split_gap_mm=` — study only

Re-clusters each seed at a smaller gap and offers the parts instead. Rejected (§3.2).

### 2.4 Provenance

`reconstruct_run` and `wft_beam._write_meta` record `sig_floor_local_mm`,
`sig_floor_local_mode`, `split_gap_mm` (selection) and the `xy_pairing` features
(multi-track / reco config) in every `.meta.json`.

## 3 · Validation

### 3.1 On the overlay bench (run_145 stat090_0000, 2 327 overlays)

Same donor pairs, same strips; only the reconstruction differs. Both tracks found and
correctly paired:

| variant | A ≥ 24 mm | C ≥ 24 mm | A 12–24 mm | C 12–24 mm |
|---|---:|---:|---:|---:|
| production | 47 % | 39 % | 18 % | 17 % |
| pairing | 49 % | 44 % | 21 % | 20 % |
| local floor 16 mm, replace | 70 % | 63 % | 26 % | 25 % |
| local floor 16 mm, rescue | 69 % | 59 % | 26 % | 25 % |
| **pairing + rescue 16 mm** | **71 %** | **66 %** | **29 %** | **28 %** |
| pairing + rescue 16 mm + split 6 mm | 71 % | 66 % | 48 % / 72 % ‡ | 44 % / 70 % ‡ |

‡ 12–18 mm / 18–24 mm. Swapped assignments among time-coincident pairs with both tracks
present: A 24 → 14 %, C 23 → 7 % with pairing. Tracks lost at seeding ≥ 24 mm: 16–17 % →
< 1 % with either floor. Below 12 mm nothing recovers: those tracks share one cluster
and need a joint two-track fit.

### 3.2 Single tracks: every changed production trigger re-fitted

`intra_bench floor-ab` seeds each tag of run_145 stat090_0000 with and without the change,
re-fits every production-fitted trigger whose seed lists differ (the others fit identically
by construction), and matches its gated tracks to the frozen pass (both planes within
3 mm). A: 22 434 fitted triggers, 7 497 gated tracks; C: 22 788 and 7 089.

| change | seeds changed (A / C) | production tracks not recovered | clean single tracks lost | events losing a track | verdict |
|---|---:|---:|---:|---:|---|
| replace floor 16 mm | 2 216 / 3 791 | 142 (1.9 %) / 519 (7.3 %) | 4 / 11 | 64 / 238 | rejected |
| replace floor 40 mm | 1 757 / 3 079 | 106 (1.4 %) / 359 (5.1 %) | 0 / 1 | 42 / 162 | rejected |
| rescue 16 mm, before the ranking rule | 1 319 / 1 726 | 14 (0.19 %) / 17 (0.24 %) | 0 / 0 | 0 / 0 | fixed → |
| **rescue 16 mm** | 1 319 / 1 726 | **0 / 0** | 0 / 0 | 0 / 0 | **keep** |
| split 6 mm | 2 403 / 5 941 | 1 103 (14.7 %) / 1 902 (26.8 %) | 2 / 5 | 58 / 255 | rejected |
| split 8 mm | 1 814 / 3 771 | 820 (10.9 %) / 1 121 (15.8 %) | 1 / 4 | 50 / 180 | rejected |

With rescue, 99.6 % of the recovered tracks are bit-identical and none moves by more than
0.1 mm; 279 (A) and 406 (C) triggers gain a track. Pairing cannot touch a single-track
event by construction, and all 112 single-donor re-fits on the bench are unchanged with it.

**Why replace and split fail:** both change the clusters of real single tracks. A single
track in a busy plane had its faint tails removed by the plane-wide floor; the local floor
keeps them, the window widens and the fit moves (one C track: 19 → 34 strips, tan moved
0.63). Real clusters also have holes, so splitting fragments real tracks far more often
than the clean bench donors (< 1 %) suggested.

### 3.3 Defaults are production

With every option off, 40 run_145 events (tag 000, 20 per chamber, 1–3 tracks) and 44
more (tag 001) reproduce the production candidates, track ids, gates and `n_tracks`. One
C event (8531) differs in p0 by 4 µm on two candidates with identical tracks — and
identically on the pre-change HEAD, so it is the laptop against the condor nodes, not
this change. Unit tests: `wft/tests/test_multitrack.py` (time-degenerate pairing, charge
pairing, rescued ranking) and `test_seed_and_select.py` (local floor, rescue, split).

## 4 · What this does not establish

- **One sub-run, two chambers.** run_145 stat090_0000, A and C. B and D are untested, and
  D's hot-channel seeding (`hot=`) has not been combined with the rescue floor.
- **The bench donors are clean.** Efficiencies are for two clean tracks overlaid; real
  two-track events are busier, so the gain on data will be smaller.
- **The single-track A/B matches tracks to the frozen pass, not to truth.** It proves the
  change does not move production answers; it does not say the added tracks are real.
  Tracks added on data need their own look (vertex tests, `intra_vertex`).
- **Pairing rules were chosen on the bench sub-run**; their calibration is from the other
  two local sub-runs of the same run. Other runs and the noisy-side condition need their
  own `xy_pairing` (calibration is per detector and per run condition).
- **< 12 mm is untouched.** A joint two-track fit of one window is the only route there.

## 5 · Turning it on

1. Put each arm's pairing calibration into its bundle's `xy_pairing` (the files are
   `<out>/intra_bench/xy_pairing_<arm>.json` for run_145; `intra_bench calib-pairing`
   rebuilds them). `wft_beam.make_bundle` does not do this yet — it needs the same kind of
   merge step as `run_beam_job.py --hot`.
2. Set `WFT_SIG_FLOOR_LOCAL_MM=16` for the reconstruction (the mode defaults to `rescue`;
   `WFT_SIG_FLOOR_LOCAL_MODE` only needs setting to study `replace`). Never set
   `WFT_SPLIT_GAP_MM`.
3. Re-run a fixed sub-run and repeat `intra_bench floor-ab --local-mm 16` there before the
   full pass, then re-run `intra_vertex --multiplicity`.
