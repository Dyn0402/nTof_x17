# Multi-track generalisation of wft — 2026-08-12

**The reconstruction now returns every candidate track, not just the winner.
Single-track output is bit-identical (verified, 250 det3 bench events); the
multi-track information is additive: an `n_tracks` column on the events table
and a `*.candidates.parquet` sidecar with the full ranked candidate list.**

## Why this was mostly plumbing, and where the real work was/is

The single-track chain already fit every candidate cluster on every event:
`wft.seed` offers up to 3 (bench) / 5 (beam) spatial clusters per plane, and
`fit_plane_candidates` fits all of them to pick the winner — the ranked list
existed at runtime and was discarded by the output layer. Three tiers:

1. **Spatially separated tracks** (clusters > 12 mm apart per plane) — DONE
   here. Zero extra fit cost; output plumbing only.
2. **X↔Y pairing of ≥ 2 simultaneous tracks** — DONE here at the wft level
   (`select_tracks`, time coincidence); the charge-balance tie-break
   (TRACK_PLAN_03, `microtpc_lib.pair_planes`) stays downstream where the
   3D-segment table is built, and IS needed — see the ghost-swap failure below.
3. **Two tracks merged into ONE cluster** (< 12 mm apart) — NOT done, real
   model work: a two-column design matrix (2K NNLS basis, 6 outer parameters)
   plus a 1-vs-2-track model-selection penalty. Such events today fit as one
   compromise track with large chi2/dof (`quality_ok` False at chi2/dof > 300)
   — that flag is the "multi-track suspect" marker until this tier exists.

## What changed

- `wft.reco.select_tracks(cand_fits, ftst_diff, cal, max_tracks=3)` — ranked
  disjoint time-coincident (x, y) candidate pairs. Pair 0 is `select_pair`'s
  choice by construction (same key, same maximum — unit-tested), kept even
  when ungated, so single-track behaviour cannot move. Every further pair
  must be time-coincident AND both-plausible: the double-counting guard
  against split clusters and coincident noise.
- `wft.reco.candidate_rows(...)` — flattens the ranked per-plane candidate
  fits (all PlaneFit fields + `plausible`, `dchi2`, `rank`, `track_id`,
  `track_gated`, `isochronous`, `ftst`) into the sidecar table.
- `_worker_fit` emits both; `reconstruct_run` (and the beam driver) write
  `<out>.candidates.parquet` next to the events table.
  `WFT_EMIT_CANDIDATES=0` turns the sidecar off.
- **`ntof_tracking/wft_beam.py` bug fix**: it still passed a scalar
  `ftst_diff` where `_worker_fit`'s payload has carried the per-plane `ftst`
  dict since the t0-prior work (2026-08-11). Under the old signature every
  beam event with `ftst_x != ftst_y` raised inside the worker's try and
  reconstructed as None. Any run_79 table produced with wft at or after the
  t0-prior merge and the old wft_beam should be regenerated.

## Reading `n_tracks` — it is conservative by design

`n_tracks` counts GATED pairs. `n_tracks == 0` does **not** mean "no track":
the winner is still in the `x_*`/`y_*` columns and `x_ok/y_ok` remain the
single-track authority. It means the winning pair did not pass the
coincidence+plausibility gate. Measured on det3 bench (400 cached events,
free-t0 bundle): 27–32 % of good single-muon events sit at `n_tracks == 0`,
and the failures cluster at Δt0 offsets of ±2–4 × 60 ns — the known t0
basin degeneracy of free fits (doc §22 / T1.1), not mis-pairing. With the
t0-prior bundle the coincidence leg tightens (see gate results). For n_TOF,
where no per-event t0 prior exists yet, treat `n_tracks >= 2` as the
double-track flag and do the definitive pairing downstream with
charge-balance + time-IoU (TRACK_PLAN_03).

## Validation (det3, saturday scan long_run 490 V/1000 V)

- **Non-regression**: `wft.cli reco sat_det3` (calib_bundle_lp2, 250 events,
  HEAD vs this change): all 51 shared columns bit-identical; only additions.
- **Unit tests**: `wft/tests/test_multitrack.py` (19 checks) + the existing
  suite (seed/select, model regression vs R&D code, share modes, dead mask)
  all pass.
- **Synthetic doubles** (`mx_june_wft/16_multitrack_gate.py`): merge the
  candidate windows of two clean single-track cache events ≥ 40 mm apart in
  both planes into one payload, require both tracks back.

  | bundle | both tracks found | both (x,y) within 3 mm | x↔y swaps | ghost rate (399 singles) | n_tracks==0 (singles) |
  |---|---|---|---|---|---|
  | free t0 (cache default)  | 34/40 (85 %) | 33/40 | 1 | 1.5 % | 27 % |
  | t0 prior (lp2_t0p, σ=5)  | 37/40 (92 %) | 33/40 | 4 | 1.0 % | 20 % |

  The prior finds more doubles but SWAPS more: it pins every candidate of a
  plane to the same per-ftst t0 prediction, erasing the time separation the
  coincidence pairing keys on, so simultaneous tracks become
  assignment-degenerate and greedy dchi2 decides. Measured, not just
  predicted. Which x goes with which y for near-simultaneous tracks is
  therefore the downstream charge-balance pull's job (σ68 ≈ 0.07, PLAN_38 /
  TRACK_PLAN_03) — the wft-level gate answers "how many tracks", not
  "which pairing", when Δt0 between the tracks is ≲ the 60 ns basin scale.
- **Ghost rate** on real single-muon events: 4–6/399 = **1.0–1.5 %** report
  `n_tracks >= 2`. Inspected examples are large multi-cluster,
  high-charge topologies (delta rays / shower pairs) — at least partly REAL
  second particles, not selection artifacts; most fail `quality_ok`, which
  remains available as a downstream cut.

## For n_TOF (run_79 and the July/August MM data)

- The beam seeder (`wft_beam.seeds_from_hits_beam`) already offers 5
  candidates/plane with per-plane busy vetoes — nothing to change there.
- Double-track events appear in the candidates sidecar with `track_id`
  0 and 1; build 3D segments downstream per TRACK_PLAN_03 (time IoU +
  charge balance), which also resolves the t0-degenerate ghost swaps the
  wft-level gate cannot.
- Merged-cluster doubles (< 12 mm) are NOT resolved — they surface as one
  bad-chi2 track (`quality_ok` False). Tier 3 above is the follow-up if the
  physics needs them.
