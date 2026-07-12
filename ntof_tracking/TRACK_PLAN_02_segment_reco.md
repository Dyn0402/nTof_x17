# TRACK_PLAN_02 — per-plane segment reconstruction in beam windows

**Core stage: turn each track-candidate window into per-plane track segments
with bench-grade positions and angles. Deliverable: `beam_segments.py`
producing a per-plane segment table, validated first on the cosmics-at-nTOF
runs (Jun 27–29) where the bench models should reproduce ~2° out of the box.**

## The two-layer design

A beam window (6–8 µs) can contain several tracks at different times, unlike
the bench's single-track 1.9 µs window. So reconstruction is two-layered:

1. **Pattern recognition** — split the window's hits into per-plane track
   candidates. REUSE the existing, working machinery in
   `beam_track_finding.py` (imported, not rewritten): isolated-hit removal
   (no neighbour within 5 mm AND 200 ns), road-following late→early time
   (seed amp>600, road 3 mm / seed road 7 mm, MAX_TIME_GAP 250 ns, dead-strip
   aware, MIN_TRACK_HITS 2), strip-cluster endpoint extension, plus the
   cross-dimensional guided merge from `ntof_may_analysis/track_explorer.py`
   (`_cross_dim_merge`, `find_tracks_1d_pass2`) to de-fragment.
2. **Measurement** — for each candidate's hits, apply the bench chain from
   `ntof_tracking/microtpc_lib.py`:
   - `anchored_time_fit(pos, time, amp)` → position anchor
     (`mesh_position_mm` = earliest-strip, the mesh crossing point — robust to
     the edge, bench σ 0.73–0.94 mm at low angle), `S_prod`, `duration_ns`,
     `extent_mm`, `n_strips`, `red_chi2`.
   - `hit_features(pos, amp, time, tot)` → the 6-feature signature
     (needs `time_over_threshold`; lead strip ≥ A_LEAD_MIN=300 — revisit this
     value against the beam gain; record coverage lost to it).
   - `apply_tan_regression(model_plane, F, restandardize=True)` +
     `apply_sign(wg, …, fallback_sign=S_prod)` → signed tanθ per plane.
     Model = `models/mx17_<n>_hits6.json` for THIS detector (det3's for C/D
     only until PLAN_05 retrains them — their own bench models exist too;
     start with each detector's own model, they were all trained).
   - Hybrid switch (`hybrid_tan`) only becomes meaningful once PLAN_06
     supplies unshared time-fit segments; until then the REGRESSION is the
     angle (bench: regression-only plateau 2.1°, low-angle 1.8°).

## Restandardization protocol (important subtlety)

`restandardize=True` recomputes feature mu/sd on the target population. On the
bench this fully recovered retraining (README table) — but the bench target
populations were all cosmics (same angular mix). Beam tracks from a target
have a DIFFERENT angular distribution, which shifts feature means for
physical (not instrumental) reasons and can re-bias the calibration.
Protocol:
1. Restandardize on a large, taxonomy-clean candidate population per
   (run, detector, HV setting) — never per event.
2. Commission on the **cosmics-at-nTOF runs first** (Jun 27–29, and any
   beam-off cosmic runs): cosmic angular mix ≈ bench, so the frozen_rs
   numbers must reproduce ~1.7–1.9° (det3/A) — if they don't, the problem is
   the data interface, not the physics.
3. In-spill, treat frozen_rs angles as *bootstrap quality* until PLAN_04/05
   provides inter-chamber truth to re-anchor mu/sd and retrain. Record which
   model+standardization produced every segment (provenance columns).

## Quality flags per segment

- `n_strips`, `red_chi2`, `q_cluster`, `saturated_any`
- edge fiducial: anchor > 25 mm (angles unreliable below; bench script 32) —
  flag, don't cut; positions stay valid to the edge.
- flash/spark proximity: window class from PLAN_01.
- window-truncation: candidate's `duration_ns` within ~2 T_sat of the window
  end → angle biased (bench 500 V lesson) — flag.

## Output schema (per-plane segments, parquet or csv)

`run, subrun, eventId, det, plane, t0_ns, pos_anchor_mm, tan_reg, sign_src,
tan_prod, S_prod, n_strips, extent_mm, duration_ns, q_cluster, q_frac,
a_lead, tot_lead, red_chi2, flags, model_id, restd_id`

## Acceptance

- Cosmics-at-nTOF: angle correlation between chambers (after PLAN_04 rough
  alignment) consistent with two ~2° devices; rate of segments/window sane.
- In-spill: hits-per-candidate, segments/window, and angle distributions
  stable across subruns at fixed HV; no pileup of segments at the window
  edges (would indicate truncation mishandling).

## Gotchas

- Road-following parameters embed a drift-velocity assumption
  (`DRIFT_VELOCITY_MM_US=22` in beam_track_finding) — only pattern-level;
  keep loose, and re-tune MAX_TIME_GAP once PLAN_05 pins v for the July gas
  (at v~20–35 µm/ns and 20 ns sampling, one sample ≈ 0.4–0.7 mm of drift).
- Two tracks crossing in position but separated in time must NOT be merged —
  IoU pairing in PLAN_03 needs honest per-candidate time spans.
- The bench 12 mm gap-cluster inside `anchored_time_fit` assumes ONE cluster
  per candidate; road-following should already deliver that. If a candidate
  yields n_dropped > 0 repeatedly, pattern recognition is under-splitting.
