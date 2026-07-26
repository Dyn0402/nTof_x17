# TRACK_PLAN_04 — inter-chamber alignment, track linking, vertexing

**This is the stage that REPLACES the M3 telescope: particles crossing ≥2
chambers give straight-line references built from the chambers' own
sub-mm position anchors. Everything calibration-grade (PLAN_05) hangs off
this. Deliverable: aligned chamber geometry + linked-track table + vertex
observables.**

## 0. Get the as-installed geometry (blocking, human-in-the-loop)

The May run_config carried NO chamber positions (`det_center_coords` all
zero) — assume July is the same until proven otherwise. Needed per chamber:
z along the stacking axis, transverse offsets, drift-axis direction (which
side the mesh faces), in-plane rotation (~90° convention as bench). Sources:
July logbook / install photos / survey, `MX17_Simulation/detector_config.py`
for the DESIGN (two arms around the target, MM at 25 cm), and the DAQ-side
`run_config.json` of the physics runs. **If absolute numbers are unavailable,
proceed with a nominal layout ±cm — relative alignment below absorbs it; only
vertexing to the target needs the absolute scale (~mm from survey or from
beam-spot reconstruction itself).** Record whatever is adopted in
`ntof_tracking/geometry_july.json` with provenance comments.

## 1. Relative alignment from through-going tracks

Sample: **in-spill through-goers ONLY — there are no cosmics-at-nTOF and no
beam-off cosmic runs (2026-07-12).** Through-goers = segments in ≥2 chambers,
time-coincident within the same DREAM window, |Δt0| < 150 ns ≈ 5σ of the bench
33 ns timing. This is the sole alignment sample; it is rarer and more
forward-peaked than a cosmic sample would be, so expect to pool many subruns to
accumulate enough inclined links.

Procedure (mirrors the bench alignment loop, `03_alignment_and_tpc.py`
pattern — iterate translation → rotation → z):
1. Match segments pairwise between chambers (position road ~10 mm after
   nominal geometry, tighten iteratively; use PLAN_03 t0 coincidence to kill
   combinatorics).
2. Fit per-pair residuals → per-chamber transverse offsets (median residual),
   in-plane rotation (residual-vs-position slope), then z (residual-vs-angle
   slope — needs inclined tracks; with NO cosmics these must come from the
   inclined beam tracks themselves — the run_30 first-look shows ~8–15° tracks
   exist but are rare, so z will converge slowly and only after pooling many
   subruns; if inclined statistics are too thin, hold z at the nominal/survey
   value and align only translation+rotation).
3. Iterate to convergence (<0.1 mm shifts). Bench experience: converges in
   2–3 iterations; θ lands near 90°±1.
4. Ship `geometry_july.json` v2 with the fitted constants + residual maps.

## 2. Track linking

For each window: collect 3D segments (PLAN_03) across chambers, cluster by
t0 coincidence, fit straight lines through segment POSITIONS (weights from
per-plane σ ~0.8 mm; angles are NOT used in the fit — they validate it).
Accept links by χ² and by segment-angle-vs-line-angle consistency (each
within ~3× its per-chamber angular σ from bench_constants/BENCH_ANGLES,
det-specific). Output per track: line parameters, χ², contributing segments,
per-segment residuals (these residuals ARE the in-situ resolution
measurement), and the per-chamber angle pulls (the training residuals for
PLAN_05).

**Geometry note:** with two arms of two chambers (design), most tracks cross
2 chambers within one arm — the lever arm is the inter-chamber spacing; a
2-point line has no χ² but still gives the reference ANGLE to ~
σ_pos·√2/Δz — for σ=0.8 mm, Δz=50 mm that is ~1.3°, comparable to the
micro-TPC angle itself; at Δz=100 mm it is ~0.65°, i.e. a genuine reference.
Compute and store the per-link reference-angle covariance so PLAN_05 can
deconvolve it exactly like PAPER PLAN_37 does for M3.

## 3. Vertexing & pair topology

- Extrapolate linked tracks (or single-chamber 3D segments when only one
  chamber fired) to the target plane; beam spot = the 2D distribution — its
  centroid/width is also the absolute-alignment cross-check.
- Pair finder: two tracks in one window with compatible t0 and vertex
  distance-of-closest-approach < cut → opening angle ψ, invariant-mass proxy
  under e⁺e⁻ hypothesis (needs energies from calorimetry/scint — out of
  scope here; deliver the geometry+timing part: vertex, ψ, per-track E_n
  window from PLAN_07).
- Deliverables to physics: `tracks.parquet` (linked tracks) and
  `pairs.parquet` (vertexed pairs with ψ, DCA, t0, E_n band).

## Acceptance

- Alignment residuals: flat maps, widths consistent with quadrature of two
  chambers' position σ (~1.1–1.3 mm per axis for 0.8 mm chambers).
- Linked beam tracks (no cosmic sample exists): micro-TPC segment angles vs
  link angles — per-chamber σ can be MEASURED here for the first time (there is
  no independent cosmic cross-check), and compared to the bench numbers (det3
  ~1.8°, det2 ~2.4°, det6/7 ~3–4° with frozen models) as a plausibility guide,
  not a validation. This link residual IS the in-situ resolution.
- Beam spot at the target position with a width compatible with the beam
  profile + extrapolation error.

## Gotchas

- Time coincidence across chambers relies on all FEUs sharing the trigger
  time base (same DREAM event) — verified structure at bench; confirm once at
  beam by histogramming Δt0 between chambers for obvious through-goers
  (should be σ ~ 45 ns, not µs).
- In-spill through-goers are forward-peaked and there is NO cosmic sample to
  fall back on: z-alignment (needs angles) must come from the inclined beam
  tracks alone. If it will not converge, freeze z at survey/nominal and align
  only translation+rotation rather than fitting a poorly-constrained z.
- Arms on opposite sides of the target NEVER share a track — and with no
  cosmics crossing everything, there is NO track-based handle for cross-arm
  global alignment. It must come from survey and/or the reconstructed beam-spot
  position (each arm extrapolated to the target must agree). Plan for that.
