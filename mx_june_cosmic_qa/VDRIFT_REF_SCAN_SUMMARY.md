# Back-to-basics drift-velocity scan against the M3 reference (2026-07-17/18)

Scripts: `46_vdrift_ref_metric_scan.py` (scan, metric, displays, unshared +
three-way recon comparison), `46b_bias_anatomy.py` (angle dependence +
time-free extent), `46c_gas_scale_and_anglebins.py` (cross-field gas test +
per-angle-bin scan minima). Outputs in
`<Analysis>/mx17_3/vdrift_ref_metric_scan/` (det3 sat long run, 1000 V).
Assumed working gap: **29 mm** (user request; mechanical 30 mm).

**Full write-up: `vdrift_ref_metric_scan/report/main.pdf` (11 pp) and
`vdrift_ref_metric_scan/slides/slides.pdf` (12 frames)** — presents the test
result at face value (raw minimum 41–43 ≈ gap-filling 42.1, threads at 82%
clusters<1) and then the four independent discriminators:
1. angle-binned scan minima 53.0 / 42.0 / 38.5 / 36.5 µm/ns over
   |tanθ| 0.11→0.41 (a constant velocity cannot; floor model w≈2 mm fits);
2. time-free extent slope: visible column 24.5 mm ≠ 29 mm (implied v 35.5);
3. identical metric on unshared hits: min 35.0, per-cluster median 34.3;
4. cross-field: 29 mm/T_sat(E) = 29.7/39.0/43.2/45.2 fits NO surveyed gas
   (best RMS 2.65 = excluded Ar/iso 80/20; unshared series fits 95/5+1% H2O
   at RMS 0.59; even dry 95/5 only reaches 39 at 333 V/cm).

## The test (as requested)

Raw production hits, largest cluster per plane, `z = (t − t_earliest)·v`;
M3 line = mesh anchor + tangents rotated to the raw frame (same chain as the
3-D displays). Metric per hit: `pull = d / √(σ_ref² + (σ'_ref·z)² + σ_hit(z)²)`
with σ_ref = 0.21/0.24 mm (M3 self-resolution, raw frame), σ_hit(z) = 0.9+0.06z
(measured depth-resolved width). Cluster closeness = √(mean pull²). Scan v.

## Result of the scan on RAW hits

* The scan **does find a clean single minimum**: anchored median|d| **43.0**,
  offset-floated **41.25**, pull² **46.25** µm/ns — *not* 34, not 28.
* It sits almost exactly on the **gap-filling velocity 42.1** (= 29 mm / 689 ns
  median saturated time span). Superficially self-consistent with "v≈42 and the
  cloud fills the whole gap".
* Agreement there: med|pull| 0.48, median cluster closeness 0.70, 82 % of
  clusters < 1 (vs 0.62 / 0.79 / 77 % at v=34). In reference-only units the
  median hit sits 3.4 σ_ref (~0.7 mm) off the line — σ_ref is not the limit.
* Per-cluster floated v* median **49** (all strips) — and estimator-dependent
  (54 with the regression direction flipped, 42 core strips only): first hint
  that this is not a physical velocity.

## Why the raw minimum is NOT the drift velocity (three independent breaks)

1. **A real velocity is angle-independent; the raw one is not.**
   Binned in |tan θ_ref|: v* falls 62→39 (all strips) / 50→37 (core) from
   tan 0.10→0.32, tracking `v_true + w/(|tan|·T_span)` with w ≈ 2 mm — the
   resistive-spread spatial floor. It crosses *both* the 42 and 34 lines; no
   constant velocity does that. Extrapolated to steep angles it heads to ~34.
2. **A time-free column measurement contradicts gap-filling.**
   Cluster spatial extent vs |tan θ_ref| (positions only, no drift times):
   slope = visible column = **24.5 mm** (floor 2.0 mm), vs **29 mm** required
   by the gap-filling hypothesis. Implied v = 24.5 mm/689 ns = **35.5 µm/ns**.
3. **The identical metric on unshared hits lands on 34.**
   Same 400 most-inclined events: raw production min 39.0, raw wf-re-extract
   38.2, **unshared 35.0**, per-cluster median **34.3** — converging with
   v_geom (34) and Magboltz for the measured gas.

The 41 ≈ 42.1 coincidence is partly structural: scan-min ≈
(z_vis + w/tan_med)/T_span ≈ (24.5 + 2/0.2)/0.689 ≈ 42 for the population's
median angle — the floor happens to make up almost exactly the 29−24.5 mm
difference at ~11°. It is not evidence that the cloud fills the gap.

## What the test DID legitimately expose

* Drawing **raw** hits with v = 34 is an inconsistent pairing: raw clusters
  genuinely thread the reference best near v ≈ 41 (displays confirm visually).
  The 3-D displays should either use unshared hits at 34, or state that the
  raw depth axis uses an *effective* raw-ladder velocity, not the physical one.
* ~6.5 % of raw hits land above the 29 mm gap at the raw minimum (edge/RC
  strips) — a useful sanity flag for any future display velocity choice.
