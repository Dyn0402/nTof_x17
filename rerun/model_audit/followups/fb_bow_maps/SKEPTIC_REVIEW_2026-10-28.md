# Skeptical review of the chamber-bow conclusion (FB, 2026-10-27) — 2026-10-28

Asked for: an objective check of "the bow is real: mesh centre 1.5–2.3 mm lower than its rim on all five chambers, drift
direction tilts with it". Re-ran the position route with alternative estimators and selections on the five golden rows,
regressed the cell heights on the bias a slope-scale error would leave, and tested the M3 telescope itself for a bow
on the per-station rays. Scripts and raw output in `skeptic/`; nothing in `results/` or the report was changed.

## Verdict in three lines

1. **A bowed surface in the drift volume is real.** The drift-direction tilt (w0 ramp, and the amplitude-only
   narrowest-cluster route) is the solid part: it is M3-independent, agrees between the two routes in magnitude
   (not just sign), and none of the alternatives below can produce it. Field tilt at 150 mm from the centre: 0.5–1.7°,
   outward.
2. **That the bowed surface is the *mesh* is probable but not airtight, and its depth is 1.2–1.9 mm with a
   ±0.4 mm systematic per chamber, not 1.5–2.3 ± 0.2.** The position route (the only route that names the mesh)
   carries a reconstruction-side bias of the same form as the signal; on det6 it is most of the map, on det4 a
   quarter, on det2/det3 negligible. The 50 µm kapton cathode can sag by a comparable amount and the tilt routes
   cannot tell the two electrodes apart (both sagging the same way give the same tilt).
3. **The M3 telescope is not the explanation.** A common dish of the four M3 planes mimics the position route exactly;
   measured on the per-station rays it is 0.0–0.6 mm at 150 mm depending on cuts (≤ 30 % of the chamber number), and
   plane-to-plane bows are ≤ 0.3–0.5 mm. This tightens the report's "0.5–1.3×" to roughly "≥ 0.7×".

## What was checked and what it gave

### A. Alternatives for the position route (r = p0 − ref_mesh, h = dr/dtan)

| Alternative | Test | Result |
|---|---|---|
| Acceptance leak: a position-only quadratic residual (local pitch/alignment) leaking into tan·U² through corr(tan, position) | add U², W², UW without tan to the basis | every curvature term unchanged to ±0.02 → excluded |
| M3 plane-scale | residual scale bu, bw | ≤ 0.15 % (report) → excluded |
| Common bow of the M3 planes (δ(x) = k x² on all four) | per-station rays `sat_perplane_0to3.root`: 4-point residual pattern ∝ tan³·(dz² − mean) = (+,−,−,+) | X: +0.34 ± 0.14 (χ² < 5), +0.10 ± 0.12 (χ² < 2), −0.01 ± 0.19 (|tan| < 0.3) mm at 150 mm; Y: +0.20 ± 0.11, +0.63 ± 0.10, +0.62 ± 0.17, +0.14 ± 0.26. **Bound: \|δ_M3(150)\| ≲ 0.6 mm.** The per-plane tan³ coefficients are dominated by a (+,−,+,−) pattern (mirror-mounted planes within each doublet: the planes' own angle response), orthogonal to the dish pattern |
| Relative bow of one M3 plane | unbiased residual slope vs tan in 8 bins of position on the plane | ≤ ±0.3 mm structure across ±250 mm (X); Y bottom doublet shows a relative *tilt* (±1 mm over 500 mm), not a bow |
| Slope-scale position bias: if the fitted line is right at the charge centroid and wrong at t0 (w ≠ v·tan locally), p0 is biased by (v − b)·τ̄·tan → a fake height (v − b)·τ̄ that follows the local ladder slope b | regress the 5×5 cell heights on (v − b)·τ̄ (τ̄ = median charge-arrival time after t0) | **det6: slope +1.19 / +1.26 (X / Y), r² 0.59 / 0.71** — unit slope, i.e. the mechanism as predicted; det4: slope 0.5 / 0.7, r² 0.13; det2, det3: the term's spread is 0.3 mm, nothing to explain; det7: r² ≈ 0 |
| same, as an estimator | pivot-anchored residual r + (w − v·tan)·τ̄, full 2-D fit | depth at 150 mm: det2 1.86, det3 1.41, det4 1.53, det6 1.15, det7 1.54 (report: 1.77, 1.48, 1.85, 1.79, 2.31). det6 map spread 1.58 → 1.0 mm with X–Y agreement unchanged |
| t0 treatment | report's pinned variant | individual terms move by up to ±40 % (det2 X: 1.51 → 1.10, 1.33 → 1.89); it is a ±30 % systematic on the terms, not "79–97 % survives" |
| Same-surface test: X and Y strips must give the same kuu and the same kww | xy pulls from `summary.json` | kww: X 1.33 / Y 2.21 (det2), 1.07 / 1.92 (det3), 1.20 / 1.88 (det6): 2.8–3.6σ apart; the own-axis term is 1.2–1.8× the other view's reading of the same term on 9/10 views, under either estimator. A per-view systematic of ~±30 % on the curvature, acknowledged in the report as "edge truncation" but not propagated |
| Steep vs shallow tracks (geometry is tan-independent) | \|tan\| < 0.2 vs > 0.2 | other-axis terms stable within ~2σ; own-axis terms not (det7 X: +2.44 ± 0.37 vs −1.47 ± 0.96) |
| Shape: parabola or rim hook | 1-D profiles (9 bins) and inner-250 mm fits | own-axis profiles: flat over the central ±100 mm, rising in the outer 75 mm, on most views (det2 X: +1.5, +1.5, 0.2, 0.1, 0.0, −0.1, −0.1, +0.6, +1.8). Other-axis profiles (away from the strip-plane edges): det2 X, det3 X, det4 Y, det7 X parabola-like; det3 Y, det4 X, det6 Y flat-middle + hook; det6 X, det7 Y tilt-dominated. Inner-region curvature errors are ~±1 mm per view, so the positions **cannot resolve the shape inside ±100 mm**; the parabola is established by the w0 route (linear through the centre, errors 0.03 µm/ns), not by the positions |
| det7 X faulty half | ok-fraction by x: 0.54–0.93 for x < 200 mm vs 0.98 above (standing rule: quote det7 X from ch 256–511 only) | det7 X's own-axis profile is +6.4 mm in the first bin and flat elsewhere; det7's 2.31 mm includes the faulty half and should be redone masked |

### B. Alternatives for the drift tilt (w0 ramp, narrowest cluster)

| Alternative | Why it fails |
|---|---|
| Edge truncation of wide clusters | shifts the narrowest-cluster vertex to *positive* tan at the +u rim (deep charge beyond the edge is lost); the dish gives negative; measured slopes are negative on 9/10 views, and agree with the w0 route in magnitude: per m, X: det2 −0.061 / −0.099 (meas / pred), det3 −0.117 / −0.089, det4 −0.102 / −0.103, det6 −0.176 / −0.163, det7 −0.128 / −0.080; Y: −0.185 / −0.156, −0.133 / −0.098, −0.263 / −0.184, +0.158 / −0.184, −0.100 / −0.135 |
| M3 angle error ∝ position (plane scale) | needs 6 % at the chamber, measured ≤ 0.15 % |
| Field cage / fringe | decays within ~gap/π ≈ 10 mm of the frame; w0 ramps linearly through the centre on 8/10 views |
| Lateral gain gradient within the footprint | centroid bias σ²·g'/g ~ 10⁻³ mm; two orders too small |
| Asymmetric, delayed sharing varying across the board (resistive/readout strip registration) | would need a registration offset of a few hundred µm varying linearly across a lithographed board; and the X-view ramp is along x, the Y-view ramp along y (an along-strip RC effect would be the other way round) |
| M3 common bow | does not move tan_M3 at tan = 0 (the w0 intercept) nor the amplitude-only vertex |

So the tilt is real. What the tilt measures is the *sum* of the two electrodes' sags: with mesh height m(x) = k x² and
cathode height c(x) = G + k' x² (both lower at the centre), the lateral drift per unit depth is α = (k + k')·x. A
cathode foil sagging **toward** the mesh at the centre gives the same outward tilt as a mesh sagging **away** from
it (checked with the wedge-field picture both ways; a cathode doming away gives an *inward* tilt and is excluded).
From the w0 gradients (cu·100/v, ×1.125 for 150 mm) the sum k + k' at 150 mm is: X views det2 1.2, det3 0.66,
det4 0.92, det6 1.95, det7 0.83 mm; Y views 2.1, 0.97, 1.77, 2.2, 1.4 mm.

### C. Which electrode

- The position route names the mesh (the t0 plane). After the M3 bound it says the mesh is dished by 1.2–1.9 mm,
  with the reconstruction caveat above (det6 unreliable, det7 X unmasked).
- The cathode is a 50 µm kapton + copper foil (`02_design.tex`). Under its own weight (≈1.2 Pa) a 0.4 m square
  membrane sags ≈ 0.014 m²·Pa / T: 1.4 mm at 10 N/m tension, 0.3 mm at 50 N/m. Gravity pulls it *toward* the mesh
  (cathode up), i.e. the direction that adds to the tilt. "Cathode mostly flat" is therefore not guaranteed by physics;
  it is what the closure r_f = (k + k')/(2k) = 0.4–0.9 says, and with the position-route systematics that is a
  statement at the ±1 mm level (k' = k(2r_f − 1) = −0.2 k to +0.8 k).
- The July X-plane column maps (`GAP_STUDY_2026-07-30.md`: det2 30.5 mm in the interior drooping to ~29.3 in the outer
  50–70 mm; det3 longest at the centre) have the gap *longer* at the centre, which is the mesh-sagging-away direction
  and against a cathode sagging toward the mesh. Weak (±1 mm calibration systematic, amplitude-dependent endpoints),
  but it is the only route that separates the electrodes, and it points the report's way.
- The Y-view tilt gradient is 1.5–1.9× the X-view's on det2, det3, det4, det7 (equal on det6), and the amplitude-only
  route shows the same anisotropy on det2, det3, det4. An elliptical dish, more curved along y, is what a board
  resting on two rails running along x would do. Which way the rails run is a drawing/photo question, not a
  measurement, and would be a clean falsifiable check of "it is the board".

### D. Stability claim

15/19 runs within 2σ is consistent with "static", but the per-row errors (0.14–0.8 mm) are too large to see a 20 %
change, and the two det3 bottom-slot rows disagree with each other (0.93 ± 0.14 on 6-25 vs 1.70 ± 0.14 on 6-23, the
latter with broken timing). Mesh-HV and drift-HV rows have 380–780 muons and ±0.3–1.2 mm errors: uninformative.
"Static per chamber to ~20 %" is not demonstrated; "no run contradicts a static bow at the ±0.5 mm level" is.

## What I would change in the wording (REPORT.md, `08b_bow.tex`, the slide note)

- "Three routes with no shared systematics" → two routes share the wft line fit (p0 and w come from the same fit;
  the t0 trade-off moves both); the narrowest-cluster route is the only fit-independent one.
- Depth: "1.2–1.9 mm, ±0.4 mm systematic per chamber (estimator, view, t0 treatment); det6 dominated by a
  reconstruction term, det7 X unmasked". Drop the ±0.2.
- M3: "a common bow of the telescope is measured at 0.0–0.6 mm (per-station rays) → at most a third of the chamber
  value" instead of "cannot be excluded, 0.5–1.3×".
- Shape: the parabola is from the w0 route; the positions only give rim-minus-centre.
- "Cathode mostly flat": state it as r_f = 0.4–0.9 ⇒ cathode sag between −0.2 and +0.8 of the mesh's, and that a
  kapton foil at low tension sags by that much.
- Keep: the tilt, its sign (outward at the rim), its magnitude, that it is static enough to transfer (94 % of the gain),
  and the correction. Those stand.

## Not done / limits

- The M3 test uses one run (det3 weekend, the only per-station file). The alternating (+,−,+,−) response pattern is
  orthogonal to the dish pattern by construction but its tan-odd part is large (±5 mm per unit tan³), so the bound is
  a bound, not a measurement.
- The slope-bias mimic was tested by regression and by an alternative estimator, not by a toy of the wft fit; the
  unit slope on det6 is the evidence.
- No new data anywhere; nothing here needs any.

Files: `skeptic/refits.py` (A: extra terms, inner region, tan subsets, cell maps, t0 maps; output `refits.out`),
`skeptic/pivot_test.py`, `skeptic/mimic_reg.py`, `skeptic/other_axis.py`, `skeptic/m3_sag_test.py`,
`skeptic/m3_robust.py`. Inputs: `~/x17/cosmic_bench/model_audit/ws_v_w0/det*.parquet`,
`mx_june_cosmic_qa/m3_self_resolution/sat_perplane_0to3.root`.
