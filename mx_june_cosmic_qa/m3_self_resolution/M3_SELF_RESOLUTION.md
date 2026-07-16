# M3 four-plane telescope self-resolution & DUT pointing uncertainty

**Date:** 2026-07-15 · **Data:** MX17 det3 weekend "saturday" long run
(`MX17_long_run_resist_490V_drift_1000V_datrun_260627_22H49`, files 0–3),
resist 490 V / drift 1000 V. · **Analysis dir:** `mx_june_cosmic_qa/m3_self_resolution/`

This closes the paper's biggest open measurement gap ([[mx17-june-paper-status]] topic 9,
PLAN_37): the M3 reference-tracker pointing error at the DUT plane was never quantified, so
every DUT residual (0.47–0.63 mm) was an un-deconvolved DUT⊕M3 sum. Here we measure the M3
planes' own intrinsic resolution, bound multiple scattering, and propagate the reference
pointing uncertainty to the DUT — and answer *where a DUT out-resolves the reference*.

---

## 1. Why this is feasible — geometry

The "M3" reference tracker is **8 MultiGen-v2 (MGv2) strip planes in 4 stations**, each station
carrying one X plane and one Y plane (config `CosmicBench.MGv2`, `cosmic_bench_m3_tracking`):

| Layer | z (mm) | planes | doublet |
|:-----:|:------:|:------:|:--------|
| L0 | 1302 | X, Y | top    |
| L1 | 1185 | X, Y | top    |
| L2 | 144  | X, Y | bottom |
| L3 | 24   | X, Y | bottom |

So **each coordinate is an independent 4-point straight-line fit** → a genuine 4-plane beam
telescope. Structure = two close **doublets** (Δz ≈ 117 / 120 mm) separated by a **~1041 mm
lever arm**, with the DUT slots (z = 232, 702 mm) *interpolated* inside the gap. The producer
fits X and Y separately, unweighted (`Ray_2D::process` → `TGraph::Fit`), `Chi2 = Σ residual²`
(mm², no errors), `NClus` = planes used (≤4).

## 2. Data path — producer patch + reprocess

The rays tree stored only the fitted line endpoints (`X_Up/Down` = `eval_X(Z_Up/Down)`),
`Chi2`, `NClus` — **not** the per-plane hits. Minimal patch to expose them
(`cosmic_bench_m3_tracking`, uncommitted working-tree change):

- `include/Tray.h`, `src/Tray.cpp`: added branches `mX0..3`, `mY0..3` (aligned cluster
  position per station, NaN if absent) and `zX0..3`, `zY0..3` (plane z). Filled in
  `Tray::fillTree` from `Ray::get_clus()` (splitting on `get_is_X()` / `get_layer()`,
  reading `get_pos_mm()` / `get_z()`; the clones are `delete`d). **Does not change the fit.**
- Build: `make tracking` on ROOT 6.36.06 (`~/Software/root_6_36_06`).
- Reprocess (local, files 0–3): `DataReader config_sat_ped.json read|ped` →
  `DataReader config_sat_cos_all.json analyse` → `tracking config_sat_cos_all.json rays
  sat_perplane_0to3.root`. 41 455 rays over 47 452 events (matches the DAQ v2 rays count).

**Validation** (`validate.py`): recomputing `Σ residual²` from `(zX,mX)` reproduces the
producer's `Chi2X/Y` with **median |Δ| = 7×10⁻⁴ mm²** → the emitted positions *are* the fit
inputs. (A ~2%-on-χ² tail on some tracks is the producer's internal drop-layer / `perp_pos`
readout bookkeeping; irrelevant because the analysis refits independently.) No circularity from
the `angle_z` correction: it multiplies a track-derived perp position by tan(angle_z ≤ 0.005),
i.e. angle_z × (sub-mm perp error) ≈ few µm ≪ 0.4 mm.

## 3. Method

Single-track events only (`rayN==1`), 4-hit tracks (`NClus==4`) per coordinate.
Robust **Gaussian-core** widths throughout (iterative ±2.5σ fit) so the multiple-scattering
tails do not bias the intrinsic estimate.

**Per-plane intrinsic σ — geometric mean (primary).** For each plane *k*:
`σ_incl,k` = width of the *biased* residual (all 4 in the fit), `σ_excl,k` = width of the
*unbiased* residual (fit the other 3, predict *k*). Then
`σ_k = √(σ_incl,k · σ_excl,k)`.
Exact identity `σ_incl² = σ_k²(1−h_kk)`, `σ_excl² = σ_k²/(1−h_kk)` ⇒ product `= σ_k⁴`
cancels the leverage/geometry factor `h_kk`. Standard beam-telescope estimator.

**Cross-checks.** (a) *Equal-σ global*: assume one σ, `E[Σ residual²] = σ²(N−2) = 2σ²` from the
biased core. (b) *Simultaneous unbiased solve* `Var(u_k) = σ_k² + Σ_{j≠k} c_kj² σ_j²`:
**ill-conditioned** here (cond ≈ 2×10¹⁶ — within-doublet leave-one-out coefficients blow up),
so it is reported only as a caution, not used.

**Multiple scattering.** Inter-doublet kink `Δslope = slope_top − slope_bot`;
`Var(Δslope) = σ²_slope,top + σ²_slope,bot + θ_MS²` with the slope errors fixed by σ_k and the
(short) doublet lever arms. Highland cross-check with a rough budget
(air 1278 mm + 4 chambers ≈ 0.4% X₀ each + DUT ≈ 0.3% X₀ ⇒ x/X₀ ≈ 0.023).

**Pointing uncertainty.** For the full 4-plane *unweighted* fit (what the reconstruction
actually does), `ŷ(z) = Σ_k g_k(z) y_k` with `g_k(z)` pure geometry, so
`P(z) = √(Σ_k g_k(z)² σ_k²)`. This is the reference pointing error at any z; evaluate at the
DUT slots. `σ_DUT = √(σ_resid² − P(z_DUT)²)` deconvolves the DUT intrinsic resolution.

## 4. Results

### 4.1 Per-plane intrinsic resolution (Gaussian core)

| plane | z (mm) | σ_incl (µm) | σ_excl (µm) | **σ_k = √(incl·excl)** (µm) |
|:--|:--:|:--:|:--:|:--:|
| X L0 | 1302 | 278 | 618 | **415** |
| X L1 | 1185 | 304 | 553 | **410** |
| X L2 | 144  | 304 | 552 | **409** |
| X L3 | 24   | 275 | 613 | **411** |
| Y L0 | 1302 | 343 | 762 | **511** |
| Y L1 | 1185 | 373 | 679 | **503** |
| Y L2 | 144  | 331 | 600 | **445** |
| Y L3 | 24   | 307 | 683 | **458** |

- **X planes are uniform at σ ≈ 0.41 mm**; equal-σ global = 411.2 µm vs geo-mean average
  411.3 µm — agreement to 0.1 µm validates the equal-resolution premise (and the whole method).
- **Y planes σ ≈ 0.45–0.51 mm**, systematically worse than X and worse in the top doublet.
  Consistent with Y being the resistive charge-spreading direction ([[mx17-june-paper-status]]
  topic 1/4: extent floors y 4.7 vs x 3.1 mm; f = q_X/(q_X+q_Y) ≈ 0.49).
- These are the MGv2 **reference** planes' own resolution — first direct measurement.

### 4.2 Multiple scattering

| coord | σ(Δslope) meas | resolution-only | θ_MS (excess) | UL |
|:--|:--:|:--:|:--:|:--:|
| X | 7.42 mrad | 6.94 mrad | **2.6 mrad** | 2.7 |
| Y | 9.52 mrad | 8.12 mrad | **5.0 mrad** | 5.0 |

Highland: θ₀ = 1.8 mrad @1 GeV, 0.59 mrad @3 GeV. The measured core kink (θ_MS ≈ few mrad) sits
**above** the high-momentum limit, as expected for the soft cosmic component; the slope
resolution (≈7–8 mrad over the 120 mm doublet arm) **dominates** the kink, so MS is a
sub-dominant, poorly-resolved term here. Crucially, MS enters the DUT as the **non-Gaussian
tails** of the residual (visible in `figs/residuals_per_plane.png`), *not* as a broadening of
the core — which is exactly why the DUT residual keeps shrinking with tighter M3 χ²
([[mx17-active-area-and-m3-cut]]) with no plateau. The core σ_k and core P(z) below are
MS-free by construction (core fits), so MS is **not** double-counted.

### 4.3 Reference pointing at the DUT, and deconvolution

Pointing minimizes near the telescope centroid z̄ = 663.75 mm (that is why mid-gap is best):

| z (mm) | P_X (µm) | P_Y (µm) | **P_M3 (mean, µm)** | DUT resid core (µm) | **σ_DUT (µm)** | reference share of resid variance |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 232 | 255 | 282 | **269** | 470 | **386** | 33 % |
| 702 | 206 | 242 | **224** | 470 | **413** | 23 % |

(DUT residual core = 0.47 mm at the χ²<1 & NClus4 recipe, from
`det3_recofar_analysis` / [[mx17-active-area-and-m3-cut]]. The **sat det3 DUT sits at z = 702**
— FEU 7/8; z = 232 is the other slot, shown for completeness.)

**Deconvolved DUT intrinsic resolution σ_DUT ≈ 0.39–0.41 mm** (0.41 at the real z = 702 slot),
consistent with the "intrinsic chamber σ ≲ 0.3–0.45 mm" estimate from the χ²-scan floor.

### 4.4 Crossover — where does the DUT beat the reference?

`P_M3(z)` core exceeds σ_DUT (≈ 0.40 mm) only in the **extrapolation** regions:

| coord | above z ≈ | below z ≈ |
|:--|:--:|:--:|
| X | 1629 mm | −310 mm |
| Y | 1373 mm | −179 mm |

**Inside the entire M3 tracking volume (24 ≤ z ≤ 1302 mm) the reference always out-points the
DUT in the Gaussian core** (P ≈ 0.21–0.28 mm < σ_DUT ≈ 0.40 mm). A DUT plane out-resolves the
reference only if placed *outside* the stations (extrapolation), or when compared against the
**MS-degraded tail** of the reference (loose M3 χ²), where the effective P_M3 blows past the
DUT resolution — this is the regime the "reference-limited" language referred to.

## 5. Bottom line for DUT analyses

- **The M3 reference points to the DUT with a core σ ≈ 0.22 mm (z=702) / 0.27 mm (z=232).**
  It contributes ~23 % (z=702) / ~33 % (z=232) of the *variance* of the measured DUT residual —
  real but sub-dominant in the core; the DUT's own ≈ 0.40 mm resolution dominates.
- Subtract it in quadrature to report the DUT intrinsic: σ_DUT = √(σ_resid² − P_M3²).
- The reference is **not** the core bottleneck; its **tails** (MS-scattered, low-p muons) are.
  Tightening M3 χ² removes those and shrinks the residual, but there is no core plateau to gain
  below P_M3 — the DUT resolution is the floor.
- A DUT can only be validated as "better than M3" by comparing to the core reference, and in
  the current geometry that never happens between the stations. To *measure* a sub-0.2 mm DUT
  you would need either tighter/weighted M3 fits, a DUT placed to exploit the low-P mid-gap, or
  an independent higher-resolution reference.

## 6. Caveats / future sharpening

1. **DUT residual is an external input** (0.47 mm). A direct join of these reprocessed rays to
   the det3 `combined_hits` at z=702 (script-03 style) would make the deconvolution fully
   self-contained and give σ_DUT with a statistical error.
2. **Unweighted producer fit.** A σ-weighted (or GBL) fit would slightly reduce P(z); the
   quoted P is the as-reconstructed value, which is the honest one for existing DUT residuals.
3. **MS θ_MS is core-fit-level**, not a spectrum unfold; Y's larger value partly reflects its
   less-uniform σ_k. Treat θ_MS ≈ 2.6 mrad (X, cleaner) as the representative number.
4. One run / one gas point (det3, 490/1000 V). σ_k is a property of the MGv2 reference planes,
   expected stable, but re-run on another file set to get a systematic.

## 7. Reproduce

```bash
# (producer, one-time) patch Tray.{h,cpp} already applied; rebuild + reprocess:
cd ~/CLionProjects/cosmic_bench_m3_tracking
source ~/Software/root_6_36_06/bin/thisroot.sh && make tracking
./DataReader config_sat_ped.json read && ./DataReader config_sat_ped.json ped
./DataReader config_sat_cos_all.json analyse
./tracking config_sat_cos_all.json rays sat_perplane_0to3.root
# (analysis)
cd mx_june_cosmic_qa/m3_self_resolution
../../.venv/bin/python validate.py           # chi2 reproduction check
../../.venv/bin/python analyze.py            # -> results.json + figs/{residuals,pointing,sigma}
../../.venv/bin/python make_geometry_fig.py  # -> figs/geometry.png
pdflatex report.tex                          # human report
```

Files: `analyze.py`, `validate.py`, `make_geometry_fig.py`, `results.json`,
`sat_perplane_0to3.root`, `figs/`, `report.tex` → `report.pdf`.
