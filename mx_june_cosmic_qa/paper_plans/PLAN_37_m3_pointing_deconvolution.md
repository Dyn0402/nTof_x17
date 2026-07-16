# PLAN 37 — M3 pointing budget + intrinsic-resolution deconvolution

**Paper point 9 (spatial resolution). Priority 1 — every spatial σ quoted in the paper
is currently the convolution detector ⊕ telescope; this plan separates them.**

> **2026-07-14: the production M3 recipe changed to χ²<1.0 & NClus=4** (was χ²<5 &
> NClus≥3) — see `../det3_recofar_analysis/M3_CUT_AND_ACTIVE_AREA_NOTE.md`. This plan's
> "load rays WITHOUT the chi2 cut" step is unaffected (it needs the untruncated
> distribution regardless of what the production cut is), but every reference below to
> "the production chi2<5 cut" and the fleet core-σ table now means chi2<1.0 / the
> 2026-07-14 numbers — updated below. Deconvolving against a tighter production cut
> should find a SMALLER telescope-pointing contribution to subtract (less truncation to
> begin with), not a larger one.

## Goal

Compute the M3 reference telescope's pointing resolution at the detector plane
(σ_M3(z_det), per axis), then quote the detector's **intrinsic** resolution
σ_det = √(σ_meas² − σ_M3²) for the headline numbers: the fleet core-σ table and the
position-estimator benchmark (script 36). Also state (with one line of math) why the
M3 contribution to the ANGULAR resolution is negligible.

## Background you need

- M3 is a 4-station telescope. Station z positions: **24, 144, 1185, 1302 mm**
  (Z_Down = 24, Z_Up = 1302; see `TODO_m3_reference_6-23.md` around line 21 — verify
  there before hardcoding). The detector under test sits INSIDE the span:
  z ≈ 232 mm (bottom slot) or z ≈ 702–714 mm (top slot; use the run's
  `alignment.json` `z_mean`, e.g. 714 mm for `sat_det3`). So the ray position at z_det
  is an INTERPOLATION — error is modest but not negligible at the 0.6 mm level.
- The ray files expose only fitted-track quantities (see
  `cosmic_bench_analysis/M3RefTracking.py`): `X_Up/Y_Up` at `Z_Up`, `X_Down/Y_Down`
  at `Z_Down`, plus **`Chi2X`, `Chi2Y` = unweighted sum of squared residuals in mm²**
  (per coordinate) and **`NClusX`, `NClusY`** = number of stations used (3 or 4).
  Per-station cluster positions are NOT in these files.
- That is enough: for an unweighted straight-line least-squares fit of N points with
  common per-station resolution σ_st, **E[Chi2] = σ_st²·(N−2)**, and the fitted-line
  position variance at any z is the standard formula
  **σ_track²(z) = σ_st² · ( 1/N + (z − z̄)² / Σᵢ(zᵢ − z̄)² )**
  with z̄ = mean of the station z's used. So σ_st comes from the Chi2 distribution and
  σ_M3(z_det) follows analytically.
- Measured residual σ to deconvolve: production earliest-strip
  σx/σy = 0.76/0.83 mm (core Gaussian), and the script-36 σ68 table
  (`position/position_summary.csv`), e.g. early-charge centroid 0.61/0.72 mm @ θ<5°.

## Inputs (all verified to exist)

- Rays: `CFG.m3_tracking_dir` for run key `sat_det3` (use `qa_config.py`).
- Alignment: `.../mx17_3/alignment_tpc_veto50/alignment.json` (z_mean).
- Benchmark to correct: `.../alignment_tpc_veto50/position/position_summary.csv`
  (columns: estimator, coverage, σ68 per axis per angle band) and
  `position_estimates.csv` (per-event: `eid,plane,n_strips,cog_raw,cog_u,lead_u,early_raw,early_u,fit_t0`).
- Fleet core-σ values to correct (from `JUNE_RESULTS_SUMMARY.md` §1, 2026-07-14
  changelog, chi2<1.0 & NClus=4 recipe): det3 0.47, det2 0.46, det3(6-22) 0.46, det6 0.45,
  det7 0.59, det4 0.67 mm (these are 5 mm-fid core Gaussians — correct them with each
  run's own z_det; slot z from `qa_config.py`). Superseded chi2<5 values: det3 0.63,
  det2 0.64, det6 0.59, det7 0.86, det4 0.89 mm — do not use.

## Method

1. **Load rays WITHOUT the chi2 cut** for the σ_st measurement:
   `M3RefTracking(dir, chi2_cut=1e9, min_nclus=3)`. The production chi2<1.0 cut (since
   2026-07-14; was chi2<5) truncates the Chi2 distribution and would bias σ_st low —
   measure on the (nearly) untruncated sample, then note what fraction the recipe cut
   removes (a much larger fraction now than at chi2<5 — expect a bigger reported "removed"
   number, that is not a red flag).
2. **Split by NClus** (3 vs 4) and by coordinate (X, Y). For each subset compute
   Chi2/(N−2) per track. Estimate σ_st² two ways and require agreement:
   (a) robust location of Chi2/(N−2) — for a scaled χ²₍N−2₎ distribution the MEDIAN of
   Chi2/σ_st² is 0.455·(N−2) for N=3 (χ²₁ median) and 1.386 for N=4 (χ²₂ median), so
   σ_st² = median(Chi2)/χ²-median(N−2); (b) a fit of the Chi2 histogram to a scaled
   χ²₍N−2₎ shape over the bulk (exclude the far tail, e.g. fit below the 90th
   percentile). Report σ_st per coordinate; expect a few hundred µm (M3 strip pitch
   scale — sanity: if you get >2 mm or <0.1 mm something is wrong).
3. **Propagate to the detector plane.** For NClus=4 (all four stations
   z = {24,144,1185,1302}, z̄ = 663.75, Σ(zᵢ−z̄)² ≈ 1.358e6 mm²):
   σ_M3(z)² = σ_st²·(0.25 + (z−663.75)²/1.358e6). At z=714 the geometric factor is
   ≈0.252, i.e. σ_M3 ≈ 0.502·σ_st. For NClus=3 the factor depends on WHICH station
   dropped — you can't know per-track, so compute the factor for each of the four
   drop-one combinations and use the spread as the systematic; weight by the NClus=3
   sample fraction. Produce σ_M3(z) curves per coordinate with an uncertainty band
   (σ_st stat + NClus-3 mixture), and mark z=232 and z=714.
4. **Deconvolve.** For each entry of `position_summary.csv` and each fleet core-σ:
   σ_det = √(σ_meas² − σ_M3²) (per axis; use the run's z_det and the run's NClus mix).
   If σ_meas² − σ_M3² goes negative for any entry, do NOT clip silently — report it
   (it means σ_st is overestimated; revisit step 2).
5. **Data-driven cross-check (important, cheap).** Per matched event you have the
   residual (from `position_estimates.csv` vs the ray) and the track's own
   Chi2/(N−2) = per-track σ_st² estimate. Bin events by Chi2/(N−2) and plot residual
   variance vs it: it should rise linearly with slope ≈ the geometric factor from
   step 3, and the **intercept at Chi2→0 is σ_det² directly** — an independent
   deconvolution that never uses the analytic E[Chi2] calibration. Do this for the
   production estimator and the early-charge centroid, θ<5° band. Agreement of the
   intercept method with step 4 within ~10 % is the acceptance test of the whole plan.
6. **Angles (one paragraph, no big machinery).** var(slope) = σ_st²/Σ(zᵢ−z̄)² →
   σ_slope ≈ σ_st/1166 mm ≈ 3–7·10⁻⁴ in tan-space ≈ 0.02–0.04° — negligible vs the
   1.8° hybrid resolution. Compute the actual number and write it down.

## Outputs

- Script: `37_m3_pointing_budget.py sat_det3` (should also run for any key).
- `.../alignment_tpc_veto50/m3_pointing/m3_pointing_budget.png` — 4 panels:
  (a) Chi2/(N−2) distributions + scaled-χ² fits per coordinate; (b) σ_M3(z) curves with
  slot markers; (c) the step-5 residual-variance-vs-Chi2 scatter + linear fit;
  (d) before/after table: σ_meas → σ_det for the headline estimators.
- `.../m3_pointing/m3_pointing.csv` — σ_st (x,y), σ_M3(z_det), corrected σ_det for
  every row of position_summary.csv + the fleet core-σ list.
- Appended results section in this file.

## Acceptance checks

- σ_st(x) ≈ σ_st(y) within ~30 % (same telescope design both axes).
- Step-5 intercept σ_det agrees with step-4 quadrature σ_det within ~10 %.
- Corrected det3 production σ stays in a physically sensible range
  (0.5 mm < σ_det < σ_meas); the early-charge centroid at θ<5° should land somewhere
  around 0.5–0.6 mm if σ_M3(714) ≈ 0.2–0.4 mm.
- NClus=3 vs NClus=4 subsets give consistent σ_st after the (N−2) scaling.

## Gotchas

- Do NOT measure σ_st with the chi2<1.0 cut applied (truncation bias) — but DO apply the
  standard recipe (chi2<1.0 & NClus=4) when selecting the matched events whose residuals
  you correct (consistency with every other quoted number).
- `ref_x_sign` is applied to M3 x on load in the analysis scripts — irrelevant for
  Chi2 work, but if you recompute residuals from scratch follow the pattern in
  `09_efficiency_breakdown.py` lines ~50–75 (attach_reference_positions).
- Chi2 is UNWEIGHTED (mm², not normalized) — that is exactly why E[Chi2]=σ²(N−2) works;
  do not divide by an assumed σ first.
- The residual σ68 in position_summary.csv had per-estimator offsets removed — when
  redoing residuals in step 5, remove the median offset per estimator the same way.
- det4/6/7 fleet σ corrections use THEIR z (bottom slot z≈232 has geometric factor
  ≈(0.25 + (232−663.75)²/1.358e6) ≈ 0.387 → σ_M3 ≈ 0.62·σ_st — bigger than top slot).
