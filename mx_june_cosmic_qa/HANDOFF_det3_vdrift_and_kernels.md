# Handoff — det3 v_drift puzzle + new plotting tools (2026-06-27/28)

Context for a fresh session. Two threads: (A) an UNRESOLVED det3 angular-correlation /
v_drift problem the user wants followed up, and (B) new plotting features added this session.

> **UPDATE 2026-07-01 — RESOLVED.** The universal off-diagonal angle correlation is
> explained and the drift velocity is now measured properly. See
> `DET3_WEEKEND_ANALYSIS.md` (same dir) for the full story. Short version:
> every muon crosses the full ~19 mm drift gap so the cluster TIME span is fixed
> (~gap/v), while the cluster SPATIAL width has a ~2 mm non-geometric floor from
> resistive charge spreading → the strip-fit slope is inflated additively:
> tanθ_det ≈ tanθ_ref ± w/gap  (w/gap ≈ 0.115, i.e. the ridge sits ~5-6° outside
> the diagonal, sign-following, angle-independent, identical in every detector).
> The σ-scan v_drift (30-36 µm/ns) is biased by this; the bias-free estimator is a
> straight-line fit of the (rotated) strip slope vs tanθ_ref per track sign
> (slope = v_drift, intercept = ±v·w/gap): `13_tpc_angle_bias.py` (diagnosis) and
> `14_drift_velocity_scan.py` (v vs drift HV from the 6-27 saturday drift scan).
> v_drift(1000 V) = 28.1 ± 0.7 µm/ns; the 6-27 drift scan maps the full v(E) curve
> with 1-3 % errors per 15-min point.

## A. det3 v_drift / angle correlation — OPEN, "something still wrong"

**Symptom.** For det3 (6-22 long_run, key `g_det3`), the `angle_correlation_corrected.png`
looks badly correlated and the `v_drift_scan.png` does not find a clean minimum — unlike
other detectors. User: "something is still wrong here, will follow up later." Treat my
conclusion below as PARTIAL, not closed.

**What I established (solid):**
- It is NOT the rotation handling and NOT the minimiser. `03_alignment_and_tpc.py` already
  passes `params=best` to `cm.plot_angle_correlation` (line ~204), so the 90° det→M3
  rotation IS applied. det2 (same run, same code, identically rotated) gives a textbook
  U-shaped minimum: v_drift X≈35 Y≈37 µm/ns, σ≈2.0–2.4°.
- det3 is plane-asymmetric:
  - **X plane (FEU 3): no minimum.** σ(v) decreases monotonically toward the
    σ-of-reference-angles asymptote as v→∞ (rails to scan max: 50 with [25,50], 100 when I
    widened to [25,100]). Signature that the X det-angle carries ~no information → the fit's
    best move is to shrink θ_det,X to 0.
  - **Y plane (FEU 4): weak real minimum** at v≈45, σ≈5.3° (vs det2 ≈2°).
- det3 POSITION resolution is good (~0.69 mm). So strips localise position fine; it's the
  DRIFT-TIME / micro-TPC slope (→ angle) that's poor, worst on X.

**Why the user may be right that more is wrong (open hypotheses to check next):**
1. FEU 3 (X) per-event `n_strips` / cluster size — if X clusters are tiny (≤ min_strips=4),
   `x_fit.slope_mm_per_ns` is noise. Check the distribution of `r.x_fit.n_strips` and
   `red_chi2` for det3 vs det2.
2. Per-strip TIMING on FEU 3 — pedestal/threshold or a time-reference offset could destroy
   the drift slope while leaving the position centroid intact.
3. The diagonal-projection scan uses `d = |θ_ref| - |θ_det|` (ABS of each) inside
   `plot_angle_correlation`'s v-loop (cosmic_micro_tpc_analysis.py ~line 2258), while the
   corrected scatter uses signed angles. The abs() folds sign and could mask/!distort the
   minimum — worth re-deriving with signed, sign-paired residuals.
4. Slope SIGN / axis swap on a 90° detector: verify det-X↔ref-Y, det-Y↔ref-X pairing is
   right for det3 specifically (the corrected scatter's lobe orientation tells you).
5. Cross-check against a DIFFERENT det3 run (e.g. 6-25 det3 day/long_run) to see if the bad
   X-drift is run-specific or intrinsic to that physical detector.

**Diagnostic recipe used:** edit the v-range in `03` line ~206 (`v_scan_min/max/steps`),
rerun `../venv/bin/python 03_alignment_and_tpc.py g_det3 --full` (reuses the per-event
cache, just re-aligns + replots, ~2 min), view `…/g_det3/alignment_tpc_veto50/
v_drift_scan.png` and `angle_correlation_corrected.png`. det2 is the control. NB I reverted
`03` to the production range [25,50] and reverted the unrelated library call
(`run_full_analysis` line ~209, which `03` does NOT use — leave its `params=None` for the
legacy det_4/n_TOF θ≈0 path).

Possible quick win: warn when a fitted v_drift lands on the scan boundary (railed) — that's
the honest "drift-angle unreliable" flag.

---

## B. New plotting features added this session (kept, committed)

1. **Correlation density plots** (`cosmic_bench_analysis/cosmic_micro_tpc_analysis.py`):
   `plot_position_correlation` and `plot_angle_correlation`'s corrected figure are now 2×2 —
   top row the existing scatter, bottom row a log-density 2-D histogram (`_density_hist2d`
   helper; viridis, LogNorm, cmin=1, shares the scatter's limits, draws y=x). Regenerates on
   `03 … --full`. Validated on det7 (Y-plane banding) and det3.

2. **Adaptive (k-NN) sliding efficiency kernel** (`12_efficiency_map_sliding.py`):
   `--adaptive --target=N [--maxkernel=R]` → each grid point uses the SMALLEST radius holding
   N rays (constant statistics, variable/locally-finest resolution). Output
   `efficiency_map_adaptive.png` + `.json` (incl. radius min/median/p90/p95/max and area
   fraction ≤2/5/10 mm). It also PRINTS a recommended edge-safe FIXED radius (≈ adaptive p90)
   so you can instead run a fixed kernel that still gives N rays at the edges
   (`--kernel=<p90> --min=N`). Fixed mode unchanged (used by build_final_pdf at --kernel=25).
   - On g_det3 (47.6k rays): target=20 → median radius 5.4 mm, p90 6.3, max 9.5.
     **mm resolution NOT reachable at these stats — needs ~0.5–1 M rays (~20×).** Combine
     subruns/runs to push finer; the adaptive map tightens automatically as stats grow.
   - Guidance: fixed edge-safe (~6–7 mm) for uniform-density dets (cleaner, undistorted map);
     adaptive when density varies a lot.

   TODO if wanted: wire `--adaptive` (or the edge-safe fixed radius) into `build_final_pdf.py`.

---

## C. Current bench run gotcha (2026-06-26/27)

`mx17_det3_weekend_scan_6-26-26` (det3-only resist×drift grid + hv_scan subruns; det3 now in
the TOP slot, FEU 3/4, z=702). The user noted they'd picked the WRONG FEUs. Indeed the
detector combined_hits are near-empty: `short_initial_resist_490V_drift_1000V` has just **2
events** (only FEU 4) vs **40,560 M3 rays** — so no alignment / efficiency / correlation is
possible on it yet (det3 essentially not reading out, or detector data still decoding while
the run is live). The "previous long_run" det3 data used for all testing above is the 6-22
long_run (`g_det3`), which is the good one. If revisiting the weekend run, re-check det3's
FEU mapping/readout first.

---

## Pointers
- Per-detector overview: `JUNE_RESULTS_SUMMARY.md`, `build_final_pdf.py`, `run_full_june_qa.sh`.
- HV scans: `build_hv_scan_pdf.py`, `run_hv_scans.sh`, `run_hv_scan_new.sh`.
- venv: `../venv/bin/python`. Analysis outputs under `~/x17/cosmic_bench/Analysis/<run>/…`.
