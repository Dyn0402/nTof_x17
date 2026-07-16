# ACTION NOTE — stricter M3 reference cut + true active-area outline

> **✅ EXECUTED (2026-07-14).** Job 1 step 0/1 (centralise + apply the recipe) and Job 1
> step 2 (re-run alignment + efficiency/residual fleet-wide) are done — see
> `../JUNE_RESULTS_SUMMARY.md` 2026-07-14 changelog and `../REPROCESSING_CONSTANTS.md` for
> the measured fleet numbers (det3 headline: 0.63→0.47 mm, 88.8→92.9 %, matching this
> note's predicted table almost exactly). Job 2 (outline overlay) is done for the golden
> 2-D position plots (08/09/12, the shared `plot_efficiency_map`/`plot_resolution_map`/
> `plot_resolution_map_sliding` used by 03, occupancy 04, edge/fringe 32, charge-balance
> 38, spark position maps) via `common/mx17_active_area.alignment_transform()`.
> **2026-07-14 follow-up: chi2<0.5 comparison (det3, both runs, +NClus=4).** Direct
> measurement (not just the differential extrapolation): efficiency is essentially FLAT
> (g_det3_wknd 92.9→93.0 %, sat_det3 93.4→93.5 %) while core σ keeps falling with no
> plateau (g_det3_wknd 0.473→0.432 mm, sat_det3 0.478→0.433 mm, ~9 % further) — confirming
> the note's original "no plateau" finding continues below chi2<1. Cost: stats kept drops
> from ~43 % to ~30 % (reconstructed events −29 %). Consistent with `m3_cut_scan.py`'s
> independent differential estimate (0.43 mm both ways). This supports the note's original
> tiering: chi2<1.0+NClus=4 for headline efficiency/position (current), chi2<0.5 as the
> **dedicated spatial-resolution number** where the extra stats cost is worth it and
> efficiency doesn't care. Full numbers: `m3_cut_scan.json`, `fig4_m3_cut_scan.png`.
>
> **2026-07-14 follow-up: Job 1 step 3 done.** The waveform-dependent chain (26–42:
> unsharing, calibration, micro-TPC metrics, hybrid tracking, position estimators, charge
> balance, time resolution) re-run fleet-wide (`sat_det3`, `o22_long_det2`, `g_det4`,
> `g_det6_long`, `g_det7_long`). Hybrid angular σ68 barely moved (≤0.28° fleet-wide,
> several detectors slightly improved) — confirms angle resolution is only weakly coupled
> to the M3 position cut. `build_final_pdf.py` rebuilt; HV scans re-run for det2/det3
> (peak +2.4/+5.0 pts). Two real bugs found + fixed along the way (empty-calibration-bin
> crash in `34_hybrid_tracking.py` on low-stats det4; missing malformed-frame guard in
> `36_position_estimators.py`, never previously run on det4). Full numbers:
> `REPROCESSING_CONSTANTS.md` 2026-07-14 section, `JUNE_RESULTS_SUMMARY.md` changelog.
> **2026-07-14 follow-up: det6/det7 HV scans + engineer package done.** All 4 remaining
> HV-scan runs completed (`g_det6_hv`/`g_det7_hv` dedicated 400–500V scans,
> `g_det6_long`/`g_det7_long` overnight 505–530V/480–505V points) — det6 peak
> 76.2→78.8% @480V, det7 peak 63.1→64.1% @440V, same clean pattern as det2/det3.
> `build_hv_scan_pdf.py` rebuilt with all 5 detectors. The engineer-package LaTeX report
> (`report/main.tex`, 23 pages, all 34 M3-recipe-dependent figures regenerated + every
> headline number reconciled including the operating/spark-free distinction) and the
> slide deck (`make_slide_deck.py`, 18 slides) are both rebuilt and recompiled cleanly
> (0 missing figures, verified against `pdftotext`). Full numbers:
> `REPROCESSING_CONSTANTS.md`, `JUNE_RESULTS_SUMMARY.md`.
> **2026-07-14 follow-up: last two items done.** (1) det3-frozen-model transfer
> cross-check re-verified on the new recipe (re-saved det3's model fresh, applied
> untouched to det2/4/6/7): every detector's degrade/improve direction and magnitude
> matches the pre-recipe-change pattern exactly (det2 +0.31° vs old +0.38°, det4 −0.42°
> vs −0.51°, det6 −0.53° vs −0.60°, det7 +1.13° vs +1.24°) — confirms the near-identical
> weights are a genuine design property, stable across the recipe change. (2) Measured
> the Y passivation on det2/4/6/7 too (generalised `edge_chi2_extract.py`/`_plots.py`
> from det3-only to any qa_config key): **17.9–20.4 mm across all 5 detectors, a ~2.5 mm
> spread consistent with measurement noise** — strong confirmation this is a common
> construction feature, not a det3-specific one. X passivation was NOT reliably
> re-measurable on det4/6/7 (their efficiency-vs-X profiles are corrupted by genuine
> hardware pathologies — det6's mesh defect, det7's weak-X gain gradient, det4's low
> gain — that fool the edge-finder into spurious tiny "active" windows; visually
> confirmed in `fig3_edges_passivation_g_det{4,6,7}*.png` panel c). X stays at the
> nominal sharp-edge value for every detector (confirmed clean on det2 and det3).
> `common/mx17_active_area.py` now has a `TRUE_ACTIVE_BY_DET` fleet table and
> `det_name=` threading through `draw_outlines()`/`active_area_corners()` and all ~20
> call sites (08/09/12/04/32/38, spark maps, the shared `plot_efficiency_map`/
> `plot_resolution_map`/`plot_resolution_map_sliding`) — every detector's plots now draw
> ITS OWN measured outline, not det3's, verified visually on det4 and det6. Efficiency
> maps (08/09/12) re-run fleet-wide to pick up the corrected outlines; engineer package
> report/deck recompiled clean. **All Job 1 + Job 2 work, plus both follow-ups, is now
> complete.**

**Written 2026-07-13** off the det3 position-miss investigation
(`report_v2.pdf`, this directory). Two follow-up jobs the owner asked for, both
for **another instance to execute** — this note is the brief, not the work.

---

## The finding (why this matters)

The detector's measured position residual is **limited by the M3 reference
tracker, not by the chamber.** On the current v2 recipe (χ²<5, NClus≥3) the det3
core σ(|r|<15) is **0.63 mm**, but that number is inflated: tightening the M3
χ² cut keeps improving it with **no plateau** inside χ²<5 (0.63 mm → 0.46 mm at
χ²<0.3). A differential fit puts the chamber's *intrinsic* core σ at
**≲0.3–0.45 mm** (cleanest bin measures 0.45 mm directly). ~Half of every >2 mm
"position miss" sits on an elevated-χ² M3 track that still passes the cut.
Identical on two independently-aligned runs (`g_det3_wknd`, `sat_det3`).

**Consequence:** every June residual / efficiency / angle number is quoted
against a reference that is dirtier than it needs to be. Reprocessing on a
stricter M3 cut sharpens the detector's true performance.

Reproduce: `edge_chi2_extract.py [KEY]` → `edge_chi2_plots.py [KEY]` →
`m3_cut_scan.py`. Numbers in `edge_chi2_meta_*.json`, `m3_cut_scan.json`.

---

## JOB 1 — reprocess EVERYTHING on a stricter M3 cut

### Recommended cut
**`chi²  < 1.0` on both planes  AND  `NClus = 4` on both planes.**

Rationale (both runs agree):

| cut | core σ | reco_far (>5mm) | within-2mm | stats kept | headline efficiency |
|---|---|---|---|---|---|
| χ²<5, NClus≥3 (current) | 0.63 mm | 6.5 % | 78.5 % | 100 % | 88.8 % |
| **χ²<1.0, NClus=4 (rec.)** | **0.47 mm** | **3.3 %** | **89.5 %** | **43 %** | **~91.5 %** |
| χ²<1.0, NClus≥3 (hi-stats fallback) | 0.52 mm | 4.0 % | 86.4 % | 68 % | 91.5 % |
| χ²<0.5 (resolution quote only) | 0.48 mm | 3.6 % | 89.0 % | 49 % | 91.8 % |

- There is **no natural knee** — σ keeps falling. χ²<1.0 captures the steep part
  of the gain; 43 % of tens-of-thousands of June crossings is still ample.
- If per-bin statistics are tight (fine 2-D maps), drop to `NClus≥3` (68 % kept).
- For the **dedicated spatial-resolution number**, push to χ²<0.5 (or quote the
  deconvolved intrinsic ≲0.3–0.45 mm), since σ has not converged even at χ²<1.
- **Efficiency rises** (88.8 → 91.5 %) and **reco_far nearly halves** (6.5 → 4.0 %)
  because reference-error events leave the sample entirely — this is expected and
  correct, not a regression. Re-quote the headline accordingly.

### Where to change it
`CHI2_CUT = 5.0` is **duplicated per-script** (grep `CHI2_CUT =` in
`mx_june_cosmic_qa/*.py` — ~25 scripts) and **NClus is only the M3RefTracking
default (`min_nclus=3`)**, not passed by the scripts. So:

0. **(recommended first) centralise** the recipe: add `M3_CHI2_CUT = 1.0` and
   `M3_MIN_NCLUS = 4` to `qa_config.py`, import them everywhere, and make each
   `M3RefTracking(...)` call pass `min_nclus=M3_MIN_NCLUS`. One lever afterwards.
   Alternatively bump the `M3RefTracking.__init__` defaults (`chi2_cut=1.0,
   min_nclus=4`) — but several scripts still pass `chi2_cut=20`/`5` explicitly, so
   audit all call sites either way.
1. Change every `CHI2_CUT = 5.0` / `M3_CHI2_CUT` / `chi2_cut=5.0` (and the stale
   `chi2_cut=20` in `05,06,07,16` and `cosmic_micro_tpc_analysis.py:160`) to the
   new recipe. Scripts that pass `min_nclus` implicitly (none) must now pass 4.
2. **Alignment uses the same cut** (`03_alignment_and_tpc.py`). Re-run alignment
   first on the stricter cut (cleaner, but fewer tracks — confirm each det still
   aligns; det4/6/7 have lower M3 yield). Everything downstream keys off the
   alignment, so this is step 1 of the actual reprocessing.
3. Re-run the golden chain per `REPROCESSING_PLAN.md` (efficiency 08/09, angle/
   resolution 31/33/34, unsharing 26/27, etc.), re-quote headline numbers, rebuild
   the overview PDF and `JUNE_RESULTS_SUMMARY.md` / `PAPER_STATUS.md`.

### Caveats / gotchas
- Do **not** change the spark veto (>50 strips) or the active-box logic — this job
  is only the M3 track quality.
- The efficiency active box (0.5–99.5 pct of reco positions) already excludes the
  passivated Y strips (see Job 2); keep that behaviour.
- Cross-check the resolution improvement transfers to det2 (has M3) before quoting
  fleet-wide; det4/6/7 have weaker M3 yield so the stricter cut costs more there —
  may need `NClus≥3` fallback for those.

---

## JOB 2 — draw the 40×40 and TRUE active-area outlines on all 2-D plots

### Permanent record (already committed)
`common/mx17_active_area.py` — the measured active area, importable:

- `NOMINAL_ACTIVE` = 40×40 cm (strips 0–398.58 mm both axes).
- `TRUE_ACTIVE` = X ∈ [0, 398.6] mm (full), **Y ∈ [18.0, 379.9] mm**
  (passivated **18.0 mm low + 18.7 mm high**, ~2 cm each — **Y plane only**;
  X has sharp geometric edges, no passivation). Efficient size ≈ **39.9 × 36.2 cm**.
- Helpers: `nominal_corners()`, `active_area_corners()`, and
  `draw_outlines(ax, transform=None, ...)` which plots both rectangles.

Measured on det3 (identical on both runs → physical). **Only measured on det3
so far** — the passivation is very likely a common construction feature, but
confirm per detector (re-run `edge_chi2_extract`/`_plots` for each) before
drawing the *measured* outline on det2/4/6/7 plots; until then draw det3's as
"nominal passivation" or just the 40×40.

### The task
Overlay **both** outlines (nominal 40×40 dashed + true active solid red) on
every 2-D area plot: efficiency maps (`08_efficiency_maps.py`,
`09_efficiency_breakdown.py` reco/nonreco position panels), spatial-resolution /
residual maps, occupancy, spark-position maps, the engineer package figures, etc.

**Frame handling (the one real subtlety):** `common/mx17_active_area.py`
constants are in **detector-local strip mm**. Plots come in three frames:
- *Detector-local strip frame* (0–398 mm): call `draw_outlines(ax)` directly.
- *Aligned / M3 frame* (most efficiency maps, e.g. `det_x_aligned_mm`): pass a
  `transform` that applies the run's alignment to the corners —
  i.e. `_rotate_det_positions`-equivalent forward map
  (θ, centre, offset from the run's `alignment.json`):
  `x' = cosθ·(x−cx) − sinθ·(y−cy) + cx + xoff`, `y' = sinθ·(x−cx) + cosθ·(y−cy) + cy + yoff`.
  (The inverse — aligned→local — is worked out in `edge_chi2_plots.py:inv()`.)
- *Reference/ray frame*: same alignment transform; watch `ref_x_sign`.

Add a small `alignment→transform` adapter (load `alignment.json`, return a
closure) so any script can do `draw_outlines(ax, transform=adapter(CFG))`.
Keep the legend entries consistent (`nominal 40×40 cm`, `true active area`).

---

## Pointers
- Finding + figures: `report_v2.pdf` (this dir), figs `fig1–fig4_*`.
- Permanent active area: `common/mx17_active_area.py`.
- Reprocessing context: `../REPROCESSING_PLAN.md`, `../REPROCESSING_CONSTANTS.md`,
  `../MICROTPC_RUNBOOK.md`.
- Memory: `june-spark-and-recofar-analysis`.
