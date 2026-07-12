# PLAN 41 — Publication figures + housekeeping

**Priority 5 — run AFTER plans 37–40 so their outputs are included. The paper should be
figure-driven (author's explicit preference: plots that tell the story, few words).
Everything below re-presents EXISTING results; no new physics. Do not change any
number — if a regenerated figure disagrees with the documented value, stop and report.**

## Ground rules

- One message per figure, single panel unless physically paired. No embedded
  monospace text tables, no 6–9-panel QA dashboards.
- Common style for all: a tiny shared module `paper_figs/style.py`
  (font ≥11 pt, consistent colors per detector — det3 and det2 keep the same two
  colors across ALL figures — vector PDF + 300 dpi PNG, no titles [captions carry the
  text], labeled axes with units).
- Output dir: `mx_june_cosmic_qa/paper_figs/` (in-repo, since the paper will live
  near the repo; sources under Analysis stay where they are).
- Sources of record for every number: `DET3_WEEKEND_ANALYSIS.md`, `MICROTPC_RUNBOOK.md`
  §0, `JUNE_RESULTS_SUMMARY.md`, plus plans 37–40 results sections. The LaTeX report
  rev 4 is STALE — do not copy numbers from it.
- Prefer regenerating from the cached CSVs (paths in `../PAPER_STATUS.md` figure
  inventory) over screen-scraping the QA pngs.

## Figure list (story order)

1. **F1 sharing constants / X-vs-Y spreading** *(new figure, paper point 1)* — from
   26's measured distributions (`bias_study/unsharing_sharing.png` inputs): per-plane
   neighbour/lead amplitude-ratio profile vs strip offset (−2…+2), x and y overlaid,
   annotated c1/c2; optionally det2 as hollow markers (design property). This is the
   "resistive strips matter, and here's the direction" figure.
2. **F2 the bias mechanism** — ONE clean strip-time ladder display (from
   `unsharing_ladders.png` inputs): measured ladder before vs after unsharing on a
   single representative inclined track, with the fitted slope lines. Plus (2b, may be
   a schematic) the existing `report_det3_weekend/figures/diagram_sharing.png` —
   regenerate via `report_det3_weekend/mk_diagrams.py` if style needs to match.
3. **F3 estimator convergence** — v_drift(1000 V) by method: production time-fit,
   4 waveform re-timings, unshared time-fit x/y, geometry, core-geometry, hybrid v_sig
   — as a dot-with-errorbar column chart converging on 34.3 ± 0.3. Numbers from
   runbook §0/§0b + `hybrid_vdrift_scan.csv`. (This one figure carries the whole
   "why unsharing is justified" argument.)
4. **F4 v(E) drift scan** — from `geometry_vdrift_scan.csv` + `hybrid_vdrift_scan.csv`
   + Magboltz curves (15's loader): v_geom and v_sig points 700–1100 V, Magboltz
   Ar/iso + {0,1,1.5}% H₂O bands, the 500 V point OPEN-symbol with a "window-truncated"
   annotation, ≤300 V absent. det2's 1000 V point included.
5. **F5 attachment money plot pair** — amplitude vs drift TIME (curves split) and vs
   DEPTH (curves collapse, single λ fit) side by side; from 19's inputs
   (`gap_attachment_test.csv` / recompute). Then **F5b det2/det3 overlay** = PLAN_40c's
   figure, restyled.
6. **F6 hybrid method schematic** *(new drawing)* — decision flow: waveform features →
   |tanθ| regression + L/R sign ⟶ (tan_reg > 0.09 & segment?) → time fit : signed
   regression. Follow `mk_diagrams.py` matplotlib-drawing patterns. Keep to ~6 boxes.
7. **F7 hybrid performance** *(the headline)* — σ68 vs |θ_ref| for production /
   track-only / regression / hybrid (det3 solid, det2 dashed), coverage as a light
   second panel or line. Rebuild from `hybrid_summary.csv` + `hybrid_summary_transfer.csv`
   binned curves (34 caches what's needed; if only band summaries exist, extend 34 to
   dump the per-bin arrays rather than re-deriving).
8. **F8 correlation before/after** — θ_det vs θ_ref 2D: production (exclusion gap
   visible) vs hybrid (continuous diagonal). From 34's inputs / `angle_correlation_*`
   equivalents.
9. **F9 position estimators** — σ68 vs angle band for production / early-charge /
   COMBO (from `position/position_summary.csv`), WITH the PLAN_37 M3-deconvolved
   values as the filled markers (convolved as hollow). Pitch/√12 and pitch lines drawn.
10. **F10 edge/fringe** — two panels from `32`'s data: (a) efficiency vs
    distance-to-edge turn-on (0→96 % over 0–25 mm), (b) outward δtanθ profile with the
    −0.06 dip and the ±0.007 systematics band; fiducial recommendation marked at
    25/40 mm.
11. **F11 HV operating curve** — det3 6-22 scan: efficiency + spark fraction (twin
    axis) vs resist HV from `hv_scan/mx17_3/efficiency_vs_hv.csv`; optionally the
    4-detector summary overlay as a second panel.
12. **F12 spark physics summary** — restyle 2–3 panels from `det3_spark_analysis/`
    (Poisson intervals, muon-enhancement bar, edge occupancy) + PLAN_39's ceiling
    curve as the closing panel.
13. **F13 scoreboard table** (LaTeX, not a figure) — consolidated: per-detector
    eff/σ/angular σ/spark (from JUNE_RESULTS_SUMMARY §1) and the 4-estimator × det3/det2
    hybrid table (from `hybrid_summary*.csv`), with PLAN_37 corrected σ where relevant.

## Housekeeping (do alongside)

- **Regenerate `angle_calibration.png` into the live v2 tree**: run
  `../.venv/bin/python 28_angle_calibration.py sat_det3 --veto=50` — it currently
  exists only under `alignment_tpc_veto50_prev2_backup/bias_study/`. Verify b_x≈+0.033,
  b_y≈+0.029 reproduce.
- **Add the missing script-23 figure**: extend `23_core_geometry_vdrift.py` to save a
  png of v vs core-fraction (20/30/40 %) next to its CSV.
- **Persist the unsharing before/after table**: extend `26`/`27` to write their
  stdout summary numbers (v before/after per plane, bias/resolution per band) to
  `bias_study/unsharing_summary.csv`. Do not change any computation.
- Re-render `report_det3_weekend/mk_diagrams.py` schematics only if styles are
  harmonized; otherwise leave.

## Acceptance

- Every figure's numbers cross-check against the markdown digests (spot-check each);
  any mismatch is a STOP-and-report, not a silent fix.
- Each figure has a one-sentence caption draft appended to this file (they become the
  paper captions — remember: the figures carry the narrative).
