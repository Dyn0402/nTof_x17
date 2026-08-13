# The June fleet report and plot explorer

Two products, both under `Analysis/fleet_report/`, both regenerated from the
promoted `wft/` products — never hand-edited.

| file | what it is |
|---|---|
| `report.html` | the june_grand_qa.pdf layout: fleet summary, then one page per detector. The summary. |
| `explorer.html` | one plot per subject, zoomable, each with its data as CSV. The place to look closely. |
| `explorer_selfcontained.html` | the same page with every image inlined — for the site / offline reading |
| `plot_data/fleet.csv` | the fleet table, machine-readable |
| `<OUT_BASE>/wft/plot_data/rays.csv` | per detector: one row per reference ray. Every per-detector figure is a projection of it. |

## Rebuilding, in order

```bash
P=../../.venv/bin/python                       # from mx_june_wft/report/
# 0. accounting must be current for the keys you are about to plot
#    (01_alignment, 02_efficiency x2, 03_angles, 04_maps, digest — see below)
$P ../quality_investigation/corrected_angles.py --keys g_det3_wknd,g_det2,...
$P export_plot_data.py                          # -> plot_data/rays.csv + summary.json
$P make_june_figs.py                            # -> the report's per-detector figures
$P make_grand_report.py                         # -> report.html
$P make_plot_explorer.py --selfcontained        # -> explorer.html (+ portable copy)
```

**Do not use `run_chain.sh` to refresh accounting on a promoted product.** Its
first stage rebuilds the bundle from `hyper_v2.json` and its reco stage
overwrites the parquet — both destroy what the campaign promoted. Call the
numbered stages directly (`condor_campaign/rerun_20260813/run_accounting.sh` is
the working example).

## Things that will bite you

- **`angles_w0corr/` is what the report and digest quote**, not `angles/`.
  Since 2026-08-13 reco applies w0/kw itself and stamps
  `events.meta.json:angle_constants.applied`; `corrected_angles.py` then
  re-runs the accounting on the live table instead of correcting it twice.
  Skipping it entirely — the obvious implementation — leaves the PREVIOUS
  generation's angles sitting there to be quoted beside today's efficiencies.
- **`corrected_angles.py` iterates `FLEET`**, which carries `sat_det3` for
  det3, while the report's Detector A is `g_det3_wknd`. Use `--keys` for
  anything outside the fleet list, or A never gets refreshed.
- **`make_june_figs.py` picks its table from the stamp.** Corrected-in-reco →
  the live parquet; older tables → `angles_w0corr/events_w0corr.parquet`.
  Both write `angles_fullcoverage.json` into `angles_w0corr/`.
- **Detector B is deliberately two entries.** `g_det2` (6-22 long_run, 19,054
  rays) is the one to look at; `o22_long_det2` (longer_run, 3,678 rays) is what
  the June PDF used, and the report keys B on it so the continuity table
  compares like with like. Changing that means re-basing the continuity row.

## The explorer's own conventions

- 1-D plots are **SVG** (sharp at any zoom); 2-D densities are **PNG at 260 dpi**,
  sliding maps at 170 dpi (smooth fields — the extra dpi is empty detail).
- Every plot writes the numbers actually drawn (`bins`, profile points, grid
  cells) next to it. The full ray table stays in the Analysis tree.
- **Sliding scans step 2 mm; the circle is NOT 2 mm.** At June ray densities a
  2 mm circle holds ~2 rays. The radius is set to hold ~150 rays (`TARGET_N`),
  and every caption states the radius, rays per circle and the binomial spread
  at that size, so noise can be told from structure.
- The sliding width estimator re-tightens its window onto the core. It is a
  windowed RMS and lands BETWEEN the report's Gaussian core sigma and sigma68
  (A: 0.53 vs 0.44 and 0.63) — compare map to map, not map to headline.
- The sliding time map is the X-Y plane time difference over root two
  (trigger jitter cancels), ~59 ns/plane. That is **sampling-limited** at 60 ns
  DAQ sampling and is NOT the scintillator-referenced time resolution.
- The chi2 scan can only **tighten** the reference cut: reco ran on rays
  already inside the frozen recipe. `NClus` is uniformly 4 in these runs, so
  there is no NClus scan to make.
- Scan points are reconstructed with each detector's **nominal** bundle, so
  away from nominal the reconstruction is off-calibration even where the
  detector is fine: read charge as physical response, reco fraction as
  calibration transfer.
