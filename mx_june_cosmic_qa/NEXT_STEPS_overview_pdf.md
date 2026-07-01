# June overview / HV-scan PDFs — status & next steps

Status as of 2026-07-01. Deliverables under `~/x17/cosmic_bench/Analysis/` (not in repo;
regenerate from `run_full_june_qa.sh` / `run_hv_scans.sh`, or the direct calls below).

## Where we stand

**`june_detectors_overview.pdf`** — 5 pages (one per detector), best high-stats long run:

| Det | run / subrun | rays | eff (≤5 mm) | core σ |
|----:|---|---:|---:|---:|
| 2 | 6-22 overnight / long_run | 34.9k | 76.4 % | 0.84 mm |
| **3** | **6-28 weekend / long_run_p2 (top slot, FEU7/8)** | 53.0k | **79.7 %** | 0.80 mm |
| 4 | 6-24 daytime / long_run | 39.6k | 8.4 % | 1.24 mm |
| 6 | 6-26 overnight / long_run (7 files) | 32.3k | 55.4 % | 0.72 mm |
| 7 | 6-26 overnight / long_run (7 files) | 35.5k | 36.8 % | 1.36 mm |

- det3 page is now the **weekend 6-28** run (key `g_det3_wknd`), which wins the Detector-3
  page by ray count (`build_final_pdf.select_keys` = most rays per DET_NAME). Older 6-22
  det3 (bottom slot) was a consistent 80.7 %.
- Built with: `build_final_pdf.py g_det2 g_det3 g_det3_wknd g_det4 g_det6 g_det6_longer
  g_det6_long g_det7 g_det7_longer g_det7_long --out=.../june_detectors_overview.pdf`.
- Full QA per key (`process_key` in `run_full_june_qa.sh`): 01,02,04, 03 --full, 03
  --no-veto, 08, 09, plot_amplitude_vs_strip, 12_efficiency_map_sliding (--kernel=25 --grid=120).
- Data pulled from **lxplus** (rays DAQ PC was down); the run drivers assume rays, so a
  fresh machine needs the lxplus AFS path instead.

**`june_hv_scans.pdf`** — per-detector efficiency + resolution vs resist HV, now WITH a
**spark-rate overlay** (twin axis): det2 78 %@490 V, det3 81 %@490 V, det6 71 %@480 V,
det7 55 %@440 V. Spark = fraction of firing events with >50 strips; <10 % through the
plateau, climbs to 40–57 % where efficiency rolls off. Built by `build_hv_scan_pdf.py`;
metric added in `10_hv_scan_efficiency.py` (`spark_frac` column, `--spark=N`).

## Overview-page improvements — DONE 2026-07-01

All five implemented and the overview PDF regenerated for det2/3/4/6/7:

1. **Spark-rate stat card** ✅ — `09_efficiency_breakdown.py --spark=N` (default 50) writes
   `spark_frac=<x>%` into `efficiency_breakdown.txt`; `build_final_pdf.parse_breakdown` reads
   it and adds a 5th header card right of "Reconstructed" (green<8% / amber<20% / red≥20%).
   Measured: det2 7.4%, det3 9.1%, det4 5.6%, det6 27.3%, det7 38.9%.
2. **Sliding-efficiency maps for all detectors** ✅ — regenerated for every key.
3. **Fine per-detector kernel** ✅ — `12_efficiency_map_sliding.py --edge-hits=10` derives the
   SMALLEST fixed kernel giving ~10 rays at the active-area edge (from ray density), floored at
   `--kmin` (2.5 mm). Landed 4.1–5.5 mm per detector (was 25 mm), grid 140. The old fixed
   `--kernel=` / `--adaptive` k-NN modes still work.
4. **Resolution map vmax = 1 mm** ✅ — `plot_resolution_map_sliding(..., sigma_vmax=1.0)` (new
   optional arg; default None = old autoscale) wired from `03 --full`. Surface structure now
   visible (esp. det7, where the σ map shows the saturated-Y-plane bands).
5. **Overview layout** ✅ — `detector_page` now shows position + angular correlation 2D hists
   (`position_correlation.png`, `angle_correlation_corrected.png`) in the gs[2] row, and the
   efficiency breakdown moved to a full-width bottom row (gs 6×2).

### Possible follow-ups
- The correlation quad-plots + resolution 3-panel are dense at A4; could split to 2 pages/detector.
- det7 saturation veto (`local_max ≳ 3900`) to recover its true efficiency — still open (see
  JUNE_RESULTS_SUMMARY §5).
