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

## Next time (improvements to `build_final_pdf.py` / `12_efficiency_map_sliding.py` / `03`)

1. **Spark rate as a per-detector stat card** on the overview page — add a 5th header card
   to the RIGHT of "Reconstructed". Needs the long-run spark fraction surfaced first:
   compute it in `09_efficiency_breakdown.py` (or 08) and write it into
   `efficiency_breakdown.txt`, then read it in `build_final_pdf.parse_breakdown` + add the card.
2. **Sliding-efficiency plots for the detectors that lack them** — verify every page has
   `efficiency/efficiency_map_sliding.png`; run `12_efficiency_map_sliding.py <key>` for any
   missing (check det2 `g_det2`, det4 `g_det4` in particular).
3. **Shrink the sliding kernel radius to ~2.5 mm** (currently `--kernel=25`). Target as small
   a kernel as possible that still gives ~10 hits/kernel at the *edges* — likely per-detector
   (tune `--kernel` per key; low-efficiency dets like det4 need a bigger kernel for the same
   edge count). `12_efficiency_map_sliding.py` `--kernel=` (+ maybe raise `--grid`).
4. **Resolution map colour scale: fix vmax = 1 mm** so the true surface variation is visible
   and not washed out by the large edge values. In the sliding resolution map plotting
   (`plot_resolution_map_sliding` in `cosmic_micro_tpc_analysis`, called from `03 --full`) set
   `vmax=1.0` (expose as an arg rather than autoscale).
5. **Overview page layout (`detector_page`)**: replace the *binned efficiency map* (gs[2,0])
   with the **position-correlation and angular-correlation 2D hists**; **move the efficiency
   breakdown down to the bottom row** to make room. (Position corr = `alignment_tpc_veto50/
   position_correlation.png`; angle corr = the `plot_angle_correlation` output from `03 --full`.)
