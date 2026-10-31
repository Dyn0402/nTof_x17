# Figure provenance against pass 1 (2026-10-31)

PASS_PLAN §4: date every `papers/*/figures.toml` figure against the from-scratch pass. Pass 1 means code 6a0bb1f,
EOS `Analysis_pass1`, 2026-10-30, with post-fixes b5a29af/a72b889. Below, "basis" is the reduction the figure was made
from and when.

**Checks made:**
- **Byte-compare.** Every `current` source in the repo was compared with its committed copy in `papers/<paper>/figures/`
  (md5). All match except `bow_profiles` (see its row).
- **Creation dates.** Read from the PDF metadata.
- **Re-renders.** None of the data figures could be re-rendered here. Their inputs (`~/x17/...` reductions, hits
  tables, Gerbers) are on the desktop only, so they were dated, not re-rendered.

**Status keys:**
- **current**: pass-independent (a schematic or a raw display) or already on pass 1.
- **rebuilt**: re-made on pass 1 in this round.
- **pending-vt**: needs angles or v on pass 1, so it waits for the v(t) cross-check (`rerun/vt_all/`).
- **pending-sps**: needs the SPS final pass now running (`rerun/sps_audit/`).
- **stale**: predates pass 1 and must be re-made before submission.
- **placeholder**: nothing produced yet.

## Paper I

| figure | source (state) | basis, date | status | note |
|---|---|---|---|---|
| chamber_stack | none (placeholder) | spec page | placeholder | schematic, pass-independent |
| readout_layout | none (placeholder) | production drawing (Stephan) | placeholder | pass-independent |
| microtpc_principle | engineer_package 02 (current) | schematic, 07-12 | current | |
| bench_setup | engineer_package 00 (current) | schematic, 07-12 | current | |
| sps_setup | mpgd26 render (placeholder, source missing) | schematic | placeholder | inputs listed in `mpgd26/slides/HANDOFF_offline_rebuild.md` |
| hv_scan | engineer_package 30 (current) | June hits QA, 07-14 | **stale** | predates the 7-24 reprocessing (+40 % hits), the 10-23 re-decode and pass 1. Re-make efficiency vs HV from `Analysis_pass1/*/efficiency/` (efficiency session) |
| spark_deadtime | engineer_package 44 (current) | June hits QA, 07-14 | **stale** | allowed content (hit timing QA), but on the pre-7-24, pre-re-decode hits |
| charge_balance | `rerun/paper1_charge/figures/charge_balance.pdf` (current) | pass 1, 10-31 | **rebuilt** | was the 07-12 hits figure 61. §5's text numbers (det3 0.487, det2 0.531, σ68 0.07) are the hits proxy; pass 1 gives 0.485 / 0.513, s68 0.014 / 0.016 (`rerun/paper1_charge/README.md`) |
| efficiency_map | engineer_package 20 (current) | July recipe on hits, 07-14 | **stale** | re-make from the pass-1 `maps/` (manifest note D1) |
| edge_turnon | engineer_package 80 (superseded) | hits angle, 07-12 | superseded | stays faded; the turn-on panel needs a pass-1 remake |
| gas_vE | `rerun/paper1_res/figures/gas_vE.pdf` (current) | pass 1, 10-31 | **rebuilt** | det3 6-27 drift scan, angle-scale v (v_drift × mean kw: carries kw, not a timing v) vs Magboltz bench gas 0.95 % H2O; v(t) not relevant (short rows) (`rerun/paper1_res/`) |
| attachment | none (placeholder) | cloud_basics §16–27 (bench + run71), outside the repo | **pending-sps** | run71 is being re-run with the seed fix |
| timing | engineer_package 51 (superseded) | hit times | superseded | waveform timing (D3) not done |
| tracking_perf | `rerun/paper1_res/figures/tracking_perf.pdf` (current) | pass 1, 10-31 | **rebuilt** | σθ vs angle on interior_good, headline rows, v(t) arm where a map exists; efficiency from the pass-1 effbudget (`rerun/paper1_res/PASS1_RESOLUTION_2026-10-31.md`) |
| sps_resolution | spatial_resolution decomposition.png (placeholder, source missing) | run 53 hits centroid, 08-13 | **pending-sps** | the final SPS pass decides the run and the estimator |
| bow_profiles | `rerun/paper1_res/figures/bow_profiles.pdf` (current) | pass 1 golden rows, 10-31 | **rebuilt** | fb_bow_maps' estimator unchanged on pass-1 tables (`bow_pass1.py`), v(t) arm; pass arm `bow_profiles_pass.pdf`. The audit render stays in `fb_bow_maps/figures/` |
| bow_closure | `rerun/paper1_res/figures/bow_closure.pdf` (current) | pass 1 golden rows, 10-31 | **rebuilt** | as above |
| det4_stripes | none (placeholder) | `14_board_map.py`: hits efficiency on `alignment_tpc_veto50` + Gerbers | placeholder (stale basis) | that alignment is retired (calib-reference lesson). Re-make from pass-1 det4 maps |
| loss_band … loss_multi_deposit (14) | `~/x17/cosmic_bench/satstudy/appendix/*.pdf` (current; source missing on this machine) | Analysis_rr1 `final_phys` det3 table + phys bundle, 10-26 | **stale** | event displays with the pre-pass-1 fit overlaid. Re-pick and re-draw on the pass-1 det3 table with the A1 fractions (effbudget session) |

## Paper II

| figure | source (state) | basis, date | status | note |
|---|---|---|---|---|
| sharing_cartoon | engineer_package 03 (current) | schematic, 07-12 | current | manifest asks to check it shows c2 < c1 |
| kernel_forms | sharing_kernel forms_y.png (placeholder, missing) | run71 RAW stacks | **pending-sps** | |
| angled_kernel | angled_kernel kernel_ab.png (current; source PNG gitignored, not on this machine) | run57, August | **pending-sps** | run57 rows are among the seed-fix re-runs |
| ladder_compression | none (placeholder) | waveform_first_threading 05 (hits vs M3) | placeholder | pass-independent study; never produced |
| event_waveforms | stephan_cosmics display (current) | raw waveforms, 07-12 | current | manifest asks to check there is no hits overlay |
| forward_model | none (placeholder) | diagram | placeholder | |
| toy_floor | none (placeholder) | script 18 on the R&D model | placeholder | needs a re-run on the production model (R6) |
| tiers_benchmark | none (placeholder) | R1–R5 | placeholder | not a pass product: needs the hits tiers (H0–H2c) reconstructed on the same events; v(t) no longer blocks it (10-31) |
| threshold_neighbours | none (placeholder) | R4 | placeholder | |
| known_angle_geometry | angled_kernel geometry.png (current; PNG not on this machine) | schematic | current | |
| known_angle | none (placeholder) | SPS rotated runs | **pending-sps** | SPS tan sign not pinned (only \|tan\|), `COORDINATES.md` §4b |
| fleet | none (placeholder) | pass-1 angles + a hits arm | placeholder | the waveform side is in `rerun/paper1_res/results/summary.json` (headline s68, both arms); the hits side needs the hits tiers on pass-1 events, not produced. v(t) no longer blocks it (10-31) |

## Summary

| status | count | figures |
|---|---|---|
| current | 5 | microtpc_principle, bench_setup, sharing_cartoon, event_waveforms, known_angle_geometry |
| rebuilt on pass 1 | 5 | charge_balance; gas_vE, tracking_perf, bow_profiles, bow_closure (10-31, `rerun/paper1_res/`) |
| stale | 17 | hv_scan, spark_deadtime, efficiency_map, the 14 loss_* displays |
| pending-vt | 0 | (v(t) settled 10-31: the four data figures rebuilt; tiers_benchmark and fleet moved to placeholder, they need a hits arm) |
| pending-sps | 5 | attachment, sps_resolution, kernel_forms, angled_kernel, known_angle |
| superseded | 2 | edge_turnon, timing |
| placeholder | 10 | chamber_stack, readout_layout, sps_setup, det4_stripes (stale basis), ladder_compression, forward_model, toy_floor, threshold_neighbours, tiers_benchmark, fleet |

New pass-1 figures available but not in either manifest (no `.tex` references them yet):
`rerun/paper1_charge/figures/{gain_curves, drift_field, unsat_bias, golden_spectra}.pdf`.
