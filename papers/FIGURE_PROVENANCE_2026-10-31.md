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
| hv_scan | `rerun/paper1_ops/figures/hv_scan.pdf` (current) | pass 1, 10-31 | **rebuilt** | was the 06-29 hits figure 30 (det3 only). Now det2/det3/det6/det7: interior_good efficiency (effbudget), tracks lost to discharges vs bursts, discharges per trigger; golden rows starred (`rerun/paper1_ops/README.md`) |
| spark_deadtime | `rerun/paper1_ops/figures/spark_deadtime.pdf` (current) | pass 1, 10-31 | **rebuilt** | all five chambers on the rr1 hits + the budget's rays: efficiency and charge vs time since the previous discharge; dead time per discharge ≤ 1.2–2.6 ms (95 %, det2/3/6/7), det4 ≤ 15 ms |
| charge_balance | `rerun/paper1_charge/figures/charge_balance.pdf` (current) | pass 1, 10-31 | **rebuilt** | was the 07-12 hits figure 61. §5's text numbers (det3 0.487, det2 0.531, σ68 0.07) are the hits proxy; pass 1 gives 0.485 / 0.513, s68 0.014 / 0.016 (`rerun/paper1_charge/README.md`) |
| efficiency_map | `papers/paper1/figures/efficiency_map.pdf` (current) | pass 1, 10-31 | **rebuilt** | all five chambers, plateau pools, 8 mm disk, the loss budget's denominator (`rerun/paper1_figs/make_effmap_figure.py`). The caption still says "(det3)" |
| edge_turnon | `papers/paper1/figures/edge_turnon.pdf` (current) | pass 1, 10-31 | **rebuilt** | det3: turn-on, the perpendicular view's tan bias against M3, the column duration (`rerun/paper1_figs/README.md`: the x sides carry a drift-field fringe over 20–30 mm). Was the hits-angle dashboard |
| gas_vE | `rerun/paper1_res/figures/gas_vE.pdf` (current) | pass 1, 10-31 | **rebuilt** | det3 6-27 drift scan, angle-scale v (v_drift × mean kw: carries kw, not a timing v) vs Magboltz bench gas 0.95 % H2O; v(t) not relevant (short rows) (`rerun/paper1_res/`) |
| gas_vE_timing | `rerun/paper1_gas/figures/gas_vE_timing.pdf` (current) | pass 1, 10-31 | **new** | det3 6-27 drift scan: timing v (gap / T, 500–1100 V) and angle-scale v (all fields) on Magboltz 0.97 % (fine grid air_merged), the other chambers' golden timing v; recommended for §6 over gas_vE (`rerun/paper1_gas/`) |
| attachment | `rerun/paper1_spsfigs/figures/attachment.pdf` (current) | SPS gas fit 4127f9ea (gap 27.9 mm) + bench §6 summary, 11-01 | **rebuilt** | water (bench golden rows vs SPS epochs) and O₂ at SPS; the bench det4 point uses the old 30.48 mm gap (337f7fd1: 28.6 mm) |
| timing | `rerun/paper1_timing/figures/timing.pdf` (current) | pass 1, 10-31 | **rebuilt** | waveform column-start timing (D3): HV scans, the uncorrelated part vs amplitude with det4 at SPS, the golden-row split (`rerun/paper1_timing/`) |
| tracking_perf | `rerun/paper1_res/figures/tracking_perf.pdf` (current) | pass 1, 10-31 | **rebuilt** | σθ vs angle on interior_good, headline rows, v(t) arm where a map exists; efficiency from the pass-1 effbudget (`rerun/paper1_res/PASS1_RESOLUTION_2026-10-31.md`) |
| sps_resolution | `rerun/paper1_spsfigs/figures/sps_resolution.pdf` (current) | final SPS chain, telescope tables on the final rule (sps_junk final_rule.json), 11-01 | **rebuilt** | 3-zone decomposition, run63 X/Y, run53 Y; X zone fits fail (χ²/1 7–88), drawn open |
| bow_profiles | `rerun/paper1_res/figures/bow_profiles.pdf` (current) | pass 1 golden rows, 10-31 | **rebuilt** | fb_bow_maps' estimator unchanged on pass-1 tables (`bow_pass1.py`), v(t) arm; pass arm `bow_profiles_pass.pdf`. The audit render stays in `fb_bow_maps/figures/` |
| bow_closure | `rerun/paper1_res/figures/bow_closure.pdf` (current) | pass 1 golden rows, 10-31 | **rebuilt** | as above |
| det4_stripes | `papers/paper1/figures/det4_stripes.pdf` (current) | pass 1 bench g_det4 + SPS final telescope tables, 10-31 | **rebuilt** | X-view efficiency against x, bench (M3) and SPS run63/run53 (telescope): stripe positions agree on 82–85 % of 1 mm bins (`rerun/paper1_figs/det4_stripes.py`) |
| loss_band … loss_multi_deposit (14) | `papers/paper1/figures/loss_*.pdf` (current) | pass 1 sat_det3 table + rr1 waveforms, 10-31 | **rebuilt** | `rerun/paper1_figs/a1_displays.py`. Each class keeps the 10-26 deck event where its pass-1 class is unchanged (8 of 11); band, merged and the four classes without a deck event were re-picked by eye. The deck's band and head-on events are now *detected* on pass 1; loss_headon now shows the one fit_failed event (the head-on class is 0.02 %) |

## Paper II

| figure | source (state) | basis, date | status | note |
|---|---|---|---|---|
| sharing_cartoon | engineer_package 03 (current) | schematic, 07-12 | current | manifest asks to check it shows c2 < c1 |
| kernel_forms | `rerun/paper1_spsfigs/figures/kernel_forms.pdf` (current) | run71 RAW on the final cache builder (kernel_job.py, condor 4605384), 11-01 | **rebuilt** | ±1..3 responses and time forms, X/Y, 96/156/243 V/cm; supersedes the old Y c2/c1 0.42–0.47 |
| angled_kernel | `rerun/paper1_spsfigs/figures/angled_kernel.pdf` (current) | run63 flat700 vs run63_rot25 d425/d325 (not run57) on the final builder, 11-01 | **rebuilt** | the August kernel_ab.png read run63's own A/B from staging npz no longer on disk; remade on the pass cache |
| ladder_compression | none (placeholder) | waveform_first_threading 05 (hits vs M3) | placeholder | pass-independent study; never produced |
| event_waveforms | stephan_cosmics display (current) | raw waveforms, 07-12 | current | manifest asks to check there is no hits overlay |
| forward_model | none (placeholder) | diagram | placeholder | |
| toy_floor | none (placeholder) | script 18 on the R&D model | placeholder | needs a re-run on the production model (R6) |
| tiers_benchmark | none (placeholder) | R1–R5 | placeholder | H0 vs W on the same events now exists (`rerun/paper1_floor/hits_arm.py`, the fleet figure); the H1/H2a/H2b/H2c tiers are not pass products and H2c (forward fit on hit lists, R3) is new work, not a re-run: left open (10-31) |
| threshold_neighbours | none (placeholder) | R4 | placeholder | |
| known_angle_geometry | angled_kernel geometry.png (current; PNG not on this machine) | schematic | current | |
| known_angle | `rerun/paper1_spsfigs/figures/known_angle.pdf` (current) | final SPS pass tables + column-duration v (vbox.py, 19 new rows), 11-01 | **rebuilt** | 25.4° mount: 0.981 ± 0.03 of the stage tan with the column v (13 rows), 0.927 with the registry v |
| fleet | `rerun/paper1_floor/figures/fleet.pdf` (current) | pass 1 + rr1 hits chain, 10-31 | **rebuilt** | H0 (rr1 event_results_veto50, angle scale calibrated on M3 per row) vs W (pass-1 v(t) arm) on the same interior_good events of the five calibration rows: angle s68, position core σ, efficiency within 5 mm (`rerun/paper1_floor/hits_arm.py`) |

## Summary

| status | count | figures |
|---|---|---|
| current | 5 | microtpc_principle, bench_setup, sharing_cartoon, event_waveforms, known_angle_geometry |
| rebuilt on pass 1 | 31 | attachment, sps_resolution, kernel_forms, angled_kernel, known_angle (11-01, `rerun/paper1_spsfigs/`); timing (10-31, `rerun/paper1_timing/`); charge_balance; gas_vE, tracking_perf, bow_profiles, bow_closure (10-31, `rerun/paper1_res/`); fleet (10-31, `rerun/paper1_floor/`); hv_scan, spark_deadtime (10-31, `rerun/paper1_ops/`); efficiency_map, edge_turnon, det4_stripes, the 14 loss_* displays (10-31, `rerun/paper1_figs/`) |
| stale | 0 | |
| pending-vt | 0 | (v(t) settled 10-31) |
| pending-sps | 0 | (SPS final 11-01) |
| superseded | 0 | |
| placeholder | 8 | chamber_stack, readout_layout, sps_setup, ladder_compression, forward_model, toy_floor, threshold_neighbours, tiers_benchmark |

New pass-1 figures available but not in either manifest (no `.tex` references them yet):
`rerun/paper1_charge/figures/{gain_curves, drift_field, unsat_bias, golden_spectra}.pdf`.
