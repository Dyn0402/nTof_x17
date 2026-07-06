# Micro-TPC analysis runbook (scripts 13–30)

*Operational guide: everything needed to re-run, extend, or audit the
micro-TPC / drift-velocity / gas / unsharing chain without reading the
history. Physics narrative: `report_det3_weekend/main.pdf` (rev 4).
Results digest: `DET3_WEEKEND_ANALYSIS.md`. Written 7-06.*

## 0. TL;DR of what this chain established

| quantity | value | script that owns it |
|---|---|---|
| drift velocity det3 (1000 V, 6-27) | **34 ± 1.5 µm/ns** | 21/23 (geometry), confirmed by 26/27 (unshared time fit) |
| why time-fits are 10–20 % low | ~50 % prompt charge sharing between strips distorts the strip-time ladder | 22, 24, 25 (toy closure), diagram in report |
| sharing constants (design property) | c1 ≈ 0.45–0.52, c2 ≈ 0.05–0.15 (x/y), +69 ns delay | 26 (measured from vertical tracks) |
| final micro-TPC angles | 0.19° plateau bias, 1.9° resolution (68 %) | 26→27→28 chain |
| gap / recorded column | 30 mm mechanical; ~23 mm recorded (attachment+threshold) | 17, 21 |
| gas | Ar/iso 95/5 + ~1 % H2O + ~1 % air; det3 was >3 % H2O before 6-24 (dried) | 15, 30 + `garfield_sim/` tables |
| field conversion | E = HV / 3.0 cm; drift HV varies per run/slot — decode from hv_monitor.csv | 30 (`DRIFT_HV` map) |

**Data is frozen** (campaign over, no re-runs, no dry-gas scan). Decoded
waveforms exist locally for det3 6-27 long run and det2 6-22 longer_run;
all other runs' `decoded_root` are on lxplus:
`~dneff/x17/cosmic_bench/june_tests/<run>/<subrun>/decoded_root/`
(`ssh lxplus`, user dneff; rsync only the `*_<FEU>.root` you need, ~260 MB each).

## 1. Environment

- **Analysis python**: `nTof_x17/.venv/bin/python` (pandas, uproot, scipy,
  matplotlib). NOT `../venv` (stale docs mention it).
- Run all scripts FROM `mx_june_cosmic_qa/`:
  `../.venv/bin/python <script>.py <run_key> [--veto=50]`.
- **Magboltz/Garfield**: system `python3` locally (env vars global), or on
  lxplus: `source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh`
  then `ROOT.gSystem.Load('libGarfield')` (there is no python `Garfield`
  module there — see `garfield_sim/mm_water_grid_lxplus.py` for the pattern).
- LaTeX report: `cd report_det3_weekend && pdflatex main.tex` (×2);
  schematic figures regenerate via `../../.venv/bin/python mk_diagrams.py`.

## 2. Data layout (per run)

```
/home/dylan/x17/cosmic_bench/<bench>/<run>/<subrun>/
    combined_hits_root/   per-strip pulse scalars (eventId, feu, channel,
                          amplitude, time[ns]=sample*60, left/right_sample,
                          time_of_max, ...) -- NO waveforms, NO CNS
    decoded_root/         full waveforms: per event vectors sample-major ->
                          reshape(32,512) = [sample, channel]; 12-bit ADC,
                          baseline ~220; eventIds continuous across files
    m3_tracking_root/     M3 reference rays (see M3 note in Sec. 7)
    hv_monitor.csv        channel map: 0:7=top drift, 0:6=bottom drift,
                          3:4/3:3=top/bottom resist, 0:8-0:11=M3 @500 V
/home/dylan/x17/cosmic_bench/Analysis/<run>/<subrun>/<det>/
    cache/event_results.pkl          un-vetoed per-event micro-TPC results
    cache/event_results_veto50.pkl   veto: <=50 hit-rows/event (sparks out)
    alignment_tpc_veto50/alignment.json
```

Run keys live in `qa_config.py` (`RUNS` dict). For runs without a key,
`30_fleet_gas_survey.py` shows how to construct `_Config` directly with
FEU auto-detection (`autodetect_feus`) — FEU↔detector mapping changes
between runs, never assume.

## 3. Prerequisites before any script 13+ works on a run

1. `combined_hits_root` + `m3_tracking_root` present (else see
   `HANDOFF_recofar_and_m3_tracking.md` / memory notes for
   producing them — M3 producer now runs locally, see Sec. 7).
2. `03_alignment_and_tpc.py <key> --veto=50` → produces the veto50 cache
   and `alignment_tpc_veto50/alignment.json`. Everything downstream loads
   these. (Alignment gotchas: θ converges near 89–90°, use `--flipy` era
   notes in memory if starting from scratch on a new setup.)

## 4. The pipeline, in dependency order

| # | script | needs | produces / measures |
|---|---|---|---|
| 13 | `13_tpc_angle_bias.py` | cache+align | off-diagonal diagnosis: bias profiles, cluster geometry, w/z ratio |
| 14 | `14_drift_velocity_scan.py` | caches for all drift points | ridge-fit v per drift HV (KNOWN 10–20 % LOW — kept as reference) → `drift_velocity_scan.csv` |
| 20 | `20_ridge_systematics.py` | cache+align | ridge convexity (split windows), per-plane extent slopes, floors |
| 21 | `21_geometry_vdrift_scan.py` | 14's csv + caches | **v_geom(E)** = extent-slope/T_sat per drift point → `geometry_vdrift_scan.csv` (the physics velocity) |
| 23 | `23_core_geometry_vdrift.py` | caches | core-only geometry v (systematics check on 21) |
| 15 | `15_drift_velocity_vs_magboltz.py [--gap=3.0]` | 21's csv + `garfield_sim/results/*.json` | gas ranking (RMS per mixture) |
| 17 | `17_gap_attachment_test.py` | caches + hits, prefers 21's csv for the depth scale | arrival-depth distributions, λ_att per HV → `gap_attachment_test.csv` |
| 18/19 | attachment closure & money plots | 17's csv | `attachment_vs_magboltz.png`, `amplitude_attachment.png` |
| 16 | `16_gap_from_pulse_width.py` | cache + hits | single-strip ToT cross-check |
| 22 | `22_strip_timing_and_estimators.py` | cache + hits | pulse-bar displays, skirt residuals, hits-level estimator shoot-out |
| 24 | `24_waveform_investigation.py` | **decoded_root** | waveform-level re-timing (4 estimators), shape systematics; caches `wf_strip_times.csv` |
| 25 | `25_signal_formation_toy.py` | nothing (standalone MC) | estimator-bias closure; `--unshare` validates the correction; knobs `--vtrue --lam --xt1 --xt2` |
| 26 | `26_unsharing_analysis.py` | decoded_root + cache | **measures c1/c2/Δt from vertical tracks**; unshare → before/after ridge v |
| 27 | `27_unsharing_refinement.py` | decoded_root + cache | kernel α scan (prompt vs delayed), before/after ladders, angle bias/resolution |
| 28 | `28_angle_calibration.py` | decoded_root + cache | additive tan-space calibration constants b per plane → final angles |
| 29 | `29_det2_validation.py` | det2 decoded+cache | the whole chain on another detector (template for any detector) |
| 30 | `30_fleet_gas_survey.py` | all cached runs | v_geom + λ_att + implied H2O % per detector per date |

Typical invocation: `../.venv/bin/python 21_geometry_vdrift_scan.py sat_det3 --veto=50`.
Runtimes: hits-level scripts ~1–5 min; waveform scripts (24/26/27/28) ~5–15 min
(they stream decoded files in 400-event chunks; ~1 GB RAM when buffering).

## 5. Constants you must know (and where they come from)

- `CSHARE = {7: (0.449, 0.052), 8: (0.516, 0.151)}` in 27/28 — **det3's**
  measured sharing. For another detector, measure with 26 first (vertical
  tracks; needs ≥ a few hundred leads) — do NOT reuse blindly (though det2
  came out nearly identical: design property).
- Kernel `ALPHA = 0.5` (prompt fraction; result robust ±0.5 µm/ns over 0–1).
- Calibration `b_x = +0.033, b_y = +0.029` (28; det3 6-27, post-unsharing).
  Gas-quality dependent — remeasure per run era.
- Thresholds: hits `THR_HIT = 100` ADC, waveform timing `THR_WF = 150`,
  `CORE_FRAC = 0.30`, `MIN_STRIPS = 4` before unsharing, **3 after**
  (unshared clusters shrink to the direct footprint).
- Ridge/regression windows: `0.06 < |tanθ| < 0.55`; extent-slope profile
  bins 0.06–0.44; T_sat = median span for |θ_ref| > 10°.
- λ_att fit window: z ∈ [8, 22] mm on the v_geom depth scale.
- Geometry: pitch 0.78 mm, gap 30 mm, E = HV/3.0 cm, DREAM 32×60 ns.
- Detector-frame math (everywhere): with alignment angle θ,
  `t_x_det = cosθ·tan_rx + sinθ·tan_ry`, `t_y_det = −sinθ·tan_rx + cosθ·tan_ry`;
  the M3 x angles get `ref_x_sign` applied on load. Physical plane X pairs
  `s_x` with `t_x_det`.

## 6. How to run the chain on a NEW run/detector (recipe)

```bash
# 0) key in qa_config.py (or _Config + autodetect_feus, see 30)
# 1) alignment + veto50 cache
../.venv/bin/python 03_alignment_and_tpc.py <key> --veto=50
# 2) velocity + gas observables (hits only)
../.venv/bin/python 20_ridge_systematics.py <key> --veto=50      # sanity
#    (single point: use 29's geometry block; scans: 14 then 21)
# 3) drift HV: read <run>/<subrun>/hv_monitor.csv, decode with the channel
#    map above; NEVER assume 1000 V (6-23: 600 V, 6-25: 500 V?, 6-26: 700 V)
# 4) waveform chain (needs decoded_root; pull from lxplus if not local)
../.venv/bin/python 26_unsharing_analysis.py <key> --veto=50     # measures c1/c2
#    then edit CSHARE in 27/28 for this detector and run them
# 5) compare v_geom vs unshared time-fit: they must agree (~3 %);
#    disagreement means something new (fit windows, FEU map, HV, gas).
```

## 7. Cross-cutting gotchas (each cost real time once)

- **Charge sharing ~50 %**: never extract velocities/angles from raw strip
  times without unsharing; never interpret cluster width as geometric.
- **Common-mode noise**: production hits have NO CNS; FEU 6/8 raw σ~115.
  All waveform work must do pedestal (per channel median over ~300 events)
  + per-chip (64-ch) per-sample median subtraction (pattern in 24).
- **eventIds** are continuous across datrun files (verified) — safe to
  concatenate hits and match by eventId. Small gaps (missing ids) exist.
- **M3 tracking (updated 7-05/06 by follow-up sessions)**: the producer now
  runs locally and bit-reproduces the DAQ; a drop-layer fix adds ~25 % rays;
  the recommended v2 recipe is NClusX/Y ≥ 3 & chi2 < 5 (see
  `HANDOFF_m3_v2_reprocessing.md`). **Scripts 13–30 results in the report
  were produced pre-v2 with `chi2_cut=20`** — rerunning on v2 rays will
  shift samples slightly (expected: more/cleaner rays, same physics). The
  714-vs-702 mm z puzzle is resolved as an origin-offset convention; M3
  geometry is sound.
- **Sparks**: the efficiency breakdown (09) historically counted spark
  centroids inside reco_near/reco_far; separated since the 7-05 spark
  deep-dive (see memory/`june-spark-and-recofar-analysis`). The veto50
  cache excludes >50-row events from micro-TPC fits, but sub-veto
  discharges (30–50 strips) survive — relevant for tails.
- **det3's gas changed during the week** (>3 % H2O before ~6-24, ~1 %
  after): never average det3 quantities across those dates; the 6-23
  bottom-slot det3 point is broken (v_geom ≈ 1, excluded).
- Magboltz: velocity converges at ncoll=2; **attachment needs ncoll=5**
  (rare process). Killing a spawn-Pool Magboltz parent orphans workers —
  kill worker PIDs too. Existing tables: `garfield_sim/results/`
  (`water_grid.json`, `attachment_*.json`, `townsend_compare.json`,
  `drift_velocity_*.json`).
- Toy MC (25) forbids nothing but note the closure config that matches
  data: `--xt1=0.35 --xt2=0.12 --lam=10` (effective hits-level sharing).

## 8. Documents map

- `report_det3_weekend/main.pdf` — full physics write-up (29 pp): includes
  the estimator table, before/after conclusions, all figures.
- `DET3_WEEKEND_ANALYSIS.md` — results digest + tables.
- `HANDOFF_recofar_and_m3_tracking.md` — spark/reco_far + M3 audit threads
  (both since picked up; see the newer handoffs and memory).
- This file — how to run it all.
