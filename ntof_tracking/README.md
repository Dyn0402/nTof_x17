# nTOF track reconstruction — bench-informed, telescope-free

**Mission: reconstruct true 3D tracks (ultimately e⁺e⁻ pairs from the target) in
the July 2026 nTOF beam data, where there is NO reference telescope, by
transferring the June cosmic-bench characterization of these exact chambers.**

This directory contains (a) a standalone reconstruction library distilled from
the bench analysis, (b) frozen per-detector "factory calibration" models, (c) a
completed transfer validation proving the day-1 strategy on bench data, and
(d) the numbered plan documents (TRACK_PLAN_01…07) that take another Claude
model from raw beam files to physics-ready track tables. **Read this README
fully, then execute the plans in order.** Each plan is self-contained.

> **CORRECTION (2026-07-12): there are NO cosmics at nTOF.** The plans as
> originally written assume a cosmics-at-nTOF commissioning dataset (Jun 27–29)
> and beam-off cosmic runs. **These do not exist.** The ONLY tracks available at
> nTOF are the beam tracks we reconstruct ourselves. Consequences, applied
> throughout PLAN_02/04/05 below:
> - **No external validation of the frozen bench angle models.** "Commission on
>   cosmics where the angular mix matches the bench" is impossible. The frozen_rs
>   angles cannot be checked against a known-good cosmic sample — trust is
>   established only by internal consistency (X-plane vs Y-plane, micro-TPC
>   segment angle vs inter-chamber link angle).
> - **All calibration is bootstrapped from self-found beam tracks.** The
>   inter-chamber links of PLAN_04 are the sole truth source (replacing both M3
>   AND the assumed cosmic sample). The loop must converge from beam tracks alone.
> - **z-alignment cannot lean on cosmics for inclined tracks.** It must use the
>   inclined beam tracks we find. First real nTOF micro-TPC tracks (run_30
>   scintOff, detector A) are inclined ~8–15° (evt 1017: θx −7.7°/θy −14.5°;
>   evt 162: −13.5°/−13.3°), so inclined beam tracks DO exist — but they are rare
>   (~3 % of compact clusters; most nTOF hits are isochronous point-like deposits,
>   fundamentally unlike a gap-crossing cosmic — see
>   `ntof_july_analysis/run30_microtpc.py`).
> - **Opposite arms never share a track** (no cosmics crossing everything), so
>   global cross-arm alignment has no track-based handle — it needs survey/beam-spot.

## What already exists here

### `reco/` — beam reconstruction package (NEW, 2026-07-17, run_48 era)

First working implementation of the PLAN_01/02/03 chain, built against the
run_48 scint-DOUBLES data (32 smp × 60 ns windows, Ar/iso 95/5, dr800):

| module | what |
|---|---|
| `reco/io.py` | full-schema combined_hits loader + vectorized (feu,channel)→(det,plane,pos_mm) mapping (centred local coords) |
| `reco/noise.py` | residual-noise model: coherent time-band finder (the ~1.3 MHz whole-plane oscillation → 50-400 strips in one ~30 ns slice; amp-rescue for track hits crossing a band), isolated-hit removal, hot-channel list |
| `reco/segments.py` | per-plane spatio-temporal clustering (Chebyshev link 4 mm/250 ns) + guided fragment merge (union robust-fit must stay track-grade) + robust line fits + cluster taxonomy {track, point, band_fragment, blob} with a strip-occupancy guard |
| `reco/pairing.py` | X/Y pairing (time IoU + PLAN_38 charge balance via microtpc_lib) → 3D local segments; drift-time → depth via DriftModel |
| `reco/geometry.py` | global frame from run_config det_center_coords/det_orientation (verified = Geant strip-plane centres incl. pinwheel); ACTIVE volumes only (MM drift gas, 16 instrumented SiPM bars per DetectorConstruction.cc, 2 PVT bars, LS liquid, He-3 gas polycone); local→global transforms; per-volume line-crossing tests (→ trigger-bar predictions); bench drift-velocity curve. NB: Geant `plot_geometry.py` has the SiPM read-out window shift sign FLIPPED (+u, bars 3–18) vs the sim itself (−u toward the MM, bars 1–16, DetectorConstruction.cc:489-500) — sim taken as truth; fix the plot script upstream. |
| `reco/display.py` | per-plane event displays with reco overlays + 3-view global extrapolation figure (active volumes; measured segment drawn only in its own arm's panels) |
| `reco/search.py` | event sifting: reconstruction-evidence score (3D pair ≫ 1-plane track ≫ nothing) + physics priors (2026-07-17): beamline-pointing bonus (DCA to Y axis < 400 mm, + He-3-gas crossing bonus), ≥3-chamber pair events down-ranked as pile-up (`multi_det`), burst veto (≥3 dets with >120 clean strips). Priors bias ranking only — odd-angle tracks stay findable. |

Driver: `ntof_july_analysis/run48_tracking.py` (`event N` / `search`).
First results (run_48 `_00`, 1047 events, 2026-07-17): evt 1107 → clean 3D
track in mx17_C (x: n=28, r²=0.98; y: n=23, r²=0.98; IoU 0.95), extrapolation
crosses D's SiPM wall + plastics (= the C×D doubles trigger), dca(origin)
≈ 250 mm. Full sift: 42 events with a 3D X/Y pair (~4 %), 379 more with a
single-plane track segment, 104 prompt multi-detector bursts vetoed
(BUSY_DET_STRIPS=120 clean strips, ≥3 dets); evt 1107 ranks #1.
Outputs → `analysis/July_HV_Scan/run48_tracking/<subrun>/` (candidates.csv
+ per-event planes/global figures).
Full-run sift (all 12 subruns, 2026-07-17): 457 3D pairs after the multi-det
cut; `run48_tracking.py ensemble` pools them, keeps DCA(target/origin)
< 75 mm → **97 pairs (A 57 / B 11 / C 20 / D 9)** drawn together on the
3-view global figure and a matplotlib 3D model of the active Geant4 volumes
(`display.plot_global_ensemble` / `plot_global_3d`) →
`run48_tracking/ensemble_dca75/`. `beamproj` mode gives the source profile
along the beamline (median beam_y ≈ −25 mm, σ68 ~ few 100 mm).
**Open calibrations:** DAQ t0 (≈450 ns first estimate), in-plane strip-axis
signs/offsets (survey-unverified — pointing conclusions provisional until
PLAN_04 links pin them).

### bench-transfer layer (June cosmic distillation)

| file | what |
|---|---|
| `microtpc_lib.py` | numpy/pandas-only library: gap clustering + anchored time fit, 6-feature signature extraction, \|tanθ\| regression train/apply (+ in-situ restandardization), Fisher sign model, hybrid rule, extent-slope/T_sat velocity, X/Y pairing (time-IoU + charge balance), model I/O. Bit-faithful to bench scripts 21/34/35/36. |
| `bench_constants.py` | every transferable bench number, with per-block TRANSFERABILITY notes (what carries over vs what must be remeasured). |
| `models/mx17_{3,2,6,7}_hits6.json` | frozen hits-level angle models per detector (trained on M3 truth, even-eid; holdout metrics + provenance inside). det3 = fleet default. |
| `validate_transfer_bench.py` | the dress rehearsal (already run — results below). Rerun after any lib change. |
| `validation/transfer_validation.{csv,png}` | its outputs. |

## The proven day-1 strategy (validation results, 2026-07-12)

Frozen det3 model, **mu/sd restandardized on the target data** (`frozen_rs` —
unsupervised, no truth needed), scored against M3 on the odd-eid holdout,
regression band 3.4–24°:

| condition | σ68 frozen_rs (x/y) | σ68 self-trained | v_sig vs v_geom |
|---|---|---|---|
| det3 @1000 V (train) | 1.73/1.78° | 1.74/1.76° | +4.1 % |
| det3 @700 V | 1.82/1.88° | 1.84/1.96° | −0.6 % |
| det3 @900 V | 1.70/1.82° | 1.80/1.90° | −0.4 % |
| det3 @1100 V | 1.87/1.70° | 1.87/1.82° | −0.4 % |
| det2 @1000 V | 1.97/2.36° | 1.94/2.19° | +3.9 % |
| det6 @700 V | 2.56/3.28° | 2.59/2.91° | +17 % (weak regr.) |
| det7 @700 V | 3.73/3.14° | 3.15/2.93° | +18 % (weak regr.) |

Low-angle (|θ|<5°, signed with the frozen Fisher sign model): det3 1.7–1.8°,
det2 2.3/2.9°, det6/7 3.3–4.3°; sign accuracy 85–98 %.

**Conclusions that shape everything downstream:**
1. **Restandardization ≈ retraining** for condition changes on the same
   detector (35's "per-point training required" was mostly feature-scale
   shift). Day-1 beam angles: frozen per-detector model + restandardize.
2. Pure-frozen (no restandardization) develops multi-degree biases
   off-condition — never ship it.
3. det6/det7 gain ~15 % from self-training → retrain them first once in-situ
   truth exists (TRACK_PLAN_04/05).
4. The telescope-free velocity (v_sig) is good to <1 % where the regression is
   strong and the window is not truncated — it is the in-situ gas monitor.
5. Caveat carried into PLAN_02: restandardization assumes the target angular
   distribution is broadly similar; beam tracks (from a target) differ from
   cosmics. Treat frozen_rs as bootstrap; converge via PLAN_04→05 truth.

## Architecture (execution order)

```
PLAN_01  beam data interface: run configs, schemas, flash ID, dead strips,
         thresholds, event taxonomy                        [foundation]
PLAN_02  per-plane, per-window segment reco: pattern recognition
         (road-following) + bench-grade features/angles     [core]
PLAN_03  X/Y pairing (time IoU + charge balance) → 3D segments per chamber
         + the output table schema                          [core]
PLAN_04  inter-chamber alignment + track linking + vertexing — this CREATES
         the truth source that replaces M3                  [key enabler]
PLAN_05  in-situ calibration loop: v_drift, model retraining, f_balance,
         T_sat, monitoring                                  [closes the loop]
PLAN_06  DAQ-side waveform upgrades on daq_lxplus: unsharing, early-charge
         centroid, n_u, walk                                [precision tier]
PLAN_07  timing & neutron ToF: gamma-flash anchor, per-track interaction
         time, E_n                                          [physics tier]
```
01→02→03 are strictly sequential. 04 needs 03's segments from ≥2 chambers.
05 needs 04. 06 and 07 can run in parallel with 04/05 once 03 exists.

## Shared conventions (read once)

- **Machines.** Laptop: repo `/home/dylan/PycharmProjects/nTof_x17`, python
  `.venv/bin/python` (pandas/uproot live ONLY in the venv). DAQ machine:
  `ssh daq_lxplus` (user mx17, host mx17-daq via lxplus jump; there is also
  `ssh daq` direct when on-site). DAQ repo clone:
  `~/PycharmProjects/nTof_x17` with its own working `.venv`; DAQ data:
  `~/beam_july/runs/<run>/<subrun>/` (July analysis README references
  `/mnt/data/x17/beam_july/` — verify they are the same volume before writing
  anything). DAQ has 6 cores, ~98 GB free. The DAQ clone has UNPUSHED
  ntof_july_analysis commits — `git pull/push` there before assuming states
  match.
- **July detector fleet = the bench-characterized chambers.**
  mx17_A/B/C/D = det3/det2/det6/det7 (bench letters; `det_labels.py`), July
  FEUs A:[3,4] B:[5,6] C:[7,8] D:[1,2]. Which FEU is X vs Y: read
  `run_config.json detectors[].dream_feus` or autodetect from the strip map
  (`autodetect_feus` pattern, `30_fleet_gas_survey.py:83`). NEVER assume.
- **Beam DAQ differs from bench**: sample period **20 ns** (bench 60), window
  **300–400 samples ≈ 6–8 µs** (bench 32 ≈ 1.9 µs). Both are per-run knobs in
  `run_config.json` (`sample_period`, `n_samples_per_waveform`) — read them,
  never hardcode. Hit `time` is always already ns.
- **combined_hits schema at beam** (verified on run_8): `eventId,
  trigger_timestamp_ns, channel, amplitude, time, time_of_max, sample,
  max_sample, local_baseline, local_max, left_sample, right_sample,
  time_over_threshold, integral, saturated, feu`. `amplitude` is
  **saturation-corrected** (can exceed 4095; `saturated` flags it) — PLAN_38
  bench finding, unchanged at beam. Everything the hits6 feature set needs is
  present.
- **decoded_root schema at beam**: tree `nt`, per-event jagged `eventId,
  timestamp, ftst, sample, channel, amplitude` (run_8: 204800 = 400 samples ×
  512 channels). `ftst` (10 ns steps) is applied to hit `time` upstream.
- **Pedestal contamination gotcha** (July README): every subrun's raw dir gets
  a copy of the shared pedestal acquisition whose filename contains
  `_datrun_`; all loaders must exclude `_pedestals_` files.
- **Amplitude thresholds**: bench used THR_HIT=100 ADC; July quick-look QA
  used 400 against beam noise. PLAN_01 re-derives the threshold per run from
  the pedestal/noise data; the lib takes it as a parameter.
- **The gas is NOT the bench gas.** May ran Ar/CF4/iso and others; candidate
  July gases have Garfield tables in `garfield_sim/results/*_CERN_450m.json`
  (Ne/iC4H10, He/C2H6, Ar/iC4H10). v_drift, T_sat, recorded column,
  attachment: all must be re-established in situ (PLAN_05). Sharing constants
  c1/c2, the feature/regression machinery, charge balance width, spark
  behaviour: transfer (design properties).
- **Bench sources of record** if a plan references a bench result:
  `mx_june_cosmic_qa/{PAPER_STATUS.md, DET3_WEEKEND_ANALYSIS.md,
  MICROTPC_RUNBOOK.md}`.
- **When done with a plan**: append a dated RESULTS section to the plan file
  (numbers + output paths), same convention as `mx_june_cosmic_qa/paper_plans/`.
