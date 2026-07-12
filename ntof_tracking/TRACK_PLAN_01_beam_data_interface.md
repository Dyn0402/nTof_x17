# TRACK_PLAN_01 — beam data interface & event taxonomy

**Foundation for everything. Deliverable: `ntof_tracking/beam_io.py` + a
per-run QA figure, such that PLAN_02 can ask for "the preprocessed hits of
window (run, subrun, eventId)" and trust what it gets.**

## Inputs

- Runs: `/media/dylan/data/x17/beam_july/runs/<run>/<subrun>/` (laptop, if
  synced) or `~/beam_july/runs/...` on daq_lxplus. Each subrun has
  `combined_hits_root/`, `decoded_root/`, and the run dir has
  `run_config.json`.
- Strip map: repo `mx17_m1_map.csv` + `common/Mx17StripMap.py`'s
  `RunConfig(run_config_path, map_csv).get_detector(name)` → `det.map_hit(feu,
  ch)` → `(x_mm, y_mm)` (exactly one finite). This is the same machinery the
  bench used; July detector names come from the config's `included_detectors`.

## Steps

1. **Run-config parser.** From `run_config.json` extract per run:
   `sample_period` (ns), `n_samples_per_waveform`, per-detector
   `dream_feus` (FEU↔plane map), per-subrun `sub_runs[].hvs` decoded through
   `detectors[].hv_channels` (July: resist = card 5 ch 1–4 for A–D, drift =
   card 9 ch 0–3 — but READ it from the config, don't trust this note), gas
   fields if present, target, trigger. Return a plain dict; cache as JSON next
   to the analysis output. Sanity: run_8 gives sample_period=20,
   window = n_samples×20 ns ≈ 8 µs.
2. **Hits loader.** Load `combined_hits_root/*_datrun_*feu-combined_hits.root`
   tree `hits` (full verified schema in README), EXCLUDING `_pedestals_`
   files and files without a `hits` tree (live-run truncation). Attach
   per-hit `pos_mm` and `plane` via the strip map. Keep `saturated`,
   `integral`, `time_over_threshold` — the feature set needs them.
3. **Noise floor & threshold per run.** Use the pedestal acquisition (in
   `<run>/pedestals/` or the per-subrun copies) or, lacking that, the
   amplitude spectrum of off-window hits: fit the noise shoulder, set
   `THR_HIT_BEAM = ceil(5σ_noise)` per FEU (expect O(100–400) ADC; July QA
   used 400). Record per run; the lib takes it as a parameter everywhere.
4. **Dead/hot strip maps** per (run, FEU): rate per channel over many events;
   dead = <10 % of plane median (existing `compute_dead_strips` convention in
   `beam_track_finding.py`), hot = >10× median (mask, they fake clusters).
   Persist as JSON; PLAN_02's road-following is dead-strip aware.
5. **Event taxonomy.** Classify every eventId of a subrun:
   - **gamma flash**: existing rule (`ntof_may_analysis/
     dream_timber_time_sync_flash.py`): any FEU with >200 hits of amp>500 and
     time<1000 ns. One per beam pulse; keep (it is the ToF anchor, PLAN_07)
     but exclude from tracking.
   - **discharge/spark**: >50 strips on ONE detector outside the flash
     signature. Bench PLAN_39/40 established sparks are global common-mode
     steps with NO after-effects — veto the window, trust the neighbours.
   - **empty/noise**: no hit above threshold.
   - **track candidate**: the rest. Report the funnel per subrun (the May
     `beam_qa.plot_track_candidate_summary` pattern).
6. **Window time structure QA** per subrun: hit-time histogram with the July
   window edges [0, 800, 1150, 3500, 7000, 10000] ns overlaid (flash /
   mid-window turn-off structure); occupancy per plane; hits-per-candidate
   distribution. One PNG per subrun.

## Acceptance checks

- run_8 subrun loads; hits count matches `uproot` row count; every kept hit
  maps to a finite position on exactly one plane.
- Flash events found at ~1 per beam pulse (cross-check against
  `trigger_timestamp_ns` spacing ~ PS cycle).
- Threshold: hit-rate per event above threshold is stable across subruns of
  the same HV (if not, the noise floor moved — investigate before continuing).

## Gotchas

- `time = sample × sample_period` is already ns — never rescale by 60.
- `trigger_timestamp_ns` granularity must be VERIFIED (histogram the diffs)
  before PLAN_07 leans on it.
- ZS runs: absent samples are suppressed, not zero — occupancy per channel is
  biased low for small pulses; dead-strip detection must use rate RELATIVE to
  neighbours at the same threshold.
- FEU numbering collides across campaigns (bench det3 was FEU 7/8; July det C
  is FEU 7/8) — never reuse bench FEU-keyed constants without going through
  the detector identity (bench_constants.DETECTORS).
