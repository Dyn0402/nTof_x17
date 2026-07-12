# TRACK_PLAN_06 — waveform-level upgrades on the DAQ machine

**The precision tier: unsharing, early-charge centroid positions, the n_u
feature, and walk corrections all need decoded waveforms, which are big —
this work runs ON daq_lxplus next to the data. Deliverable:
`ntof_tracking/daq_waveforms.py` (post-processor of decoded_root for
candidate windows) + upgraded segment columns.**

## Environment (verified 2026-07-12)

- `ssh daq_lxplus` → mx17@mx17-daq (ProxyJump lxplus; `ssh daq` direct
  on-site). 6 cores, ~98 GB free on /. Python 3.12 system;
  **repo clone `~/PycharmProjects/nTof_x17` with a working `.venv`
  (uproot/pandas confirmed)**. Data: `~/beam_july/runs/<run>/<subrun>/
  {combined_hits_root,decoded_root}/`; also `~/beam_july/{pedestals,
  analysis,dream_config}`. NB the DAQ repo clone has UNPUSHED commits
  (ntof_july_analysis) — sync git both ways before working there.
- The C++ production chain lives in `~/CLionProjects/mm_strip_reconstruction`
  (decoder / feu_hit_combiner / clusterizer / waveform_analysis /
  orchestrator / qa_waveforms). `WaveformAnalyzer.cpp` is where the 30 %
  constant-fraction hit `time` and the ftst correction are applied
  (bench-verified, PLAN_42). **Phase 1 below is pure-python post-processing;
  do NOT modify the C++ chain until the algorithms are frozen** — then port
  (the unsharing pre-filter slot is between decode and hit-building).

## Decoded data at beam

Tree `nt`: per-event jagged `eventId, timestamp, ftst, sample, channel,
amplitude`; run_8 shows 204800 values/event = **400 samples × 512 channels**
(8 µs at 20 ns). VERIFY the (sample, channel) ordering by reshaping and
checking that a known hit's channel/sample from combined_hits lines up —
bench was sample-major; do not assume.

## Phase 1 tasks (python, candidate windows only)

Process ONLY windows PLAN_02 marked as track candidates (decimation is what
makes 6 cores enough: sparse candidate rate × 400×512 arrays).

1. **Pedestal + common-mode.** Per channel pedestal from the run's pedestal
   acquisition (median per channel); per-chip common-mode = median over each
   64-channel group per sample (the bench 24-pattern). CM subtraction is
   NON-OPTIONAL at beam: bench 40_spark_waveforms showed discharges are
   global CM steps, and the flash is worse.
2. **Unsharing** (`microtpc_lib` has no waveform code — implement here,
   porting `27_unsharing_refinement.unshare` exactly): banded solve
   (I + α·c1·E1 + α·c2·E2)X[:,s] = W[:,s] − (1−α)(c1·E1·X[:,s−d1] +
   c2·E2·X[:,s−d2]), α=0.5, (c1,c2) per detector/plane from
   bench_constants.CSHARE. **The delayed-kernel lag is a TIME, not a sample
   count**: bench +69 ns ≈ 1 sample at 60 ns → at 20 ns use d1≈3–4 samples.
   FIRST remeasure the neighbour delay from beam near-vertical tracks
   (the 26 recipe: lead ≥500 ADC, neighbour ratios + CFD Δt) — c1/c2 should
   reproduce ~0.45/0.05 (x) 0.52/0.15 (y); the delay in ns should match ~69.
   That single measurement validates the whole transfer.
3. **Unshared re-timing + n_u** → per-plane unshared segment slope S_seg
   (26/27's core-OLS ladder: THR_WF=150, CORE_FRAC=0.30, min 3 core strips)
   and the unshared footprint n_u → upgrade PLAN_02 segments to the FULL
   hybrid (34: time fit above ~5°, regression below; bench gain: plateau
   2.1°→1.86°) and to the 7-feature model (retrain via PLAN_05 once links
   exist; bench 7-feature models are in the June tree if needed).
4. **Early-charge centroid position** (36): integrate the FIRST 120 ns after
   cluster start (bench EARLY_K=2 samples at 60 ns → **6 samples at 20 ns**),
   RAW (not unshared — sharing is the interpolator), centroid over strips.
   Bench gain at θ<5°: 0.73/0.94 → 0.61/0.72 mm; COMBO rule: early-charge if
   n_strips ≤ 9 else production anchor, per-branch offsets.
5. **Walk**: per-strip walk is negligible (30 % CF handles it — bench);
   remeasure the inter-plane walk (~−100 ns per unit charge-asym at bench)
   and apply to pairing Δt0 and PLAN_07 timing.

## Phase 2 (only after PLAN_05 convergence)

Port the frozen unsharing pre-filter + early-charge estimator into
`mm_strip_reconstruction` (C++), so combined_hits carries unshared timing and
the sub-pitch position natively. Keep the python path as the reference
implementation for regression tests (bit-compare on a pinned run).

## Acceptance

- c1/c2/delay remeasured at beam ≈ bench values (design property — if c1
  differs by >0.05 something structural changed; stop and report).
- Unshared time-fit v (from S_seg vs regressed angle) converges toward the
  PLAN_05 v within ~3 % — the bench three-estimator convergence, in situ.
- Early-charge centroid beats the production anchor on low-angle cosmic
  links by ≥10 % (bench: −16/−23 %).
- Runtime: full candidate set of a typical subrun processes in < the subrun's
  wall-clock (else the monitor falls behind — decimate harder or move cuts
  earlier).

## Gotchas

- 12-bit saturation: unsharing assumes linearity — exclude saturated strips
  from the solve (bench did; flash-adjacent windows especially).
- ZS holes: suppressed samples are ABSENT, not zero — rebuild dense (channel
  × sample) arrays with explicit zeros before the banded solve, and remember
  a suppressed neighbour biases c1 measurement low (require neighbours
  present).
- Memory: 400×512 float32 = 0.8 MB/event — chunk at ≤200 events (bench used
  400×32-sample chunks at ~1 GB RSS).
