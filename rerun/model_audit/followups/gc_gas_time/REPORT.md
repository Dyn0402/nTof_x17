# GC — Gas change vs time, and its consequence for the reconstruction (round 3, item 3, 2026-10-28)

## Verdict: can wait until after the reprocessing, with two cheap changes before it
**v does not enter the waveform fit.**
- The forward model fits (p0, w, t0) in time units; the diffusion term (Dp²·u) and the kernel are in ns.
- v is used only in three places:
  - `tan = (w − w0)/(kw·v)` (`reco.tan_of`, `tan_bow`);
  - seeding (`init_guess`, `_two_sided`, the two-track grid starts);
  - the u_window plausibility window (0.30–1.60 × 30 mm/v).
- Changing v by ±10 % gives bit-identical w on 120 det4 events (FA).

So v(t), and any per-row scale, is a correction to the tables (tan, mid-gap position, depth). p0 at the mesh does not
depend on v. Nothing needs refitting.

**Before the reprocessing:**
1. **Write the DREAM `timestamp` and the acquisition tag (file stem) into every bench event record.** Joining afterwards
   on (run, subrun, eventId) is not safe everywhere: eventId restarts between acquisitions in det3 6-22 short_run (two
   stems, 130 duplicate ids in 14 193). n_TOF tables already carry `tag`.
2. **Give the flushing rows their own v group in `pass_registry.csv`** (refit = v). They inherit an angle scale 9–32 %
   off (table below): det4 6-23 short/longer, det6 and det7 6-26 short/longer, the det6 6-26 afternoon HV scan. This is
   post-correctable too. Doing it first keeps the seeding and the u_window centred: det4 and det6 sit only ~9 % inside
   the u_window upper edge (1.46 vs 1.60).

## Method
- **Bench, v(t):** FA's model-free column duration T (`colsum.py`), on every trigger of all 159 non-excluded registry
  rows. That is 150 new condor jobs plus FA's 9. Clock = file-stem acquisition start + DREAM timestamp × 10 ns (the
  timestamp is in `decoded_root`, just not in the event record).
- **Bench angle scale per row:** FA's M3 ladder (`tilt_rows.json`, 47 rows).
- **n_TOF, two independent routes:**
  - the production capsule-band k per sub-run (36 runs, arms A/C; k ∝ 1/v);
  - the stacked strip-sum falling edge relative to the trigger, on 76 of 79 sub-runs (all sub-runs of runs 79, 86, 104,
    145, 152, 162, plus one sub-run each of 13 other runs).
- **T is gain-confounded.** On det3's two 6-27 mesh scans, T rises 18 % from 460 to 525 V (the scans agree with each
  other), while the ladder scale is flat to ±1 %, dropping 3 % only at 520–525 V where tracks saturate. So T measures v
  only at stable gain; across HV scans and gain ramps the ladder counts.

## Results
**Bench, within a row** (10-min blocks, tan error relative to the row mean):

| row | duration | rms | worst | note |
|---|---|---|---|---|
| det2 6-22 golden longer_run | 2.0 h | 0.27 % | < 1 % | |
| det2 6-23 long_run | 9.8 h | 1.25 % | 2.3 % | |
| det3 sat golden | 2.35 h | 0.18 % | — | |
| det3 6-27/28 night | 7.6 h | 0.33 % | 0.8 % | |
| det3 6-25 | 3.1 h | 0 % | 0.3 % | |
| det4 golden 6-24 | 6 h | 1.7 % | 4.7 % | ladder moves 1.45× faster than 1/T → up to ~7 % |
| det6 golden 6-26 | 4.3 h | 3.0 % | 6.7 % | monotonic +11 %; ladder 1.68× faster than 1/T |
| det7 golden 6-26 | 4.3 h | 0.32 % | 1.1 % | |

**Bench, inherited rows against their calibration row** (ladder; tan_reco = (1 + ε) tan_true):

| row | ε |
|---|---|
| det3 6-27 mesh scans (18 rows) | −1.5…+0.3 % (−3 % at 520/525 V) |
| det2 6-22 short / 6-23 long | +1.1 / −0.7 % |
| **det4 6-23 short / longer** | **−31.9 / −22.7 %** |
| **det6 6-26 short / longer** | **−31.6 / −23.5 %** |
| **det6 6-26 HV scan, 495 V row** | **+19 %** |
| **det7 6-26 short / longer** | **−14.1 / −8.9 %** |
| det7 6-26 HV scan, 495 V row | +2 % |

This is gas still exchanging after installation:
- det6: v 17.8 → 31.0 µm/ns over 12 h (from 00:58 on 6-26);
- det7: 31.6 → 37.5 over the same 12 h;
- det4: 24.9 → 28.3 → 36.6 over its first night.

**n_TOF.** Between runs v moves by 14 % (arm C) and 4.5–6.6 % (arm A) over the campaign. The two routes agree (C: r = 0.97,
slope 0.84; A: r = 0.73, slope 1.05). This is already absorbed by the one k per run × arm in stage 3.

Within a run (column end, 30-min bins):

| run | arm A rms / worst | arm C rms / worst |
|---|---|---|
| 79 | 0.56 / 1.34 % | 0.68 / 1.39 % |
| 86 | 0.40 / 0.89 % | 1.01 / 1.62 % |
| 104 | 0.94 / 2.70 % | 0.66 / 1.85 % |
| 145 | 0.29 / 0.59 % | 0.13 / 0.47 % |
| 152 | 0.54 / 1.33 % | 0.95 / 1.86 % |
| 162 | 0.47 / 1.61 % | 0.48 / 1.43 % |

- The band k noise floor is 0.8 % (A) / 1.1 % (C). Its intrinsic within-run rms is ≤ 1.7 %, largest on C in runs 116,
  118, 147 and 156.
- **Arms B and D:** the column end falls outside the 1.2 µs window, so their within-run v is not measured.

**SPS det4:** not measured. The stacked-edge estimator scatters ±15 % between blocks on ZS H4 data and misses the field
steps.

## Consequences with one v per run
Δθ ≈ ε·sinθ·cosθ: 0.25° per 1 % at 30°. Mid-gap position error = 15 mm·ε·tanθ; depth error at the cathode = 30 mm·ε.
Threshold: 0.2° at 30°, i.e. ε = 0.8 %. Bench resolution is 1.06–1.76°.

- **Above threshold, must be corrected:**
  - the inherited flushing rows (9–32 % → 2–8° at 30°, 1–3 mm mid-gap, 3–10 mm in depth);
  - det6 golden (worst 6.7 %, up to 11 % on the ladder → 1.7–2.7° at a run end, 1.2° rms smear);
  - det4 golden (up to 7 % → ≤ 1.7°, 0.6° rms).
- **Borderline:**
  - det2 6-23 long_run: 0.3° rms / 0.6° worst;
  - n_TOF within a run: ≤ 0.25° rms, ≤ 0.7° worst. Small next to the resolution, but systematic in time.
- **Below threshold:** det2/det3 golden, det3 nights, the det3 HV scans, det7 golden.

## After the reprocessing (table-level)
- **Bench:** per-row, per-time-block kw from the M3 ladder (not from T, which depends on gain).
- **n_TOF arms A and C:** per-bin v(t) from the column end, joined on (run, subrun, tag, event_id).
- **n_TOF arms B and D:** no within-run route exists; stated limitation.

## Files (`rerun/model_audit/followups/gc_gas_time/`)
- **Drivers:** `make_jobs.py`, `gc_row.sh`/`.sub`, `ntof_colsum.py` + `ntof_row.*`, `sps_colsum.py` + `sps_row.*`.
- **Analysis:** `gc_common.py`, `gc_bench.py`, `gc_ladder.py`, `gc_ntof.py`, `gc_sps.py`, `gc_consequence.py`,
  `gc_figures.py`.
- **Results:** `results/*.json`, `results/consequences.md`.
- **Figures:** `figures/gc_bench_ladder.png`, `gc_bench_within.png`, `gc_ntof_campaign.png`, `gc_consequence.png`.
- **EOS:** `/eos/user/d/dneff/x17rerun/followups/gc/{Analysis,ntof,sps}/`. Local bench pull:
  `~/x17/cosmic_bench/model_audit/followups/gc/`.

## Limitations
- **T is gain-confounded** (~18 % across a mesh scan). The T-based within-row numbers on det6/det7 6-26 short/longer and
  det3 6-22/6-23 are not v.
- **The ladder covers only FA's 47 rows.** det6 HV-scan rows other than 495 V and the det4 6-23 HV rows have no ladder
  number; their ε is unknown (probably large on det6).
- **The ladder-vs-T factor** (1.45 det4, 1.68 det6) means the T-based within-row drifts on those chambers are
  underestimates.
- **The n_TOF column end** assumes a fixed latency and covers arms A and C only. The n_TOF band k is provisional; only
  its run-to-run ratio is used.
- **SPS within-plateau drift is not measured.** Plateaus are ≤ 77 min, ≥ 16 h after the CF₄ switch (τ = 2.3–4.6 h), so
  the drift is expected to be small, but this is unverified. Condor cluster 4583840 (25 SPS jobs) was still running when
  the agent stopped; the extra output won't help until the estimator is fixed. Cluster 4583853 (last n_TOF sub-runs):
  rerun `gc_ntof.py` to add them.
