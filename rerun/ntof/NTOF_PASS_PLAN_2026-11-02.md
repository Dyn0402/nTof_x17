# n_TOF pass 1: plan to reprocess the n_TOF tracks from scratch on the final chain

*Drafted 2026-11-02, before any code. It lists what the bench/SPS pass 1 built, what the old n_TOF chain (prod
fullpass → is2_v2) learned, how the two are combined, and the order of work. Section 8 lists the decisions that are
Dylan's to make.*

**Goal.** Reconstruct every n_TOF trigger of ours with the reconstruction that pass 1 locked on the bench and SPS
(`code_6a0bb1f`, `Analysis_pass1`). Calibrate it from nothing on n_TOF's own conditions, and carry every n_TOF
lesson in code or in the registry, not in somebody's memory. The rule is the same as for pass 1: **one reconstruction
for all detectors**, no `WFT_*` switches, and a ledger that proves every row ran on the archived code.

**What this plan is not.** It does not take new data: the dataset is final. Questions that the existing data cannot
answer become stated limitations.

---

## 1. What we start from

### 1a. The bench + SPS chain (pass 1, done 10-30)
Maps: `rerun/PASS1_STATUS_2026-10-30.md`, `rerun/PASS_PLAN_2026-10-28.md`, `docs/PARAMETERS.md`,
`rerun/history/LESSONS.md`.

- **Model (locked 10-29):**
  - physical RC kernel (`build_matrix_rc`; c2 < c1 by construction), one σ0 = 0.30 mm for both views;
  - X template drives both views; noise-only weights; 3550 ADC clip rule; no gain map; no shape term;
  - dead/short channels censored; seeder bridges dead strips (`DEAD_SEED`); `COL_GUARD`, `RC_T0_INTERP` and
    `PIVOT_START` built in.
- **Fitter:** LM variable projection (`FITTER='lm'`, `LM_HOPS=0`); two-track on with a 4-pitch minimum; column-time
  x–y gate (`TC_TOL` 200 ns); χ²/dof quality flag at 5 × the table median per view (a flag, never a gate).
- **Calibration DAG** (`make_pass_dag.py` CAL0–CAL4 per golden row): templates → D_rc,y (pinned t0) → bundle →
  waveform alignment → w0/kw → bow → iterate to convergence (`convergence.py`). v is the timing v = gap / column
  duration; it is never a waveform χ²(v).
- **The calibration DAG needs M3 at every step**: matched RECO, ALIGN, ANG, W0KW, BOW, DRC's ref-pinned p0, and the
  templates from inclined M3 tracks. **None of it can run as-is on n_TOF.**
- **SPS is the closest analogue** (`rerun/model_audit/sps_reco.py`, `sps_build_cache.py`, `sps_mask.py`):
  - the pass-1 bundle with bow, v_map, w0 and t0_abs dropped, and t0 free;
  - seeding from the waveforms, not hits;
  - a channel layer measured at H4, with the hot-probe / near-oscillator / coupled-channel rules;
  - K = v_span + 15;
  - per-row timestamp, acq_tag and quality_flags.

### 1b. The old n_TOF chain (prod fullpass 09-10 → is2_v2 10-09)
Map: `ntof_cosmics/HANDOFF_TRACKING_2026-10-06.md` §14–22, `sept26_prelim_analysis/` (DATASET.md,
SAME_CHAMBER_PAIRS.md), `ntof_tracking/wft_beam.py`.

| stage | what | where |
|---|---|---|
| 0 | sample freeze: 36 production runs, 293 sub-runs, 25.6 M triggers, by run title | `freeze_sample.py` → `/media/dylan/data/x17/sept26_prelim/stage0/sample.csv` |
| 1 | hits candidate filter | abandoned: it kept 12.8 % of two-track triggers, so the pass is blind (every trigger) |
| 2 | waveform fit per (run, subrun, arm, tag) | `ntof_tracking/condor/run_beam_job.py` → `wft_beam.py`; EOS `analysis/sept26_fullpass_is2_v2` |
| 3 | track tables (x–y pairing, `gated`, `tan = k·tan_raw`, y up) | `build_tracks.py`, `campaign_tracks.py` → `stage3_*` |
| k | angle scale per (run, arm): in-beam muons × capsule band; ×1.12 capsule focus (provisional) | `k_insitu.py` → `kcal_is2_v2[_cf]` |
| 4/5 | pairs, capsule pose/focus, scint stack, calorimetry, qa-tracks | `sept26_prelim_analysis/*`, `ntof_scint_stack`, `ntof_calorimetry` |

All four arms of the beam-off cosmics were also reconstructed once (`analysis/cosmics_fullpass`, run_149 per arm and
file tag); that is the old in-situ truth sample. is2_v2 covered **A and C only**. It used old copy-amplitude r06/lp kernels, in-situ v A 38.0 / C 28.7 from the
run_149 A–C cosmic line, no dead/hot masks, the NM fitter, two-track off, min strips 3, two-sided search, and
`TAN_MAX` 1.0 raw. **It can no longer be reproduced with today's code:** its switches are now diag overrides and
the defaults behind them have changed. That is not a problem for a from-scratch pass, but is2_v2 is only a
reference, not a baseline we can rerun.

---

## 2. What has to carry over: the checklist

Each item names where it must live in the new chain. ✱ = not in any chain today.

### Inputs and conditions (registry columns)
| # | lesson | carried by |
|---|---|---|
| N1 | Ours by **run title**, never by run number or recency; last beam run 224715 | registry builder reads `run_config.json` titles + `x17-match.json` |
| N2 | Production readout: latency 27, **20 samples × 60 ns (1.2 µs)**, ZS off. Pre-production: 35 / 32 | registry `latency`, `n_samples` per sub-run (set_nsamp per window, as on the bench) |
| N3 | Noise era: ≤ run_67 quiet, run_68 unplaced, ≥ run_69 ×2 noise (`RdClk_Div` 6→4) | registry `noise_era`; run_68 excluded or checked directly |
| N4 | Chamber A x connector 8 (FEU 3 ch 448–511) dead through run_79 ✱ | n_TOF channel layer, per era (§3, P2) |
| N5 | Chamber A common mode ×3.3 from 21 to 27 July; the 27 July access is a condition step (run_79/81 INTER ×2) ✱ | registry `access_epoch`, `cm_episode` |
| N6 | Gas **Ar/iso 90/10** (bench was 95/5), in series A→B→C→D; water A 0.15–0.3 %, C/D 0.35–0.7 %; no O₂ attachment at n_TOF (O₂ is an SPS finding) | Dp and v are per gas (§3, P3); Magboltz `drift_9010_contam_cern.json` |
| N7 | HV per sub-run (production resist A540/B540/C525/D520, drift 700 V); E = V/gap with the timing-free gaps (det3 28.08, det2 31.12, det7 29.44, det6 28.0) — not the 27.9/30.5/30.0 in `BEAM_DETS` ✱ | registry `hv_*`, `gap_mm` from `physics_inputs.json` |
| N8 | Chamber map A=det3 (FEU 3/4), B=det2 (5/6), C=det6 (7/8), D=det7 (1/2); global pose from `ntof_tracking/reco/geometry.py` | registry + geometry module, one source |
| N9 | n_TOF cabling differs from the bench: bench masks must be remapped (`wft/calib.py:361`); `wft_beam.make_bundle` never calls the remap ✱ | P2 |
| N10 | Clock: DREAM↔n_TOF match 95.8 %, 6 ns (`ntof_dream_merge/DREAM_NTOF_CALIBRATION.md`); `pulse_match` supercycle mis-lock on 25.7 % of beam: fix state not recorded ✱ | P1 audit; ToF per event in the stage-2 record |
| N11 | n_TOF file choice: `prod_v12` (`v12_liqpileup`), `prod_v12_recover` for 224526, 224576 only `prod_v11` | registry `ntof_file` |
| N12 | run_104 stat090_0016 A/C decoded empty; run_145 stat090_0003 a stub; run_120/137 never joined | registry `excluded` with a reason |

### Reconstruction (code defaults)
| # | lesson | carried by |
|---|---|---|
| R1 | Never geometry from `combined_hits` times; hits for candidates/QA only | wft tables only |
| R2 | Blind full pass (every trigger), not a hits pre-filter | one stage-2 node per (run, subrun, tag), all arms |
| R3 | Flash: chambers dead 0–0.5 ms; veto 1 ms; 1–10 ms flash junk; pairs ToF ≥ 12 ms; late tracks t0 > 300 ns unreliable (17 %) | stage-2 records ToF; cuts are stage-3 columns, never a reco gate |
| R4 | Busy plane > 150 hits (not the bench spark veto 50); up to 5 candidates per plane; isochronous veto | **reconcile with the bench seeder (§3, P4)**: one rule with a beam/bench parameter, or a stated difference |
| R5 | Seeder minimum 3 strips (+40–52 % wall-confirmed); bound at definition time, so patch both the constant and the call | default in the new driver, not an env var |
| R6 | Steep-track mirror basin (13–40 % at |tan| ≥ 0.3 one-sided): two-sided search / PIVOT_START | check that LM + PIVOT_START alone fixes it on n_TOF; if not, two-sided becomes a default (§3, P5) |
| R7 | `TAN_MAX` is in **raw** tan, so its true reach moves with v; `W_SCAN_HALF` reach 0.55–0.73 | set from the chamber's true v so that the true reach is ≥ 1.0 (§3, P3) |
| R8 | Depth grid 18 bins rails `q_uend` at 1080 ns on ~40 % of tracks | K covers the 1.2 µs window (K = 20, plus the bench margin rule) |
| R9 | Unregularised NNLS ran away on 30 % of n_TOF tracks | `COL_GUARD` (built in) |
| R10 | Saturation: `SAT_ADC_BEAM` 3700 vs locked bench clip 3550 | **one rule** (§3, P3; decision Q5) |
| R11 | Bands absent at n_TOF (window 1.2 µs < `BAND_MIN_WINDOW_NS`); `DROP_FLAT_HITS` bench only | inert by construction; state it in the meta |
| R12 | Bench v_map and w0 never at n_TOF (per bench mounting); bench t0_abs never (an n_TOF latency is measured instead, Q2). **The bow block now transfers** (Dylan 11-02, §5) | the n_TOF bundle keeps the bow with provenance; the driver refuses v_map / bench w0 / bench t0_abs |
| R13 | LM with free t0 relabels t0/p0 by one 60 ns bin on 13–33 % of planes (same angle); x–y pairing on run_145 unchecked | P5 gate; goes away if t0 is pinned (Q2) |
| R14 | χ²/dof flag purity at n_TOF unmeasured (~10 % flagged) | P5: measure with x–y charge balance / A–C pointing |
| R15 | Two-track thresholds F = 1200 (A) / 2400 (C) were set on old bundles; two-track was off in is2_v2 | refit F on the new bundles (P5) |
| R16 | Busy multi-particle planes are not refitted | stated limitation unless P5 finds it costs pairs |

### Angle scale, coordinates and geometry
| # | lesson | carried by |
|---|---|---|
| G1 | **+y up**; x view flips position, y view flips tan (`COORDINATES.md`); never decide a sign by self-consistency | stage-3 writes `y_convention: "up"`; sign test pinned to cosmic muons going down (P5) |
| G2 | `tan = k·tan_raw` (not /k; DATASET.md and the `campaign_tracks.py` docstring are wrong) | new stage 3 stores true tan from the bundle v; k only as a check |
| G3 | v at n_TOF: in-situ A 38.0 / C 28.7 from the run_149 cosmic line; D wall 1.525 → v ≈ 28; B never certifies (no field-shaping ring) | P3 measures v from the n_TOF column duration (the bench rule); in-situ line and D wall are the checks |
| G4 | C's v moves 3–5 % within a run (needs per-sub-run v on runs 79, 104, 116, 118, 124, 147, 152, 156); A one per run; B/D no route | v per sub-run from the column duration (P3); B/D stated limitation if their column is unusable |
| G5 | Capsule focus wants ×1.12, in-beam muons say ≈ 1.00: **open, 12 %** | P5 must close or state it; the new chain's v changes the input |
| G6 | Beam electrons scatter: the wall is not angle truth for beam (0.89 vs cosmic 1.15) | wall used only for cosmics |
| G7 | Source at y −31 (A) / −35 (C) mm, x ≈ −9.5; `pointing()` still uses the axis at x = 0 ✱ | fix `pointing()` with the measured source (P6) |
| G8 | Positions from pointing (band crossing), never from the shape of a distribution | rule for every alignment step in P3/P5 |
| G9 | EAR2 beam: r50 10.5 mm, bore 0.46 @ 19.5 m; Lambda2D is the resolution function | downstream only |

### Process
| # | lesson | carried by |
|---|---|---|
| X1 | One frozen `git archive` per pass, stage_job pattern, STAGES json, no overrides, ledger | new n_TOF phase in the pass tooling (P6) |
| X2 | EOS: only fresh `git archive` extracts work on workers; RETRY 3; outputs under `/eos/experiment/...` through `root://eosexperiment` (the old stage-2 wrapper still pushes via `eosuser`: a bug) ✱ | P6 |
| X3 | Every promotion idempotent; retired products get a RETIRE record | P7 |
| X4 | Condor jobs write to EOS from inside the job and clean scratch (AFS quota incident 10-07) | P6 job script |
| X5 | Each analysis ships a generated `report.html` | P5/P8 |

---

## 3. Work packages (in order)

**P0. Registry** — `rerun/ntof/make_ntof_registry.py` → `rerun/ntof/ntof_registry.csv`
- One row per (run, subrun, acquisition tag), built from `stage0/sample.csv`, each `run_config.json` and the dylan-cern-site
  match files.
- Columns: class (production / beam-off cosmic / scan / pulser / DAQ-scan), title, n_TOF runs and file version,
  latency, n_samples, ZS, HV per chamber, drift field, gas era, noise era, access epoch, CM episode, mask era,
  excluded + reason, calib_key (which v / latency row it takes).
- Scope (Q1): production 36 runs + the 41 beam-off cosmic runs at the production point (calibration and truth).
  Pre-production scans only if Dylan wants them.

**P1. Inputs audit (no physics yet)**
- **Decoder:** n_TOF `decoded_root` predates fa80c06. Re-decode a sample (≥ 1 sub-run per run, 36 + a few cosmic) with
  the pinned decoder and byte-compare, as on the bench (1278/1278 identical there).
  - Identical → keep the existing files.
  - Different → re-decode the campaign (compute is not a constraint) and archive the decoder hash.
- **Hits** (seeding/QA only): decide whether to regenerate with the rr1 recipe (plain-median CM, matched filter
  `--mf 5`, `--tps 60`).
  - The beam seeder reads hits through `read_hits_tag`.
  - The SPS stage seeds from the waveforms and needs no hits. Prefer that if it reaches the same seeding efficiency
    (Q4).
- **n_TOF join:** confirm the pulse-match mis-lock fix is in `prod_v12` and record the margin; verify ToF on 3 runs.
- **Pedestals:** every acquisition has its own pedestal (56 on EOS); check none was taken with a connector off (the
  g_det4 lesson).

**P2. n_TOF channel layer** — `rerun/ntof/channel_layer/`
- Start from each chamber's pass-1 bench `dead` list (census dead + shorts), remapped through the n_TOF cabling.
- Add n_TOF-measured dead/hot per era with the SPS rules (hot probe per 1000-event block, > 3 % seeding and > 10× live
  ±4 neighbours; near-oscillator; coupled-channel). The old 102-channel run_145 mask and D's hot cells
  (`HANDOFF_D_NOISY_CHANNELS.md`) are the cross-check.
- A connector 8 through run_79; the early-July / 9 August connector-8 dropouts on A-y/B-y.
- Output: one mask per (chamber, era), named by the registry.

**P3. n_TOF calibration phase, from nothing** — `rerun/ntof/calib/`
- **Carried from pass 1 (chamber properties):** physical RC kernel form, σ0 0.30, D_rc,y (det3/det2/det6/det7 pass-1
  values), X template.
  - The template cannot be measured on the 20-sample window (`wft/calibrate.py:470-480`), so check that it holds:
    residuals of unsaturated head-on columns at n_TOF.
  - The template is the strip self-response, but its ion tail could move with the gas (90/10 vs 95/5). If the
    residual check fails, that becomes a stated systematic.
- **Re-derived for 90/10 at n_TOF:**
  - **Dp:** Magboltz 90/10 + water, checked on the n_TOF unsaturated head-on stack (cloud_basics stack rules).
  - **v:** the timing v = gap / column duration, per chamber and per sub-run, using the timing-free gaps. This is the
    bench rule and it needs no M3. The 1.2 µs window holds a 28 mm column at v ≥ 24 µm/ns. Check that C (v ≈ 28)
    does not touch the window end.
    - It replaces the capsule-band k as the primary angle scale.
    - k (in-beam muons, capsule band), the run_149 A–C line and D's wall become independent checks.
  - **t0 latency** (if Q2 = pin): measure C at n_TOF per chamber and per latency setting (`measure_latency`), as on
    the bench. **A, C and D are treated alike (Dylan 11-05):** all three pinned, cosmic latency on cosmic rows and beam
    latency on beam rows. D's cosmic latency is measured (run_149); D's beam latency is not measurable (oscillation)
    and is D's cosmic latency + the mean beam-minus-cosmic offset of A and C (§6, 11-05). B: positions only, t0 free.
  - **kw (and w0 = 0):** from the beam-off cosmic A–C through-goers (the only geometric angle truth at n_TOF,
    `ntof_cosmics/insitu_calib.py`, kwmed rule). B and D have no partner line. D uses its wall on cosmics; B stays
    positions-only (no field-shaping ring).
  - **Alignment:** chamber poses from the cosmic A–C line and the capsule band crossing (pointing estimators only).
- **K** = 20 (window) with the bench margin rule; `TAN_MAX` raw set from the chamber's v for a true reach ≥ 1.0.
- **Saturation:** one rule (Q5).
- **Convergence:** iterate v / kw / alignment until they stop moving, by the same criteria as `convergence.py`.

**P4. Driver** — `rerun/ntof/ntof_reco.py`, the n_TOF counterpart of `sps_reco.py`
- Built on `wft.reco` with the pass-1 defaults (LM, two-track, COL_GUARD, PIVOT_START, DEAD_SEED), the bundle rules
  of R12, and the registry for every per-row number.
- One seeder: either the beam seeder (busy 150, min 3, 5 candidates) folded into `wft/seed.py` as a named beam
  parameter set, or the SPS waveform seeder. Chosen on P5 numbers, not by default.
- Record per event: ToF, t_since_flash, DREAM timestamp, acq_tag, quality_flags, the n_TOF bunch/pulse id, and
  `x/y_bow_h = 0` explicitly.
- Stage 3 rebuilt on the new record: true tan from v (no k multiplication), `gated` as a column, y up by
  construction, sidecar with code hash + registry hash.

**P5. Local tests on ucla** — `$WORK_DATA/ntof_pass1/` (= `/media/ucla/work/ntof_pass1/`)
Stage the following decoded_root + hits + run_config from EOS. Sizes checked on EOS 11-02: a production sub-run is 6–12 GB
decoded + 0.5–1.2 GB hits, a run_149 sub-run 2.4 GB. About 10 sub-runs is ~100 GB, against 843 GB free. Delete when done;
nothing unique is stored there:

| sample | why |
|---|---|
| run_149 (beam-off cosmics, 21.8 h; a few sub-runs) | geometric truth: A–C line, D wall, y sign from muons going down |
| run_145, 2–3 sub-runs | the is2 reference run: hot mask, pairs, LM label check (R13) |
| run_79, 1 sub-run | A connector 8 dead, pre-access, CM episode |
| one of C's v-drifting runs (e.g. run_116) | per-sub-run v |
| run_162 (last) | late-campaign gas and gain |

Tests, in order:
1. Decoder byte-compare (P1).
2. Bundle build (P3) on run_149 + run_145.
3. A/B against is2_v2 on the same events: efficiency per arm, wall-confirmed fraction, angle s68 on the A–C cosmic
   line, capsule band crossing y, x–y pairing rate, χ²-flag purity (x–y charge balance), two-track F scan.
4. Seeder choice (P4).
5. Mirror-basin check with and without two-sided (R6).
6. t0 pinned vs free (if Q2 needs data).
7. `wft/tests` + an n_TOF regression test on a frozen run_149 sub-run.
8. g4_digi closure (`ntof_cosmics/g4_digi/`) re-run with the new driver: electrons vs muons, angle scale.

Each test is written up in `rerun/ntof/tests/<name>/report.html`.

**N-G1 gate (n_TOF analogue of G1).** On run_149 + run_145, against is2_v2 on the same events:
- efficiency and wall-confirmed fraction ≥ is2_v2 − 0.5 pt per arm;
- angle s68 on the cosmic A–C line ≤ is2_v2;
- source y within its error of −31 / −35;
- the capsule-focus vs muon scale tension (G5) either closed or quantified.

A failing arm is held, as on the bench.

**P6. Condor pass on lxplus**
- An `ntof` phase in the pass tooling (`make_pass_dag.py ntof` or `rerun/ntof/make_ntof_dag.py`): calib nodes →
  N-G1 → reco nodes, one per (run, subrun, tag), **all four arms** → stage 3 → ledger.
- Output: `/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/` (Q3), through `root://eosexperiment`; the stage-2
  output-host bug fixed.
- Before submitting: dry run `--limit 300` on one sub-run, `lxstore status`, `git archive` to EOS, registry hash.
- Size guide: is2_v2 was 6,466 jobs for 2 arms with the two-sided search. Four arms ≈ 13 k jobs. Compute is not a
  constraint.
- Ledger `rerun/ntof/ntof_ledger.py`: every non-excluded row present, code = archive, no overrides, per-row
  efficiency and v within bounds.

**P7. Downstream migration**
- Repoint, in this order: same-chamber pairs, capsule pose/focus/band, scint stack (track extrapolation; its
  `tanx = k·tan_raw` calibration becomes `tanx = tan`), calorimetry, Athens imaging, `capsule_tof`,
  `tight_coincidence`, `chamber_b`, vt_all, qa-tracks shards (snappy), the analysis board.
- Retire `stage3_fullpass`, `stage3_is2_v2[_cf]`, the `kcal_*` sets and the is2 bundles with a RETIRE record. Keep
  them on EOS as references.

**P7a. Drift velocity vs time, a standard pass product (Dylan 11-05)** — `rerun/ntof/vt/`
- v does not enter the waveform fit (`gc_gas_time/REPORT.md`: ±10 % in v gives identical w), so v(t) is a table-level
  correction and a gas monitor; nothing is refitted for it.
- **A and C:** the model-free column duration T (`colsum.GridFit`, pass-1 X template) on clean single-cluster tracks of
  the pass output, per sub-run and in 30-min bins, v = gap / T with the timing-free gaps. Over the whole campaign, beam
  and cosmic rows separately (beam and cosmic samples differed by ~3 % on A: a selection systematic, kept apart).
- **D:** from the beam-off cosmic runs (D's beam column is oscillation-dominated; its cosmic box fits work on the
  fa80c06 decodes, 29.9–32). That gives D at the cosmic-run cadence over the six weeks, and an independent cosmic v
  for A and C at the same times. **B:** no v (no box).
- Cross-checks: the per-run capsule-band k (∝ 1/v; the old route, `gc_gas_time` C r = 0.97) and, where lines exist,
  the per-sub-run A–C line scale.
- Output: `vt_<arm>.parquet` (run, subrun, tag, t_start, t_end, v, err, n, trigger class) next to the stage-3 tables;
  stage 3 joins it on (run, subrun, tag[, time bin]) and writes `tan_vt` beside `tan` (one v per chamber) so either can
  be used. A campaign figure per chamber with HV changes, the 23 July clock change, the 27 July access and gas
  events marked, in the pass `report.html`.
- Caveat to state: T is gain-confounded (+18 % across a det3 mesh scan), so v(t) is quoted only at fixed HV; across an
  HV change it is flagged, not interpreted.

**P8. Docs**
- `docs/PARAMETERS.md` n_TOF section.
- n_TOF lessons into `rerun/history/timeline_data.py` (regenerate LESSONS.md).
- CLAUDE.md: replace "n_TOF out of scope" with the new pass.
- One generated `report.html` for the pass.

---

## 4. Known limitations to state, not to chase
- **B:** no field-shaping ring, so no angle. Positions only, and ~13 % cosmic efficiency.
- **B:** no v route (column not a box). **D:** v(t) only from the beam-off cosmic runs (P7a), not within beam runs.
- **Template:** not measurable on the 20-sample window. It is carried from the bench (P3 check).
- **Window:** 1.2 µs, so a column slower than ~24 µm/ns would be cut (C is closest).
- **Busy planes:** multi-particle planes are not refitted (R16).
- **Gas:** n_TOF gas is 90/10 and only Magboltz + in-situ v anchor it; there is no SPS-style gas fit at n_TOF.

---

## 5. Decisions (Dylan, 2026-11-02)
- **Q1, scope.** The minimum now: the 36 production runs and the beam-off cosmics at the production point
  (`make_ntof_registry.py --scope pass1`). Pre-production goes in a last pass at the end (`--scope all`).
- **Q2, t0.** Pin it, but test before the algorithm is locked: pinned against free on run_145/run_149 (P5 test 6).
  The first look at is2_v2 is below (§6). It shows the prompt peak, and also structure the pin has to deal with.
- **Q3, output location.** Yes: `/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/`.
- **Q4, hits.** Hits only decide which strips and windows to fit. Choose on run_145 numbers: rr1-style regenerated
  hits vs waveform seeding (the SPS seeder is single-track, so it needs extending for pairs).
- **Q5, saturation.** Probably 3550, but check it on n_TOF (P5).
- **Q6, A/B sequence.** run_145 + run_149 first; expand to all five samples when those pass. Compare CPU time per
  event too (is2_v2's two-sided search cost ~20×).
- **Bow (new).** Carry each chamber's pass-1 bow block to n_TOF instead of dropping it. Then check bow-on against
  bow-off on the run_149 A–C cosmic line and on the capsule focus.
  - The block is in the chamber's own strip-map frame (U, W from the plate centre; `wft.reco.tan_bow`/`bow_h`), and
    that frame is shared by bench, SPS and n_TOF.
  - The bench review found the dish static and mostly fabrication/mesh tension (pressure ≤ 1/3), so it should be a
    chamber property. The open part is orientation: on the bench the chambers lie flat; at n_TOF they stand vertical
    around the capsule.
  - `wft_beam.make_bundle`, `reconstruct_subrun` and `sps_reco._init` drop or refuse the block today. The n_TOF driver
    has to keep it, as an explicit bundle field with provenance.
  - B has no field-shaping ring, so its drift field (and its bow-induced tilt) is not the bench's: B carries no angle
    either way.

- **Re-decode (11-02, approved).** Full campaign, fa80c06, into `decoded_root_fa80c06/` next to the old files:
  `rerun/ntof/redecode/` (job per acquisition, 3,937 jobs, condor 4612805; one record per FEU file in
  `.../july_beam/Analysis_pass1/checks/redecode/`). The pass reads only `decoded_root_fa80c06`.
- **Angle limit per dataset (11-02).** `CalibrationBundle.tan_max` (None = the reco default 0.6, which every bench/SPS
  bundle keeps); `wft.reco.tan_max()` applies it and a `WFT_DIAG reco.TAN_MAX` still wins. Registry column `tan_max`
  is **0.6, tentative** after the first test (§6). Old study scripts that assign
  `wreco.TAN_MAX` directly (g4_digi, yield_gate) are overridden by a bundle that sets its own; they must pass
  `--tan-max` into the bundle instead.

## 6. Findings so far
- **Registry (P0, 11-02).** 3,941 acquisitions in scope; 3,937 kept: production 36 runs / 3,233 acquisitions / 2.83 TB
  decoded, cosmics 40 runs / 704 acquisitions / 0.58 TB. Every kept acquisition has latency 27, 20 samples, Ar/iso
  90/10, HV 540/540/525/520 + drift 700 V, and the noisy era. Excluded: run_148 (no data), and three 0-byte
  acquisitions (run_104 stat090_0004 and 0009 tag _012, run_124 stat090_0007 tag _012). 255 acquisitions (run_79 +
  run_81 start) are pre-access with A connector 8 dead.
- **Decoder (P1, spot check `rerun/ntof/decode_spot/`, condor 4612778).** The first acquisition of every run was
  re-decoded with fa80c06 and compared with EOS: 320 of the first 334 FEU files are identical. The rest:
  - **Truncated EOS decodes.** run_130 stat090_0000 `_000` holds 654 of 8,387 events. A campaign-wide size check
    (decoded/raw ratio, 0.572–0.616 on 99 % of acquisitions) finds four such acquisitions: run_81 stat090_0001 `_001`,
    run_124 stat090_0006 `_005` and stat090_0010 `_011`, run_130 stat090_0000 `_000`. They are the registry's
    `redecode` column. is2_v2 and every older pass saw only that fraction of them.
  - **Fused triggers.** The fresh decode splits a few fused events (run_126 +8, run_147 +1 per file). It is the same
    eventID-split fix the June nights needed.
  - **The independent audit does not fit the n_TOF frame format.** It reports 12 % (FEU 01) and 7 % (FEU 03) of
    events "incomplete" (< 20 samples), but the C++ decode of those events has 20 distinct samples per event with
    normal noise (5.5 vs 6.0 ADC, no duplicated sample). The audit counts ~56 k "orphan" 64-word data runs outside a
    DREAM block header and ignores them. Before the audit can certify n_TOF it has to learn that layout: open item,
    the C++ decode stands meanwhile.
  - **Recommendation:** re-decode the whole campaign with fa80c06 into `decoded_root_fa80c06/` next to the old files,
    as the bench did, so the pass has one decoder provenance (3.4 TB). Needs Dylan's go-ahead (EOS space).
- **Channel maps.** Bench and n_TOF channel → strip positions are identical on all four chambers, both views, all 512
  channels (only the FEU numbers differ). Dead lists and the bow block carry by channel number and strip frame, so
  N9's remap is not needed. The n_TOF census (P2) still checks the masks.
- **First n_TOF reconstruction on the pass-1 algorithm** (`rerun/ntof/ntof_reco.py`, run_145 stat090_0000 tag `_000`,
  arm A, 300 events, bundle `ntof_bundle.py` v0: pass-1 det3 + bow, v 38.0, K 20, t0 free).
  - It runs end to end.
  - The NNLS charge runaway (q_sum > 1e5) falls from 84 to 1 of 300 events.
  - Per-view ok is the same as is2_v2 (seeding unchanged).
  - t0 is far narrower: 16–84 % −71…+329 ns, against is2 −172…+450.
  - The bow mesh height reaches 5 mm at the plate corners (quadratic extrapolation beyond the bench fit: check).
  - Cost: 2.6 CPU-s per event.
- **t0 in is2_v2, run_145 stat090_0000** (x–y consistent tracks, |t0x − t0y| < 60 ns):
  - A prompt peak at −60…0 ns holds most tracks. Its width (~150 ns FWHM) is several times the bench's.
  - A second peak at +420–480 ns holds ~15 % (A) / ~12 % (C).
  - A flat pedestal runs −600…+700 ns.
  - The prompt median does not step with ftst by 10 ns as on the bench.
  - **The +450 ns population is the NNLS charge runaway, not physics:** tan exactly 0, q_sum ~1e7 (prompt 1.5e3),
    q_u50 at the grid edge. `COL_GUARD` (built in since) removes it. The real distribution is the prompt peak.
  - The pin still applies to the trigger track only; the second track of a pair keeps a free t0 (Q2 test).

- **tan_max 0.6 vs 1.0 (11-02, `rerun/ntof/ac_line.py`, run_149 cosbounce_cos_0000, first 1,500 seeded events per
  acquisition, v0 bundles: v not yet measured in situ).** Scored against the A–C cosmic line (residual angle between
  each chamber's direction and the line through both mesh crossings, sep < 60 mm).
  - 1.0 moves **no** track on the 158 A–C events both settings find (no p0 change > 1 mm).
  - 1.0 adds 33 A–C events and loses 7. Of the 33, ~27 are truly shallow tracks (line tans < 0.6 in both views)
    given a wrong steep fit in one arm; ~6 are genuinely steep and fitted roughly. Median residual of the 33: 16° (A),
    11° (C).
  - At 0.6 the line's 31–40° bin is still reconstructed (A 68 % residual 2.8°): plausibility is per view in raw tan,
    and a 35° line splits between the views.
  - **The two-sided search** (`WFT_DIAG reco.TWO_SIDED=1`, is2_v2's fix) costs 12–14 CPU-s per event (5×) and does
    not improve either setting on the line (0.6: A 68 % 2.6 → 2.8°; 1.0 adds the same kind of junk).
  - **On beam (run_145, 1,500 events) the limit is a preference, not a cut.** Per-plane ok is identical; even at 0.6,
    21–33 % of winners have |tan| > 0.6 (the best implausible candidate is kept when no plausible one exists); 1.0
    swaps the winner on 6–13 % of planes.
  - **Cost:** cosmics 2.1–2.9 CPU-s per event; beam A 2.6–3.5, C 6.7–8.1. 1.0 costs +20–35 % on beam.
  - **Verdict (tentative): 0.6, no two-sided.** Statistics are one sub-run (~170 line events). Re-check on the full
    staged run_149 (12 sub-runs) once v is measured in situ, and look at what the beam's steep winners are (25 % of
    beam planes: real steep electrons, flash-tail junk, or mis-fits).
- **Full re-decode done (11-02, condor 4612805).**
  - 31,496 / 31,496 FEU files, 0 errors, in `<subrun>/decoded_root_fa80c06/`; 246.2 M FEU-events (~30.8 M triggers).
  - In 79 acquisitions the eight FEUs differ by ±1 event. That is the last trigger at the file boundary; the reco
    pairs FEUs by eventId.
  - The decoder reports 0 missing eventIDs on every file whose log tail carries the count. 1,067 records' 12-line
    tails cut that line, so for those the count is not recorded rather than zero.
  - **Next:** point `ntof_reco.py` / `wft.io` at `decoded_root_fa80c06`, and decide hits (Q4) before any production
    node.
- **Capsule pointing as the beam truth (11-02, `rerun/ntof/capsule_point.py`).** Distance of each paired track's line
  to the capsule axis (vertical, x −9.5, z −3.7 mm). On run_145 (1,500 events per arm, v0 bundles), tracks shallower
  than 0.6 point (< 15 mm) 30–45 % of the time; steeper-than-0.6 tracks 5–8 %. limit 1.0 vs 0.6: A 223 vs 208 pointing
  tracks out of 794 vs 548, C 186 vs 165 out of 856 vs 521. So 1.0 gains 7–13 % pointing tracks at ~250–330
  non-pointing ones.
  - Caveat: a steep track's pointing is very sensitive to the v scale. **The limit scan (0.5…1.0 in 0.1) is to be run
    after v is measured in situ**, scored on both truths (A–C line purity/efficiency; capsule-pointing yield vs
    non-pointing).
- **The bench pairing problem (Paper I session, 11-02).** The production chain pairs two-particle bench events
  correctly only ~80 % of the time (98 % for the bare charge-ratio pairing), because a plausibility check tuned on
  beam data drops one track. A separate session is fixing it. **Every n_TOF pair/multi-track number here is
  provisional until that fix lands**, and the n_TOF scans must be redone on it.
- **Q4 seeding study set up.** Three arms, all feeding the same clustering and candidate logic:
  - old `combined_hits_root`;
  - rr1-recipe hits regenerated on the fa80c06 decodes (`rerun/ntof/rehit/`, condor 4613963, `--tps 60 --mf 5`,
    plain-median CM, the bench job with scratch binaries);
  - waveform-native hits (`wft_beam.hits_from_waveforms`: per-channel peak or 3-sample running sum above 5σ of its
    own measured noise). On run_145 A tag `_000` it gives 362 k hits vs the old 235 k (211 k shared); the old
    analyzer's non-robust pedestal RMS makes its threshold stricter. Threshold to be scanned, judged downstream.
- **Column duration and t0 at n_TOF, first look** (`gc_gas_time/ntof_colsum.py` on run_145 stat090_0000, two
  acquisitions per FEU, the pass-1 X templates):
  - **A and C:** columns where expected. A: T peaks ~720–780 ns (v ≈ 33–37 for gap 28.08). C: broad T 840–1140 ns
    (v ≈ 28), with a tail at the window/grid end.
  - **B:** mostly T at the grid minimum. No field-shaping ring: its column is not a box.
  - **D: the box start sits ~200 ns earlier than A/C** (median −210 ns vs ≈ 0, at the grid edge) and T piles at the
    grid maximum. Either D's window cuts the mesh-side start of its columns, or the estimator fails on D. Check on
    cosmics before believing it.
  - **t0 scatter ~80 ns (robust) in every arm, and it does not step with ftst** (−6…+5 ns per class against −10 on the
    bench). If this is trigger-to-particle jitter, a pinned t0 is wrong at n_TOF.
  - **Test:** t0_A − t0_C on run_149 A–C cosmics (the same particle, ~2 ns apart, so trigger jitter cancels) measures
    the chambers' own t0 resolution against the common jitter.
- **t0 at n_TOF, cosmics (11-02, run_149 cosbounce_cos_0000/0001).**
  - **The column-sum box start is a coarse t0 estimator.** Within one chamber t0x − t0y has a robust σ of 75–110 ns
    (≈ 55 ns per view), so the 80 ns scatter seen on beam is mostly the estimator.
  - **D's −210 ns beam offset is not a window offset.** On cosmics D's box starts with the others (median −45 ns vs
    A −35, C −30). What differs on beam is D's beam events (hot cells / flash), to be looked at; the earlier D finding
    is withdrawn.
  - **Fitted t0 on the same muon** (165 A–C line events, v0 bundles, free t0): t0_A robust σ 49 ns, t0_C 73 ns,
    t0_A − t0_C 86 ns. That gives a common (trigger) jitter of only ~13 ns, against a free-fit t0 resolution of
    47 (A) / 72 (C) ns, so the trigger knows t0 3–5× better than the free fit. Offsets differ per chamber (A − C −53 ns),
    so latency is per (arm, view).
  - **Latency** (median of t0 + 10·ftst over ~600–700 single cosmic tracks, `bundles/latency_<arm>_r149.json`):
    A x −5 / y +1 ns, C x +35 / y +27 ns. The ftst slope is −5…−7 ns per class against the bench's −10 (diluted by the
    free fit's lattice?). Open.
  - **Pinned (σ 15 ns) vs free on the A–C line** (474 candidate events): no measurable change. 171 vs 165 line events;
    common 158: A 68 % residual 2.75 vs 2.70°, C 3.64 vs 3.67°; median line separation 8.3 vs 7.9 mm.
  - **Not conclusive yet:** the line residuals are dominated by the uncalibrated angle scale (v, kw), which pinning
    cannot fix. Decide t0 after v and kw are measured in situ; the beam trigger (PS + singles) also needs its own check.
- **Order of the remaining calibration studies:** v in situ (column duration on full-gap tracks, A/C; B/D
    limitations), then kw from the A–C line, then t0 pin vs free, then the tan_max scan, then seeding (Q4 runs
    meanwhile: its conclusion does not depend on the angle scale).
- **v in situ from the column duration (11-02).** Box fit (`colsum.GridFit`, the pass-1 X template) to the median
  stacked strip sum, stacked on the trigger (jitter ~13 ns); bootstrap errors.
  - **A–C line muons (true full-gap through-goers, run_149 cos_0000):** A 38.4 ± 0.7 (x) / 38.9 ± 0.8 (y); C 27.7 ± 0.7
    / 28.6 ± 0.7 µm/ns. This agrees with is2's geometric in-situ v (A 38.0, C 28.7) by an independent, timing-only
    route.
  - **All clean single-cluster events (two cosmic sub-runs, and run_145 beam):** A 39.4–40.0, C 27.3–28.5, D 29.9–32.0
    (cosmics). B is not measurable (its column is not a box). D on beam fails again (box start at the grid edge), so D's
    beam events need their own look.
  - **Adopted for the v1 bundles:** A 38.6 (line muons; the all-cluster selection reads 3 % faster, a selection
    systematic), C 28.1, D 31.0 (± gap 29.44 ± 1.12 mm, i.e. ± 4 %; the old D-wall estimate was ~28), B 42.6 prior
    (no angle).
  - Gaps are the bench timing-free gaps (`physics_inputs.json`), so v carries their errors: A ± 0.7 %, D ± 3.8 %, C has
    no error (adopted 28 mm).
- **X/Y pairing fix on main (743d6d31, Paper I session, 11-02).** `wft.reco.select_tracks` now has
  `xy_pairing['select'] = 'cost'`: pair by the lowest summed `xy_pair_cost`, then gate. The old `'gate'` mode let the
  beam-tuned `plausible` flag decide which pairs survive (80 % correct on bench two-particle cosmics, `cost` 96.7 %).
  - `ntof_bundle.py` writes `'gate'` explicitly, so n_TOF output is unchanged for now.
  - On run_145, `cost` would change 12–23 % of events and lose the gated track in 2–3 %.
  - The same `plausible` flag also gates the joint two-track fit (`_cross_plane_mismatch`, the `both_plausible` split
    guard). On the bench it blocks nearly every recovery of two particles merged in one projection
    (`rerun/effbudget/single/MERGED_PROJECTION_2026-11-02.md`, in progress).
  - Notes: `rerun/effbudget/single/PAIRING_FIX_2026-11-02.md`, `rerun/ntof/NOTE_PAIRING_FIX_2026-11-02.md` (coming).
  - **Decision before the n_TOF bundles freeze (Q7):** `select` gate vs cost, and the two-track plausibility guards,
    chosen on a beam truth test: overlay pairs (`sept26_prelim_analysis/intra_bench.py`, real single beam tracks
    superimposed), plus capsule pointing. `plausible` contains tan_max, so the tan_max scan and Q7 are one study and
    are scored together.
- **Q4 seeding result (11-02, `/media/ucla/work/ntof_pass1/q4/`; same triggers for every arm: run_145 `_000` ids
  1–2000, run_149 cos_0000 ids 1–5000; tm0.6 v0 bundles).**
  - **The old n_TOF hits ARE the rr1 recipe:** run_145 `_000` old vs regenerated hits are identical field by field
    (1,908,783 hits), and so are the reconstructions. So "old vs rr1" only differs on the truncated/fused acquisitions.
    The pass uses `combined_hits_root_rr1_cm0` from the fa80c06 decodes for one provenance.
  - **Waveform-native seeding** (5σ peak or 3-sample sum) vs hits, on beam:
    - A: 491 vs 335 paired tracks, but 132 vs 124 pointing to the capsule (27 % vs 37 % pointing).
    - C: 353 vs 326 paired tracks, 101 vs 104 pointing (29 % vs 32 %).
    The extra tracks are almost all non-pointing. CPU is 2–3× (C 22 vs 8 CPU-s per event).
  - On cosmics there are too few A–C lines in 5,000 triggers (30–34) to separate the two.
  - **Decision: seed from hits** (the analyzer's matched filter). A waveform threshold scan (6–7σ) could narrow the
    gap, but nothing here suggests it would beat hits; not pursued unless the hits fail somewhere.
  - Timing noise: identical work (old vs rr1) ran 3.1–3.9 and 7.6–9.1 CPU-s per event, so only differences > ~25 %
    mean anything.
- **kw in situ (11-02).** v1 bundles (timing v) on the cos_0000 A–C line: the ratio fitted/true tan is A 0.947 ± 0.021
  (x) / 0.958 ± 0.010 (y), C 0.942 ± 0.021 / 0.934 ± 0.014. So **in-situ kw = A 0.982 / 0.973, C 1.013 / 1.008**:
  with the column-timing v, kw ≈ 1 at n_TOF, and the bench kw (1.02–1.08) does not carry. Held-out check on
  cos_0001, and the pinned-t0 retest on these bundles (`v1k`, `v1kpin`), running.
- **Held-out check on run_149 cos_0001 (11-02; kw measured on cos_0000; 3,455 / 2,990 events, 161–169 A–C lines).**
  - **kw closes:** the fitted/true scale on the line is A 1.003 ± 0.020 (x) / 0.994 ± 0.011 (y), C 1.000 ± 0.014 /
    1.007 ± 0.014. The angle scale at n_TOF is set by the timing v with kw ≈ 1 (A 0.98/0.97, C 1.01/1.01).
  - **Pinned t0 (σ 15 ns, cos_0000 latency) vs free, same bundles:**
    - 169 vs 161 line events.
    - On the 158 common events: A residual-angle 68 % 3.26° vs 3.56° (A x tan 0.0367 vs 0.0429, −14 %; A y 0.0349
      vs 0.0335), C 2.93° vs 2.94° (unchanged).
    - Under the pin the scale moves by −1…−2 % (A 0.984/0.977), so a pinned pass needs its kw measured under the pin.
    - **Mildly in favour of pinning (A), neutral for C;** with ~160 lines that is not yet significant.
    - **Next:** all 12 staged run_149 sub-runs (~2,000 lines), kw re-measured under the pin, then a beam-side check
      (beam trigger, capsule pointing) before deciding.
- **Remaining before the bundles freeze:** the pin decision (above); the tan_max scan with Q7 (pairing select +
  two-track guards) on beam overlay pairs and capsule pointing; D's beam events (column start at the grid edge); B and
  D v/kw routes (D wall, cosmics without a line); Dp for 90/10; the template residual check; sat 3550 vs 3700.

### 11-04: the pre-freeze studies (one session, four parallel studies + condor)
- **Inputs for the pass are complete.** Campaign rr1_cm0 hits on the fa80c06 decodes: 3,937 / 3,937 acquisitions
  (condor 4616842, `rerun/ntof/rehit/rehit_pass1.sub`, `CM_MODES=0`; the per-FEU + combined hits are ~184 GB on EOS for
  runs ≥ 100). Every acquisition in scope now has `decoded_root_fa80c06/` and `combined_hits_root_rr1_cm0/`.
- **P6 tooling exists and is validated** (`rerun/ntof/condor/`): one job per (acquisition, arm); the job stages only the
  arm's two FEU files from `decoded_root_fa80c06` (as `decoded_root`) and the rr1_cm0 combined hits (as
  `combined_hits_root`), runs `ntof_reco.py --tag` from a frozen code archive with the lock venv, and writes
  `run_<r>_<sub>_<tag>_<arm>.{parquet,candidates,splits,meta}` to EOS. Full scope = 15,748 jobs. Dry run (code
  `89478e52`, run_145 + run_149, A + C, 300 events per tag): 4 / 4 ok, and the condor output is **bit-identical** to the
  desktop on the same acquisition (582 / 582 candidates, every field). Caveat: the χ²/dof quality-flag median is per
  table, so per acquisition (~10³ tracks) rather than per sub-run. Known flake: EOS FUSE I/O errors loading the lock
  venv's scipy `.so` files on a few jobs (Q7 cluster); `max_retries 2` covers it, and the ledger must check presence.
- **P2 channel layer done** (`rerun/ntof/channel_layer/`, README + report.html): eight masks, (A, B, C, D) × (E1 = before
  the 27 July access, E2 = after); registry column `mask_era`; `ntof_bundle.py --mask-era`. Dead x/y E2: A 1/35,
  B 8/24, C 38/21, D 154/61 (+ D hot 20/21). The bench lists carry poorly (A's bench mask is bench-electronics; C-x
  210–219 fire at n_TOF and are released). New: A-y connector 8 gains 23 shorts after the access; D's "hot cells" are
  beam-induced (wildcard, not censored). **Tested:** on the A–C line the mask is neutral (Δ68 −0.02 ± 0.03° A,
  +0.02 ± 0.04° C, 15 of 1,909 lines lost); on beam (run_145) it removes 4.4 % of A's tracks, almost all non-pointing
  (pointing fraction 32.9 → 34.0 %, pointing tracks −1 %); C neutral.
- **Q2 decided: pin t0, latency per trigger class** (`rerun/ntof/tests/t0pin/`; 12 run_149 sub-runs, A∧C hit-coincident
  triggers only, 100 % of the earlier line events kept; kw measured on even sub-runs, scored on odd).
  - Cosmic A–C line, ~1,850 lines (was ~160), held-out at each config's own scale, pinned vs free:
    A 68 % residual 2.88 → 2.62° (−0.25 ± 0.08°), A-x tan −11 %; C 3.33 → 3.23° (−0.10 ± 0.09°); +6 % line events.
  - **The beam trigger has its own latency:** free-fit prompt t0 + 10·ftst on run_145 sits +11 / +19 ns (A x/y) and
    +17 / +26 ns (C) after the cosmic one. Beam latency files `bundles/latency_{A,C}_r145beam.json` (A 6.9 / 20.0,
    C 52.4 / 52.5 ns). The ftst slope on beam is ~−7 ns per class, as on cosmics (the pin assumes −10: open, ≤ 15 ns
    at ftst 5).
  - Capsule pointing on the same run_145 tracks (matched by position): pinned (beam latency) vs free, paired DCA
    −0.006 ± 0.008 mm (A), −0.06 ± 0.02 mm (C); pinned finds 3–5 % more pointing tracks; the pointing *fraction* is
    lower because the pin also gates more non-pointing tracks. Beam latency beats cosmic latency on C (−0.075 ±
    0.026 mm paired).
  - So: pinned, σ 15 ns; beam rows take the beam latency, cosmic rows the cosmic one (bundle per trigger class).
- **Model checks** (`rerun/ntof/tests/model_checks/FINDINGS.md`): template holds (A, C within 0.02 of peak of the
  bench; D's peak excess is lateral), systematic 0.1° / 0.15 mm. **sat stays 3550** (the rail is 4095 − ped − CM, ~4080
  on beam, ~3650–3750 on cosmics; 3700 misses more clips and buys back ~25 %; the per-sample rail is the real fix —
  diff proposed for `wft_beam._windows_for_tag`, not applied). **Dp 90/10:** A 0.0245 ± 0.0014 (stack; bench ×1.2 det3
  excess), C 0.0136 (Magboltz; the old 0.018 was never measured), D 0.0157 (Magboltz); `ntof_bundle.py --dp`.
- **v2 candidate bundles (A, C):** pin + E2 mask + new Dp + kw carried under the pin. On the line, against free:
  A −0.24 ± 0.07°, C −0.12 ± 0.11°; scale ratio train 0.996 / 1.000 (A), 1.008 / 1.009 (C): one more kw step
  (A 0.976 / 0.965, C 1.009 / 1.000) is < 1 %, i.e. converged. C-y train vs test differs by ~1.5 % (0.992 vs 1.007
  under v1k; 1.009 vs 1.016 under v2): sub-run to sub-run variation of C, consistent with C's drifting v; a per-sub-run v
  for C is the open item.
- **D and B** (`rerun/ntof/tests/d_b/FINDINGS.md`): **kw_D = 0.880 ± 0.014 (x) / 0.881 ± 0.010 (y), ± 0.03 syst** at
  v 31.0, from B–D cosmic lines through cluster centroids (validated on A–C to ~1 %); D's timing v and angle scale differ
  by 12 % (kw·v ≈ 27.3). **D on beam carries a ~360 ns (2.8 MHz) beam-gated oscillation** on 82–90 % of triggers
  (from ~10 ms after the flash), which is why the beam box fit fails; it is not a window or latency offset. D's beam v is
  not measurable and its beam angles are limited (stated limitation; a hum flag / sinusoid subtraction is the only
  route, untested). B: positions only; its v moves no position (≤ 1.2 mm at the 95th pct via the bow term) but changes
  B's efficiency through the raw-tan gate.
- **Q7 decided (11-05, `rerun/ntof/tests/q7/FINDINGS.md`, report.html): `select='cost'`, `split_corrob='charge'`,
  `plausible` keeps the raw tan_max, raw tan_max 0.6 confirmed.** Beam overlay truth (A 695 / C 783 overlays, steep
  donors included): both members found and correctly paired 26 → 61 % (A), 25 → 57 % (C); accepted ghosts per overlay
  0.24 → 0.11; the shallow member of a shallow+steep pair 36 → 71 / 68 %. Single tracks, the A–C line (1,647 → 1,645
  lines, same residuals) and the capsule-pointing excess (A 1463 ± 69 → 1446, C 1206 ± 61 → 1201) do not move. With
  cost + charge the in-acceptance efficiency is flat in tan_max; tracks above 0.6 carry no pointing excess; 1.0 and
  column-window only add line mis-fits and +30–60 % CPU. Applied in `ntof_bundle.py` (explicit, with provenance). One
  left-out acquisition (C `_006`, column window) resubmitted as condor 4617508; it cannot change the decision.
- **Remaining before the bundles freeze:** build the pass bundles: (A, B, C, D) × (E1, E2) ×
  (beam, cosmic latency) with v / kw / Dp above, ~~D t0 free~~ D pinned (11-05 below), B no angle; then N-G1 on
  run_145 + run_149 against is2_v2, and the full submission.

### 11-05: D pinned like A and C; v(t) becomes a pass product (Dylan 11-05)
- **Dylan: one t0 treatment for A, C and D unless there is a reason not to.** D was left free only because its beam
  latency is not measurable. Checks: `rerun/ntof/tests/d_b/dpin_run.sh` (reco) + `dpin_score.py` (scores); bundles
  `d_b/bundles/d2_D` (free), `d2pin_D` (cosmic latency), `d2pinb_D` (cosmic + borrowed beam offset), all v 31.0,
  kw 0.880/0.881, Dp 0.0157, E2 mask, tan_max 0.6, cost + charge. Output `/media/ucla/work/ntof_pass1/d_b/reco_pin/`.
- **D cosmic latency measured** (run_149, 12 sub-runs, same rule as A/C: iterated median of free t0 + 10·ftst on single
  quality tracks): **x +18.3 ± 3.4, y +20.0 ± 2.2 ns** (n ≈ 1,050; ftst slope −8.8 / −5.8). `bundles/latency_D_r149.json`.
- **D beam latency is not measurable by the free fit** (run_145: +65 / +150 ns, 110–125 ns wide, ftst slope +2: the
  oscillation). Borrowing A/C's beam−cosmic offset (+14.3 / +22.5 ns) gives `bundles/latency_D_r145beam.json`
  (x 32.6, y 42.5) — but **D's own-trigger late beam tracks pull the pinned t0 to −12 / −15 ns below it** (and only
  −4 / −7 ns below the cosmic latency), i.e. D's beam data sit near D's cosmic latency, not cosmic + 18.
- **Beam (run_145 `_000`, 1,500 triggers), free → pinned (borrowed beam lat.) → pinned (cosmic lat.):**
  gated tracks 463 → 512 → 524; pointing < 15 mm 68 → 91 → 99 (frac 14.7 → 17.8 → 18.9 %); < 10 mm 50 → 61 → 70.
  On the same tracks the DCA 68 % goes 15.8 → 13.1 mm (n < 10 mm 34 → 38 / 36 → 40); ok / quality fractions and χ²/dof
  unchanged (own-trigger late χ²/dof x 6.7 → 5.6 / 5.5). **The pin helps D on beam**, and the cosmic latency does at
  least as well as the borrowed offset.
- **Cosmic kw under the pin (B–D centroid lines, strict B, miss < 60 mm):** free d2 0.983 ± 0.018 (x) / 1.009 ± 0.012
  (y) — kw_D 0.880/0.881 still closes on the new bundle (new mask, Dp, cost + charge) — and pinned 0.970 ± 0.014 /
  0.990 ± 0.018, the same −1…−2 % shift as A. **kw_D under the pin = 0.854 (x) / 0.872 (y)**, ± 0.03 syst unchanged.
  Line residuals pinned vs free on the same lines: B–D 68 % tan 0.0343 → 0.0362 (x), 0.0349 → 0.0331 (y); A–D
  0.0574 → 0.0540, 0.0635 → 0.0615: neutral to slightly better.
- **Decision (proposed): D pinned, σ 15 ns, with D's measured cosmic latency on BOTH trigger classes** (beam data prefer
  it; the borrowed offset is not a measurement). kw_D 0.854 / 0.872. Same σ and rule as A/C; the only difference is that
  D's beam latency = its cosmic latency, stated with the evidence above. B stays positions-only, t0 free.
- Not checked: one more kw iteration on the final D bundle (A converged at < 1 % on the next step); E1-era D.
- **v(t):** added as P7a (§3). A/C from the pass output's column duration per sub-run / 30-min bin; D from the beam-off
  cosmic runs; joined in stage 3 as `tan_vt` beside `tan`.
