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
| R12 | Bench bow, v_map, w0, t0_abs never at n_TOF; `reconstruct_subrun` refuses a bundle that carries them | keep the refusal; the n_TOF bundle is built without them |
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
    the bench.
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

**P8. Docs**
- `docs/PARAMETERS.md` n_TOF section.
- n_TOF lessons into `rerun/history/timeline_data.py` (regenerate LESSONS.md).
- CLAUDE.md: replace "n_TOF out of scope" with the new pass.
- One generated `report.html` for the pass.

---

## 4. Known limitations to state, not to chase
- **B:** no field-shaping ring, so no angle. Positions only, and ~13 % cosmic efficiency.
- **B/D:** no within-run v route unless their column duration proves usable.
- **Template:** not measurable on the 20-sample window. It is carried from the bench (P3 check).
- **Window:** 1.2 µs, so a column slower than ~24 µm/ns would be cut (C is closest).
- **Busy planes:** multi-particle planes are not refitted (R16).
- **Gas:** n_TOF gas is 90/10 and only Magboltz + in-situ v anchor it; there is no SPS-style gas fit at n_TOF.

---

## 5. Decisions needed (Q1–Q6)
- **Q1, scope:** production 36 runs + 41 beam-off cosmic runs? Include pre-production (run_55 HV scan, drift scans,
  run_75/76)?
- **Q2, t0:** pin it to the trigger latency, as on the bench, or leave it free, as at SPS? The n_TOF trigger is the
  scintillator stack, like the bench, so the latency should be fixed up to the ftst phase. But a non-trigger second
  track (pairs, pile-up, late tracks) has its own t0. Proposal: pin the trigger track and leave the second track of a
  two-track fit free. Decide on P5 numbers.
- **Q3, output location:** `/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/` (next to the data, like the bench).
- **Q4, hits:** regenerate n_TOF hits with the rr1 recipe, or seed from waveforms as SPS does?
- **Q5, saturation:** one clip rule (bench 3550) or keep 3700 at n_TOF? The ADC is the same DREAM. 3550 is a
  model-mismodelling rule, not the rail, so it should transfer.
- **Q6, scale of the A/B:** the full is2_v2 comparison (P5 test 3) on all 5 test samples, or run_145 + run_149 only?
