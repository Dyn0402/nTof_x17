# Reprocessing to-do (bench + SPS from-scratch pass) — started 2026-10-28

The single checklist for the from-scratch reprocessing. **Dylan (10-28): before the pass, every calibration is redone
from scratch — no shortcuts, no reuse of intermediate products from the audit.** `RERUN_PLAN.md` holds the plan and
its history (§11 = the pass machinery, §12 = blockers), `model_audit/FOLLOWUPS.md` the audit answers (round 3 outcomes
table), the reports `model_audit/followups/*/REPORT.md`. Tick items here; record the evidence in the item.

Status key: [x] done · [ ] open · [~] partly / in progress

## A. Decided and in the code (main, 2026-10-28)
- [x] `fe` recipe = the one reconstruction: pinned t0 (bench), one σ0 = 0.30 mm both views, chamber bow block, one-rule
      dead lists, band censoring + re-seed, column-time x–y gate, per-class pairing (`fe_unify2/REPORT.md`; GF: `fe`
      over `fefa`). Not "promoted": the pass rebuilds every bundle with this recipe.
- [x] χ²/dof quality flag = 5 × the table median per view (`wft.reco.CHI2DOF_REL`, `quality_flags`; GE).
- [x] DREAM `timestamp` + `acq_tag` in every event row, `acq_tag` in candidates (GC). Validated end to end on det3.
- [x] Own v (w0/kw) for every registry row with ≥ 500 clean triggers; smaller rows inherit the nearest-in-time
      calibrated row of their group (`make_registry.py`, `pass_registry.csv`: 117 own-v, 37 inherit; GC).
- [x] Spark veto stays > 50 seeding hits (GD; 40 is the purity-side option); seed constants read at call time.
- [x] Gain map: not used (slot 1.0; GA). M3 reference term 0.05° (GB): bench angles quoted as the chamber's own.

## B. Before the pass (blocking)

> **2026-10-28 evening:** the run sheet is `PASS_PLAN_2026-10-28.md`. Dylan's decisions: the calibration is rebuilt
> in the DAG templates included (new `calib` phase, E1–E3); bow on all five chambers, refit by the same recipe; LM
> merged only if its gate passes; trigger-track rule downstream. Items found by the history review (F7–F12, E1–E5,
> `history/LESSONS.md` "found today") are listed there. Done this session: SPS bench bow/v_map/w0 dropped, SPS 15°
> mount tangent, stale ZS_CENSOR texts, CLAUDE.md / RECONSTRUCTION_BASIS / RERUN_PLAN §11 wording.
1. [ ] **GD det4 row**: lxplus job 4583837 (started 10-28 09:18). When done: `model_audit/followups/gd_sparks/pull.sh det4`,
       `gd_score.py`; confirm det4 does not change the verdict.
2. [ ] **Double-check the 14 GA bad channels** (`model_audit/followups/fe_unify2/masks/bench_det{3,4}.json`, key
       `ga_added`): det4 X weak 94 125 196 376 400 435 439 455 472 498 → dead, det4 X 274 hot (beside the chamber-side
       dead run 271–273: short partner?), det3 X 62 hot, det3 Y 41 / 510 → dead. Flank test on the hits + a waveform
       display each; drop any that fail.
3. [ ] **Chamber bow**: the separate skeptical review (HANDOFF "Chamber bow (FB): skeptical review") — its verdict on
       det6 (reco bias) and det7 X (unmasked) decides the bow block content per chamber.
4. [ ] **B1 remainder** (`RERUN_PLAN.md` §12): the LM variable-projection outer solver (HANDOFF "Fitter: LM ...",
       worktree `lm-varpro`), gated on the golden rows; retire the remaining `WFT_*` switches into one algorithm;
       Dylan's calls: the trigger-track rule (reco or downstream) and `PIVOT_START` built in.
5. [ ] **B2 fleet gate** on the final code, all five golden rows (s68, far, χ², efficiency, held out); no chamber worse
       or the cost written down.
6. [ ] **B3 SPS** in the pass: `rerun/sps_registry.csv` + `model_audit/sps_reco.py` / `sps_stage.sh` (wiring done);
       the SPS stage must also write the timestamp and `quality_rule` (check `sps_reco.py` uses `wft.reco` writers).
7. [ ] **Pass DAG check**: `make_pass_dag.py vrefit` now gets ~117 own-v rows (was 14 groups): dry-run, size the condor
       load (one reco of ≤ 8000 events + align + angles + W0KW each), check rows with < 8000 events and the
       `--keep-v --fold-kw` path; the `pass` phase's own-alignment branch for every own-v row.
8. [ ] **Code archive** (`code_<hash>` on EOS) of the final main; regenerate `pass_registry.csv`
       (`python rerun/make_registry.py`; reads `model_audit/followups/gc_gas_time/results/bench_rows.json` for row
       times and clean-trigger counts).
9. [ ] **From scratch**: golden finalize (S1) → vrefit (S1b, every own-v row) → pass (S2+S3), `RERUN_PLAN.md` §11
       order. `lxstore status` on lxplus before and after.

## C. After the pass (table-level, no refit)
- [ ] v(t) within a row: per-time-block kw from the M3 ladder (bench; not the column duration, which is gain-confounded),
      column end on n_TOF arms A/C; join on (run, subrun, acq_tag, event_id). Largest: det6 golden ~11 %, det4 ~7 %,
      det4 6-23 short_run (inherits longer_run, ~9 % apart). n_TOF B/D: no within-run route (limitation). GC.
- [ ] SPS gas fit (H₂O + O₂) with Magboltz (FOLLOWUPS item 9).
- [ ] Noise-band hotspot census, all chambers (item 13).
- [ ] FD budget update with GA (gain row withdrawn) and GB (M3 0.05°); re-quote resolutions (also the INFN–IN2P3
      abstract numbers).
- [ ] n_TOF: the χ²/dof flag's purity at n_TOF is unmeasured (~10 % flagged); check with x–y charge balance or
      inter-chamber pointing before an analysis cuts on it. Busy-plane multi-particle-like planes not refitted (GD).

## D. Stated limitations (no new data)
- det4 dead stripes and det7 X (M1) tails: χ² blind, gain map cannot recover (GA, GE).
- SPS within-plateau v not measured (GC); n_TOF B/D within-run v not measured.
- det4 has no partner chamber: its M3 term rests on simulation (GB).
