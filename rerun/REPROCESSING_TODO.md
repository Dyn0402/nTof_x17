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
3. [x] **Chamber bow** — decided (Dylan 10-28): bow block on all five chambers, refit inside the new `calib` phase by
       the same FB recipe (det6 included, det7 X unmasked). Paper wording follows the skeptic review (1.2–1.9 ± 0.4 mm):
       REPORT.md / `08b_bow.tex` / slide note still to edit (downstream, not blocking).
4. [~] **B1 remainder**: switches already retired into `wft/diag.py` (10-27; stage_job refuses overrides);
       PIVOT_START is built in (reco.py:86); trigger-track rule = **downstream selection** (Dylan 10-28). Left: the LM
       fitter — merge `lm-varpro` only if its fleet gate passes (condor 4591065–74, DAG `lm_r1`, started 15:09; 6/10
       done at 16:20; `lm_gate_pairs.py` in the worktree), then `FITTER='lm'`, `LM_HOPS=0` as code defaults (PASS_PLAN E5).
5. [ ] **E1 bundle from nothing**: `rerun/calib/make_bundle.py` + `rerun/calib/physics_inputs.json` (σ0 0.30, Dp,
       det4 D_rc,y 2.2e-4, gaps, timing v; provenance) — PASS_PLAN §2.
6. [ ] **E2 bow node**: `rerun/calib/bow_node.py` (FB `set_bow` reading EOS tables + M3 + alignment, not the laptop mirror).
7. [ ] **E3 `make_pass_dag.py calib` phase** CAL0–CAL4 per golden chamber into `Analysis_pass1`, convergence
       criteria → `CALIB_CONVERGENCE.json`; **`--table` on every W0KW node** (finalize/vrefit/pass, make_pass_dag.py:89,
       112, 130 — without it they abort on a bow bundle, set_w0kw.py:95–97); vrefit keeps the golden bow.
8. [ ] **E4 SPS phase** (was B3): one node per `sps_registry.csv` row with the new det4 bundle (bench bow/v_map/w0/t0
       dropped — done 10-28 in sps_reco), `quality_flags`/timestamp in `sps_reco`, K fallback +5 → +15.
9. [ ] **Pre-flight checks** (PASS_PLAN F7–F12): the 14 GA channels (item 2); H4 ch 372 oscillation → `det4_sps_h4.json`;
       the 10 rows without gas-time entries (det2 6-17 ×8, det3 6-23 short, det3 6-26 quick); det4 6-23 rows' M3
       clean fraction (skeptical if < ~40 %); g_det4 seeding on FEU 6 conn 7–8; optional S0b: decode + M3 + hits of
       the never-decoded det6/det7 6-26 points 455/460/470/475/490 V (raw in EOS `june_tests/`).
10. [ ] **Dry run**: `calib` on det3 `--limit 300` + one vrefit and one pass row on its output; pytest wft/tests.
11. [ ] **Code archive** (`code_<hash>` on EOS) of the final main; `lxstore status` (AFS 88 % on 10-28: offload old
       `code_*` first); regenerate `pass_registry.csv` (`python rerun/make_registry.py`).
12. [ ] **Submit**: `calib` (S1) → gate G1 (convergence + held-out vs the audit fe numbers, `gate_compare.py`) →
       `vrefit --keep-v` (117 rows) → `pass` (159) + `sps` (71). Order and gates: PASS_PLAN §3.

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
