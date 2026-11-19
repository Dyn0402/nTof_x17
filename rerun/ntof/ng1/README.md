# N-G1 gate on n_TOF pass 1 (2026-11-16)

**Verdict: no arm fails, none is held.** A passes every criterion. C passes, with two conditions: its raw angles
need the P7a v(t) scale, and its source y sits 3 mm above is2_v2's. B and D have no is2_v2 reference (is2_v2 had A and
C only), so N-G1 cannot pass or fail them. G5 is quantified, not closed: A +8.1 ± 1.1 %, C +4.5 ± 1.2 %, down from
+12.6 / +6.0 % for is2_v2 on the same run.

Criteria: `../NTOF_PASS_PLAN_2026-11-02.md` §3 "N-G1 gate". Pass 1 = EOS `Analysis_pass1/stage2/` (code ed0b1162,
bundles frozen 11-07), run_145 beam (24 acquisitions, A B C D) and run_149 cosmic (87 sub-runs). |tan| < 0.6 on both
chains throughout. **D's beam rows are on the L1 pin** (rerun_dbeam_L1, condor 4677439; rescored 2026-11-19: 28.8 → 29.0 % gated/trigger,
5.61 → 5.63 % confirmed/trigger on the L2 → L1 pin). `wall` caches `results/wall_<chain>_<arm>.parquet`: delete an arm's
file when its inputs change.

| arm | C1 efficiency + wall | C2 A–C line s68 | C3 source y | G5 focus vs muons |
|---|---|---|---|---|
| A | **PASS**: gated/trigger +5.6 pt, wall-confirmed/trigger +2.8 pt; same tracks −0.05 ± 0.11 pt | **PASS**: rescaled Δs68 x −0.17 ± 0.08°, y −0.05 ± 0.07°; raw −0.07 / +0.02° | **PASS**: −31.6 ± 0.4 mm (−1.6σ) | 1.073 ± 0.011 × 1.007 → **+8.1 ± 1.1 %** |
| C | **PASS**, same-track purity at the tolerance: +5.7 pt, +2.6 pt; same tracks −0.58 ± 0.17 pt | **PASS on resolution**: rescaled x −0.14 ± 0.13°, y −0.23 ± 0.13°; raw +0.23 / +0.10° (C reads 4–5 % steep: P7a) | **PASS within the ±3 mm estimator systematic**: −33.2 ± 0.4 (+4.0σ stat vs −35); +3.0 mm vs is2_v2 on the same events | 1.006 ± 0.012 × 1.038 → **+4.5 ± 1.2 %** |
| B | n/a: 8.4 % gated/trigger, 2.2 % confirmed/trigger, purity 0.26 | n/a (positions only) | n/a | n/a |
| D | n/a: 29.0 % gated/trigger, 5.6 % confirmed/trigger, purity 0.19 (L1 pin) | n/a | n/a | n/a |

Notes on the numbers:
- **C1.** One common trigger set per sub-run. The pass confirms less often per gated track (A 0.544 vs 0.563, C 0.570
  vs 0.636), and that is all composition. On the same single-track events both chains confirm equally. The ~11k extra
  tracks per arm that only the pass has are 46–48 % confirmed, against a ~1 % control. On the cosmic truth events both
  chains fit every plane. The quality-level gap there (A 0.91 vs 0.98) comes from the reference: the quality rule is
  relative to the table's median χ²/dof, and is2ts_s3 ran on a truth-only allowlist, which makes its cut up to 2–3×
  looser on C.
- **C2.** Paired on the 14-sub-run in-situ truth (`ntof_cosmics/insitu_calib.py`) against `is2ts_s3`, which uses
  is2_v2's stage-2 settings. Over all of run_149 the pass's own line gives 3D s68 A 2.87 ± 0.04°, C 3.78 ± 0.04°, and
  scale fit/true A 1.007 / 1.009, C 1.038 / 1.041.
- **C3.** `y_guard.late_y_crossing` (≥ 20 ms). C's shift persists on the same events, so it comes from C's
  reconstruction, not from the added tracks. It brings C toward A: A − C is 1.7 mm on the pass against 3.8 mm on is2_v2.
- **G5.** run_145, 20–80 ms, x view, source dx −9.5 mm, |tan x| 0.1–0.5. Tension = focus s × the A–C muon fit/true.

What it does not rule out:
- Only one beam run and one cosmic run are scored here.
- The wall says a track is real, not that its angle is right (G6).
- Whether −33.2 or −35.8 is closer to C's true source y is not decided.
- G5's remaining 8 % on A rests on Geant4's ±3 % focus-bias estimate with the simulated source.

Files: `ng1_score.py build|eff|wall|cos|line|srcy|focus|all` (work dir `$WORK_DATA/ntof_pass1/ng1_pass`: EOS copy in
`pass/`, scoring track tables in `s3/`), `make_report.py` → `report.html` (copied here with `figures/` and the small
`results/` tables).
