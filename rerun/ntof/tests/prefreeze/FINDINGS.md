# Pre-freeze scoring of the n_TOF pass-1 bundles — findings (2026-11-06)

Code `ed0b1162` (archive `~/x17rerun/code_ed0b1162` on lxplus). Outputs on EOS
`july_beam/Analysis_pass1/tests/prefreeze_ed0b1162/{lat,cospin,samples,ng1,twosided,samples_dL1,bundles}`.
Scorer `pf_score.py {pull,lat,line,beam,all}`; tables `$WORK_DATA/ntof_pass1/prefreeze/results/`
(`latency_by_run.csv`, `line_by_run.csv`, `beam_by_run.csv`). On the laptop (no ucla) the scorer uses `~/work_data/`.
Plan and decisions: `rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md` §5–§6.

**Status: every set is complete except the run_79/81 beam samples** (23 acquisition×arm jobs, A/C/D, need > 8 h each;
re-submitted on the 24 h flavour as 4634655) **and the D-L1 beam check** (3/25 done, 4620578). The numbers below for
runs 79/81 are partial; nothing else will change.

## 1. Latency per run (free t0, all quality single tracks, dithered iterated median of t0 + 10·ftst)

| arm | E1/L1 (runs 80–103) | L2 (runs 105–159) | note |
|---|---|---|---|
| A x / y | −5.4…−2.5 / 0.5…5.0 | −0.4…+10.5 / 5.2…18.1 | +5–10 ns at the 07-29→07-30 step, back by run_133; inside the 15 ns pin |
| C x / y | 16.1…25.4 / 6.2…11.8 | 22.6…33.9 / 9.4…27.4 | run_109 (1 acq.) 48 / 33 ± 6–9; same small step as A |
| D x / y | −18.3…+4.4 / −9.6…+16.4 | +18.6…+57.5 / +24.3…+62.6 | **~40 ns step between run_103 (07-29 14:25) and run_105 (07-30 11:49)**; run_133 D y 115 (outlier) |

→ D gets two latencies (registry `lat_epoch`, decided 11-06): L1 x −17.8 ± 5.4, y −11.2 ± 4.4 ns
(`bundles/latency_D_L1.json`), L2 keeps the r149 pin x +18.3, y +20.0. A and C: one latency each.

Selection dependence: at run_149 this estimator reads A +2/+4, C −9/−6, D +11/+18 ns from the bundle pins (measured
on line/wall candidates). Not followed up; inside the 15 ns prior.

The t0 lattice lock (20–65 % of fitted t0 exactly on the absolute 10 ns grid; template tabulated on 10 ns, `np.interp`)
is a model artefact with no geometry effect (tan 2–5e-4 at 68 %); reco left untouched for the freeze, estimator dithered.

## 2. A–C line (angle scale and resolution)

- run_149, t0 pinned (`ng1`, 1,896 lines): A res68 2.67 ± 0.08°, C 3.30 ± 0.10°; ratios A 1.002/0.995, C 0.995/1.005.
- Per run (`cospin`, ~300 lines each): A's scale 0.988–1.026; **C's 0.994–1.086** (run_89 1.083/1.086, run_133
  0.994/1.001, run_159 1.058/1.051), errors ~0.01–0.015. C res68 follows it (2.9–4.5°). Same pattern with free t0
  (`lat`). Reads as v(t), the P7a product; **open for Dylan: does it block the freeze or wait for P7a?**
- Mirror test (two-sided vs one-sided, run_149): identical below 31°; 31–40° A 2.91 → 2.93°, C 3.42 → 3.64° (+7 tracks
  each). The A–C line has no tracks above 40°: the test can't reach where a mirror basin would matter (limitation).

## 3. Beam vs is2_v2 on the same triggers (A, C)

| run | arm | acq | gated tracks / trigger | pointing < 15 mm (n) | frac pointing |
|---|---|---|---|---|---|
| 79* | A | 8 | 0.266 → 0.329 (+24 %) | 3081 → 4193 (+36 %) | 0.327 → 0.359 |
| 79* | C | 10 | 0.231 → 0.275 (+19 %) | 3025 → 4024 (+33 %) | 0.331 → 0.369 |
| 81* | A | 9 | 0.259 → 0.322 (+25 %) | 3300 → 4411 (+34 %) | 0.335 → 0.359 |
| 81* | C | 3 | 0.235 → 0.276 (+17 %) | 673 → 915 (+36 %) | 0.323 → 0.374 |
| 116 | A | 13 | 0.352 → 0.393 (+12 %) | 4795 → 6327 (+32 %) | 0.313 → 0.370 |
| 116 | C | 13 | 0.270 → 0.328 (+21 %) | 3526 → 4881 (+38 %) | 0.310 → 0.354 |
| 162 | A | 12 | 0.345 → 0.388 (+12 %) | 5044 → 6474 (+28 %) | 0.320 → 0.365 |
| 162 | C | 12 | 0.260 → 0.320 (+23 %) | 3482 → 4800 (+38 %) | 0.308 → 0.344 |
| 145 | A | 24 | 0.358 → 0.393 (+10 %) | 9595 → 12382 (+29 %) | 0.306 → 0.359 |
| 145 | C | 24 | 0.258 → 0.323 (+25 %) | 6577 → 9824 (+49 %) | 0.298 → 0.356 |

\* partial (run_79/81 samples still running). B and D have no is2_v2 reference: D frac pointing 0.12–0.16, B 0.07–0.10.

## 4. D L1 pin on beam (the check that the cosmic L1 value transfers)

Same acquisitions, D, L2 (r149) pin → L1 pin: run_79 (2 acq., 11,153 triggers) pointing < 15 mm 450 → 476,
< 10 mm 313 → 347, median DCA 90.2 → 89.7 mm; run_81 (1 acq.) 117 → 129, 81 → 84, DCA 86.0 → 85.9 mm.
**No worse with L1 (slightly better) on 3/25 acquisitions**: pass criterion met so far; final when 4620578 completes.

## 5. Job accounting (rule: a failure rate is a bug to explain)

- 4618426–30 (782 jobs): 37 without output. **32 of them were killed at the 8 h `workday` wall limit** (28 in samples,
  4 in ng1; condor `009` "wall time exceeded"), not EOS I/O as first logged; the rest were the pre-lockpy EOS-FUSE venv
  errors. The resubmit 4620473 (workday again) lost 23/37 the same way — all run_79/81 beam samples.
- Run_79/81 beam acquisitions take up to > 8 h per (tag, arm) at 4 cores (median samples job 3.1 h). **The full pass
  must use the `tomorrow` flavour (or split those acquisitions)** for those runs.
- 4620474 (84), cospin (144), ng1 (240): complete. 4620578: limit raised in place to 24 h (`condor_qedit MaxRuntime`).
