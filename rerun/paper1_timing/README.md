# Paper I §7 — chamber timing from the waveforms (pass 1, 2026-10-31)

Report: [`report.html`](report.html) (built by `make_report.py`). Paper figure: `figures/timing.pdf`
(`papers/paper1/figures.toml` → `fig.timing`).

## Verdict

- **On a cosmic muon the chamber times itself to σ ≈ 40–46 ns** (s68, event time against the scintillator trigger;
  det2, det3, det7 golden rows, M3-matched single tracks). That is **1.5–1.7 mm of drift**.
  - It is measured from the **start of the charge column**: the arrival of the charge made next to the mesh, the
    particle's crossing time plus a fixed delay.
  - It is the same at every track angle, head-on included.
- **Two parts.**
  - **≈ 22–28 ns per view is not shared by X and Y** (noise, 60 ns sampling, view-specific response). It falls steeply
    with the signal size (HV).
  - **≈ 35–40 ns is common to X and Y.** It is the arrival of the first electrons, read by both views, and it is the
    floor at high gain.
- **The trigger is not a term on the clean runs.** cov(start, end) of the column is ≈ 0: a trigger error would shift
  both column ends together.
  - **The exception is the det4-era runs (6-23 and 6-24):** a common shift of both ends of ≈ 40–60 ns rms.
- **SPS (det4 at H4) is not rescued.** The same estimator gives 160–200 ns per view on ZS data. That is the
  low-amplitude end of the bench curve (det4's gain, CF₄ at 14 µm/ns, ZS), an upper bound and not det4's intrinsic floor.
  The SPS session's limitation stands, now with the reason measured.
- **The hits-based 33 ns** (PLAN_42, MPGD proceedings) is the X−Y plane-to-plane number of single-strip leading edges.
  - The waveform equivalent on the same row (sat_det3) is **22 ns** per view.
  - The hits "absolute" 37.7 ns compares with the waveform event time **42 ns**.
  - So the hits value was the uncorrelated part only. Paper I should quote the event time (≈ 40–46 ns), not 33 ns.

## Headline numbers (M3-matched single tracks, s68)

| chamber | row | n | event time | X−Y/√2 | common to X, Y | trigger √cov | v [µm/ns] | event time [mm] |
|---|---|---|---|---|---|---|---|---|
| det2 | 6-22 long_run (quoted golden) | 5659 | **41.7 ns** | 27.6 | 36.9 | ≈ 0 | 38.3 | 1.60 |
| det2 | o22_long (calib golden) | 1153 | 39.9 | 28.7 | 34.3 | ≈ 0 | 38.3 | 1.53 |
| det3 | 6-27 sanity long_run (quoted golden) | 10044 | **40.3 ns** | 22.2 | 37.1 | ≈ 0 | 36.6 | 1.47 |
| det3 | sat_det3 (calib golden) | 3523 | 41.6 | 22.4 | 38.4 | ≈ 0 | 36.0 | 1.50 |
| det7 | g_det7_long (all X channels) | 433 | **45.6 ns** | 25.5 | 41.9 | ≈ 0 | 36.8 | 1.68 |
| det6 | g_det6_long | 366 | 78.1 | 52.5 | 68.7 | ≈ 0 | 24.8 | 1.93 |
| det4 | g_det4 (6-24) | 2109 | 110.9 | 38.7 | 107.5 | **42** | 32.7 | 3.63 |

Statistical errors (bootstrap) are 0.2–1 ns on det2 and det3 and 2–4 ns on det6 and det7 (`results/rows_matched.csv`).

- **HV** (figure (a)): σ_evt falls and flattens near 40 ns.
  - det3 425 → 500 V: 102 → 41 ns.
  - det2 450 → 500 V: 61 → 44 ns.
  - det6 415 → 465 V: 121 → 56 ns.
  - det7 415 → 445 V: 64 → 46 ns.
  - X−Y/√2 keeps falling (det3: 83 → 21 ns).
- **Drift field** (det3, 490 V; `figures/drift.png`): 114 / 79 / 52 / 41 / 38.5 ns at 104 / 174 / 243 / 313 / 382 V/cm,
  i.e. ≈ 2.3 → 1.4 mm.
- **Angle** (`figures/angle.png`): the median start moves ≤ 8 ns from head-on to |tan θ| = 0.5 on det2/det3, and the
  width is flat (head-on bin 36–43 ns).
- **Amplitude walk** of the event time: ≤ 12 ns across the early-amplitude quartiles (golden rows).

## Method

- **Input.** colsum.py's per-trigger strip sum S(t) of the track's cluster ±2 strips, X and Y.
  - Every lateral term of the forward model cancels in the sum, so S is the column's charge-arrival profile convolved
    with the electronics response, for any angle.
  - No hit times, no track model, no M3 in the estimator (the reconstruction basis).
  - Sources: the pass-bundle colsum of the golden and v(t) rows, and the GC/FA colsum of 145 more rows.
  - S(t) is a raw strip sum: the bundle only set the cluster finder's dead mask.
- **Estimator `edge25`** (`estimators.py`).
  - A step of the row's own pass template is fitted to the rising edge of S: the samples up to the first one above 25 %
    of the maximum, 1 ns grid plus a parabola.
  - It is offset by 10 ns × ftst. ftst is the DREAM fine timestamp of each view's FEU (= timestamp mod 6); the
    bundles' t0_abs falls 10 ns per step, and after the correction every ftst class has the same median.
- **Why this estimator.** On det3 golden, all clean triggers, σ_evt by estimator:

  | estimator | σ_evt | note |
  |---|---|---|
  | colsum's own box fit | 85 ns | its t0 trades off against the column duration T |
  | box with T fixed | 73 ns | |
  | cf 30 % | 77 ns | |
  | cf 10 % | 49 ns | |
  | edge15 | 44 ns | |
  | **edge25** | **42 ns** | |
  | edge40 | 48 ns | |

  The more of the column a time uses, the more it follows the column's charge profile, which X and Y share. A noise-set
  cut (`dev_edgeabs.py`) gives 37–48 ns but fails on 1.5 % of triggers; the common part stays 33–44 ns.
- **Decomposition.**
  - σ_xy = s68(t_X − t_Y)/√2.
  - σ_evt = s68((t_X + t_Y)/2).
  - σ_common = √(σ_evt² − σ_xy²/2).
  - The trigger test is cov(start, end) of the column (end = last 30 % crossing of S), central 90 %.
    - It is +σ_trig² for a trigger error, which shifts both ends.
    - The ionization at the mesh and at the cathode is independent.
    - It is −63 ± 19 ns² on sat_det3, against +1500 ns² if the 39 ns common part were the trigger.
- **Samples.**
  - Clean triggers: both views ≥ 5 pulse strips, one cluster, unsaturated.
  - The quoted sample adds an M3 match: the pass table's events, both views ok, one track. That drops non-track
    triggers; det7's all-trigger width is 164 ns, its matched width 46 ns.
- **mm.** v_row = v_S4 (masked interior, S4 re-check) × T_golden / T_row, with T = colsum's median column duration.

## The trigger term on 6-23 / 6-24 and det4's positions

- **Seen on the column.** The 6-23 (det3+det4 overnight) and 6-24 (det4) runs carry √cov(start, end) ≈ 40–66 ns, on
  both chambers of 6-23. Every other run reads ≈ 0.
- **The pass pins t0 to the trigger** (t0_abs[ftst], σ = 0.05 ns). A trigger error δt therefore moves p0 by
  tan × v × δt (`t0_position.py`).
- **The signed regression agrees in size.** On g_det4 the residual p0 − M3 against tan × δt has a slope of
  6.3 ± 0.6 µm/ns in X; a jitter of this size predicts 6.0. On sat_det3, with the column mid-time, it is 0.3 ± 1.8.
- **The cost is not established.** A jitter should widen the residual only at large |tan|, and that isn't seen:
  events with a large |δt| are wider at every angle, so |δt| mostly flags poor events. What this costs det4's positions
  is open.

## SPS (det4 at H4, `sps_edge.py`, `sps_timing.py`)

- **Selection.** The footprint selection (seed ≥ 50 ADC, final oscillator rule, archive 48244778), both views clustered,
  q ≥ 150 ADC. Template: g_det4 (the SPS pass's).
- **ZS rows** (run55/56/63/66/70):
  - edge25: X−Y/√2 = 160–205 ns, against the trigger 160–235 ns, with or without the ftst correction;
  - cf20: X−Y/√2 = 115–145 ns.
- **RAW run71:** see `results/sps_timing.json` (filled when its condor rows finish).
- **The amplitude curve** (figure (b)): at an early amplitude of 1000–4000 ADC the bench reads 20–45 ns and SPS
  110–160 ns. In drift-distance units (14 against ~36 µm/ns) that is 1.5–2.2 mm against 0.8–1.6 mm: the same physics
  at a worse S/N and with ZS, not a different mechanism. No intrinsic det4 number can be quoted from SPS.

## What this does not rule out

- **A trigger part below ~10 ns** inside the common term. The cov method carries an estimator bias of about −60 ns².
  The scintillator hardware note says ~5 ns.
- **The physical origin of the common 35–40 ns** (primary-cluster positions near the mesh, gain of the first electrons,
  a threshold set relative to the column maximum) is inferred, not simulated.
- **det6's uncorrelated part is 2× the others at the same HV**: not chased.
- **det4 bench timing** can't be separated from the 6-23/6-24 trigger term.
- **The 6-22 det3 bottom-slot rows** (skeptical) read 87 ns, and the 6-23 det3 rows read 119 ns (trigger); neither is
  quoted.

## Open questions for Dylan

1. **What does Paper I quote?** The event time (40–46 ns, 1.5–1.7 mm: how well the chamber alone fixes the absolute
   depth), with the split into the per-view part (22–28 ns) and the common part (35–40 ns)? Recommended: the event
   time, with the split in the text.
2. **The MPGD proceedings' "33 ns / 1.1 mm"** is the hits plane-to-plane number. Replace it with the waveform event time,
   or keep it labelled as hits?
3. **What changed in the trigger on 6-23 / 6-24** (the det4 era)? The run notes may say: a different paddle pair or
   discriminator? Nothing in the data identifies it.

## Files

- `estimators.py`: box / fixT / edge / edgeabs / cf.
- `timing_row.py`: one row, all estimators, decomposition, cov, walk.
- `make_jobs.py` → `jobs.txt`; `condor/`: `timing.sub`, `sps_edge.sub`, `deploy.sh`.
- `matched_rows.py`: the M3-matched widths of every row.
- `angle.py`, `t0_position.py`, `amplitude.py`, `sps_edge.py`, `sps_timing.py`, `collect.py`, `show.py`.
- `make_report.py` → `report.html`, `figures/`.
- `results/`: `rows.csv` (all clean triggers), `rows_matched.csv`, `golden.csv`, `angle.json`, `amplitude.json`,
  `sps_timing.json`, `t0_position.json`.
- EOS: `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_timing/` (`rows/`, `sps/`, `code_*`). Condor:
  `~/x17rerun/paper1_timing/` (lxplus).
