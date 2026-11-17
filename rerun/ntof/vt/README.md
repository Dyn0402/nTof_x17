# P7a: drift velocity v(t) on n_TOF (2026-11-17)

Plan: `../NTOF_PASS_PLAN_2026-11-02.md` P7a and §6 "11-17". Outputs: EOS `Analysis_pass1/vt_table/` (laptop copy
`~/work_data/ntof_pass1/vt/`): `report.html` (the measurements), `smooth.html` (the smooth model, environment, validation).

## Chain

| step | script | where |
|---|---|---|
| column duration per (acquisition, arm) on the pass's clean tracks | `vt_acq.py` via `vt_stage.sh`, `vt.sub` (`make_vt_jobs.py`) | condor 4677358, EOS `Analysis_pass1/vt/` |
| per-acquisition and per-sub-run v | `vt_table.py --vt <npz copy>` | local, ~2.5 h single core |
| campaign report | `make_vt_report.py` | local |
| smooth shared v(t), environment, validation | `vt_smooth.py` (needs `env/gve_t_recent.csv`, MeteoSwiss Genève-Cointrin 10-min, and `ntof_run_report/data/he3_pressure_5min.csv`) | local, ~20 min |
| A/B: does the frozen v shape the pass? | `make_vt_ab.py` (`ab_pick.csv`), `ab_compare.py` | condor 4677861, EOS `Analysis_pass1/tests/vt_ab/` |

Times: vt and the capsule log are local (CEST, from the DAQ); MeteoSwiss is UTC and is shifted +2 h.

## Results

- v moves with time by far more than its error, and the chambers move together: per sub-run corr(A, C) 0.85 (beam);
  the shared-curve model has C 3.0x and D 2.9x A's swing. HV never changed during the campaign.
- The frozen bundles match the measured v at their calibration run (run_145: A 0.998, C 0.989); the campaign medians
  are +3 % (A), +9 % (C) higher because most of the campaign is faster than that run. D reads +10 % against its
  bundle even on its own run: its absolute scale needs an angle anchor.
- Smooth model (`vt_smooth.py`): predicts a held-out acquisition to +-1.1 % (A), +-1.4 % (C), D sub-runs +-3.3 %;
  whole 6 h blocks held out: +-1.2 / +-1.5 %; pulls rms 0.92-1.06.
- Environment: outdoor water vapour (24 h causal smoothing) alone r = -0.84 with C; with the ³He capsule's daily swing
  (a hall thermometer) R^2 = 0.78 (C), 0.48 (A), 0.56 (D). -1.9 %/hPa and -2.2 %/K on C. The signs and size are not
  a gas-density effect; water entering the gas (permeation / outgassing, more downstream) is the reading, not shown.

## A/B test: rerun or table correction? (condor 4677861, 30/30 exit 0, first attempt)

30 beam acquisitions at the fastest times (C 20, v 12-15 % above the frozen v; A 10, 3.5-5 %), re-reconstructed with
only `v_drift` set to the measured sub-run v, against the main pass (`ab_summary.json`, `ab_per_acq.csv`):

| | A | C |
|---|---|---|
| events, planes with a fit | identical | identical |
| same chosen track (p0 < 0.05 mm, t0 < 0.5 ns) | 95.1 % | 96.1 % |
| on those, w unchanged (< 1e-6) | 100 % | 99.99 % |
| on those, tan_B / (tan_A v_A/v_B) - 1 | +0.04 % (68 % ±0.10 %) | +0.15 % (±0.53 %) |
| all tracks (any rank) found in both | 98.7 % | 99.1 % |
| tracks with true \|tan\| < 0.5 found in both | 98.6 % | 99.2 % |
| track count A → B | 53,212 → 53,166 | 133,460 → 133,638 |
| chosen fits, true \|tan\| 0.5-0.6 | 2,850 → 3,094 | 6,359 → 9,273 |

- **The table correction is exact for the fits**: v does not enter the waveform fit (w identical), and tan scales as
  v_frozen / v to 0.1-0.5 % (the rest is the bow term, subtracted in tan units).
- **What the frozen v does change** is which candidate is ranked first, and so which ~1 % of tracks exist (both
  directions, net -0.1 % / +0.1 %). Two mechanisms: the uncorrected-tan edge (|tan_raw| < 0.6) and the column-end
  window (1.6 x 30 mm / v on a 60 ns grid); a replay of the stored candidates predicted the top-choice change rate
  (4.7 / 3.8 % vs 4.9 / 3.9 % observed).
- **The |tan| < 0.5 cut** removes the v-dependent acceptance edge (C's 0.5-0.6 band changes by +46 % with v; inside 0.5
  the track count changes by -0.6 % (A) / -0.2 % (C)), but not the ~1 % reshuffle inside it, which comes from the window.
- **Recommendation (Dylan to decide): no rerun.** Correct tan with v(t) at table level, cut true |tan| < 0.5 so the acceptance does not move with
  v, and carry ~1 % track-set churn at the fastest times as a stated systematic (it is a ranking among candidates, not a
  loss: the counts change by < 0.6 %). These acquisitions are the campaign's extremes; typical ones sit closer to the
  frozen v. A rerun on v(t) bundles would change ~1 % of tracks and is not justified by this.

Does not rule out: a v-dependent effect on two-track splits specifically (not separated here); the A/B covers beam
triggers on A and C only (D has no beam v; B has no v).
