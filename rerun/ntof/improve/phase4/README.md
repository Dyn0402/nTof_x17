# Phase 4: the test re-reconstruction (started 2026-11-23)

**To run the campaign, read [`CAMPAIGN.md`](CAMPAIGN.md)** (the handoff of 2026-11-24: what is prepared, the checks,
the commands, what decides each switch). This file describes the switches and keeps the history.

Phase 4 of `../AUDIT_FIX_ROADMAP.md` collects the fixes that act only through a new reconstruction. Each fix is in
the code behind a switch whose default is the pass-1 behaviour, so pass 1 is unchanged. We re-reconstruct five test
sub-runs with each switch on alone, and with all of them on together, and compare the results with pass 1. Dylan then
decides whether a pass 2 is worth it.

## The switches

| roadmap | what it changes | how it is turned on |
|---|---|---|
| 4.3 | The children of a two-track split get the same t0 prior at the trigger as the one-track fit. With pass 1 they have no prior, and 27-28 % of them rail at the ±180 ns box edge. | `WFT_DIAG=reco.TWO_TRACK_T0_PRIOR=1` |
| 4.4 | The clusters offered to the fit are ranked by charge among those with at least 3 strips, and the cap of five counts the clusters that survive the wide-band veto. Pass 1 ranks them by strip count and applies the cap before the veto, so a wide noise band can take a slot from a short bright track. | `WFT_DIAG=seed.SEED_RANK=charge,seed.SEED_CAP_AFTER_VETO=1` |
| 4.5 | A plane the clustering left empty gets the bench's 2-strip head-on seeds and the whole-plane rescue window. | `ntof_reco.py --headon-rescue` |
| 4.6 | `quality_ok` takes its reference median from triggers at ≥ 12 ms. Pass 1 takes it from all triggers, so the post-flash ringing sets it. This is a stage-3 switch: it needs no re-reconstruction. | `X17_QUALITY_REF_LATE_MS=12` when building stage 3 |
| 4.7 | The candidates table carries `bow_h`. | always (a new column) |
| 4.9 | Chamber D's permanently oscillating channels (19 x, 42 y not yet masked) are added to its hot list. | the bundles `bundles_osc/` (`../../ringing/make_osc_bundles.py`) |
| 4.10 | The gated ringing correction of the A/C waveforms (`deflect`, mode `deflect_hits`). | `ntof_reco.py --ringing deflect_hits` |
| 4.11 | The candidates table carries each candidate's 20-bin depth profile (`q_profile`). On in every configuration since 2026-11-24. | `WFT_DIAG=reco.STORE_PROFILES=1` |
| 4.1 | The x–y column-time tolerance of n_TOF (A 151, C 296, D 535 ns) in place of the bench value of the bundles (197, 456, 726 ns). | `ntof_reco.py --tc-tol <ns>` |
| 2.8 | Every candidate carries `n_sig5` and `q_sig5`: the strips of its fit window above 5σ and their summed peaks. With the ringing correction on, `n_sig5 ≥ 1` in both views is the waveform test of the ringing rule (`quality_join.py --low-charge r28`). | always (new columns) |
| 4.12 | The seeding significance floor is judged within 30 mm instead of over the whole plane. | `WFT_DIAG=seed.SIG_FLOOR_LOCAL_MM=30` |

Item 4.8 (three small output fixes) is not done yet. The wider t0 box (4.2) was tested by injection and gains little
(`../t0_study/README.md`), so it is not in the campaign.

## The test

- **Sub-runs:** run_145 stat090_0000, 0001 and 0002, run_104 stat090_0003, and run_139 stat090_0003. Chambers A, C
  and D (D is not run on run_104, whose D latency epoch is unplaced).
- **Configurations:** `ref` (pass 1, on this code), `all` (every switch on), and one configuration per switch:
  `floor30`, `t0prior`, `seedrank`, `headon`, `ringing` (A and C only) and `osc` (D only).
- **Code:** `code_e86028b5` on lxplus.
- **Jobs:** one per acquisition and chamber, as pass 1 (`../../condor/ntof_stage.sh`). The job lists come from
  `make_jobs.py`. The submit file is `phase4.sub`.
- **Outputs:** `/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/phase4_e86028b5/out/<cfg>/`.

## Status

- 2026-11-23: test cluster 4729099 (12 jobs of `all`, the heaviest configuration) submitted to size the campaign.
  9 of 12 done by 23:30: CPU efficiency 98 % on 4 cores, peak memory 957 MB, median 40 min, max 104 min. The 3 D
  jobs were still running after more than 4 h.
- The campaign (8 configurations, 896 jobs, clusters 4729137-44, `cpus=4 mem=4000 flavour=tomorrow maxmat=30`) was
  submitted and then removed the same night: Dylan wants to review it before a long campaign runs. Estimate about
  5,000 core-hours.
- 2026-11-24 (re-audit, `../reaudit/report.html`):
  - The sizing cluster finished: 12 of 12 jobs, exit 0. The three large D jobs ran 6.9, 7.6 and 7.6 h, close to the
    8 h limit of `workday`. Use `tomorrow` for D.
  - `ref` reproduces pass 1. Cluster 4729151 ran `ref` on the 8 A and C acquisitions of the sizing cluster (outputs
    in `out/reftest/`). Against `stage3_r2_final`: 20,251 tracks in both, all gated tracks matched, largest difference
    1.3e-6 mm.
  - `stage3_chain.sh` failed on every sub-run before this date, because the test has no chamber-B files. It now runs
    stage 3 with `--allow-missing` and clears `vt/`. Options: `PHASE4_DIR`, `PHASE4_OUT`, `PHASE4_SUBS`,
    `PHASE4_FETCH`, `PHASE4_JOBS`.
  - `compare.py` is the comparison script: `compare.py --cfg <dir>/final --ref <ref dir>/final --out <dir> --name <cfg>`.
    Compare each configuration with `ref` built by the same chain, and `ref` with `stage3_r2_final`.
  - A `ref` job takes about 3 times longer than an `all` job on A and C (A about 2 h, C 3.6 to 5 h on 4 cores),
    because the ringing correction removes many early fits. The campaign costs about 10,000 to 12,000 core-hours.

## Before submitting (re-audit 2026-11-24, roadmap 6.6)

**Done on the afternoon of 2026-11-24** (items 1 to 6; see `CAMPAIGN.md` for the state and what is left). The list is
kept as the record of what the re-audit asked for.

1. Correct the seed ranking (4.4): rank by significance and apply the cap after the wide-band veto.
2. Make `--ringing` write the per-track waveform flag, and store the depth profiles in every configuration.
3. Add a configuration with the n_TOF x-y time tolerance in the bundles (A 151, C 296, D 535 ns).
4. Run D on run_104 (canon bundle `D_E2_beam_L1`), and add two or three beam-off cosmic sub-runs (for example of
   run_149): they are the only angle truth.
5. Give D the single switches before `all`. With every switch on, D lost 131 real late tracks on 4 acquisitions.
6. Redeploy the code on lxplus (`deflect.py` and `ntof_stage3.py` changed after `e86028b5`), and regenerate the job lists.
7. Free AFS space first: the quota was at 90 % and `log_pass1_2026-11` at 93 % of its name slots on 11-24.
8. Re-run the injection test (`../t0_study/inject.py`) on the new code for 4.3 and 4.12: the five sub-runs hold only
   158 (A) and 103 (C) two-track events from 12 ms on.

What decides each switch is Table 2 of `../reaudit/report.html`.
