# Phase 4: the test re-reconstruction (started 2026-11-23)

Phase 4 of `../AUDIT_FIX_ROADMAP.md` collects the fixes that act only through a new reconstruction. Each fix is in
the code behind a switch whose default is the pass-1 behaviour, so pass 1 is unchanged. We re-reconstruct five test
sub-runs with each switch on alone, and with all of them on together, and compare the results with pass 1. Dylan then
decides whether a pass 2 is worth it.

## The switches

| roadmap | what it changes | how it is turned on |
|---|---|---|
| 4.3 | The children of a two-track split get the same t0 prior at the trigger as the one-track fit. With pass 1 they have no prior, and 27-28 % of them rail at the ±180 ns box edge. | `WFT_DIAG=reco.TWO_TRACK_T0_PRIOR=1` |
| 4.4 | The clusters offered to the fit are ranked by charge among those with at least 3 strips. Pass 1 ranks them by strip count, so a wide noise band can take a slot from a short bright track. | `WFT_DIAG=seed.SEED_RANK=charge` |
| 4.5 | A plane the clustering left empty gets the bench's 2-strip head-on seeds and the whole-plane rescue window. | `ntof_reco.py --headon-rescue` |
| 4.6 | `quality_ok` takes its reference median from triggers at ≥ 12 ms. Pass 1 takes it from all triggers, so the post-flash ringing sets it. This is a stage-3 switch: it needs no re-reconstruction. | `X17_QUALITY_REF_LATE_MS=12` when building stage 3 |
| 4.7 | The candidates table carries `bow_h`. | always (a new column) |
| 4.9 | Chamber D's permanently oscillating channels (19 x, 42 y not yet masked) are added to its hot list. | the bundles `bundles_osc/` (`../../ringing/make_osc_bundles.py`) |
| 4.10 | The gated ringing correction of the A/C waveforms (`deflect`, mode `deflect_hits`). | `ntof_reco.py --ringing deflect_hits` |
| 4.11 | The candidates table carries each candidate's 20-bin depth profile (`q_profile`). | `WFT_DIAG=reco.STORE_PROFILES=1` |
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
- Not done yet: the check that `stage3_chain.sh ref` reproduces `stage3_r2_final` on the five sub-runs, and the
  comparison script.
