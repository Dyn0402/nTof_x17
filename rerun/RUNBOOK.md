# Re-run runbook: running a stage on lxplus condor (W10, 2026-10-23)

How any stage of the re-run (RERUN_PLAN.md §4) runs on lxplus without the desktop bench tree. Nothing in the
analysis code is edited for it: the code reads the desktop layout, and the job builds that layout in its scratch dir.

## The pieces

| piece | what it does |
|---|---|
| `rerun/inputs.py` | **The one place that says which inputs the pass reads.** `GENERATIONS['rr1']`: `decoded_root_fa80c06` (C11), `combined_hits_root_rr1_cm0` (C12/C13), `m3_tracking_root_v2` (C13). `EXCLUDED`: known false-start acquisitions. `stage()` copies a manifest row's inputs from EOS into `<bench>/<area>/<run>/<subrun>/` under the canonical names (`decoded_root`, `combined_hits_root`, `m3_tracking_root_v2`), skipping acquisitions absent from the M3 dir, and writes `INPUTS.json` (generation, sources, every file with its size) and `incomplete_events.json` (decoder-audit eventIds per acquisition and FEU: not measured, never a miss). |
| `qa_config.py` | `X17_BENCH_ROOT` replaces `/home/dylan/x17/cosmic_bench` in every registered run; `X17_ANALYSIS_ROOT` replaces the `Analysis/` root. Unset = unchanged. |
| `rerun/stage_job.py` | One stage for one row: stage inputs, pull the row's Analysis subtree from EOS, set the two variables, register the row in `qa_config.RUNS` (registered key, or synthesized from the manifest row), refuse to run if the config resolves outside the staged tree, run the stage in-process, write `STAGES/<time>_<pid>_<stage>.json` (stage, args, rc, time, code commit, input generation, files written), push back only the files the stage wrote (parallel stages of one row never overwrite each other). |
| `rerun/stage_job.sh` | condor entry point: LCG_110 view + the lock venv, `WFT_RND_DIR` for the regression test, cleans the scratch dir (condor copies leftovers back to AFS). |
| `rerun/stage.sub` | `condor_submit stage.sub code=<code dir> jobfile=<lines of stage_job.py args> [cpus=N mem=MB flavour=...]` |

Keys: a manifest key (`campaign_manifest.csv`) or a run registered in `qa_config.RUNS` (`sat_det3`,
`g_det3_wknd`, `o22_long_det2`, `g_det4`, `g_det6_long`, `g_det7_long`, ...).

Outputs: `/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_rr1/<run>/<subrun>/<det>/` (the pass);
`/eos/user/d/dneff/x17rerun/Analysis_dryrun/` for tests (`--eos-analysis`).

## Recipe

```bash
# 1. freeze: commit, then ship a git archive of that commit (AFS holds code only)
C=$(git rev-parse --short HEAD)
git archive --format=tar HEAD | ssh lxplus "mkdir -p ~/x17rerun/code_$C && tar -x -C ~/x17rerun/code_$C"
scp rerun/stage_job.sh lxplus:x17rerun/stage_job.sh; scp rerun/stage.sub lxplus:x17rerun/stage/

# 2. a job file: one line per (row, stage); the line is stage_job.py's arguments
#    sat_det3 --parts run_config,m3,hits --script mx_june_cosmic_qa/03_alignment_and_tpc.py
#    sat_det3 --module wft.calibrate -- --jobs 4
#    sat_det3 --module wft.cli -- reco {key} --jobs 4
#    sat_det3 --parts run_config,m3 --script mx_june_wft/01_alignment.py
# 3. dry-run the submit and read the expanded arguments, then submit
ssh lxplus 'cd ~/x17rerun/stage && condor_submit -name bigbird101.cern.ch -dry-run /tmp/d.txt stage.sub \
    code=$HOME/x17rerun/code_<C> jobfile=jobs.txt && condor_submit -name bigbird101.cern.ch stage.sub \
    code=$HOME/x17rerun/code_<C> jobfile=jobs.txt cpus=4 mem=12000'
```

Stages of one row depend on each other through the Analysis tree, so submit them in order (one job file per
stage, the next after the previous finished), or use a DAG.

## Rules

- `--parts` stages only what the stage reads: hits-only stages do not need `decoded` (25–50 GB per row).
- A stage that pickles worker functions (`wft.calibrate`) runs as `--module`, never as a script.
- The input generation is recorded in `INPUTS.json` (inputs) and `STAGES/*.json` (outputs). A table without them is not
  from the pass.
- `condor_q`/`condor_rm` need `-name bigbird101.cern.ch`.
