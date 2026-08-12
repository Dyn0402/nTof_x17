# lxplus condor for the wft chain

> **MPGD26 full-June reco campaign (2026-08-12):** the manifest, job scripts
> and two-phase runbook live here too — `make_manifest.py`,
> `make_reco_package.py`, `run_reco_job.py`, `reco.sub`, `gate_eval.py`,
> `collect_results.py`. The runbook is `../FREEZE_MPGD26_2026-08-12.md` §5.
> The rest of this file documents the older drift-gap-fit workflow.

# Running the drift-gap fits on lxplus condor

The gap study splits cleanly in two, and only the cheap half needs the bench
data:

| stage | needs | where |
|---|---|---|
| `wft.cli reco`, `bench/build_cache.py` | `decoded_root` waveforms (25-50 GB per dataset) | **local** — the June waveforms are not on EOS |
| `bench/gap_fit.py` (the fits) | `bench_cache.pkl` (16-49 MB) + a bundle (5 kB) | **condor** |
| `bench/gap_merge.py`, `gap_compare.py`, maps | the shard parquets | local, seconds |

The fits are ~95 % of the CPU: every variant re-fits ~12 k events at ~0.1 s
each. One shard of 8 is a few minutes on one core, so a whole systematics sweep
that takes hours serially at home finishes in one condor pass.

## Recipe

```bash
# 1. stage code + caches + bundles + the job list (local)
../../.venv/bin/python mx_june_wft/condor/make_package.py --shards 8
#    add --cross to fit every dataset with every OTHER dataset's bundle
#    (the calibration-systematic sweep)

# 2. ship it (needs a kerberos ticket: kinit <user>@CERN.CH)
rsync -av /home/dylan/x17/cosmic_bench/condor_wft/ lxplus:~/wft_gap/

# 3. submit
ssh lxplus 'cd ~/wft_gap && condor_submit gap_fit.sub && condor_q'

# 4. bring the results back and merge (local)
rsync -av lxplus:~/wft_gap/out/ /home/dylan/x17/cosmic_bench/condor_wft/out/
cd /home/dylan/x17/cosmic_bench/condor_wft/out && for f in *.tar.gz; do tar xzf $f; done
../../../../PycharmProjects/nTof_x17/.venv/bin/python \
    mx_june_wft/bench/gap_merge.py --dir out/out --label sat_det3 \
    --bundle <the bundle that fit was run with>
```

`gap_merge.py` writes `gap_study.json` + `event_profiles.parquet` in exactly the
format `bench/gap_study.py` produces locally, so `gap_compare.py`,
`gap_map_hires.py` and `gap_charge_check.py` all consume it unchanged.

## Job shape

- `request_cpus = 1`, `request_memory = 2500MB`, `microcentury` (1 h) flavour.
  A shard of ~1.5 k events runs in 3-6 min; the memory is dominated by holding
  one shard of windows.
- Inputs travel with the job (`transfer_input_files`); the largest is a 49 MB
  cache. Nothing is read from EOS, so there is no xrdcp / token dance.
- Environment is LCG_105 (`numpy`, `scipy`, `pandas`, `pyarrow`). No ROOT: the
  fitting half of `wft/` imports only numpy and scipy. If `pyarrow` is missing
  the shard falls back to `.csv.gz` and `gap_merge.py` still reads it.
- `max_retries = 2` covers evicted/held workers; a shard is idempotent, so a
  rerun simply overwrites its own parquet.

## Watch out for

- **The bundle is part of the measurement.** A fit is labelled
  `<data>__with__<bundle>` whenever the two differ; do not merge shards from
  different bundles into one result — the endpoint moves by ~1.7 mm between
  legitimate bundles of the same detector (see `GAP_CONSISTENCY_2026-07-30.md`).
- Shards are split `events[i::N]`, so every shard samples the whole run
  uniformly — a missing shard biases statistics, not geometry, but merge only
  complete sets when quoting numbers.
- `bench_cache.pkl` is built from the LOCAL production reco (windows + M3
  truth + the active box). Rebuild and restage it whenever the analyzer, the
  seeding or the alignment changes.
