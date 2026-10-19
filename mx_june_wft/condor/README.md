# lxplus condor for the wft chain

> **MPGD26 full-June reco campaign (2026-08-12):** the manifest, job scripts
> and two-phase runbook live here too — `make_manifest.py`,
> `make_reco_package.py`, `run_reco_job.py`, `reco.sub`, `gate_eval.py`,
> `collect_results.py`. The runbook is `../FREEZE_MPGD26_2026-08-12.md` §5.
> The rest of this file documents the older drift-gap-fit workflow.

## The reco campaign, as it stands after the 2026-08-13 rerun

Read this before submitting a reco campaign again. Full account:
`../quality_investigation/JUNE_CONTINUITY_2026-08-13.md` §7–8.

```bash
# 1. matched lists — ALWAYS, for every row, computed locally
../../.venv/bin/python mx_june_wft/condor/make_matched_lists.py \
    --out /home/dylan/x17/cosmic_bench/condor_campaign/rerun_<date>/matched_lists_all

# 2. package (refuses a dirty tree; --t0p-dets IS the gate decision)
../../.venv/bin/python mx_june_wft/condor/make_reco_package.py \
    --t0p-dets mx17_2,mx17_4 \
    --matched-lists <the dir from step 1>

# 3. ship, DRY-RUN, then submit
rsync -av --exclude back/ --exclude results/ --exclude log/ \
    /home/dylan/x17/cosmic_bench/condor_campaign/ lxplus:~/wft_campaign/
ssh lxplus 'cd ~/wft_campaign && condor_submit -dry-run /tmp/d.txt \
    jobfile=jobs_rest.txt reco.sub && grep -m2 TransferInput /tmp/d.txt'
ssh lxplus 'cd ~/wft_campaign && condor_submit jobfile=jobs_gate.txt reco.sub \
    && condor_submit jobfile=jobs_rest.txt reco.sub'

# 4. collect into a DATED tree, then promote (dry-run first)
../../.venv/bin/python mx_june_wft/condor/collect_results.py \
    --back <...>/back_<date> --results <...>/results_<date>
../../.venv/bin/python mx_june_wft/condor/promote_rerun.py \
    --src <...>/results_<date> [--apply]
```

### Rules that are enforced in code, and why

- **Every job carries its own matched list.** LCG_105 mis-resolves the NClus
  branches of **v1** rays files (184 of 214 rows), silently degrading the
  recipe to chi2-only: 55.4 % of what those jobs reconstructed fell outside
  `chi2<1 & NClus>=4`. `run_reco_job.py --matched-list` refuses a list whose
  recipe, row or key disagrees, and the worker never opens a rays file. Rows
  that cannot be listed are refused, not downgraded.
- **`--t0p-dets` is the gate decision, not a default.** The phase-2 gate
  adopted the t0 prior for mx17_2 and mx17_4. Omitting the flag rebuilds those
  detectors on the un-adopted bundle — verify against a promoted product's
  `events.meta.json`, not against the submit files.
- **Tier B (`--vrefit`) cannot run on a worker at all.** `wft.calibrate` needs
  the hits-chain event cache *and* `alignment_tpc_veto50/alignment.json`,
  neither of which is staged. Run those rows locally with `--local`. Scan
  subruns that have a cache but no alignment get one from
  `seed_scan_alignment.py` (long run seeds z/theta, translation refitted).
- **Refit bundles carry no w0/kw** — `wft/calibrate.py` has no notion of them —
  so tier-B tables' ANGLES are on the uncorrected mapping and are not
  quotable. Their v values are. `promote_rerun.py` refuses them, along with
  off-conditions rows and tagged arms.

### Condor gotchas that cost real time

- **`queue` splits items on whitespace as well as commas**, and only the LAST
  variable absorbs the remainder of the line. A middle column holding
  space-separated flags silently becomes one token with the rest swallowed by
  the next variable. Column order is `row,tag,mlist,extra`. **Always
  `condor_submit -dry-run` and read the expanded `Arguments`/`TransferInput`.**
- **`condor_q` AND `condor_rm` need `-name <schedd>`.** A cluster lives on the
  schedd it was submitted from; from a login node both commands quietly
  address the local one and report nothing found. Get the schedd with
  `condor_q -global -constraint 'ClusterId==N' -af GlobalJobId | cut -d'#' -f1`.
- **`combined_hits_root` is fetched per acquisition**, like `decoded_root`.
  EOS can hold acquisitions the M3 chain rejected, old enough to predate the
  `significance` branch; fetching the directory wholesale kills seeding.

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
