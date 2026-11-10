# Chamber A two-track overlay on the final reconstruction: batch run (prepared 2026-11-10, NOT submitted)

Produces `papers/paper1_v2/figs/out/two_track_separation.{pdf,png,json}`: probability that both muons are
reconstructed against their separation, opening-angle resolution, separation error. Method and definitions: the
docstring of `../make_two_track_separation.py`.

## Why batch

The fit is the cost, and it is far over the laptop limit (2 processes, ~20 min):

| measured on the laptop (smoke, 80 fits, `out/two_track_separation_smoke.json`) | CPU s per fit |
|---|---|
| single donor / empty-partner check | 0.8 |
| overlay, separation 0–24 mm (two-track split attempts) | 8–15 |
| overlay, 24–80 mm | 2–6 |
| **mean over the 8 bins** | **8.4** |

300 overlays per bin x 8 bins + 2 x 200 checks = ~20 300 CPU s (5.6 core-hours); even 150 per bin is ~86 min on
2 processes. Inputs are not the problem (2.1 GB, all already on EOS and on the laptop). Memory: 470 MB main process +
~50 MB per worker beyond the shared pages.

## Inputs (all on EOS, verified with xrdfs 2026-11-10)

`/eos/experiment/ntof/data/x17/cosmic_bench/`
- `june_tests/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/decoded_root_fa80c06/*_0{7,8}.root`
  (FEU 7 = X, 8 = Y; 4 acquisitions, 1.93 GB) and `combined_hits_root_rr1_cm0/*.root` (70 MB, seeding only)
- `june_tests/mx17_det3_saturday_scan_6-27-26/run_config.json` (strip map)
- `Analysis_pass1_twotrack/<same rel>/mx17_3/wft/events.parquet, events.candidates.parquet, calib_bundle/` (final
  tables and bundle: `xy_pairing.split_corrob = charge`, `select = cost`; the script asserts the former)
- `x17rerun/paper1_eff/budget/<same rel>/mx17_3/rays_det3.parquet` (reference rays in the strip frame, interior flag)

## Commands

1. Commit `papers/paper1_v2/figs/make_two_track_separation.py` and `two_track_batch/` (they are new, uncommitted
   files), then ship the commit to AFS as every other job (`rerun/RUNBOOK.md`):
   ```
   C=$(git rev-parse --short HEAD)
   git archive --format=tar HEAD | ssh lxplus "mkdir -p ~/x17rerun/code_$C && tar -x -C ~/x17rerun/code_$C"
   ```
2. On lxplus: `lxstore status`, then
   ```
   mkdir -p ~/x17rerun/two_track/log && cd ~/x17rerun/two_track
   condor_submit code=/afs/cern.ch/user/d/dneff/x17rerun/code_$C ~/x17rerun/code_$C/papers/paper1_v2/figs/two_track_batch/two_track.sub
   ```
   8 jobs x 4 CPUs, `longlunch`; expected ~10–30 min each (300 overlays + 50 checks per shard, ~2 600 CPU s at 8.4 CPU s on the
   laptop; condor nodes may be up to 2x slower). Each job xrdcp's the 2.1 GB of inputs into its scratch dir, checks
   them with `root_complete`, runs Python through `/afs/cern.ch/user/d/dneff/bin/lockpy`, writes
   `run_<shard>_{overlays,candidates,events}.parquet` + `run_<shard>_timing.json` (a few MB) to
   `/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_v2/two_track/` with `eosput`, and empties the scratch dir.
3. After the cluster: compare job starts with jobs in `log/tt.log` (`grep -c '^001'` vs `grep -c '^005'`) and read
   every non-empty `log/tt_*.err`. A failed attempt is a bug to explain, not a flake.
4. Merge and plot on the laptop (needs the local tables, or set `TT_TABLES` / `TT_RAYS` to copies):
   ```
   D=$WORK_DATA/claude-jobs/v2_twotrack/batch; mkdir -p $D
   for f in $(xrdfs root://eospublic.cern.ch/ ls /eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_v2/two_track); do
     xrdcp -s root://eospublic.cern.ch/$f $D/; done
   .venv/bin/python papers/paper1_v2/figs/make_two_track_separation.py plot --work $D --prefix run
   ```
   Expect 8 shards, 2 400 overlays + 400 checks. Check `check_single` and `check_noise` in the json first (both should
   be ~100 %: the donors were reconstructed by the same chain, so this is a relative efficiency; position s68 ~0.4 mm,
   angle s68 ~0.9° X / ~0.55° Y on the smoke sample).

The pair list is deterministic (seed 7): every shard rebuilds it and keeps rows `index % nshard == shard`, so the
shards are disjoint and their union is the whole sample. Outputs differ in the last digits between worker nodes
(`rerun/condor/README.md`), so compare with a tolerance.

## Local smoke (done 2026-11-10)

`make_two_track_separation.py smoke --n 8 --jobs 2`: 8 overlays per bin + 8 singles + 8 empty-partner checks, 80 fits,
4.6 min of fit wall time on 2 processes, all `fitted`. Shard path tested with `run --shard {0,1} --nshard 400` + `plot`.
