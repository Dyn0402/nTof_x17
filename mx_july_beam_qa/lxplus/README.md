# Running the July-beam read pass on lxplus / HTCondor

Instead of pulling each 13-18 GB official root file local, run the **read pass** on
lxplus next to the EOS data via a short condor job, and only bring back the ~2.4 MB of
`.npz` caches + calib JSON. All plotting stays local.

**Status:** benchmarked and validated 2026-07-16 on run224461 (feasible, correct output),
but at ~89 min/run it is a batch/overnight workflow, not interactive. Kept here as the
foundation for on-the-fly processing during data taking.

## Why this split is clean

The 6 read-pass scripts — `01_signal_qa`, `02_coincidence_scan`, `03_time_offsets`,
`06_wall_geometry_test`, `07_mip_amplitude`, `09_late_inclusive` — are **pure read**
(no matplotlib) and emit only small caches. The plotters (`01b`, `02b`, `07b`, `08`,
`10`, `11`) run locally from those caches. Every script takes the run file/stem as
`argv[1]`. So: condor produces caches → rsync 2.4 MB down → plot local.

Dependency order (enforced by the wrapper): `01` first (bunch-selection cache the rest
read), then `02`; `07` needs `03`'s `calib/time_offsets_*.json`, `09` needs `06`'s cache.

## One-time setup on lxplus

```bash
ssh lxplus                       # alias in ~/.ssh/config: user dneff, GSSAPI/Kerberos
mkdir -p ~/x17qa
```
Software is the LCG_105 view (numpy 1.23.5 + uproot 4.3.7) — sourced by the wrapper, no
install. (Local venv is uproot 5.7 / numpy 1.26; no compatibility issues seen.)

## Per-run

From the repo, stage scripts + job files, then submit:
```bash
rsync -av -e 'ssh -o ControlPath=none' mx_july_beam_qa/*.py        lxplus:x17qa/
rsync -av -e 'ssh -o ControlPath=none' mx_july_beam_qa/lxplus/     lxplus:x17qa/
ssh lxplus 'cd ~/x17qa && chmod +x readpass_wrapper.sh && myschedd bump && condor_submit readpass.sub run=224461'
```
`myschedd bump` prints the chosen schedd (e.g. `bigbird25.cern.ch`). Monitor:
```bash
condor_q   <id> -name bigbird25.cern.ch     # status; re-pass -name each fresh session
condor_tail <id> -name bigbird25.cern.ch    # live stdout of a running job
```
On completion condor writes `cache/` and `calib/` into `~/x17qa`. Pull them down:
```bash
rsync -av -e 'ssh -o ControlPath=none' lxplus:x17qa/cache/*run224461* ./mx_july_beam_qa/cache/
rsync -av -e 'ssh -o ControlPath=none' lxplus:x17qa/calib/*run224461* ./mx_july_beam_qa/calib/
```
Then plot locally, e.g. `python 07b_geometry_mip_plots.py run224461`.

## Benchmark (run224461, 18.4 GB, node-local scratch)

| step | 01 | 02 | 03 | 06 | 07 | 09 | xrdcp | total |
|------|----|----|----|----|----|----|-------|-------|
| sec  | 477 | 258 | 1365 | 996 | 1369 | 654 | 145 | **~89 min** |

- The coincidence scripts (03/06/07) dominate: Python per-bunch pairing loops, **not** I/O.
  run224461 is ~2× the hits of run224404 (no SiPM-wall outage), so this is near worst case.
- **Peak RSS ~15.9 GB** → `request_memory = 16 GB` (in `readpass.sub`).
- Data source: xrootd `root://eosexperiment.cern.ch//eos/experiment/ntof/processing/official/done/runNNN.root`
  (xrdcp 18 GB → scratch in 145 s). EOS-fuse also works (~86 MB/s) but xrdcp-to-scratch is cleaner.

## lxplus gotchas (why condor, not an interactive node)

Interactive read attempts failed three ways that condor avoids entirely:
- **connection sharing** (`ControlMaster` in `~/.ssh/config`) pins you to one login node —
  use `-o ControlPath=none -o ControlMaster=no` for long/parallel ssh;
- **AFS close-to-open**: an actively-written log is invisible from other nodes until closed;
- **nohup jobs die** on ssh channel close.

Also: don't pipe live monitoring through `grep` — it block-buffers to a pipe and looks hung.
Condor gives node-independent status (`condor_q`) and auto-transfers outputs.

## On-the-fly during data taking

Official processing drops finished runs into `.../done/runNNN.root` automatically
(224461/462/463 all appeared 2026-07-16). A watcher on that dir can `condor_submit
readpass.sub run=<new>` as each lands, so caches are waiting by the time you sit down.
Not built yet — this is the next step.

## Files

- `readpass_wrapper.sh` — the condor executable (xrdcp + 6 scripts + adc_to_mv gen).
- `readpass.sub` — submit description; `condor_submit readpass.sub run=NNN`.
