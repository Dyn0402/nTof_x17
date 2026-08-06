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

---

# run_58 drift-column scan (2026-07-30)

`run58_columns.py` + `run58_columns_wrapper.sh` + `run58_columns.sub`, staged the
same way as the read pass above. **One job per sub-run**, 76 of them: each xrdcp's
that sub-run's ~137 MB of `combined_hits_root` to node scratch and returns a ~240 kB
parquet of per-cluster drift-column statistics. **21 s wall, 632 MB peak RSS per job**;
the whole scan is minutes, ~10 GB read on the farm and ~8 MB back.

Why it exists: run_58 sweeps the drift voltage **700 → 200 V in 9 points** with a
**64-sample (3.84 µs)** window that contains the full column at every point, so it is
the only July dataset that can test whether a chamber's drift field actually responds
to its supply. See `../HANDOFF_2026-07-30_readout_window_and_detB.md` §4.4a.

```bash
# 1. freeze the strip map (local, once) — keeps the worker repo-free
.venv/bin/python mx_july_beam_qa/lxplus/make_run58_stripmap.py

# 2. stage + list the sub-runs (EOS is POSIX-mounted on lxplus)
rsync -av -e 'ssh -K -o ControlPath=none' mx_july_beam_qa/lxplus/run58_* lxplus:x17run58/
ssh -K lxplus 'mkdir -p ~/x17run58/logs ~/x17run58/out &&
  ls /eos/experiment/ntof/data/x17/july_beam/runs/run_58 | grep "^sngPS" > ~/x17run58/subruns.txt'

# 3. submit
ssh -K lxplus 'cd ~/x17run58 && myschedd bump && condor_submit run58_columns.sub'

# 4. collect + aggregate locally
rsync -av -e 'ssh -K -o ControlPath=none' lxplus:x17run58/out/ mx_july_beam_qa/cache/run58_columns/
.venv/bin/python mx_july_beam_qa/run58_column_scan.py
```

Gotchas, both hit while building this:

- **`condor_submit` parses the whole RHS of a `+Attribute` as an expression**, so an
  inline comment on `+JobFlavour = "espresso"   # 20 min` is a *parse error*. Keep
  comments on their own line. (`readpass.sub` above has the same latent issue.)
- **run_58 predates the 2026-07-24 analyzer**: no `significance`, `trunc_left` or
  `trunc_right` branch. The worker therefore selects on an **absolute amplitude cut**
  (300 ADC) rather than the relative significance floor, so that every sub-run — and
  every run compared against it — is selected identically. Comparing hit populations
  across analyzer versions without a common absolute cut is meaningless.
- The worker is deliberately **standalone** (no repo imports): batch nodes have only
  the LCG view, so the strip map is shipped as a 17 kB npz built by
  `make_run58_stripmap.py`.

---

# Two-source plastic calibration, runs 224588-224596 (2026-07-28 data)

`srccal.sub` (read) and `srccalfit.sub` (fit) — the Y-88 + Cs-137 campaign;
physics and method in `../SRCCAL_2026-07-28.md`.

The **read pass** is nine jobs, one per run, and is genuinely quick: these are
6-minute source runs, so the official files are only 0.3-0.6 GB and the whole
set came back in **~3 minutes wall**, ~1 MB of caches total. Nothing about it
resembles the 89-minute beam read pass above.

```bash
rsync -av -e 'ssh -o ControlPath=none' \
      ../33_srccal_spectra.py ../srccal_runs.py ../adc_mv.py srccal.sub \
      srccal_wrapper.sh lxplus:x17qa/
ssh -K lxplus 'cd ~/x17qa && chmod +x srccal_wrapper.sh && myschedd bump \
               && condor_submit srccal.sub'
ssh -K lxplus 'cd ~/x17qa && for t in srccal_out_*.tgz; do tar xzf "$t"; done'
rsync -av -e 'ssh -o ControlPath=none' lxplus:x17qa/cache/33_srccal_\*.npz ../cache/
```

The **fit pass** (`34` + `35`) normally runs on the laptop from those caches, but
it is a few thousand `curve_fit` calls and `srccalfit.sub` offloads it when the
local machine is loaded — one job, ~15 min, returns `srccalfit_out.tgz` with the
calib JSON, the figures and the results markdown. Ship
`calib/y88_energy_calib.json` along (it is in `transfer_input_files`) or the
07-17 transport table comes back empty.

Gotcha found here: **condor rejects a trailing comment after a `request_*`
value** (`request_disk = 3 GB  # ...` is a parse error), even though the same
form parses elsewhere in a submit file.
