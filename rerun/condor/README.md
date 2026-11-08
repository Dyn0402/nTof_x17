# Condor jobs and the lock venv: the "usual EOS read errors" (2026-11-05)

**Answer:** the errors were our own doing. Every job ran Python straight from the lock venv on EOS
(`/eos/user/d/dneff/venvs/ntof-lock-cvmfs`, 14,140 files) through the worker's EOS FUSE mount. When hundreds of jobs
start together and import numpy, scipy, pandas and pyarrow, FUSE reads fail. **About half of all job starts died.**
Retries hid it, and sessions called it a flake for a week. Fixed by staging the venv into the job's scratch dir with
`lockpy`; outputs are byte-identical to running from the FUSE venv. Two guards now refuse new job scripts that run
software from EOS.

## What the failures look like

```
OSError: [Errno 5] Input/output error                 (importlib get_data on a venv .py)
ImportError: .../site-packages/scipy/.../_reordering.cpython-313-x86_64-linux-gnu.so: cannot read file data: Input/output error
ImportError: The `scipy` install you are using seems to be broken, (extension modules cannot be imported)
pyarrow.lib.ArrowKeyError: No type extension with name arrow.py_extension_type found   (a half-imported pandas/pyarrow)
exit status 126                                        (the read failed on the interpreter or the code archive itself)
```

All of them are reads of *software* from `/eos`. Copying data from EOS in the same jobs (`cp` with retries,
`ntof_stage.sh`) works fine.

## How big it was

n_TOF pre-freeze checks, code `ed0b1162`, clusters 4618427–30 (`~/x17rerun/ntof/pass1/log/ntof.log` on lxplus), by
15:30 on 11-05:

| cluster | jobs finished | job starts | failed after retries |
|---|---|---|---|
| 4618427 `cospin` | 143 | 209 | 3 |
| 4618428 `samples` | 26 | 285 | 2 |
| 4618429 `ng1` | 154 | 359 | 1 |
| 4618430 `twosided` | 54 | 94 | 0 |

That is 947 starts for about 380 jobs. The failures were spread over many nodes, so this is load on the mount, not
one bad machine. At that moment 122 of the 482 `.err` files (each holds only the latest attempt of its job) held a traceback, nearly all imports from the venv. The history: the 10-29 dry run saw status 126
and added `RETRY 3` (`make_pass_dag.py`); the 10-30 pass saw 18 status-126 failures and excluded one node; the n_TOF
plan wrote "a few jobs … `max_retries 2` covers it".

## The fix: `lockpy`

`rerun/condor/lockpy` (deployed to `/afs/cern.ch/user/d/dneff/bin/lockpy`):

1. Copies `ntof-lock-cvmfs.tgz` (214 MB, sha256 in `ntof-lock-cvmfs.tgz.sha256`, both next to the venv on EOS) into
   `$_CONDOR_SCRATCH_DIR/.lockpy` with `xrdcp`, over xrootd rather than FUSE. It retries up to 5 times with back-off.
2. Checks the sha256, unpacks the tarball, and test-imports numpy, scipy, pandas and pyarrow.
3. Runs that copy's python. Further `lockpy` calls in the same job reuse it, with a `flock` so parallel calls don't
   race. `lockpy --prefix` prints the directory instead, for wrappers that want `python3` on `PATH`.

Outside condor (no `$_CONDOR_SCRATCH_DIR`) it runs the EOS venv directly; at interactive load that's fine.
`LOCKPY_FUSE=1` forces that path inside a job, which is only useful for comparisons. A venv is relocatable when called
as `bin/python`: `bin/python3.13` is a symlink to the LCG_110 Python on CVMFS, and no file in `site-packages` holds the
EOS path (checked).

**After rebuilding the venv** (`/eos/user/d/dneff/venvs/mk_cvmfs2.sh`), run `rerun/condor/pack_lock_venv.sh` on lxplus
(~8 min: reading the venv's 14k files through FUSE is itself the slow part).

Every job script in the repo (66) and MX17_Geant's `pairsim_reco.sh` call `lockpy`. The 4 July-beam
`mx_july_beam_qa/lxplus/*_wrapper.sh` use `--prefix`.

## Checks

- **Condor, 6 real n_TOF jobs** (run_80, A/C/D × 2 sub-runs, `max_retries 0`): all exit 0.
- **Same machine, staged vs FUSE venv** (lxplus, run_80 `_000` A, 200 events): tracks, candidates and splits parquet
  **byte-identical**. Staged 2 m 39 s against 3 m 50 s from FUSE.
- **Nothing copied back to AFS:** a `should_transfer_files = YES` job returned only its top-level file. Condor does not
  transfer the `.lockpy/` subdirectory (610 MB).
- The deployed `~/x17rerun/ntof/pass1/ntof_stage.sh` was swapped by atomic rename while 4618428–30 ran, so their
  retries use `lockpy`. The previous version is kept as `ntof_stage.sh.pre_lockpy`.

## Side finding: results are not bit-reproducible across worker nodes

The same job rerun on a different node gives the same shapes (1348 tracks, 2620 candidates), but the values differ in
the last digits: angles ~1e-5°, positions ~1e-6 mm, t0 < 1e-3 ns. Only the error columns of near-singular fits (x_p0_err
etc.) differ by large absolute amounts. On one machine, staged and FUSE runs are byte-identical, so this is the
hardware (CPU-dependent SIMD/BLAS code paths), not the venv. That last step is an inference: I did not pin two runs to
two named CPU types. The plan's earlier "condor output bit-identical to the desktop" held because the hardware was the
same. **Compare pass outputs with a tolerance, not by bytes.**

## Guards, so it isn't reintroduced

Both live in `~/Software/claude-skills` and apply to every repo and machine:

- **`claude/hooks/no-eos-software.py`**: a Claude Code PreToolUse hook (in `settings.shared.json`). It blocks writing
  a job script (`.sh/.sub/.jdl/.dag`, or text mentioning condor) that runs `/eos/…/bin/python`, `activate`, `pip`,
  anything inside a `venv/conda/miniforge` directory on EOS, or `source /eos/…`. A line carrying `eos-software-ok`
  is exempt, for code that doesn't run inside a job.
- **`claude/lxplus/condor_submit`**: deployed as lxplus `~/bin/condor_submit` plus a `condor_submit_dag` symlink;
  `~/bin` comes before `/usr/bin` on lxplus `PATH`. It reads each submit file's `executable`, and for a DAG every JOB's
  submit file. It refuses the submission if the executable is on EOS, or if it, or a `.sh` it calls in the same
  directory, runs software from EOS. `CONDOR_EOS_OK=1` overrides once. Tested: refuses the old stage script, passes
  the fixed one, catches it inside a DAG.

The rule itself is written in the global `CLAUDE.md` (claude-skills), in this repo's `CLAUDE.md`, and in
`rerun/history/LESSONS.md`. **And more generally: a job failure rate is a bug to explain.** After a cluster, compare
job starts (`001` events) with jobs (`005`) in the user log.

## Writing to EOS: use `eosput`, not `cp` through FUSE (2026-11-08)

During the n_TOF re-decode, 5 of 31,496 files written with `cp` + size check + `mv` ended up cut on EOS, with exit 0, clean logs and
a passing size check. EOS calls such a file healthy, because it checksums what arrived. `eosput <local> <eos path>` (`~/bin`, master
here) uploads over XRootD with a server-checked adler32 atomically and re-reads the size and checksum. `root_complete FILE...` is a
stdlib check that a ROOT file is as long as its header says: use it on inputs of tools that exit 0 on a cut file. Evidence and the
list of files: `rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md`, "11-08".
