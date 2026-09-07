# Processing n_TOF runs ourselves

**We do not need n_TOF to process a run, and we do not need the merge.** This is
the recipe, proven twice: 224573-224579 in July (as `prod_v11`) and 224688 on
2026-08-10 (as `prod_v12`). The merge node fails and **that is fine** — see §5.

Evidence that the partials are sufficient:
[`skip_diagnosis/README.md`](skip_diagnosis/README.md).

---

## 1. Prerequisites

```bash
ssh -K lxplus          # MANDATORY. without delegated credentials there is no
                       # AFS token, no condor auth, and /eos/user looks absent
```

The UserInput must be staged on AFS **with absolute paths** to its 26 pulse-shape
templates — `RunProcessing.sh` does not resolve bare filenames:

```bash
# from the repo, once per variant
./ntof_processing/deploy_userinput.sh v12_liqpileup <local-staging-dir> \
     /afs/cern.ch/work/d/dneff/x17_reproc/userinputs
rsync -a -e "ssh -K" <local-staging-dir>/ \
     dneff@lxplus:/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/
```

v12 is already staged at
`/afs/cern.ch/work/d/dneff/x17_reproc/userinputs/v12_liqpileup/UserInput.h`.

## 2. Run it

```bash
W=/afs/cern.ch/work/d/dneff/x17_reproc
RUN=224690
A=$W/aux_prod_$RUN; mkdir -p $A; cd $A          # run FROM the aux dir
OUT=/eos/user/d/dneff/x17/reproc/prod_v12/$RUN; mkdir -p $OUT

/eos/experiment/ntof/repositories/processingscripts/RunProcessing.sh \
    -y 2026 -a EAR2 -c X17_measurement -r $RUN \
    -p $W/userinputs/v12_liqpileup/UserInput.h -o $OUT
```

It prepares the file lists, builds a DAG and **submits immediately** — there is
no dry-run. A range is `-r <first> -l <last>`, and `-s 1` skips runs that already
have output.

**`-o` cannot point at ntof storage, and this is not negotiable.**
`ProcessFileList.sh` is a compiled binary that validates the output path against
exactly two prefixes — the only ones in its string table are:

```
/eos/user/
/eos/project-
```

Anything under `/eos/experiment/...` gives `Output path ... is not supported at
the moment!` and every job exits 255. And even if it were accepted,
`/eos/experiment/ntof/processing/official/completed/` is **not writable** by us
(owner `ntofpro`), so our output can never sit beside n_TOF's own. `/eos/project-n/ntof`
is a root-owned empty stub from 2017, not a usable project space.

So: stage on `/eos/user/d/dneff/...`, then move to ntof storage (§6). Both
`/eos/experiment/ntof/data/x17/reproc/` and `/eos/experiment/ntof/processing/Users/`
are writable by us, and the ntof quota is `ignored` (unenforced).
`process_missing_runs.sh` does the whole cycle so the user quota only ever holds
one batch.

## 3. Watch it

```bash
condor_status -schedd -af Name                  # find your schedd
condor_q -name bigbird15.cern.ch -totals
```

**Always query by schedd name.** Each `ssh lxplus` can land on a different login
node, and a bare `condor_q` then reports an empty queue for jobs that are running
fine. Treat an empty queue as real only when the command also exits 0.

Expect ~4 raw files per job, so `ceil(raw_files / 4)` jobs, ~45-60 min each,
running in parallel. 224688 (173 raw files, 44 jobs) took ~85 min wall.

## 4. What you get

```
<OUT>/completed/<run>/run<run>_0001.root … _NNNN.root     the partials
<OUT>/completed/<run>/history_<run>.root                  the UserInput used
```

## 5. The merge fails. Ignore it.

The DAG's final `Merge_<run>` node dies within minutes and DAGMan exits 1:

```
max total download bytes exceeded (max=1024 MB, this file=764 MB)
Job removed by SYSTEM_PERIODIC_REMOVE due to disk usage exceeded allowed max
Disk (KB): 58332564 used, 3000 requested
```

That is condor's file-transfer cap and the merge job's 3 MB disk request, not
wall clock. **A DAGMan exit status of 1 with all `Process_*` nodes done is a
SUCCESSFUL run for our purposes.** Check the node tally, not the exit code:

```bash
grep -E "^[0-9/]+ [0-9:]+ +[0-9]+ +[0-9]+ +[0-9]+" $A/$RUN/run$RUN.dag.dagman.out | tail -1
# want:  <N> done, 0 queued, 1 failed   where the 1 failure is the merge
```

**We do not need the merge.** `ntof_dream_merge/ntof_io.ntof_paths()` and
`slim_pipeline/config.ntof_files()` both chain partials in order by design, the
`index` tree is replicated in full in every partial, and a slim built off
partials passes all 19 clock-QA checks. `STATUS.md` has said "Never merge a run"
since July.

## 6. Verify, then move off the user quota

```bash
ssh -K lxplus 'source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
               python3 skip_diagnosis/verify_partials.py <run> ...'
```

checks contiguity, `N == ceil(raw/4)`, 16 trees with sane entry counts, and the
`history`. **Note on `history` for OUR products:** the md5 will NOT equal
n_TOF's, because line 0 is the staged filename (`UserInput.h` vs
`UserInput_2026_EAR2_X17_v4.h`) and the template addresses carry our AFS prefix.
Compare the **parameter columns with the `.txt` addresses dropped**, plus the
template basenames — on 224688 those are identical to `done/run224572.root`.

### ⚠ Deleting from `/eos/user/` does not free quota by default

The staging tree inherits

```
sys.recycle="/eos/home-i00/proc/recycle/"
```

so an `rm` moves files to the EOS recycle bin **where they keep counting against
your quota**, and a normal user **cannot purge it** — `eos recycle purge`, the
by-date form and even `eos -r 0 0` all answer `you cannot purge your recycle bin
without being a sudo or having an admin role`. Retention is 42 days
(`Object-Lifetime 3628800`). This is how the 2026-08-10 campaign silently pushed
CERNBox from 82 % to 99 % **while harvesting**: 161 GB moved to the ntof disk,
0 GB returned.

**The fix, applied 2026-08-11 — remove the attribute on the staging subtree only:**

```bash
EOS_MGM_URL=root://eosuser.cern.ch eos attr rm sys.recycle /eos/user/d/dneff/x17/reproc
```

Measured on run 224697: quota 993.60 → 964.23 GB (−29 GB) with the recycle entry
count unchanged at 598. Home and everything else keep their recycle protection.
Reverse with `eos attr set sys.recycle=/eos/home-i00/proc/recycle/ <path>`.

Anything already in the bin stays there — 695 GB of prod_v11/prod_v12 staging is
stuck until it ages out or an admin purges it. It is all duplicate: the originals
are verified on the ntof disk.

**Quota bounds the staging, not the campaign.** `/eos/user/d/dneff` is 2.00 TB
with ~370 GB free as of 2026-08-10 and a run costs ~35 GB — so all 30 remaining
runs (~1.05 TB) will not fit at once. They do not need to.
`process_missing_runs.sh` runs a rolling pipeline: at most `MAX_INFLIGHT` runs
staged (default 6, ~210 GB), and each run is verified and moved to the ntof disk
the moment its partials are complete, freeing the slot. One invocation handles
any number of runs.

```bash
ssh -K lxplus
nohup ./process_missing_runs.sh 224689 224690 ... 224718 > campaign.log 2>&1 &
```

The destination defaults to `/eos/experiment/ntof/data/x17/reproc/prod_v12/` and
is overridable with `$X17_FINAL`. It copies, compares every file's size, and only
then deletes the staged copy — an unverified run is never removed.

## 7. What still needs doing

**224688-224718 minus 224688** — 30 runs, ~12 TB of raw, all staged. These are
the only n_TOF runs in the campaign with no processed output at all, and they
carry 61 h of DREAM beam time (run_150 onward).

## 8. Gotchas, all paid for

* **Never `condor_rm -all` while a DAG is up** — DAGMan reads the removals as
  node failures, burns all three `RETRY`s and aborts the DAG.
* **The 2 h `longlunch` wall still kills the occasional node, and RETRY does not
  help** — the retry re-runs identical work under the identical limit. Of 1 300+
  jobs across the 31-run campaign exactly one hit it: **224709 node 0023**, which
  was killed at 2h00m, retried, and would have burned all three attempts.
  **Fix: raise the flavour in the node's own submit file. DAGMan re-reads it on
  every retry**, so no resubmission is needed:

  ```bash
  A=<aux>/aux_prod_<run>/<run>
  sed -i 's/"longlunch"/"workday"/' $A/run<run>_<NNNN>_process.sub   # 2 h -> 8 h
  condor_rm -name <schedd> <clusterid>      # kill the doomed attempt; retry picks up the edit
  ```

  224709/0023 then finished in **5h06m** — 5x the slowest normal job in that run
  (0.99 h) and far beyond any 2 h limit.
* **A stalled node is not always a heavy one — check CPU before theorising.**
  0023 burned only ~21 min of CPU in 249 min of wall time (8 % utilisation), so
  it was I/O-blocked, not compute-bound. Its data volume (17.1 GB) was ordinary:
  job 0032 did 17.4 GB in 0.99 h. Its four raw files read at 48-205 MB/s from
  lxplus, so the inputs were fine. Read `RemoteUserCpu`/`RemoteSysCpu` from
  `condor_q` before assuming a slow job is a big job — and note that a blocked
  job's runtime cannot be extrapolated from a working job's.
* `history` is a ROOT **string object**, not a TTree: read it with
  `uproot.open(f)['history'].member('fString')`; `.keys()` raises.
* A zero-byte `run<run>.root` in `done/` is a failed merge, not a processed run.
  Test size, never existence.
