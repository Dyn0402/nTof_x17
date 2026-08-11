# The n_TOF merge: what it does to large runs, and what we actually have

**The 5-7 August pass reconstructed far more than `done/` suggests. For 55 runs
the processing finished and only the MERGE is absent** — a complete, contiguous,
v12 partial set is sitting in
`/eos/experiment/ntof/processing/official/completed/<run>/`, 2 223 partials and
1 816 GB of finished output that nothing was reading, because the pass publishes
only merged files.

Full inventory of all 445 campaign runs: `inputs/inventory_2026-08-10.csv`,
regenerate with `inventory.sh`. Audit of a partial set: `verify_partials.py`.

> **This supersedes the wall-time diagnosis this file first carried.** Processing
> jobs *are* sometimes killed by the 2 h `longlunch` wall (§4, measured) but that
> is not why these runs lack output — their processing completed.

## 1. The state of the campaign, 2026-08-10

| state | runs | meaning |
|---|---|---|
| `MERGED` | 359 | non-empty `done/run<run>.root` — the normal case |
| `PARTIALS_ONLY` | 53 | processed, not merged. **Usable now** |
| `MERGE_EMPTY` | 2 | a **zero-byte** `done/` file; partials are the truth |
| `RAW_ONLY` | 31 | nothing processed; raw staged (12.76 TB) |
| `NOTHING` | 0 | — |

The 31 `RAW_ONLY` runs are **224688-224718**, a contiguous block: everything
after the pass stopped at 08-07 19:56, through to the end of data taking on
08-10. These are the only runs that genuinely need processing.

224649 and 224650, which the 08-08 request asked to recall from tape, **have no
DAQ directory at all** — they are not runs with data sitting on this path, and
the "bracketed" window we gave them was a guess. Drop them from the ask.

## 1a. Availability in beam time, 2026-08-10

`availability.py`, over 289.0 h and 342 DREAM sub-runs (data taking ended
2026-08-10; DREAM now runs to run_162, so this is ~36 h more than the 08-08
request knew about):

| | hours | share |
|---|---|---|
| **AVAILABLE** — merged *or* unmerged partials | **223.3** | **77 %** |
| **NEEDS PROCESSING** — raw staged, nothing processed | 60.9 | 21 % |
| no n_TOF run live | 4.7 | 2 % |

The split is clean in time, not scattered: every DREAM run through **run_147 is
88-100 % available**, and every run from **run_150 onward is 0 %**. That
boundary is the pass stopping at 08-07 19:56.

**Proven end to end.** run_116/stat090_0001 slimmed directly off the unmerged
224632 partials: **PASS on all 19 clock-QA checks, 0 warnings** — efficiency
94.23 % (held-out 94.16 %), accidental 0.0498 %, residual RMS 6.87 ns, 1 146 of
1 146 bunches fitted. The per-arm offsets come out within 0.6 ns of the locked
run_79 v12 values (A -17.6 vs -16.81, B +7.7 vs +7.55, C +1.2 vs +1.62,
D -1.2 vs -0.83), which confirms the processing from the physics rather than
from a checksum.

## 2. Is the no-merge deliberate? The evidence says no

We looked for a lock or marker and **there is none**:

* the not-merged directories contain only `run<run>_NNNN.root` and
  `history_<run>.root` — no dotfiles, no flag file, nothing (`ls -A`);
* merged runs **keep their partials too** (224636 still has 54, 224572 has 39),
  so `completed/` is not a staging area that gets cleaned on success, and the
  presence of a partial set carries no signal either way;
* `official/` and `done/` hold no manifest, skip-list or per-run marker.

**Size correlates strongly, but not deterministically.** Over the 413 runs that
were processed at all:

| output size (sum of partials) | runs | merged | rate |
|---|---|---|---|
| 0-1 GB | 200 | 199 | 100 % |
| 1-15 GB | 66 | 66 | 100 % |
| 15-20 GB | 9 | 9 | 100 % |
| 20-25 GB | 17 | 13 | 76 % |
| 25-30 GB | 42 | 31 | 74 % |
| 30-35 GB | 65 | 37 | 57 % |
| 35-40 GB | 8 | 2 | 25 % |
| > 40 GB | 6 | 1 | 17 % |

**Below 20 GB, 275 of 275 merged — not one failure.** Above it the rate falls
monotonically with size. But in the overlap region the two populations sit on top
of each other: merged runs span **20.5-42.0 GB**, unmerged **20.3-39.5 GB**. A
run of 42.0 GB (224644) merged; a run of 20.3 GB (224513) did not.

**That is the shape of a limit being approached, not a rule being applied.** A
deliberate "too big, do not merge" policy would be deterministic and leave a
clean cut with no overlap. Two further points argue the same way:

* **the two zero-byte files** — `done/run224405.root` and `done/run224667.root`,
  both 2026-08-05. A deliberate skip does not create an empty output file. Those
  are merges that started and died;
* run number does not explain it — the merge rate for big runs is 64 %, 65 % and
  55 % across the 224400s, 224500s and 224600s, so it is not "the pass gave up
  partway".

**How the merge actually fails — measured, not guessed.** Our own 224688 run
(44 partials, 34.0 GB) reached its `Merge_224688` node on 2026-08-10 and died in
**five minutes**:

```
004 Job was evicted. Code 33 Subcode 0
    Reason: STARTER ... failed to send file(s) ...: 49 total failures:
    first failure: sending file .../run224688_0002.root:
    max total download bytes exceeded (max=1024 MB, this file=764 MB)
012 Job was held.
009 Job was aborted.
    Job removed by SYSTEM_PERIODIC_REMOVE due to disk usage exceeded allowed max
    Disk (KB): 58332564 used, 3000 requested
```

So it is **condor's 1024 MB file-transfer cap and a 3 MB disk request against
58 GB of real usage** — not wall clock. It failed in 5 minutes of a 60-minute
allowance, which rules the `microcentury` flavour out as the binding constraint.
This is exactly the failure `ntof_dream_merge/ntof_io.py` recorded in July.

**An earlier version of this file blamed the one-hour wall. That was wrong.**

**What this does NOT settle.** It cannot be the whole story for n_TOF, because
they merged 66 runs in the 1-20 GB band whose partials also far exceed 1 GB of
transfer. Their merge must be configured or invoked differently from what
`RunProcessing.sh` generates for us — with a raised transfer budget, a bigger
disk request, or off-condor entirely. **We cannot see their configuration**, so
what we can say is: the merge step as we can run it cannot handle these runs at
all, large runs are exactly the ones missing from `done/`, and two of them left
zero-byte files. "Intentional policy" remains inconsistent with the size overlap
and with the empty files, but we cannot prove their mechanism from outside.

## 3. It does not matter much, because the partials are as good

`ntof_dream_merge/ntof_io.ntof_paths()` has chained partials by design since
July, and `slim_pipeline/config.ntof_files()` now falls back to
`completed/<run>/` automatically. The merge adds no content.

Every one of the 28 runs from the 08-08 request was audited and passes:

* **contiguous** `_0001 … _NNNN`, no gap;
* **complete** — N equals `ceil(raw files / 4)` (224667 and 224405 predate the
  07-08 split change and match `ceil(raw / 10)`);
* **v12** — the `history` string is byte-identical to the reference
  `done/run224572.root`, md5 `e51a3ef3dc0b32c1803e59ad18639a7c`, 10 696 bytes;
* **readable** — first/middle/last partial open with 16 trees, ~3.4 M entries.

Corroborated independently in the fit: slimming run_116 × 224632 off the partials
gives wall top/bottom offsets within **±5.5 ns**, the known v12 signature (the
old official processing gives ±32-39 ns).

**A zero-byte merged file is a real failure mode, so test size and not
existence.** `config._merged_ok()` does; without it 224405 and 224667 resolve to
an empty file while a complete partial set sits next door.

## 4. The wall-time kills, real but a separate thing

`RunProcessing.sh` submits processing nodes at `longlunch` (2 h). Of our 78 July
jobs over 224573-224579, three were killed:

```
009 (11878665.000.000) 07/28 23:33:48 Job was aborted.
	Job removed by SYSTEM_PERIODIC_REMOVE due to wall time exceeded allowed max.
```

Completed jobs ran to a median 1.33 h, max 1.92 h — the tail was touching the
wall. All three were absorbed by `RETRY 3`. `walltime_diagnosis.py` has the
tables. The split has since been tightened (mtime 2026-08-07 11:55): 10 raw
files per job in July, 4 today, so per-job load fell ~2.4x and this pressure is
largely gone.

## 5. What to ask n_TOF

1. **Re-run the merge** for the 55 runs — not the reconstruction, which is done.
2. **Process 224688-224718** (31 runs, 12.76 TB raw). We are doing some of these
   ourselves; 224688 was run on 2026-08-10 as the proof.
3. **Tell them about the two zero-byte files**, which any existence check treats
   as success.
4. Drop 224649/224650 from the request — no DAQ directory.

Draft: [`../NTOF_MERGE_REQUEST_2026-08-10.md`](../NTOF_MERGE_REQUEST_2026-08-10.md).

## 6. Regenerating

```bash
scp inventory.sh lxplus:~   # then
ssh -K lxplus 'bash inventory.sh' > inputs/inventory_2026-08-10.csv

ssh -K lxplus 'source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
               python3 verify_partials.py 224573 224576 ... 224673'

ssh -K lxplus 'python3 joblog_csv.py 224573 224574 224575 224576 224577 224578 224579' \
    > inputs/july_job_walltimes.csv
```

`ssh -K` is mandatory — without delegated credentials there is no AFS token.

## 7. Operational notes learned here

* **`-o` is validated and most EOS roots are rejected.** `ProcessFileList.sh`
  answers `Output path ... is not supported at the moment!` and exits 255 for
  `/eos/experiment/ntof/data/...`. Write to `/eos/user/d/dneff/...`.
* **Never `condor_rm -all` while a DAG is up** — DAGMan reads the removals as
  node failures, burns all three retries and aborts the DAG.
* **Always query condor BY SCHEDD NAME.** Each `ssh lxplus` can land on a
  different login node, and a bare `condor_q` then reports an empty queue for
  jobs that are running perfectly well on the schedd you submitted to. This read
  as "the DAG finished" while 11 nodes were still running on bigbird15. Find the
  jobs with `condor_status -schedd -af Name` and then
  `condor_q -name <schedd> ...`; treat an empty queue as real only when the
  command also **exits 0**. Same failure mode as `SLIM_CAMPAIGN_2026-08-09.md`
  section 5 (there it was a dead schedd, here a different one), and the same
  cure.
* **`history` is a ROOT string object, not a TTree.** Read it with
  `uproot.open(f)['history'].member('fString')`; `.keys()` raises.
