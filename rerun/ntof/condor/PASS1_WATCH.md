# Watching the n_TOF pass 1 on condor: a checklist

Written 2026-11-11 for a session whose only job is to keep the pass moving until it finishes. Follow the steps
exactly. **Don't analyse, resubmit, release, remove or edit anything that this page doesn't list.** If something
doesn't match what's described here, stop and report it to Dylan. Don't improvise a fix.

## What is running

| Cluster | Jobs | What it is |
|---|---|---|
| **4636418** | 15,550 | main pass, all n_TOF triggers → EOS `Analysis_pass1/stage2/` |
| 4636419 | 198 | chamber D run_104, L1 pin → `Analysis_pass1/side/run104_D_L1/` |
| 4636420 | 198 | chamber D run_104, L2 pin → `Analysis_pass1/side/run104_D_L2/` |
| 4638252 | 1 | rerun of main proc 1727 (whose exit 1 is expected and already explained) |

The main cluster keeps at most 2,000 jobs in the queue (`max_materialize`), so new jobs appear as old ones finish.
**Known problem:** newly added jobs sometimes get CERN's requirements rewrite applied twice, which creates a cycle
that matches no slot. The jobs then sit idle forever, with 0 running. It happened on 11-10 and again on 11-11
(1,980 jobs, procs 11029 onward, stuck from 06:22 to 11:15). The repair is `fix_req_cycle.sh`, which
`watch_pass1.sh` runs for you. The cause is presumed to be on CERN's side; nothing of ours edits the jobs.

State at 11-11 11:20, after the second repair: main 11,008 done (11,007 exit 0, plus proc 1727) out of 15,550,
1,974 idle, 26 running and rising. 4636419: 102 done, 36 running, 60 idle. 4636420: 198 idle, waiting for slots.
That leaves about 4,500 main jobs. Expect another 2–3 days (an estimate: it depends on our condor share).

## Rules

- Reach lxplus only as `ssh lxplus` (never a node name: it triggers a 2FA prompt for Dylan). Use
  `ssh -o BatchMode=yes lxplus ...` with a `timeout`. If it asks for a password or fails with "Permission denied",
  the shared connection has expired. Stop and ask Dylan to log in (`! ssh lxplus` in his prompt).
- The AFS user log `~/x17rerun/ntof/pass1/log/ntof.log` has been stale since 11-09. **Don't count from it.** Use
  `condor_history` (the `--hist` option below).
- Never `condor_rm`, `condor_release`, `condor_submit`, or `condor_qedit` by hand.
- Don't delete anything on AFS or EOS.

## Each check (every 3–4 hours)

```
timeout 200 ssh -o BatchMode=yes lxplus 'bash ~/x17rerun/ntof/pass1/watch_pass1.sh'
```

Read the output:

1. **"jobs with the cyclic requirement: N" on 4636418.** If N > 0, the script repaired them and then prints
   "after repair: 0 still bad". That's fine: write down the time and N. If it says anything other than 0 still
   bad, stop and report.
2. **Cyclic requirement on 4636419 or 4636420.** The script won't fix those. Stop and report.
3. **"HELD JOBS".** Don't release them. Copy the hold reasons into the report and stop. "Disk quota exceeded"
   means AFS is full: run `ssh lxplus lxstore status` and include its output.
4. **Running count.** If main shows no `2=` (nothing running) while there are idle jobs, the requirements are all
   good, and the latest job start is over 2 hours old, run
   `ssh lxplus 'condor_q 4636418 -constraint "JobStatus==1" -af ClusterId ProcId | head -1'`, then
   `condor_q <that id> -better-analyze`, and report the "slots match" lines. A low running count with recent
   starts is normal: it's our fair share.
5. Once a day, add `--hist` (about 3 more minutes) to count finished jobs:
   `timeout 900 ssh -o BatchMode=yes lxplus 'bash ~/x17rerun/ntof/pass1/watch_pass1.sh --hist'`.
   Lines read `<count> 4 <exitcode>`. Exit 0 is good. On main, exactly one `4 1` (proc 1727) is expected. Any
   other non-zero exit code is a failure: list the ProcIds with
   `condor_history 4636418 -constraint 'ExitCode != 0' -af ProcId ExitCode` and report them. Don't resubmit.

Append one line per check to the log at the bottom of this file (time, main done/idle/running, repairs made).
Commit it with a short message and push `origin main`.

## When it's finished

The pass is done when `watch_pass1.sh` prints `(none)` for all three clusters. Then:

1. Run `watch_pass1.sh --hist` and record the final counts. Main should total 15,550 completed (15,549 exit 0 +
   proc 1727). 4636419/20 should be 198 exit 0 each, and 4638252 one exit 0.
2. Run `ssh lxplus lxstore status` and record the quota.
3. **Stop here and tell Dylan the pass has finished.** The next steps (`audit_eos.py outputs`, checking a
   `.meta.json` per job, reading each failure's `.err`, N-G1, the D run_104 L1/L2 choice) need judgement. They're
   listed in `HANDOFF.md` § "n_TOF pass 1" and belong to a full analysis session.

## Check log

| Time (CET) | Main done / idle / running | Sides 19 / 20 | Notes |
|---|---|---|---|
| 11-10 12:55 | 9,028 / 2,000 / 0 | — | first cycle (procs 9029–11028) repaired; running 93 a minute after |
| 11-11 11:20 | 11,008 / 1,974 / 26 | 102 done, 36 run / 198 idle | second cycle (1,980 jobs from proc 11029, stuck since 06:22) repaired; slots match again |
| 11-11 11:25 | ~11,0xx / 1,975 / 25 | 35 run, 60 idle / 198 idle | babysit session start; 1 cyclic job on main repaired (0 bad after); held 0 |
| 11-11 12:47 | n/a / 1,937 / 63 | 28 run, 60 idle / 198 idle | 12 cyclic jobs on main repaired (0 bad after); held 0 |
| 11-11 18:47 | n/a / 1,759 / 241 | 0 run, 60 idle / 198 idle | 50 cyclic jobs on main repaired (0 bad after); held 0 |
