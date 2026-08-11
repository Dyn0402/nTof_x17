#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walltime_diagnosis.py -- why the 5-7 August pass skipped the large runs.

`slim_study/why_skipped.py` established from the outside that the skipped runs
are the *big* ones (nothing below 0.35 TB was ever skipped) and called it "the
shape of a resource limit a large job sometimes misses and sometimes makes".
This script names the limit, from the inside, using our own condor logs.

The mechanism
-------------
`RunProcessing.sh` submits every processing node with

    +JobFlavour = "longlunch"       # a hard 2 h wall

and builds a DAG in which the single `Merge_<run>` node is a CHILD OF EVERY
processing node.  So one node that exhausts its `RETRY 3` takes the merge with
it, and the run ends with **no output file of any kind** -- which is exactly
what the 41 missing runs look like from outside.

In July we processed 224573-224579 ourselves through the same script.  Of 78
processing jobs, **three were killed outright**:

    009 (...) 07/28 23:33:48 Job was aborted.
        Job removed by SYSTEM_PERIODIC_REMOVE due to wall time exceeded allowed max.

and they survived only because a retry landed on a faster machine.  Several
more finished within five minutes of the wall.  At that per-job load the upper
tail of the distribution is already touching 2 h; it does not need to grow much
before the *bulk* of a run's jobs cross it and all three retries fail.

Why per-file size is the right axis
-----------------------------------
The raw stream1 files are fixed-duration chunks, so GB-per-file is a proxy for
instantaneous DATA RATE, and the work in a job is (files per job) x (GB per
file).  Total run size mixes rate with duration; per-file size does not, and it
separates skipped from processed runs more sharply.

The split changed under us
--------------------------
`RunProcessing.sh` has mtime 2026-08-07 11:55 -- i.e. it was modified DURING
the 5-7 August pass.  Measured:

    July  (our 224573 run) : 156 raw files -> 16 job lists = 10 files/job
    August (our 224632 run): 250 raw files -> 63 job lists =  4 files/job

so per-job load fell by ~2.4x.  That is consistent with the fix having been
made in response to these failures, and it predicts that re-running the 41
today simply works.  `../skip_diagnosis/README.md` records whether it did.

Inputs (regenerate with the commands in README.md):
  inputs/july_job_walltimes.csv    run,job,cluster,attempt_s,outcome,mem_mb,nfiles
  inputs/staged_raw_2026-08-10.txt run n_files total_MB done|SKIPPED
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WALL_S = 2 * 3600           # the "longlunch" flavour

# raw geometry of the runs we processed in July, from the 08-08 census
# (their stream1 has since aged off the EOS disk, so it cannot be re-measured)
JULY_RAW = {          # run: (n_raw_files, raw_TB)
    224573: (156, 0.46), 224574: (152, 0.45), 224575: (17, 0.05),
    224576: (150, 0.43), 224577: (166, 0.45), 224578: (71, 0.21),
    224579: (1, 0.00),
}


def load_walltimes():
    rows = []
    with (HERE / 'inputs' / 'july_job_walltimes.csv').open() as fh:
        for r in csv.DictReader(fh):
            r['attempt_s'] = int(r['attempt_s'])
            r['nfiles'] = int(r['nfiles'])
            r['run'] = int(r['run'])
            rows.append(r)
    return rows


def load_staged():
    rows = []
    for ln in (HERE / 'inputs' / 'staged_raw_2026-08-10.txt').read_text().split('\n'):
        f = ln.split()
        if len(f) == 4 and int(f[1]) >= 20:
            rows.append((int(f[0]), int(f[1]), int(f[2]) / 1024.0, f[3]))
    return rows


def july_table(rows):
    print('== July 2026: our own pass over 224573-224579, 10 raw files per job ==\n')
    print(f'{"run":>7} {"GB/file":>8} {"jobs":>5} {"median":>8} {"max":>8} '
          f'{">1h45":>6} {"KILLED":>7}')
    by_run = defaultdict(list)
    for r in rows:
        by_run[r['run']].append(r)
    tot_killed = tot_jobs = 0
    for run in sorted(by_run):
        js = by_run[run]
        nf, tb = JULY_RAW[run]
        gpf = tb * 1024 / nf if nf else 0
        good = sorted(j['attempt_s'] for j in js if j['outcome'] == 'completed')
        killed = [j for j in js if j['outcome'] == 'killed_walltime']
        near = sum(1 for s in good if s > 6300)
        tot_killed += len(killed)
        tot_jobs += len(js)
        med = good[len(good) // 2] if good else 0
        print(f'{run:>7} {gpf:8.2f} {len(js):5d} {med/3600:7.2f}h '
              f'{max(good)/3600 if good else 0:7.2f}h {near:6d} {len(killed):7d}')
    print(f'\n{tot_jobs} processing jobs, {tot_killed} killed by the 2 h wall '
          f'({100*tot_killed/tot_jobs:.1f} %).')
    good = sorted(r['attempt_s'] for r in rows if r['outcome'] == 'completed')
    print(f'completed-job wall time: median {good[len(good)//2]/3600:.2f} h, '
          f'p90 {good[int(0.9*len(good))]/3600:.2f} h, max {max(good)/3600:.2f} h '
          f'against a {WALL_S/3600:.0f} h limit.')

    # what per-file size puts the MEDIAN job at the wall
    ref = [r for r in rows if r['run'] == 224573 and r['outcome'] == 'completed']
    ref_s = sorted(x['attempt_s'] for x in ref)
    gpf573 = JULY_RAW[224573][1] * 1024 / JULY_RAW[224573][0]
    med, mx = ref_s[len(ref_s) // 2], max(ref_s)
    print(f'\nScaling from 224573 ({gpf573:.2f} GB/file, 10 files/job): work per job is '
          f'linear in GB/file, so the wall is reached at')
    print(f'  median job : {gpf573 * WALL_S / med:5.2f} GB/file')
    print(f'  slowest job: {gpf573 * WALL_S / mx:5.2f} GB/file  <- failures start here')


def skip_table(rows):
    print('\n\n== The pass, from outside: skip rate vs per-file size ==')
    print('   (135 runs whose stream1 was still staged on 2026-08-10)\n')
    edges = [0, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 4.0, 99]
    print(f'{"GB/file":>13} {"runs":>5} {"skipped":>8} {"rate":>6}')
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in rows if lo <= r[2] / r[1] < hi]
        if not sel:
            continue
        k = sum(1 for r in sel if r[3] == 'SKIPPED')
        print(f'{lo:5.1f}-{hi:<7.1f} {len(sel):5d} {k:8d} {100*k/len(sel):5.0f}%')
    gpf = lambda r: r[2] / r[1]
    dn = sorted(gpf(r) for r in rows if r[3] == 'done')
    sk = sorted(gpf(r) for r in rows if r[3] == 'SKIPPED')
    print(f'\nmedian GB/file  processed {dn[len(dn)//2]:.2f}   '
          f'skipped {sk[len(sk)//2]:.2f}')


def main() -> int:
    july_table(load_walltimes())
    skip_table(load_staged())
    print('\nThe two numbers to compare: failures start around 3.4 GB/file from '
          '\nthe inside, and the skip rate crosses 50 % between 3.0 and 3.4 from '
          '\nthe outside. Same wall, measured two ways.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
