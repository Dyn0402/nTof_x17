#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
campaign_status.py -- what has the slim campaign actually produced?

Run from ~/x17slim on lxplus:

    python3 campaign_status.py                  # summary
    python3 campaign_status.py --missing        # + every segment not yet done
    python3 campaign_status.py --cluster 13353824

`condor_q` answers "is the job running". This answers the question that matters
afterwards: for every (DREAM sub-run x n_TOF run) segment that segments.py says
is ready, is there a slim file on disk, and does its QA look sane?

Segments, not jobs, is the right unit: one job covers every sub-run overlapping
its n_TOF run, and a job that exits 0 having silently skipped a sub-run (below
slim_run's MIN_EVENTS gate, say) still looks like a success to condor.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Expected-segment list comes from the same code the submission used.
sys.path.insert(0, str(HERE))
try:
    from ntof_processing.slim_pipeline import segments as S
except ImportError:                                   # running from ~/x17slim
    sys.path.insert(0, str(Path.cwd()))
    from ntof_processing.slim_pipeline import segments as S

# Anything below these is worth a human look rather than a silent pass.
MIN_EFFICIENCY = 0.90
MAX_ACCIDENTAL = 0.005


def expected():
    """{(run, subrun, ntof_run)} -- every ready segment."""
    return {(p.dream_run, p.dream_subrun, p.ntof_run): p
            for p in S.propose() if p.reprocessed}


def found(root: Path):
    """{(run, subrun, ntof_run): (root_path, qa)} for everything on disk."""
    got = {}
    pat = re.compile(r'ntof_hits_(?P<run>run_\d+)_(?P<sub>\w+?)_(?P<ntof>\d{6})\.root$')
    for f in sorted(root.glob('out*/runs/*/*/ntof_hits/ntof_hits_*.root')):
        m = pat.search(f.name)
        if not m:
            continue
        qa_f = f.parent / 'qa.json'
        qa = json.loads(qa_f.read_text()) if qa_f.is_file() else {}
        got[(m['run'], m['sub'], int(m['ntof']))] = (f, qa)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--missing', action='store_true',
                    help='list every segment with no slim file yet')
    ap.add_argument('--cluster', help='also summarise this condor cluster')
    a = ap.parse_args()

    exp, got = expected(), found(a.root)
    done = {k: v for k, v in got.items() if k in exp}
    extra = {k: v for k, v in got.items() if k not in exp}

    runs_exp = sorted({k[2] for k in exp})
    runs_done = sorted({k[2] for k in done})
    runs_full = [r for r in runs_exp
                 if all(k in done for k in exp if k[2] == r)]

    nbytes = sum(f.stat().st_size for f, _ in got.values())
    print(f'segments : {len(done):>4} of {len(exp)} done '
          f'({len(done)/max(len(exp),1):.0%})')
    print(f'n_TOF run: {len(runs_full):>4} of {len(runs_exp)} complete, '
          f'{len(runs_done)} started')
    print(f'on disk  : {len(got)} file(s), {nbytes/1e9:.2f} GB')
    if extra:
        print(f'  note: {len(extra)} file(s) present that are not in the ready '
              f'list (an older or superseded run?)')

    # Every file in one campaign must have been cut with the SAME window.
    # A stale condor retry carrying an older sandbox will happily drop a
    # narrower file into the tree, and nothing downstream would notice: the
    # file is valid, just cut differently. Compare, do not assume.
    windows = {}
    for k, (f, _) in done.items():
        cal = f.parent / 'calibration.json'
        if cal.is_file():
            w = json.loads(cal.read_text()).get('slim_ns')
            windows.setdefault(w, []).append(k)
    if len(windows) > 1:
        print(f'\n!! MIXED SLIM WINDOWS in this tree -- do not publish as one '
              f'dataset:')
        for w, ks in sorted(windows.items(), key=lambda x: -len(x[1])):
            print(f'   {w} ns: {len(ks)} file(s)' +
                  (f'   e.g. {ks[0][0]}/{ks[0][1]} x {ks[0][2]}'
                   if len(ks) < 5 else ''))
    elif windows:
        print(f'slim window: {list(windows)[0]:g} ns on all {len(done)} file(s)')

    bad = []
    for k, (f, qa) in sorted(done.items()):
        if not qa:
            bad.append((k, 'no qa.json'))
        elif qa.get('efficiency', 0) < MIN_EFFICIENCY:
            bad.append((k, f'efficiency {qa["efficiency"]:.2%}'))
        elif qa.get('accidental', 1) > MAX_ACCIDENTAL:
            bad.append((k, f'accidental {qa["accidental"]:.2%}'))
    if done:
        eff = [q.get('efficiency') for _, q in done.values() if q.get('efficiency')]
        if eff:
            print(f'efficiency: min {min(eff):.2%}  median '
                  f'{sorted(eff)[len(eff)//2]:.2%}  max {max(eff):.2%}')
    print(f'suspect  : {len(bad)}')
    for k, why in bad:
        print(f'   {k[0]}/{k[1]} x {k[2]}   {why}')

    if a.missing:
        miss = sorted(k for k in exp if k not in done)
        print(f'\nnot yet done ({len(miss)}):')
        for run, sub, ntof in miss:
            print(f'   {run}/{sub} x {ntof}')

    if a.cluster:
        import subprocess
        q = subprocess.run(['condor_q', a.cluster, '-af', 'JobStatus', 'Args'],
                           capture_output=True, text=True).stdout.split('\n')
        names = {1: 'idle', 2: 'running', 5: 'held'}
        live = {}
        for ln in q:
            p = ln.split()
            if len(p) == 2:
                live.setdefault(names.get(int(p[0]), p[0]), []).append(p[1])
        print(f'\ncluster {a.cluster}: ' +
              ('  '.join(f'{k} {len(v)}' for k, v in sorted(live.items()))
               or 'nothing left in the queue'))
        for k in ('held',):
            if k in live:
                print(f'   {k}: {" ".join(sorted(live[k]))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
