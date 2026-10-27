#!/usr/bin/env python3
"""make_jobs.py -- the FD condor job list (one line of arguments per job) for fd.sub, and the code tarball.

    make_jobs.py <staging dir>   -> <dir>/jobs.txt, <dir>/fd_code.tgz
"""
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')


def main():
    st = sys.argv[1]
    os.makedirs(st, exist_ok=True)
    lines = []
    for det in DETS:
        for b in (20, 30, 60, 90):
            for cfg in ('full', 'no_landau'):
                lines.append(f'scan_a --det {det} --dtbin {b} --config {cfg} --n 100')
    for p in (0.5, 1.2):
        for b in (30, 60):
            lines.append(f'scan_a --det det3 --dtbin {b} --config full --pitch {p} --n 100')
    for s0 in (0.10, 0.45):
        lines.append(f'scan_a --det det3 --dtbin 60 --config full --s0 {s0} --n 100')
    for det in DETS:
        for rung in ('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'):
            for ang in (3, 9, 16):
                lines.append(f'ladder --det {det} --rungs {rung} --angles {ang} --n 150')
                if rung not in ('L0', 'L1'):
                    lines.append(f'ladder --det {det} --rungs {rung} --angles {ang} --n 150 --alone')
    open(os.path.join(st, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
    with tarfile.open(os.path.join(st, 'fd_code.tgz'), 'w:gz') as t:
        t.add(os.path.join(REPO, 'wft'), 'wft', filter=lambda ti: None if '__pycache__' in ti.name or
              '/archive' in ti.name or '/tests' in ti.name else ti)
        t.add(os.path.join(REPO, 'rerun', 'model_audit', 'ws_floor', 'floor_toy.py'),
              'rerun/model_audit/ws_floor/floor_toy.py')
        for f in ('ladder.py', 'scan_a.py'):
            t.add(os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', f),
                  f'rerun/model_audit/followups/fd_floor_realism/{f}')
        for det in DETS:
            for f in ('bundle.json', 'arrays.npz'):
                t.add(os.path.join(AUDIT, det, 'calib_bundle', f), f'bundles/{det}/calib_bundle/{f}')
    print(len(lines), 'jobs')


if __name__ == '__main__':
    main()
