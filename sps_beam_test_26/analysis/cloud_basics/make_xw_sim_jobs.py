#!/usr/bin/env python3
"""make_xw_sim_jobs.py [--nrep 100] [--chunk 20] [--sets ...] [--arms ...] -> jobs_xw_sim.txt (set arm seed0 nrep)"""
import argparse
from xw_sim import SETS, ARMS
ap = argparse.ArgumentParser()
ap.add_argument('--nrep', type=int, default=100); ap.add_argument('--chunk', type=int, default=20)
ap.add_argument('--sets', default=','.join(SETS)); ap.add_argument('--arms', default=','.join(ARMS))
ap.add_argument('--out', default='jobs_xw_sim.txt')
a = ap.parse_args()
with open(a.out, 'w') as f:
    for s in a.sets.split(','):
        for i, arm in enumerate(a.arms.split(',')):
            for c in range(0, a.nrep, a.chunk):
                f.write(f'{s} {arm} {100000 * (i + 1) + c} {min(a.chunk, a.nrep - c)}\n')
print(open(a.out).read().count('\n'), 'jobs ->', a.out)
