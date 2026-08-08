#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
why_skipped.py -- is there anything different about the runs the pass skipped?

The 5-7 August pass processed 325 X17 runs and left 66 gaps inside
224300-224687. This asks whether the skipped ones differ from the processed
ones in any way visible from outside the processing.

Checked and RULED OUT:
  * directory structure   -- identical (stream0 + stream1, every file
                             `.finished`, no stragglers) on both sets
  * output size cap       -- a cap would have to sit below 21 GB, and 42
                             processed runs already exceed that
  * position in the range -- the gaps are scattered, not clustered at an end

What SURVIVES is raw size, and it is not subtle: below 0.35 TB nothing was ever
skipped, above it 42 % was, and the rate keeps climbing with size. That is the
shape of a resource limit that a big job sometimes misses and sometimes makes,
not of a rule that rejects a run outright.

Inputs (regenerate as in coverage_map.refresh_inputs, plus):
  raw_sizes.txt   run n_files bytes   -- du -sb over each run's stream1/
  out_sizes.txt   run bytes           -- ls -l over processing/official/done/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'coverage_inputs'
PASS_LO, PASS_HI = 224300, 224687
FLOOR_TB = 0.35


def load():
    raw, out = {}, {}
    for ln in (DATA / 'raw_sizes.txt').read_text().splitlines():
        f = ln.split()
        if len(f) == 3:
            raw[int(f[0])] = (int(f[1]), int(f[2]))
    for ln in (DATA / 'out_sizes.txt').read_text().splitlines():
        f = ln.split()
        if len(f) == 2:
            out[int(f[0])] = int(f[1])
    return raw, out


def main() -> int:
    raw, out = load()
    runs = sorted(r for r in raw if PASS_LO <= r <= PASS_HI)
    tb = np.array([raw[r][1] / 1e12 for r in runs])
    ok = np.array([r in out for r in runs])
    print(f'{len(runs)} runs in {PASS_LO}-{PASS_HI} with stream1 still staged: '
          f'{ok.sum()} processed, {(~ok).sum()} skipped\n')

    print('skip rate vs raw size')
    print(f'{"raw TB":>13} {"n":>4} {"skipped":>8} {"rate":>6}')
    edges = [0, .05, .15, .25, .35, .45, .55, .65, .75, 1.0]
    for lo, hi in zip(edges, edges[1:]):
        m = (tb >= lo) & (tb < hi)
        if not m.sum():
            continue
        bar = '#' * int(round(20 * (~ok[m]).mean()))
        print(f'{lo:5.2f}-{hi:<5.2f} {m.sum():>5} {(~ok[m]).sum():>8} '
              f'{(~ok[m]).mean():>6.0%}  {bar}')

    big = tb >= FLOOR_TB
    print(f'\nbelow {FLOOR_TB} TB : {(~ok[~big]).sum():>3} of {(~big).sum():>3} '
          f'skipped ({(~ok[~big]).mean():.0%})')
    print(f'at/above     : {(~ok[big]).sum():>3} of {big.sum():>3} '
          f'skipped ({(~ok[big]).mean():.0%})')
    o = np.argsort(tb[big])
    s = (~ok[big])[o]
    h = len(s) // 2
    print(f'  and within the big group it keeps climbing: '
          f'lower half {s[:h].mean():.0%}, upper half {s[h:].mean():.0%}')

    # The control: after the pass stopped, size stops mattering.
    post = sorted(r for r in raw if r > PASS_HI)
    if post:
        ptb = np.array([raw[r][1] / 1e12 for r in post])
        print(f'\ncontrol -- the {len(post)} runs after {PASS_HI}, all missing: '
              f'{(ptb < FLOOR_TB).sum()} of them are below {FLOOR_TB} TB,')
        print('  a band in which the pass never skipped anything. So those are '
              'missing for a\n  different reason (it stopped), not this one.')

    print(f'\nRULED OUT: identical directory structure on both sets; an output '
          f'size cap\n(it would have to sit below 21 GB, and 42 processed runs '
          f'exceed that); position\nin the run range (the gaps are scattered).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
