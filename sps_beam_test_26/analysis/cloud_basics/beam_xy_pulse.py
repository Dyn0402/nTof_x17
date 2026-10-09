#!/usr/bin/env python3
"""beam_xy_pulse.py -- is the beam's late-pulse decline common to X and Y?

Unbiased head-on stacks (fixed-threshold alignment on the 9-strip sum of each
view, no per-event normalisation, plain mean), for strip sums of +-0, +-1, +-2,
+-4 around the centre.  Gas attachment removes electrons before they reach the
mesh, so it must tilt BOTH views' all-strip sums identically; charge spreading
out of a narrow sum tilts narrow sums more than wide ones; an electronics
(baseline / AC) effect tilts every sum of a view by the same factor of its
own signal.  Also: X and Y aligned on the SAME (X) threshold time, so the two
views' all-strip sums can be compared sample by sample."""
import json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES, SAMPLE_NS

GRID = np.arange(-300.0, 3600.0, 60.0)
THR = 60.0
COLS = (0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000)


def run(name):
    ev = pickle.load(open(SOURCES[name], 'rb'))
    acc = {}; cnt = {}; n = 0
    for e in ev.values():
        if abs(e['tan_x']) > 0.03 or abs(e['tan_y']) > 0.03:
            continue
        sums = {}
        for p in ('x', 'y'):
            W = np.asarray(e[p]['W'], float)
            ic = int(np.argmax(W.sum(1)))
            if ic < 4 or ic > len(W) - 5:
                break
            for h in (0, 1, 2, 4):
                sums[(p, h)] = W[ic - h:ic + h + 1].sum(0)
        else:
            s = sums[('x', 4)]
            k = next((k for k in range(1, len(s)) if s[k - 1] < THR <= s[k]), None)
            if k is None or k < 3:
                continue
            t = (np.arange(len(s)) - (k - 1 + (THR - s[k - 1]) / (s[k] - s[k - 1]))) * SAMPLE_NS
            for key, y in sums.items():
                yy = np.interp(GRID, t, y, left=np.nan, right=np.nan)
                ok = np.isfinite(yy)
                acc.setdefault(key, np.zeros(len(GRID)))[ok] += yy[ok]
                cnt.setdefault(key, np.zeros(len(GRID)))[ok] += 1
            n += 1
    m = {k: acc[k] / np.maximum(cnt[k], 1) for k in acc}
    pkx = np.nanmax(m[('x', 4)])
    print(f'== {name}  n={n} (both views head-on), all normalised to the X +-4 peak; t from the X threshold')
    print('            ' + ' '.join(f'{c:6d}' for c in COLS))
    for key in sorted(m):
        y = m[key] / pkx
        print(f'  {key[0]} +-{key[1]}     ' + ' '.join(f'{np.interp(c, GRID, y):6.3f}' for c in COLS))
    # shape only: each sum normalised to its own value at +300 ns
    print('  shape (each / its value at 300 ns):')
    for key in sorted(m):
        y = m[key]; y = y / np.interp(300, GRID, y)
        print(f'  {key[0]} +-{key[1]}     ' + ' '.join(f'{np.interp(c, GRID, y):6.3f}' for c in COLS))
    return {f'{k[0]}{k[1]}': (m[k] / pkx).tolist() for k in m}


NAMES = sys.argv[1:] or ['sps700', 'sps450', 'sps275']
out = {name: run(name) for name in NAMES}
out['t'] = GRID.tolist()
json.dump(out, open(f'{HERE}/results/xy_pulse_{"_".join(NAMES)}.json', 'w'))
