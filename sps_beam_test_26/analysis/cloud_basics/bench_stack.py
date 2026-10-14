#!/usr/bin/env python3
"""bench_stack.py -- the bench's arriving-current stack, both views, every chamber.

The sum over ALL strips of a view at each sample is the current arriving at the mesh,
whatever the track angle: the angle only redistributes it across strips.  So every
track with |tan_view| <= TAN_MAX whose whole ladder (30 mm * |tan| plus +-3 mm of
spread) lies inside the cached strip window contributes, not only head-on ones.
Events are placed by the trigger: t = sample * 60 ns - t0_abs[view][ftst] (the
bundle's per-fine-timestamp-class absolute t0).  No pulse-based alignment.  Per-event,
per-strip baseline from the first 4 samples; bootstrap band over events.

    bench_stack.py det2=<big_cache.pkl>:<bundle_dir> ...
Output: results/bench_stack.json
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNS = 60.0
TAN_MAX = 0.2
MARGIN = 3.0
GRID = np.arange(-400.0, 1600.0, 20.0)
NBOOT = 100


# det4's amplification stripes run across X: an inclined X track walks over live and
# dead stripes and loses depth-dependent parts of its charge.  Head-on only there.
TAN_MAX_DET = {('det4', 'x'): 0.03}


def rows(ev, view, t0a, tan_max=TAN_MAX):
    out = []
    for e in ev.values():
        if view not in e or abs(e[f'tan_{view}']) > tan_max:
            continue
        P = e[view]; pos = np.asarray(P['pos']); W = np.asarray(P['W'], float)
        lo = e[f'ref_mesh_{view}'] + min(0.0, 30.0 * e[f'tan_{view}']) - MARGIN
        hi = e[f'ref_mesh_{view}'] + max(0.0, 30.0 * e[f'tan_{view}']) + MARGIN
        if pos.min() > lo or pos.max() < hi:
            continue                                   # ladder not contained in the window
        t0 = t0a.get(str(int(e[f'ftst_{view}'])))
        if t0 is None:
            continue
        W = W - W[:, :4].mean(1, keepdims=True)
        s = W.sum(0)
        t = np.arange(len(s)) * SNS - t0
        out.append(np.interp(GRID, t, s, left=np.nan, right=np.nan))
    return np.array(out)


def stack(R):
    with np.errstate(invalid='ignore'):
        S = np.nanmean(R, axis=0)
    return S - np.nanmean(S[GRID < -150])


def main():
    rng = np.random.default_rng(17)
    res = {'grid': GRID.tolist(), 'tan_max': TAN_MAX}
    for arg in sys.argv[1:]:
        det, rest = arg.split('=', 1)
        cache, bdir = rest.split(':', 1)
        B = json.load(open(os.path.join(bdir, 'bundle.json')))
        if 't0_abs' not in B:
            print(f'{det}: no t0_abs in {bdir} -- skipped'); continue
        ev = pickle.load(open(cache, 'rb'))
        res[det] = {'v_bundle': B['v_drift']}
        for view in ('x', 'y'):
            R = rows(ev, view, B['t0_abs'][view], TAN_MAX_DET.get((det, view), TAN_MAX))
            S = stack(R); pk = np.nanmax(S)
            band = np.nanstd([stack(R[rng.integers(0, len(R), len(R))]) / pk for _ in range(NBOOT)], axis=0)
            res[det][view] = dict(n=int(len(R)), stack=(S / pk).tolist(), band=band.tolist())
            print(f'{det} {view}: n={len(R):5d}  stack/peak every 100 ns: ' +
                  ' '.join(f'{x:.2f}' for x in (S / pk)[::5]))
    json.dump(res, open(os.path.join(HERE, 'results', 'bench_stack.json'), 'w'))


if __name__ == '__main__':
    main()
