#!/usr/bin/env python3
"""bench_ladder.py -- template-free charge vs depth on inclined bench tracks, BOTH views.

The bench control for ladder_profile.py.  Full waveforms (no zero suppression,
no packet loss).  For each view, tracks with ANG_LO <= |tan| <= ANG_HI cross
several strips; strip k's depth is z = (pos - ref_mesh) / tan (M3 reference; the
same placement as charge_vs_depth.py).  Per strip: integral Q (whole window) and
peak A.  Each event is normalised by its mean interior Q (a per-event constant,
which cannot tilt the profile), then the per-depth-bin mean is taken.  A flat
profile = no attachment.  Y also carries RC spreading along the strip direction,
which moves charge between neighbouring Y strips but does not create a slope.

    bench_ladder.py det2=<big_cache.pkl> det3=... ...
Output: results/bench_ladder.json
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANG_LO, ANG_HI = 0.15, 0.45
ZB = np.arange(0.0, 30.01, 3.0)
NBOOT = 300


def prof(ev, view):
    rows = []
    for e in ev.values():
        if view not in e:
            continue
        t = e[f'tan_{view}']
        if not ANG_LO <= abs(t) <= ANG_HI:
            continue
        P = e[view]
        pos = np.asarray(P['pos']); W = np.asarray(P['W'], float)
        z = (pos - e[f'ref_mesh_{view}']) / t
        inside = (z > 1.5) & (z < 28.5)
        if inside.sum() < 4:
            continue
        Q = W.sum(1); A = W.max(1)
        nq, na = Q[inside].mean(), A[inside].mean()
        if nq <= 0 or na <= 0:
            continue
        rows.append((z[inside], Q[inside] / nq, A[inside] / na))
    return rows


def binned(rows, idx):
    z = np.concatenate([rows[i][0] for i in idx])
    q = np.concatenate([rows[i][1] for i in idx])
    a = np.concatenate([rows[i][2] for i in idx])
    out = []
    for lo, hi in zip(ZB[:-1], ZB[1:]):
        m = (z >= lo) & (z < hi)
        out.append((q[m].mean() if m.sum() > 20 else np.nan, a[m].mean() if m.sum() > 20 else np.nan))
    return np.array(out)


def main():
    rng = np.random.default_rng(3)
    res = {'z': (0.5 * (ZB[:-1] + ZB[1:])).tolist()}
    for arg in sys.argv[1:]:
        det, path = arg.split('=', 1)
        ev = pickle.load(open(path, 'rb'))
        res[det] = {}
        for view in ('x', 'y'):
            rows = prof(ev, view)
            if len(rows) < 30:
                continue
            B = binned(rows, range(len(rows)))
            boots = [binned(rows, rng.integers(0, len(rows), len(rows))) for _ in range(NBOOT)]
            err = np.nanstd(boots, axis=0)
            # deep / shallow: 18-27 mm over 3-12 mm
            def ratio(b, col):
                return np.nanmean(b[6:9, col]) / np.nanmean(b[1:4, col])
            rq = ratio(B, 0); rqe = np.nanstd([ratio(b, 0) for b in boots])
            ra = ratio(B, 1); rae = np.nanstd([ratio(b, 1) for b in boots])
            res[det][view] = dict(n=len(rows), Q=B[:, 0].tolist(), A=B[:, 1].tolist(),
                                  Q_err=err[:, 0].tolist(), A_err=err[:, 1].tolist(),
                                  deep_over_shallow_Q=[float(rq), float(rqe)],
                                  deep_over_shallow_A=[float(ra), float(rae)])
            print(f'{det} {view}: n={len(rows):4d}  Q(z) ' + ' '.join(f'{x:.2f}' for x in B[:, 0]) +
                  f'   deep/shallow Q {rq:.3f}±{rqe:.3f}  A {ra:.3f}±{rae:.3f}')
    json.dump(res, open(os.path.join(HERE, 'results', 'bench_ladder.json'), 'w'))


if __name__ == '__main__':
    main()
