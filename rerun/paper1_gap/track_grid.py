#!/usr/bin/env python3
"""track_grid.py <work> -- which part of a cell map helps chamber A's angles, and how fine it should be.
Maps (w0 and v per cell, robust ladder line) are built on one run with the RECONSTRUCTED mid-gap position and applied to
the other run (bilinear between cell centres); w0, v constants re-measured on the run itself.  Arms per grid n x n:
w0+v, w0 only, v only.  -> results/track_grid.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gapmap as GM, track_test as TT, cellobs as CO  # noqa: E401,E402

UL, WL = (-1.7, 1.7), (-1.51, 1.51)


def table(work, R, p):
    d = GM.load_table(work, R['rel'])
    U = (d['x_p0'].values + 14 * d['x_tan_theta'].values - 200) / 100
    W = (d['y_p0'].values + 14 * d['y_tan_theta'].values - 200) / 100
    w = d[f'{p}_w'].values * 1e3; tr = d[f'tan_ref_{p}'].values; tp = d[f'{p}_tan_theta'].values
    s = np.abs(tr) <= GM.TAN_LADDER
    return U[s], W[s], w[s], tr[s], tp[s]


def cells(U, W, w, tr, n):
    eu, ew = np.linspace(*UL, n + 1), np.linspace(*WL, n + 1)
    i, j = np.searchsorted(eu, U) - 1, np.searchsorted(ew, W) - 1
    out = []
    for a in range(n):
        for b in range(n):
            m = (i == a) & (j == b)
            w0, v, *_ = CO.line(tr[m], w[m])
            out.append(dict(U=0.5 * (eu[a] + eu[a + 1]), W=0.5 * (ew[b] + ew[b + 1]), w0=w0, v=v))
    vm = np.nanmean([c['v'] for c in out]); wm = np.nanmean([c['w0'] for c in out])
    for c in out:
        c['vrel'] = c['v'] / vm; c['dw0'] = c['w0'] - wm
    return out


def main():
    work = sys.argv[1]; out = {}
    other = {'run1': 'run2', 'run2': 'run1'}
    for run, R in GM.RUNS.items():
        out[run] = {}
        for p in 'xy':
            U, W, w, tr, tp = table(work, R, p)
            Uo, Wo, wo, tro, _ = table(work, GM.RUNS[other[run]], p)
            res = {'pass1': TT.metrics(tp, tr, U, W)}
            for n in (4, 6, 8, 10):
                C = cells(Uo, Wo, wo, tro, n)
                vr = TT.bilinear(C, 'vrel', U, W); dw = TT.bilinear(C, 'dw0', U, W)
                for arm, (dd, vv) in {'w0+v': (dw, vr), 'w0': (dw, np.ones_like(vr)), 'v': (np.zeros_like(dw), vr)}.items():
                    c = TT.robust_const(tr, w, dd, vv)
                    res[f'{n}x{n} {arm}'] = TT.metrics((w - c[0] - dd) / (c[1] * vv), tr, U, W)
            out[run][p] = res
            print(run, p, '  '.join(f'{k}: {v["s68"]:.3f}' for k, v in res.items()), flush=True)
    json.dump(out, open(f'{HERE}/results/track_grid.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
