#!/usr/bin/env python3
"""bench_kernel.py -- measure the BENCH detector's sharing kernel the same
model-free way, and settle whether the beam's constants transfer.

The beam pins det4's Y-plane kernel at tau ~ 1000 ns, c ~ 0.65.  Transplanting
that onto det3's bench calibration makes the bench WORSE (sigma_theta_Y
1.14 -> 1.51 deg).  Either the constant does not transfer between chambers, or
one of the two measurements is wrong.  The cross-relation settles it without
either fit being involved.

Near-vertical cosmics are as good a source as the beam for this: at normal
incidence the ionisation column sits at ONE transverse position, so every strip
is driven by the same C(t) and

        n_0 (*) W_d  ==  n_d (*) W_0

holds exactly, whatever the column's depth extent.  Selecting |tan| < 0.05 from
the calibration cache gives that geometry directly, from the same events the
hyper fit trains on.

    ../../../.venv/bin/python bench_kernel.py [--view y] [--tan-max 0.05]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares

import forms

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/dylan/PycharmProjects/nTof_x17'
CACHE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
         'long_run_resist_490V_drift_1000V/mx17_3/wft/calib_work/calib_cache.pkl')
PITCH = 0.78
SNS = 60.0
DLIST = (1, -1, 2, -2)


def trim_mean(A, frac=0.20):
    out = np.zeros(A.shape[1])
    for s in range(A.shape[1]):
        c = A[:, s]
        c = c[np.isfinite(c)]
        if len(c) == 0:
            continue
        c = np.sort(c)
        k = int(len(c) * frac)
        out[s] = (c[k:len(c) - k] if len(c) > 2 * k else c).mean()
    return out


def build(view='y', tan_max=0.05, nrel=12, cache=CACHE, q_lo=80.0):
    with open(cache, 'rb') as f:
        events = pickle.load(f)
    rows = {d: [] for d in (0,) + DLIST}
    nev = 0
    for eid in sorted(events):
        ev = events[eid]
        if view not in ev or abs(ev[f'tan_{view}']) > tan_max:
            continue
        P = ev[view]
        W = np.asarray(P['W'], float)
        pos = np.asarray(P['pos'], float)
        # per-strip baseline from the first three samples, then the leading
        # strip -- the same convention as the beam stacks
        W = W - W[:, :3].mean(axis=1)[:, None]
        pk = W.max(axis=1)
        i0 = int(np.argmax(pk))
        if pk[i0] < q_lo:
            continue
        s0 = int(np.argmax(W[i0]))
        if not (nrel <= s0 < W.shape[1] - nrel // 2):
            continue
        sidx = np.round((pos - pos[i0]) / PITCH).astype(int)
        cols = s0 + np.arange(-nrel, nrel + 1)
        ok = (cols >= 0) & (cols < W.shape[1])
        for d in (0,) + DLIST:
            j = np.flatnonzero(sidx == d)
            v = np.full(2 * nrel + 1, np.nan)
            if len(j):
                v[ok] = W[j[0], cols[ok]] / pk[i0]
            rows[d].append(v)
        nev += 1
    A = {d: np.array(rows[d], float) for d in rows}
    t = (np.arange(2 * nrel + 1) - nrel) * SNS
    return A, t, nev


SPEC = {
    'cascade': (('tau', 'c', 'q1', 'q1m', 'q2', 'q2m'),
                (600.0, 0.4, 0.25, 0.25, 0.05, 0.05),
                (30.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (4000.0, 0.95, 2.0, 2.0, 1.0, 1.0)),
    'delay':   (('tau', 'c1', 'c2', 'sigma_s', 'q1', 'q1m', 'q2', 'q2m'),
                (150.0, 0.3, 0.1, 60.0, 0.25, 0.25, 0.05, 0.05),
                (20.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0),
                (2000.0, 1.5, 1.5, 3000.0, 2.0, 2.0, 1.0, 1.0)),
    'geom':    (('q1', 'q1m', 'q2', 'q2m'), (0.4, 0.4, 0.1, 0.1),
                (0.0, 0.0, 0.0, 0.0), (3.0, 3.0, 2.0, 2.0)),
}


def fit(form, W, lo, hi):
    nm, x0, blo, bhi = SPEC[form]
    n = len(W[0])

    def res(x):
        p = dict(zip(nm, x))
        q = {0: 1.0, 1: p['q1'], -1: p['q1m'], 2: p['q2'], -2: p['q2m']}
        return forms.cross_resid(forms.build_n(form, 2, q, p, n), W, DLIST,
                                 lo, hi)

    r = least_squares(res, np.array(x0, float), bounds=(blo, bhi),
                      xtol=1e-12, ftol=1e-12)
    p = dict(zip(nm, r.x))
    q = {0: 1.0, 1: p['q1'], -1: p['q1m'], 2: p['q2'], -2: p['q2m']}
    nn = forms.build_n(form, 2, q, p, n)
    sc = max(np.abs(np.convolve(nn[0], W[1])[:n]).max(), 1e-12)
    return dict(par={k: float(v) for k, v in zip(nm, r.x)},
                rms_pct=float(100 * np.sqrt(np.mean(r.fun ** 2)) / sc),
                x=[float(v) for v in r.x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', default='y')
    ap.add_argument('--tan-max', type=float, default=0.05)
    ap.add_argument('--nrel', type=int, default=12)
    ap.add_argument('--nboot', type=int, default=200)
    args = ap.parse_args()

    A, t, nev = build(args.view, args.tan_max, args.nrel)
    print(f'bench det3, {args.view.upper()} view, |tan| < {args.tan_max}: '
          f'{nev} events, window {t[0]:+.0f} .. {t[-1]:+.0f} ns')
    W = {d: trim_mean(A[d]) for d in A}
    lo, hi = 2, len(t)
    print('  peak ratios:  ' + '  '.join(
        f'd={d:+d}: {W[d].max():.3f}' for d in (0,) + DLIST))

    rng = np.random.default_rng(20260818)
    out = {'n_events': nev, 'view': args.view, 'tan_max': args.tan_max}
    for form in ('cascade', 'delay', 'geom'):
        f = fit(form, W, lo, hi)
        bs = []
        for _ in range(args.nboot):
            idx = rng.integers(0, nev, nev)
            Wb = {d: trim_mean(A[d][idx]) for d in A}
            try:
                bs.append(fit(form, Wb, lo, hi)['x'])
            except Exception:
                pass
        B = np.array(bs, float)
        nm = SPEC[form][0]
        err = {k: float(B[:, i].std()) for i, k in enumerate(nm)} if len(B) > 5 else {}
        f['err'] = err
        out[form] = f
        print(f'  {form:8} rms {f["rms_pct"]:6.2f} %   ' + '  '.join(
            f'{k}={v:.4g}' + (f'+-{err[k]:.3g}' if k in err else '')
            for k, v in f['par'].items()))

    p = os.path.join(HERE, f'bench_kernel_{args.view}.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {p}')


if __name__ == '__main__':
    main()
