#!/usr/bin/env python3
"""decompose.py -- prompt vs delayed neighbour charge, the central strip as basis.

On the head-on stacks of neighbour_vs_time.py (trimmed means of W(offset, t),
per event normalised to the centre peak):

    W(+-1) = a1 W0 + b1 LP_tau(W0)          W(+-2) = a2 W0 + b2 LP_tau(LP_tau(W0))

LP_tau = one-pole low-pass with unit area (the share_lp kernel form), tau
scanned on a grid, (a, b) by linear least squares; +-2 shares tau.  The centre
strip is its own impulse x charge-profile basis, so neither the template nor
the drift profile enters.  a1 is the PROMPT neighbour fraction (the cloud /
footprint), b1 the delayed copy amplitude, tau its RC time.

    decompose.py [--tmax 1300]
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = 30.0


def lp(x, tau):
    a = np.exp(-STEP / tau)
    y = np.empty_like(x); acc = 0.0
    for i, v in enumerate(x):
        acc = acc * a + v * (1 - a); y[i] = acc
    return y


def fit(t, m, tmax):
    w0 = np.nan_to_num(np.array(m[2], float))
    n1 = 0.5 * (np.array(m[1], float) + np.array(m[3], float))
    n2 = 0.5 * (np.array(m[0], float) + np.array(m[4], float))
    ok = np.isfinite(n1) & np.isfinite(n2) & (t <= tmax) & (t >= -200)
    best = None
    for tau in np.arange(30, 1500, 10):
        l1 = lp(w0, tau); l2 = lp(l1, tau)
        A1 = np.c_[w0, l1][ok]; A2 = np.c_[w0, l2][ok]
        c1, r1 = np.linalg.lstsq(A1, n1[ok], rcond=None)[:2]
        c2, r2 = np.linalg.lstsq(A2, n2[ok], rcond=None)[:2]
        s = float(r1[0] + r2[0]) if len(r1) and len(r2) else np.inf
        if best is None or s < best[0]:
            best = (s, tau, c1, c2)
    s, tau, c1, c2 = best
    # tau from +-1 alone (does +-2 pull it?)
    b1 = None
    for t1 in np.arange(30, 1500, 10):
        A1 = np.c_[w0, lp(w0, t1)][ok]
        c, r = np.linalg.lstsq(A1, n1[ok], rcond=None)[:2]
        if b1 is None or r[0] < b1[0]:
            b1 = (r[0], t1, c)
    return dict(tau=float(tau), a1=float(c1[0]), b1=float(c1[1]), a2=float(c2[0]), b2=float(c2[1]),
                tau_pm1_only=float(b1[1]), a1_pm1_only=float(b1[2][0]), b1_pm1_only=float(b1[2][1]),
                rms=float(np.sqrt(s / ok.sum())), tmax=float(t[ok].max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tmax', type=float, default=1300.0)
    a = ap.parse_args()
    d = json.load(open(f'{HERE}/results/neighbour_vs_time.json'))
    out = {}
    print(f'{"":8s}   tau   a1(prompt)  b1(delayed)   a2     b2   | tau(+-1 only) a1 b1 | rms')
    for name, v in d.items():
        for p in ('x', 'y'):
            q = v[p]; t = np.array(q['t'])
            m = [[np.nan if x is None else x for x in row] for row in q['m']]
            r = fit(t, m, a.tmax)
            out[f'{name}_{p}'] = r
            print(f'{name:7s}{p} {r["tau"]:5.0f}   {r["a1"]:.3f}      {r["b1"]:.3f}     '
                  f'{r["a2"]:+.3f} {r["b2"]:.3f} | {r["tau_pm1_only"]:5.0f} {r["a1_pm1_only"]:.3f} '
                  f'{r["b1_pm1_only"]:.3f} | {r["rms"]:.4f}  (to {r["tmax"]:.0f} ns)')
    json.dump(out, open(f'{HERE}/results/decompose_tmax{int(a.tmax)}.json', 'w'), indent=1)


main()
