#!/usr/bin/env python3
"""edges.py --bundle B --out o.json CACHE.pkl [...] [--n 1000] [--boot 200]

Column duration from the two EDGES of the stacked strip-sum waveform, model-free.  The strip sum of any track is the
box of arriving charge convolved with the electronics response (boxfit.py docstring); its rising edge is where the
first (mesh-side) charge arrives and its falling edge where the last (cathode-side) charge arrives.  T_edge = t(steepest
fall) - t(steepest rise), each located by a parabola through the extreme sample of the first difference.  A constant
pre-level (beam pile-up, ZS) does not move either edge, and a linear charge gradient along the column moves them by
the same amount to first order: the closure on box (x) template toys (``--toys``) gives the residual bias, which is
quoted.  Errors by bootstrap over events.
"""
import argparse, json, os, pickle, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
from wft.calib import CalibrationBundle          # noqa: E402
from boxfit import Kern, TS                       # noqa: E402

DTS = 60.0


def _unused_vertex(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]; d = a - 2 * b + c
    return float(i + (0.5 * (a - c) / d if d != 0 else 0.0))


def _deriv_guess(s):
    d = np.diff(s)
    ir = 6 + int(np.argmax(d[6:25]))
    j = min(ir + 15, len(d) - 2)
    ifl = j + int(np.argmin(d[j:]))
    return (ir + 1.0) * DTS, (ifl + 1.0) * DTS


def _edge_fit(t, y, tc, rising):
    """Local fit of one edge: y = B + (P + s (t - te)) Phi(+-(t - te)/sig) on te +- 480 ns."""
    from scipy.optimize import curve_fit
    from scipy.special import ndtr
    m = np.abs(t - tc) <= 480
    sg = 1.0 if rising else -1.0

    def f(x, B, P, s, te, sig):
        return B + (P + s * (x - te)) * ndtr(sg * (x - te) / sig)
    P0 = np.nanmax(y[m]) - np.nanmin(y[m])
    try:
        p, _ = curve_fit(f, t[m], y[m], p0=[np.nanmin(y[m]), P0, 0.0, tc, 120.0],
                         bounds=([-1, 0, -1e-2, tc - 400, 20], [1, 2, 1e-2, tc + 400, 600]), maxfev=20000)
        return float(p[3])
    except Exception:
        return np.nan


def edge_times(st, guess=None):
    """(t_rise, t_fall): erf-edge fits around the steepest rise (390-1530 ns) and the steepest fall (>= 900 ns later);
    ``guess`` fixes the two fit centres (bootstrap replicas use the full stack's)."""
    s = np.asarray(st, float)
    ok = np.isfinite(s)
    if ok.sum() < len(s):
        s = np.interp(np.arange(len(s)), np.flatnonzero(ok), s[ok])
    t = np.arange(len(s)) * DTS
    gr, gf = guess if guess is not None else _deriv_guess(s)
    return _edge_fit(t, s, gr, True), _edge_fit(t, s, gf, False)


def stack(C, view, n, ids=None):
    out = []
    for eid in (ids if ids is not None else sorted(C)[:n]):
        ev = C[eid]
        if view not in ev:
            continue
        W = np.asarray(ev[view]['W'], float); S = W.sum(0); ok = ~(W == 0).all(0)
        if ok.sum() < 30:
            continue
        out.append(np.where(ok, S / S[ok].max(), np.nan))
    return np.array(out)


def toys(kern):
    """T_edge - T_true for box (x) template toys: flat and with the measured gradients, a 30 ns end smear."""
    rows = []
    for T in (1500, 2000, 2400, 2800):
        for g in (0.0, -0.3, -0.5, -0.8):
            for t0 in (600, 630, 660):
                y = kern(t0, T) @ np.array([1.0, g]); y = y / y.max()
                tr, tf = edge_times(y)
                rows.append((T, g, t0, tf - tr - T))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('caches', nargs='*'); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1000); ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--views', default='xy')
    a = ap.parse_args()
    cal = CalibrationBundle.load(a.bundle)
    tm = np.asarray(cal.tmpl['x'], float); kern = Kern(np.asarray(cal.grid, float), tm / tm.max(), 30.0)
    R = toys(kern)
    res = dict(toys=dict(rows=R.tolist(), bias_med=float(np.median(R[:, 3])), bias_min=float(R[:, 3].min()), bias_max=float(R[:, 3].max())))
    print(f'toys: T_edge - T  median {np.median(R[:, 3]):+.0f} ns, range {R[:, 3].min():+.0f} .. {R[:, 3].max():+.0f}')
    rng = np.random.default_rng(1)
    for c in a.caches:
        C = pickle.load(open(c, 'rb')); key = os.path.basename(c).replace('.pkl', ''); res[key] = {}
        for view in a.views:
            S = stack(C, view, a.n)
            if len(S) < 30:
                continue
            tr, tf = edge_times(np.nanmedian(S, 0))
            bs = []
            for _ in range(a.boot):
                Sb = S[rng.integers(0, len(S), len(S))]
                r2, f2 = edge_times(np.nanmedian(Sb, 0), guess=(tr, tf)); bs.append((r2, f2, f2 - r2))
            bs = np.array(bs)
            res[key][view] = dict(n=int(len(S)), t_rise=tr, t_fall=tf, T_edge=tf - tr, T_err=float(np.nanstd(bs[:, 2])),
                                  rise_err=float(np.nanstd(bs[:, 0])), fall_err=float(np.nanstd(bs[:, 1])),
                                  stack=[None if not np.isfinite(x) else float(x) for x in np.nanmedian(S, 0)])
            print(f'{key} {view}: n {len(S)}  rise {tr:.0f}  fall {tf:.0f}  T_edge {tf - tr:.0f} ± {np.nanstd(bs[:, 2]):.0f} ns', flush=True)
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
