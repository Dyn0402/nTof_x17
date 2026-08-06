#!/usr/bin/env python3
"""
rc_line_step2.py — does the Y drain depend on position along the strip?

Step 1 showed T_Y = T_X (x) drain(tau_g ~ 7 us). If that drain is resistive
evacuation along the strip, tau_g must vary with where the charge lands
(distance to the grounded end) — an electronics difference would not. Build
per-position-bin Y templates from the calibration cache (the track's x from
the reference gives the coordinate along the Y strip) and fit tau_g per bin.
Mirror test for X binned in y.

    ../../.venv/bin/python mx_june_wft/bench/rc_line_step2.py sat_det3
"""
import argparse
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

TEMPLATE_GRID = np.arange(-360, 1400, 10.0)
TAN_MIN = 0.10
MIN_AMP = 300.0
DT = 2.0
T_MAX = 1800.0


def t50(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.5 * a > w[k - 1]:
            return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
    return np.nan


def build_template(cands, sample_ns=60.0):
    acc = []
    for w in cands:
        a = w.max()
        c = t50(w)
        if not np.isfinite(c):
            continue
        tt = (np.arange(len(w)) - c) * sample_ns
        acc.append(np.interp(TEMPLATE_GRID, tt, w / a, left=np.nan, right=np.nan))
    if len(acc) < 8:
        return None, len(acc)
    t = np.nanmedian(np.array(acc), axis=0)
    t -= np.nanmedian(t[TEMPLATE_GRID < -250])
    return np.nan_to_num(t), len(acc)


def drained(tmpl_grid, te, tau_g):
    """T_e convolved with (delta + d/dt exp(-t/tau_g)) on the fine grid."""
    te_t = np.arange(tmpl_grid[0], T_MAX, DT)
    tef = np.interp(te_t, tmpl_grid, te, left=0, right=0)
    tg = np.arange(0.0, T_MAX, DT)
    g = -(1.0 / tau_g) * np.exp(-tg / tau_g)
    g[0] += 1.0 / DT
    full = np.convolve(tef, g)[:len(tef)] * DT
    return te_t, full / full.max()


def fit_tau(tmpl_grid, te, target):
    fit_t = np.arange(-100.0, 1400.0, 10.0)
    tm = np.interp(fit_t, tmpl_grid, target)

    def loss(lt):
        tau = float(np.exp(lt))
        t, r = drained(tmpl_grid, te, tau)
        return float(((np.interp(fit_t, t, r, left=0, right=0) - tm) ** 2).sum())

    res = minimize_scalar(loss, bounds=(np.log(300), np.log(300000)),
                          method='bounded', options=dict(xatol=1e-3))
    return float(np.exp(res.x)), res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    import pickle
    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    cal = CalibrationBundle.load(os.path.join(W, 'calib_bundle'))
    events = pickle.load(open(os.path.join(W, 'calib_work', 'calib_cache.pkl'),
                              'rb'))
    print(f'{len(events)} cached events')

    # collect bright-strip waveforms per plane with the ALONG-strip coordinate
    cands = {'x': [], 'y': []}
    for ev in events.values():
        for plane, other in (('x', 'y'), ('y', 'x')):
            if plane not in ev or abs(ev[f'tan_{plane}']) < TAN_MIN:
                continue
            along = ev.get(f'ref_mesh_{other}')     # position along this strip
            if along is None or not np.isfinite(along):
                continue
            Wf = np.asarray(ev[plane]['W'], np.float32)
            amax = Wf.max(axis=1)
            for i in np.argsort(amax)[::-1][:2]:
                w = Wf[i]
                a = w.max()
                ipk = int(np.argmax(w))
                ns = len(w)
                if a < MIN_AMP or a > 3550 or ipk < 6 or ipk > ns - 12:
                    continue
                cands[plane].append((along, w))

    for plane in ('y', 'x'):
        arr = cands[plane]
        if not arr:
            continue
        alongs = np.array([a for a, _ in arr])
        edges = np.percentile(alongs, [0, 25, 50, 75, 100])
        te = np.asarray(cal.tmpl['x' if plane == 'y' else 'y'], float)
        # reference electronics template: use the OTHER plane's measured
        # template only for Y (X-as-electronics); for the X test use X's own
        # global template as the reference shape
        te = np.asarray(cal.tmpl['x'], float)
        te = te / te.max()
        print(f'\n== {plane} templates binned by position along the strip '
              f'(n={len(arr)})')
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [w for a, w in arr if lo <= a < hi or (hi == edges[-1] and a == hi)]
            t, n = build_template(sel)
            if t is None:
                print(f'  [{lo:6.1f},{hi:6.1f}) n={n}: too few')
                continue
            tau, resid = fit_tau(TEMPLATE_GRID, te, t / t.max())
            us = np.nanmin(t)
            print(f'  along [{lo:6.1f},{hi:6.1f}) mm  n={n:3d}: '
                  f'undershoot {us:+.3f}  tau_g {tau/1000:6.2f} us  '
                  f'(resid {resid:.4f})')


if __name__ == '__main__':
    main()
