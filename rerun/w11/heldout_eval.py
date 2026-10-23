#!/usr/bin/env python3
"""heldout_eval.py -- score a (cache, bundle) pair on the cache's held-out events against the M3 reference.

Production fit (blind start from init_guess, t0 prior sigma 5 ns from the bundle's t0 table) with the NNLS column
guard on (W2) and the bundle's depth grid.  Per plane: angle residual (s68, median), position residual (s68 around
its median: the reference is pinned to the hits alignment, so only the spread is meaningful), chi2/dof, far
fraction (|dtheta| > 10 deg), runaway fraction (q_sum > 1e6 ADC).  Events = cache ids from 180 on (the first 180 are
the calibration training set).

    heldout_eval.py --cache cache_cm5.pkl --bundle <bundle dir> [--depth-bins 24] --out eval.json [--max 1300]
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
_EV = None
T0_SIGMA = 5.0


def _init(cache, bundle, k, guard):
    global _EV
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle))
    if k:
        wm.set_depth_bins(k)
    wm.COL_GUARD = guard


def one(eid):
    from wft import model as wm
    cal = wm.CAL
    ev = _EV[eid]
    out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        ft = ev.get(f'ftst_{plane}')
        t0p = (cal.t0_abs or {}).get(plane, {}).get(str(ft), (cal.t0_abs or {}).get(plane, {}).get(ft))
        p0i, wi, t0g = wm.init_guess(P, plane, 0.0, None, cal.v_drift)
        prior = (float(t0p), T0_SIGMA) if t0p is not None else None
        try:
            r = wm.fit_plane_raw(P, plane, p0i, wi, float(t0p) if t0p is not None else t0g, t0_prior=prior)
        except Exception:
            r = None
        if r is None or not np.isfinite(r['chi2']):
            continue
        tan = (r['w'] * 1e3 - cal.w0.get(plane, 0.0)) / (cal.kw.get(plane, 1.0) * cal.v_drift)
        out[plane] = dict(tan=float(tan), tr=float(ev[f'tan_{plane}']), p0=float(r['p0']),
                          p0r=float(ev[f'ref_mesh_{plane}']), q=float(np.sum(r['q'])),
                          chi2=float(r['chi2'] / max(r['dof'], 1)))
    return eid, out


def s68(x):
    x = np.asarray(x, float)
    return float(np.percentile(np.abs(x - np.median(x)), 68.27)) if len(x) else float('nan')


def summarise(res):
    S = {}
    for plane in ('x', 'y'):
        rows = [r[plane] for r in res.values() if plane in r]
        if not rows:
            continue
        th = np.degrees(np.arctan([r['tan'] for r in rows])) - np.degrees(np.arctan([r['tr'] for r in rows]))
        dp = np.array([r['p0'] - r['p0r'] for r in rows])
        ok = np.abs(th) < 10
        S[plane] = dict(n=len(rows), s68_deg=s68(th[ok]), bias_deg=float(np.median(th[ok])),
                        frac_far=float(1 - ok.mean()), pos_s68_mm=s68(dp[ok]),
                        chi2_median=float(np.median([r['chi2'] for r in rows])),
                        frac_q_gt_1e6=float(np.mean([r['q'] > 1e6 for r in rows])))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--depth-bins', type=int, default=None)
    ap.add_argument('--guard', type=float, default=0.01)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=1300)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    ids = sorted(pickle.load(open(a.cache, 'rb')))[180:180 + a.max]
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.depth_bins, a.guard)) as pool:
        res = dict(pool.map(one, ids, chunksize=8))
    S = summarise(res)
    for p, s in S.items():
        print(p, json.dumps(s))
    json.dump(dict(cache=a.cache, bundle=a.bundle, depth_bins=a.depth_bins, guard=a.guard, summary=S, events=res),
              open(a.out, 'w'), default=float)


if __name__ == '__main__':
    main()
