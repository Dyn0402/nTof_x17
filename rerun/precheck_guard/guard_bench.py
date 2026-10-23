#!/usr/bin/env python3
"""guard_bench.py -- the NNLS column guard (wft.model.COL_GUARD) on real bench events, paired against production.

Pre-run work package W2 (rerun/RERUN_PLAN.md).  Held-out events of a bench cache (the first 180 ids are the
calibration training set and are skipped), each plane fitted twice with the production fit (blind start from
init_guess, t0 prior sigma 5 ns from the bundle's t0 table): COL_GUARD = 0 and = --guard.  Scored against the M3
reference: angle residual, position, charge sum, chi2.  The rule the guard must pass: fits whose charge is observable
are unchanged, the runaway class (q_sum > 1e6 ADC) disappears, and resolution is not worse.

    guard_bench.py --cache big_cache.pkl --bundle <bundle dir> --out guard_<det>.json [--max 1500] [--guard 0.01]
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


def _init(cache, bundle):
    global _EV
    os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle))


def one(payload):
    eid, guard = payload
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
        res = {}
        for tag, g in (('prod', 0.0), ('guard', guard)):
            wm.COL_GUARD = g
            try:
                r = wm.fit_plane_raw(P, plane, p0i, wi, float(t0p) if t0p is not None else t0g,
                                     t0_prior=prior)
            except Exception:
                r = None
            finally:
                wm.COL_GUARD = 0.0
            if r is None or not np.isfinite(r['chi2']):
                res[tag] = None
                continue
            tan = (r['w'] * 1e3 - cal.w0.get(plane, 0.0)) / (cal.kw.get(plane, 1.0) * cal.v_drift)
            res[tag] = dict(tan=float(tan), p0=float(r['p0']), t0=float(r['t0']), q=float(np.sum(r['q'])),
                            chi2=float(r['chi2'] / max(r['dof'], 1)))
        out[plane] = dict(tr=float(ev[f'tan_{plane}']), p0r=float(ev[f'ref_mesh_{plane}']), **res)
    return eid, out


def s68(x):
    x = np.asarray(x, float)
    return float(np.percentile(np.abs(x - np.median(x)), 68.27)) if len(x) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=1500)
    ap.add_argument('--guard', type=float, default=0.01)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    ev = pickle.load(open(a.cache, 'rb'))
    ids = sorted(ev)[180:180 + a.max]
    del ev
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle)) as pool:
        res = dict(pool.map(one, [(e, a.guard) for e in ids], chunksize=8))
    S = {}
    for plane in ('x', 'y'):
        rows = [r[plane] for r in res.values() if plane in r and r[plane].get('prod') and r[plane].get('guard')]
        tr = np.array([r['tr'] for r in rows])
        th = {t: np.degrees(np.arctan([r[t]['tan'] for r in rows])) - np.degrees(np.arctan(tr)) for t in ('prod', 'guard')}
        q = {t: np.array([r[t]['q'] for r in rows]) for t in ('prod', 'guard')}
        dt = np.array([r['guard']['tan'] - r['prod']['tan'] for r in rows])
        dp = np.array([r['guard']['p0'] - r['prod']['p0'] for r in rows])
        same = (np.abs(dt) < 1e-9) & (np.abs(dp) < 1e-9)
        ok = {t: np.abs(th[t]) < 10 for t in th}
        S[plane] = dict(n=len(rows), frac_identical=float(same.mean()),
                        frac_dtheta_gt_0p1deg=float(np.mean(np.abs(np.degrees(np.arctan(dt))) > 0.1)),
                        **{f'{t}_s68_deg': s68(th[t][ok[t]]) for t in th},
                        **{f'{t}_bias_deg': float(np.median(th[t][ok[t]])) for t in th},
                        **{f'{t}_frac_far': float(1 - ok[t].mean()) for t in th},
                        **{f'{t}_frac_q_gt_1e6': float(np.mean(q[t] > 1e6)) for t in q},
                        **{f'{t}_chi2_median': float(np.median([r[t]['chi2'] for r in rows])) for t in ('prod', 'guard')})
        print(plane, json.dumps(S[plane]))
    json.dump(dict(cache=a.cache, bundle=a.bundle, guard=a.guard, summary=S, events=res), open(a.out, 'w'), default=float)


if __name__ == '__main__':
    main()
