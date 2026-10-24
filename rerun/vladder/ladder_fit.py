#!/usr/bin/env python3
"""ladder_fit.py -- does the per-event reconstruction's transverse speed w depend on how the fit is started or on its
trigger t0 prior?  (rerun/vladder, 2026-10-24)

The per-event fit never uses v: it fits (p0, w, t0) and reports tan = (w - w0) / (kw v).  So dw/dtan_ref over held-out
events is a drift velocity of its own (the "ladder" v).  This refits held-out events of one (cache, bundle) under one
fit configuration and stores the raw w per plane, so configurations can be compared event by event.

    ladder_fit.py --cache cache.pkl --bundle <dir> --mode prod|noprior|seeded|shift20|shiftm20 --out o.json [--n 400]

  prod      production heldout_eval fit: blind start (tan 0), t0 prior sigma 5 ns at the bundle's t0_abs
  noprior   blind start, no t0 prior (t0 seeded from the earliest half-maximum)
  seeded    start at the reference line (w = v_bundle tan_ref, p0 = ref_mesh), prior as prod
  shift20   prior centre moved +20 ns (later); shiftm20: -20 ns
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

_EV = None
_MODE = None


def _init(cache, bundle, k, guard, mode, repo):
    global _EV, _MODE
    sys.path.insert(0, repo)
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle))
    if k:
        wm.set_depth_bins(k)
    wm.COL_GUARD = guard
    _MODE = mode


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
        tr = float(ev[f'tan_{plane}'])
        if _MODE == 'seeded':
            p0i, wi, t0g = wm.init_guess(P, plane, tr, float(ev[f'ref_mesh_{plane}']), cal.v_drift)
        else:
            p0i, wi, t0g = wm.init_guess(P, plane, 0.0, None, cal.v_drift)
        shift = {'shift20': 20.0, 'shiftm20': -20.0}.get(_MODE, 0.0)
        if _MODE == 'noprior' or t0p is None:
            prior, t0s = None, t0g
        else:
            prior, t0s = (float(t0p) + shift, 5.0), float(t0p) + shift
        try:
            r = wm.fit_plane_raw(P, plane, p0i, wi, t0s, t0_prior=prior)
        except Exception:
            r = None
        if r is None or not np.isfinite(r['chi2']):
            continue
        q = np.asarray(r['q'])
        out[plane] = dict(w=float(r['w'] * 1e3), tr=tr, p0=float(r['p0']), t0=float(r['t0']),
                          t0p=None if t0p is None else float(t0p), chi2=float(r['chi2']), dof=int(r['dof']),
                          q=q.tolist())
    return eid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--depth-bins', type=int, default=24)
    ap.add_argument('--guard', type=float, default=0.01)
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    repo = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ids = sorted(pickle.load(open(a.cache, 'rb')))[180:180 + a.n]
    with ProcessPoolExecutor(a.jobs, initializer=_init,
                             initargs=(a.cache, a.bundle, a.depth_bins, a.guard, a.mode, repo)) as pool:
        res = dict(pool.map(one, ids, chunksize=4))
    json.dump(dict(mode=a.mode, bundle=a.bundle, events=res), open(a.out, 'w'))


if __name__ == '__main__':
    main()
