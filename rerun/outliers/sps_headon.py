#!/usr/bin/env python3
"""sps_headon.py -- the head-on far rate of det4 at H4 (run_71 RAW, flat mount, 700 V plateau) under the pivot fitter,
against the bench with the same fitter and bundle family.

SPS events: plane_ratio/sps_cache.py (both views +-8 strips around the uRWELL prediction, truth tan = 0, beam
divergence 0.5 mrad). Bundle: the det4 phys bundle (bench kernel + templates, beam D_rc 2.2e-4, det3 sigma_0
footprint, the sigma_0 profile), v_drift = 14.0 um/ns (CF4-epoch timing v at 233 V/cm, TWOGAS_HEADON section 5),
64 samples, K = 40 depth bins (the 30 mm column at 14 um/ns is ~2.1 us; K = 24 would truncate it, C5);
Dp scanned (wet CF4 not measured): --dp.
Bench events (--bench): the vladder det4 cache, head-on |tan_ref| < 0.1, held-out [180:], with the phys bundle as is.
far = |atan(w / v_ladder)| > 5 deg, v_ladder = v_drift for SPS (truth 0, no ladder) and the timing v for the bench.
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
_EV = None


def _init(cache, bundle, k, v, dp, nsamp):
    global _EV
    sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'seedfit')]
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _EV = pickle.load(open(cache, 'rb'))
    cal = CalibrationBundle.load(bundle)
    if v:
        cal.v_drift = v
    if dp:
        cal.hyper['Dp'] = dp
    wm.use_calibration(cal); wm.set_depth_bins(k); wm.COL_GUARD = 0.01
    if nsamp:
        wm.set_nsamp(nsamp)


def one(eid):
    from wft import model as wm
    import fitters as F
    ev = _EV[eid]; out = {'eid': int(eid)}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        try:
            r = F.fit_pivot(P, plane, None)
        except Exception:
            continue
        W = np.asarray(P['W'])
        out[plane] = dict(w=float(r['w'] * 1e3), p0=float(r['p0']), chi2=float(r['chi2']), wmax=float(W.max()),
                          q=float(np.clip(W, 0, None).sum()), nsat=int((W > 3500).sum()), tr=float(ev[f'tan_{plane}']),
                          ref=float(ev[f'ref_mesh_{plane}']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--v', type=float, default=None); ap.add_argument('--vlad', type=float, required=True)
    ap.add_argument('--dp', type=float, default=None); ap.add_argument('--nsamp', type=int, default=None)
    ap.add_argument('--bench', action='store_true'); ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--k', type=int, default=24, help='depth bins of 60 ns; must cover the drift column (C5): K >= T/60 + 2')
    a = ap.parse_args()
    C = pickle.load(open(a.cache, 'rb')); ids = sorted(C)
    if a.bench:
        ids = [e for e in ids[180:] if abs(C[e]['tan_x']) < 0.1 or abs(C[e]['tan_y']) < 0.1]
    ids = ids[:a.n]; del C
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.cache, a.bundle, a.k, a.v, a.dp, a.nsamp)) as pool:
        res = list(pool.map(one, ids, chunksize=4))
    summ = {}
    for v in 'xy':
        R = [r[v] for r in res if v in r and abs(r[v]['tr']) < 0.1]
        th = np.degrees(np.arctan(np.array([e['w'] for e in R]) / a.vlad) - np.arctan(np.array([e['tr'] for e in R])))
        far = np.abs(th) > 5
        summ[v] = dict(n=len(R), far_pct=float(100 * far.mean()), s68=float(np.percentile(np.abs(th[~far]), 68)),
                       far_pct_ci=float(100 * np.sqrt(far.mean() * (1 - far.mean()) / max(len(R), 1))))
    json.dump(dict(args=vars(a), summary=summ, events=res), open(a.out, 'w'))
    print(a.out, json.dumps(summ))


if __name__ == '__main__':
    main()
