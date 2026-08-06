#!/usr/bin/env python3
"""
val_calib.py — score calibration bundles on reference-free fits of the
calibration cache's validation split (events beyond the training 180).

For each bundle: free-fit each val event's planes exactly as production does
(init_guess -> global start -> fit), then compare the fitted angle to the M3
reference. The windows come from the ref corridor (that is what the calib
cache is), so absolute numbers are slightly optimistic; comparisons between
bundles on the same cache are fair.

    ../../.venv/bin/python mx_june_wft/bench/val_calib.py g_det7_long \
        --bundles <dir1> <dir2> ... --n 200 --jobs 4
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

_EV = None
_CAL = None


def _init(cache_path, bundle):
    global _EV, _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    with open(cache_path, 'rb') as f:
        _EV = pickle.load(f)
    _CAL = CalibrationBundle.load(bundle)
    wm.use_calibration(_CAL)


def _fit_one(eid):
    from wft import model as wm
    from wft import reco as wr
    ev = _EV[eid]
    out = {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        try:
            p0s, _w, t0s = wm.init_guess(P, plane)
            p0s, w0, t0s = wr._global_start(P, plane, p0s, t0s, wm.HYPER)
            r = wm.fit_plane_raw(P, plane, p0s, w0, t0s)
        except Exception:
            continue
        if r is None or not np.isfinite(r['chi2']):
            continue
        tan_ref = ev[f'tan_{plane}']
        q = np.asarray(r['q'], float)
        u50 = np.nan
        if q.sum() > 0:
            cgrid = np.cumsum(q) / q.sum()
            u50 = float(np.interp(0.5, cgrid, wm.UK[:len(q)]))
        out[plane] = dict(w=r['w'], p0=r['p0'],
                          dp=r['p0'] - ev[f'ref_mesh_{plane}'],
                          tan_ref=tan_ref,
                          tan_fit=r['w'] * 1e3 / _CAL.v_drift,
                          chi2dof=r['chi2'] / max(r['dof'], 1), u50=u50)
    return eid, out


def robust_sigma(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


def _v_of(bundle):
    import json
    with open(os.path.join(bundle, 'bundle.json')) as f:
        return float(json.load(f)['v_drift'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bundles', nargs='+', required=True)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--skip', type=int, default=180,
                    help='training events to skip (they saw the calibration)')
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.run_key)
    cache_path = args.cache or os.path.join(cfg.OUT_BASE, 'wft', 'calib_work',
                                            'calib_cache.pkl')
    with open(cache_path, 'rb') as f:
        ev_all = pickle.load(f)
    ids = sorted(ev_all)[args.skip:args.skip + args.n]
    print(f'{len(ids)} validation events from {cache_path}')

    for bundle in args.bundles:
        t0 = time.time()
        acc = {'x': [], 'y': []}
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(cache_path, bundle)) as pool:
            for eid, out in pool.map(_fit_one, ids, chunksize=4):
                for p, d in out.items():
                    acc[p].append(d)
        name = os.path.basename(bundle.rstrip('/'))
        print(f'\n=== {name}  ({time.time() - t0:.0f} s)')
        for p in ('x', 'y'):
            a = acc[p]
            if not a:
                print(f'  {p}: no fits')
                continue
            tr = np.array([d['tan_ref'] for d in a])
            tf = np.array([d['tan_fit'] for d in a])
            dp = np.array([d['dp'] for d in a])
            w = np.array([d['w'] for d in a])
            rel = np.abs(tr) >= 0.08
            dth = np.degrees(np.arctan(tf[rel])) - np.degrees(np.arctan(tr[rel]))
            vimp = np.where(np.abs(tr) >= 0.08, w * 1e3 / tr, np.nan)
            u50s = np.array([d.get('u50', np.nan) for d in a])
            u50m = float(np.nanmedian(u50s))
            print(f'  {p}: n={rel.sum():3d}  bias {np.median(dth):+.2f}  '
                  f'sigma {robust_sigma(dth):.2f}  s68 '
                  f'{np.percentile(np.abs(dth - np.median(dth)), 68):.2f}  '
                  f'| dp0 med {np.median(dp):+.2f} rsig {robust_sigma(dp):.2f} mm '
                  f'| implied-v med {np.nanmedian(vimp):.1f} um/ns '
                  f'| chi2/dof med {np.median([d["chi2dof"] for d in a]):.0f} '
                  f'| u50 med {u50m:.0f} ns (column ~ {2*u50m*_v_of(bundle)/1e3:.1f} mm)')


if __name__ == '__main__':
    main()
