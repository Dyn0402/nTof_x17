#!/usr/bin/env python3
"""
11_tauy_refit.py — does the SHIPPED kernel representation want a slower Y
copy, and how much? (T2.1, done right.)

The direct measurement says Y's neighbour copy is ~1.8x slower than X's
(tau 230 vs 410 ns), and the RC-ladder production line used kTauY = 1.78 —
but porting that constant into the shipped share_lp kernel regressed Y badly
(bench 2026-08-12): RC constants are representation-dependent (F19). So the
per-plane factor must be FIT under this kernel. This fits (tau_y_fac, kY)
ref-pinned with everything else held at the lp2 values; kY must be co-fitted
because slowing the copy moves its peak amplitude, which is what kY was
absorbing.

    ../.venv/bin/python mx_june_wft/11_tauy_refit.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/tauy_refit.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--maxiter', type=int, default=40)
    ap.add_argument('--n-train', type=int, default=180)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle

    cfg = get_config(args.run_key)
    bundle = args.bundle or os.path.join(cfg.OUT_BASE, 'wft', 'calib_bundle_lp2')
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')
    out_dir = cfg.out_dir('wft', 'kernel_arms')
    os.makedirs(out_dir, exist_ok=True)

    cal = CalibrationBundle.load(bundle)
    base_hyper = dict(cal.hyper)
    base_hyper.pop('kTauY', None)      # the inert RC-ladder constant
    v = cal.v_drift
    with open(cache, 'rb') as f:
        eids = sorted(pickle.load(f).keys())
    train = eids[:args.n_train]
    warm = {e: {} for e in train}
    print(f'[tauy] {len(train)} train events, bundle {os.path.basename(bundle)}')

    with ProcessPoolExecutor(max_workers=args.jobs, initializer=wc._init_hyper,
                             initargs=(cache, bundle)) as pool:
        neval = [0]

        def total_chi2(fac, ky):
            hyper = dict(base_hyper)
            hyper['tau_y_fac'] = float(fac)
            hyper['kY'] = float(ky)
            c = 0.0
            for eid, tot, t0s in pool.map(
                    wc._event_chi2, [(e, hyper, v, warm[e]) for e in train],
                    chunksize=6):
                c += tot
                warm[eid] = t0s
            neval[0] += 1
            return c

        t0 = time.time()
        c0 = total_chi2(1.0, base_hyper['kY'])
        print(f'[tauy] chi2 at (fac=1, kY={base_hyper["kY"]:.3f}): {c0:.5e} '
              f'({time.time() - t0:.0f} s/eval)', flush=True)

        def obj(x):
            fac, ky = x
            if not (0.4 <= fac <= 6.0 and 0.3 <= ky <= 6.0):
                return 2 * c0
            c = total_chi2(fac, ky)
            print(f'[tauy]   eval{neval[0]:3d} fac={fac:.3f} kY={ky:.3f} '
                  f'{c:.5e}', flush=True)
            return c

        x0 = np.array([1.0, base_hyper['kY']])
        res = minimize(obj, x0, method='Nelder-Mead',
                       options=dict(initial_simplex=np.array(
                           [x0, x0 + [0.5, 0], x0 + [0, 0.4]]),
                           xatol=0.02, fatol=c0 * 1e-4, maxiter=args.maxiter))

    out = dict(tau_y_fac=float(res.x[0]), kY=float(res.x[1]),
               chi2=float(res.fun), chi2_init=float(c0),
               kY_init=float(base_hyper['kY']), n_train=len(train),
               bundle=bundle, note='fit under the SHIPPED share_lp kernel; '
               'the RC-ladder kTauY=1.78 does not transfer (F19)')
    path = os.path.join(out_dir, 'tauy_refit.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[tauy] tau_y_fac {out["tau_y_fac"]:.3f}, kY {out["kY"]:.3f} '
          f'(was {out["kY_init"]:.3f}), chi2 {out["chi2"]:.5e} vs {c0:.5e}')
    print(f'[tauy] wrote {path}')


if __name__ == '__main__':
    main()
