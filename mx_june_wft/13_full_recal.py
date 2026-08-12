#!/usr/bin/env python3
"""
13_full_recal.py — T2.1 full recalibration with the per-plane copy speed free.

The 2-parameter refit (`11_tauy_refit.py`) showed the shipped share_lp kernel
wants tau_y_fac ~ 1.13 with kY ~ 1.96 (not lp2's 2.88, which was RC-ladder-
optimal) — but benching that pair with the OTHER hypers held at lp2 values was
a trade, not a win (`KERNEL_ARMS_2026-08-12.md` §3): the rest of lp2's hyper
set is self-consistent around the too-strong kY, so a 2-param patch leaves it
internally inconsistent. This does it properly: Nelder-Mead over all seven
kernel/geometry hypers PLUS tau_y_fac, ref-pinned on the calibration cache,
v pinned at the bundle value (a property of gas+field; freeing it re-opens the
v <-> sharing degeneracy that wrecked det7).

Warm-started from lp2 with (tau_y_fac, kY) at the 2-param refit optimum.
Judge the result on the bench (implied-v flatness), NEVER on this chi2 —
both 8-12 arms bought −23 % chi2 with zero geometry gain (doc §35).

    ../.venv/bin/python mx_june_wft/13_full_recal.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/full_recal.json
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

NAMES = ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp', 'tau_y_fac')
STEPS = (0.03, 0.03, 0.30, 20.0, 3.0, 0.05, 0.003, 0.15)
# c1 floor = calibrate.C1_MIN, the physical floor that stops the c1 -> 0 /
# kY -> 6 collapse the det7 free fit fell into (WFT §17.2). A first run of
# this script without it slid straight to c1 = 0.028, kY = 3.9 — the same
# valley, so the floor is needed even with v pinned.
LO = (0.05, 0.0, 0.3, 30.0, 1.0, 0.10, 0.001, 0.4)
HI = (0.60, 0.6, 6.0, 400.0, 60.0, 1.50, 0.100, 6.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--maxiter', type=int, default=300)
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
    base_hyper.pop('kTauY', None)          # inert RC-ladder constant
    extra = {k: v for k, v in base_hyper.items() if k not in NAMES}
    v = cal.v_drift
    with open(cache, 'rb') as f:
        eids = sorted(pickle.load(f).keys())
    train = eids[:args.n_train]
    warm = {e: {} for e in train}

    # warm start: lp2 values with (tau_y_fac, kY) from the 2-param refit
    x0 = np.array([base_hyper[k] if k != 'tau_y_fac' else 1.0 for k in NAMES])
    refit_path = os.path.join(out_dir, 'tauy_refit.json')
    if os.path.exists(refit_path):
        with open(refit_path) as f:
            r2 = json.load(f)
        x0[NAMES.index('tau_y_fac')] = r2['tau_y_fac']
        x0[NAMES.index('kY')] = r2['kY']
        print(f'[recal] warm start from tauy_refit: fac={r2["tau_y_fac"]:.3f} '
              f'kY={r2["kY"]:.3f}')
    print(f'[recal] {len(train)} train events, v pinned {v:.2f}, '
          f'extra={extra}, bundle {os.path.basename(bundle)}')

    with ProcessPoolExecutor(max_workers=args.jobs, initializer=wc._init_hyper,
                             initargs=(cache, bundle)) as pool:
        neval = [0]

        def total_chi2(x):
            hyper = dict(zip(NAMES, (float(q) for q in x)))
            hyper.update(extra)
            c = 0.0
            for eid, tot, t0s in pool.map(
                    wc._event_chi2, [(e, hyper, v, warm[e]) for e in train],
                    chunksize=6):
                c += tot
                warm[eid] = t0s
            neval[0] += 1
            return c

        t0 = time.time()
        c0 = total_chi2(x0)
        print(f'[recal] initial chi2 {c0:.5e} ({time.time() - t0:.0f} s/eval)',
              flush=True)

        def obj(x):
            if any(not (lo <= q <= hi) for q, lo, hi in zip(x, LO, HI)):
                return 2 * c0
            c = total_chi2(x)
            if neval[0] % 10 == 0:
                print(f'[recal]   eval{neval[0]:4d}  chi2 {c:.5e}  ' +
                      ' '.join(f'{n}={q:.3g}' for n, q in zip(NAMES, x)),
                      flush=True)
            return c

        simplex = np.vstack([x0] + [x0 + np.eye(len(NAMES))[i] * STEPS[i]
                                    for i in range(len(NAMES))])
        res = minimize(obj, x0, method='Nelder-Mead',
                       options=dict(initial_simplex=simplex, xatol=1e-3,
                                    fatol=c0 * 5e-5, maxiter=args.maxiter,
                                    adaptive=True))

    out = dict({n: float(q) for n, q in zip(NAMES, res.x)},
               chi2=float(res.fun), chi2_init=float(c0), n_eval=int(neval[0]),
               converged=bool(res.success), v_pinned=float(v),
               x0={n: float(q) for n, q in zip(NAMES, x0)},
               extra=extra, n_train=len(train), bundle=bundle,
               note='T2.1 full recalibration: all 7 hypers + tau_y_fac free, '
                    'v pinned; judge on the bench, not this chi2 (doc §35)')
    path = os.path.join(out_dir, 'full_recal.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('[recal] ' + '  '.join(f'{n}={out[n]:.4g}' for n in NAMES))
    print(f'[recal] chi2 {out["chi2"]:.5e} vs init {c0:.5e} '
          f'({100 * (out["chi2"] / c0 - 1):+.1f} %), {neval[0]} evals')
    print(f'[recal] wrote {path}')


if __name__ == '__main__':
    main()
