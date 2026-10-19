#!/usr/bin/env python3
"""
19_ratio_recal.py -- refit a bench run with c2 SLAVED to c1, in the shipped
kernel form.

This is the minimal, transferable fix for the c2 > c1 inversion.  The larger
programme (replace the translated-copy kernel with the RC form the beam
actually measures) does NOT transfer as it stands: the beam's absolute
constants are window-dependent (the fitted tau walks 662 -> 1040 ns as the fit
window is lengthened from 720 to 1800 ns, so the tail is heavier than one
exponential) and det3 shares visibly more than det4 (+-1/centre peak ratio
0.48 vs 0.31).  Transplanting them costs sigma_theta_Y 1.14 -> 1.51 deg
(18_ladder_bench).

What DOES survive both detectors and every form tested is the ORDERING.  The
beam's own model-free fit in the shipped translated-copy form gives
c2/c1 = 0.45 +- 0.03 at all three drift fields; near-vertical bench cosmics on
det3 give 0.63 +- 0.10.  Neither is anywhere near the > 1 the shipped bundles
carry.  So this arm keeps the kernel form the bench data can actually support,
pins the ratio, and fits SIX hypers instead of seven.

Judge on the bench (19 -> 18_ladder_bench), never on this chi2.

    ../.venv/bin/python mx_june_wft/19_ratio_recal.py sat_det3 --ratios 0.45,0.6,0.8
Output: <OUT_BASE>/wft/kernel_arms/ratio_recal.json
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

NAMES = ('c1', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')
STEP = dict(c1=0.03, kY=0.30, tau_s=20.0, sigma_s=10.0, sigma_p0=0.05,
            Dp=0.003)
# c1 keeps calibrate.C1_MIN as its floor.  It is NOT what fixes the inversion
# -- the freeold arm (seeded physical, no floor) still slid to c1 = 0.022 --
# but with c2 slaved the floor no longer has an inverted basin to fall into.
LO = dict(c1=0.05, kY=0.3, tau_s=30.0, sigma_s=1.0, sigma_p0=0.10, Dp=0.001)
HI = dict(c1=0.60, kY=6.0, tau_s=400.0, sigma_s=400.0, sigma_p0=1.50, Dp=0.100)


# This tool's whole purpose is to refit AWAY from an inverted kernel, so it is
# the one place allowed to read one. The gate added 2026-08-21
# (wft.calib.check_kernel_ordering) refuses c2 > c1 everywhere else, including
# in the worker processes, which inherit this environment.
os.environ.setdefault('WFT_ALLOW_INVERTED_KERNEL', '1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--bundle-name', default='calib_bundle_lp2_t0p',
                    help='frozen MPGD26 set: calib_bundle_lp2_t0p '
                         'for det3, calib_bundle_lp for det2/4/6/7')
    ap.add_argument('--ratios', default='0.45,0.6,0.8')
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--maxiter', type=int, default=400)
    ap.add_argument('--n-train', type=int, default=180)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle

    cfg = get_config(args.run_key)
    bundle = args.bundle or os.path.join(cfg.OUT_BASE, 'wft',
                                         args.bundle_name)
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')
    out_dir = cfg.out_dir('wft', 'kernel_arms')
    os.makedirs(out_dir, exist_ok=True)

    cal = CalibrationBundle.load(bundle)
    base = {k: float(q) for k, q in cal.hyper.items() if k != 'kTauY'}
    v = float(cal.v_drift)
    with open(cache, 'rb') as f:
        eids = sorted(pickle.load(f).keys())
    train = eids[:args.n_train]
    print(f'[ratio] {bundle} share_mode={cal.share_mode} v={v:.2f} '
          f'{len(train)} train events')

    out = {}
    for rs in args.ratios.split(','):
        r = float(rs)
        warm = {e: {} for e in train}

        def expand(x):
            h = {k: float(q) for k, q in zip(NAMES, x)}
            h['c2'] = 0.0                      # slaved; kept for the dict shape
            h['c2_over_c1'] = r
            for k in ('share_lp',):
                if k in base:
                    h[k] = base[k]
            return h

        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=wc._init_hyper,
                                 initargs=(cache, bundle)) as pool:
            neval = [0]

            def chi(h):
                c = 0.0
                for eid, tot, t0s in pool.map(
                        wc._event_chi2, [(e, h, v, warm[e]) for e in train],
                        chunksize=6):
                    c += tot
                    warm[eid] = t0s
                neval[0] += 1
                return c

            # start from production's set with c1 raised to where the pinned
            # ratio reproduces production's c2 -- so the seed is the closest
            # allowed point to the shipped answer, not an arbitrary one
            x0 = np.array([max(base['c2'] / r, LO['c1'])] +
                          [base[k] for k in NAMES[1:]], float)
            t0 = time.time()
            c0 = chi(expand(x0))
            print(f'\n[r={r:.2f}] seed ' +
                  '  '.join(f'{n}={q:.4g}' for n, q in zip(NAMES, x0)))
            print(f'[r={r:.2f}] initial chi2 {c0:.5e} '
                  f'({time.time() - t0:.0f} s/eval)', flush=True)

            def obj(x):
                h = expand(x)
                if any(not (LO[n] <= h[n] <= HI[n]) for n in NAMES):
                    return 2 * c0
                c = chi(h)
                if neval[0] % 20 == 0:
                    print(f'[r={r:.2f}]  eval{neval[0]:4d} chi2 {c:.5e}  ' +
                          '  '.join(f'{n}={h[n]:.4g}' for n in NAMES),
                          flush=True)
                return c

            simplex = np.vstack([x0] + [x0 + np.eye(len(NAMES))[i] * STEP[NAMES[i]]
                                        for i in range(len(NAMES))])
            res = minimize(obj, x0, method='Nelder-Mead',
                           options=dict(initial_simplex=simplex, xatol=1e-3,
                                        fatol=c0 * 5e-5, maxiter=args.maxiter,
                                        adaptive=True))
        h = expand(res.x)
        h['c2_implied'] = h['c1'] * r
        out[f'ratio{r:g}'] = dict(hyper=h, chi2=float(res.fun),
                                  chi2_seed=float(c0), ratio=r,
                                  n_eval=int(neval[0]), v_pinned=v,
                                  bundle=bundle, n_train=len(train))
        print(f'[r={r:.2f}] ' + '  '.join(f'{n}={h[n]:.4g}' for n in NAMES) +
              f'   -> c2={h["c2_implied"]:.4g}')
        print(f'[r={r:.2f}] chi2 {res.fun:.5e} ({neval[0]} evals)\n',
              flush=True)

    path = os.path.join(out_dir, 'ratio_recal.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[ratio] wrote {path}')


if __name__ == '__main__':
    main()
