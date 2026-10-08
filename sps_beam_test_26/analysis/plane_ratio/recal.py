#!/usr/bin/env python3
"""recal.py -- refit one per-view kernel ARM on a bench calibration cache.

Same objective, training set and conventions as mx_june_wft/19_ratio_recal.py
(the tool that made r06): ref-pinned chi2 summed over the first N_TRAIN cache
events, v fixed at the seed bundle's value, Nelder-Mead inside box bounds,
seeded from the production (r06) hypers.  What changes is which hypers are
free -- the arms below.  Self-contained (explicit paths, no qa_config) so it
runs unchanged on condor.

    recal.py --cache calib_cache.pkl --bundle calib_bundle_r06 --arm pv \
             --out arm_pv.json [--jobs 8] [--maxiter 600]

Production arms keep c2 < c1 on EACH view (ratio bound < 1).  ``diag_*`` arms
lift that bound and are diagnostics only: they say where the data would go,
and a bundle is never made from one (CLAUDE.md: c2 < c1 always).
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)

# the six shared hypers 19_ratio_recal.py fits, with its steps and bounds
BASE = ('c1', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')
STEP = dict(c1=0.03, kY=0.30, tau_s=20.0, sigma_s=10.0, sigma_p0=0.05, Dp=0.003,
            c2_over_c1=0.1, c2_over_c1_x=0.1, c2_over_c1_y=0.1, cX=0.3,
            c1_asym_x=0.2, sigma_p0_x=0.05, sigma_p0_y=0.05, tau_y_fac=0.3)
LO = dict(c1=0.05, kY=0.3, tau_s=30.0, sigma_s=1.0, sigma_p0=0.10, Dp=0.001,
          c2_over_c1=0.0, c2_over_c1_x=0.0, c2_over_c1_y=0.0, cX=0.3,
          c1_asym_x=-0.95, sigma_p0_x=0.03, sigma_p0_y=0.03, tau_y_fac=0.3)
HI = dict(c1=0.60, kY=6.0, tau_s=400.0, sigma_s=400.0, sigma_p0=1.50, Dp=0.100,
          c2_over_c1=0.95, c2_over_c1_x=0.95, c2_over_c1_y=0.95, cX=6.0,
          c1_asym_x=0.95, sigma_p0_x=1.50, sigma_p0_y=1.50, tau_y_fac=4.0)

#: arm -> (free hypers, fixed extras, seed overrides, upper-bound overrides)
ARMS = {
    # control: r06's own refit, must land at r06's chi2
    'g06': (BASE, dict(c2_over_c1=0.6), {}, {}),
    # the asked-for change: one ratio per view
    'pv': (BASE + ('c2_over_c1_x', 'c2_over_c1_y'), {},
           dict(c2_over_c1_x=0.6, c2_over_c1_y=0.6), {}),
    # + X's own copy amplitude and its one-sidedness
    'pv_cx': (BASE + ('c2_over_c1_x', 'c2_over_c1_y', 'cX', 'c1_asym_x'), {},
              dict(c2_over_c1_x=0.6, c2_over_c1_y=0.6, cX=1.0, c1_asym_x=0.0), {}),
    # + a prompt lateral spread per view (instead of the shared one)
    'pv_sp': (tuple(n for n in BASE if n != 'sigma_p0') +
              ('c2_over_c1_x', 'c2_over_c1_y', 'sigma_p0_x', 'sigma_p0_y'), {},
              dict(c2_over_c1_x=0.6, c2_over_c1_y=0.6), {}),
    # everything per view
    'pv_all': (tuple(n for n in BASE if n != 'sigma_p0') +
               ('c2_over_c1_x', 'c2_over_c1_y', 'cX', 'c1_asym_x',
                'sigma_p0_x', 'sigma_p0_y', 'tau_y_fac'), {},
               dict(c2_over_c1_x=0.6, c2_over_c1_y=0.6, cX=1.0, c1_asym_x=0.0,
                    tau_y_fac=1.0), {}),
    # DIAGNOSTIC: Y ratio unbounded above -- where does the data want it?
    'diag_pv_free': (BASE + ('c2_over_c1_x', 'c2_over_c1_y'), {},
                     dict(c2_over_c1_x=0.6, c2_over_c1_y=0.6),
                     dict(c2_over_c1_x=3.0, c2_over_c1_y=3.0)),
}


_OPT = dict(t0_grid=np.arange(120.0, 961.0, 20.0), p0_profile=False)


def _init_worker(cache, bundle, k_bins, t0_lo, t0_hi, p0_profile):
    """Worker setup: load cache + bundle, and set everything the objective
    reads explicitly (not via inherited module state)."""
    from wft import calibrate as wc
    from wft import model as wm
    wc._init_hyper(cache, bundle)
    if k_bins:
        wm.set_depth_bins(k_bins)
    _OPT['t0_grid'] = np.arange(t0_lo, t0_hi + 1.0, 20.0)
    _OPT['p0_profile'] = bool(p0_profile)


def _plane_chi2(wm, plane, W, noise, pos, sat, p0, wline, hyper, grid):
    chis = np.array([wm.chi2_plane(plane, W, noise, pos, sat, p0, wline,
                                   float(t), hyper)[0] for t in grid])
    j = int(np.argmin(chis))
    best, t0b = chis[j], grid[j]
    if 0 < j < len(grid) - 1:
        for t in t0b + np.array([-10.0, -5.0, 5.0, 10.0]):
            c = wm.chi2_plane(plane, W, noise, pos, sat, p0, wline, float(t),
                              hyper)[0]
            if c < best:
                best, t0b = c, t
    return best, t0b


def _event_chi2_cold(payload):
    """Per-event ref-pinned chi2 with t0 profiled COLD and GLOBALLY on every
    evaluation (20 ns grid over the whole window + a 5 ns refinement).

    wft.calibrate._event_chi2 warm-starts t0 from the previous evaluation in a
    +-60 ns window, and this chi2 surface has near-degenerate minima ~60 ns
    apart -- so its objective depends on the optimiser's PATH.  Arms compared
    on that objective are compared on their histories.  This one is a
    deterministic function of the hypers.

    With p0_profile, p0 is also profiled (+-1 mm, 0.1 mm steps, t0 re-searched
    +-20 ns) around the reference: the reference's pointing error (~0.3 mm
    for M3 on the bench) is otherwise absorbed into sigma_p0."""
    from wft import calibrate as wc
    from wft import model as wm
    eid, hyper, v = payload
    ev = wc._EV[eid]
    grid = _OPT['t0_grid']
    tot = 0.0
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        wline = ev[f'tan_{plane}'] * v * 1e-3
        p0r = ev[f'ref_mesh_{plane}']
        W, noise, pos, sat = wm.prep_plane(P, plane)
        best, t0b = _plane_chi2(wm, plane, W, noise, pos, sat, p0r, wline,
                                hyper, grid)
        if _OPT['p0_profile'] and np.isfinite(best):
            for dp in np.arange(-1.0, 1.01, 0.1):
                if abs(dp) < 1e-9:
                    continue
                for t in (t0b - 20.0, t0b, t0b + 20.0):
                    c = wm.chi2_plane(plane, W, noise, pos, sat, p0r + dp,
                                      wline, float(t), hyper)[0]
                    if c < best:
                        best = c
        if np.isfinite(best):
            tot += float(best)
    return eid, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--arm', required=True,
                    help=f'one of {sorted(ARMS)}, or a free label with --free')
    ap.add_argument('--free', default=None,
                    help='custom arm: comma-separated free hypers (others '
                         'fixed at --seed-json / the bundle)')
    ap.add_argument('--seed-json', default=None,
                    help='start (and fix the non-free hypers) from a recal.py '
                         'output -- e.g. a bench arm carried to the beam')
    ap.add_argument('--depth-bins', type=int, default=None)
    ap.add_argument('--t0-lo', type=float, default=120.0)
    ap.add_argument('--t0-hi', type=float, default=960.0)
    ap.add_argument('--p0-profile', action='store_true',
                    help='profile p0 per event instead of pinning it to the '
                         'reference (cold objective only)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--maxiter', type=int, default=600)
    ap.add_argument('--n-train', type=int, default=180)
    ap.add_argument('--objective', default='cold', choices=('cold', 'warm'),
                    help='cold = deterministic global t0 profile (default); '
                         'warm = wft.calibrate._event_chi2, path-dependent')
    a = ap.parse_args()

    if a.arm.startswith('diag_'):
        os.environ['WFT_ALLOW_INVERTED_KERNEL'] = '1'
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle

    if a.free:
        free, fixed, seed_over, hi_over = tuple(a.free.split(',')), {}, {}, {}
    else:
        free, fixed, seed_over, hi_over = ARMS[a.arm]
    hi = dict(HI, **hi_over)
    cal = CalibrationBundle.load(a.bundle)
    base = {k: float(q) for k, q in cal.hyper.items() if k != 'kTauY'}
    v = float(cal.v_drift)
    if a.seed_json:
        sj = json.load(open(a.seed_json))
        base = {k: float(q) for k, q in sj['hyper'].items()}
        if a.free:                       # keep the arm's own c2 handling
            fixed = {k: base[k] for k in ('c2_over_c1',) if k in base}
    with open(a.cache, 'rb') as f:
        train = sorted(pickle.load(f).keys())[:a.n_train]

    start = dict(base)
    start.pop('c2_over_c1', None)
    start.update(fixed)
    start.update(seed_over)
    # per-view sigma_p0 arms start both views at the shared value
    for p in ('x', 'y'):
        if f'sigma_p0_{p}' in free:
            start.setdefault(f'sigma_p0_{p}', base['sigma_p0'])
    if 'c2_over_c1' not in fixed and 'c2_over_c1' not in free:
        # per-view arms: drop the global so only the per-view keys act
        start.pop('c2_over_c1', None)
    start['c2'] = 0.0

    def expand(x):
        h = dict(start)
        h.update({k: float(q) for k, q in zip(free, x)})
        return h

    x0 = np.array([start[k] for k in free], float)
    print(f'[recal {a.arm}] {a.bundle}\n  v={v:.2f} train={len(train)} '
          f'free={list(free)} fixed={fixed}', flush=True)

    warm = {e: {} for e in train}
    log = []
    with ProcessPoolExecutor(a.jobs, initializer=_init_worker,
                             initargs=(a.cache, a.bundle, a.depth_bins, a.t0_lo,
                                       a.t0_hi, a.p0_profile)) as pool:
        def chi(h):
            if a.objective == 'warm':
                c = 0.0
                for eid, tot, t0s in pool.map(
                        wc._event_chi2, [(e, h, v, warm[e]) for e in train],
                        chunksize=4):
                    c += tot
                    warm[eid] = t0s
                return c
            return sum(t for _e, t in pool.map(
                _event_chi2_cold, [(e, h, v) for e in train], chunksize=4))

        t0 = time.time()
        c0 = chi(expand(x0))
        print(f'[recal {a.arm}] seed chi2 {c0:.6e} ({time.time() - t0:.0f} s/eval)',
              flush=True)
        n = [0]

        def obj(x):
            h = expand(x)
            if any(not (LO[k] <= h[k] <= hi[k]) for k in free):
                return 2 * c0
            c = chi(h)
            n[0] += 1
            log.append([float(c)] + [float(q) for q in x])
            if n[0] % 10 == 0:
                print(f'[recal {a.arm}] eval {n[0]:4d} {c:.6e} ' +
                      ' '.join(f'{k}={h[k]:.4g}' for k in free), flush=True)
            return c

        simplex = np.array([x0] + [x0 + np.eye(len(free))[j] * STEP[free[j]]
                                   for j in range(len(free))])
        res = minimize(obj, x0, method='Nelder-Mead',
                       options=dict(initial_simplex=simplex, xatol=1e-3,
                                    fatol=c0 * 1e-5, maxiter=a.maxiter))
    h = expand(res.x)
    out = dict(arm=a.arm, objective=a.objective, p0_profile=a.p0_profile,
               depth_bins=a.depth_bins, hyper=h, v=v, chi2=float(res.fun), chi2_seed=float(c0),
               n_train=len(train), n_eval=n[0], converged=bool(res.success),
               message=str(res.message), free=list(free), bundle=a.bundle,
               cache=a.cache, trace=log[-50:])
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[recal {a.arm}] chi2 {c0:.6e} -> {res.fun:.6e} '
          f'({100 * (res.fun / c0 - 1):+.2f} %) after {n[0]} evals, '
          f'converged={res.success}')
    print('  ' + ' '.join(f'{k}={h[k]:.4g}' for k in free))


if __name__ == '__main__':
    main()
