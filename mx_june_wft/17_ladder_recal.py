#!/usr/bin/env python3
"""
17_ladder_recal.py -- recalibrate a bench run under the LADDER kernel, with the
RC time constant measured on the beam rather than fitted here.

WHY.  The shipped bundles carry c2 > c1 -- the +-2 strip receiving MORE than
the +-1 strip, which no lateral transport can do.  That is not a bug in a
floor or a bound: the ref-pinned cosmic chi2 is genuinely flat in that
direction (sloppy-mode analysis, 2026-08-17), so the fit is free to walk there
and does.  The cure is to stop asking the cosmic chi2 for what it cannot see.

WHAT CHANGED.  The run_71 RAW head-on beam data measures the kernel's SHAPE
directly, through the cross-relation n_0 (*) W_d == n_d (*) W_0 -- no
deconvolution, no regularisation (sps_beam_test_26/analysis/sharing_kernel).
It picks the cascade of one-poles (wft's share_mode='lp') over the shipped
translated-copy form by a factor 2 in residual, and it pins:

  * tau, the RC constant -- field-invariant over a 2.6x range of drift field,
  * c2 = c1^2, the ladder constraint -- confirmed to 6 %.

So this arm fits FOUR hypers (c1, kY, sigma_p0, Dp) where production fits
seven, with tau_s and tau_y_fac pinned from the beam, c2 slaved by
``share_ladder`` and sigma_s held at the production value.  If four physical
parameters reach production's geometry, the other three were never measured.

ONLY Y IS PINNED.  The beam pins the Y plane and only the Y plane.  The flat
mount carries a 0.2-0.4 deg residual tilt in x, which shows up in the head-on
fit as an asymmetry q_{+1} != q_{-1} that grows as the drift field falls, and
X's fitted tau walks 390 -> 610 ns across the three plateaus instead of sitting
still.  X's constants are therefore not measured and are left free here.  The
LADDER (c2 = c1^2) is structural and applies to both planes regardless.

Parameterised per plane -- c1x/c1y and taux/tauy -- and mapped onto the model's
(c1, kY, tau_s, tau_y_fac) at the end, because that is the pairing the beam
measures and the model's shared-then-scaled convention is not.

ARMS
  ladder_pinY  c1y and tauy pinned from the beam; c1x, taux, sigma_p0, Dp free
  ladder_free  the same kernel with c1y and tauy ALSO free -- the transfer
               test.  det4 at H4 and det3 on the bench are different chambers
               of the same board design, so a bench tau_y that lands on the
               beam value is the evidence the constant transfers.

    ../.venv/bin/python mx_june_wft/17_ladder_recal.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/ladder_recal.json
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

# Beam-measured Y-plane constants, run_71 RAW, mean of the three drift
# plateaus (sharing_kernel/fit_kernel.json).  BEAM_TAU_Y is quoted on the
# model's 10 ns template grid: the beam fit runs on the 60 ns sample grid, and
# the discrete one-pole's mean a*dt/(1-a) differs by 2.4 % between the two, so
# the raw fitted 1019 ns becomes 1015 ns here.  See sharing_kernel/README.md.
# MATCHED TO THE BENCH'S WINDOW.  A single one-pole is not the true form -- the
# measured tail is heavier -- so the fitted (c, tau) depend on how much tail the
# fit window contains: over 1800 ns the beam returns c 0.656 / tau 1040 ns, over
# the +-720 ns the 32-sample bench window actually spans it returns c 0.525 /
# tau 664 ns.  The bench must be pinned with the value measured over ITS span;
# pinning the long-window pair instead cost sigma_theta_Y 1.14 -> 1.51 deg.
# At the matched window det3's own near-vertical cosmics give c 0.42 +- 0.05 and
# tau 375 +- 198 ns -- consistent with the beam, which is the transfer evidence.
BEAM_C1Y = 0.525             # +- 0.014 over 95-243 V/cm, at the bench window
BEAM_TAU_Y = 639.0           # +- 40 ns, on the model's 10 ns template grid

ARMS = {
    'ladder_pinY': dict(free=('c1x', 'taux', 'sigma_p0', 'Dp'),
                        fix=dict(c1y=BEAM_C1Y, tauy=BEAM_TAU_Y)),
    'ladder_long': dict(free=('c1x', 'taux', 'sigma_p0', 'Dp'),
                        fix=dict(c1y=0.647, tauy=995.0)),
    'ladder_free': dict(free=('c1x', 'taux', 'c1y', 'tauy', 'sigma_p0', 'Dp'),
                        fix={}),
}
STEP = dict(c1x=0.03, c1y=0.05, sigma_p0=0.05, Dp=0.003, taux=60.0,
            tauy=120.0)
LO = dict(c1x=0.02, c1y=0.02, sigma_p0=0.02, Dp=0.001, taux=30.0, tauy=30.0)
HI = dict(c1x=0.90, c1y=0.90, sigma_p0=1.50, Dp=0.100, taux=3000.0,
          tauy=3000.0)
X0 = dict(c1x=0.25, c1y=BEAM_C1Y, sigma_p0=0.25, Dp=0.009, taux=500.0,
          tauy=BEAM_TAU_Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--maxiter', type=int, default=400)
    ap.add_argument('--n-train', type=int, default=180)
    ap.add_argument('--arms', default='ladder_pinY,ladder_free')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle

    cfg = get_config(args.run_key)
    src = args.bundle or os.path.join(cfg.OUT_BASE, 'wft', 'calib_bundle_lp2_t0p')
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')
    out_dir = cfg.out_dir('wft', 'kernel_arms')
    os.makedirs(out_dir, exist_ok=True)

    # The kernel FORM lives on the bundle, not in the hyper dict: model.py
    # reads SHARE_MODE once, in use_calibration().  So the arm needs its own
    # bundle with share_mode='lp' -- passing share_lp in the hyper dict does
    # nothing (it is inert in model.py; calibrate.py only carries it through).
    cal = CalibrationBundle.load(src)
    lp_path = os.path.join(out_dir, 'ladder_provisional_bundle')
    cal.share_mode = 'lp'
    cal.save(lp_path, note='share_mode=lp, for 17_ladder_recal')
    print(f'[ladder] source bundle {os.path.basename(src)} -> {lp_path} '
          f'(share_mode=lp)')

    base = dict(cal.hyper)
    base.pop('kTauY', None)
    sigma_s = float(base.get('sigma_s', 12.07))
    v = float(cal.v_drift)
    with open(cache, 'rb') as f:
        eids = sorted(pickle.load(f).keys())
    train = eids[:args.n_train]

    res_all = {}
    for name in args.arms.split(','):
        spec = ARMS[name]
        free = list(spec['free'])
        fix = dict(spec['fix'])
        warm = {e: {} for e in train}

        def expand(xf):
            p = dict(X0)
            p.update(fix)
            p.update({n: float(q) for n, q in zip(free, xf)})
            # per-plane -> the model's shared-then-scaled convention
            return dict(sigma_s=sigma_s, share_ladder=1.0, c2=0.0,
                        sigma_p0=p['sigma_p0'], Dp=p['Dp'],
                        c1=p['c1x'], kY=p['c1y'] / p['c1x'],
                        tau_s=p['taux'], tau_y_fac=p['tauy'] / p['taux'],
                        _p=p)

        with ProcessPoolExecutor(max_workers=args.jobs,
                                 initializer=wc._init_hyper,
                                 initargs=(cache, lp_path)) as pool:
            neval = [0]

            def total_chi2(h):
                h = {k: q for k, q in h.items() if k != '_p'}
                c = 0.0
                for eid, tot, t0s in pool.map(
                        wc._event_chi2, [(e, h, v, warm[e]) for e in train],
                        chunksize=6):
                    c += tot
                    warm[eid] = t0s
                neval[0] += 1
                return c

            x0 = np.array([X0[n] for n in free], float)
            t0 = time.time()
            c0 = total_chi2(expand(x0))
            print(f'[{name}] free={free} fix={fix}')
            print(f'[{name}] initial chi2 {c0:.5e} '
                  f'({time.time() - t0:.0f} s/eval)', flush=True)

            def obj(xf):
                h = expand(xf)
                if any(not (LO[n] <= h['_p'][n] <= HI[n]) for n in free):
                    return 2 * c0
                c = total_chi2(h)
                if neval[0] % 15 == 0:
                    print(f'[{name}]   eval{neval[0]:4d} chi2 {c:.5e}  ' +
                          '  '.join(f'{n}={h["_p"][n]:.4g}' for n in free),
                          flush=True)
                return c

            simplex = np.vstack([x0] + [x0 + np.eye(len(free))[i] * STEP[free[i]]
                                        for i in range(len(free))])
            r = minimize(obj, x0, method='Nelder-Mead',
                         options=dict(initial_simplex=simplex, xatol=1e-3,
                                      fatol=c0 * 5e-5, maxiter=args.maxiter,
                                      adaptive=True))
        h = expand(r.x)
        plane = h.pop('_p')
        h['c2_effective_x'] = h['c1'] ** 2
        h['c2_effective_y'] = (h['c1'] * h['kY']) ** 2
        res_all[name] = dict(hyper={k: float(q) for k, q in h.items()},
                             per_plane={k: float(q) for k, q in plane.items()},
                             chi2=float(r.fun), chi2_init=float(c0),
                             n_eval=int(neval[0]), free=free, fix=fix,
                             v_pinned=v, bundle=lp_path, n_train=len(train))
        print(f'[{name}] ' + '  '.join(f'{n}={plane[n]:.4g}' for n in free))
        print(f'[{name}] chi2 {r.fun:.5e} vs init {c0:.5e} '
              f'({100 * (r.fun / c0 - 1):+.1f} %), {neval[0]} evals\n',
              flush=True)

    # merge, so running a subset of the arms does not drop the others
    path = os.path.join(out_dir, 'ladder_recal.json')
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(res_all)
    with open(path, 'w') as f:
        json.dump(prev, f, indent=1)
    print(f'[ladder] wrote {path}')


if __name__ == '__main__':
    main()
