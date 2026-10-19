#!/usr/bin/env python3
"""x_width_beam.py -- is the bench's extra X width growth (x_width.py, FINDINGS §29) there on the beam?

No beam bundle (template) is on disk, so this uses rc_drift's template-free X model: the stack's own
strip sum S is the drive and charge in source sample m arrived u = max(t_m - t_a, 0) after the first,
    sigma^2(m) = sigma_0^2 + dd2 u,     dd2 = D_T^2 v / 10   [D_T mm/sqrt(cm), v mm/ns]
with sigma_0, dd2 and t_a all free (t_a is the electronics delay of the arrival onset on the
threshold-aligned clock).  The same fit on the bench head-on caches calibrates the method against
the template model.  Beam v: run_71 composition (gasmodel 'beam', 1.51 % water, 0.078 % air).

    x_width_beam.py [--nboot 20]  -> results/x_width_beam.json
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rc_diffusion as rc                                  # noqa: E402
from rc_drift import predict                               # noqa: E402
from width_vs_time import SOURCES                          # noqa: E402

CACHES = '/home/dylan/x17/cosmic_bench/cloud_basics/xwidth/caches'
DATA = {'sps700': (SOURCES['sps700'], 243.0), 'sps450': (SOURCES['sps450'], 150.0),
        'sps275': (SOURCES['sps275'], 92.0),
        'sat_det3': (f'{CACHES}/headon_sat_det3__long_run_resist_490V_drift_1000V.pkl', None),
        'g_det4': (f'{CACHES}/headon_g_det4.pkl', None),
        'g_det7_long': (f'{CACHES}/headon_g_det7_long.pkl', None),
        'o22_long_det2': (f'{CACHES}/headon_o22_long_det2.pkl', None),
        'g_det6_long': (f'{CACHES}/headon_g_det6_long.pkl', None)}


def fit(M):
    ok = np.all(np.isfinite(M), axis=0)
    S = np.nan_to_num(M).sum(0); S[~ok] = 0

    def res(q):
        return (predict((q[0], 0.0, q[2]), S, q[1], 'x')[:, ok] - M[:, ok]).ravel()
    best = None
    for ta in (-200.0, 0.0, 200.0):
        r = least_squares(res, [0.45, 2e-4, ta], bounds=([0.02, 0.0, -600.0], [1.5, 0.01, 900.0]),
                          x_scale=[0.05, 1e-4, 50.0])
        if best is None or r.cost < best.cost:
            best = r
    return dict(sig0=float(best.x[0]), dd2=float(best.x[1]), ta=float(best.x[2]),
                rms=float(np.sqrt((best.fun ** 2).mean())))


def one(args):
    path, seed = args
    ev = pickle.load(open(path, 'rb'))
    if seed is not None:
        ks = list(ev); rng = np.random.default_rng(seed)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    n, M = rc.stack(ev, 'x')
    r = fit(M); r['n'] = n
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    from gasmodel import GasGrid
    gb = GasGrid('beam', 'air_hs')
    xw = json.load(open(f'{HERE}/results/x_width.json'))
    out = {}
    with ProcessPoolExecutor(a.workers) as pool:
        for name, (path, E) in DATA.items():
            rs = list(pool.map(one, [(path, None)] + [(path, 1000 + b) for b in range(a.nboot)]))
            c, bs = rs[0], rs[1:]
            if E is not None:
                g = gb(1.51, 0.078, E); v = g['v'] * 1e-3; mb = g['DT']
            else:
                v = xw[name]['all']['v'] * 1e-3; mb = xw[name]['magboltz']['DT_um_rtcm']
            DT = lambda d: float(np.sqrt(10 * d / v) * 1e3)
            c.update(v=v * 1e3, DT=DT(c['dd2']), DT_err=float(np.std([DT(b['dd2']) for b in bs])),
                     sig0_err=float(np.std([b['sig0'] for b in bs])), ta_err=float(np.std([b['ta'] for b in bs])),
                     DT_magboltz=mb, DT_template=xw.get(name, {}).get('all', {}).get('DT'))
            # in-quadrature extra over Magboltz, per unit depth
            c['Dx'] = float(np.sign(c['DT'] ** 2 - mb ** 2) * np.sqrt(abs(c['DT'] ** 2 - mb ** 2)))
            c['Dx_err'] = float(np.std([np.sign(DT(b['dd2']) ** 2 - mb ** 2) * np.sqrt(abs(DT(b['dd2']) ** 2 - mb ** 2))
                                        for b in bs]))
            out[name] = c
            print(f'{name:14s} n {c["n"]:5d} v {c["v"]:5.1f}  sig0 {c["sig0"]:.3f}±{c["sig0_err"]:.3f}  t_a {c["ta"]:+5.0f}±{c["ta_err"]:3.0f}  '
                  f'D_T {c["DT"]:4.0f}±{c["DT_err"]:3.0f}  (template model {c["DT_template"] or float("nan"):4.0f}; Magboltz {mb:4.0f})  '
                  f'D_x {c["Dx"]:+4.0f}±{c["Dx_err"]:3.0f}  rms {c["rms"]:.4f}', flush=True)
    json.dump(out, open(f'{HERE}/results/x_width_beam.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
