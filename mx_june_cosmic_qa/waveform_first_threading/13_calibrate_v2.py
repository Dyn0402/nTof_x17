#!/usr/bin/env python3
"""Calibrate model v2 hypers (ref-pinned), same protocol/split as v1."""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
train = [e for e in split['train'] if e in events]
print(f'train {len(train)}', flush=True)


def solve_plane_t0(P, plane, p0l, wline, hyper, t0_grid):
    W, noise, pos, sat = fm2.prep_plane(P, plane)
    chis = np.empty(len(t0_grid))
    for gi, t0 in enumerate(t0_grid):
        chis[gi], _ = fm2.chi2_plane(plane, W, noise, pos, sat, p0l, wline,
                                     float(t0), hyper)
    j = int(np.argmin(chis))
    if 0 < j < len(t0_grid) - 1 and np.isfinite(chis[j - 1:j + 2]).all():
        a, b, c = chis[j - 1], chis[j], chis[j + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den > 0 else 0.0
        t0b = t0_grid[j] + frac * (t0_grid[1] - t0_grid[0])
    else:
        t0b = t0_grid[j]
    return chis[j], float(t0b), bool(j in (0, len(t0_grid) - 1))


def event_chi2(args):
    eid, hyper, v, warm = args
    ev = events[eid]
    tot = 0.0
    t0s = {}
    for plane in ('x', 'y'):
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        wline = tn * v * 1e-3
        wt0 = warm.get(plane)
        for attempt in range(3):
            grid = (np.arange(150.0, 900.0, 30.0) if wt0 is None
                    else np.arange(wt0 - 60.0, wt0 + 61.0, 15.0))
            chi, t0b, edge = solve_plane_t0(ev[plane], plane, p0l, wline,
                                            hyper, grid)
            if not edge or wt0 is None:
                break
            wt0 = t0b
        if np.isfinite(chi):
            tot += chi
            t0s[plane] = t0b
    return eid, tot, t0s


NAMES = ['c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp', 'v']

if __name__ == '__main__':
    warm = {e: {} for e in train}
    pool = ProcessPoolExecutor(max_workers=14)
    neval = [0]

    def total_chi2(hv):
        c1, c2, kY, tau, sig_s, sp0, Dp, v = hv
        hyper = dict(c1=c1, c2=c2, kY=kY, tau_s=tau, sigma_s=sig_s,
                     sigma_p0=sp0, Dp=Dp)
        c = 0.0
        for eid, tot, t0s in pool.map(
                event_chi2, [(e, hyper, v, warm[e]) for e in train], chunksize=6):
            c += tot
            warm[eid] = t0s
        neval[0] += 1
        return c

    x0 = np.array([0.306, 0.057, 1.0, 47.0, 87.0, 0.098, 0.0114, 36.65])
    scale = np.array([0.05, 0.03, 0.15, 15.0, 20.0, 0.06, 0.005, 2.0])
    t_ = time.time()
    c0 = total_chi2(x0)
    print(f'initial chi2 {c0:.4e} ({time.time()-t_:.0f}s/eval)', flush=True)

    def obj(x):
        x = np.asarray(x)
        if (x[:3] < 0).any() or x[3] < 0 or x[4] < 0 or x[5] < 0.03 or \
                x[6] < 0 or not (20 < x[7] < 60):
            return 2 * c0
        c = total_chi2(x)
        print(f'  eval{neval[0]:3d}', np.round(x, 4), f'{c:.5e}', flush=True)
        return c

    simplex = [x0] + [x0 + np.eye(8)[i] * scale[i] for i in range(8)]
    res = minimize(obj, x0, method='Nelder-Mead',
                   options=dict(initial_simplex=np.array(simplex),
                                xatol=1e-3, fatol=c0 * 1e-4, maxiter=130))
    out = {k: float(vv) for k, vv in zip(NAMES, res.x)}
    out['chi2'] = float(res.fun); out['chi2_init'] = float(c0)
    out['n_train'] = len(train)
    json.dump(out, open(os.path.join(BASE, 'hyper_v2.json'), 'w'), indent=1)
    print('final', res.x, res.fun, flush=True)
    print('saved hyper_v2.json', flush=True)
    pool.shutdown()
