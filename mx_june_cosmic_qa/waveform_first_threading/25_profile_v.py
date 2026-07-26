#!/usr/bin/env python3
"""Profile-likelihood chi2(v): at each v, re-fit the timing hypers
(tau_s, sigma_s, sigma_p0) on the training set. Quantifies the v valley
including sharing-parameter degeneracy."""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
train = [e for e in split['train'] if e in events]

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
        return chis[j], float(t0_grid[j] + frac * 15.0), bool(j in (0, len(t0_grid) - 1))
    return chis[j], float(t0_grid[j]), bool(j in (0, len(t0_grid) - 1))

def event_chi2(args):
    eid, hyper, v, warm = args
    ev = events[eid]
    tot, t0s = 0.0, {}
    for plane in ('x', 'y'):
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        wt0 = warm.get(plane)
        for _ in range(3):
            grid = (np.arange(150.0, 900.0, 30.0) if wt0 is None
                    else np.arange(wt0 - 60.0, wt0 + 61.0, 15.0))
            chi, t0b, edge = solve_plane_t0(ev[plane], plane, p0l,
                                            tn * v * 1e-3, hyper, grid)
            if not edge or wt0 is None:
                break
            wt0 = t0b
        if np.isfinite(chi):
            tot += chi
            t0s[plane] = t0b
    return eid, tot, t0s

if __name__ == '__main__':
    pool = ProcessPoolExecutor(max_workers=12)
    out = {}
    for v in np.arange(33.0, 39.5, 0.75):
        warm = {e: {} for e in train}

        def total(hv):
            tau, sig_s, sp0 = hv
            hyper = dict(H0, tau_s=tau, sigma_s=sig_s, sigma_p0=sp0)
            c = 0.0
            for eid, tot_, t0s in pool.map(
                    event_chi2, [(e, hyper, v, warm[e]) for e in train],
                    chunksize=6):
                c += tot_
                warm[eid] = t0s
            return c

        x0 = np.array([H0['tau_s'], H0['sigma_s'], H0['sigma_p0']])
        r = minimize(lambda x: (2e18 if (x < 0).any() or x[2] < 0.03
                                else total(x)),
                     x0, method='Nelder-Mead',
                     options=dict(xatol=0.5, fatol=200.0, maxiter=45))
        out[float(v)] = dict(chi2=float(r.fun), tau_s=float(r.x[0]),
                             sigma_s=float(r.x[1]), sigma_p0=float(r.x[2]))
        print(f'v={v:.2f}: chi2={r.fun:.5e} tau={r.x[0]:.0f} sig_s={r.x[1]:.0f} '
              f'sp0={r.x[2]:.3f}', flush=True)
    json.dump(out, open(os.path.join(BASE, 'profile_v.json'), 'w'), indent=1)
    print('saved profile_v.json')
    pool.shutdown()
