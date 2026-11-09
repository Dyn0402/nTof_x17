#!/usr/bin/env python3
"""fit_kernel.py <window pickles> --view x|y --free a,b,.. --set k=v,.. --out F.json [--n N] -- global fit of the same-time readout kernel.

Parameters shared by all planes: sigma (prompt Gaussian width, mm), Dp (drift diffusion, mm/sqrt(ns)), rc (rc_D_y, mm^2/ns, Y only),
k1, k2 (same-time, same-shape nearest and next-nearest strip coupling), kd (derivative of the nearest neighbours).
Per plane: the charge profile q by NNLS; the track (p0, w, t0) fixed at the pass values.
Objective: mean of ln(chi2) over the planes, so bright planes with large model error do not decide it.  Nelder-Mead, positive
parameters through bounds by penalty.  Halves by event-id parity give a reproducibility check (--parity 0/1).
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))]
import emulate as E  # noqa: E402
from observables import kernel, padded  # noqa: E402

G = {}
START = dict(sigma=0.15, Dp=0.023, rc=3.8e-4, k1=0.05, k2=0.01, kd=0.0)
SCALE = dict(sigma=0.05, Dp=0.002, rc=4e-5, k1=0.02, k2=0.005, kd=0.02)
BOUND = dict(sigma=(0.01, 0.6), Dp=(0.005, 0.06), rc=(1e-6, 3e-3), k1=(0.0, 0.5), k2=(0.0, 0.2), kd=(-0.5, 0.5))


def chi_plane(args):
    i, par = args
    wm, p = G['wm'], G['planes'][i]
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    pos2, g2 = padded(pos, g)
    hy = dict(G['hyper'], sigma_p0=par['sigma'], sigma_p0_x=par['sigma'], sigma_p0_y=par['sigma'], Dp=par['Dp'], rc_D_y=par['rc'])
    B = wm.build_matrix(v, pos2, p['p0'], p['w'], p['t0'], hy)
    B = B.reshape(len(pos2), -1, B.shape[1]) * g2[:, None, None]
    B = kernel(B, par['k1'], par['k2'])
    if par['kd'] != 0:
        D = np.zeros_like(B)
        D[:, 1:-1] = 0.5 * (B[:, 2:] - B[:, :-2])
        C = B.copy()
        C[1:] += par['kd'] * D[:-1]
        C[:-1] += par['kd'] * D[1:]
        B = C
    B = B[4:-4].reshape(-1, B.shape[2])
    chi, q = wm._solve_nnls(B, W, nz, p['censor'].astype(bool), True)
    return float(np.log(max(chi, 1.0))) if q is not None else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--view', required=True)
    ap.add_argument('--free', required=True)
    ap.add_argument('--set', default='')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--parity', type=int, default=-1)
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--maxfev', type=int, default=260)
    ap.add_argument('--inject', default='', help='closure: sigma:Dp:rc:k1:k2 of the generated data')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes = []
    for f in a.pickles:
        planes += pickle.load(open(f, 'rb'))['planes']
    planes = [p for p in planes if p['view'] == a.view and len(p['pos']) >= 5 and (a.parity < 0 or p['event_id'] % 2 == a.parity)]
    idx = np.random.default_rng(3).permutation(len(planes))[:a.n]
    planes = [planes[i] for i in idx]
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    G.update(wm=wm, planes=planes, hyper=dict(cal.hyper))
    if a.inject:                      # closure: replace W by the model of known parameters on the plane's own track and q, plus white noise
        sg, dp, rc, k1, k2 = (float(x) for x in a.inject.split(':'))
        for p in planes:
            W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
            pos2, g2 = padded(pos, g)
            hy = dict(cal.hyper, sigma_p0=sg, sigma_p0_x=sg, sigma_p0_y=sg, Dp=dp, rc_D_y=rc)
            B = wm.build_matrix(p['view'], pos2, p['p0'], p['w'], p['t0'], hy)
            B = kernel(B.reshape(len(pos2), -1, B.shape[1]) * g2[:, None, None], k1, k2)[4:-4]
            rng = np.random.default_rng(int(p['event_id']))
            p['W'] = ((B @ p['q'].astype(float)) + rng.normal(size=W.shape) * nz[:, None]).astype(np.float32)
            p['censor'] = np.zeros(W.shape, bool)
    par0 = dict(START, rc=float(cal.hyper['rc_D_y']), Dp=float(cal.hyper['Dp']))
    fixed = dict(k1=0.0, k2=0.0, kd=0.0)
    for x in filter(None, a.set.split(',')):
        k, v = x.split('=')
        fixed[k] = float(v)
    free = a.free.split(',')
    if a.view == 'x':
        free = [f for f in free if f != 'rc']

    def params(x):
        P = dict(par0, **fixed)
        P.update({k: float(v) * SCALE[k] for k, v in zip(free, x)})
        return P

    with mp.Pool(a.jobs) as pool:
        def obj(x):
            P = params(x)
            pen = sum(((max(BOUND[k][0] - P[k], 0) + max(P[k] - BOUND[k][1], 0)) / SCALE[k]) ** 2 for k in free)
            for k in free:
                P[k] = float(np.clip(P[k], *BOUND[k]))
            r = np.array(pool.map(chi_plane, [(i, P) for i in range(len(planes))], chunksize=24))
            return float(np.nanmean(r)) + 10 * pen
        x0 = np.array([(fixed[k] if k in fixed and k in a.set else dict(par0)[k]) / SCALE[k] for k in free])
        t = time.time()
        f0 = obj(x0) if len(free) else None
        if free:
            simplex = np.vstack([x0, x0 + np.eye(len(x0))])
            r = minimize(obj, x0, method='Nelder-Mead', options=dict(initial_simplex=simplex, maxfev=a.maxfev, xatol=0.02, fatol=2e-5))
            best = params(r.x)
            fbest, nfev = float(r.fun), int(r.nfev)
        else:
            best, fbest, nfev = params(np.zeros(0)), obj(np.zeros(0)), 1
    res = dict(view=a.view, free=free, n=len(planes), parity=a.parity, start=f0, mean_ln_chi2=fbest, nfev=nfev, sec=time.time() - t, params=best, set=a.set)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in dict(view=a.view, set=a.set, f=fbest, **{k: best[k] for k in ('sigma', 'Dp', 'rc', 'k1', 'k2', 'kd')}).items()}))


if __name__ == '__main__':
    main()
