#!/usr/bin/env python3
"""Global hyperparameter calibration of the forward model (fast version).

Training: medium-angle events, track line FIXED to the M3 reference
(p0 = ref_mesh, w = tan_ref * v with v a hyperparameter), t0 free per
event/plane via warm-started grid + parabolic refine, charge profile NNLS.
Hypers: c1, c2, tau_s, sigma_s, sigma_p0, Dp, v.
--mode prod: fix line to production ladder (v then inert, frozen).
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model as fm

BASE = fm.BASE
MODE = 'ref' if '--mode' not in sys.argv else sys.argv[sys.argv.index('--mode') + 1]
N_TRAIN = 180
SEED = 12345

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']

rng = np.random.default_rng(SEED)
cand = []
for eid, ev in events.items():
    t3 = np.hypot(ev['tan_x'], ev['tan_y'])
    if not (0.10 < t3 < 0.40):
        continue
    ok = True
    for plane in ('x', 'y'):
        W = ev[plane]['W'].astype(np.float32)
        if (W >= fm.SAT).sum() > 6 or W.max() < 400:
            ok = False
    if ok:
        cand.append(eid)
rng.shuffle(cand)
train = cand[:N_TRAIN]
test = cand[N_TRAIN:]
json.dump(dict(train=[int(e) for e in train], test=[int(e) for e in test]),
          open(os.path.join(BASE, 'split_ref.json'), 'w'))
print(f'candidates {len(cand)}, train {len(train)}', flush=True)

_orig_grid, _orig_tmpl = fm.TGRID.copy(), fm.TMPL.copy()


def solve_plane(W, noise, pos, p0l, wline, hyper, sm, t0_grid):
    """chi2(t0) on grid, return (best chi2 refined, best t0)."""
    mask = (W < fm.SAT)
    y = (W / noise[:, None]).reshape(-1)[mask.reshape(-1)]
    Wt = np.repeat(1.0 / noise, 32)[mask.reshape(-1)]
    c1, c2, tau = hyper['c1'], hyper['c2'], hyper['tau_s']
    F = fm.strip_fractions(pos, p0l, wline, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    from scipy.optimize import nnls
    chis = np.empty(len(t0_grid))
    for gi, t0 in enumerate(t0_grid):
        M = np.zeros((n, 32, fm.K))
        for k in range(fm.K):
            h0 = np.interp(fm.TS - (t0 + fm.UK[k]), _orig_grid, _orig_tmpl, left=0, right=0)
            h1 = np.interp(fm.TS - (t0 + fm.UK[k] + tau), _orig_grid, sm, left=0, right=0)
            h2 = np.interp(fm.TS - (t0 + fm.UK[k] + 2 * tau), _orig_grid, sm, left=0, right=0)
            Fk = F[:, k]
            M[:, :, k] += Fk[:, None] * h0[None, :]
            M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
            M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
            if c2 > 0:
                M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
                M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
        A = M.reshape(n * 32, fm.K)[mask.reshape(-1)] * Wt[:, None]
        try:
            _, rn = nnls(A, y, maxiter=50 * fm.K)
            chis[gi] = rn * rn
        except Exception:
            chis[gi] = np.inf
    j = int(np.argmin(chis))
    # parabolic refine on chi2 (no extra solve)
    if 0 < j < len(t0_grid) - 1:
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
    sm = gaussian_filter1d(_orig_tmpl, max(hyper['sigma_s'], 1.0) / 10.0)
    tot = 0.0
    t0s = {}
    for plane in ('x', 'y'):
        P = ev[plane]
        W = P['W'].astype(np.float64)
        pos = P['pos'].astype(np.float64)
        noise = np.maximum(P['noise'].astype(np.float64), 3.0)
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        if MODE == 'prod':
            sl = ev['prod']['slope_x'] if plane == 'x' else ev['prod']['slope_y']
            if not np.isfinite(sl) or sl == 0:
                continue
            wline = 1.0 / sl
            p0l = ev['prod']['det_x'] if plane == 'x' else ev['prod']['det_y']
        else:
            wline = tn * v * 1e-3
            p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        wt0 = warm.get(plane)
        for attempt in range(3):
            if wt0 is None:
                grid = np.arange(150.0, 900.0, 30.0)
            else:
                grid = np.arange(wt0 - 60.0, wt0 + 61.0, 15.0)
            chi, t0b, edge = solve_plane(W, noise, pos, p0l, wline, hyper, sm, grid)
            if not edge or wt0 is None:
                break
            wt0 = t0b
        if np.isfinite(chi):
            tot += chi
            t0s[plane] = t0b
    return eid, tot, t0s


NAMES = ['c1', 'c2', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp', 'v']

if __name__ == '__main__':
    warm = {e: {} for e in train}
    pool = ProcessPoolExecutor(max_workers=14)
    neval = [0]

    def total_chi2(hv):
        c1, c2, tau, sig_s, sp0, Dp, v = hv
        hyper = dict(c1=c1, c2=c2, tau_s=tau, sigma_s=sig_s, sigma_p0=sp0, Dp=Dp)
        args = [(e, hyper, v, warm[e]) for e in train]
        c = 0.0
        for eid, tot, t0s in pool.map(event_chi2, args, chunksize=6):
            c += tot
            warm[eid] = t0s
        neval[0] += 1
        return c

    x0 = np.array([0.48, 0.08, 69.0, 30.0, 0.35, 0.012, 34.0])
    scale = np.array([0.08, 0.04, 25.0, 25.0, 0.10, 0.006, 4.0])
    t_ = time.time()
    c0 = total_chi2(x0)
    print(f'initial chi2 {c0:.4e}  ({time.time()-t_:.0f}s/eval)', flush=True)

    def obj(x):
        x = np.asarray(x)
        if (x[:2] < 0).any() or x[2] < 0 or x[3] < 0 or x[4] < 0.05 or \
                x[5] < 0 or not (20 < x[6] < 60):
            return 2 * c0
        c = total_chi2(x)
        print(f'  eval{neval[0]:3d}', np.round(x, 4), f'{c:.5e}', flush=True)
        return c

    simplex = [x0] + [x0 + np.eye(7)[i] * scale[i] for i in range(7)]
    res = minimize(obj, x0, method='Nelder-Mead',
                   options=dict(initial_simplex=np.array(simplex),
                                xatol=1e-3, fatol=c0 * 1e-4, maxiter=110))
    print('final', res.x, res.fun, flush=True)
    out = {k: float(vv) for k, vv in zip(NAMES, res.x)}
    out['chi2'] = float(res.fun); out['chi2_init'] = float(c0)
    out['mode'] = MODE; out['n_train'] = len(train)
    json.dump(out, open(os.path.join(BASE, f'hyper_{MODE}.json'), 'w'), indent=1)
    print('saved', f'hyper_{MODE}.json', flush=True)
    pool.shutdown()
