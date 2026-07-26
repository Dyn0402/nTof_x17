#!/usr/bin/env python3
"""Toy calibration-bias test.

Generate training-set toys whose TRUTH sharing differs from the v2 model
family (hit-level kernel: per-plane c1/c2, tau=69 ns, NO dispersion smear,
different geometric spread), with v_true = 34.0. Then run the standard
8-hyper ref-pinned calibration on them. If the calibrated v inflates toward
36.6, sharing-model mismatch biases v; if it recovers ~34, the forward v is
robust to model mismatch.
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model2 as fm2

BASE = fm2.BASE
V_TRUE = 34.0
TRUTH = dict(
    x=dict(c1=0.449, c2=0.052), y=dict(c1=0.516, c2=0.151),
    tau=69.0, sigma_p0=0.30, Dp=0.010)
T0_TRUE = 450.0
DT, K = 60.0, fm2.K
UK = fm2.UK
TOY_CACHE = os.path.join(BASE, 'wfcache_toy34.pkl')

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
train = [e for e in split['train'] if e in events]
res1 = pickle.load(open(os.path.join(BASE, 'freefit.pkl'), 'rb'))
qbank = {}
for r in res1:
    for p in ('x', 'y'):
        if p in r and 'error' not in r[p] and 'q' in r[p]:
            q = np.asarray(r[p]['q'], float)
            if 0 < q.sum() and q.max() < 2e4:
                qbank[(r['eid'], p)] = q
# fallback q: median profile scaled
qmed = np.median(np.array([q / q.sum() for q in qbank.values()]), axis=0)


def gen_matrix_truth(plane, pos, p0, w, t0):
    """Truth model matrix: hit-level kernel, no smear."""
    tr = TRUTH[plane]
    c1, c2, tau = tr['c1'], tr['c2'], TRUTH['tau']
    tmpl = fm2.TMPL[plane]
    n = len(pos)
    M = np.zeros((n, 32, K))
    for k in range(K):
        ua, ub = k * DT, (k + 1) * DT
        pa, pb = p0 + w * ua, p0 + w * ub
        pc, half = 0.5 * (pa + pb), 0.5 * abs(pb - pa)
        sig = np.sqrt(TRUTH['sigma_p0'] ** 2 + TRUTH['Dp'] ** 2 * UK[k]
                      + half ** 2 / 3.0)
        Fk = 0.5 * (erf((pos + fm2.PITCH / 2 - pc) / (np.sqrt(2) * sig))
                    - erf((pos - fm2.PITCH / 2 - pc) / (np.sqrt(2) * sig)))
        h0 = np.interp(fm2.TS - (t0 + UK[k]), fm2.TGRID, tmpl, left=0, right=0)
        h1 = np.interp(fm2.TS - (t0 + UK[k] + tau), fm2.TGRID, tmpl, left=0, right=0)
        h2 = np.interp(fm2.TS - (t0 + UK[k] + 2 * tau), fm2.TGRID, tmpl, left=0, right=0)
        M[:, :, k] += Fk[:, None] * h0[None, :]
        M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
        M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
        M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
        M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
    return M.reshape(n * 32, K)


def build_toys():
    rng = np.random.default_rng(777)
    toys = {}
    for eid in train:
        ev = events[eid]
        toy = dict(eid=eid, ref_mesh_x=ev['ref_mesh_x'], ref_mesh_y=ev['ref_mesh_y'],
                   tan_x=ev['tan_x'], tan_y=ev['tan_y'],
                   ftst_x=ev['ftst_x'], ftst_y=ev['ftst_y'])
        for plane in ('x', 'y'):
            P = ev[plane]
            pos = P['pos'].astype(np.float64)
            noise = np.maximum(P['noise'].astype(np.float64), 3.0)
            tn = toy['tan_x'] if plane == 'x' else toy['tan_y']
            p0r = toy['ref_mesh_x'] if plane == 'x' else toy['ref_mesh_y']
            q = qbank.get((eid, plane))
            if q is None:
                q = qmed * 3000.0
            M = gen_matrix_truth(plane, pos, p0r, tn * V_TRUE * 1e-3, T0_TRUE)
            W = (M @ q).reshape(len(pos), 32)
            W += rng.normal(0, 1, W.shape) * noise[:, None]
            W = np.minimum(W, fm2.SAT + 100)
            toy[plane] = dict(W=W.astype(np.float16), pos=P['pos'],
                              noise=P['noise'],
                              ch=np.zeros(len(pos), np.int16))
        toys[eid] = toy
    pickle.dump(toys, open(TOY_CACHE, 'wb'))
    print(f'{len(toys)} toys generated (v_true={V_TRUE})', flush=True)
    return toys


# ---------- calibration on toys (same protocol as wf13) ----------
toys = build_toys() if not os.path.exists(TOY_CACHE) else pickle.load(open(TOY_CACHE, 'rb'))
fm2.GAIN['x'] = np.ones(512)
fm2.GAIN['y'] = np.ones(512)


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
    ev = toys[eid]
    tot, t0s = 0.0, {}
    for plane in ('x', 'y'):
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0l = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        wline = tn * v * 1e-3
        wt0 = warm.get(plane)
        for _ in range(3):
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
    ids = list(toys)
    warm = {e: {} for e in ids}
    pool = ProcessPoolExecutor(max_workers=5)
    neval = [0]

    def total_chi2(hv):
        c1, c2, kY, tau, sig_s, sp0, Dp, v = hv
        hyper = dict(c1=c1, c2=c2, kY=kY, tau_s=tau, sigma_s=sig_s,
                     sigma_p0=sp0, Dp=Dp)
        c = 0.0
        for eid, tot, t0s in pool.map(
                event_chi2, [(e, hyper, v, warm[e]) for e in ids], chunksize=6):
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
    out['chi2'] = float(res.fun)
    out['v_true'] = V_TRUE
    out['truth'] = str(TRUTH)
    json.dump(out, open(os.path.join(BASE, 'hyper_toy34.json'), 'w'), indent=1)
    print('CALIBRATED ON TOYS (v_true=34):', np.round(res.x, 3), flush=True)
    print('saved hyper_toy34.json', flush=True)
    pool.shutdown()
