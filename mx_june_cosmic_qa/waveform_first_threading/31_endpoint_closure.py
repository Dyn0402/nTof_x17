#!/usr/bin/env python3
"""Endpoint closure test: does the fit+estimator chain recover a known
charge-column duration U_true?

Toys: real event geometries/noise, truth charge profile = flat to U_true
(+ bin-0 amplification spike x3, Landau-ish per-bin fluctuations), truth
model = v2 (fair estimator test). Variants: U_true in {674, 720, 793} ns;
plus U_true=793 generated with 2x-deep template undershoot but FIT with the
standard template (model-mismatch axis).
Recovered with the same estimator suite as wf30.
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2
import forward_model3 as fm3

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
hj = json.load(open(os.path.join(BASE, 'hyper_v2.json')))
H0 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']
DT = 60.0
K = fm2.K
UK = fm2.UK
N_EV = 400
T0_TRUE = 430.0

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
test = [e for e in split['test'] if e in events][:N_EV]

# deep-undershoot template variant (for generation only)
_tmpl_deep = {}
for p in ('x', 'y'):
    t = fm2.TMPL[p].copy()
    neg = t < 0
    t2 = t.copy()
    t2[neg] *= 2.0
    _tmpl_deep[p] = t2


def gen_toy(args):
    eid, U_true, deep_undershoot, seed = args
    rng = np.random.default_rng(seed)
    ev = events[eid]
    toy = {}
    for plane in ('x', 'y'):
        P = ev[plane]
        pos = P['pos'].astype(np.float64)
        noise = np.maximum(P['noise'].astype(np.float64), 3.0)
        tn = ev[f'tan_{plane}']
        p0r = ev[f'ref_mesh_{plane}']
        # truth charge profile: flat to U_true with fluctuations + bin0 spike
        q = np.zeros(K)
        for k in range(K):
            lo, hi = k * DT, (k + 1) * DT
            frac = np.clip((U_true - lo) / DT, 0, 1)
            if frac > 0:
                q[k] = frac * rng.gamma(4.0, 0.25)   # mean 1, Landau-ish
        q[0] += 3.0 * rng.gamma(4.0, 0.25)
        q *= 2500.0 / max(q.sum(), 1e-9)             # typical total charge
        if deep_undershoot:
            saved = {pp: fm2.TMPL[pp] for pp in ('x', 'y')}
            fm2.TMPL.update(_tmpl_deep)
            fm2._smear_cache.clear()
        M = fm2.build_matrix(plane, pos, p0r, tn * V * 1e-3, T0_TRUE, H0)
        if deep_undershoot:
            fm2.TMPL.update(saved)
            fm2._smear_cache.clear()
        W = (M @ q).reshape(len(pos), fm2.NSAMP)
        W += rng.normal(0, 1, W.shape) * noise[:, None]
        toy[plane] = dict(W=W.astype(np.float16), pos=P['pos'], noise=P['noise'],
                          ch=np.zeros(len(pos), np.int16))
        toy[f'tan_{plane}'] = tn
        toy[f'ref_mesh_{plane}'] = p0r
    # fit with STANDARD model, ref-pinned (as in the data endpoint measurement)
    out = []
    for plane in ('x', 'y'):
        tn = toy[f'tan_{plane}']
        if abs(tn) < 0.08:
            continue
        W, noise, pos, sat = fm2.prep_plane(toy[plane], plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 700.0, 30.0):
            c, q_ = fm2.chi2_plane(plane, W, noise, pos, sat,
                                   toy[f'ref_mesh_{plane}'], tn * V * 1e-3,
                                   float(t0), H0)
            if c < best[0]:
                best = (c, q_)
        q_ = best[1]
        if q_ is not None and 0 < q_.sum() and q_.max() < 2e4:
            out.append(q_ / q_.sum())
    return out


def U50_generic(uk, prof, pb=(1, 6)):
    plat = np.median(prof[pb[0]:pb[1]])
    if plat <= 0:
        return np.nan
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= pb[1] - 1]
    if len(below) == 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1

if __name__ == '__main__':
    fm2.GAIN['x'] = np.ones(512)
    fm2.GAIN['y'] = np.ones(512)
    uk = (np.arange(K) + 0.5) * DT
    CASES = [(674.0, False), (720.0, False), (793.0, False), (793.0, True)]
    with ProcessPoolExecutor(max_workers=14) as pool:
        for U_true, deep in CASES:
            args = [(e, U_true, deep, 9000 + i) for i, e in enumerate(test)]
            t_ = time.time()
            profs = []
            for o in pool.map(gen_toy, args, chunksize=4):
                profs.extend(o)
            P = np.array(profs)
            res = {}
            for name, est in (('median', lambda A: np.median(A, axis=0)),
                              ('mean', lambda A: A.mean(axis=0)),
                              ('trim10', lambda A: np.mean(np.sort(A, axis=0)[
                                  int(.05 * len(A)):int(.95 * len(A))], axis=0))):
                res[name] = U50_generic(uk, est(P))
            lab = f'U_true={U_true:.0f}' + (' +deep-undershoot-gen' if deep else '')
            print(f'{lab}: recovered median {res["median"]:.0f} / mean '
                  f'{res["mean"]:.0f} / trim10 {res["trim10"]:.0f} ns  '
                  f'(n={len(P)}, {time.time()-t_:.0f}s)', flush=True)
