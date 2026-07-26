#!/usr/bin/env python3
"""Batch free forward-model fits with calibrated hypers + threading metrics.

For each event/plane in the test split:
  - free fit (p0, w, t0)
  - constrained fit at ref line (p0_ref, w = tan_ref * v_cal)
  - constrained fit at production ladder (det_pos, w = 1/slope_prod)
Saves a dataframe for analysis.
Usage: wf8_freefit.py [--hyper hyper_ref.json] [--n 1500] [--out freefit.pkl]
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model as fm
from scipy.ndimage import gaussian_filter1d

BASE = fm.BASE
argv = sys.argv
HYPER_F = argv[argv.index('--hyper') + 1] if '--hyper' in argv else 'hyper_ref.json'
N_MAX = int(argv[argv.index('--n') + 1]) if '--n' in argv else 2000
OUT_F = argv[argv.index('--out') + 1] if '--out' in argv else 'freefit.pkl'

hj = json.load(open(os.path.join(BASE, HYPER_F)))
HYPER = dict(c1=hj['c1'], c2=hj['c2'], tau_s=hj['tau_s'], sigma_s=hj['sigma_s'],
             sigma_p0=hj['sigma_p0'], Dp=hj['Dp'])
V_CAL = hj['v']
print('hypers', HYPER, 'v', V_CAL)

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
split = json.load(open(os.path.join(BASE, 'split_ref.json')))
test = [e for e in split['test'] if e in events][:N_MAX]
print('test events', len(test))

_orig_grid, _orig_tmpl = fm.TGRID.copy(), fm.TMPL.copy()
_sm = gaussian_filter1d(_orig_tmpl, max(HYPER['sigma_s'], 1.0) / 10.0)

def build_matrix_smear(pos, p0, w, t0, hyper):
    c1, c2, tau = hyper['c1'], hyper['c2'], hyper['tau_s']
    F = fm.strip_fractions(pos, p0, w, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    M = np.zeros((n, 32, fm.K))
    for k in range(fm.K):
        h0 = np.interp(fm.TS - (t0 + fm.UK[k]), _orig_grid, _orig_tmpl, left=0, right=0)
        h1 = np.interp(fm.TS - (t0 + fm.UK[k] + tau), _orig_grid, _sm, left=0, right=0)
        h2 = np.interp(fm.TS - (t0 + fm.UK[k] + 2 * tau), _orig_grid, _sm, left=0, right=0)
        Fk = F[:, k]
        M[:, :, k] += Fk[:, None] * h0[None, :]
        M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
        M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
        if c2 > 0:
            M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
            M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
    return M.reshape(n * 32, fm.K)

fm.build_matrix = build_matrix_smear


def fit_one(eid):
    ev = events[eid]
    out = dict(eid=eid)
    for plane in ('x', 'y'):
        P = ev[plane]
        W = P['W'].astype(np.float64)
        pos = P['pos'].astype(np.float64)
        noise = np.maximum(P['noise'].astype(np.float64), 3.0)
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0r = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        try:
            g = fm.init_guess(W, noise, pos, tn, p0r)
            free = fm.fit_event_plane(W, noise, pos, *g, hyper=HYPER)
            wref = tn * V_CAL * 1e-3
            cref = fm.fit_event_plane(W, noise, pos, *g, hyper=HYPER,
                                      fix_p0w=(p0r, wref))
            sl = ev['prod']['slope_x'] if plane == 'x' else ev['prod']['slope_y']
            pp = ev['prod']['det_x'] if plane == 'x' else ev['prod']['det_y']
            if np.isfinite(sl) and sl != 0:
                cprod = fm.fit_event_plane(W, noise, pos, *g, hyper=HYPER,
                                           fix_p0w=(pp, 1.0 / sl))
            else:
                cprod = dict(chi2=np.nan, dof=np.nan)
            out[plane] = dict(
                tan_ref=tn, p0_ref=p0r,
                p0=free['p0'], w=free['w'], t0=free['t0'],
                chi2=free['chi2'], dof=free['dof'], q=free['q'],
                chi2_ref=cref['chi2'], t0_ref=cref['t0'],
                chi2_prod=cprod['chi2'],
                w_prod=(1.0 / sl if np.isfinite(sl) and sl != 0 else np.nan),
                p0_prod=pp, amax=float(W.max()),
                nsat=int((W >= fm.SAT).sum()))
        except Exception as ex:
            out[plane] = dict(error=str(ex), tan_ref=tn)
    return out

if __name__ == '__main__':
    t0_ = time.time()
    with ProcessPoolExecutor(max_workers=14) as pool:
        res = list(pool.map(fit_one, test, chunksize=4))
    print(f'{len(res)} events fit in {time.time()-t0_:.0f}s')
    pickle.dump(res, open(os.path.join(BASE, OUT_F), 'wb'))
    print('saved', OUT_F)
