#!/usr/bin/env python3
"""Per-channel gain map from the v1 free fits (ensemble flat-field), plus the
FEU7-FEU8 t0 offset measurement for the joint fit."""
import os, sys, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward_model as fm
from scipy.ndimage import gaussian_filter1d

BASE = fm.BASE
hj = json.load(open(os.path.join(BASE, 'hyper_ref.json')))
HYPER = dict(c1=hj['c1'], c2=hj['c2'], tau_s=hj['tau_s'], sigma_s=hj['sigma_s'],
             sigma_p0=hj['sigma_p0'], Dp=hj['Dp'])
_g, _t = fm.TGRID.copy(), fm.TMPL.copy()
_sm = gaussian_filter1d(_t, max(HYPER['sigma_s'], 1.0) / 10.0)

def bm(pos, p0, w, t0, hyper):
    c1, c2, tau = hyper['c1'], hyper['c2'], hyper['tau_s']
    F = fm.strip_fractions(pos, p0, w, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    M = np.zeros((n, 32, fm.K))
    for k in range(fm.K):
        h0 = np.interp(fm.TS - (t0 + fm.UK[k]), _g, _t, left=0, right=0)
        h1 = np.interp(fm.TS - (t0 + fm.UK[k] + tau), _g, _sm, left=0, right=0)
        h2 = np.interp(fm.TS - (t0 + fm.UK[k] + 2 * tau), _g, _sm, left=0, right=0)
        Fk = F[:, k]
        M[:, :, k] += Fk[:, None] * h0[None, :]
        M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
        M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
        M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
        M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
    return M.reshape(n * 32, fm.K)

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
res = pickle.load(open(os.path.join(BASE, 'freefit.pkl'), 'rb'))

ratios = {('x'): {}, ('y'): {}}
t0_diffs = []
for r in res:
    if 'x' in r and 'y' in r and 'error' not in r['x'] and 'error' not in r['y']:
        ev = events.get(r['eid'])
        if ev is not None:
            t0_diffs.append((r['x']['t0'] - r['y']['t0'],
                             ev['ftst_x'] - ev['ftst_y']))
    for p in ('x', 'y'):
        if p not in r or 'error' in r[p]:
            continue
        ev = events.get(r['eid'])
        if ev is None:
            continue
        P = ev[p]
        W = P['W'].astype(np.float64)
        pos = P['pos'].astype(np.float64)
        M = bm(pos, r[p]['p0'], r[p]['w'], r[p]['t0'], HYPER)
        model = (M @ r[p]['q']).reshape(len(pos), 32)
        for i, ch in enumerate(P['ch']):
            m = model[i]
            wv = W[i]
            sel = (m > 80) & (wv < fm.SAT)
            if sel.sum() < 4 or m[sel].max() < 250:
                continue
            g = float((wv[sel] * m[sel]).sum() / (m[sel] * m[sel]).sum())
            ratios[p].setdefault(int(ch), []).append(g)

gain = {}
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
for i, p in enumerate(('x', 'y')):
    gm = np.ones(512)
    ns = np.zeros(512, int)
    for ch, gl in ratios[p].items():
        if len(gl) >= 5:
            gm[ch] = np.median(gl)
            ns[ch] = len(gl)
    # normalize to median of measured channels
    meas = ns >= 5
    gm[meas] /= np.median(gm[meas])
    gain[p] = gm
    axs[0, i].plot(np.where(meas)[0], gm[meas], '.', ms=3)
    axs[0, i].set_title(f'{p}: per-channel gain (n_meas={meas.sum()})')
    axs[0, i].set_xlabel('channel'); axs[0, i].grid(alpha=0.3)
    sp = 100 * (np.percentile(gm[meas], 84) - np.percentile(gm[meas], 16)) / 2
    axs[1, i].hist(gm[meas], bins=60, range=(0.7, 1.3))
    axs[1, i].set_title(f'{p}: gain spread ~{sp:.1f}% (68%)')
    print(f'{p}: {meas.sum()} channels measured, spread {sp:.1f}%')
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'gainmap.png'), dpi=110)

# t0 offset by ftst difference
t0d = np.array([t for t, _ in t0_diffs])
fd = np.array([f for _, f in t0_diffs])
out_dt = {}
for u in np.unique(fd):
    m = fd == u
    out_dt[int(u)] = float(np.median(t0d[m]))
    print(f'ftst_x - ftst_y = {u:+d}: median t0x - t0y = {np.median(t0d[m]):+.1f} ns '
          f'(n={m.sum()}, mad {1.4826*np.median(np.abs(t0d[m]-np.median(t0d[m]))):.1f})')

np.savez(os.path.join(BASE, 'gainmap.npz'), gain_x=gain['x'], gain_y=gain['y'])
json.dump(out_dt, open(os.path.join(BASE, 'dt_xy.json'), 'w'))
print('saved gainmap.npz, dt_xy.json')
