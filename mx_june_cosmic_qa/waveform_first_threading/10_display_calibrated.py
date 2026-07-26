#!/usr/bin/env python3
"""Single-event displays with calibrated hypers: data / model / residual,
with fitted line vs ref line (v_cal)."""
import os, pickle, sys, json
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
V = hj['v']
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

fm.build_matrix = bm

d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
eids = list(events)
tan3 = np.array([np.hypot(events[e]['tan_x'], events[e]['tan_y']) for e in eids])
picks = [eids[int(np.argmin(np.abs(tan3 - np.percentile(tan3, q))))]
         for q in (10, 40, 65, 85, 95)]

fig, axes = plt.subplots(len(picks) * 2, 3, figsize=(15, 5.6 * len(picks)))
for row0, eid in enumerate(picks):
    ev = events[eid]
    for pi, plane in enumerate(('x', 'y')):
        row = row0 * 2 + pi
        P = ev[plane]
        W = P['W'].astype(np.float64)
        pos = P['pos'].astype(np.float64)
        noise = np.maximum(P['noise'].astype(np.float64), 3.0)
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        p0r = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        g = fm.init_guess(W, noise, pos, tn, p0r)
        free = fm.fit_event_plane(W, noise, pos, *g, hyper=HYPER)
        model = fm.model_waveforms(pos, free['p0'], free['w'], free['t0'],
                                   free['q'], hyper=HYPER)
        resid = (W - model) / noise[:, None]
        vmax = W.max()
        for col, (img, ttl, cm_) in enumerate([
                (W, 'data', 'viridis'), (model, 'model', 'viridis'),
                (resid, 'residual', 'coolwarm')]):
            ax = axes[row, col]
            if col < 2:
                pm = ax.pcolormesh(np.append(pos - 0.39, pos[-1] + 0.39),
                                   np.append(fm.TS - 30, fm.TS[-1] + 30), img.T,
                                   cmap=cm_, vmin=-0.05 * vmax, vmax=vmax)
            else:
                pm = ax.pcolormesh(np.append(pos - 0.39, pos[-1] + 0.39),
                                   np.append(fm.TS - 30, fm.TS[-1] + 30), img.T,
                                   cmap=cm_, vmin=-8, vmax=8)
            plt.colorbar(pm, ax=ax)
            uu = np.linspace(0, 850, 2)
            ax.plot(free['p0'] + free['w'] * uu, free['t0'] + uu, 'r-', lw=1.6,
                    label='fit')
            ax.plot(p0r + tn * V * 1e-3 * uu, free['t0'] + uu, 'w--', lw=1.4,
                    label=f'ref @ v={V:.1f}')
            if col == 0:
                ax.legend(fontsize=7, loc='upper right')
                ax.set_title(f'eid {eid} {plane} data  tan_ref={tn:+.3f}')
            elif col == 1:
                ax.set_title(f"model w={free['w']*1e3:+.2f} (ref {tn*V:+.2f}) "
                             f"chi2/dof={free['chi2']/free['dof']:.1f}")
            else:
                ax.set_title('residual [sigma]')
            ax.set_xlabel(f'{plane} [mm]')
        print(f'eid {eid} {plane}: tan={tn:+.3f} w={free["w"]*1e3:+.2f} '
              f'ref w={tn*V:+.2f} p0={free["p0"]:.2f} ref {p0r:.2f} '
              f'chi2/dof {free["chi2"]/free["dof"]:.2f}')
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'forward_fit_calibrated.png'), dpi=100)
print('saved')
