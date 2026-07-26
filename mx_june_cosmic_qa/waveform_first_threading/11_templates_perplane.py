#!/usr/bin/env python3
"""Per-plane impulse templates (X=FEU7, Y=FEU8) + quantify the Y slow rise."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
d = pickle.load(open(os.path.join(BASE, 'wfcache.pkl'), 'rb'))
events = d['events']
SNS = 60.0
TAN_MIN = 0.22
SAT = 3550.0
GRID = np.arange(-360, 1400, 10.0)

def t50(w):
    ipk = int(np.argmax(w)); a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.5 * a > w[k - 1]:
            return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
    return np.nan

acc = {'x': [], 'y': []}
for eid, ev in events.items():
    for plane in ('x', 'y'):
        tn = abs(ev['tan_x'] if plane == 'x' else ev['tan_y'])
        if tn < TAN_MIN:
            continue
        W = ev[plane]['W'].astype(np.float32)
        amax = W.max(axis=1)
        for i in np.argsort(amax)[::-1][:2]:
            w = W[i]; a = w.max(); ipk = int(np.argmax(w))
            if a < 500 or a > SAT or ipk < 6 or ipk > 20:
                continue
            c = t50(w)
            if np.isfinite(c):
                tt = (np.arange(32) - c) * SNS
                acc[plane].append(np.interp(GRID, tt, w / a, left=np.nan, right=np.nan))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
tm = {}
for plane in ('x', 'y'):
    A = np.array(acc[plane])
    t = np.nanmedian(A, axis=0)
    t -= np.nanmedian(t[GRID < -250])
    tm[plane] = np.nan_to_num(t)
    # metrics
    ipk = np.nanargmax(t)
    r10 = GRID[np.argmax(t >= 0.1)]; r90 = GRID[np.argmax(t >= 0.9)]
    above = t >= 0.5
    fwhm = GRID[len(t) - 1 - np.argmax(above[::-1])] - GRID[np.argmax(above)]
    print(f'{plane}: n={len(A)}  rise10-90={r90-r10:.0f}ns  peak@{GRID[ipk]:.0f}  '
          f'FWHM={fwhm:.0f}ns  undershoot_min={np.nanmin(t):.3f}')
    axs[0].plot(GRID, t, lw=2, label=f'{plane} (n={len(A)})')
axs[0].legend(); axs[0].grid(alpha=0.3); axs[0].set_xlabel('t - t50 [ns]')
axs[0].set_title('per-plane impulse templates')
axs[1].plot(GRID, tm['y'] - tm['x'], lw=2)
axs[1].grid(alpha=0.3); axs[1].set_xlabel('t - t50 [ns]')
axs[1].set_title('Y - X template difference')
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'templates_perplane.png'), dpi=110)
np.savez(os.path.join(BASE, 'templates_perplane.npz'), grid=GRID,
         tmpl_x=tm['x'], tmpl_y=tm['y'])
print('saved templates_perplane.npz')
