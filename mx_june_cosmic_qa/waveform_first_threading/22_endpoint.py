#!/usr/bin/env python3
"""Charge-profile endpoint: template/sharing-deconvolved gap-crossing duration.

The fitted q_k (60 ns bins since t0=first arrival) is the deconvolved charge
arrival profile. For through-going muons its support is the gap-crossing
drift duration U (truncated early by attachment+noise). Ensemble-average the
normalized profiles, fit endpoint with an attenuated-uniform model:
    q(u) ~ exp(-u/tau_att) * sigmoid((U-u)/w)
Closure: v_fwd * U = charge-visible depth; compare with the 46-series
time-free extent column (24.5 mm core / floor) and the 29-30 mm gap.
"""
import os, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

B = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
     'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
res = pickle.load(open(os.path.join(B, 'freefit2.pkl'), 'rb'))
V = json.load(open(os.path.join(B, 'hyper_v2.json')))['v']
DT = 60.0
K = 18
UK = (np.arange(K) + 0.5) * DT

profs = {'x': [], 'y': []}
for r in res:
    for p in ('x', 'y'):
        if p not in r or 'error' in r[p] or 'q' not in r[p]:
            continue
        q = np.asarray(r[p]['q'], float)
        if q.sum() <= 0:
            continue
        # only well-contained fits: t0 in sane range, decent charge
        profs[p].append(q / q.sum())

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
out = {}
for i, p in enumerate(('x', 'y')):
    P = np.array(profs[p])
    med = np.median(P, axis=0)
    mean = P.mean(axis=0)
    q16, q84 = np.percentile(P, [16, 84], axis=0)
    ax = axs[i]
    ax.fill_between(UK, q16, q84, alpha=0.2)
    ax.plot(UK, med, 'o-', label='median profile')
    ax.plot(UK, mean, 's--', alpha=0.6, label='mean profile')

    # endpoint fit on the mean profile (attenuated uniform x soft edge)
    def model(u, A, tau, U, w):
        return A * np.exp(-u / tau) / (1.0 + np.exp((u - U) / w))
    try:
        popt, pcov = curve_fit(model, UK, mean,
                               p0=[mean[1] * 1.2, 500.0, 800.0, 40.0],
                               sigma=np.maximum(mean * 0.05, 1e-4),
                               maxfev=20000)
        A, tau, U, w = popt
        dU = np.sqrt(pcov[2, 2])
        uu = np.linspace(0, DT * K, 300)
        ax.plot(uu, model(uu, *popt), 'r-', lw=1.5,
                label=f'fit: U={U:.0f}±{dU:.0f}ns, tau_att={tau:.0f}ns, edge w={w:.0f}ns')
        zvis = V * U * 1e-3
        out[p] = dict(U=float(U), dU=float(dU), tau=float(tau), w=float(w),
                      z_vis_mm=float(zvis))
        print(f'{p}: U = {U:.0f} ± {dU:.0f} ns  tau_att = {tau:.0f} ns  '
              f'edge {w:.0f} ns')
        print(f'   charge-visible depth = v_fwd*U = {zvis:.1f} mm  '
              f'(v_fwd={V:.1f})')
        print(f'   v if U were full 29 mm: {29000/U:.1f}  full 30 mm: {30000/U:.1f} um/ns')
    except Exception as e:
        print(p, 'fit failed', e)
    ax.set_xlabel('u since first arrival [ns]')
    ax.set_ylabel('normalized charge / 60 ns bin')
    ax.set_title(f'{p}: deconvolved charge arrival profile (n={len(P)})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(B, 'charge_profile_endpoint.png'), dpi=110)
json.dump(out, open(os.path.join(B, 'endpoint.json'), 'w'), indent=1)
print('saved charge_profile_endpoint.png')
