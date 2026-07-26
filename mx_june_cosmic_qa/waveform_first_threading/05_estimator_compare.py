#!/usr/bin/env python3
"""Estimator-dependence of the ladder slope: production CFD vs matched-filter
t50 vs leading-edge threshold, on the same matched events."""
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

# implied v from PRODUCTION ladder slopes (slope_ns_per_mm) on matched events
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, plane in zip(axs, ('x', 'y')):
    vs = []
    for ev in events.values():
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        sl = ev['prod']['slope_x'] if plane == 'x' else ev['prod']['slope_y']
        if abs(tn) < 0.05 or not np.isfinite(sl) or sl == 0:
            continue
        v = 1.0 / (sl * tn) * 1000.0
        if 0 < v < 80:
            vs.append(v)
    vs = np.array(vs)
    ax.hist(vs, bins=60, range=(0, 80))
    ax.axvline(34, color='r', ls='--', label='v_geom=34')
    ax.axvline(np.median(vs), color='k', label=f'median {np.median(vs):.1f}')
    ax.set_title(f'{plane}: implied v from PRODUCTION slope')
    ax.legend()
    print(plane, 'production implied v median', np.median(vs).round(2),
          'q25/75', np.percentile(vs, [25, 75]).round(1), 'n', len(vs))
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'prod_ladder_v.png'), dpi=110)

# leading-edge (20% of peak) times on waveforms -> depth residual curve
rows = pickle.load(open(os.path.join(BASE, 'mf_strips.pkl'), 'rb'))
V = 34.0e-3
CORE = 0.30

def lead20(w):
    ipk = int(np.argmax(w)); a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.2 * a > w[k - 1]:
            return SNS * (k - 1 + (0.2 * a - w[k - 1]) / (w[k] - w[k - 1]))
    return np.nan

# need waveforms again: recompute lead20 for the mf-selected strips
lead_by_key = {}
for eid, ev in events.items():
    for plane in ('x', 'y'):
        P = ev[plane]
        W = P['W'].astype(np.float32)
        for i, ch in enumerate(P['ch']):
            lead_by_key[(eid, plane, float(P['pos'][i]))] = lead20(W[i])

fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))
for ax, plane in zip(axs2, ('x', 'y')):
    du_all, dt_mf, dt_le, rel_all = [], [], [], []
    for r in rows:
        if r['plane'] != plane or abs(r['tan']) < 0.08:
            continue
        t_ref = (r['pos'] - r['p0']) / (r['tan'] * V)
        okm = (t_ref > -120) & (t_ref < 1100)
        core = okm & (r['relamp'] > CORE)
        if core.sum() < 3:
            continue
        led = np.array([lead_by_key.get((r['eid'], plane, float(p)), np.nan)
                        for p in r['pos']])
        t0_mf = np.median(r['t'][core] - t_ref[core])
        okl = core & np.isfinite(led)
        if okl.sum() < 3:
            continue
        t0_le = np.median(led[okl] - t_ref[okl])
        du_all.append(t_ref[okm]); rel_all.append(r['relamp'][okm])
        dt_mf.append(r['t'][okm] - t_ref[okm] - t0_mf)
        dt_le.append(led[okm] - t_ref[okm] - t0_le)
    du = np.concatenate(du_all); rel = np.concatenate(rel_all)
    dmf = np.concatenate(dt_mf); dle = np.concatenate(dt_le)
    bins = np.arange(0, 1100, 100)
    ib = np.digitize(du, bins) - 1
    sel = rel > CORE
    med_mf = [np.median(dmf[(ib == k) & sel]) for k in range(len(bins) - 1)]
    med_le = [np.nanmedian(dle[(ib == k) & sel]) for k in range(len(bins) - 1)]
    ax.plot(bins[:-1] + 50, med_mf, 'o-', label='matched-filter t50')
    ax.plot(bins[:-1] + 50, med_le, '^-', label='leading edge 20%')
    ax.axhline(0, color='k', lw=0.5); ax.grid(alpha=0.3)
    ax.set_xlabel('u = ref drift time since mesh [ns] (v=34)')
    ax.set_ylabel('median residual [ns]  (core strips)')
    ax.set_title(f'{plane}: strip-time residual vs ref ladder, by estimator')
    ax.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(BASE, 'estimator_residual_curves.png'), dpi=110)
print('done')
