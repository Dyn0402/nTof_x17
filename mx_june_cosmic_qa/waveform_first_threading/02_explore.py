#!/usr/bin/env python3
"""First-look exploration of the waveform cache: event displays with the
reference-implied time-position line, pulse shapes, ftst sanity."""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
         'long_run_resist_490V_drift_1000V/mx17_3/waveform_first/wfcache.pkl')
OUT = os.path.dirname(CACHE)
d = pickle.load(open(CACHE, 'rb'))
meta, events = d['meta'], d['events']
SNS = meta['sample_ns']
TS = np.arange(32) * SNS
V = 34.0e-3  # mm/ns tentative display scale

print('n events', len(events))
# ftst check
f7 = np.array([e['ftst_x'] for e in events.values()])
f8 = np.array([e['ftst_y'] for e in events.values()])
print('ftst equal frac:', np.mean(f7 == f8), 'range', f7.min(), f7.max())
print('ftst diff sample:', np.unique(f8 - f7, return_counts=True))

# angle distribution
tanx = np.array([abs(e['tan_x']) for e in events.values()])
tany = np.array([abs(e['tan_y']) for e in events.values()])
print('|tan_x| deciles', np.round(np.percentile(tanx, [10, 30, 50, 70, 90]), 3))
print('|tan_y| deciles', np.round(np.percentile(tany, [10, 30, 50, 70, 90]), 3))

# pick display events: sort by 3D angle, pick around percentiles with decent amp
eids = list(events)
tan3 = np.array([np.hypot(events[e]['tan_x'], events[e]['tan_y']) for e in eids])
picks = []
for q in (5, 35, 60, 80, 92, 98):
    t_target = np.percentile(tan3, q)
    i = int(np.argmin(np.abs(tan3 - t_target)))
    picks.append(eids[i])

fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.2 * len(picks)))
for row, eid in enumerate(picks):
    ev = events[eid]
    for col, plane in enumerate(('x', 'y')):
        ax = axes[row, col]
        P = ev[plane]
        W = P['W'].astype(np.float32)
        pos = P['pos']
        # pcolormesh: x=pos, y=time
        pm = ax.pcolormesh(
            np.append(pos - 0.39, pos[-1] + 0.39), np.append(TS - 30, TS[-1] + 30),
            W.T, cmap='viridis', shading='flat')
        plt.colorbar(pm, ax=ax)
        p0 = ev['ref_mesh_x'] if plane == 'x' else ev['ref_mesh_y']
        tn = ev['tan_x'] if plane == 'x' else ev['tan_y']
        # ref-implied line: t(pos)=t0 + (pos-p0)/(tan*V); anchor t0 at
        # amp-weighted mean lead time (visual only)
        amax = W.max(axis=1)
        lead = np.array([TS[np.argmax(w >= 0.5 * w.max())] if w.max() > 60 else np.nan
                         for w in W])
        okm = np.isfinite(lead) & (amax > 100)
        if okm.sum() >= 2 and abs(tn) > 1e-3:
            zs = (pos - p0) / tn          # depth from mesh per strip [mm]
            t_pred = zs / V
            t0 = np.nanmedian(lead[okm] - t_pred[okm])
            zline = np.linspace(0, 30, 2)
            ax.plot(p0 + zline * tn, t0 + zline / V, 'r-', lw=2, alpha=0.8,
                    label='ref slope (v=34)')
            ax.axvline(p0 + 0 * tn, color='w', ls=':', lw=1)
        ax.set_title(f'eid {eid} {plane}  tan={tn:+.3f}  ({np.degrees(np.arctan(abs(tn))):.1f} deg)')
        ax.set_xlabel(f'{plane} [mm]'); ax.set_ylabel('t [ns]')
        ax.legend(fontsize=7, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'explore_event_displays.png'), dpi=110)
print('saved displays')

# ---- average normalized pulse shape of bright strips, split by |tan| ----
fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
groups = [(0.0, 0.08, 'near-vertical |tan|<0.08'),
          (0.15, 0.30, '0.15<|tan|<0.30'),
          (0.35, 1.0, '|tan|>0.35')]
for gi, (lo, hi, lab) in enumerate(groups):
    acc = []
    for eid in eids[:3000]:
        ev = events[eid]
        for plane in ('x', 'y'):
            tn = abs(ev['tan_x'] if plane == 'x' else ev['tan_y'])
            if not (lo <= tn < hi):
                continue
            W = ev[plane]['W'].astype(np.float32)
            amax = W.max(axis=1)
            i = int(np.argmax(amax))
            w = W[i]
            if w.max() < 300 or np.argmax(w) < 4 or np.argmax(w) > 22:
                continue
            # align by linear-interp 50% crossing
            ipk = np.argmax(w)
            a = w.max()
            cross = None
            for k in range(1, ipk + 1):
                if w[k] >= 0.5 * a > w[k - 1]:
                    cross = k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
                    break
            if cross is None:
                continue
            tt = (np.arange(32) - cross) * SNS
            acc.append(np.interp(np.arange(-300, 1200, 30), tt, w / a, left=np.nan, right=np.nan))
    acc = np.array(acc)
    med = np.nanmedian(acc, axis=0)
    q1, q3 = np.nanpercentile(acc, [25, 75], axis=0)
    tgrid = np.arange(-300, 1200, 30)
    axs[gi].fill_between(tgrid, q1, q3, alpha=0.3)
    axs[gi].plot(tgrid, med, lw=2)
    axs[gi].set_title(f'{lab}  (n={len(acc)})')
    axs[gi].set_xlabel('t - t50 [ns]'); axs[gi].grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, 'explore_pulse_shapes.png'), dpi=110)
print('saved pulse shapes')
