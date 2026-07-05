#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_plots.py — figures for the det3 spark characterisation report.
Reads events.npz + spark_hits.npz; writes PNGs and augments spark_meta.json."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
e = np.load(os.path.join(HERE, 'events.npz'))
h = np.load(os.path.join(HERE, 'spark_hits.npz'))
meta = json.load(open(os.path.join(HERE, 'spark_meta.json')))
box = e['box']; ax0, ax1, ay0, ay1 = [float(v) for v in box]
spark = e['spark']; hasray = e['has_ray']; rx = e['ray_x']; ry = e['ray_y']
mult = e['mult']; ts = e['ts']
DUR = meta['run_duration_s']


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=140, bbox_inches='tight')
    plt.close(fig); print('wrote', name)


# ============================================================ Q1: TIME
t = (ts.astype(float) - ts[ts > 0].min()) / 1e9
tspark = np.sort(t[spark & (ts > 0)])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
# (a) rate vs run time
nb = 40
cnt, edg = np.histogram(tspark, bins=nb, range=(0, DUR))
rate = cnt / (DUR / nb)
axes[0].step(edg[:-1] / 3600, rate, where='post', color='purple')
axes[0].axhline(len(tspark) / DUR, color='grey', ls='--', label='mean %.2f Hz' % (len(tspark) / DUR))
axes[0].set_xlabel('time in run [h]'); axes[0].set_ylabel('spark rate [Hz]')
axes[0].set_ylim(0, rate.max() * 1.3); axes[0].legend()
axes[0].set_title('(a) spark rate vs time\nflat → no HV bursts')
# (b) inter-spark interval vs exponential
dt = np.diff(tspark)
axes[1].hist(dt, bins=np.linspace(0, 20, 60), density=True, color='orange', alpha=0.8, label='data')
lam = 1 / np.mean(dt)
xx = np.linspace(0, 20, 200)
axes[1].plot(xx, lam * np.exp(-lam * xx), 'k--', label='exponential (Poisson)')
axes[1].set_yscale('log'); axes[1].set_xlabel('inter-spark interval [s]'); axes[1].set_ylabel('pdf'); axes[1].legend()
axes[1].set_title('(b) inter-spark intervals\nmatch exponential → random in time')
# (c) counts per bin vs Poisson band
axes[2].hist(cnt, bins=15, color='steelblue', alpha=0.8)
mu = cnt.mean()
axes[2].axvline(mu, color='k', ls='--', label='mean %.0f' % mu)
axes[2].axvspan(mu - np.sqrt(mu), mu + np.sqrt(mu), color='grey', alpha=0.25, label='±√mean (Poisson)')
axes[2].set_xlabel('sparks per %.0f-min bin' % (DUR / nb / 60)); axes[2].set_ylabel('# bins'); axes[2].legend()
axes[2].set_title('(c) counts/bin std=%.0f vs √mean=%.0f\n→ Poisson' % (cnt.std(), np.sqrt(mu)))
save(fig, 'fig_time.png')

# ============================================================ Q2: MUON-INDUCED?
MB = 30.0
m = hasray
inbox = m & (rx >= ax0) & (rx <= ax1) & (ry >= ay0) & (ry <= ay1)
outbox = m & ~((rx >= ax0 - MB) & (rx <= ax1 + MB) & (ry >= ay0 - MB) & (ry <= ay1 + MB))
edge = m & ~inbox & ~outbox
noray = ~hasray
cats = [('muon\ncrosses det\n(in box)', inbox), ('muon at\nedge\n(<30mm)', edge),
        ('muon\nmisses\n(>30mm)', outbox), ('no clean\nM3 track', noray)]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
rates = [100 * (spark & c).sum() / c.sum() for _, c in cats]
ns = [int((spark & c).sum()) for _, c in cats]
Ns = [int(c.sum()) for _, c in cats]
colors = ['#1a7f37', '#b26a00', '#b3261e', '#666666']
bars = axes[0].bar(range(4), rates, color=colors)
for i, (r, nsi, Ni) in enumerate(zip(rates, ns, Ns)):
    axes[0].text(i, r + 0.2, f'{r:.1f}%\n{nsi}/{Ni}', ha='center', fontsize=8)
axes[0].set_xticks(range(4)); axes[0].set_xticklabels([c[0] for c in cats], fontsize=8)
axes[0].set_ylabel('spark rate [%]'); axes[0].set_ylim(0, max(rates) * 1.25)
axes[0].set_title('(a) spark rate vs muon geometry\ncrossing muon sparks 4× more than a miss')
# (b) M3 ray positions of spark events (with a ray)
sp_ray = spark & m
axes[1].hist2d(rx[sp_ray], ry[sp_ray], bins=45, range=[[ax0, ax1], [ay0, ay1]], cmap='inferno')
axes[1].add_patch(plt.Rectangle((ax0, ay0), ax1 - ax0, ay1 - ay0, fill=False, ec='cyan', lw=1.2))
axes[1].set_xlabel('M3 ref X [mm]'); axes[1].set_ylabel('M3 ref Y [mm]'); axes[1].set_aspect('equal')
axes[1].set_title('(b) M3 crossing position of sparking muons')
# (c) spark discharge centroid (detector frame, raw strip pos)
dfh = pd.DataFrame({'eid': h['eventId'], 'isx': h['is_x'], 'pos': h['pos']})
cx = dfh[dfh['isx']].groupby('eid')['pos'].mean()
cy = dfh[~dfh['isx']].groupby('eid')['pos'].mean()
cxy = pd.concat([cx.rename('x'), cy.rename('y')], axis=1).dropna()
axes[2].hist2d(cxy['x'], cxy['y'], bins=40, range=[[0, 399], [0, 399]], cmap='inferno')
axes[2].set_xlabel('X strip pos [mm]'); axes[2].set_ylabel('Y strip pos [mm]'); axes[2].set_aspect('equal')
axes[2].set_title('(c) discharge CENTROID (detector frame)')
save(fig, 'fig_muon.png')

meta['q2'] = {c[0].replace(chr(10), ' '): dict(N=Ns[i], sparks=ns[i], rate_pct=rates[i])
              for i, c in enumerate(cats)}
meta['q2']['enhancement_in_vs_miss'] = rates[0] / rates[2]
meta['q2']['noray_frac_of_sparks_pct'] = 100 * float((spark & noray).sum()) / spark.sum()
meta['q2']['noray_frac_baseline_pct'] = 100 * float(noray.sum()) / len(noray)

# ============================================================ Q3: WHERE (strips)
isx = h['is_x']; pos = h['pos']
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
for ax, mask, nm, col in [(axes[0], isx, 'X plane (FEU%d)' % meta['feu_x'], 'steelblue'),
                          (axes[1], ~isx, 'Y plane (FEU%d)' % meta['feu_y'], 'firebrick')]:
    p = pos[mask]; p = p[np.isfinite(p)]
    ax.hist(p, bins=64, range=(0, 399), color=col, alpha=0.85)
    ax.axhline(len(p) / 64, color='k', ls='--', lw=0.8, label='uniform')
    ax.set_xlabel('strip position [mm]'); ax.set_ylabel('spark hits'); ax.legend()
    ax.set_title('(%s) %s strip occupancy in sparks' % ('a' if mask is isx else 'b', nm))
# strips per spark + spatial span
nx = dfh[dfh['isx']].groupby('eid').size(); ny = dfh[~dfh['isx']].groupby('eid').size()
axes[2].hist(nx, bins=np.arange(0, 160, 6), alpha=0.6, color='steelblue', label='X strips/spark')
axes[2].hist(ny, bins=np.arange(0, 160, 6), alpha=0.6, color='firebrick', label='Y strips/spark')
axes[2].set_xlabel('strips firing per spark'); axes[2].set_ylabel('# sparks'); axes[2].legend()
axes[2].set_title('(c) discharge size\nX med %.0f, Y med %.0f strips' % (nx.median(), ny.median()))
save(fig, 'fig_places.png')


def span(s):
    p = s.values; p = p[np.isfinite(p)]
    return (p.max() - p.min()) if len(p) > 1 else np.nan


spanx = dfh[dfh['isx']].groupby('eid')['pos'].apply(span)
spany = dfh[~dfh['isx']].groupby('eid')['pos'].apply(span)
meta['q3'] = dict(
    x_strips_med=float(nx.median()), y_strips_med=float(ny.median()),
    x_span_med_mm=float(np.nanmedian(spanx)), y_span_med_mm=float(np.nanmedian(spany)),
    frac_sat=float(h['sat'].mean()),
    x_hot_edge_share=float(np.mean(pos[isx & np.isfinite(pos)] > 360) * 100),
    y_hot_edge_share=float(np.mean(pos[~isx & np.isfinite(pos)] < 40) * 100))

# ============================================================ Q4: WALK?
tim = h['time']; ok = (tim >= -50) & (tim <= 1600)
dfw = pd.DataFrame({'eid': h['eventId'][ok], 't': tim[ok], 'pos': pos[ok], 'isx': isx[ok]})
g = dfw.groupby('eid')['t']; tstd = g.std().dropna()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
# (a) per-spark time-spread vs normal cluster
axes[0].hist(tstd, bins=np.linspace(0, 600, 60), color='purple', alpha=0.8)
axes[0].axvline(meta['tspread_norm_median_ns'], color='green', ls='--',
                label='normal cluster %.0f ns' % meta['tspread_norm_median_ns'])
axes[0].axvline(tstd.median(), color='k', ls='--', label='spark median %.0f ns' % tstd.median())
axes[0].set_xlabel('per-spark hit-time std [ns]'); axes[0].set_ylabel('# sparks'); axes[0].legend()
axes[0].set_title('(a) time spread within a spark\n1%% fire <100ns → NOT a single flash')
# (b) aggregated dpos vs dtime (centre each spark) -> walk shows as tilt
parts = []
for eid, sub in dfw.groupby('eid'):
    if len(sub) < 12:
        continue
    parts.append(np.column_stack([sub['pos'].values - np.nanmedian(sub['pos'].values),
                                  sub['t'].values - np.nanmedian(sub['t'].values)]))
D = np.vstack(parts)
Dm = np.isfinite(D[:, 0]) & np.isfinite(D[:, 1])
axes[1].hist2d(np.clip(D[Dm, 0], -200, 200), np.clip(D[Dm, 1], -600, 600),
               bins=80, cmap='inferno', norm=LogNorm())
axes[1].set_xlabel('Δ strip position [mm]'); axes[1].set_ylabel('Δ hit time [ns]')
axes[1].set_title('(b) time vs position (spark-centred)\nno tilt → no propagating front')
# (c) two example spark event displays
ex = tstd.sort_values().index.values
picks = [ex[len(ex) // 2], ex[int(len(ex) * 0.9)]]  # a typical + a spread one
for j, eid in enumerate(picks):
    sub = dfw[dfw['eid'] == eid]
    sx = sub[sub['isx']]; sy = sub[~sub['isx']]
    axes[2].scatter(sx['pos'], sx['t'], s=10, c='steelblue', label='X' if j == 0 else None, marker='o', alpha=.7)
    axes[2].scatter(sy['pos'], sy['t'], s=10, c='firebrick', label='Y' if j == 0 else None, marker='s', alpha=.7)
axes[2].set_xlabel('strip position [mm]'); axes[2].set_ylabel('hit time [ns]'); axes[2].legend()
axes[2].set_title('(c) two example sparks\nstrips light up across ~200mm, spread in time')
save(fig, 'fig_walk.png')

meta['q4'] = dict(
    tstd_med_ns=float(tstd.median()), tstd_norm_ns=float(meta['tspread_norm_median_ns']),
    frac_simultaneous_lt100=float((tstd < 100).mean()),
    frac_spread_gt300=float((tstd > 300).mean()),
    postime_corr_med=float(dfw.groupby('eid').apply(
        lambda s: np.corrcoef(s['pos'], s['t'])[0, 1]
        if (len(s) > 10 and s['pos'].std() > 1 and s['t'].std() > 1) else np.nan).abs().median()))

json.dump(meta, open(os.path.join(HERE, 'spark_meta.json'), 'w'), indent=2)
print('\nupdated spark_meta.json')
print(json.dumps({k: meta[k] for k in ('q2', 'q3', 'q4')}, indent=2, default=float))
