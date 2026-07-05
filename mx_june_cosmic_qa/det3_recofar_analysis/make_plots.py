#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_plots.py — figures for the det3 reco_far characterisation report.
Reads recofar_data.npz (from extract_recofar.py); writes PNGs into this dir and
appends derived numbers to recofar_meta.json for the LaTeX report to pick up.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, 'recofar_data.npz'))
meta = json.load(open(os.path.join(HERE, 'recofar_meta.json')))
R = float(d['R'])
box = d['box']; ax0, ax1, ay0, ay1 = [float(v) for v in box]
far = d['is_far'].astype(bool); near = ~far
rng = [[ax0, ax1], [ay0, ay1]]


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.tight_layout(); fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print('wrote', name)


# ---------------------------------------------------------------- 1. WHERE
# (a) reco_far ray-position density  (b) reco_far RATE map  (c) det-position density
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
nb = 45
Hf, xe, ye = np.histogram2d(d['ray_x'][far], d['ray_y'][far], bins=nb, range=rng)
Hn, _, _ = np.histogram2d(d['ray_x'][near], d['ray_y'][near], bins=[xe, ye])
ext = [ax0, ax1, ay0, ay1]

im0 = axes[0].imshow(Hf.T, origin='lower', extent=ext, aspect='equal', cmap='inferno')
plt.colorbar(im0, ax=axes[0], label='reco_far rays')
axes[0].set_title('(a) reco_far muon crossings\n(M3 ray position)')

rate = np.full_like(Hf, np.nan)
tot = Hf + Hn
ok = tot >= 15                                   # only bins with enough stats
rate[ok] = 100 * Hf[ok] / tot[ok]
im1 = axes[1].imshow(rate.T, origin='lower', extent=ext, aspect='equal',
                     cmap='viridis', vmin=0, vmax=np.nanpercentile(rate, 98))
plt.colorbar(im1, ax=axes[1], label='reco_far rate [%]')
axes[1].set_title('(b) reco_far RATE = far / (far+near)\nper %d mm bin' % round((ax1 - ax0) / nb))

Hd, _, _ = np.histogram2d(d['det_x'][far], d['det_y'][far], bins=nb, range=rng)
im2 = axes[2].imshow(Hd.T, origin='lower', extent=ext, aspect='equal', cmap='inferno')
plt.colorbar(im2, ax=axes[2], label='reco_far reco points')
axes[2].set_title('(c) WHERE the far reco point LANDED\n(detector position)')
for a in axes:
    a.set_xlabel('X [mm]'); a.set_ylabel('Y [mm]')
save(fig, 'fig_where.png')

# rate uniformity numbers
rate_vals = rate[ok]
meta['rate_map'] = dict(median_pct=float(np.nanmedian(rate_vals)),
                        p10_pct=float(np.nanpercentile(rate_vals, 10)),
                        p90_pct=float(np.nanpercentile(rate_vals, 90)),
                        max_pct=float(np.nanmax(rate_vals)))

# ---------------------------------------------------------------- 2. |r| + planes
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(d['r'][far], bins=np.linspace(0, 200, 100), color='orange', alpha=0.9)
axes[0].axvline(R, color='r', ls='--', label=f'{R:g} mm cut')
for xb in (10, 20, 50):
    axes[0].axvline(xb, color='grey', ls=':', lw=0.8)
axes[0].set_yscale('log'); axes[0].legend()
axes[0].set_xlabel('|r| = |reco - ray| [mm]'); axes[0].set_ylabel('reco_far events')
axes[0].set_title('(a) reco_far radial residual (log)\n'
                  'median %.1f mm, mean %.1f mm' % (np.median(d['r'][far]), np.mean(d['r'][far])))

# per-plane residual scatter (which plane is wrong)
lim = 60
h = axes[1].hist2d(np.clip(d['dx'][far], -lim, lim), np.clip(d['dy'][far], -lim, lim),
                   bins=80, range=[[-lim, lim], [-lim, lim]], cmap='inferno',
                   norm=matplotlib.colors.LogNorm())
plt.colorbar(h[3], ax=axes[1], label='events')
for s in (-R, R):
    axes[1].axvline(s, color='cyan', ls='--', lw=0.7); axes[1].axhline(s, color='cyan', ls='--', lw=0.7)
axes[1].set_xlabel('dx = reco_x - ray_x [mm]'); axes[1].set_ylabel('dy = reco_y - ray_y [mm]')
axes[1].set_title('(b) per-plane displacement (clipped ±%d mm)\n'
                  'cross arms = one plane right, one wrong' % lim)
axes[1].set_aspect('equal')
save(fig, 'fig_residual_planes.png')

T = R / np.sqrt(2)
adx, ady = np.abs(d['dx'][far]), np.abs(d['dy'][far])
meta['planes'] = dict(
    x_only_bad_pct=100 * float(((adx > T) & (ady <= T)).mean()),
    y_only_bad_pct=100 * float(((ady > T) & (adx <= T)).mean()),
    both_bad_pct=100 * float(((adx > T) & (ady > T)).mean()),
    thresh_mm=float(T))
meta['rband'] = dict(
    near_5_10_pct=100 * float(((d['r'][far] > 5) & (d['r'][far] <= 10)).mean()),
    mid_10_20_pct=100 * float(((d['r'][far] > 10) & (d['r'][far] <= 20)).mean()),
    far_gt20_pct=100 * float((d['r'][far] > 20).mean()),
    far_gt50_pct=100 * float((d['r'][far] > 50).mean()),
    median_mm=float(np.median(d['r'][far])), mean_mm=float(np.mean(d['r'][far])))

# ---------------------------------------------------------------- 3. WHAT KIND
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
# multiplicity (strips fired, both sub-spark <=50)
bins = np.arange(0, 52, 2)
axes[0].hist(d['mult'][near], bins=bins, density=True, alpha=0.6, color='green', label='reco_near')
axes[0].hist(d['mult'][far], bins=bins, density=True, alpha=0.6, color='orange', label='reco_far')
axes[0].axvline(meta['spark_thresh'], color='purple', ls='--', lw=1, label='spark cut (50)')
axes[0].set_xlabel('strips fired in event'); axes[0].set_ylabel('norm. density'); axes[0].legend()
axes[0].set_title('(a) event multiplicity\nfar median %.0f vs near %.0f'
                  % (np.median(d['mult'][far]), np.median(d['mult'][near])))
# n_strips in the reconstructed cluster (X plane shown; Y similar)
nb2 = np.arange(0, 30)
axes[1].hist(d['nsx'][near], bins=nb2, density=True, alpha=0.6, color='green', label='near')
axes[1].hist(d['nsx'][far], bins=nb2, density=True, alpha=0.6, color='orange', label='far')
axes[1].set_xlabel('n_strips in X cluster'); axes[1].set_ylabel('norm. density'); axes[1].legend()
axes[1].set_title('(b) fitted cluster size (X)\nfar median %.0f vs near %.0f'
                  % (np.median(d['nsx'][far]), np.median(d['nsx'][near])))
# cluster duration (X)  -- tail of long/multi-pulse clusters
db = np.linspace(0, 2000, 80)
axes[2].hist(np.clip(d['durx'][near], 0, 2000), bins=db, density=True, alpha=0.6, color='green', label='near')
axes[2].hist(np.clip(d['durx'][far], 0, 2000), bins=db, density=True, alpha=0.6, color='orange', label='far')
axes[2].set_xlabel('X cluster duration [ns] (clipped 2000)'); axes[2].set_ylabel('norm. density'); axes[2].legend()
axes[2].set_title('(c) cluster duration (X)\nfar mean %.0f vs near %.0f ns'
                  % (np.mean(d['durx'][far]), np.mean(d['durx'][near])))
save(fig, 'fig_what.png')

meta['quality'] = dict(
    mult_far_med=float(np.median(d['mult'][far])), mult_near_med=float(np.median(d['mult'][near])),
    mult_far_p90=float(np.percentile(d['mult'][far], 90)), mult_near_p90=float(np.percentile(d['mult'][near], 90)),
    nsx_far_med=float(np.median(d['nsx'][far])), nsx_near_med=float(np.median(d['nsx'][near])),
    nsy_far_med=float(np.median(d['nsy'][far])), nsy_near_med=float(np.median(d['nsy'][near])),
    durx_far_mean=float(np.mean(d['durx'][far])), durx_near_mean=float(np.mean(d['durx'][near])),
    dury_far_mean=float(np.mean(d['dury'][far])), dury_near_mean=float(np.mean(d['dury'][near])),
    multiray_far_pct=100 * float((d['nray'][far] > 1).mean()),
    multiray_near_pct=100 * float((d['nray'][near] > 1).mean()))

# ---------------------------------------------------------------- 4. tail vs near-miss split
# does the >20mm "genuinely wrong" tail come from higher-multiplicity events?
tail = far & (d['r'] > 20); nearmiss = far & (d['r'] <= 10)
meta['split'] = dict(
    nearmiss_5_10_mult_med=float(np.median(d['mult'][nearmiss])),
    tail_gt20_mult_med=float(np.median(d['mult'][tail])),
    nearmiss_frac_of_far=100 * float(nearmiss.sum()) / far.sum(),
    tail_frac_of_far=100 * float(tail.sum()) / far.sum())

json.dump(meta, open(os.path.join(HERE, 'recofar_meta.json'), 'w'), indent=2)
print('\nupdated recofar_meta.json')
print(json.dumps({k: meta[k] for k in ('rband', 'planes', 'quality', 'rate_map', 'split')}, indent=2))
