#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_hitmaps.py — 2D spark hitmaps + a mx17-side shower check (is a spark a few
tight clusters, or one broad discharge?). Reads spark_hits.npz + events.npz;
writes fig_hitmap.png and fig_clusters.png; augments spark_meta.json.

Run: ../../.venv/bin/python make_hitmaps.py [KEY]
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root, for common/
from common.mx17_active_area import draw_outlines
KEY = next((a for a in sys.argv[1:] if not a.startswith('-')), 'g_det3_wknd')
h = np.load(os.path.join(HERE, 'spark_hits.npz'))
e = np.load(os.path.join(HERE, 'events.npz'))
meta = json.load(open(os.path.join(HERE, 'spark_meta.json')))
FX, FY = meta['feu_x'], meta['feu_y']

eid = h['eventId']; isx = h['is_x']; pos = h['pos']; chan = h['chan']
tim = h['time']; ok = (tim >= -50) & (tim <= 1600)
df = pd.DataFrame({'eid': eid, 'isx': isx, 'pos': pos, 'chan': chan})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=140, bbox_inches='tight')
    plt.close(fig); print('wrote', name)


# ---- per-spark X/Y occupancy vectors (50 bins over 0..399 mm) ----
NB = 50
edges = np.linspace(0, 399, NB + 1)
sparks = df['eid'].unique()
cent = np.full((len(sparks), 2), np.nan)          # (Xcent, Ycent)
foot = np.zeros((NB, NB))                          # summed outer(Xocc, Yocc)
xocc_tot = np.zeros(NB); yocc_tot = np.zeros(NB)   # marginal occupancy
for i, ev in enumerate(sparks):
    s = df[df['eid'] == ev]
    px = s[s['isx']]['pos'].values; py = s[~s['isx']]['pos'].values
    px = px[np.isfinite(px)]; py = py[np.isfinite(py)]
    if len(px):
        cent[i, 0] = px.mean()
        xv = (np.histogram(px, bins=edges)[0] > 0).astype(float); xocc_tot += xv
    else:
        xv = np.zeros(NB)
    if len(py):
        cent[i, 1] = py.mean()
        yv = (np.histogram(py, bins=edges)[0] > 0).astype(float); yocc_tot += yv
    else:
        yv = np.zeros(NB)
    foot += np.outer(xv, yv)
foot_pct = 100 * foot / len(sparks)                # % of sparks lighting cell (x,y)

# ============================================================ FIG 1: hitmaps
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ext = [0, 399, 0, 399]
cc = cent[np.isfinite(cent[:, 0]) & np.isfinite(cent[:, 1])]
hh = axes[0].hist2d(cc[:, 0], cc[:, 1], bins=45, range=[[0, 399], [0, 399]], cmap='inferno')
plt.colorbar(hh[3], ax=axes[0], label='# sparks')
axes[0].set_title('(a) discharge CENTRE\n(mean of fired strips per spark)')
im = axes[1].imshow(foot_pct.T, origin='lower', extent=ext, aspect='equal', cmap='inferno')
plt.colorbar(im, ax=axes[1], label='% of sparks with X&Y hit in cell')
axes[1].set_title('(b) discharge FOOTPRINT hitmap\n(where X and Y strips co-fire)')
# marginal occupancy overlay
axc = axes[2]
xb = 0.5 * (edges[:-1] + edges[1:])
axc.plot(xb, 100 * xocc_tot / len(sparks), color='steelblue', label='X plane (FEU%d)' % FX)
axc.plot(xb, 100 * yocc_tot / len(sparks), color='firebrick', label='Y plane (FEU%d)' % FY)
axc.set_xlabel('strip position [mm]'); axc.set_ylabel('% of sparks firing this region')
axc.legend(); axc.set_title('(c) per-plane occupancy\nedges over-weighted')
for a in (axes[0], axes[1]):
    a.set_xlabel('X strip pos [mm]'); a.set_ylabel('Y strip pos [mm]')
    draw_outlines(a, det_name=meta['det'])  # detector-local strip frame, no transform needed
axes[0].legend(loc='upper right', framealpha=0.9, fontsize=7)
save(fig, 'fig_hitmap.png')

# ============================================================ FIG 2: shower check on mx17 side
# For each spark/plane: number of distinct fired-strip clusters (merge gaps <= 3 strips)
# and the "fill fraction" (fired strips / strips spanned). A few tight shower tracks ->
# several NARROW clusters, LOW fill. A discharge -> few BROAD, densely filled clusters.
PITCH = 399.0 / 512


def cluster_stats(chans):
    """(#clusters merging gaps<=3, fill fraction, #strips, widest cluster mm)."""
    c = np.unique(chans[np.isfinite(chans)].astype(int))
    if len(c) < 2:
        return 1, 1.0, len(c), len(c) * PITCH
    br = np.where(np.diff(c) > 3)[0]                  # new cluster when gap > 3 strips
    groups = np.split(c, br + 1)
    nclust = len(groups)
    fill = len(c) / (c.max() - c.min() + 1)
    widest = max((g.max() - g.min() + 1) for g in groups) * PITCH
    return nclust, fill, len(c), widest


rows = []
for ev, s in df.groupby('eid'):
    for pl, mk in [('X', s['isx']), ('Y', ~s['isx'])]:
        ch = s[mk]['chan'].values
        if len(ch) == 0:
            continue
        n, fill, nst, widest = cluster_stats(ch)
        rows.append((ev, pl, n, fill, nst, widest))
cs = pd.DataFrame(rows, columns=['eid', 'plane', 'nclust', 'fill', 'nstrips', 'widest_mm'])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
# (a) number of clusters per plane
for pl, col in [('X', 'steelblue'), ('Y', 'firebrick')]:
    v = cs[cs['plane'] == pl]['nclust']
    axes[0].hist(v, bins=np.arange(0.5, 12.5), density=True, alpha=0.6, color=col, label='%s plane' % pl)
axes[0].set_xlabel('# contiguous strip clusters per spark'); axes[0].set_ylabel('norm.')
axes[0].legend(); axes[0].set_title('(a) clusters per spark\na few broad groups, not many tight tracks')
# (b) fill fraction
for pl, col in [('X', 'steelblue'), ('Y', 'firebrick')]:
    v = cs[cs['plane'] == pl]['fill']
    axes[1].hist(v, bins=np.linspace(0, 1, 40), density=True, alpha=0.6, color=col, label='%s plane' % pl)
axes[1].axvline(cs['fill'].median(), color='k', ls='--', label='median %.2f' % cs['fill'].median())
axes[1].set_xlabel('fill fraction (fired / spanned strips)'); axes[1].set_ylabel('norm.')
axes[1].legend(); axes[1].set_title('(b) fill fraction\nhigh = solid discharge, not sparse tracks')
# (c) widest cluster width (mm) distribution -- shower tracks are ~few mm wide
axes[2].hist(cs['widest_mm'], bins=np.linspace(0, 200, 50), color='purple', alpha=0.8)
axes[2].axvline(np.median(cs['widest_mm']), color='k', ls='--', label='median %.0f mm' % np.median(cs['widest_mm']))
axes[2].set_xlabel('widest single cluster [mm]'); axes[2].set_ylabel('# spark-planes'); axes[2].legend()
axes[2].set_title('(c) widest contiguous cluster\ntens of mm wide = discharge, not a ~mm track')
save(fig, 'fig_clusters.png')

meta['hitmap'] = dict(
    centroid_x_med=float(np.nanmedian(cent[:, 0])), centroid_y_med=float(np.nanmedian(cent[:, 1])),
    fill_med=float(cs['fill'].median()),
    nclust_med=float(cs['nclust'].median()),
    widest_cluster_mm_med=float(np.median(cs['widest_mm'])),
    frac_single_cluster=float((cs['nclust'] == 1).mean()))
json.dump(meta, open(os.path.join(HERE, 'spark_meta.json'), 'w'), indent=2)
print(json.dumps(meta['hitmap'], indent=2))
