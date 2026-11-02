#!/usr/bin/env python3
"""make_effmap_figure.py --root ROOT [--out papers/paper1/figures/efficiency_map.pdf] -- Paper I §5 efficiency maps on
pass 1: det3 large, the other four small (det2, det4, det6, det7).

Each map is a chamber's plateau pool (rerun/effbudget/pass1 fine_maps.py: every row of the chamber at the golden drift
field whose interior efficiency is within 3 points of the golden row's; the pools are in
rerun/effbudget/pass1/results/map_pools.json): same denominator as the loss budget (every measured M3 ray in the active
area), numerator 'detected', 8 mm disk on a 2 mm grid, blank where fewer than 15 rays or outside the active area.  No
titles (checklist 2026-11-02): panels are labelled (a) det3 ... (e) det7; the golden-row interior efficiency the text quotes
(rows.csv eff_ig; det7 X ch 256-511) and the pooling (rows, rays) are in results/efficiency_map.json for the caption.  Solid box: active area; dashed: the interior (>= 30 mm from the x sides, >= 8 mm from
the passivated y sides).
ROOT holds budget/<run>/<subrun>/<det>/rays_<det>.parquet (the paper1_eff outputs; EOS x17rerun/paper1_eff)."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
EB = os.path.join(REPO, 'rerun', 'effbudget', 'pass1')
sys.path[:0] = [EB, REPO, os.path.join(REPO, 'papers')]
import fine_maps as fm  # noqa: E402
import paper_style as ps  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402

DETS = ('det3', 'det2', 'det4', 'det6', 'det7')     # det3 first (the large panel)
EDGE_X, EDGE_Y = 30.0, 8.0                           # interior: from the x sides / from the passivated y sides
DET7_X_SPLIT = 199.3                                 # det7 X ch 0-255 | 256-511 boundary at mid-gap [mm]


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=os.path.join(REPO, 'papers', 'paper1', 'figures', 'efficiency_map.pdf'))
    ap.add_argument('--r', type=float, default=8.0)
    a = ap.parse_args()
    ps.use()
    T = pd.read_csv(os.path.join(EB, 'results', 'rows.csv'))
    pools = json.load(open(os.path.join(EB, 'results', 'map_pools.json')))

    fig = plt.figure(figsize=(ps.WIDTH['full'], 3.25), layout='constrained')
    gs = GridSpec(2, 4, figure=fig, width_ratios=[2, 1, 1, 0.07])
    axes = {'det3': fig.add_subplot(gs[:, 0]), 'det2': fig.add_subplot(gs[0, 1]), 'det4': fig.add_subplot(gs[0, 2]),
            'det6': fig.add_subplot(gs[1, 1]), 'det7': fig.add_subplot(gs[1, 2])}
    cax = fig.add_subplot(gs[:, 3])
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('white')
    summ, csv = {}, []
    for i, d in enumerate(DETS):
        ax = axes[d]; big = d == 'det3'
        rows = T[T.key.isin(pools[d]['pool'])]
        G = T[T.key == pools[d]['golden']].iloc[0]
        R = fm.load(a.root, rows)
        e, m, n = fm.emap(R, a.r)
        b = TRUE_ACTIVE_BY_DET['mx17_' + d[-1]]
        c = 0.5 * (e[:-1] + e[1:])
        X, Y = np.meshgrid(c, c, indexing='ij')
        inside = (X >= b['x'][0]) & (X <= b['x'][1]) & (Y >= b['y'][0]) & (Y <= b['y'][1])
        m = np.where(inside, m, np.nan)
        im = ax.imshow(100 * m.T, origin='lower', extent=[e[0], e[-1], e[0], e[-1]], vmin=0, vmax=100, cmap=cmap,
                       interpolation='nearest', rasterized=True)
        ax.add_patch(Rectangle((b['x'][0], b['y'][0]), b['x'][1] - b['x'][0], b['y'][1] - b['y'][0], fill=False,
                               ec=ps.INK, lw=0.7))
        ax.add_patch(Rectangle((b['x'][0] + EDGE_X, b['y'][0] + EDGE_Y), b['x'][1] - b['x'][0] - 2 * EDGE_X,
                               b['y'][1] - b['y'][0] - 2 * EDGE_Y, fill=False, ec='w', lw=0.8, ls='--'))
        if d == 'det7':      # X ch 0-255 are not quoted: veil them
            ax.add_patch(Rectangle((b['x'][0], b['y'][0]), DET7_X_SPLIT - b['x'][0], b['y'][1] - b['y'][0],
                                   fc='white', ec='none', alpha=0.55, zorder=3))
            ax.axvline(DET7_X_SPLIT, color=ps.INK, ls=':', lw=0.8, zorder=4)
        q = G.eff_ig
        ax.text(0, 1.02, rf'$\mathbf{{({chr(97 + i)})}}$ {d}', transform=ax.transAxes, ha='left', va='bottom',
                fontsize=ps.BASE_PT, color=ps.INK)
        ax.set_xlim(-10, 410); ax.set_ylim(-10, 410); ax.set_aspect('equal'); ax.grid(False)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(True)
        if big:
            ax.set_xticks([0, 100, 200, 300, 400]); ax.set_yticks([0, 100, 200, 300, 400])
            ax.set_xlabel(r'$x$ at mid-gap [mm]'); ax.set_ylabel(r'$y$ at mid-gap [mm]')
        else:
            ax.set_xticks([0, 200, 400]); ax.set_yticks([0, 200, 400])
            if d in ('det6', 'det7'):
                ax.set_xlabel(r'$x$ [mm]')
            if d in ('det2', 'det6'):
                ax.set_ylabel(r'$y$ [mm]')
        if d in ('det4', 'det7'):
            ax.set_yticklabels([])
        if d in ('det2', 'det4'):
            ax.set_xticklabels([])
        summ[d] = dict(rows=len(rows), rays=int(len(R)), golden=pools[d]['golden'], golden_eff_ig=float(q))
        k = np.isfinite(m)
        csv.append(pd.DataFrame(dict(det=d, x_mm=X[k], y_mm=Y[k], eff=m[k], n_rays_disk=n[k])))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('detection efficiency [%]'); cb.outline.set_linewidth(0.6)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300)                                            # the paper copy (pdf, map raster 300 dpi)
    ps.save(fig, os.path.join(HERE, 'figures', 'efficiency_map'))
    # the plotted maps (2 mm grid, inside the active area), gzipped: ~150k pixels
    pd.concat(csv, ignore_index=True).round(dict(x_mm=1, y_mm=1, eff=4, n_rays_disk=0)).to_csv(
        os.path.join(HERE, 'figures', 'efficiency_map.csv.gz'), index=False)
    json.dump(summ, open(os.path.join(HERE, 'results', 'efficiency_map.json'), 'w'), indent=1)
    print(summ)


if __name__ == '__main__':
    main()
