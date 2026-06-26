#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_efficiency_map_sliding.py

Smooth sliding-window efficiency map, analogous to the spatial-resolution
sliding maps (`cm.plot_resolution_map_sliding`).  Consumes the per-ray hit/miss
list written by `08_efficiency_maps.py` (`<efficiency>/ray_hit_miss_list.csv`):

    columns: event_id, x, y, within, has_any
      x, y      = clean M3 single-track projection at the aligned detector plane
                  (reference frame, mm)
      within    = a reconstructed det hit within R mm of the projection (R baked
                  in by 08, default 5 mm) -> this is the efficiency numerator
      has_any   = the detector fired any strip in that event

For every point on a regular grid we collect all rays whose projection is within
KERNEL_RADIUS mm and report the fraction `within` (efficiency) and `has_any`.
Overlapping kernels make the map smooth, exactly like the resolution version.

Usage:
    python 12_efficiency_map_sliding.py <key> [--kernel=25] [--grid=120] [--min=30] [--r=5]

`--r` only labels the plot/JSON (the 5 mm match itself is encoded in `within`).
Outputs (under the detector's `efficiency/` dir):
    efficiency_map_sliding.png        (within / has_any / count panels)
    efficiency_map_sliding.json       (integrated numbers for the PDF builder)
"""
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
CFG = config_from_argv()


def _argf(flag, default):
    return next((float(a.split('=')[1]) for a in sys.argv if a.startswith(flag)), default)


KERNEL = _argf('--kernel=', 25.0)   # spatial smoothing radius [mm]
GRID = int(_argf('--grid=', 120))   # grid points per axis
MIN_RAYS = int(_argf('--min=', 30))  # min rays in kernel to colour a grid point
R = _argf('--r=', 5.0)              # label only (match radius baked into 'within')


def sliding_map(x, y, val, x_grid, y_grid, kernel, min_n):
    """Mean of `val` over rays within `kernel` mm of each grid point."""
    r2 = kernel ** 2
    eff = np.full((len(x_grid), len(y_grid)), np.nan)
    cnt = np.zeros_like(eff, dtype=int)
    for i, xg in enumerate(x_grid):
        dx2 = (x - xg) ** 2
        for j, yg in enumerate(y_grid):
            mask = (dx2 + (y - yg) ** 2) <= r2
            n = int(mask.sum())
            cnt[i, j] = n
            if n >= min_n:
                eff[i, j] = float(val[mask].mean())
    return eff, cnt


def main():
    eff_dir = CFG.out_dir('efficiency')
    csv = os.path.join(eff_dir, 'ray_hit_miss_list.csv')
    if not os.path.isfile(csv):
        print(f'ERROR: {csv} not found — run 08_efficiency_maps.py first.')
        sys.exit(1)

    d = pd.read_csv(csv)
    d = d[np.isfinite(d['x']) & np.isfinite(d['y'])].copy()
    for c in ('within', 'has_any'):
        d[c] = d[c].astype(str).str.lower().isin(('true', '1'))
    x = d['x'].to_numpy(float)
    y = d['y'].to_numpy(float)
    within = d['within'].to_numpy(float)
    has_any = d['has_any'].to_numpy(float)
    print(f'{CFG.DET_NAME} {CFG.RUN}/{CFG.SUB_RUN}: {len(d)} clean M3 rays, '
          f'kernel r={KERNEL:.0f} mm, {GRID}x{GRID} grid', flush=True)

    # Active-area box from the reco'd (within) ray footprint — same idea as 08.
    hit = d[d['within']]
    if len(hit) < 20:
        hit = d
    ax0, ax1 = np.percentile(hit['x'], [0.5, 99.5])
    ay0, ay1 = np.percentile(hit['y'], [0.5, 99.5])
    inact = (x >= ax0) & (x <= ax1) & (y >= ay0) & (y <= ay1)
    integ_within = float(within[inact].mean()) if inact.any() else float('nan')
    integ_hasany = float(has_any[inact].mean()) if inact.any() else float('nan')

    pad = KERNEL
    x_grid = np.linspace(ax0 - pad, ax1 + pad, GRID)
    y_grid = np.linspace(ay0 - pad, ay1 + pad, GRID)

    eff_w, cnt = sliding_map(x, y, within, x_grid, y_grid, KERNEL, MIN_RAYS)
    eff_a, _ = sliding_map(x, y, has_any, x_grid, y_grid, KERNEL, MIN_RAYS)
    n_fit = int(np.sum(~np.isnan(eff_w)))
    print(f'  fitted {n_fit}/{GRID**2} grid points; '
          f'integrated within{R:g}mm={integ_within*100:.1f}%  has_any={integ_hasany*100:.1f}%')

    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    box = dict(xy=(ax0, ay0), width=ax1 - ax0, height=ay1 - ay0)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('lightgrey')
    cmap_c = plt.get_cmap('plasma').copy(); cmap_c.set_bad('lightgrey')

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    panels = [
        (axes[0], eff_w, f'efficiency (reco within {R:g} mm)', cmap, (0, 1)),
        (axes[1], eff_a, 'has_any (fired any strip)', cmap, (0, 1)),
    ]
    for ax, data, label, cm_, (vmin, vmax) in panels:
        im = ax.imshow(data.T, origin='lower', extent=extent, aspect='equal',
                       cmap=cm_, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.3))
        ax.set_xlabel('reference X [mm]'); ax.set_ylabel('reference Y [mm]')
        ax.set_title(f'{CFG.DET_NAME}  {label}\nsliding kernel r={KERNEL:.0f} mm')

    cnt_m = np.where(cnt >= MIN_RAYS, cnt, np.nan)
    im3 = axes[2].imshow(cnt_m.T, origin='lower', extent=extent, aspect='equal', cmap=cmap_c)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='rays in kernel')
    axes[2].add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.3))
    axes[2].set_xlabel('reference X [mm]'); ax = axes[2]; ax.set_ylabel('reference Y [mm]')
    axes[2].set_title(f'rays per kernel\n(grey < {MIN_RAYS})')

    fig.suptitle(f'{CFG.DET_NAME} sliding-window efficiency — {CFG.RUN}/{CFG.SUB_RUN}',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(eff_dir, 'efficiency_map_sliding.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = dict(
        det=CFG.DET_NAME, run=CFG.RUN, sub_run=CFG.SUB_RUN,
        feus=CFG.MX17_FEUS, det_z=CFG.DET_PLANE_Z,
        r_mm=R, kernel_mm=KERNEL, grid=GRID,
        n_rays=int(len(d)), n_rays_active=int(inact.sum()),
        integrated_within=integ_within, integrated_has_any=integ_hasany,
        active_box=dict(x0=float(ax0), x1=float(ax1), y0=float(ay0), y1=float(ay1)),
    )
    with open(os.path.join(eff_dir, 'efficiency_map_sliding.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  written: {out_png}')


if __name__ == '__main__':
    main()
