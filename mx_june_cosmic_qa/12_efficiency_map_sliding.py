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


KERNEL = _argf('--kernel=', 25.0)   # FIXED-kernel spatial smoothing radius [mm]
GRID = int(_argf('--grid=', 120))   # grid points per axis
MIN_RAYS = int(_argf('--min=', 30))  # min rays in kernel to colour a grid point
R = _argf('--r=', 5.0)              # label only (match radius baked into 'within')

# Auto-kernel: pick the SMALLEST fixed kernel whose EDGE points still capture ~EDGE_HITS
# rays (the finest map the local statistics allow), floored at KMIN mm. Set --edge-hits=10
# for ~10 hits/kernel at the active-area edge; the kernel is derived per detector from the
# ray density (higher-stats detectors -> smaller kernel -> finer efficiency structure).
EDGE_HITS = int(_argf('--edge-hits=', 0))
KMIN = _argf('--kmin=', 2.5)        # kernel radius floor [mm]

# Adaptive (k-nearest-neighbour) kernel: instead of a fixed radius, each grid point
# uses the SMALLEST radius that captures TARGET rays -> the finest resolution the local
# statistics allow, with a constant per-point error (binomial with N=TARGET). The kernel
# shrinks toward mm scale where rays are dense and grows where they are sparse (capped at
# MAXKERNEL; points needing more are masked). The kernel-radius map IS the local-resolution
# map. Enable with --adaptive.
ADAPTIVE = '--adaptive' in sys.argv
TARGET = int(_argf('--target=', 60))    # rays per kernel (sets the statistical error)
MAXKERNEL = _argf('--maxkernel=', 30.0)  # cap on adaptive radius [mm]


def adaptive_map(x, y, within, has_any, x_grid, y_grid, target, max_r):
    """k-NN adaptive kernel. For each grid point, take the `target` nearest rays; the
    kernel radius is the distance to the target-th ray (= local resolution). Returns
    (efficiency, has_any, radius) maps; points whose radius exceeds `max_r` are NaN."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x, y]))
    gx, gy = np.meshgrid(x_grid, y_grid, indexing='ij')
    gpts = np.column_stack([gx.ravel(), gy.ravel()])
    k = min(target, len(x))
    dist, idx = tree.query(gpts, k=k)
    if k == 1:
        dist = dist[:, None]; idx = idx[:, None]
    radius = dist[:, -1]
    eff = within[idx].mean(axis=1)
    anyh = has_any[idx].mean(axis=1)
    bad = radius > max_r
    eff[bad] = np.nan; anyh[bad] = np.nan; radius[bad] = np.nan
    shp = (len(x_grid), len(y_grid))
    return eff.reshape(shp), anyh.reshape(shp), radius.reshape(shp)


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


def adaptive_main(eff_dir, d, x, y, within, has_any, box4, inact, integ_within, integ_hasany):
    ax0, ax1, ay0, ay1 = box4
    grid_n = max(GRID, 200)                       # adaptive deserves a fine grid
    x_grid = np.linspace(ax0, ax1, grid_n)
    y_grid = np.linspace(ay0, ay1, grid_n)
    print(f'{CFG.DET_NAME} {CFG.RUN}/{CFG.SUB_RUN}: ADAPTIVE k-NN, target={TARGET} rays/kernel, '
          f'maxR={MAXKERNEL:.0f} mm, {grid_n}x{grid_n} grid, {len(d)} rays', flush=True)
    eff_w, eff_a, radius = adaptive_map(x, y, within, has_any, x_grid, y_grid, TARGET, MAXKERNEL)

    rad = radius[np.isfinite(radius)]
    if len(rad):
        rmin, med = float(rad.min()), float(np.median(rad))
        p90, p95, rmax = (float(np.percentile(rad, q)) for q in (90, 95, 100))
        f2, f5, f10 = (float(np.mean(rad <= t)) for t in (2, 5, 10))
        print(f'  kernel radius over covered area: min={rmin:.1f} median={med:.1f} '
              f'p90={p90:.1f} p95={p95:.1f} max={rmax:.1f} mm | '
              f'<=2mm:{f2*100:.0f}%  <=5mm:{f5*100:.0f}%  <=10mm:{f10*100:.0f}%')
        print(f'  -> for a FIXED kernel matching edge stats, use ~p90={p90:.1f} mm '
              f'(--kernel={p90:.0f} --min={TARGET})')
    else:
        rmin = med = p90 = p95 = rmax = f2 = f5 = f10 = float('nan')

    extent = [x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]]
    box = dict(xy=(ax0, ay0), width=ax1 - ax0, height=ay1 - ay0)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('lightgrey')
    cmap_r = plt.get_cmap('turbo').copy(); cmap_r.set_bad('lightgrey')

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, data, label in [
        (axes[0], eff_w, f'efficiency (reco within {R:g} mm)'),
        (axes[1], eff_a, 'has_any (fired any strip)'),
    ]:
        im = ax.imshow(data.T, origin='lower', extent=extent, aspect='equal',
                       cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.3))
        ax.set_xlabel('reference X [mm]'); ax.set_ylabel('reference Y [mm]')
        ax.set_title(f'{CFG.DET_NAME}  {label}\nadaptive k-NN, {TARGET} rays/kernel')

    im3 = axes[2].imshow(radius.T, origin='lower', extent=extent, aspect='equal',
                         cmap=cmap_r, vmin=0, vmax=MAXKERNEL)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04,
                 label='kernel radius [mm]  (= local resolution)')
    axes[2].add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.3))
    axes[2].set_xlabel('reference X [mm]'); axes[2].set_ylabel('reference Y [mm]')
    axes[2].set_title(f'local kernel radius\n(smaller = finer; median {med:.1f} mm)')

    fig.suptitle(f'{CFG.DET_NAME} ADAPTIVE-kernel efficiency — {CFG.RUN}/{CFG.SUB_RUN}',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(eff_dir, 'efficiency_map_adaptive.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close(fig)

    summary = dict(
        det=CFG.DET_NAME, run=CFG.RUN, sub_run=CFG.SUB_RUN, mode='adaptive',
        target_rays=TARGET, max_kernel_mm=MAXKERNEL, grid=grid_n,
        n_rays=int(len(d)), n_rays_active=int(inact.sum()),
        integrated_within=integ_within, integrated_has_any=integ_hasany,
        kernel_radius_min_mm=rmin, kernel_radius_median_mm=med,
        kernel_radius_p90_mm=p90, kernel_radius_p95_mm=p95, kernel_radius_max_mm=rmax,
        area_frac_le2mm=f2, area_frac_le5mm=f5, area_frac_le10mm=f10,
        active_box=dict(x0=float(ax0), x1=float(ax1), y0=float(ay0), y1=float(ay1)))
    with open(os.path.join(eff_dir, 'efficiency_map_adaptive.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  written: {out_png}')


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
    _kdesc = f'auto (~{EDGE_HITS} edge hits, floor {KMIN:g} mm)' if EDGE_HITS > 0 else f'r={KERNEL:.0f} mm'
    print(f'{CFG.DET_NAME} {CFG.RUN}/{CFG.SUB_RUN}: {len(d)} clean M3 rays, '
          f'kernel {_kdesc}, {GRID}x{GRID} grid', flush=True)

    # Active-area box from the reco'd (within) ray footprint — same idea as 08.
    hit = d[d['within']]
    if len(hit) < 20:
        hit = d
    ax0, ax1 = np.percentile(hit['x'], [0.5, 99.5])
    ay0, ay1 = np.percentile(hit['y'], [0.5, 99.5])
    inact = (x >= ax0) & (x <= ax1) & (y >= ay0) & (y <= ay1)
    integ_within = float(within[inact].mean()) if inact.any() else float('nan')
    integ_hasany = float(has_any[inact].mean()) if inact.any() else float('nan')

    if ADAPTIVE:
        adaptive_main(eff_dir, d, x, y, within, has_any,
                      (ax0, ax1, ay0, ay1), inact, integ_within, integ_hasany)
        return

    kernel, min_rays = KERNEL, MIN_RAYS
    if EDGE_HITS > 0:
        area = (ax1 - ax0) * (ay1 - ay0)
        dens = inact.sum() / area if area > 0 else 0.0        # rays / mm^2 in active area
        # a straight-edge grid point sees ~half a kernel disk of active area:
        #   EDGE_HITS ~= dens * (pi k^2 / 2)  ->  k = sqrt(2 EDGE_HITS / (pi dens))
        kernel = float(np.sqrt(2 * EDGE_HITS / (np.pi * dens))) if dens > 0 else KERNEL
        kernel = max(KMIN, kernel)
        min_rays = max(5, EDGE_HITS // 2)                     # don't mask the very edge
        print(f'  auto-kernel for ~{EDGE_HITS} rays at the edge: density={dens:.3f}/mm^2 '
              f'-> kernel={kernel:.1f} mm (floor {KMIN:g}), min_rays={min_rays}')

    pad = kernel
    x_grid = np.linspace(ax0 - pad, ax1 + pad, GRID)
    y_grid = np.linspace(ay0 - pad, ay1 + pad, GRID)

    eff_w, cnt = sliding_map(x, y, within, x_grid, y_grid, kernel, min_rays)
    eff_a, _ = sliding_map(x, y, has_any, x_grid, y_grid, kernel, min_rays)
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
        ax.set_title(f'{CFG.DET_NAME}  {label}\nsliding kernel r={kernel:.1f} mm')

    cnt_m = np.where(cnt >= min_rays, cnt, np.nan)
    im3 = axes[2].imshow(cnt_m.T, origin='lower', extent=extent, aspect='equal', cmap=cmap_c)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='rays in kernel')
    axes[2].add_patch(plt.Rectangle(**box, fill=False, ec='red', lw=1.3))
    axes[2].set_xlabel('reference X [mm]'); ax = axes[2]; ax.set_ylabel('reference Y [mm]')
    axes[2].set_title(f'rays per kernel\n(grey < {min_rays})')

    fig.suptitle(f'{CFG.DET_NAME} sliding-window efficiency — {CFG.RUN}/{CFG.SUB_RUN}',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(eff_dir, 'efficiency_map_sliding.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)

    summary = dict(
        det=CFG.DET_NAME, run=CFG.RUN, sub_run=CFG.SUB_RUN,
        feus=CFG.MX17_FEUS, det_z=CFG.DET_PLANE_Z,
        r_mm=R, kernel_mm=kernel, min_rays=min_rays, edge_hits=EDGE_HITS, grid=GRID,
        n_rays=int(len(d)), n_rays_active=int(inact.sum()),
        integrated_within=integ_within, integrated_has_any=integ_hasany,
        active_box=dict(x0=float(ax0), x1=float(ax1), y0=float(ay0), y1=float(ay1)),
    )
    with open(os.path.join(eff_dir, 'efficiency_map_sliding.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  written: {out_png}')


if __name__ == '__main__':
    main()
