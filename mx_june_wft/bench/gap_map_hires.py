#!/usr/bin/env python3
"""
gap_map_hires.py — high-resolution charge-visible drift-gap map.

Sliding-window endpoint fit: at each grid point, stack the normalised NNLS
charge-arrival profiles (X plane, contained tracks) of all events within
KERNEL_R, fit the erfc endpoint, convert to mm with the plane's geometric
drift speed. Diverging colour scale centred on the 30 mm mechanical gap, so
"short of mechanical" is directly visible.

    ../../.venv/bin/python mx_june_wft/bench/gap_map_hires.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO]

KERNEL_R = 45.0        # mm
GRID_STEP = 8.0        # mm
MIN_EVENTS = 80
GAP_MECH = 30.0
U = (np.arange(18) + 0.5) * 60.0

DETS = [
    ('det3', '/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_'
     '6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/gap_study/'
     'event_profiles.parquet', 36.83),
    ('det2', '/home/dylan/x17/cosmic_bench/Analysis/mx17_det2_det3_overnight_'
     '6-22-26/longer_run/mx17_2/wft/gap_study/event_profiles.parquet', 39.04),
]


def sharp(u, A, T, sig):
    return A * 0.5 * erfc((u - T) / (np.sqrt(2) * sig))


def fit_T(P):
    m = P.mean(axis=0)
    e = np.maximum(P.std(axis=0) / np.sqrt(len(P)), 1e-5)
    sel = U < 1050
    try:
        p, c = curve_fit(sharp, U[sel], m[sel], p0=[m[:5].mean(), 700, 60],
                         sigma=e[sel], absolute_sigma=True, maxfev=20000)
        return float(p[1]), float(np.sqrt(c[1, 1]))
    except Exception:
        return np.nan, np.nan


def build_map(path, v_geom):
    df = pd.read_parquet(path)
    g = df[(df.plane == 'x') & df.contained & (df.chi2dof < 250)].copy()
    Q = g[[f'q{i}' for i in range(18)]].to_numpy()
    Q = Q / Q.sum(axis=1, keepdims=True)
    x, y = g.ref_x.to_numpy(), g.ref_y.to_numpy()
    xs = np.arange(np.percentile(x, 1), np.percentile(x, 99), GRID_STEP)
    ys = np.arange(np.percentile(y, 1), np.percentile(y, 99), GRID_STEP)
    M = np.full((len(ys), len(xs)), np.nan)
    E = np.full_like(M, np.nan)
    for j, yc in enumerate(ys):
        dy2 = (y - yc) ** 2
        for i, xc in enumerate(xs):
            s = dy2 + (x - xc) ** 2 < KERNEL_R ** 2
            if s.sum() < MIN_EVENTS:
                continue
            T, Te = fit_T(Q[s])
            if np.isfinite(T):
                M[j, i] = T * v_geom / 1000.0
                E[j, i] = Te * v_geom / 1000.0
    return xs, ys, M, E, g


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.6), layout='constrained',
                            gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
    fig.suptitle('Charge-visible drift gap from stacked charge-arrival '
                 f'profiles (X plane, contained tracks, kernel r = '
                 f'{KERNEL_R:.0f} mm)', fontsize=12)
    ims = []
    maps = {}
    for k, (det, path, vg) in enumerate(DETS):
        xs, ys, M, E, g = build_map(path, vg)
        maps[det] = (xs, ys, M, E)
        ax = axs[k]
        im = ax.pcolormesh(xs, ys, M, cmap='RdBu', vmin=25.0, vmax=35.0,
                           shading='nearest')
        ims.append(im)
        cs = ax.contour(xs, ys, M, levels=[26, 27, 28, 29, 30],
                        colors='k', linewidths=0.6, alpha=0.55)
        ax.clabel(cs, fmt='%.0f', fontsize=7)
        med = np.nanmedian(M)
        lab = 'control' if det == 'det2' else 'dished'
        ax.set_title(f'{det} ({lab}) — median {med:.1f} mm', fontsize=11)
        ax.set_xlabel('x [mm]')
        if k == 0:
            ax.set_ylabel('y [mm]')
        ax.set_aspect('equal')
        n = np.isfinite(M).sum()
        print(f'{det}: {n} grid points, median {med:.2f} mm, '
              f'p5-p95 {np.nanpercentile(M,5):.1f}-{np.nanpercentile(M,95):.1f}, '
              f'median err {np.nanmedian(E):.2f} mm')
    cb = fig.colorbar(ims[0], ax=axs[:2], shrink=0.9, pad=0.015)
    cb.set_label('charge-visible drift gap [mm]  (white = 30 mm mechanical)')

    # slices: det3 gap vs y in three x bands + det2 reference band
    ax = axs[2]
    xs, ys, M, E = maps['det3']
    bands = [(0.05, 0.35, '#a6611a', 'det3, low x'),
             (0.35, 0.65, '#d01c8b', 'det3, mid x'),
             (0.65, 0.95, '#5e3c99', 'det3, high x')]
    for lo, hi, col, lab in bands:
        i0, i1 = int(lo * len(xs)), int(hi * len(xs))
        prof = np.nanmean(M[:, i0:i1], axis=1)
        ax.plot(ys, prof, color=col, lw=2, label=lab)
    xs2, ys2, M2, _ = maps['det2']
    m2 = np.nanmean(M2, axis=1)
    ax.plot(ys2, m2, color='0.45', lw=2, ls='--', label='det2 (all x)')
    ax.axhline(GAP_MECH, color='k', lw=1, ls=':')
    ax.text(ys[2], GAP_MECH + 0.15, '30 mm mechanical', fontsize=8)
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('charge-visible gap [mm]')
    ax.set_ylim(24.5, 32.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('gap vs y: det3 dished/tilted, det2 flat at 30', fontsize=11)

    out = os.path.join(os.path.dirname(DETS[0][1]), 'gap_map_hires.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', out)
    out2 = os.path.join(REPO, 'mx_june_wft', 'gap_map_hires.png')
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    print('wrote', out2)


if __name__ == '__main__':
    main()
