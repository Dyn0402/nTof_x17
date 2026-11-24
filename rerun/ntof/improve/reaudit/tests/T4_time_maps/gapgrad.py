#!/usr/bin/env python3
"""Gap gradient implied by T, per 50 mm cell: g = g_mean (T/T_med)^(1/(1+beta)) on the x-y-mean T map (50 mm cells,
n >= 100), d g/d x_local and d g/d y_local by central differences (np.gradient, 50 mm spacing), mm per mm.
Arrays [ix, iy] on edges50; NaN where a neighbour is missing.  Adds <arm>_gapgrad_x/_y and <arm>_gap50 to tmaps.npz."""
import json
import numpy as np
Z = dict(np.load('tmaps.npz')); G = json.load(open('results_gap.json'))
for arm in 'ACDB':
    m = np.log(Z[f'{arm}_xy_T50']); m[Z[f'{arm}_xy_n50'] < 100] = np.nan
    beta = G[arm]['beta']; g0 = G[arm]['g_mean_mm'] if arm != 'B' else 30.0
    g = g0 * np.exp((m - np.nanmedian(m)) / (1 + beta))
    gx, gy = np.gradient(g, 50.0, 50.0)
    Z[f'{arm}_gap50'], Z[f'{arm}_gapgrad_x'], Z[f'{arm}_gapgrad_y'] = g, gx, gy
    inn = np.zeros_like(g, bool); inn[1:7, 1:7] = True
    print(arm, 'beta %.2f' % beta, 'interior median |grad| x %.4f y %.4f mm/mm' % (np.nanmedian(np.abs(gx[inn])), np.nanmedian(np.abs(gy[inn]))),
          'range x [%.4f, %.4f] y [%.4f, %.4f]' % (np.nanmin(gx[inn]), np.nanmax(gx[inn]), np.nanmin(gy[inn]), np.nanmax(gy[inn])))
np.savez('tmaps.npz', **Z)
