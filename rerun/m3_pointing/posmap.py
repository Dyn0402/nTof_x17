#!/usr/bin/env python3
"""posmap.py -- position dependence of the M3 pointing, measured without any chamber.

In cells of the M3 position (X, Y at the top chamber's z), the covariance of the two chambers' residuals is compared
with the covariance that the per-track model predicts for the same tracks (fit.json).  The ratio is the factor by which
the M3 term of the variance is larger (or smaller) in that part of the telescope.  The ratio is shrunk towards 1 by its
bootstrap error (weight 1/(1 + err^2/(ratio-1)^2) is not used: plain inverse-variance shrinkage with a prior width of
0.3), and the common shift of the two chambers' residual medians is reported alongside (not applied).

Writes $WORK/posmap.json: per view, the cell edges, ratio, error, shrunk factor, and the common median shift.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3common import load, WORK, s68  # noqa: E402
from fit import add_m3, pair_table, cross_P, MATCH  # noqa: E402

EDGES = [-250.0, -100.0, 0.0, 100.0, 250.0]
PRIOR = 0.3
rng = np.random.default_rng(2)


def rcov(a, b):
    return (s68(a + b) ** 2 - s68(a - b) ** 2) / 4


def pred_cov(T, v, g):
    c = MATCH[v]
    Ptb = np.array([cross_P(a, b, c) for a, b in zip(T[f'z_{v}_t'], T[f'z_{v}_b'])])
    return (Ptb * (g['a0'] + g['a1'] * (T[f'mu_v{v}_t'] - 1)) + g['ktb'] * T[f'k_v{v}_t']
            + g['gtb'] * T[f'g_v{v}_t']).to_numpy()


def cell_index(X, Y):
    i = np.clip(np.digitize(X, EDGES) - 1, 0, len(EDGES) - 2)
    j = np.clip(np.digitize(Y, EDGES) - 1, 0, len(EDGES) - 2)
    return i, j


def factor(D, v, pm):
    """the position factor of the M3 term for each ray of D (1 where no map)"""
    if pm is None:
        return np.ones(len(D))
    i, j = cell_index(D[f'm3_X_at_{v}'].to_numpy(), D[f'm3_Y_at_{v}'].to_numpy())
    return np.array(pm[v]['factor'])[i, j]


def main():
    par = json.load(open(os.path.join(WORK, 'decompose.json')))
    fr = json.load(open(os.path.join(WORK, 'fit.json')))
    T = pair_table(add_m3(load(), par))
    out = dict(edges=EDGES, prior=PRIOR)
    for v in 'xy':
        pc = pred_cov(T, v, fr[v]['glob'])
        i, j = cell_index(T[f'm3_X_at_{v}_t'].to_numpy(), T[f'm3_Y_at_{v}_t'].to_numpy())
        n = len(EDGES) - 1
        R, E, F, B, N = (np.full((n, n), np.nan) for _ in range(5))
        a, b = T[f'res_{v}_t'].to_numpy(), T[f'res_{v}_b'].to_numpy()
        # the s68-based covariance and the model's Gaussian covariance differ by a global factor (heavy tails):
        # only the pattern across the telescope is used, normalised to the whole-telescope ratio
        glob = rcov(a, b) / np.mean(pc)
        out[f'{v}_global_ratio'] = float(glob)
        for ii in range(n):
            for jj in range(n):
                m = (i == ii) & (j == jj)
                N[ii, jj] = m.sum()
                if m.sum() < 150:
                    R[ii, jj], E[ii, jj], F[ii, jj], B[ii, jj] = 1, np.inf, 1, 0
                    continue
                p = np.mean(pc[m])
                R[ii, jj] = rcov(a[m], b[m]) / p / glob
                bs = [rcov(a[m][k], b[m][k]) / p / glob for k in (rng.integers(0, m.sum(), m.sum()) for _ in range(200))]
                E[ii, jj] = np.std(bs)
                w = 1 / E[ii, jj] ** 2; w0 = 1 / PRIOR ** 2
                F[ii, jj] = max((w * R[ii, jj] + w0) / (w + w0), 0.3)
                B[ii, jj] = 0.5 * (np.median(a[m]) + np.median(b[m]))
        out[v] = dict(ratio=R.tolist(), err=np.where(np.isfinite(E), E, -1).tolist(), factor=F.tolist(),
                      common_shift=B.tolist(), n=N.tolist())
        print(f'view {v} (M3 {MATCH[v]}): rows = M3 X cells, columns = M3 Y cells')
        for ii in range(n):
            print('   ' + '  '.join(f'{R[ii, jj]:4.2f}±{min(E[ii, jj], 9):4.2f}->{F[ii, jj]:4.2f} ({B[ii, jj]:+.2f})'
                                   for jj in range(n)))
    # per row: slope of the X-view residual median along M3 Y (a scale-like shift; M3 plane scale or chamber pitch)
    D = load(); e = np.arange(-200, 201, 50.0); c = 0.5 * (e[1:] + e[:-1])
    out['row_slope_um_per_mm'] = {}
    for tag, S in D.groupby('tag'):
        med = np.array([np.median(S.res_x[(S.m3_Y_at_x >= a) & (S.m3_Y_at_x < b)])
                        if ((S.m3_Y_at_x >= a) & (S.m3_Y_at_x < b)).sum() > 80 else np.nan for a, b in zip(e[:-1], e[1:])])
        m = np.isfinite(med)
        out['row_slope_um_per_mm'][tag] = dict(slope=float(np.polyfit(c[m], med[m], 1)[0] * 1000),
                                               z=float(S.z_x.iloc[0]))
    print('row slopes [um/mm]:', {k: round(v['slope'], 2) for k, v in out['row_slope_um_per_mm'].items()})
    json.dump(out, open(os.path.join(WORK, 'posmap.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
