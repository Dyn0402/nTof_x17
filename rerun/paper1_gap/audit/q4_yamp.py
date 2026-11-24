"""q4_yamp.py -- item 4: the geometric gap varies along y about 1.4 times more than the drift-time gap.  On the 4x4 cells of
the audited analysis ($WORK/cellobs.json, both golden runs), the y-shape (slope and curvature along W, with a linear U term)
of four gap estimates: geometric X (G), geometric Y (GY), drift time alone (T, beta 0.68), and the beta-free product
v_cell x T_cell (X, Y, mean).  Weighted LSQ, model 1 + U + W + W^2.  Same cells for every estimate (those with G)."""
import json
import numpy as np

c = json.load(open('/media/ucla/work/claude-jobs/e1a7a26e/gap/cellobs.json'))
for run in ('run1', 'run2'):
    L = c[run]; U = np.array([x['U'] for x in L]); W = np.array([x['W'] for x in L])
    arr = lambda k: np.array([x[k] for x in L], float)
    T, vx, vy = arr('T'), arr('v_x'), arr('v_y'); Te, vxe, vye = arr('T_err'), arr('v_x_err'), arr('v_y_err')
    vm, vme = 0.5 * (vx + vy), 0.5 * np.hypot(vxe, vye)
    est = {'geom X (G)': (arr('G'), arr('G_err')), 'geom Y (GY)': (arr('GY'), arr('GY_err')),
           'T only, beta 0.68': (28.2 * (T / np.nanmean(T)) ** (1 / 1.68), 28.2 / 1.68 * Te / T),
           'v_x * T': (28.2 * vx * T / np.nanmean(vx * T), 28.2 * np.hypot(Te / T, vxe / vx)),
           'v_y * T': (28.2 * vy * T / np.nanmean(vy * T), 28.2 * np.hypot(Te / T, vye / vy)),
           'v_mean * T': (28.2 * vm * T / np.nanmean(vm * T), 28.2 * np.hypot(Te / T, vme / vm))}
    print(run)
    for cells in ('G', 'all'):
        base = np.isfinite(arr('G')) if cells == 'G' else np.ones(len(L), bool)
        for k, (g, e) in est.items():
            ok = base & np.isfinite(g) & np.isfinite(e) & (e > 0)
            if ok.sum() < 6:
                continue
            X = np.c_[np.ones(ok.sum()), U[ok], W[ok], W[ok] ** 2]
            A = X / e[ok][:, None]
            cf = np.linalg.lstsq(A, g[ok] / e[ok], rcond=None)[0]
            cov = np.linalg.pinv(A.T @ A)
            print(f'   cells {cells:3s} {k:18s} n {ok.sum():2d}  slope along y {cf[2]:+.2f} +- {np.sqrt(cov[2, 2]):.2f}  '
                  f'curvature along y {2 * cf[3]:+.2f} +- {2 * np.sqrt(cov[3, 3]):.2f}')
