#!/usr/bin/env python3
"""task2.py -- does the horizontal capsule pointing add information to the lines-only spline map?  (A, C, and D)

Pointing cells (pointcells.py, 25 mm, all runs, T5 validity): c = (unmapped slope - expected capsule slope e), a global
x-view residual centre with stat error err.  In the joint fit each valid cell adds one row
    c_loc = kappa (a_x(cell centre) + s_x(cell centre) e_loc) + alpha + gamma e_loc
with weight sigma_line^2 / (err^2 + floor^2); alpha, gamma free per chamber, kappa scanned.  The floor is set so that the
cells scatter with chi2/ndf = 1 about the lines-only map (all lines) after the best (kappa, alpha, gamma).
Held-out: lines of one run parity train (with ALL pointing cells: beam data, independent of every cosmic line), the
other parity tests; lambdas = the lines-only CV choice of that training half.  Floors f x {0.25, 0.5, 1, 2, 4}.

    task2.py A|C|D   -> t2_models_<arm>.csv, t2_diffs_<arm>.csv, t2_nuis_<arm>.csv, t2_floor_<arm>.json
"""
import json
import sys

import numpy as np
import pandas as pd

from r3lib import Spline, HERE
from t6lib import rows_ac, rows_d, G, mad, widths, NoMap, Free
from q34_models import diffs

KAP = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0])
LT_AC, LS_AC = np.array([0.003, 0.01, 0.03, 0.1]), np.array([1e-3, 3e-3, 1e-2, 0.1, 1.0])
LT_D, LS_D = 10.0 ** np.arange(-2, 3.1, 1.0), 10.0 ** np.arange(-3, 2.1, 1.0)
rng = np.random.default_rng(9)


def point_rows(arm, lab='all'):
    z = np.load(HERE / f'pc_{arm}_{lab}.npz')
    e = z['edges']; c = 0.5 * (e[1:] + e[:-1])
    X, Y = np.meshgrid(c, c, indexing='ij')
    v = z['valid'] & np.isfinite(z['c']) & np.isfinite(z['err'])
    gx = G[arm][0]
    return dict(x=X[v], y=Y[v], c_loc=gx * z['c'][v], e_loc=gx * z['e'][v], err=z['err'][v], valid=v, edges=e)


def floor_from(mdl, P):
    """kappa, alpha, gamma by WLS of c on [pred, 1, e] and the floor giving chi2/ndf = 1."""
    f = mdl.fields(P['x'], P['y'])
    pred = f['a_x'] + f['s_x'] * P['e_loc']
    X = np.column_stack([pred, np.ones(len(pred)), P['e_loc']])
    fl = 0.0
    for _ in range(30):
        w = 1 / (P['err'] ** 2 + fl ** 2)
        b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * P['c_loc']))
        res = P['c_loc'] - X @ b
        ndf = len(res) - 3
        lo, hi = 0.0, 0.05
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.sum(res ** 2 / (P['err'] ** 2 + mid ** 2)) > ndf:
                lo = mid
            else:
                hi = mid
        if abs(mid - fl) < 1e-6:
            break
        fl = mid
    chi0 = float(np.sum(res ** 2 / P['err'] ** 2) / ndf)
    return dict(floor=float(fl), kappa=float(b[0]), alpha=float(b[1]), gamma=float(b[2]), chi2ndf_stat=chi0,
                rms_res=float(np.std(res)), n_cells=len(res), corr=float(np.corrcoef(pred, P['c_loc'])[0, 1]))


def covered(D, P):
    e = P['edges']; nc = len(e) - 1
    ix = np.digitize(D.x, e) - 1; iy = np.digitize(D.y, e) - 1
    ok = (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nc)
    out = np.zeros(len(D), bool)
    out[ok] = P['valid'][ix[ok], iy[ok]]
    return out


def main(arm):
    D = rows_d() if arm == 'D' else rows_ac(arm)
    D['part'] = D.run.str.replace('run_', '').astype(int) % 2       # run parity, also for D (CV over runs)
    LT, LS = (LT_D, LS_D) if arm == 'D' else (LT_AC, LS_AC)
    P = point_rows(arm)
    # lines-only spline on all clean lines: floor and nuisances
    base = Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS).fit(D[D.clean])
    fl = floor_from(base, P)
    fl.update(lam_t=base.lam_t_sel, lam_s=base.lam_s_sel)
    json.dump(fl, open(HERE / f't2_floor_{arm}.json', 'w'), indent=1)
    print(arm, fl, flush=True)
    mults = (0.25, 0.5, 1.0, 2.0, 4.0)
    names = ['none', 'cells_current'] + ['lines'] + [f'joint_f{m:g}' for m in mults]
    R = {k: np.full(len(D), np.nan) for k in names}
    nuis = []
    for tr in (0, 1):
        trn = D[(D.part == tr) & D.clean]
        tst = (D.part != tr).to_numpy() & D.allok.to_numpy()
        lines = Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS).fit(trn)
        R['lines'][tst] = lines.corrected_resid(D[tst])
        R['none'][tst] = NoMap().corrected_resid(D[tst])
        cur = (Free(edges=np.arange(-200.0, 201.0, 100.0), n_min=60) if arm == 'D' else Free()).fit(trn)
        R['cells_current'][tst] = cur.corrected_resid(D[tst])
        for m in mults:
            pt = dict(P, floor=fl['floor'] * m, sig_line=np.mean(list(lines.sig.values())), kappas=KAP)
            jm = Spline(tilt='pot', scale='sep', lam_t=lines.lam_t_sel, lam_s=lines.lam_s_sel, point=pt).fit(trn)
            R[f'joint_f{m:g}'][tst] = jm.corrected_resid(D[tst])
            nuis.append(dict(train_part=tr, mult=m, floor=pt['floor'], kappa=jm.kappa, alpha=jm.nuis[0],
                             gamma=jm.nuis[1], lam_t=lines.lam_t_sel, lam_s=lines.lam_s_sel))
            print(arm, tr, m, nuis[-1], flush=True)
    cov = covered(D, P)
    c = D.clean.to_numpy()
    runs = D.run.to_numpy(); ur = np.unique(runs)
    rs = {r: np.flatnonzero(runs == r) for r in ur}
    boots = [np.concatenate([rs[r] for r in rng.choice(ur, len(ur))]) for _ in range(300)]
    rows = []
    for w in (0, 1):
        vw = (D.view == w).to_numpy()
        for reg, rm in (('all', np.ones(len(D), bool)), ('covered', cov), ('uncovered', ~cov)):
            for k, e in R.items():
                sel = vw & c & rm & np.isfinite(e)
                w68, w95 = widths(e[sel])
                rows.append(dict(arm=arm, view='xy'[w], region=reg, model=k, n=int(sel.sum()), sig=mad(e[sel]),
                                 w68=w68, w95=w95))
    T = pd.DataFrame(rows)
    T.to_csv(HERE / f't2_models_{arm}.csv', index=False)
    pd.DataFrame(nuis).to_csv(HERE / f't2_nuis_{arm}.csv', index=False)
    # paired differences vs lines-only (all lines, and covered cells only)
    out = []
    for w in (0, 1):
        vw = (D.view == w).to_numpy()
        for reg, rm in (('all', np.ones(len(D), bool)), ('covered', cov)):
            sel = vw & c & rm
            for k in names:
                if k == 'lines':
                    continue
                d0 = mad(R[k][sel]) - mad(R['lines'][sel])
                bb = [mad(R[k][b][sel[b]]) - mad(R['lines'][b][sel[b]]) for b in boots]
                out.append(dict(arm=arm, view='xy'[w], region=reg, a=k, b='lines', d_sig=d0, err=float(np.std(bb))))
    pd.DataFrame(out).to_csv(HERE / f't2_diffs_{arm}.csv', index=False)
    np.savez(HERE / f't2_resid_{arm}.npz', covered=cov, **R)
    pd.set_option('display.width', 250)
    print(T.pivot_table(index='model', columns=['region', 'view'], values='sig').round(4).to_string())
    print(pd.DataFrame(out).round(5).to_string())


if __name__ == '__main__':
    main(sys.argv[1])
