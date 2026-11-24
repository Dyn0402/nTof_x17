#!/usr/bin/env python3
"""q34_models.py -- Q3 (constrained surface models vs free cells, held-out) for A and C, and Q4 for D.

Held-out protocol of angle_map.py: train on clean lines of one parity (run parity for A, C; event parity for D, as
bd_map.py), test on the other, both directions, pooled so that every line is tested once.  Metrics on the clean test
lines (sigma = 1.4826 MAD, w68 = (q84 - q16)/2, w95 = (q97.5 - q2.5)/2, edge = |x| or |y| > 125 mm) and on all test
lines with |j| < 0.45 (sig_all, as angle_map.py).  Errors on differences: run-block bootstrap of the pooled residuals.
-> models_<arm>.csv, diffs_<arm>.csv, coeffs.json, model_cells.npz
"""
import json
import sys

import numpy as np
import pandas as pd

from t6lib import rows_ac, rows_d, Poly, Free, NoMap, mad, widths, OUT, L

NB = 400
rng = np.random.default_rng(5)


def models_ac():
    m = {'none': NoMap(), 'global': Free(globalonly=True), 'free50': Free()}
    for n in (2, 3, 4, 5):
        for sm in (0, 1, 2, 3):
            m[f'pot{n}_s{sm}'] = Poly(n=n, pot=True, m=sm)
        m[f'pot{n}_s2sep'] = Poly(n=n, pot=True, m=2, sep=True)
        m[f'pot{n}_s2off'] = Poly(n=n, pot=True, m=2, soff=True)
        m[f'ind{n}_s2'] = Poly(n=n, pot=False, m=2)
        m[f'ind{n}_s2sep'] = Poly(n=n, pot=False, m=2, sep=True)
        m[f'pot{n}_s2_rot'] = Poly(n=n, pot=True, m=2, rot=True)
        m[f'ind{n}_s2_rot'] = Poly(n=n, pot=False, m=2, rot=True)
    for n in (6, 7):
        m[f'pot{n}_s2'] = Poly(n=n, pot=True, m=2)
        m[f'ind{n}_s2'] = Poly(n=n, pot=False, m=2)
        m[f'pot{n}_s3'] = Poly(n=n, pot=True, m=3)
        m[f'ind{n}_s3'] = Poly(n=n, pot=False, m=3)
    E50 = np.arange(-200.0, 201.0, 50.0)
    for sm in (0, 1, 2, 3):
        m[f'cellA_s{sm}'] = Poly(cellA=E50, m=sm)
        m[f'cellA_s{sm}sep'] = Poly(cellA=E50, m=sm, sep=True)
    return m


def models_d():
    m = {'none': NoMap(), 'global': Free(globalonly=True, edges=np.arange(-200.0, 201.0, 100.0), n_min=60),
         'free100': Free(edges=np.arange(-200.0, 201.0, 100.0), n_min=60)}
    for n in (1, 2, 3, 4):
        for sm in (0, 1, 2):
            m[f'pot{n}_s{sm}'] = Poly(n=n, pot=True, m=sm)
            m[f'pot{n}_s{sm}sep'] = Poly(n=n, pot=True, m=sm, sep=True)
            m[f'ind{n}_s{sm}'] = Poly(n=n, pot=False, m=sm)
            m[f'ind{n}_s{sm}sep'] = Poly(n=n, pot=False, m=sm, sep=True)
    return m


def evaluate(D, models):
    """pooled held-out residuals per model on the test rows (all rows with allok), plus full-data fits."""
    R = {k: np.full(len(D), np.nan) for k in models}
    for tr in (0, 1):
        trn = D[(D.part == tr) & D.clean]
        tst = (D.part != tr).to_numpy() & D.allok.to_numpy()
        for k, mdl in models.items():
            mdl.fit(trn)
            R[k][tst] = mdl.corrected_resid(D[tst])
    return R


def metrics(D, R, models):
    c = D.clean.to_numpy()
    edge = ((D.x.abs() > 125) | (D.y.abs() > 125)).to_numpy()
    runs = D.run.to_numpy()
    ur = np.unique(runs)
    rowsets = {r: np.flatnonzero(runs == r) for r in ur}
    boots = [np.concatenate([rowsets[r] for r in rng.choice(ur, len(ur))]) for _ in range(NB)]
    rows = []
    for w in (0, 1):
        vw = (D.view == w).to_numpy()
        for k in models:
            e = R[k]
            sel = vw & c & np.isfinite(e)
            w68, w95 = widths(e[sel])
            b = [mad(e[bi][(vw & c)[bi]]) for bi in boots[:150]]
            rows.append(dict(view='xy'[w], model=k, n_clean=int(sel.sum()), sig_clean=mad(e[sel]),
                             sig_clean_err=float(np.std(b)), w68=w68, w95=w95,
                             sig_edge=mad(e[sel & edge]), sig_int=mad(e[sel & ~edge]), n_edge=int((sel & edge).sum()),
                             sig_all=mad(e[vw & np.isfinite(e)]),
                             tail_all=float(np.mean(np.abs(e[vw & np.isfinite(e)]) > 0.15)),
                             npar=int(len(models[k].beta)) if hasattr(models[k], 'beta') else np.nan))
    return pd.DataFrame(rows), boots


def diffs(D, R, boots, pairs):
    c = D.clean.to_numpy()
    edge = ((D.x.abs() > 125) | (D.y.abs() > 125)).to_numpy()
    out = []
    for w in (0, 1):
        vw = (D.view == w).to_numpy()
        for a, b in pairs:
            for reg, rm in (('all', np.ones(len(D), bool)), ('edge', edge)):
                sel = vw & c & rm
                d0 = mad(R[a][sel]) - mad(R[b][sel])
                bb = [mad(R[a][bi][sel[bi]]) - mad(R[b][bi][sel[bi]]) for bi in boots]
                out.append(dict(view='xy'[w], a=a, b=b, region=reg, d_sig=d0, err=float(np.std(bb))))
    return pd.DataFrame(out)


def cells(mdl, D0, edges):
    """tilt (local and global), scale at cell centres, index [ix, iy]."""
    c = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(c, c, indexing='ij')
    out = {}
    for w in (0, 1):
        g = (D0[D0.view == w].g.iloc[0])
        Dg = pd.DataFrame(dict(x=X.ravel(), y=Y.ravel(), view=w, j=0.0, g=g, go=1, jo=0.0, slope_o=0.0,
                               slope=0.0, r=0.0))
        t, s, _ = mdl.tilt_scale(Dg)
        out['xy'[w]] = dict(a_loc=t.reshape(X.shape), a_glob=(g * t).reshape(X.shape), s=s.reshape(X.shape))
    return out


def main(which):
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    coeffs = json.load(open(OUT / 'coeffs.json')) if (OUT / 'coeffs.json').exists() else {}
    cz = dict(np.load(OUT / 'model_cells.npz')) if (OUT / 'model_cells.npz').exists() else {}
    arms = ['A', 'C', 'D'] if which == 'all' else [which]
    for arm in arms:
        D = rows_d() if arm == 'D' else rows_ac(arm)
        models = models_d() if arm == 'D' else models_ac()
        R = evaluate(D, models)
        T, boots = metrics(D, R, models)
        T.to_csv(OUT / f'models_{arm}.csv', index=False)
        ref = 'free100' if arm == 'D' else 'free50'
        if arm == 'D':
            pairs = [(k, ref) for k in models if k != ref] + [('pot2_s0', 'ind2_s0'), ('pot2_s1', 'ind2_s1'),
                                                               ('pot3_s1', 'ind3_s1'), ('pot2_s0', 'global')]
        else:
            pairs = [(k, ref) for k in models if k != ref] + \
                [(f'pot{n}_s2', f'ind{n}_s2') for n in (2, 3, 4, 5, 6, 7)] + \
                [(f'pot{n}_s2sep', f'ind{n}_s2sep') for n in (2, 3, 4, 5)] + \
                [(f'pot{n}_s2_rot', f'ind{n}_s2_rot') for n in (2, 3, 4, 5)] + \
                [(f'pot{n}_s2sep', f'pot{n}_s2') for n in (2, 3, 4, 5)] + \
                [('cellA_s2', 'cellA_s2sep'), ('cellA_s1', 'free50'), ('cellA_s2', 'free50')]
        Dd = diffs(D, R, boots, pairs)
        Dd.to_csv(OUT / f'diffs_{arm}.csv', index=False)
        print(f'===== {arm}  ({D.line.nunique()} lines, clean rows x {int((D.clean & (D.view == 0)).sum())} '
              f'y {int((D.clean & (D.view == 1)).sum())}, runs {D.run.nunique()})')
        print(T.round(4).to_string())
        # full-data fits of every Poly model: coefficients + covariance; cells for the main ones
        coeffs[arm] = {}
        for k, mdl in models.items():
            if isinstance(mdl, Poly):
                mdl.fit(D[D.clean])
                coeffs[arm][k] = dict(spec={kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                            for kk, vv in mdl.spec.items()},
                                      names=mdl.names, beta=mdl.beta.tolist(),
                                      err=np.sqrt(np.diag(mdl.cov)).tolist(), sigma=mdl.sig, n_keep=mdl.nkeep,
                                      note='r_loc = g r (g per view, t6lib.G); u = x_local/200, v = y_local/200 (mm)')
                if not k.startswith('cellA'):
                    coeffs[arm][k]['cov'] = mdl.cov.tolist()
                for step in (50, 25):
                    e = np.arange(-200.0, 201.0, step)
                    cc = cells(mdl, D, e)
                    for vw in 'xy':
                        for q in ('a_loc', 'a_glob', 's'):
                            cz[f'{arm}_{k}_{vw}_{q}_e{step}'] = cc[vw][q]
            elif isinstance(mdl, Free) and not mdl.globalonly:
                mdl.fit(D[D.clean])
                for w in (0, 1):
                    cz[f'{arm}_{k}_{"xy"[w]}_a_glob_native'] = np.asarray(mdl.maps[w]['a'])
                    cz[f'{arm}_{k}_{"xy"[w]}_s_native'] = np.asarray(mdl.maps[w]['s'])
        json.dump(coeffs, open(OUT / 'coeffs.json', 'w'))
        np.savez(OUT / 'model_cells.npz', edges50=np.arange(-200.0, 201.0, 50), edges25=np.arange(-200.0, 201.0, 25),
                 **cz)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
