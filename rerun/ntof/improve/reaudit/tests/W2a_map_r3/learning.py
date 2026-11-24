#!/usr/bin/env python3
"""learning.py -- learning curve of the spline map (pot + per-view scale) and the per-track noise floor.

  curve A|C : train on a random fraction f of the clean lines of one run parity (lambdas re-chosen by 5-fold CV over
              runs inside that subset), test on ALL clean lines of the other parity (pooled over both directions);
              2 random draws per f.  -> lc_<arm>.csv
  noise     : A-C decomposition on the same lines, from task-1 held-out residuals (t1_resid_<arm>.npz, spl_pot_sep):
              var(r_A) = s_A^2 + s_j^2, var(r_C) = s_C^2 + s_j^2, var(r_A - r_C) = s_A^2 + s_C^2 (j cancels)
              -> noise.csv
"""
import sys

import numpy as np
import pandas as pd

from r3lib import Spline, HERE
from t6lib import rows_ac, mad

FR = (0.03, 0.06, 0.12, 0.25, 0.5, 0.75, 1.0)
LT = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
LS = np.array([1e-3, 1e-2, 0.1, 1.0])


def curve(arm):
    D = rows_ac(arm)
    rows = []
    for draw in (0, 1):
        rng = np.random.default_rng(100 + draw)
        for f in FR:
            if f == 1.0 and draw == 1:
                continue
            res = np.full(len(D), np.nan)
            ntr = 0
            for tr in (0, 1):
                trn = D[(D.part == tr) & D.clean]
                ul = trn.line.unique()
                keep = rng.choice(ul, max(int(round(f * len(ul))), 30), replace=False)
                sub = trn[trn.line.isin(keep)]
                ntr += len(keep)
                m = Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS,
                           kfold=min(5, sub.run.nunique())).fit(sub)
                tst = ((D.part != tr) & D.clean).to_numpy()
                res[tst] = m.corrected_resid(D[tst])
                rows.append(dict(arm=arm, f=f, draw=draw, train_part=tr, n_train_lines=len(keep),
                                 lam_t=m.lam_t_sel, lam_s=m.lam_s_sel))
            for w in (0, 1):
                vw = (D.view == w).to_numpy()
                rows.append(dict(arm=arm, f=f, draw=draw, view='xy'[w], n_train_lines_mean=ntr / 2,
                                 sig=mad(res[vw & np.isfinite(res)])))
            print(arm, f, draw, rows[-2:], flush=True)
    pd.DataFrame(rows).to_csv(HERE / f'lc_{arm}.csv', index=False)


def noise():
    A, C = rows_ac('A'), rows_ac('C')
    ra, rc = np.load(HERE / 't1_resid_A.npz'), np.load(HERE / 't1_resid_C.npz')
    out = []
    for k in ('none', 'free50', 'T6_cell33_s3', 'spl_pot_sep'):
        for w in (0, 1):
            m = ((A.view == w) & A.clean & C.clean).to_numpy() & np.isfinite(ra[k]) & np.isfinite(rc[k])
            sa, sc, sd = mad(ra[k][m]), mad(rc[k][m]), mad((ra[k] - rc[k])[m])
            sj2 = (sa ** 2 + sc ** 2 - sd ** 2) / 2
            out.append(dict(model=k, view='xy'[w], n=int(m.sum()), sig_rA=sa, sig_rC=sc, sig_diff=sd,
                            sig_j=float(np.sign(sj2) * np.sqrt(abs(sj2))),
                            sig_A=float(np.sqrt(max(sa ** 2 - sj2, 0))), sig_C=float(np.sqrt(max(sc ** 2 - sj2, 0))),
                            corr_rA_rC=float(np.corrcoef(ra[k][m], rc[k][m])[0, 1])))
    T = pd.DataFrame(out)
    T.to_csv(HERE / 'noise.csv', index=False)
    print(T.round(5).to_string())


if __name__ == '__main__':
    noise() if sys.argv[1] == 'noise' else curve(sys.argv[1])
