#!/usr/bin/env python3
"""d_event.py -- D held-out with T6's EVENT-parity split (bd_map.py protocol), for comparison with T6's D numbers;
the run-parity split is in t2_models_D.csv.  -> d_event.csv"""
import numpy as np, pandas as pd
from r3lib import Spline, HERE
from t6lib import rows_d, NoMap, Free, mad, widths
from task2 import LT_D, LS_D
from task1 import R2File
D = rows_d()           # part = event parity
R = {k: np.full(len(D), np.nan) for k in ('none', 'free100', 'r2_file_insample', 'spl_pot_sep')}
for tr in (0, 1):
    trn = D[(D.part == tr) & D.clean]; tst = (D.part != tr).to_numpy() & D.allok.to_numpy()
    for k, m in (('none', NoMap()), ('free100', Free(edges=np.arange(-200.0, 201, 100), n_min=60)),
                 ('r2_file_insample', R2File('D')),
                 ('spl_pot_sep', Spline(tilt='pot', scale='sep', lam_grid_t=LT_D, lam_grid_s=LS_D))):
        m.fit(trn); R[k][tst] = m.corrected_resid(D[tst])
rows = []
edge = ((D.x.abs() > 125) | (D.y.abs() > 125)).to_numpy()
for w in (0, 1):
    for k, e in R.items():
        s = (D.view == w).to_numpy() & D.clean.to_numpy() & np.isfinite(e)
        w68, w95 = widths(e[s])
        rows.append(dict(view='xy'[w], model=k, n=int(s.sum()), sig=mad(e[s]), w68=w68, w95=w95,
                         sig_edge=mad(e[s & edge]), sig_int=mad(e[s & ~edge])))
T = pd.DataFrame(rows); T.to_csv(HERE / 'd_event.csv', index=False); print(T.round(4).to_string())
