#!/usr/bin/env python3
"""task1.py -- held-out comparison of spline maps against no map, the current free 50 mm cells and T6's best, on the
clean A-C cosmic lines.  Outer protocol = T6's (train on clean lines of one run parity, test on the other, pooled, every
line tested once); the spline smoothing is chosen INSIDE the training half by 5-fold CV over runs.

    task1.py A|C   -> t1_models_<arm>.csv, t1_diffs_<arm>.csv, t1_resid_<arm>.npz, t1_lam_<arm>.csv
"""
import json
import sys

import numpy as np
import pandas as pd

from r3lib import Spline, HERE
from t6lib import rows_ac, Poly, Free, NoMap, mad, widths
from q34_models import diffs

LT = np.array([0.003, 0.01, 0.03, 0.1])
LS = np.array([1e-3, 3e-3, 1e-2, 0.1, 1.0])
REPO_MAP = '/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/stage3/angle_map/angle_map_r2.json'
rng = np.random.default_rng(5)


class R2File:
    """the shipped r2 map (trained on ALL lines: in-sample for every test line)."""

    def __init__(self, arm):
        self.maps = json.load(open(REPO_MAP))['maps']; self.arm = arm

    def fit(self, D):
        return self

    def corrected_resid(self, D):
        from rerun.ntof.stage3.map_view import interp
        out = np.full(len(D), np.nan)
        for w, p in ((0, 'x'), (1, 'y')):
            m = (D.view == w).to_numpy()
            a, s = interp(self.maps[f'{self.arm}_{p}'], D.x.to_numpy()[m], D.y.to_numpy()[m])
            out[m] = (D.slope.to_numpy()[m] - a) / (1 + s) - D.j.to_numpy()[m]
        return out


def models(arm):
    E = lambda s: np.linspace(-200, 200, int(round(400 / s)) + 1)  # noqa: E731
    sp = lambda **k: Spline(ht=20.0, hs=40.0, lam_grid_t=LT, lam_grid_s=LS, **k)  # noqa: E731
    return {'none': NoMap(), 'r2_file_insample': R2File(arm), 'free50': Free(),
            'T6_pot10_s3': Poly(n=10, m=3), 'T6_cell33_s3': Poly(cellA=E(400 / 12), m=3, ridge=1e-4),
            'spl_pot_sep': sp(tilt='pot', scale='sep'), 'spl_pot_common': sp(tilt='pot', scale='common'),
            'spl_free_sep': sp(tilt='free', scale='sep'), 'spl_free_common': sp(tilt='free', scale='common')}


def metrics(D, R):
    c = D.clean.to_numpy()
    edge = ((D.x.abs() > 125) | (D.y.abs() > 125)).to_numpy()
    runs = D.run.to_numpy(); ur = np.unique(runs)
    rs = {r: np.flatnonzero(runs == r) for r in ur}
    boots = [np.concatenate([rs[r] for r in rng.choice(ur, len(ur))]) for _ in range(300)]
    rows = []
    for w in (0, 1):
        vw = (D.view == w).to_numpy()
        for k, e in R.items():
            sel = vw & c & np.isfinite(e)
            row = dict(view='xy'[w], model=k, n_clean=int(sel.sum()), sig_clean=mad(e[sel]),
                       sig_clean_err=float(np.std([mad(e[b][sel[b]]) for b in boots[:150]])))
            row['w68'], row['w95'] = widths(e[sel])
            for lab, rm in (('edge', edge), ('int', ~edge)):
                s2 = sel & rm
                row[f'sig_{lab}'] = mad(e[s2]); row[f'w68_{lab}'], row[f'w95_{lab}'] = widths(e[s2])
                row[f'n_{lab}'] = int(s2.sum())
            a = vw & np.isfinite(e)
            row['sig_all'] = mad(e[a]); row['tail_all'] = float(np.mean(np.abs(e[a]) > 0.15))
            rows.append(row)
    return pd.DataFrame(rows), boots


def main(arm):
    D = rows_ac(arm)
    M = models(arm)
    R = {k: np.full(len(D), np.nan) for k in M}
    lam = []
    for tr in (0, 1):
        trn = D[(D.part == tr) & D.clean]
        tst = (D.part != tr).to_numpy() & D.allok.to_numpy()
        for k, mdl in M.items():
            mdl.fit(trn)
            R[k][tst] = mdl.corrected_resid(D[tst])
            if isinstance(mdl, Spline):
                lam.append(dict(model=k, train_part=tr, lam_t=mdl.lam_t_sel, lam_s=mdl.lam_s_sel,
                                cv_min=float(mdl.cvtab.cv.min())))
                mdl.cvtab.assign(model=k, train_part=tr).to_csv(HERE / f't1_cv_{arm}_{k}_{tr}.csv', index=False)
            print(arm, tr, k, flush=True)
    T, boots = metrics(D, R)
    T['arm'] = arm
    T.to_csv(HERE / f't1_models_{arm}.csv', index=False)
    pd.DataFrame(lam).to_csv(HERE / f't1_lam_{arm}.csv', index=False)
    np.savez(HERE / f't1_resid_{arm}.npz', **R)
    pairs = [(k, 'free50') for k in M if k not in ('free50',)] + \
            [(k, 'T6_cell33_s3') for k in M if k.startswith('spl')] + \
            [('spl_pot_sep', 'spl_free_sep'), ('spl_pot_common', 'spl_free_common'),
             ('spl_pot_sep', 'spl_pot_common'), ('spl_free_sep', 'spl_free_common')]
    diffs(D, R, boots, pairs).to_csv(HERE / f't1_diffs_{arm}.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.drop(columns=['arm']).round(4).to_string())


if __name__ == '__main__':
    main(sys.argv[1])
