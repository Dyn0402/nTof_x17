#!/usr/bin/env python3
"""build_maps.py -- fit the r3 candidate maps on ALL clean lines and write them in the apply_r3 format.

    r3_lines.json        lines only (A, C: A-C lines; D: B-D lines), smoothing by 5-fold CV over runs
    r3_joint_all.json    lines + horizontal pointing cells of all runs (floor = t2_floor_<arm>.json x FLOOR_MULT)
    r3_joint_g0.json     lines + pointing of run group g0 only (for the beam check on g1), and r3_joint_g1.json
The candidate (angle_map_r3_candidate.json) is chosen from these in make_candidate.py.
"""
import json
import sys

import numpy as np

from r3lib import Spline, HERE, GRID, support_mask, LO, HI
from t6lib import rows_ac, rows_d
from task2 import point_rows, KAP, LT_AC, LS_AC, LT_D, LS_D

FLOOR_MULT = 1.0
FRAMES = {'A': 'x view -> d_x/d_z, y view -> d_y/d_z (global); local x_local = +x, y_local = +y',
          'C': 'x view -> d_x/d_z, y view -> d_y/d_z (global); local x_local = -x, y_local = +y',
          'D': 'x view -> d_z/d_x, y view -> d_y/d_x (global); local x_local = -z, y_local = +y'}
GS = {'A': [1, 1], 'C': [1, -1], 'D': [-1, 1]}


def arm_entry(arm, mdl, Dc, extra):
    tilt, scale = mdl.coef_arrays()
    X, Y = np.meshgrid(GRID, GRID, indexing='ij')
    F = mdl.fields(X.ravel(), Y.ravel())
    sup = support_mask(Dc.x.to_numpy(float), Dc.y.to_numpy(float))
    clamp = {k: [float(np.min(F[k][sup.ravel()])), float(np.max(F[k][sup.ravel()]))] for k in F}
    return dict(arm=arm, frame=FRAMES[arm], g=GS[arm],
                tilt_basis=dict(type='uniform cubic B-spline, tensor product', lo=LO, hi=HI, h=mdl.ht, nb=mdl.nt),
                scale_basis=dict(type='uniform cubic B-spline, tensor product', lo=LO, hi=HI, h=mdl.hs, nb=mdl.ns),
                tilt=tilt, scale=scale, clamp=clamp,
                support=dict(grid_mm=GRID.tolist(), mask=sup.astype(int).tolist(),
                             rule='10 mm grid point whose 20 mm cell holds >= 3 clean training lines'),
                fit=dict(lam_t=mdl.lam_t_sel, lam_s=mdl.lam_s_sel, n_rows_kept=mdl.nkeep,
                         sigma_fit={'xy'[k]: v for k, v in mdl.sig.items()},
                         n_lines=int(Dc.line.nunique()), **extra))


def fit_arm(arm, point_lab=None):
    D = rows_d() if arm == 'D' else rows_ac(arm)
    Dc = D[D.clean]
    LT, LS = (LT_D, LS_D) if arm == 'D' else (LT_AC, LS_AC)
    lines = Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS).fit(Dc)
    if point_lab is None:
        return arm_entry(arm, lines, Dc, dict(pointing=None))
    fl = json.load(open(HERE / f't2_floor_{arm}.json'))
    P = point_rows(arm, point_lab)
    pt = dict(P, floor=fl['floor'] * FLOOR_MULT, sig_line=np.mean(list(lines.sig.values())), kappas=KAP)
    jm = Spline(tilt='pot', scale='sep', lam_t=lines.lam_t_sel, lam_s=lines.lam_s_sel, point=pt).fit(Dc)
    return arm_entry(arm, jm, Dc, dict(pointing=dict(cells=f'pc_{arm}_{point_lab}.npz', n_cells=int(len(P['x'])),
                                                     floor=pt['floor'], kappa=float(jm.kappa),
                                                     alpha=float(jm.nuis[0]), gamma=float(jm.nuis[1]))))


META = dict(
    model='t = (1 + s_p) t_true + a_p in the LOCAL tan (stage-3 tanx/tany); corrected t\' = (t - a_p)/(1 + s_p). '
          'Equivalently on GLOBAL slopes S = g_p t: S\' = (S - g_p a_p)/(1 + s_p). Tilt a = grad(phi) (potential), '
          'phi a tensor cubic B-spline; per-view scale s_x, s_y tensor cubic B-splines.',
    coefficients='phi, s_x, s_y: nb x nb arrays indexed [i_x][i_y]; basis B_i(u) uniform cubic, knots lo + k h; '
                 'evaluate with apply_r3.tensor_eval (a_x = d phi/dx, a_y = d phi/dy, x = x_local, y = y_local in mm)',
    signs='g = local-tan -> global-slope sign per view (x, y): A (+1,+1), C (+1,-1), D (-1,+1), as t6lib.G and '
          'map_view.slope_sign',
    edges='positions clamped to [lo, hi]; fields then clamped to [min, max] over the supported region (clamp block); '
          'NaN positions: no correction',
    built_by='W2a_map_r3/build_maps.py (r3lib.Spline), training: lines.parquet / bd_lines.parquet clean lines '
             '(t6lib.rows_ac / rows_d)')


def main():
    for lab, pl in (('lines', None), ('joint_all', 'all'), ('joint_g0', 'g0'), ('joint_g1', 'g1')):
        maps = {arm: fit_arm(arm, pl) for arm in 'ACD'}
        json.dump(dict(meta=dict(META, variant=lab), maps=maps), open(HERE / f'r3_{lab}.json', 'w'))
        print(lab, {a: (m['fit']['lam_t'], m['fit']['lam_s'], m['fit'].get('pointing')) for a, m in maps.items()},
              flush=True)


if __name__ == '__main__':
    main()
