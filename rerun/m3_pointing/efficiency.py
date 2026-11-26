#!/usr/bin/env python3
"""efficiency.py -- match each reference track within n of ITS OWN expected residual width, against the fixed 5 mm.

Denominator: Paper I's (interior_good rays of the recipe, decoder-complete).  Numerator, the any-candidate rule: in each
view the candidate nearest the reference ray (effbudget near_res_x/y).
  fixed R       hypot(res_x, res_y) <= R                                (Paper I: R = 5 mm)
  per-track n   d^2 = sum_v ((res_v - bias_v) / sigma_v,i)^2 <= n^2     (sigma from validate.py: chamber + M3 pointing)
  const n       the same with the Paper I constant pointing (sigma0)
Accidentals by event mixing, as Paper I: the production fit of another event (shift s = 1..20) against this ray, scaled
by the fraction of events with a fit.  Writes $WORK/efficiency/{curves.parquet, summary.json}.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3common import WORK  # noqa: E402
from rows import ROWS  # noqa: E402

OUT = os.path.join(WORK, 'efficiency')
R_GRID = np.round(np.arange(0.5, 15.01, 0.25), 3)
N_GRID = np.round(np.arange(1.0, 12.01, 0.25), 3)
SHIFTS = range(1, 21)
GROUPS = {'A (Runs 1+2)': ['A_run1', 'A_run2'], 'A 6-22': ['A_622'], 'B 6-22': ['B_622'], 'C 6-26': ['C_626'],
          'D 6-26': ['D_626'], 'E 6-24': ['E_624'], 'E 6-23': ['E_623'], 'A 6-23': ['A_623']}


def load(tags):
    D = pd.concat([pd.read_parquet(os.path.join(WORK, 'validate', f'sigma_{t}.parquet')) for t in tags],
                  ignore_index=True)
    return D[~D.not_measured].copy()


def dist(D, rx, ry, kind):
    if kind == 'R':
        return np.hypot(rx, ry)
    s = '' if kind == 'n' else '0'
    bx, by = D[f'bias{s}_x'].to_numpy(), D[f'bias{s}_y'].to_numpy()
    return np.hypot((rx - bx) / D[f'sig{s}_x'].to_numpy(), (ry - by) / D[f'sig{s}_y'].to_numpy())


def curves(D):
    out = {}
    rx, ry = D.near_res_x.to_numpy(), D.near_res_y.to_numpy()
    f_fit = float(np.mean(D.x_ok & D.y_ok))
    for kind, grid in (('R', R_GRID), ('n', N_GRID), ('n0', N_GRID)):
        d = np.nan_to_num(dist(D, rx, ry, kind), nan=1e9)
        eff = np.array([np.mean(d <= g) for g in grid])
        # mixing within each row (the event order of the row): fit of event i+s against the ray of event i
        acc = np.zeros(len(grid)); nmix = 0
        for _, S in D.groupby('tag'):
            S = S.sort_values('event_id')
            px, py = S.x_p0.to_numpy(), S.y_p0.to_numpy(); ok = (S.x_ok & S.y_ok).to_numpy()
            for s in SHIFTS:
                j = np.roll(np.arange(len(S)), -s)
                mx, my = px[j] - S.mesh_x.to_numpy(), py[j] - S.mesh_y.to_numpy()
                dm = np.nan_to_num(dist(S, mx, my, kind), nan=1e9)
                dm[~ok[j]] = 1e9
                acc += np.array([np.sum(dm <= g) for g in grid]); nmix += len(S)
        out[kind] = dict(grid=grid, eff=eff, acc=acc / nmix)
    out['f_fit'] = f_fit
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, summ = [], {}
    for name, tags in GROUPS.items():
        D = load(tags)
        c = curves(D)
        for kind in ('R', 'n', 'n0'):
            for g, e, a in zip(c[kind]['grid'], c[kind]['eff'], c[kind]['acc']):
                rows.append(dict(group=name, kind=kind, cut=g, eff=100 * e, acc=100 * a, n=len(D)))
        C = pd.DataFrame([r for r in rows if r['group'] == name])
        R5 = C[(C.kind == 'R') & (C.cut == 5.0)].iloc[0]
        s = dict(n=len(D), eff_R5=R5.eff, acc_R5=R5.acc)
        for kind in ('n', 'n0'):
            K = C[C.kind == kind]
            for n in (3, 4, 5, 6):
                k = K[K.cut == n].iloc[0]; s[f'eff_{kind}{n}'] = k.eff; s[f'acc_{kind}{n}'] = k.acc
            # the cut with the same accidental rate as R = 5 mm, and its efficiency
            j = int(np.argmin(np.abs(K.acc.to_numpy() - R5.acc)))
            s[f'{kind}_at_accR5'] = float(K.cut.iloc[j]); s[f'eff_{kind}_at_accR5'] = float(K.eff.iloc[j])
        # the tracks the two rules disagree on (n = 5 vs R = 5 mm)
        dR = np.nan_to_num(dist(D, D.near_res_x.to_numpy(), D.near_res_y.to_numpy(), 'R'), nan=1e9) <= 5
        dn = np.nan_to_num(dist(D, D.near_res_x.to_numpy(), D.near_res_y.to_numpy(), 'n'), nan=1e9) <= 5
        s['gained_n5'] = int(np.sum(dn & ~dR)); s['lost_n5'] = int(np.sum(dR & ~dn))
        s['gained_chi2_median'] = float(np.median(np.maximum(D.chi2_X, D.chi2_Y)[dn & ~dR])) if s['gained_n5'] else None
        summ[name] = s
        print(f"{name:13s} n {len(D):6d}  R5 {R5.eff:5.2f} % (acc {R5.acc:.3f} %) | per-track n=3 {s['eff_n3']:5.2f} "
              f"n=4 {s['eff_n4']:5.2f} n=5 {s['eff_n5']:5.2f} (acc {s['acc_n5']:.3f} %) | at R5's acc: n={s['n_at_accR5']}"
              f" {s['eff_n_at_accR5']:5.2f} %, const n={s['n0_at_accR5']} {s['eff_n0_at_accR5']:5.2f} % | n5 vs R5: "
              f"+{s['gained_n5']} -{s['lost_n5']}", flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, 'curves.parquet'))
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
