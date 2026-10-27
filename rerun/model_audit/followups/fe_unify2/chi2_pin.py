#!/usr/bin/env python3
"""chi2_pin.py [arm ...] -- FE item 7: what the t0 pin does to the fit quality figures that gate decisions read.

Per golden row and view, on the events both the gate3 'unified' table (free t0) and the FE arm reconstructed with both
views: median chi2/dof (ratio), the fraction failing the quality cut (chi2/dof >= CHI2DOF_BAD = 300), the two-track split
rate, and the fraction of fits whose t0 sits at the pin (|t0 - (C - 10 ftst)| < 1 ns).  -> results/chi2_pin.json/.md
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.expanduser('~/x17/cosmic_bench/model_audit')
T = os.path.join(M, 'fe_unify2', 'gate_tree')
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}


def main():
    arms = sys.argv[1:] or ['fe1', 'fe']
    out, L = {}, ['| chamber | arm | view | chi2/dof median unified -> arm (ratio) | quality-cut fail unified -> arm | '
                  'splits unified -> arm | t0 at the pin |', '|---|---|---|---|---|---|---|']
    for det, row in ROWS.items():
        U = pd.read_parquet(os.path.join(M, 'gate_full', det, 'unified', 'events.parquet')).drop_duplicates('event_id').set_index('event_id')
        for arm in arms:
            p = os.path.join(T, row, 'wft', arm, 'events.parquet')
            if not os.path.exists(p):
                continue
            F = pd.read_parquet(p).drop_duplicates('event_id').set_index('event_id')
            b = json.load(open(os.path.join(T, row, 'wft', f'calib_bundle_{arm}', 'bundle.json')))
            ids = U.index.intersection(F.index)
            u, f = U.loc[ids], F.loc[ids]
            for v in 'xy':
                k = u[f'{v}_ok'].astype(bool) & f[f'{v}_ok'].astype(bool)
                cu = (u[f'{v}_chi2'] / u[f'{v}_dof'].clip(lower=1))[k]; cf = (f[f'{v}_chi2'] / f[f'{v}_dof'].clip(lower=1))[k]
                C = float(b['t0_abs'][v]['0'])
                pin = (np.abs(f[f'{v}_t0'] - (C - 10 * f[f'{v}_ftst'])) < 1.0)[k & (f[f'{v}_ftst'] >= 0)]
                r = dict(n=int(k.sum()), chi2dof_unified=float(cu.median()), chi2dof_arm=float(cf.median()),
                         ratio=float(cf.median() / cu.median()), bad_unified=float((cu >= 300).mean()),
                         bad_arm=float((cf >= 300).mean()), pinned=float(pin.mean()),
                         splits_unified=int(u.get('n_splits', pd.Series(0)).sum()), splits_arm=int(f.get('n_splits', pd.Series(0)).sum()))
                out[f'{det}|{arm}|{v}'] = r
                L.append(f'| {det} | {arm} | {v} | {r["chi2dof_unified"]:.1f} -> {r["chi2dof_arm"]:.1f} ({r["ratio"]:.2f}) | '
                         f'{100 * r["bad_unified"]:.2f} -> {100 * r["bad_arm"]:.2f} % | {r["splits_unified"]} -> {r["splits_arm"]} | '
                         f'{100 * r["pinned"]:.0f} % |')
    md = '\n'.join(L) + '\n'; print(md)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'chi2_pin.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'chi2_pin.md'), 'w').write(md)


if __name__ == '__main__':
    main()
