#!/usr/bin/env python3
"""collect.py -- GE: one row per (chamber, view, M3-pointed event) of the `fe` gate, with everything the chi2/dof cut
question needs.  -> ~/x17/cosmic_bench/model_audit/ge_chi2cut/bench_views.parquet

Inputs (the FE gate tree, `fe` arm, five golden rows, 8000 M3 muons each):
  events.parquet            the event-level fit per view (chi2, dof, q_sum, n_sat_strips, n_band_strips, n_tracks, ...)
  angles/per_event.npz      tan_ref (M3, rotated into the strip frame) and tan_fit for every fitted view with a ray
  budget/rays_<det>.parquet M3 pointing: region, position residual res_x/res_y, hit multiplicity, the 5 mm category
  events.splits.parquet     two-track split attempts (fstat, accepted)

Angle residual d_theta = theta_fit - theta_ref - median (per chamber/view), in degrees.  No quality cut anywhere.
"""
import os
import numpy as np
import pandas as pd

T = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/gate_tree')
OUT = os.path.expanduser('~/x17/cosmic_bench/model_audit/ge_chi2cut')
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
COLS = ['chi2', 'dof', 'q_sum', 'n_sat_strips', 'n_band_strips', 'n_strips', 'n_dropped', 'n_candidates',
        'isochronous', 'tan_err', 'p0_err', 'slope_reliable', 'quality_ok', 'n_flagged_strips', 'w', 'ftst']


def main(arm='fe'):
    os.makedirs(OUT, exist_ok=True)
    parts = []
    for det, row in ROWS.items():
        d = os.path.join(T, row, 'wft', arm)
        E = pd.read_parquet(os.path.join(d, 'events.parquet'))
        E = E[~E.event_id.duplicated(keep=False)].set_index('event_id')
        R = pd.read_parquet(os.path.join(d, 'budget', f'rays_{det}.parquet'))
        R = R[~R.event_id.duplicated(keep=False)].set_index('event_id')
        Z = np.load(os.path.join(d, 'angles', 'per_event.npz'))
        S = pd.read_parquet(os.path.join(d, 'events.splits.parquet'))
        for v in 'xy':
            eid = Z[f'{v}_eid']
            df = pd.DataFrame({'event_id': eid, 'tan_ref': Z[f'{v}_tan_ref'], 'tan_fit': Z[f'{v}_tan_fit']})
            df = df[df.event_id.isin(E.index) & df.event_id.isin(R.index)]
            e = E.loc[df.event_id]
            for c in COLS:
                df[c] = e[f'{v}_{c}'].to_numpy()
            for c in ('n_tracks', 'n_splits', 'spark', 'n_hits'):
                df[c] = e[c].to_numpy()
            # the other view's chi2/dof (a shower/second particle shows in both)
            o = 'y' if v == 'x' else 'x'
            df['chi2dof_other'] = (e[f'{o}_chi2'] / e[f'{o}_dof'].clip(lower=1)).to_numpy()
            df['other_ok'] = e[f'{o}_ok'].astype(bool).to_numpy()
            r = R.loc[df.event_id]
            df['region'] = r['region'].to_numpy()
            df['res'] = r[f'res_{v}'].to_numpy()
            df['res_other'] = r[f'res_{o}'].to_numpy()
            df['mult'] = r['mult'].to_numpy()
            df['cat5'] = r['cat_xy_5'].to_numpy()
            s = S[S.plane == v].groupby('event_id').agg(split_fstat=('fstat', 'max'), split_try=('fstat', 'size'),
                                                         split_acc=('accepted', 'max'))
            df = df.join(s, on='event_id')
            df['split_try'] = df['split_try'].fillna(0).astype(int)
            df['split_acc'] = df['split_acc'].astype('boolean').fillna(False).astype(bool)
            df['det'], df['view'] = det, v
            df['chi2dof'] = df.chi2 / df.dof.clip(lower=1)
            th = np.degrees(np.arctan(df.tan_fit) - np.arctan(df.tan_ref))
            df['dth'] = th - np.median(th)
            parts.append(df)
    A = pd.concat(parts, ignore_index=True)
    A.to_parquet(os.path.join(OUT, f'bench_views_{arm}.parquet'))
    print(A.groupby(['det', 'view']).size())


if __name__ == '__main__':
    import sys
    main(*(sys.argv[1:] or ['fe']))
