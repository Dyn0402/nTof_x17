#!/usr/bin/env python3
"""dveto_score.py --sets DIR --rays rays_<det>.parquet --runs DIR [--pass1 events.parquet] [--out json] [--label L]
-- score the variants of dveto_rereco.py against M3 (DISCHARGE_VETO_2026-11-02.md §3).

Truth is the M3 reference.  Per event of each set (dveto_events.py):
  track      the events-table winner fitted in both views (x_ok and y_ok): what the efficiency counts
  real       a track within R = 5 mm of the M3 ray at the mesh (raw strip frame, hypot of the per-view residuals, as
             rerun/effbudget/budget.py 'r')
  fake       a track in an M3-tagged event farther than 5 mm, or any track in a trigger with no M3 ray at all
  credit     any sane candidate within 5 mm in each view (budget 'det_any'), the Paper I multi-track credit
  extra      paired candidates (track_id >= 1) farther than 5 mm in an M3 event, or any paired candidate with no M3
Quality of real tracks: per-view residual s68 [mm] and |dtan| s68 against the M3 tangents (raw frame).
Identity: every column of the ordinary events must be equal between the variants (and to pass 1 if given).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [REPO]
R_MM = 5.0
TAN_SANE, P0_SANE = 3.0, (-5.0, 405.0)


def s68(x):
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 68.27)) if len(x) else np.nan


def score(E, C, sets, R):
    E = E.drop_duplicates('event_id').set_index('event_id')
    out = {}
    M = R.drop_duplicates('event_id').set_index('event_id')
    C = C[C['rank'] >= 0] if C is not None and len(C) else C
    for name, ids in sets.groupby('set').event_id:
        ids = ids.to_numpy()
        T = E.reindex(ids)
        has = (T.x_ok.fillna(False) & T.y_ok.fillna(False)).to_numpy(bool)
        m3 = name.endswith('_m3')
        o = dict(n=int(len(ids)), in_table=int(T.x_ok.notna().sum()), tracks=int(has.sum()))
        if 'spark_rescued' in T:
            o['rescued'] = int(T.spark_rescued.fillna(False).astype(bool).sum())
        if m3:
            mm = M.reindex(ids)
            rx, ry = T.x_p0.to_numpy() - mm.mesh_x.to_numpy(), T.y_p0.to_numpy() - mm.mesh_y.to_numpy()
            r = np.hypot(rx, ry)
            real = has & (np.nan_to_num(r, nan=1e9) <= R_MM)
            inter = (mm.region == 'interior').to_numpy()
            o.update(real=int(real.sum()), fake=int((has & ~real).sum()),
                     interior=int(inter.sum()), real_interior=int((real & inter).sum()),
                     fake_interior=int((has & ~real & inter).sum()))
            dtx = T.x_tan_theta.to_numpy() - mm.tan_x.to_numpy()
            dty = T.y_tan_theta.to_numpy() - mm.tan_y.to_numpy()
            o.update(res_s68_x=s68(rx[real]), res_s68_y=s68(ry[real]), dtan_s68_x=s68(dtx[real]), dtan_s68_y=s68(dty[real]),
                     res_med_x=float(np.nanmedian(rx[real])) if real.any() else np.nan,
                     dtan_med_x=float(np.nanmedian(dtx[real])) if real.any() else np.nan)
            if C is not None and len(C):
                Q = C[C.event_id.isin(ids)].join(mm[['mesh_x', 'mesh_y']], on='event_id')
                Q = Q[np.isfinite(Q.tan_theta) & (Q.tan_theta.abs() <= TAN_SANE) & Q.p0.between(*P0_SANE)]
                Q['ares'] = np.where(Q.plane == 'x', Q.p0 - Q.mesh_x, Q.p0 - Q.mesh_y).__abs__()
                near = Q.groupby(['event_id', 'plane']).ares.min().unstack()
                cr = near.reindex(ids)
                credit = ((cr.get('x', pd.Series(np.nan, index=ids)) <= R_MM) &
                          (cr.get('y', pd.Series(np.nan, index=ids)) <= R_MM)).to_numpy()
                o['credit'] = int(credit.sum())
                o['credit_interior'] = int((credit & inter).sum())
                P = C[C.event_id.isin(ids) & (C.track_id >= 1)]
                if len(P):
                    P = P.pivot_table(index=['event_id', 'track_id'], columns='plane', values='p0').reset_index()
                    P = P.join(mm[['mesh_x', 'mesh_y']], on='event_id')
                    far = np.hypot(P.get('x') - P.mesh_x, P.get('y') - P.mesh_y) > R_MM
                    o['extra_far'] = int(far.sum())
                else:
                    o['extra_far'] = 0
        else:
            o['fake'] = int(has.sum())
            if C is not None and len(C):
                P = C[C.event_id.isin(ids) & (C.track_id >= 1)]
                o['extra_far'] = int(P.drop_duplicates(['event_id', 'track_id']).shape[0])
        out[name] = o
    return out


def identity(A, B, ids):
    A = A.drop_duplicates('event_id').set_index('event_id')
    B = B.drop_duplicates('event_id').set_index('event_id')
    ids = sorted(set(ids) & set(A.index) & set(B.index))
    cols = [c for c in A.columns if c in B.columns and c not in ('timestamp', 'acq_tag', 'spark_rescued')]
    a, b = A.loc[ids, cols], B.loc[ids, cols]
    diff, maxd = {}, {}
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in 'fc':
            ne = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            ne = x != y
        if ne.any():
            diff[c] = int(ne.sum())
            if x.dtype.kind in 'fc':
                d = np.abs(x.astype(float) - y.astype(float))
                maxd[c] = float(np.nanmax(d)) if np.isfinite(d).any() else np.nan
    return dict(n=len(ids), n_cols=len(cols), columns_differing=diff, max_abs_diff=maxd,
                missing_in_b=len(set(A.index) & set(ids)) - len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', required=True)
    ap.add_argument('--rays', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--pass1', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    S = pd.read_parquet(os.path.join(a.sets, 'sets.parquet'))
    R = pd.read_parquet(a.rays)
    res = dict(label=a.label, n_sets=S.set.value_counts().to_dict(), variants={}, identity={})
    tabs = {}
    for v in sorted(os.listdir(a.runs)):
        p = os.path.join(a.runs, v, 'events.parquet')
        if not os.path.exists(p):
            continue
        E = pd.read_parquet(p)
        cp = p.replace('.parquet', '.candidates.parquet')
        C = pd.read_parquet(cp) if os.path.exists(cp) else None
        tabs[v] = E
        res['variants'][v] = score(E, C, S, R)
        meta = json.load(open(p.replace('.parquet', '.meta.json')))
        res['variants'][v]['_rescue'] = meta['selection'].get('spark_rescue')
    if a.pass1:
        # the pass-1 table (matched-only: M3-tagged events) on the same events, the baseline of what the pass recovers
        E1 = pd.read_parquet(a.pass1)
        cp = a.pass1.replace('.parquet', '.candidates.parquet')
        res['variants']['pass1'] = score(E1, pd.read_parquet(cp) if os.path.exists(cp) else None,
                                         S[S.set.str.endswith('_m3')], R)
    ordi = S[S.set.str.startswith('ord')].event_id.tolist()
    if 'off' in tabs:
        for v, E in tabs.items():
            if v != 'off':
                res['identity'][f'off_vs_{v}'] = identity(tabs['off'], E, ordi)
        if a.pass1:
            res['identity']['pass1_vs_off'] = identity(pd.read_parquet(a.pass1), tabs['off'],
                                                       S[S.set == 'ord_m3'].event_id.tolist())
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        open(a.out, 'w').write(txt)
    for v, d in res['variants'].items():
        print(f'== {a.label} {v}  rescue {d.get("_rescue")}')
        for k in ('veto_m3', 'veto_nom3', 'ord_m3', 'ord_nom3'):
            if k in d:
                print(f'   {k:10s} ' + ' '.join(f'{x}={y:.3g}' if isinstance(y, float) else f'{x}={y}' for x, y in d[k].items()))
    for k, d in res['identity'].items():
        print('identity', k, d)


if __name__ == '__main__':
    main()
