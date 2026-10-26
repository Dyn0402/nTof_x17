#!/usr/bin/env python3
"""summarize.py -- every number the det3 loss-category note (make_deck.py) shows, from the outputs of this package and
rerun/effbudget + rerun/outliers, into ~/x17/cosmic_bench/satstudy/summary.json. Parts whose input run has not
finished are written as null (the deck then says "pending").

Detection against M3 is the efficiency budget's (rerun/effbudget/budget.py): both views fitted, |r| <= 5 mm in the raw
strip frame, n_dropped <= 2 in both views. Per view "agrees" is the C22 outlier rule: |dθ| <= 5° (ladder-scaled) and
|p0 - ref_mesh| <= 3 mm.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
from olib import load  # noqa: E402

EFF = os.path.expanduser('~/x17/cosmic_bench/effbudget')
OLB = os.path.expanduser('~/x17/cosmic_bench/outliers')


def rays():
    R = pd.read_parquet(os.path.join(EFF, 'rays_det3.parquet'))
    return R[(R.region == 'interior') & ~R.not_measured].set_index('event_id')


def lad():
    return load(S.DET).attrs['ladder']


def view_agree(p0, w, mesh, tan, L):
    s, c = L
    tl = (np.asarray(w) * 1e3 - c) / s
    dl = np.degrees(np.arctan(tl) - np.arctan(tan))
    return (np.abs(dl) <= 5) & (np.abs(np.asarray(p0) - mesh) <= 3), dl, tl


def detected(E, R):
    """Per event of a reco table E (events.parquet), the budget's detection against the rays R."""
    E = E.drop_duplicates('event_id', keep=False).set_index('event_id')
    j = E.join(R[['mesh_x', 'mesh_y']], how='inner')
    ok = j.x_ok.fillna(False).astype(bool) & j.y_ok.fillna(False).astype(bool)
    r = np.hypot(j.x_p0 - j.mesh_x, j.y_p0 - j.mesh_y)
    nd = (j.x_n_dropped <= 2) & (j.y_n_dropped <= 2)
    return pd.DataFrame(dict(fitted=ok, near=ok & (r <= 5), detected=ok & (r <= 5) & nd, r=r))


def budget():
    B = json.load(open(os.path.join(EFF, 'budget_det3.json')))
    return dict(interior=B['budget']['interior|xy|5'], n_rays=B['n_rays_recipe'], table=B['table'])


def sat_dependence():
    d = load(S.DET); out = {}
    for v in 'xy':
        u = d[d[f'use_{v}']]
        far = (u[f'dl_{v}'].abs() > 5) | (u[f'rres_{v}'].abs() > 3)
        ns = u[f'h{v}_nsat'].fillna(0)
        clean = u[f'h{v}_ncl2'].fillna(0) <= 1
        bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 99)]
        rows = []
        for a, b in bins:
            m = (ns >= a) & (ns <= b)
            for nm, mm in (('clean', m & clean), ('all', m)):
                rows.append(dict(lo=a, hi=b, sel=nm, n=int(mm.sum()), k=int(far[mm].sum()),
                                 far=float(far[mm].mean()) if mm.any() else None))
        s = u[(ns >= 3) & far]
        out[v] = dict(bins=rows, frac_events_nsat3=float((ns >= 3).mean()),
                      too_steep=float((s[f'tl_{v}'].abs() > s[f'ref_tan_{v}'].abs()).mean()), n_satfar=int(len(s)),
                      dtan_med=float((s[f'tl_{v}'].abs() - s[f'ref_tan_{v}'].abs()).median()),
                      chi2dof=dict(sat_far=float(s[f'chi2dof_{v}'].median()),
                                   sat_ok=float(u[(ns >= 3) & ~far][f'chi2dof_{v}'].median()),
                                   unsat=float(u[ns == 0][f'chi2dof_{v}'].median())))
    return out


def arms():
    p = os.path.join(S.OUT, 'fits_det3.parquet')
    if not os.path.exists(p):
        return None
    F = pd.read_parquet(p); R = rays(); L = lad(); d = load(S.DET).set_index('event_id')
    F = F[F.event_id.isin(d.index)]
    ref = d.loc[F.event_id]
    ag, dl, tl = [], [], []
    for v in 'xy':
        m = (F.view == v).to_numpy()
        a, b, c = view_agree(F.p0[m], F.w[m], ref[f'ref_mesh_{v}'].to_numpy()[m], ref[f'ref_tan_{v}'].to_numpy()[m], L[v])
        F.loc[m, 'agree'] = a; F.loc[m, 'dl'] = b; F.loc[m, 'tl'] = c
        F.loc[m, 'res'] = F.p0[m].to_numpy() - ref[f'ref_mesh_{v}'].to_numpy()[m]
    F['agree'] = F.agree.fillna(False).astype(bool) & F.ok
    F['chi2dof'] = F.chi2 / F.dof.clip(lower=1)
    out = {}
    for (g, arm), h in F.groupby(['group', 'arm']):
        out[f'{g}|{arm}'] = dict(n=int(len(h)), agree=float(h.agree.mean()), ok=float(h.ok.mean()),
                                 far=float(1 - h.agree.mean()), chi2dof=float(h.chi2dof.median()),
                                 n_cens=float(h.n_cens.median()))
    # per event: both views agree, interior rays only (the budget's denominator)
    P = F.pivot_table(index=['event_id', 'arm'], columns='view', values='agree', aggfunc='first').reset_index()
    P = P[P.event_id.isin(R.index)]
    P['both'] = P.get('x', False).fillna(False).astype(bool) & P.get('y', False).fillna(False).astype(bool)
    sat = set(pickle_ids('sat'))
    for arm, h in P[P.event_id.isin(sat)].groupby('arm'):
        out[f'event|{arm}'] = dict(n=int(len(h)), both=float(h.both.mean()))
    # switch matrix prod -> best arm, per view, on the saturated group
    piv = F[F.group == 'sat'].pivot_table(index=['event_id', 'view'], columns='arm', values='agree', aggfunc='first')
    for arm in piv.columns:
        if arm == 'prod':
            continue
        out[f'switch|{arm}'] = dict(gained=int((~piv['prod'].astype(bool) & piv[arm].astype(bool)).sum()),
                                    lost=int((piv['prod'].astype(bool) & ~piv[arm].astype(bool)).sum()),
                                    n=int(len(piv)))
    F.to_parquet(os.path.join(S.OUT, 'fits_det3_scored.parquet'))
    return out


def pickle_ids(k):
    import pickle
    return pickle.load(open(os.path.join(S.OUT, 'wf_det3.pkl'), 'rb'))[k]


def sparks():
    T = pd.read_parquet(os.path.join(S.OUT, 'sparks_det3.parquet'))
    out = dict(n=int(len(T)), by_cls=T.cls.value_counts().to_dict(),
               med=T.groupby('cls')[['mult', 'x_non', 'y_non', 'x_nsatstrip', 'y_nsatstrip', 'x_nneg', 'y_nneg']]
               .median().to_dict('index'))
    p = os.path.join(S.OUT, 'reco_nospark', 'events.parquet')
    if os.path.exists(p):
        D = detected(pd.read_parquet(p), rays()).join(T.set_index('event_id').cls, how='right')
        D = D.fillna({'fitted': False, 'near': False, 'detected': False})
        out['nospark'] = {c: dict(n=int(len(g)), fitted=int(g.fitted.sum()), near=int(g.near.sum()),
                                  detected=int(g.detected.sum())) for c, g in D.groupby('cls')}
        # candidate level: does the chosen fit / any candidate agree with M3 in BOTH views (|dθ| <= 5°, |res| <= 3 mm)
        C = pd.read_parquet(p.replace('.parquet', '.candidates.parquet')); R = rays(); L = lad()
        E = pd.read_parquet(p).set_index('event_id')
        for e, c in T.set_index('event_id').cls.items():
            r = R.loc[e]; ok = {}
            for v in 'xy':
                g = C[(C.event_id == e) & (C.plane == v)]
                a = view_agree(g.p0, g.w, r[f'mesh_{v}'], r[f'tan_{v}'], L[v])[0] if len(g) else np.array([False])
                ok[v] = bool(np.any(a)); ok[v + 'w'] = bool(np.any(np.asarray(a)[(g['rank'] == 0).to_numpy()])) if len(g) else False
            q = out['nospark'][c]
            q['any_both'] = q.get('any_both', 0) + int(ok['x'] and ok['y'])
            q['win_both'] = q.get('win_both', 0) + int(ok['xw'] and ok['yw'])
            if ok['xw'] and ok['yw'] and e in E.index:
                q.setdefault('nd_win', []).append(int(max(E.loc[e, 'x_n_dropped'], E.loc[e, 'y_n_dropped'])))
    return out


def fit_failed():
    import pickle
    E = pickle.load(open(os.path.join(S.OUT, 'wf_examples.pkl'), 'rb'))
    ids = E['picks']['fit_failed']; R = rays()
    out = dict(n=len(ids))
    p2, p3 = (os.path.join(S.OUT, f'reco_min{k}', 'events.parquet') for k in (2, 3))
    if os.path.exists(p2) and os.path.exists(p3):
        A, B = pd.read_parquet(p2), pd.read_parquet(p3)
        DA, DB = detected(A, R), detected(B, R)
        f = [i for i in ids if i in DA.index]
        out['min2'] = dict(detected=int(DA.loc[f].detected.sum()), near=int(DA.loc[f].near.sum()), n=len(f))
        out['min3'] = dict(detected=int(DB.reindex(f).detected.fillna(False).sum()))
        ctl = sorted(set(DA.index) & set(DB.index) - set(ids))
        a = A.drop_duplicates('event_id').set_index('event_id').loc[ctl]
        b = B.drop_duplicates('event_id').set_index('event_id').loc[ctl]
        moved = sum(int(((a[f'{v}_p0'] - b[f'{v}_p0']).abs() > 0.05).sum()) for v in 'xy')
        out['controls'] = dict(n=len(ctl), det_min2=int(DA.loc[ctl].detected.sum()), det_min3=int(DB.loc[ctl].detected.sum()),
                               views_moved=moved)
    return out


def multi():
    T = pd.read_parquet(os.path.join(S.OUT, 'multi_det3.parquet'))
    B = T[~T.win_agrees & T.any_agrees]
    tw = json.load(open(os.path.join(OLB, 'report', 'twotrack_det3.json')))
    row = {r['cls']: r for r in tw['off']}['MULTI']
    return dict(n_views=int(len(T)), n_wrong=int((~T.win_agrees).sum()),
                frac_cand=float(T[~T.win_agrees].any_agrees.mean()),
                sep_q={k: float(v) for k, v in B.d_win.abs().quantile([.1, .25, .5, .75, .9]).items()},
                sep=B.d_win.abs().tolist(), dt0=B.dt0.tolist(),
                q_ratio_med=float(B.q_ratio.median()), rules={k: row[k] for k in row if k.startswith('rule_') or k in ('any', 'finalize')})


def delta():
    X = pd.read_parquet(os.path.join(OLB, 'classified_det3.parquet'))
    o = X[X.outlier & (X.cls == 'DELTA')]; c = X[~X.outlier]
    w = lambda t: np.nanmax(np.c_[t.wx_wid1, t.wy_wid1], axis=1)
    tw = json.load(open(os.path.join(OLB, 'report', 'twotrack_det3.json')))
    row = {r['cls']: r for r in tw['on']}['DELTA']
    return dict(width_delta=w(o).tolist(), width_ctrl=w(c).tolist(), twotrack_any=row['any'], finalize=row['finalize'])


def main():
    S_ = dict(budget=budget(), sat=sat_dependence(), rail=json.load(open(os.path.join(S.OUT, 'railstats.json'))),
              arms=arms(), sparks=sparks(), fit_failed=fit_failed(), multi=multi(), delta=delta())
    for k in ('multitrack', 'headon'):     # multitrack.py count, headon.py
        p = os.path.join(S.OUT, f'{k}.json')
        S_[k] = json.load(open(p)) if os.path.exists(p) else None
    json.dump(S_, open(os.path.join(S.OUT, 'summary.json'), 'w'), indent=1, default=float)
    for k in ('arms', 'sparks', 'fit_failed'):
        print(k, json.dumps(S_[k], default=float)[:1500])


if __name__ == '__main__':
    main()
