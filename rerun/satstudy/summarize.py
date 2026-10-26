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


def _pair_runs(a, b, ids_csv, n_sig):
    """Production (a; None = the production table itself) vs a variant (b) on ids_csv: the first n_sig ids are the
    signal set, the rest controls (which must come back unchanged for the comparison to hold)."""
    pa = EV_PROD if a is None else os.path.join(S.OUT, a, 'events.parquet')
    pb = os.path.join(S.OUT, b, 'events.parquet')
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    R = rays(); ids = pd.read_csv(os.path.join(S.OUT, ids_csv)).event_id.astype(int).tolist()
    sig, ctl = ids[:n_sig], ids[n_sig:]
    A, B = pd.read_parquet(pa), pd.read_parquet(pb)
    DA, DB = detected(A, R), detected(B, R)
    out = dict(n=len(sig), det_prod=int(DA.reindex(sig).detected.fillna(False).sum()),
               det_var=int(DB.reindex(sig).detected.fillna(False).sum()),
               near_var=int(DB.reindex(sig).near.fillna(False).sum()))
    c = sorted(set(ctl) & set(DA.index) & set(DB.index))
    a = A.drop_duplicates('event_id').set_index('event_id').loc[c]; b_ = B.drop_duplicates('event_id').set_index('event_id').loc[c]
    out['controls'] = dict(n=len(c), det_prod=int(DA.loc[c].detected.sum()), det_var=int(DB.loc[c].detected.sum()),
                           views_moved=sum(int(((a[f'{v}_p0'] - b_[f'{v}_p0']).abs() > 0.05).sum()) for v in 'xy'))
    return out


def band_fix():
    """WFT_DROP_FLAT_HITS=1 (wft.reco) on every interior band event (bands.py) + 400 controls, against production."""
    n = len(pd.read_csv(os.path.join(S.OUT, 'ids_band.csv'))) - 400
    r = _pair_runs(None, 'reco_band_flat', 'ids_band.csv', n)
    if r:
        R = rays(); ids = pd.read_csv(os.path.join(S.OUT, 'ids_band.csv')).event_id.astype(int).tolist()[:n]
        r['prod_cats'] = R.reindex(ids).cat_xy_5.value_counts().to_dict()
        B = detected(pd.read_parquet(os.path.join(S.OUT, 'reco_band_flat', 'events.parquet')), R)
        r['recovered_by_cat'] = {c: int(B.reindex(g.index).detected.fillna(False).sum())
                                 for c, g in R.reindex(ids).groupby('cat_xy_5')}
    return r


def headon_wft():
    """WFT_HEADON_SIG=20 + WFT_HEADON_RESCUE=1 (wft) on the head-on failures + 400 controls and on the 742 triggers whose
    M3 track misses det3 (fakes), against production."""
    r = _pair_runs(None, 'reco_ho_wft', 'ids_minstrips.csv', 17)
    p, q = (os.path.join(S.OUT, x, 'events.parquet') for x in ('reco_out_min3', 'reco_out_ho_wft'))
    if r and os.path.exists(q):
        A, B = pd.read_parquet(p), pd.read_parquet(q)
        r['outside'] = dict(n=len(pd.read_csv(os.path.join(S.OUT, 'ids_outside.csv'))),
                            views_prod=int(A.x_ok.sum() + A.y_ok.sum()), views_new=int(B.x_ok.sum() + B.y_ok.sum()),
                            both_prod=int((A.x_ok & A.y_ok).sum()), both_new=int((B.x_ok & B.y_ok).sum()))
    return r


def two_track():
    """The joint two-track fit (WFT_TWO_TRACK_FIT=1; reco_tt default, reco_ttg scale=two + grid + all candidates) on the
    multi-particle events (multi_sub.parquet) + 400 controls: any candidate on M3 in both views (multitrack's 'any')."""
    T = pd.read_parquet(os.path.join(S.OUT, 'multi_sub.parquet')); R = rays(); out = {}
    ids = pd.read_csv(os.path.join(S.OUT, 'ids_multi_s.csv')).event_id.astype(int).tolist(); ctl = ids[len(T):]
    for arm in ('reco_tt', 'reco_ttg'):
        p = os.path.join(S.OUT, arm, 'events.candidates.parquet')
        if not os.path.exists(p):
            continue
        C = pd.read_parquet(p); C = C[C['rank'] >= 0]
        C['mesh'] = np.where(C.plane == 'x', R.mesh_x.reindex(C.event_id).to_numpy(), R.mesh_y.reindex(C.event_id).to_numpy())
        near = C[(C.p0 - C.mesh).abs() <= 5]
        anyb = near.groupby('event_id').plane.nunique() == 2
        r = {k: dict(n=int(len(g)), any_prod=int(g['any'].sum()), any_tt=int(anyb.reindex(g.index).fillna(False).sum()))
             for k, g in T.groupby('sub')}
        E = pd.read_parquet(p.replace('.candidates', ''))
        P0 = pd.read_parquet(EV_PROD).drop_duplicates('event_id').set_index('event_id')
        e = E.drop_duplicates('event_id').set_index('event_id').reindex(ctl).dropna(subset=['x_p0'])
        r['controls'] = dict(n=int(len(e)), split=int((e.get('n_splits', pd.Series(0, index=e.index)).fillna(0) > 0).sum()),
                             views_moved=sum(int(((e[f'{v}_p0'] - P0.reindex(e.index)[f'{v}_p0']).abs() > 0.05).sum()) for v in 'xy'))
        out[arm] = r
    return out


def recat():
    """The budget re-categorised (Dylan 10-26): every interior ray in one class, first match wins:
      detected      production
      band          an event with a noise band (bands.py) lost to the n_dropped cut or the spark veto, not a discharge
      multi:*       det3 sees >= 2 particle clusters in a view (multitrack.py class M): m3_picked (the old 'found, cut'
                    second particle) / other_picked / m3_not_found
      spark:*       the rest of the spark veto, by sparks.py class
      far:*         the C22 far classes of single-particle events (DELTA, FIT_SAT, FIT, EDGE, POS_TAIL, MULTI = a
                    second deposit C22 sees that is not particle-like)
      head_on       the old 'fit failed' (a 2-strip x cluster)
      other         charge elsewhere etc."""
    R = rays(); D = pd.read_parquet(os.path.join(S.OUT, 'multitrack_det3.parquet'))
    T = pd.read_parquet(os.path.join(S.OUT, 'multi_sub.parquet'))
    Sp = pd.read_parquet(os.path.join(S.OUT, 'sparks_det3.parquet')).set_index('event_id').cls
    Zb = np.load(os.path.join(S.OUT, 'bands_scan.npz'))
    import bands as Bd
    band = set()
    for v in 'xy':
        band |= set(Zb[f'eid_{v}'][Bd.band_events(Zb, v).any(1)].tolist())
    cat = []
    for e, r in R.iterrows():
        c = r.cat_xy_5
        if c == 'detected':
            k = 'detected'
        elif e in band and c in ('rejected_cut', 'spark') and Sp.get(e, '') != 'discharge':
            k = 'band'
        elif e in T.index:
            k = 'multi:' + T.loc[e, 'sub']
        elif c == 'spark':
            k = 'spark:' + str(Sp.get(e, 'unclassified'))
        elif c == 'far':
            k = 'far:' + str(r.cls)
        elif c == 'fit_failed':
            k = 'head_on'
        elif c == 'rejected_cut':
            k = 'cut:other'
        else:
            k = 'other'
        cat.append(k)
    R['recat'] = cat
    R[['recat']].to_parquet(os.path.join(S.OUT, 'recat_det3.parquet'))
    n = len(R)
    return dict(n=n, pct={k: 100 * v / n for k, v in R.recat.value_counts().items()},
                counts=R.recat.value_counts().to_dict())


def multi_flow():
    """The multi-particle accounting as one closed flow (Dylan 10-26: make the M3 double-track accounting explicit),
    recomputed from multitrack_det3.parquet (class S/M/V, prod/noband/any per interior ray), multi_sub.parquet (the M
    sub-classes and x x y topology) and the budget's own categories on the rays (cat_xy_5, C22 cls), so every number
    on the flow slides is a count of rows, not a copy of another summary."""
    R = rays(); D = pd.read_parquet(os.path.join(S.OUT, 'multitrack_det3.parquet'))
    T = pd.read_parquet(os.path.join(S.OUT, 'multi_sub.parquet'))
    assert len(D) == len(R) and (D.index == R.index).all()
    c22 = R.cls.astype(str)
    out = dict(n=len(D))
    for k in 'SMV':
        g = D[D.cls == k]
        out[k] = dict(n=int(len(g)), prod=int(g['prod'].sum()), noband=int(g.noband.sum()), any=int(g['any'].sum()),
                      noband_or_any=int((g.noband | g['any']).sum()), cat=g.cat.value_counts().to_dict())
    big = lambda t: ('2x2' if t == '2x2' else ('1x2/2x1' if t in ('1x2', '2x1') else 'with 3'))
    out['sub'] = {}
    for k, g in T.groupby('sub'):
        old = [(c if c != 'far' else 'far:' + c22[e]) for e, c in g['cat'].items()]
        out['sub'][k] = dict(n=int(len(g)), any=int(g['any'].sum()), topo=g.topo.map(big).value_counts().to_dict(),
                             topo_full=g.topo.value_counts().to_dict(), old=pd.Series(old).value_counts().to_dict())
    rc = D[D.cat == 'rejected_cut']
    out['old_found_cut'] = dict(n=int(len(rc)), M=int((rc.cls == 'M').sum()), S=int((rc.cls == 'S').sum()),
                                S_noband=int(rc[rc.cls == 'S'].noband.sum()))
    fm = D[(D.cat == 'far') & (c22 == 'MULTI')]
    out['old_far_multi'] = dict(n=int(len(fm)), M=int((fm.cls == 'M').sum()), S=int((fm.cls == 'S').sum()))
    fe = D[(D.cat == 'far') & (c22 == 'EDGE')]
    out['old_far_edge'] = dict(n=int(len(fe)), M=int((fe.cls == 'M').sum()), S=int((fe.cls == 'S').sum()))
    out['total'] = dict(prod=int(D['prod'].sum()), any=int(D['any'].sum()), noband=int(D.noband.sum()),
                        noband_or_any=int((D.noband | D['any']).sum()))
    return out


EV_PROD = os.path.expanduser('~/x17/cosmic_bench/outliers/final/mx17_det3_saturday_scan_6-27-26/'
                             'long_run_resist_490V_drift_1000V/mx17_3/wft/final_phys/events.parquet')


def main():
    S_ = dict(budget=budget(), sat=sat_dependence(), rail=json.load(open(os.path.join(S.OUT, 'railstats.json'))),
              arms=arms(), sparks=sparks(), fit_failed=fit_failed(), multi=multi(), delta=delta())
    S_['band_fix'] = band_fix()
    nb = len(pd.read_csv(os.path.join(S.OUT, 'ids_band.csv'))) - 400
    S_['band_fix_left'] = _pair_runs(None, 'reco_band_left', 'ids_band.csv', nb)   # WFT_DROP_FLAT_HITS=left
    S_['two_track'] = two_track(); S_['recat'] = recat(); S_['headon_wft'] = headon_wft()
    S_['multi_flow'] = multi_flow()
    for k in ('multitrack', 'headon', 'bands', 'multi_sub', 'pairing_toy', 'overlay', 'headon2', 'headon3'):
        # multitrack.py count / sub, headon.py, bands.py report, pairing.py toy, overlay.py score, headon2/3.py
        p = os.path.join(S.OUT, f'{k}.json')
        S_[k] = json.load(open(p)) if os.path.exists(p) else None
    json.dump(S_, open(os.path.join(S.OUT, 'summary.json'), 'w'), indent=1, default=float)
    for k in ('arms', 'sparks', 'fit_failed'):
        print(k, json.dumps(S_[k], default=float)[:1500])


if __name__ == '__main__':
    main()
