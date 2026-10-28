#!/usr/bin/env python3
"""gd_score.py [--dets det2 det3 det4 det6 det7] -- GD: score the spark criteria against M3 on the fe gate sample.

Inputs (pull.sh): per chamber ~/x17/cosmic_bench/model_audit/gd_sparks/<det>/{hitfeat,wffeat}.parquet,
nospark/events.parquet (lx_sparks.py on lxplus, the fe algorithm with the spark veto off, candidates only) and the fe
gate products (fe_unify2/gate_tree/<row>/wft/fe/{events.parquet,budget/rays_<det>.parquet}).

Per M3 ray in the interior (the budget's denominator; ~not_measured), the event's fit is the no-veto fit for a
candidate (floored count > 30) and the fe fit otherwise; the identity of the two on 200 ordinary events and on the
candidates the fe arm did fit is checked first.  A criterion vetoes a set of events; then
  detected   both views fitted, |r| <= 5 mm (raw strip frame), n_dropped <= 2 in both, not vetoed (the budget rule)
  bad        among the accepted fits (both views fitted, n_dropped <= 2, not vetoed): |r| > 5 mm, or |dtheta| > 5 deg in
             a view against M3 (dtheta = atan(fit tan) - atan(M3 tan), median-centred on the clean sample)
  s68        68 % half-width of dtheta over the detected rays, per view
For waveform criteria the non-candidates are represented by the random controls (weight N_rest / N_ctl).
Writes gd_sparks/results/{score.json,roc_<det>.csv,classes.json} (in the repo) and prints the tables.
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.expanduser('~/x17/cosmic_bench/model_audit/gd_sparks')
G = os.path.expanduser('~/x17/cosmic_bench/model_audit/fe_unify2/gate_tree')
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
FIT_COLS = ['x_ok', 'y_ok', 'x_p0', 'y_p0', 'x_n_dropped', 'y_n_dropped', 'x_tan_theta', 'y_tan_theta', 'x_chi2',
            'y_chi2', 'x_dof', 'y_dof', 'n_hits', 'n_tracks', 'x_n_sat_strips', 'y_n_sat_strips']
R_MM, DTH_BAD = 5.0, 5.0


def table(p):
    E = pd.read_parquet(p)
    return E.drop_duplicates('event_id', keep=False).set_index('event_id')


def load(det):
    d = os.path.join(L, det)
    F = pd.read_parquet(os.path.join(d, 'hitfeat.parquet'))
    W = pd.read_parquet(os.path.join(d, 'wffeat.parquet'))
    meta = json.load(open(os.path.join(d, 'gd_meta.json')))
    N = table(os.path.join(d, 'nospark', 'events.parquet'))
    fe = os.path.join(G, ROWS[det], 'wft', 'fe')
    E = table(os.path.join(fe, 'events.parquet'))
    R = pd.read_parquet(os.path.join(fe, 'budget', f'rays_{det}.parquet')).drop_duplicates('event_id', keep=False)
    R = R[(R.region == 'interior') & ~R.not_measured].set_index('event_id')
    return F, W, meta, N, E, R


def identity(N, E, ids):
    """Max |difference| of the fit columns between the no-veto run and the fe table on common events."""
    c = sorted(set(ids) & set(N.index) & set(E.index))
    out = {'n': len(c)}
    for k in ('x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_chi2', 'y_chi2'):
        a, b = N.loc[c, k].astype(float), E.loc[c, k].astype(float)
        both = a.notna() & b.notna()
        out[k] = float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0
        out[k + '_nan_mismatch'] = int((a.isna() != b.isna()).sum())
    return out


def per_ray(det):
    F, W, meta, N, E, R = load(det)
    cand = set(W.index[W.cand]) | set(F.index[(F.mult > meta['min_cand']) | (F.n_hits > meta['min_cand'])])
    idc = json.load(open(os.path.join(L, det, 'idcheck.json')))
    ident = dict(ordinary=identity(N, E, idc), candidates_fitted_by_fe=identity(N, E, cand))
    # the fit each event gets with no veto: the candidates' no-veto fits, everyone else's fe fit
    M = pd.concat([E[~E.index.isin(cand)][FIT_COLS], N[N.index.isin(cand)][FIT_COLS]]).rename(columns={'n_hits': 'n_hits_fit'})
    X = R[['mesh_x', 'mesh_y', 'tan_x', 'tan_y', 'cat_xy_5', 'mult']].join(M, how='left')
    X = X.join(F.drop(columns=['mult']), how='left').join(W, how='left', rsuffix='_w')
    X['cand'] = X.index.isin(cand)
    X['is_ctl'] = X.index.isin(set(W.index[~W.cand]))
    for v in 'xy':
        X[f'{v}_ok'] = X[f'{v}_ok'].astype('boolean').fillna(False).astype(bool)
    X['fitted'] = X.x_ok & X.y_ok
    X['nd_ok'] = (X.x_n_dropped.fillna(99) <= 2) & (X.y_n_dropped.fillna(99) <= 2)
    X['r'] = np.hypot(X.x_p0 - X.mesh_x, X.y_p0 - X.mesh_y)
    X['near'] = X.fitted & (X.r <= R_MM)
    for v in 'xy':
        X[f'dth_{v}'] = np.degrees(np.arctan(X[f'{v}_tan_theta']) - np.arctan(X[f'tan_{v}']))
    clean = X.near & X.nd_ok & ~X.cand
    for v in 'xy':
        X[f'dth_{v}'] -= X.loc[clean, f'dth_{v}'].median()
    X['dth_bad'] = (X.dth_x.abs() > DTH_BAD) | (X.dth_y.abs() > DTH_BAD)
    X['accepted'] = X.fitted & X.nd_ok
    X['det_nv'] = X.near & X.nd_ok                                     # detected with no veto
    X['bad_nv'] = X.accepted & (~X.near | X.dth_bad)
    # weights: candidates 1; non-candidates are all scored by the hit criteria, and for waveform criteria the
    # controls stand for them
    n_rest, n_ctl = int((~X.cand).sum()), int((X.is_ctl & ~X.cand).sum())
    X['w_wf'] = np.where(X.cand, 1.0, np.where(X.is_ctl, n_rest / max(n_ctl, 1), 0.0))
    # waveform-derived event features (max over views)
    for k in ('n_on6', 'n_on20', 'n_neg10', 'n_rail', 'n_band', 'cm_step_max', 'n_blk_step', 'span_mm'):
        if f'x_{k}' in X:
            X[k] = X[[f'x_{k}', f'y_{k}']].max(axis=1)
    X['n_hits_eff'] = X.n_hits_reseed.where(X.n_hits_reseed.notna(), X.n_hits)    # the fe veto quantity
    X['nseed_max'] = X[['x_nseed', 'y_nseed']].max(axis=1)
    X['nseed_min'] = X[['x_nseed', 'y_nseed']].min(axis=1)
    X['nsat'] = X.x_nsat + X.y_nsat
    return X, ident, meta


def evaluate(X, veto, wf):
    """veto: boolean Series over X (True = event dropped).  wf: a waveform criterion -- known on the candidates and the
    controls only: it is applied to the candidates, and what it would do to the non-candidates is estimated from the
    controls (their per-ray loss, scaled to the non-candidate population) and quoted separately (ctl_*)."""
    veto = veto.fillna(False).astype(bool)
    out = {}
    if wf:
        ctl = X.is_ctl & ~X.cand
        n_rest = int((~X.cand).sum())
        out['ctl_vetoed'] = 100 * float(veto[ctl].mean()) if ctl.any() else 0.0
        out['ctl_det_lost'] = 100 * float((veto & X.det_nv)[ctl].mean()) * n_rest / len(X) if ctl.any() else 0.0
        veto = veto & X.cand
    n = len(X)
    acc = X.accepted & ~veto
    D = X[X.det_nv & ~veto]
    out.update(detected=100 * float((X.det_nv & ~veto).sum()) / n - out.get('ctl_det_lost', 0.0),
               bad=100 * float((X.bad_nv & ~veto).sum()) / max(int(acc.sum()), 1),
               n_bad=int((X.bad_nv & ~veto).sum()), n_det=int((X.det_nv & ~veto).sum()),
               vetoed=100 * float(veto.sum()) / n,
               s68_x=float(np.percentile(np.abs(D.dth_x.dropna()), 68)),
               s68_y=float(np.percentile(np.abs(D.dth_y.dropna()), 68)),
               tail3=100 * float(((D.dth_x.abs() > 3) | (D.dth_y.abs() > 3)).mean()))
    return out


def bins(X):
    """Per bin of the veto quantity: rays, detected / accepted-but-bad / not accepted with no veto."""
    b = pd.cut(X.n_hits_eff, [-1, 20, 30, 40, 50, 65, 80, 100, 130, 200, 1e9])
    g = X.groupby(b, observed=True)
    T = pd.DataFrame(dict(n=g.size(), det=g.det_nv.sum(), bad=g.bad_nv.sum(),
                          not_acc=g.apply(lambda d: int((~d.accepted).sum()))))
    T['det_pct'] = 100 * T.det / T.n
    T['purity_of_accepted'] = 100 * T.det / (T.det + T.bad).clip(lower=1)
    T.index = T.index.astype(str)
    return T


def criteria(X):
    """name -> (veto Series, is waveform criterion)."""
    C = {'none': (pd.Series(False, index=X.index), False)}
    for t in (30, 40, 50, 60, 80, 100, 130, 160, 200, 300):
        C[f'n_hits>{t}'] = (X.n_hits_eff > t, False)
    for t in (30, 50, 80, 120, 200):
        C[f'plane_max>{t}'] = (X.nseed_max > t, False)
        C[f'plane_both>{t}'] = (X.nseed_min > t, False)
    for t in (5, 20, 100, 300, 500):
        C[f'rail>{t}'] = (X.n_rail > t, True)
    for t in (100, 150, 250, 350):
        C[f'on6>{t}'] = (X.n_on6 > t, True)
    for t in (300, 1000, 2000):
        C[f'cm_step>{t}'] = (X.cm_step_max > t, True)
    for t in (50, 100, 200):
        C[f'neg10>{t}'] = (X.n_neg10 > t, True)
    for t in (10, 20, 40):
        C[f'nsat_hits>{t}'] = (X.nsat > t, False)
    # combinations: the hit count AND a discharge signature
    dis = (X.n_rail > 20) | (X.n_on6 > 150) | (X.n_neg10 > 100)
    C['discharge(rail>20|on6>150|neg10>100)'] = (dis, True)
    for t in (50, 100):
        C[f'n_hits>{t} & discharge'] = (dis & (X.n_hits_eff > t), True)
    return C


def classify(X):
    """The events the current veto (fe: n_hits after band re-seed > 50) drops, by what they are."""
    V = X[X.n_hits_eff > 50].copy()
    dis = (V.n_rail > 20) | (V.n_on6 > 150) | (V.n_neg10 > 100)
    V['cls'] = np.select(
        [dis & V.det_nv, dis & ~V.det_nv, ~dis & V.det_nv, ~dis & V.accepted, ~dis & V.fitted],
        ['discharge, M3 track found', 'discharge, lost', 'no discharge, M3 track found', 'no discharge, wrong track',
         'no discharge, competing clusters'], 'no discharge, not fitted')
    return V


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dets', nargs='*', default=list(ROWS))
    a = ap.parse_args()
    res = {}
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    for det in a.dets:
        if not os.path.exists(os.path.join(L, det, 'nospark', 'events.parquet')):
            print(det, 'missing'); continue
        X, ident, meta = per_ray(det)
        X.to_parquet(os.path.join(L, det, 'scored.parquet'))
        rows = []
        for name, (veto, wf) in criteria(X).items():
            r = evaluate(X, veto, wf); r['criterion'] = name; r['waveform'] = wf; rows.append(r)
        T = pd.DataFrame(rows).set_index('criterion')
        base = T.loc['n_hits>50']
        T['d_detected'] = T.detected - base.detected; T['d_bad'] = T.bad - base.bad; T['dn_det'] = T.n_det - base.n_det; T['dn_bad'] = T.n_bad - base.n_bad
        T.to_csv(os.path.join(HERE, 'results', f'roc_{det}.csv'))
        V = classify(X)
        fe_det = float((X.cat_xy_5 == 'detected').mean() * 100)
        res[det] = dict(n_interior=len(X), n_cand=int(X.cand.sum()), identity=ident, fe_detected=fe_det,
                        reproduced_detected=float(base.detected), vetoed=dict(V.cls.value_counts()),
                        n_vetoed=len(V), roc=T.reset_index().to_dict('records'))
        print(f'\n== {det}: {len(X)} interior rays, {int(X.cand.sum())} candidates; fe detected {fe_det:.2f} %, '
              f'reproduced with n_hits>50: {base.detected:.2f} %')
        print('identity:', json.dumps(ident))
        print('vetoed by n_hits>50 (', len(V), '):', V.cls.value_counts().to_dict())
        B = bins(X); res[det]['bins'] = B.reset_index().to_dict('records')
        print(B.round(1).to_string())
        cols = ['detected', 'd_detected', 'n_det', 'bad', 'd_bad', 'n_bad', 'vetoed', 's68_x', 's68_y', 'tail3']
        cols += [c for c in ('ctl_vetoed', 'ctl_det_lost') if c in T]
        print(T[cols].round(3).to_string())
    json.dump(res, open(os.path.join(HERE, 'results', 'score.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
