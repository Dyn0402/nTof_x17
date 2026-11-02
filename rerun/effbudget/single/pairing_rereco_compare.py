#!/usr/bin/env python3
"""pairing_rereco_compare.py --root R --rereco DIR [--events xy_pairing_events.parquet]
-- score pairing_rereco.py's two re-reconstructions (PAIRING_FIX_2026-11-02.md §3).

Per row: (1) 'gate' re-reco against the pass-1 table on the same events (events table and candidates: must agree,
or the comparison below is not a statement about pass 1); (2) 'cost' against 'gate': events-table columns, the
candidate fits (p0, w, t0, q_sum, chi2: must be bit-equal: select_tracks cannot change a fit; only the two-track
split's revert reads it), n_tracks, track_id; (3) the M3 pairing test of xy_pairing.py on both.
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
FIT = ['p0', 'w', 't0', 'q_sum', 'chi2']
EV = ['x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_chi2', 'y_chi2', 'x_ok', 'y_ok']


def cand(path):
    C = pd.read_parquet(path)
    return C.set_index(['event_id', 'plane', 'rank']).sort_index()


def same(a, b, cols):
    j = a[cols].join(b[cols], how='outer', lsuffix='_a', rsuffix='_b')
    bad = np.zeros(len(j), bool)
    for c in cols:
        x, y = j[f'{c}_a'], j[f'{c}_b']
        bad |= ~((x == y) | (x.isna() & y.isna()) | np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-9,
                                                              equal_nan=True))
    return int(bad.sum()), len(j)


def m3(C, E):
    out = {}
    for r in E.itertuples():
        try:
            t = {k: int(C.loc[(r.event_id, p, rk), 'track_id']) for k, p, rk in
                 (('xm', 'x', r.xm_rank), ('ym', 'y', r.ym_rank), ('xo', 'x', r.xo_rank), ('yo', 'y', r.yo_rank))}
        except KeyError:
            out['missing'] = out.get('missing', 0) + 1
            continue
        if t['xm'] >= 0 and t['xm'] == t['ym']:
            k = 'correct'
        elif (t['xm'] >= 0 and t['xm'] == t['yo']) or (t['ym'] >= 0 and t['ym'] == t['xo']):
            k = 'wrong'
        else:
            k = 'unpaired'
        out[k] = out.get(k, 0) + 1
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rereco', required=True)
    ap.add_argument('--events', default=os.path.join(here, 'results', 'xy_pairing_events.parquet'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    X = pd.read_parquet(a.events)
    res = {}
    for key, rel in ROWS.items():
        d = os.path.join(a.rereco, key)
        if not os.path.exists(os.path.join(d, 'cost', 'events.parquet')):
            continue
        Eg = pd.read_parquet(os.path.join(d, 'gate', 'events.parquet')).set_index('event_id')
        Ec = pd.read_parquet(os.path.join(d, 'cost', 'events.parquet')).set_index('event_id')
        Ep = pd.read_parquet(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/events.parquet')).set_index('event_id')
        Ep = Ep.loc[Ep.index.intersection(Eg.index)]
        Cg, Cc = cand(os.path.join(d, 'gate', 'events.candidates.parquet')), cand(os.path.join(d, 'cost', 'events.candidates.parquet'))
        Cp = cand(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
        Cp = Cp[Cp.index.get_level_values(0).isin(Eg.index)]
        Xk = X[X.row == key]
        Xk = Xk[Xk.event_id.isin(Eg.index)]
        ntr = pd.DataFrame({'g': Eg.n_tracks, 'c': Ec.n_tracks.reindex(Eg.index)})
        res[key] = dict(
            events=int(len(Eg)),
            gate_vs_pass1=dict(events_table_diff=same(Eg, Ep, EV), candidate_fit_diff=same(Cg, Cp, FIT),
                               track_id_diff=same(Cg, Cp, ['track_id', 'track_gated']),
                               n_tracks_diff=int((Eg.n_tracks != Ep.n_tracks.reindex(Eg.index)).sum())),
            cost_vs_gate=dict(events_table_diff=same(Ec, Eg, EV), candidate_fit_diff=same(Cc, Cg, FIT),
                              track_id_diff=same(Cc, Cg, ['track_id', 'track_gated']),
                              tan_diff=same(Cc, Cg, ['tan_theta']),
                              n_tracks_transitions={f'{o}->{n}': int(c) for (o, n), c in
                                                    ntr[ntr.g != ntr.c].groupby(['g', 'c']).size().items()},
                              n_splits_diff=int((Ec.get('n_splits', 0) != Eg.get('n_splits', 0)).sum())
                              if 'n_splits' in Eg else None),
            m3_truth=dict(n=int(len(Xk)), pass1=m3(Cp, Xk), gate=m3(Cg, Xk), cost=m3(Cc, Xk)))
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        open(a.out, 'w').write(s)


if __name__ == '__main__':
    main()
