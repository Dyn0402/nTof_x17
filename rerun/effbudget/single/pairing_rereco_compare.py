#!/usr/bin/env python3
"""pairing_rereco_compare.py --root R --rereco DIR [--row KEY=REL ...] [--out F]
-- score pairing_rereco.py's two re-reconstructions (PAIRING_FIX_2026-11-02.md §3, §7).

Per row (<rereco>/<KEY>/{gate,cost}/): (1) 'gate' re-reco against the pass-1 table on the same events (events table and
candidates: must agree, or the comparison is not a statement about pass 1); (2) 'cost' against 'gate': events-table
columns, the candidate fits (p0, w, t0, q_sum, chi2: must be bit-equal -- select_tracks cannot change a fit; only the
two-track split's revert reads it), n_tracks, track_id; (3) the M3 pairing test (pairing_m3_all.m3_ranks: M3 marks
exactly one X and one Y candidate) on pass 1, gate and cost, for all re-reconstructed events and the >= 2-per-view subset.
Default rows: the two det3 golden rows.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pairing_m3_all import m3_ranks  # noqa: E402

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
FIT = ['p0', 'w', 't0', 'q_sum', 'chi2']
EV = ['x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'x_chi2', 'y_chi2', 'x_ok', 'y_ok']


def cand(path):
    return pd.read_parquet(path).set_index(['event_id', 'plane', 'rank']).sort_index()


def same(a, b, cols):
    j = a[cols].join(b[cols], how='outer', lsuffix='_a', rsuffix='_b')
    bad = np.zeros(len(j), bool)
    for c in cols:
        x, y = j[f'{c}_a'], j[f'{c}_b']
        bad |= ~((x == y) | (x.isna() & y.isna()) | np.isclose(x.astype(float), y.astype(float), rtol=0, atol=1e-9,
                                                              equal_nan=True))
    return int(bad.sum()), len(j)


def m3(C, ranks, events):
    out = {'correct': 0, 'wrong': 0, 'unpaired': 0}
    for e in events:
        rx, ry = ranks[(e, 'x')], ranks[(e, 'y')]
        try:
            a, b = int(C.loc[(e, 'x', rx), 'track_id']), int(C.loc[(e, 'y', ry), 'track_id'])
        except KeyError:
            continue
        out['correct' if a >= 0 and a == b else 'wrong' if (a >= 0 or b >= 0) else 'unpaired'] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rereco', required=True)
    ap.add_argument('--row', nargs='*', default=None, help='KEY=REL (default: the two det3 golden rows)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = dict(r.split('=', 1) for r in a.row) if a.row else ROWS
    res = {}
    for key, rel in rows.items():
        d = os.path.join(a.rereco, key)
        if not os.path.exists(os.path.join(d, 'cost', 'events.parquet')):
            continue
        Eg = pd.read_parquet(os.path.join(d, 'gate', 'events.parquet')).set_index('event_id')
        Ec = pd.read_parquet(os.path.join(d, 'cost', 'events.parquet')).set_index('event_id')
        Ep = pd.read_parquet(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/events.parquet')).set_index('event_id')
        Ep = Ep.loc[Ep.index.intersection(Eg.index)]
        Cg = cand(os.path.join(d, 'gate', 'events.candidates.parquet'))
        Cc = cand(os.path.join(d, 'cost', 'events.candidates.parquet'))
        Cp = cand(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
        Cp = Cp[Cp.index.get_level_values(0).isin(Eg.index)]
        ntr = pd.DataFrame({'g': Eg.n_tracks, 'c': Ec.n_tracks.reindex(Eg.index)})
        rk = m3_ranks(a.root, rel) or {}
        ev = [e for e in Eg.index if (e, 'x') in rk and (e, 'y') in rk]
        ncx = Cp[Cp.index.get_level_values(2) >= 0].groupby(level=[0, 1]).size().unstack(fill_value=0)
        two = [e for e in ev if e in ncx.index and ncx.loc[e].get('x', 0) >= 2 and ncx.loc[e].get('y', 0) >= 2]
        res[key] = dict(
            rel=rel, events=int(len(Eg)),
            gate_vs_pass1=dict(events_table_diff=same(Eg, Ep, EV), candidate_fit_diff=same(Cg, Cp, FIT),
                               track_id_diff=same(Cg, Cp, ['track_id', 'track_gated']),
                               n_tracks_diff=int((Eg.n_tracks != Ep.n_tracks.reindex(Eg.index)).sum())),
            cost_vs_gate=dict(events_table_diff=same(Ec, Eg, EV), candidate_fit_diff=same(Cc, Cg, FIT),
                              track_id_diff=same(Cc, Cg, ['track_id', 'track_gated']),
                              tan_diff=same(Cc, Cg, ['tan_theta']),
                              n_tracks_transitions={f'{o}->{n}': int(c) for (o, n), c in
                                                    ntr[ntr.g != ntr.c].groupby(['g', 'c']).size().items()},
                              n_splits_diff=int((Ec.n_splits.reindex(Eg.index) != Eg.n_splits).sum())
                              if 'n_splits' in Eg else None),
            m3=dict(n=len(ev), pass1=m3(Cp, rk, ev), gate=m3(Cg, rk, ev), cost=m3(Cc, rk, ev)),
            m3_two_particle=dict(n=len(two), pass1=m3(Cp, rk, two), gate=m3(Cg, rk, two), cost=m3(Cc, rk, two)))
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        open(a.out, 'w').write(s)


if __name__ == '__main__':
    main()
