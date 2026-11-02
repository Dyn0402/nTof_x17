#!/usr/bin/env python3
"""merged_gate.py --gate DIR --rays-root R [--out results/merged_gate.json]
-- the fleet false-split gate for the 'charge' split corroboration, all five bench chambers (MERGED_PROJECTION_2026-11-02.md).

Inputs: merged_rereco.py's 'cost' and 'charge' re-reconstructions of each chamber's gate events (<gate>/<det>/<variant>/)
and the event sets of the gate builder (<gate>/sets_<det>.json):
  clean     the ws_unify definition (split_census.py): interior M3 ray, production track within 3 mm in both views
            (cat_xy_3 == 'detected');
  matched   interior M3 ray matched within 3 mm by some candidate in both views, any multiplicity;
  mismatch  clean or matched events whose pre-split candidate counts differ between views (the only events the 'charge'
            rule can touch); control: 500 random clean singles.
Per chamber and view: the ws_unify split rate (accepted splits of the rank-0 candidate on clean singles) under 'cost' and
'charge'; on all re-reconstructed events, the splits 'charge' adds, and what they do to the M3 muon: detected within 3 / 5 mm
in both views before and after, its distance in the split view before and after, and n_tracks transitions.  The rates are
over the whole clean / matched population (events outside the re-reconstructed set cannot change: no count mismatch).
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}


def dist(C, R):
    """Per event and view: distance of the nearest live candidate to the M3 ray."""
    C = C[C['rank'] >= 0].copy()
    C['mesh'] = np.where(C.plane == 'x', R.mesh_x.reindex(C.event_id).to_numpy(), R.mesh_y.reindex(C.event_id).to_numpy())
    C['d'] = (C.p0 - C.mesh).abs()
    return C.groupby(['event_id', 'plane']).d.min().unstack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', required=True)
    ap.add_argument('--rays-root', required=True, nargs='+')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'merged_gate.json'))
    a = ap.parse_args()
    res = {}
    for det, rel in ROWS.items():
        d = os.path.join(a.gate, det)
        if not os.path.exists(os.path.join(d, 'charge', 'events.parquet')):
            continue
        sets = json.load(open(os.path.join(a.gate, f'sets_{det}.json')))
        rp = next(p for p in (os.path.join(r, 'x17rerun/paper1_eff/budget', rel, f'rays_{det}.parquet') for r in a.rays_root)
                  if os.path.exists(p))
        R = pd.read_parquet(rp).drop_duplicates('event_id').set_index('event_id')
        V = {}
        for v in ('cost', 'charge'):
            C = pd.read_parquet(os.path.join(d, v, 'events.candidates.parquet'))
            sp = os.path.join(d, v, 'events.splits.parquet')
            S = pd.read_parquet(sp) if os.path.exists(sp) else pd.DataFrame(columns=['event_id', 'plane', 'rank', 'accepted'])
            E = pd.read_parquet(os.path.join(d, v, 'events.parquet')).set_index('event_id')
            V[v] = (C, S, E, dist(C, R))
        ids = V['cost'][2].index.intersection(V['charge'][2].index)
        clean, matched = set(sets['clean']), set(sets['matched'])
        out = dict(n_clean=len(clean), n_matched=len(matched), n_rereco=int(len(ids)),
                   n_rereco_mismatch=len(set(sets['mismatch']) & set(ids)))
        for v, (C, S, E, D) in V.items():
            acc = S[S.accepted.astype(bool)] if len(S) else S
            r = {}
            for pl in 'xy':
                a0 = acc[(acc.plane == pl) & (acc['rank'] == 0)] if len(acc) else acc
                r[f'split_rate_clean_{pl}_pct'] = 100.0 * len(set(a0.event_id) & clean) / max(len(clean), 1)
                a1 = acc[acc.plane == pl] if len(acc) else acc
                r[f'split_rate_matched_{pl}_pct'] = 100.0 * len(set(a1.event_id) & matched) / max(len(matched), 1)
            out[v] = r
        # what 'charge' adds
        Sc = V['charge'][1]
        add = Sc[Sc.get('accepted_charge', pd.Series(False, index=Sc.index)).astype(bool)
                 & ~Sc.get('accepted_count', pd.Series(False, index=Sc.index)).astype(bool)] if len(Sc) else Sc
        Dc, Dq = V['cost'][3].reindex(ids), V['charge'][3].reindex(ids)
        Ec, Eq = V['cost'][2].reindex(ids), V['charge'][2].reindex(ids)
        ev = sorted(set(add.event_id) & set(ids)) if len(add) else []
        det_c = {r: ((Dc.x < r) & (Dc.y < r)) for r in (3.0, 5.0)}
        det_q = {r: ((Dq.x < r) & (Dq.y < r)) for r in (3.0, 5.0)}
        rows = []
        for e in ev:
            pl = add[add.event_id == e].plane.iloc[0]
            rows.append(dict(event_id=int(e), plane=pl, d_before=float(Dc.loc[e, pl]), d_after=float(Dq.loc[e, pl]),
                             clean=e in clean, matched=e in matched,
                             det5_before=bool(det_c[5.0].get(e, False)), det5_after=bool(det_q[5.0].get(e, False)),
                             det3_before=bool(det_c[3.0].get(e, False)), det3_after=bool(det_q[3.0].get(e, False)),
                             ntr_before=int(Ec.loc[e, 'n_tracks']), ntr_after=int(Eq.loc[e, 'n_tracks'])))
        A = pd.DataFrame(rows)
        tr = pd.DataFrame({'a': Ec.n_tracks, 'b': Eq.n_tracks})
        out['charge_adds'] = dict(
            n_events=int(len(A)),
            in_clean=int(A.clean.sum()) if len(A) else 0, in_matched=int(A.matched.sum()) if len(A) else 0,
            rate_clean_pct=100.0 * (int(A.clean.sum()) if len(A) else 0) / max(len(clean), 1),
            rate_matched_pct=100.0 * (int(A.matched.sum()) if len(A) else 0) / max(len(matched), 1),
            muon_det5=dict(lost=int((A.det5_before & ~A.det5_after).sum()) if len(A) else 0,
                           gained=int((~A.det5_before & A.det5_after).sum()) if len(A) else 0),
            muon_det3=dict(lost=int((A.det3_before & ~A.det3_after).sum()) if len(A) else 0,
                           gained=int((~A.det3_before & A.det3_after).sum()) if len(A) else 0),
            d_split_view_median_before=float(A.d_before.median()) if len(A) else None,
            d_split_view_median_after=float(A.d_after.median()) if len(A) else None,
            closer=int((A.d_after < A.d_before - 0.05).sum()) if len(A) else 0,
            farther=int((A.d_after > A.d_before + 0.05).sum()) if len(A) else 0,
            events=rows)
        out['all_rereco'] = dict(
            det5_cost=int(det_c[5.0].sum()), det5_charge=int(det_q[5.0].sum()),
            det3_cost=int(det_c[3.0].sum()), det3_charge=int(det_q[3.0].sum()),
            n_tracks_transitions={f'{i}->{j}': int(c) for (i, j), c in tr[tr.a != tr.b].groupby(['a', 'b']).size().items()},
            control_changed=int(sum(1 for e in sets['control'] if e in ids and
                                    not np.allclose(Dc.loc[e].fillna(-1).to_numpy(), Dq.loc[e].fillna(-1).to_numpy()))))
        res[det] = out
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'charge_adds'} | {
        'charge_adds': {kk: vv for kk, vv in v['charge_adds'].items() if kk != 'events'}} for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    main()
