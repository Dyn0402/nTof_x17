#!/usr/bin/env python3
"""merged_compare.py --root R --rereco DIR [--oneview results/oneview_events.parquet] [--out results/merged_projection.json]
-- score merged_rereco.py's variants on the det3 'merged in projection' events (MERGED_PROJECTION_2026-11-02.md).

Per event of single_track_v2.py's one-view class (merged in projection + unclear), per variant:
  * what the joint two-track fit did on the merged view: no attempt (and why: which trigger was missing) or an attempt
    and which guard refused it (fstat, distinguishable, column_shared, both_plausible) -- or accepted;
  * the muon (the M3 ray at det3): a candidate within R_MM in each view (credit), and the two forming ONE track
    (X/Y pairing correct);
  * the second particle: its multi-view candidate (single_track_v2's 'sec') in a track whose merged-view member is a
    split child, i.e. both particles reconstructed as tracks.
And over every re-reconstructed event: the transitions between variants (n_tracks, accepted splits, credit), so the
change's effect outside the 195 is visible on the same events (the count-mismatch events and a random 600 per row).
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
R_MM = 5.0


def load(d):
    C = pd.read_parquet(os.path.join(d, 'events.candidates.parquet'))
    sp = os.path.join(d, 'events.splits.parquet')
    S = pd.read_parquet(sp) if os.path.exists(sp) else pd.DataFrame(columns=['event_id', 'plane', 'accepted'])
    E = pd.read_parquet(os.path.join(d, 'events.parquet')).set_index('event_id')
    return C, S, E


def score_event(e, C, S, ray, mv, sec_p0):
    """One event under one variant."""
    ov = 'y' if mv == 'x' else 'x'                     # the merged view
    g = C[(C.event_id == e) & (C['rank'] >= 0)]
    r = {}
    near = {}
    for v in 'xy':
        gv = g[g.plane == v]
        d = (gv.p0 - ray[f'mesh_{v}']).abs()
        near[v] = gv[d < R_MM]
        r[f'd_{v}'] = float(d.min()) if len(d) else np.nan
        r[f'n_{v}'] = int(len(gv))
    r['credit'] = bool(len(near['x']) and len(near['y']))
    tx = set(near['x'].track_id[near['x'].track_id >= 0]); ty = set(near['y'].track_id[near['y'].track_id >= 0])
    r['muon_track'] = bool(tx & ty)
    r['muon_track_gated'] = bool(set(near['x'].track_id[(near['x'].track_id >= 0) & near['x'].track_gated])
                                 & set(near['y'].track_id[(near['y'].track_id >= 0) & near['y'].track_gated]))
    # second particle: its multi-view candidate, the track it is in, and that track's merged-view member
    gm = g[g.plane == mv]
    s = gm.iloc[(gm.p0 - sec_p0).abs().argsort()[:1]] if len(gm) else gm
    r['sec_track'] = False
    if len(s) and abs(float(s.p0.iloc[0]) - sec_p0) < 0.5 and int(s.track_id.iloc[0]) >= 0:
        tid = int(s.track_id.iloc[0])
        mem = g[(g.plane == ov) & (g.track_id == tid)]
        mu_mem = g[(g.plane == ov) & g.track_id.isin(tx | ty)]
        r['sec_track'] = bool(len(mem) and not (tid in (tx & ty)))
        r['sec_track_split'] = bool(len(mem) and bool(mem.split_child.iloc[0]))
    r['split_merged_view'] = bool(g[g.plane == ov].split_child.any())
    a = S[(S.event_id == e) & (S.plane == ov)]
    if not len(a):
        r['fit'] = 'no attempt'
    elif a.accepted.any():
        r['fit'] = 'accepted'
    else:
        b = a.sort_values('fstat', ascending=False).iloc[0]
        if b.get('reverted_lost_track', False) is True:
            r['fit'] = 'reverted (lost track)'
        else:
            why = [k for k in ('distinguishable', 'column_shared', 'both_plausible') if not b[k]]
            if b.fstat < b.threshold:
                why = ['fstat'] + why
            r['fit'] = 'refused: ' + '+'.join(why) if why else 'refused: ?'
        for k in ('fstat', 'threshold', 'corroborated', 'distinguishable', 'column_shared', 'both_plausible', 'sep',
                  'corrob_charge', 'cc_match', 'cc_z_sum', 'a_plausible', 'b_plausible'):
            if k in b:
                r[f'att_{k}'] = b[k]
    if len(a):
        r['att_n'] = int(len(a))
        r['att_trig'] = '+'.join(k for k in ('residual', 'width', 'cross_plane') if a[f'trig_{k}'].any()) or 'charge'
    return r


def no_attempt_reason(e, C, mv, tc_tol):
    """Why the 'count' rule made no attempt on the merged view: does the multi view have two coincident PLAUSIBLE
    candidates (the count trigger's condition)?"""
    g = C[(C.event_id == e) & (C['rank'] >= 0)]
    gm = g[g.plane == mv]
    pl = gm[gm.plausible]
    if len(pl) < 2:
        return 'multi view: < 2 plausible candidates'
    t = (pl.t0 + pl.q_u50.fillna(0)).to_numpy()
    if not any(abs(t[i] - t[j]) <= tc_tol for i in range(len(t)) for j in range(i + 1, len(t))):
        return 'multi view: plausible pair not coincident'
    return 'count trigger satisfied (residual/width/selection)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rereco', required=True)
    ap.add_argument('--oneview', default=os.path.join(HERE, 'results', 'oneview_events.parquet'))
    ap.add_argument('--variants', default='p1,cost,charge')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'merged_projection.json'))
    ap.add_argument('--events-out', default=None)
    a = ap.parse_args()
    O = pd.read_parquet(a.oneview)
    O = O[O.ov_cls.isin(['merged in projection', 'unclear'])]
    res, per = {}, []
    for key, rel in ROWS.items():
        B = pd.read_parquet(os.path.join(a.root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet')
                            ).drop_duplicates('event_id').set_index('event_id')
        Cp = pd.read_parquet(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
        Sp = pd.read_parquet(os.path.join(a.root, 'Analysis_pass1', rel, 'wft/events.splits.parquet'))
        bundle = json.load(open(os.path.join(a.rereco, key, 'p1', 'calib_bundle', 'bundle.json')))
        tc_tol = bundle['xy_pairing']['tc_tol']
        V = {'pass1': (Cp, Sp, None)}
        for v in a.variants.split(','):
            if os.path.exists(os.path.join(a.rereco, key, v, 'events.parquet')):
                V[v] = load(os.path.join(a.rereco, key, v))
        Ok = O[O.row == key]
        for o in Ok.itertuples():
            ray = B.loc[o.event_id]
            # sec_p0: the multi-view second candidate's p0, from the pass-1 table (single_track_v2's choice)
            gm = Cp[(Cp.event_id == o.event_id) & (Cp.plane == o.multi_view) & (Cp['rank'] >= 0)].copy()
            gm['d'] = (gm.p0 - ray[f'mesh_{o.multi_view}']).abs()
            gm = gm.sort_values('d')
            sec_p0 = float(gm.p0.iloc[1]) if len(gm) >= 2 else np.nan
            for vn, (C, S, _E) in V.items():
                r = score_event(o.event_id, C, S, ray, o.multi_view, sec_p0)
                r.update(row=key, event_id=int(o.event_id), variant=vn, ov_cls=o.ov_cls, multi_view=o.multi_view,
                         sec_sep=o.sec_sep, other_d=o.other_d)
                if vn == 'pass1' and r['fit'] == 'no attempt':
                    r['no_attempt_reason'] = no_attempt_reason(o.event_id, C, o.multi_view, tc_tol)
                per.append(r)
        # transitions over every re-reconstructed event
        tr = {}
        base = V.get('cost')
        for vn in V:
            if vn in ('pass1',) or base is None:
                continue
            E0, E1 = base[2], V[vn][2]
            ix = E0.index.intersection(E1.index)
            t = pd.DataFrame({'a': E0.n_tracks.reindex(ix), 'b': E1.n_tracks.reindex(ix)})
            tr[vn] = dict(events=int(len(ix)),
                          n_tracks_transitions={f'{i}->{j}': int(c) for (i, j), c in
                                                t[t.a != t.b].groupby(['a', 'b']).size().items()},
                          n_splits=int(E1.get('n_splits', pd.Series(0, index=E1.index)).reindex(ix).sum()),
                          events_table_changed=int(sum(
                              ((E0[c].reindex(ix) - E1[c].reindex(ix)).abs() > 1e-9).sum()
                              for c in ('x_p0', 'y_p0'))))
        res[key] = dict(transitions_vs_cost=tr, tc_tol=tc_tol)
    P = pd.DataFrame(per)
    summ = {}
    for (cls, vn), g in P.groupby(['ov_cls', 'variant']):
        summ.setdefault(cls, {})[vn] = dict(
            n=int(len(g)), credit=float(g.credit.mean()), muon_track=float(g.muon_track.mean()),
            muon_track_gated=float(g.muon_track_gated.mean()), both_tracks=float(g.sec_track.mean()),
            split_merged_view=float(g.split_merged_view.mean()),
            fit={k: int(v) for k, v in g.fit.value_counts().items()})
    for cls in summ:
        g = P[(P.ov_cls == cls) & (P.variant == 'pass1') & (P.fit == 'no attempt')]
        summ[cls]['pass1_no_attempt_reason'] = {k: int(v) for k, v in g.no_attempt_reason.value_counts().items()}
    res['summary'] = summ
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    if a.events_out:
        P.to_parquet(a.events_out)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
