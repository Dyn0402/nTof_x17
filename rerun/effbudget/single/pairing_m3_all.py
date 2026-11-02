#!/usr/bin/env python3
"""pairing_m3_all.py [--root R] [--replay pairing_replay_events.parquet] -- the M3 pairing test on EVERY event.

xy_pairing.py's 552 events need >= 2 candidates in both views.  This widens the test to every interior M3 event
where M3 marks exactly one X and one Y candidate (|p0 - M3 at the mesh| < 5 mm, the next > 10 mm), including events
with one candidate in a view, and scores each select_tracks mode of pairing_replay.py: the M3 X and Y candidates in
one track (correct), one of them joined to another candidate (wrong), or neither (unpaired).  Split by candidate
counts: 1x1, one candidate in one view, >= 2 in both (the two-particle subset).  Per chamber and per row when the
replay table covers several rows (pairing_replay.py --all-rows); M3 at the mesh from the budget rays
(x17rerun/paper1_eff/budget/<rel>/rays_det<N>.parquet: interior, good_x, measured).
"""
import argparse, ast, json, os
import numpy as np
import pandas as pd

R_IN, R_OUT = 5.0, 10.0


def m3_ranks(root, rel):
    det = rel.split('/')[-1].replace('mx17_', 'det')
    rp = os.path.join(root, 'x17rerun/paper1_eff/budget', rel, f'rays_{det}.parquet')
    if not os.path.exists(rp):
        return None
    C = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'),
                        columns=['event_id', 'plane', 'rank', 'p0'])
    C = C[C['rank'] >= 0]
    B = pd.read_parquet(rp).drop_duplicates('event_id').set_index('event_id')
    B = B[(~B.not_measured) & (B.region == 'interior') & B.good_x]
    C = C[C.event_id.isin(B.index)].copy()
    C['d'] = (C.p0 - np.where(C.plane == 'x', B.mesh_x.reindex(C.event_id), B.mesh_y.reindex(C.event_id))).abs()
    out = {}
    for (e, p), g in C.groupby(['event_id', 'plane']):
        d = np.sort(g.d.to_numpy())
        if d[0] < R_IN and (len(d) == 1 or d[1] > R_OUT):
            out[(int(e), p)] = int(g['rank'].iloc[int(np.argmin(g.d.to_numpy()))])
    return out


def score(pairs, rx, ry):
    tx = {i: t for t, (i, j, _g) in enumerate(pairs)}
    ty = {j: t for t, (i, j, _g) in enumerate(pairs)}
    a, b = tx.get(rx, -1), ty.get(ry, -1)
    if a >= 0 and a == b:
        return 'correct'
    if a >= 0 or b >= 0:
        return 'wrong'
    return 'unpaired'


def summarise(S, modes):
    res = {}
    for cls, g in [('all', S)] + list(S.groupby('cls')):
        res[cls] = dict(n=int(len(g)), **{md: {c: int((g[md] == c).sum()) for c in ('correct', 'wrong', 'unpaired')}
                                          for md in modes},
                        changed=int((g[modes[0]] != g[modes[-1]]).sum()),
                        transitions={f'{o}->{n}': int(c) for (o, n), c in g.groupby([modes[0], modes[-1]]).size().items()
                                     if o != n})
    return res


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--replay', default=os.path.join(here, 'results', 'pairing_replay_events.parquet'))
    ap.add_argument('--out', default=os.path.join(here, 'results', 'pairing_m3_all.json'))
    ap.add_argument('--events-out', default=None, help='per-event scores (parquet)')
    a = ap.parse_args()
    D = pd.read_parquet(a.replay)
    if 'rel' not in D:          # the two-row det3 replay of the first version
        D['rel'] = D.row.map({'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
                              'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'})
    if 'id_unique' in D:        # event ids repeated across acquisitions cannot be matched to M3 rays by id
        D = D[D.id_unique]
    modes = [c[6:] for c in D.columns if c.startswith('pairs_')]
    rows, no_rays = [], []
    for rel, G in D.groupby('rel'):
        m = m3_ranks(a.root, rel)
        if m is None:
            no_rays.append(rel)
            continue
        for r in G.itertuples():
            rx, ry = m.get((r.event_id, 'x')), m.get((r.event_id, 'y'))
            if rx is None or ry is None:
                continue
            rows.append(dict(rel=rel, chamber=rel.split('/')[-1], event_id=r.event_id, nx=r.nx, ny=r.ny,
                             **{md: score(ast.literal_eval(getattr(r, f'pairs_{md}')), rx, ry) for md in modes}))
    S = pd.DataFrame(rows)
    S['cls'] = np.where((S.nx == 1) & (S.ny == 1), '1x1', np.where((S.nx >= 2) & (S.ny >= 2), '>=2 both', '1 in one view'))
    res = dict(modes=modes, rows_without_rays=no_rays, pooled=summarise(S, modes),
               chambers={ch: summarise(g, modes) for ch, g in S.groupby('chamber')})
    two = S[S.cls == '>=2 both']
    res['rows_two_particle'] = {rel: {md: {c: int((g[md] == c).sum()) for c in ('correct', 'wrong', 'unpaired')}
                                      for md in modes} for rel, g in two.groupby('rel')}
    json.dump(res, open(a.out, 'w'), indent=1)
    if a.events_out:
        S.to_parquet(a.events_out)
    for ch, r in res['chambers'].items():
        t = r.get('>=2 both', {})
        print(ch, 'all', r['all']['n'], {md: r['all'][md]['correct'] for md in modes},
              '| two-particle', t.get('n'), {md: t.get(md) for md in modes})


if __name__ == '__main__':
    main()
