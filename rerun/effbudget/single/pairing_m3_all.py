#!/usr/bin/env python3
"""pairing_m3_all.py [--root R] [--replay pairing_replay_events.parquet] -- the M3 pairing test on EVERY event.

xy_pairing.py's 552 events need >= 2 candidates in both views.  This widens the test to every interior M3 event of the
two det3 rows where M3 marks exactly one X and one Y candidate (|p0 - M3 at the mesh| < 5 mm, the next > 10 mm) --
including events with one candidate in a view, where the other view's extra candidates are fragments or noise --
and scores both select_tracks modes of pairing_replay.py: the M3 X and Y candidates in one track (correct), one of
them joined to another candidate (wrong), or neither (unpaired), split by the candidate counts (n_x, n_y).
"""
import argparse, ast, json, os
import numpy as np
import pandas as pd

ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
R_IN, R_OUT = 5.0, 10.0


def m3_ranks(root, rel):
    C = pd.read_parquet(os.path.join(root, 'Analysis_pass1', rel, 'wft/events.candidates.parquet'))
    C = C[C['rank'] >= 0]
    B = pd.read_parquet(os.path.join(root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet'))
    B = B.drop_duplicates('event_id').set_index('event_id')
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


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--replay', default=os.path.join(here, 'results', 'pairing_replay_events.parquet'))
    ap.add_argument('--out', default=os.path.join(here, 'results', 'pairing_m3_all.json'))
    a = ap.parse_args()
    D = pd.read_parquet(a.replay)
    modes = [c[6:] for c in D.columns if c.startswith('pairs_')]
    rows = []
    for key, rel in ROWS.items():
        m = m3_ranks(a.root, rel)
        for r in D[D.row == key].itertuples():
            rx, ry = m.get((r.event_id, 'x')), m.get((r.event_id, 'y'))
            if rx is None or ry is None:
                continue
            rows.append(dict(row=key, event_id=r.event_id, nx=r.nx, ny=r.ny,
                             **{md: score(ast.literal_eval(getattr(r, f'pairs_{md}')), rx, ry) for md in modes}))
    S = pd.DataFrame(rows)
    S['cls'] = np.where((S.nx == 1) & (S.ny == 1), '1x1', np.where((S.nx >= 2) & (S.ny >= 2), '>=2 both', '1 in one view'))
    res = {}
    for cls, g in [('all', S)] + list(S.groupby('cls')):
        res[cls] = dict(n=int(len(g)), **{md: g[md].value_counts().to_dict() for md in modes},
                        changed=int((g[modes[0]] != g[modes[-1]]).sum()),
                        transitions={f'{o}->{n}': int(c) for (o, n), c in g.groupby([modes[0], modes[-1]]).size().items()
                                     if o != n})
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
