#!/usr/bin/env python3
"""twotrack_validate.py --root R [--side DIR] -- validate the chamber A two-track re-pass (Analysis_pass1_twotrack,
bundles select=cost + split_corrob=charge) against pass 1 (MERGED_PROJECTION_2026-11-02.md §3d, results/merged_effcheck.json).
Same credit / rank-0 definitions as merged_effcheck.py, on the full pass tables (rays from pass 1's budget).
Writes results/twotrack_validate.json."""
import argparse, json, os
import numpy as np
import pandas as pd
from merged_effcheck import ROWS, det

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--analysis', default='Analysis_pass1_twotrack')
    ap.add_argument('--oneview', default=os.path.join(HERE, 'results', 'oneview_events.parquet'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'twotrack_validate.json'))
    a = ap.parse_args()
    O = pd.read_parquet(a.oneview)
    ref = json.load(open(os.path.join(HERE, 'results', 'merged_effcheck.json')))
    res = {}
    for key, rel in ROWS.items():
        B = pd.read_parquet(os.path.join(a.root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet')
                            ).drop_duplicates('event_id').set_index('event_id')
        B = B[B.region == 'interior']
        P = {v: pd.read_parquet(os.path.join(a.root, d, rel, 'wft/events.parquet')) for v, d in
             (('p1', 'Analysis_pass1'), ('tt', a.analysis))}
        Cd = {v: pd.read_parquet(os.path.join(a.root, d, rel, 'wft/events.candidates.parquet')) for v, d in
              (('p1', 'Analysis_pass1'), ('tt', a.analysis))}
        cr, pr = {}, {}
        for v in Cd:
            cr[v], pr[v] = det(Cd[v], B)
        ix = B.index
        cls = O[O.row == key].set_index('event_id').ov_cls.reindex(ix).fillna('other')
        T = pd.DataFrame({'c0': cr['p1'].reindex(ix).fillna(False), 'c1': cr['tt'].reindex(ix).fillna(False),
                          'p0': pr['p1'].reindex(ix).fillna(False), 'p1': pr['tt'].reindex(ix).fillna(False), 'cls': cls})
        out = {'interior_rays': int(len(T)), 'events_rows': {k: int(len(v)) for k, v in P.items()},
               'events_unique': {k: int(v.event_id.nunique()) for k, v in P.items()} if 'event_id' in P['p1'] else None,
               'ref_merged_effcheck': {c: (ref[key][c]['credit_cost'], ref[key][c]['credit_charge'], ref[key][c]['n'])
                                       for c in ref[key] if isinstance(ref[key][c], dict)}}
        for c, g in list(T.groupby('cls')) + [('ALL', T)]:
            out[c] = dict(n=int(len(g)), credit_p1=int(g.c0.sum()), credit_tt=int(g.c1.sum()),
                          gained=int((~g.c0 & g.c1).sum()), lost=int((g.c0 & ~g.c1).sum()),
                          rank0_p1=int(g.p0.sum()), rank0_tt=int(g.p1.sum()),
                          r0_gained=int((~g.p0 & g.p1).sum()), r0_lost=int((g.p0 & ~g.p1).sum()))
        res[key] = out
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
