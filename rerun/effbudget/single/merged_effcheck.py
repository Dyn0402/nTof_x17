#!/usr/bin/env python3
"""merged_effcheck.py --root R --rereco DIR [--oneview ...] [--out results/merged_effcheck.json]
-- does the 'charge' split corroboration move the det3 single-track efficiency? (MERGED_PROJECTION_2026-11-02.md §4d)

The 'charge' rule can act only on events whose candidate counts differ between views (it needs fewer candidates in the
merged view than in the other).  merged_rereco.py re-reconstructed EVERY such event of both det3 golden rows (plus the
219 one-view events and 600 random events per row), under 'cost' (the bench baseline) and 'charge'.  Per event: the
multi-track credit of single_track.py / A1 (a candidate within R_MM of the M3 ray in both views) and the rank-0
('production') detection, before and after, split by single_track_v2's one-view class.  Events outside the re-reconstructed
set cannot change, so the transitions here are the whole effect on the efficiency numerators.
"""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'p2_det3': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
R_MM = 5.0


def det(C, B):
    C = C[C['rank'] >= 0].copy()
    C = C[C.event_id.isin(B.index)]
    C['d'] = (C.p0 - np.where(C.plane == 'x', B.mesh_x.reindex(C.event_id).to_numpy(),
                               B.mesh_y.reindex(C.event_id).to_numpy())).abs()
    a = C.groupby(['event_id', 'plane']).d.min().unstack()
    r0 = C[C['rank'] == 0].groupby(['event_id', 'plane']).d.min().unstack()
    return ((a.x < R_MM) & (a.y < R_MM)), ((r0.x < R_MM) & (r0.y < R_MM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--rereco', required=True)
    ap.add_argument('--oneview', default=os.path.join(HERE, 'results', 'oneview_events.parquet'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'merged_effcheck.json'))
    a = ap.parse_args()
    O = pd.read_parquet(a.oneview)
    res = {}
    for key, rel in ROWS.items():
        B = pd.read_parquet(os.path.join(a.root, 'x17rerun/paper1_eff/budget', rel, 'rays_det3.parquet')
                            ).drop_duplicates('event_id').set_index('event_id')
        B = B[B.region == 'interior']
        cr, pr = {}, {}
        for v in ('cost', 'charge'):
            C = pd.read_parquet(os.path.join(a.rereco, key, v, 'events.candidates.parquet'))
            cr[v], pr[v] = det(C, B)
        ix = cr['cost'].index.union(cr['charge'].index)
        cls = O[O.row == key].set_index('event_id').ov_cls.reindex(ix).fillna('other')
        T = pd.DataFrame({'c0': cr['cost'].reindex(ix).fillna(False), 'c1': cr['charge'].reindex(ix).fillna(False),
                          'p0': pr['cost'].reindex(ix).fillna(False), 'p1': pr['charge'].reindex(ix).fillna(False),
                          'cls': cls})
        out = {'interior_events_rereco': int(len(T))}
        for c, g in T.groupby('cls'):
            out[c] = dict(n=int(len(g)), credit_cost=int(g.c0.sum()), credit_charge=int(g.c1.sum()),
                          credit_gained=int((~g.c0 & g.c1).sum()), credit_lost=int((g.c0 & ~g.c1).sum()),
                          rank0_cost=int(g.p0.sum()), rank0_charge=int(g.p1.sum()),
                          rank0_gained=int((~g.p0 & g.p1).sum()), rank0_lost=int((g.p0 & ~g.p1).sum()))
        res[key] = out
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
