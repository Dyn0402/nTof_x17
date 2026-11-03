#!/usr/bin/env python3
"""fiducial_variants.py [--root R --analysis A --eff E] -- Paper I §5 'Area' / §8 edge: det3 efficiency over three areas,
from the chamber A golden rows (Runs 1+2 pooled). Writes results/fiducial_variants.json (the file had no producer before
2026-11-03; its old entries mixed two single-track definitions, see below).

Areas (ray footprint distance to the active-area edge, budget dist_x / dist_y, mm):
  full area            every measured ray inside the active area (region != outside)
  interior 30x/8y      region == 'interior' (>= 30 mm from the X edges, >= 8 mm from the Y edges; the sample of the angles)
  30 mm all sides      dist_x >= 30 and dist_y >= 30
Per area:
  production pick      rank-0 candidate within 5 mm in both views (budget cat_xy_5 == 'detected')
  credit (all rays)    some candidate within 5 mm in both views (det_any_5)
  single-track         THE QUOTED ONE: single_track_area.py's definition, nothing excused (two-particle classes removed from
                       the sample, every other ray kept, losses included); computed in all three areas
  single-track v2      single_track_v2.py's sample (independent / merged / vetoed two-particle events removed, POS_TAIL and
                       EDGE classes out): needs the waveform A1 classes, so defined only inside the interior
Run after single_track.py, single_track_v2.py and single_track_area.py (reads their per-event files in results/).
"""
import argparse, json, os
import numpy as np
import pandas as pd
import side_tree
from single_track_area import ROWS, classify, RES
from single_track_v2 import CREDIT

SIDE_MM = 30.0


def binom(k, n):
    p = k / n
    return dict(n=int(n), k=int(k), eff=round(100 * p, 2), err=round(100 * np.sqrt(p * (1 - p) / n), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('COSMIC_ROOT', '/eos/experiment/ntof/data/x17/cosmic_bench'))
    ap.add_argument('--analysis', default=side_tree.PASS1)
    ap.add_argument('--eff', default=side_tree.EFF)
    ap.add_argument('--res', default=RES, help='directory with the per-event parquets of the single_track*.py scripts')
    a = ap.parse_args()
    acc = {}
    def add(area, nm, n, k):
        v = acc.setdefault(f'{area} | {nm}', [0, 0]); v[0] += int(n); v[1] += int(k)
    ov = pd.read_parquet(os.path.join(a.res, 'oneview_events.parquet'))
    for key, rel in ROWS.items():
        B, cls, _ = classify(a.root, rel, a.analysis, a.eff)
        ev = pd.read_parquet(os.path.join(a.res, f'single_track_area_events_{key}.parquet'))
        assert (ev.index == B.index).all()
        det = (B.cat_xy_5 == 'detected').to_numpy(); credit = ev.credit.to_numpy()
        single = ev.cls.eq('single').to_numpy()          # incl. the spark-vetoed muon+other class of single_track_area.py
        areas = {'full area': np.ones(len(B), bool), 'interior 30x/8y': (B.region == 'interior').to_numpy(),
                 '30 mm from every side': ((B.dist_x >= SIDE_MM) & (B.dist_y >= SIDE_MM)).to_numpy()}
        D = pd.read_parquet(os.path.join(a.res, f'single_track_events_{key}.parquet'))
        o = ov[ov.row == key].set_index('event_id').ov_cls.reindex(D.index)
        drop = D.recat.isin(['far:POS_TAIL', 'far:EDGE'])
        v2 = ~D.indep_2nd & ~o.eq('merged in projection') & ~D.recat.eq('spark:muon+other') & ~drop
        v2cr = D.recat.isin(CREDIT)
        for ar, m in areas.items():
            add(ar, 'production pick', m.sum(), (m & det).sum())
            add(ar, 'credit (all rays)', m.sum(), (m & credit).sum())
            add(ar, 'single-track (nothing excused)', (m & single).sum(), (m & single & credit).sum())
            add(ar, 'two-particle events', (m & ~single).sum(), (m & ~single & credit).sum())
            if ar != 'full area':
                mv = pd.Series(m, index=B.index).reindex(D.index).fillna(False).to_numpy()
                add(ar, 'credit, ref/geom out', (mv & ~drop).sum(), (mv & ~drop & v2cr).sum())
                add(ar, 'single-track v2', (mv & v2).sum(), (mv & v2 & v2cr).sum())
    out = {nm: binom(k, n) for nm, (n, k) in acc.items()}
    json.dump(out, open(os.path.join(a.res, 'fiducial_variants.json'), 'w'), indent=1)
    for nm, v in out.items():
        print(f"{nm:60s} n={v['n']:6d}  {v['eff']:6.2f} +- {v['err']:.2f}")


if __name__ == '__main__':
    main()
