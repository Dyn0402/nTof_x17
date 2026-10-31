#!/usr/bin/env python3
"""bench_yscan.py <dir> [--out results/bench_yscan.json] -- bench_gap.py's estimator on the Y view of the Y-inclined, X-flat
caches (bench_ycache.py: <dir>/<name>_y.pkl): det3 (sat_det3) and det2 (o22) validate, det4 (g_det4, 6-23 long / longer)
measures, per row and pooled, at three peak thresholds.  Errors: bootstrap of the slope (L against |tan_y|) and of the
median L/|tan_y|."""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_gap as B                                        # noqa: E402
import bench_scan as S                                       # noqa: E402
import gap_geom as G                                         # noqa: E402

ROWS = [('sat_det3', ['sat_det3']), ('o22_det2', ['o22_det2']), ('g_det4', ['g_det4']),
        ('d4_623long', ['d4_623long']), ('d4_623longer', ['d4_623longer']),
        ('det4_all', ['g_det4', 'd4_623long', 'd4_623longer'])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                                                                       'bench_yscan.json'))
    a = ap.parse_args()
    cache = {}
    B.VIEW = 'y'
    out = []
    for name, parts in ROWS:
        evs = []
        for p in parts:
            if p not in cache:
                cache[p] = list(pickle.load(open(os.path.join(a.dir, f'{p}_y.pkl'), 'rb')).values())
            evs += cache[p]
        for pl, pm in S.THRESH:
            G.PK_LINE, G.PK_MIN = pl, pm
            R = []
            for e in evs:
                t = abs(e['tan_y'])
                if B.TAN_LO <= t <= B.TAN_HI:
                    r = B.extent(e, 0.63, np.asarray(e['y']['W']).shape[1])
                    if r:
                        R.append((t, r[0]))
            row = dict(row=name, parts=parts, pk_line=pl, pk_min=pm, n_events=len(evs), n=len(R))
            if len(R) >= 20:
                R = np.array(R); s, ds, c, n = S.slope_fit(R[:, 0], R[:, 1])
                ratio = R[:, 1] / R[:, 0]
                row.update(gap_slope=s, gap_slope_err=ds, intercept_mm=c, n_used=n, gap_ratio_median=float(np.median(ratio)),
                           gap_ratio_err=G.boot_median(ratio))
            print(json.dumps(row), flush=True)
            out.append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
