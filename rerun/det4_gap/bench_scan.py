#!/usr/bin/env python3
"""bench_scan.py <dir with <key>_incl.pkl> [--out results/bench_scan.json] -- bench_gap.py's estimator over views, chambers and
peak thresholds: det3 is the validation (gap_extent 28.0-28.2 mm), det4 the measurement.  The bench Y view needs lower
thresholds than SPS (det4 is gain-limited; an inclined column spreads its charge), so each case runs at (PK_LINE, PK_MIN)
= (150, 60), (90, 40), (60, 30) ADC.  Y view: |tan_x| < 0.05 (one x, so det4's x-interval dead stripes do not cut it)."""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_gap as B                                        # noqa: E402
import gap_geom as G                                         # noqa: E402

CASES = [('sat_det3', 'x', None), ('sat_det3', 'y', 0.05), ('g_det4', 'y', 0.05), ('g_det4', 'x', None)]
THRESH = [(150.0, 60.0), (90.0, 40.0), (60.0, 30.0)]


def slope_fit(t, L, seed=3):
    A = np.vstack([t, np.ones(len(t))]).T; ok = np.ones(len(t), bool)
    for _ in range(5):
        c = np.linalg.lstsq(A[ok], L[ok], rcond=None)[0]; r = L - A @ c
        mad = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))); ok = np.abs(r) < 3 * mad
    rng = np.random.default_rng(seed); idx = np.flatnonzero(ok)
    bs = [np.linalg.lstsq(A[j], L[j], rcond=None)[0][0] for j in (rng.choice(idx, len(idx)) for _ in range(200))]
    return float(c[0]), float(np.std(bs)), float(c[1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                                                                       'bench_scan.json'))
    a = ap.parse_args()
    out = []
    cache = {}
    for key, view, cm in CASES:
        if key not in cache:
            cache[key] = pickle.load(open(os.path.join(a.dir, f'{key}_incl.pkl'), 'rb'))
        D = cache[key]; o = 'y' if view == 'x' else 'x'
        ev = [e for e in D.values() if B.X_LO <= e['ref_mesh_x'] <= B.X_HI and B.Y_LO <= e['ref_mesh_y'] <= B.Y_HI
              and (cm is None or abs(e[f'tan_{o}']) < cm)]
        B.VIEW = view
        for pl, pm in THRESH:
            G.PK_LINE, G.PK_MIN = pl, pm
            R = []
            for e in ev:
                t = abs(e[f'tan_{view}'])
                if B.TAN_LO <= t <= B.TAN_HI:
                    r = B.extent(e, 0.63, np.asarray(e[view]['W']).shape[1])
                    if r:
                        R.append((t, r[0]))
            row = dict(key=key, view=view, cross_max=cm, pk_line=pl, pk_min=pm, n=len(R))
            if len(R) >= 20:
                R = np.array(R); s, ds, c, n = slope_fit(R[:, 0], R[:, 1])
                row.update(gap_slope=s, gap_slope_err=ds, intercept_mm=c, n_used=n, gap_ratio_median=float(np.median(R[:, 1] / R[:, 0])))
            print(json.dumps(row), flush=True)
            out.append(row)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
