#!/usr/bin/env python3
"""amplitude.py --bench-rows DIR --rows-csv results/rows.csv --sps DIR --bundle-sps B --out results/amplitude.json
-- the column-start time resolution against the signal size, bench and SPS on one axis (Paper I §7).

x axis: the EARLY amplitude, i.e. the strip sum S at the edge estimator's cut sample (the first sample above 25 % of the
maximum), summed over X and Y [ADC].  It is the signal the rising edge is timed on.  A timing jitter set by noise goes as
noise / slope ~ 1 / early amplitude.
Bench: timing_row.py's per-trigger times of every row with a clean trigger (cov(start, end) consistent with zero: not the
6-23 / 6-24 runs) and not skeptical, per chamber.  SPS: the same estimator on the sps_edge.py strip sums (det4, g_det4
template), ZS rows and RAW run71 separately.
Per bin: s68(t_X - t_Y)/sqrt 2 (sigma_xy) and, bench only, s68 of the event time against the trigger (sigma_evt).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimators as E  # noqa: E402
from timing_row import s68  # noqa: E402

EDGES = np.geomspace(30, 60000, 23)
BAD_RUNS = ('mx17_det3_det4_overnight_6-23-26', 'mx17_det4_day_6-24-26')


def binned(amp, tx, ty, with_evt=True):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (amp >= lo) & (amp < hi) & np.isfinite(tx) & np.isfinite(ty)
        if s.sum() < 80:
            continue
        r = dict(amp=float(np.median(amp[s])), n=int(s.sum()), sigma_xy=s68(tx[s] - ty[s]) / np.sqrt(2))
        if with_evt:
            r['sigma_evt'] = s68((tx[s] + ty[s]) / 2)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench-rows', required=True); ap.add_argument('--rows-csv', required=True)
    ap.add_argument('--sps', required=True); ap.add_argument('--bundle-sps', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = pd.read_csv(a.rows_csv)
    rows = rows[rows.skip.isna() & (rows.cls != 'skeptical') & ~rows.run.isin(BAD_RUNS)]
    res = {'bench': {}, 'sps': {}}
    for det, g in rows.groupby('det'):
        A, X, Y = [], [], []
        for k in g.key:
            f = os.path.join(a.bench_rows, f'{k}.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            # each row's own median time is removed (rows differ in delay by the template and the drift field)
            tx, ty = z['tx'].astype(float), z['ty'].astype(float)
            m = np.nanmedian((tx + ty) / 2)
            A.append(z['amp']); X.append(tx - m); Y.append(ty - m)
        if A:
            A, X, Y = map(np.concatenate, (A, X, Y))
            res['bench'][det] = dict(rows=int(len(g)), bins=binned(A, X, Y))
            print(det, len(g), 'rows', len(A), 'triggers')
    b = np.load(os.path.join(a.bundle_sps, 'arrays.npz'))
    kern = E.kernel(b['grid'], b['tmpl_x'])
    E.EDGE_FRAC = 0.25
    for tag, pat in (('zs', 'run[56]*.npz'), ('raw', 'run71*.npz')):
        A, X, Y = [], [], []
        for f in sorted(glob.glob(os.path.join(a.sps, pat))):
            z = np.load(f)
            ok = z['x_ok'] & z['y_ok']
            ns = z['x_S'].shape[1]; ed = E.Edge(kern, ns)
            t, early = {}, {}
            for p in 'xy':
                S = z[f'{p}_S'][ok].astype(float)
                t[p] = np.array([ed.fit(s) for s in S])
                early[p] = np.array([s[: int(np.argmax(s > 0.25 * s.max())) + 1].max() if s.max() > 0 else np.nan for s in S])
            m = np.nanmedian((t['x'] + t['y']) / 2)
            A.append(early['x'] + early['y']); X.append(t['x'] - m); Y.append(t['y'] - m)
        if A:
            A, X, Y = map(np.concatenate, (A, X, Y))
            res['sps'][tag] = dict(files=len(glob.glob(os.path.join(a.sps, pat))), bins=binned(A, X, Y, with_evt=True))
            print('sps', tag, len(A), 'events')
    json.dump(res, open(a.out, 'w'), indent=1)
    for grp in res:
        for k, v in res[grp].items():
            print(grp, k, ' '.join(f'[{r["amp"]:.0f}: {r["sigma_xy"]:.0f}]' for r in v['bins']))


if __name__ == '__main__':
    main()
