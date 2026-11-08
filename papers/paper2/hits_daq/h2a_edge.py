#!/usr/bin/env python3
"""h2a_edge.py '<glob tiers tables>' [--col u_dream] [--json OUT] -- why the binned remap H2a reads shallow on the steepest tracks.

H2a (tiers.cv_fit) is a map tan = f(u) built as the median of tan_ref in quantile bins of u, flat outside the outermost
bin medians (np.interp).  Three things to separate, all cross-validated by event-id parity as in tiers.py:
  clamp      fraction of planes per |tan_ref| bin whose u lies beyond the map's outermost node (f is flat there)
  forward    the map as built: median(tan_ref | u).  A regression of the truth on a noisy measurement: it shrinks
             toward the training sample's angle distribution, more where that distribution falls fast
  inverse    the map of the other direction: median(u | tan_ref) in quantile bins of tan_ref, made monotone and inverted
             with linear extrapolation beyond the end nodes.  Unbiased against the truth at fixed angle, no prior
  nbins      forward with 8, 20, 40 bins: if the shallow bias were bin statistics it would change with the bin count
Scores: tiers.score (s68 within 5 deg, steep = median d sign(tan_ref), per |tan| bin).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiers import score, TBINS  # noqa: E402


def nodes(x, y, m, nb):
    q = np.quantile(x[m], np.linspace(0, 1, nb + 1))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        b = m & (x >= lo) & (x < hi)
        if b.sum() > 3:
            xs.append(np.median(x[b]))
            ys.append(np.median(y[b]))
    return np.array(xs), np.array(ys)


def forward(u, tr, eid, nb=20):
    pred, clamp = np.full(len(u), np.nan), np.zeros(len(u), bool)
    ok = np.isfinite(u) & np.isfinite(tr)
    for h in (0, 1):
        trn, te = ok & (eid % 2 != h), ok & (eid % 2 == h)
        xs, ys = nodes(u, tr, trn, nb)
        pred[te] = np.interp(u[te], xs, ys)
        clamp[te] = (u[te] < xs[0]) | (u[te] > xs[-1])
    return pred, clamp


def inverse(u, tr, eid, nb=20):
    """u = g(tan): median u per tan_ref quantile bin, made monotone, inverted; slope of the end segment outside"""
    pred = np.full(len(u), np.nan)
    ok = np.isfinite(u) & np.isfinite(tr)
    for h in (0, 1):
        trn, te = ok & (eid % 2 != h), ok & (eid % 2 == h)
        ts, us = nodes(tr, u, trn, nb)                  # x = tan_ref, y = median u
        us = np.maximum.accumulate(us)
        keep = np.r_[True, np.diff(us) > 1e-6]
        ts, us = ts[keep], us[keep]
        p = np.interp(u[te], us, ts)
        lo, hi = u[te] < us[0], u[te] > us[-1]
        p[lo] = ts[0] + (u[te][lo] - us[0]) * (ts[1] - ts[0]) / (us[1] - us[0])
        p[hi] = ts[-1] + (u[te][hi] - us[-1]) * (ts[-1] - ts[-2]) / (us[-1] - us[-2])
        pred[te] = p
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tables')
    ap.add_argument('--col', default='u_dream')
    ap.add_argument('--json')
    a = ap.parse_args()
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(a.tables))], ignore_index=True)
    T = T[~T.duplicated(['event_id', 'view'], keep=False)]
    out = {}
    for view, g in T.groupby('view'):
        u, tr, eid = g[a.col].to_numpy(float), g.tan_ref.to_numpy(float), g.event_id.to_numpy()
        f20, cl = forward(u, tr, eid)
        arms = {'forward20': f20, 'forward8': forward(u, tr, eid, 8)[0], 'forward40': forward(u, tr, eid, 40)[0],
                'inverse20': inverse(u, tr, eid)}
        S = {k: score(v, tr) for k, v in arms.items()}
        S['W'] = score(g.tan_W.to_numpy(float), tr)
        bins = [f'{lo}-{hi}' for lo, hi in zip(TBINS[:-1], TBINS[1:])]
        clamp = {b: float(cl[(np.abs(tr) >= float(b.split('-')[0])) & (np.abs(tr) < float(b.split('-')[1])) & np.isfinite(f20)].mean()) for b in bins}
        out[view] = dict(n=len(g), clamp=clamp, scores=S)
        print(f'[{view}] n {len(g)}  {a.col}   all: ' + '  '.join(f'{k} {S[k]["all"]["s68"]:.2f}' for k in S))
        print('   bin        n   clamp | steep ' + ' '.join(f'{k:>10s}' for k in S))
        for b in bins:
            print(f'   {b:8s} {S["forward20"][b]["n"]:5d}  {clamp[b]:.2f}  |       ' +
                  ' '.join(f'{(S[k][b]["steep"] if S[k][b]["steep"] is not None else np.nan):+10.2f}' for k in S))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
