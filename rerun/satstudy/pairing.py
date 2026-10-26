#!/usr/bin/env python3
"""pairing.py -- x<->y pairing of two tracks in one chamber by what the two views share (Dylan 10-26: "a very important
result for the reconstruction paper ... where does the 96 % come from, and is there a toy / simulation / SPS check?").

    ../../.venv/bin/python pairing.py toy        # data-driven toy from real single tracks -> pairing_toy.json
    ../../.venv/bin/python pairing.py real       # the real two-track events (M3 truth on one track) -> pairing_real.json
    (overlay.py: the overlay bench with truth on both tracks)

The rule (as wft.reco.xy_pair_cost): for two tracks with x candidates (a, b) and y candidates (c, d), take the
assignment with the lower summed cost, cost(x, y) = sum_f min(((f_x - f_y) - median_f)^2 / rsig_f^2, 25), features
  lq   log(q_sum_x / q_sum_y)   one particle leaves the same charge in both views (both read one avalanche)
  u50, u90                     difference of the charge-arrival quantiles: the particle's depth profile
  t0                           only for real events (in the toy every track has its own trigger)
median_f and rsig_f (robust sigma) from clean single tracks. Calibration and test are disjoint halves (odd / even
event id) in the toy.

Toy: every random pair of two different clean single tracks (production rank-0 fits, both views, detected) is a
two-track event at infinite separation, with each track's own x and y as truth. It measures the information in the
features alone: no merging, no shared strips, no common-mode coupling. The overlay bench adds those.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
import multitrack as MT  # noqa: E402

FEATS = dict(q=['lq'], q_u=['lq', 'u50', 'u90'], u=['u50', 'u90'])
N_PAIRS = 400_000


def singles():
    D = pd.read_parquet(os.path.join(S.OUT, 'multitrack_det3.parquet'))
    C = pd.read_parquet(MT.CAND)
    ids = D.index[(D.cls == 'S') & D['prod']]
    w = C[C.event_id.isin(ids) & (C['rank'] == 0)].pivot(index='event_id', columns='plane',
                                                          values=['q_sum', 'q_u50', 'q_u90', 't0'])
    return w.dropna()


def diffs(qx, qy, ux50, uy50, ux90, uy90, t0x=None, t0y=None):
    d = dict(lq=np.log(qx / qy), u50=ux50 - uy50, u90=ux90 - uy90)
    if t0x is not None:
        d['t0'] = t0x - t0y
    return d


def calib(w):
    d = diffs(w.q_sum.x, w.q_sum.y, w.q_u50.x, w.q_u50.y, w.q_u90.x, w.q_u90.y, w.t0.x, w.t0.y)
    return {k: (float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v))))) for k, v in d.items()}


def cost(d, cal, keys):
    return sum(np.minimum(((d[k] - cal[k][0]) / cal[k][1]) ** 2, 25.0) for k in keys)


def toy():
    w = singles(); odd = w.index % 2 == 1
    cal = calib(w[odd]); T = w[~odd]
    rng = np.random.default_rng(1)
    i = rng.integers(0, len(T), N_PAIRS); j = rng.integers(0, len(T), N_PAIRS); k = i != j; i, j = i[k], j[k]
    A, B = T.iloc[i], T.iloc[j]
    g = lambda X, Y: diffs(X.q_sum.x.to_numpy(), Y.q_sum.y.to_numpy(), X.q_u50.x.to_numpy(), Y.q_u50.y.to_numpy(),
                           X.q_u90.x.to_numpy(), Y.q_u90.y.to_numpy())
    straight, swap = (g(A, A), g(B, B)), (g(A, B), g(B, A))
    lqr = np.abs(np.log(np.sqrt(A.q_sum.x.to_numpy() * A.q_sum.y.to_numpy()) /
                        np.sqrt(B.q_sum.x.to_numpy() * B.q_sum.y.to_numpy())))
    out = dict(n_singles=int(len(w)), n_test=int(len(T)), n_pairs=int(len(i)), cal=cal, right={}, by_ratio={})
    edges = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 10]
    for nm, ks in FEATS.items():
        ok = (cost(straight[0], cal, ks) + cost(straight[1], cal, ks)) < (cost(swap[0], cal, ks) + cost(swap[1], cal, ks))
        out['right'][nm] = float(ok.mean())
        b = np.digitize(lqr, edges) - 1
        out['by_ratio'][nm] = [dict(lo=edges[q], hi=edges[q + 1], n=int((b == q).sum()), right=float(ok[b == q].mean()))
                               for q in range(len(edges) - 1)]
    out['ratio_hist'] = np.histogram(lqr, bins=edges)[0].tolist()
    # an analytic cross-check: lq alone, Gaussian x/y ratio of width s, tracks with log-charge ratio r -> swap prob
    s = cal['lq'][1]
    out['lq_sigma'] = s
    json.dump(out, open(os.path.join(S.OUT, 'pairing_toy.json'), 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('n_singles', 'n_pairs', 'right', 'lq_sigma')}, indent=1))
    for nm in FEATS:
        print(nm, [(r['lo'], r['hi'], r['n'], round(r['right'], 3)) for r in out['by_ratio'][nm]])


if __name__ == '__main__':
    {'toy': toy}[sys.argv[1]]()
