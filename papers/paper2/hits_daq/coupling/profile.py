#!/usr/bin/env python3
"""profile.py <window pickles> --out OUT.parquet --arms name:sigma:Dp:rc:k1:k2,... [--bundle B] -- where along the track hit times shift.

On inclined planes (0.1 < |tan| < 0.45), V0 hits (emulate.dream) on the data W and on each arm's model (white noise at each strip's
sigma, charge profile by NNLS on the data with that arm's kernel).  For every strip with a hit:
    x  = fractional position along the track projection: (pos - p0) / (w T), T = gap / v: 0 = mesh end, 1 = cathode end
    dt = t_hit - (t0 + (pos - p0) / w)       [ns]: the hit time against the time the track is above that strip
A coupling acts on both sides of an interior strip and so moves interior hits together; only the track ends see it one-sidedly. Diffusion
grows with depth.  Also returns the ladder slope ratio u/w per arm.
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))]
import emulate as E  # noqa: E402
from observables import kernel, padded  # noqa: E402
from tiers import ladder_u  # noqa: E402

G = {}
GAP, V = 28.08, 35.8e-3


def one(p):
    wm = G['wm']
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    tan = (p['w'] * 1e3 - G['w0'][v]) / (G['kw'][v] * 35.8)
    if not 0.1 < abs(tan) < 0.45 or len(pos) < 5:
        return []
    p0, w, t0 = float(p['p0']), float(p['w']), float(p['t0'])
    T = GAP / V
    pos2, g2 = padded(pos, g)
    sat = p['censor'].astype(bool)
    rng = np.random.default_rng(int(p['event_id']) * 2 + (v == 'y'))
    rows = []

    def add(name, Y):
        h = E.dream(Y, nz, nsig=5.0)
        k = np.isfinite(h['t'])
        u, n = ladder_u(pos, h['t'], h['a'])
        for i in np.flatnonzero(k):
            x = (pos[i] - p0) / (w * T)
            if -0.5 < x < 1.5:
                rows.append((v, name, round(float(tan), 3), float(x), float(h['t'][i] - (t0 + (pos[i] - p0) / w)), u / w))
    add('data', W)
    for name, (s0, dp, rc, k1, k2) in G['arms'].items():
        hy = dict(G['hyper'], sigma_p0=s0, sigma_p0_x=s0, sigma_p0_y=s0, Dp=dp, rc_D_y=rc)
        B = wm.build_matrix(v, pos2, p0, w, t0, hy)
        B = B.reshape(len(pos2), -1, B.shape[1]) * g2[:, None, None]
        B = kernel(B, k1, k2)[4:-4]
        Br = B.reshape(-1, B.shape[2])
        chi, q = wm._solve_nnls(Br, W, nz, sat, True)
        if q is None:
            continue
        add(name, (Br @ q).reshape(W.shape) + rng.normal(size=W.shape) * nz[:, None])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--jobs', type=int, default=14)
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    arms = {x.split(':')[0]: tuple(float(y) for y in x.split(':')[1:]) for x in a.arms.split(',')}
    G.update(wm=wm, hyper=dict(cal.hyper), arms=arms, w0=meta['w0'], kw=meta['kw'])
    with mp.Pool(a.jobs) as P:
        R = [r for rr in P.imap_unordered(one, planes, chunksize=16) for r in rr]
    T = pd.DataFrame(R, columns=['view', 'arm', 'tan', 'x', 'dt', 'uratio'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    T.to_parquet(a.out)
    print(f'[profile] {len(planes)} planes, {len(T)} hits -> {a.out}')


if __name__ == '__main__':
    main()
