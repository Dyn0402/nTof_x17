#!/usr/bin/env python3
"""observables.py <window pickles> --out DIR [--bundle B] [--jobs N] [--arms name:s0:k1:k2,...] -- model-free signatures of coupling.

Per near-normal plane (|w| < WMAX: the track moves < 0.3 mm in 240 ns), from the event's own waveforms:
  - the early-charge (mesh-end) centroid and p0 for the periodicity test: charge imaged from the 0.80 mm resistive strips is
    pinned to that grid, a 0.78 mm readout alone is not;
  - neighbour waveforms relative to the centre strip's peak, sample by sample from t0 (grid of 60 ns samples), for near (+1) and
    far (-1, -2) strips, with the same quantity for model arms (design matrix with NNLS q on the same plane);
  - per-strip early charge for the linearity test.
All planes (no angle cut): the pre-signal noise correlation between strips by distance.
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

WMAX = 1.2e-3
PAD = 4
GRID = np.arange(-3, 19)          # samples (60 ns) from t0
G = {}


def kernel(B, k1, k2):
    if k1 == 0 and k2 == 0:
        return B
    C = B.copy()
    C[1:] += k1 * B[:-1]
    C[:-1] += k1 * B[1:]
    C[2:] += k2 * B[:-2]
    C[:-2] += k2 * B[2:]
    return C


def padded(pos, g):
    pitch = np.median(np.diff(pos))
    return (np.r_[pos[0] - pitch * np.arange(PAD, 0, -1), pos, pos[-1] + pitch * np.arange(1, PAD + 1)],
            np.r_[np.full(PAD, np.median(g)), g, np.full(PAD, np.median(g))])


def design(wm, v, pos2, g2, p, hyper, k1, k2):
    B = wm.build_matrix(v, pos2, p['p0'], p['w'], p['t0'], hyper)
    B = B.reshape(len(pos2), -1, B.shape[1]) * g2[:, None, None]
    B = kernel(B, k1, k2)[PAD:-PAD]
    return B.reshape(-1, B.shape[2])


def resample(y, t0):
    """y on the 60 ns sample grid -> values at (t - t0) / 60 = GRID"""
    ts = np.arange(len(y)) * E.SNS
    return np.interp(t0 + GRID * E.SNS, ts, y, left=np.nan, right=np.nan)


def one(p):
    wm = G['wm']
    v = p['view']
    W, nz, g, pos = (p[k].astype(float) for k in ('W', 'noise', 'gain', 'pos'))
    t0 = float(p['t0'])
    ts = np.arange(W.shape[1]) * E.SNS
    noise_rows = []
    pre = ts < t0 - 150.0
    if pre.sum() >= 3 and len(pos) >= 5:
        X = W[:, pre] / nz[:, None]
        for d in range(1, min(6, len(pos))):
            noise_rows.append((v, d, float(np.mean(X[:-d] * X[d:])), float(np.mean(X * X))))
    out = dict(noise=noise_rows)
    if abs(p['w']) > WMAX or len(pos) < 7:
        return out
    pitch = np.median(np.diff(pos))
    pc = p['p0'] + p['w'] * 90.0
    c = int(np.argmin(np.abs(pos - pc)))
    if c < 2 or c >= len(pos) - 2:
        return out
    phi = (pc - pos[c]) / pitch
    sgn = 1 if phi >= 0 else -1
    idx = dict(near=c + sgn, far=c - sgn, near2=c + 2 * sgn, far2=c - 2 * sgn)
    if any(not 0 <= i < len(pos) for i in idx.values()):
        return out
    ek = (ts >= t0 - 60) & (ts <= t0 + 180)
    Q = W[:, ek].sum(1)
    sat = p['censor'].astype(bool)
    S = list(idx.values()) + [c]
    A = float(np.max(W[c]))
    if A < 100 or sat[S][:, ek].any():
        return out
    k3 = np.abs(np.arange(len(pos)) - c) <= 2
    qc = np.maximum(Q[k3], 0)
    cen = float(np.sum(qc * pos[k3]) / np.sum(qc)) if np.sum(qc) > 0 else np.nan
    r = dict(event_id=int(p['event_id']), view=v, phi=abs(phi), sphi=phi, p0=float(p['p0']), pc=float(pc), cen=cen, c_pos=float(pos[c]),
             Qc=float(Q[c]), A=A, tan=float(p['w']) * 1e3 / 35.8, **{f'Q_{k}': float(Q[i] / Q[c]) for k, i in idx.items()})
    cent = resample(W[c], t0)
    wf = {'data': {k: resample(W[i], t0) / A for k, i in idx.items()}}
    wf['data']['centre'] = cent / A
    pos2, g2 = padded(pos, g)
    for name, (s0, k1, k2, ov) in G['arms'].items():
        hy = dict(G['hyper'], sigma_p0=s0, sigma_p0_x=s0, sigma_p0_y=s0, **ov)
        B = design(wm, v, pos2, g2, p, hy, k1, k2)
        chi, q = wm._solve_nnls(B, W, nz, sat, True)
        if q is None:
            continue
        M = (B @ q).reshape(W.shape)
        Am = np.max(M[c]) if np.max(M[c]) > 0 else np.nan
        wf[name] = {k: resample(M[i], t0) / Am for k, i in idx.items()}
        wf[name]['centre'] = resample(M[c], t0) / Am
    out.update(row=r, wf=wf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bundle')
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--arms', default='pass:0.3:0:0,bare:0.07:0:0,k1:0.07:0.06:0,k1k2:0.07:0.06:0.015')
    a = ap.parse_args()
    from wft import model as wm
    from wft.calib import CalibrationBundle
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    cal = CalibrationBundle.load(a.bundle or E.bundle_path(meta))
    wm.use_calibration(cal)
    wm.set_depth_bins(len(planes[0]['q']))
    arms = {}
    for x in a.arms.split(','):
        f = x.split(':')
        arms[f[0]] = (float(f[1]), float(f[2]), float(f[3]), dict(Dp=float(f[4])) if len(f) > 4 else {})
    G.update(wm=wm, hyper=dict(cal.hyper), arms=arms)
    os.makedirs(a.out, exist_ok=True)
    with mp.Pool(a.jobs) as P:
        R = list(P.imap_unordered(one, planes, chunksize=16))
    rows, stacks, noise = [], {}, []
    for o in R:
        noise += o['noise']
        if 'row' not in o:
            continue
        rows.append(o['row'])
        v, phibin = o['row']['view'], min(int(o['row']['phi'] * 5), 2)        # |phase| 0-0.2, 0.2-0.4, 0.4+
        for name, W in o['wf'].items():
            for strip, y in W.items():
                stacks.setdefault((v, phibin, name, strip), []).append(y)
    T = pd.DataFrame(rows)
    T.to_parquet(os.path.join(a.out, 'near_normal.parquet'))
    pd.DataFrame(noise, columns=['view', 'd', 'cov', 'var']).to_parquet(os.path.join(a.out, 'noise.parquet'))
    out = {}
    for k, ys in stacks.items():
        Y = np.array(ys)
        out['|'.join(map(str, k))] = dict(n=len(Y), med=np.nanmedian(Y, 0), err=1.253 * np.nanstd(Y, 0) / np.sqrt(np.sum(np.isfinite(Y), 0)))
    pickle.dump(dict(grid=GRID, stacks=out), open(os.path.join(a.out, 'stacks.pkl'), 'wb'))
    print(f'[obs] {len(planes)} planes, {len(T)} near-normal -> {a.out}')


if __name__ == '__main__':
    main()
