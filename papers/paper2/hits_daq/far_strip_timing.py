#!/usr/bin/env python3
"""far_strip_timing.py <boot pickle> --out X.parquet [--part i/n] | --summary <glob> [--json OUT]

Data-model hit timing as a function of a strip's lateral distance from the track (Paper II diagnostic, HANDOFF step 2).
The H2c near-normal Y excess was traced (2026-11-06, ad hoc) to strips 3-4 pitches out that fire 105-150 ns EARLIER
in the data W than in the pass model M, with amplitudes within ~10 %.  This makes that measurement reproducible and
asks: is it on X too (X has no resistive RC spreading in the model), and does it depend on the track angle?  An early,
amplitude-correct signal far from the track is what a prompt capacitive coupling would do (the literature's
neighbour cross-talk, Flierl 15 %), while the model brings far charge in late through the RC kernel.

Per plane and per strip hit in BOTH W and M (V0-like hit, p0_xbias.hits_v0):
  d    = signed lateral distance from the track, in pitches, to the nearest point of the track's projection
         [p0, p0 + w T] (0 inside the projection); sign + = beyond the cathode end
  dt   = t_le(W) - t_le(M)   [ns]        (also the peak time)
  ra   = a(W) / a(M)
The pass model's own fit (p0, w, t0, q) defines the track, so this is a data-model comparison, not a truth one.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
sys.path.insert(0, HERE)

from p0_xbias import hits_v0  # noqa: E402

PITCH = 398.58 / 512
TANB = (0.0, 0.05, 0.15, 0.3, 0.6)
DB = (0, 1, 2, 3, 4, 5, 6, 8)


def run(a):
    import pandas as pd
    D = pickle.load(open(a.pickle, 'rb'))
    v = D['meta']['v_drift'] / 1000.0
    DT = 60.0                                      # wft.model.DT, ns per depth bin
    ip, npart = (int(x) for x in a.part.split('/'))
    rows = []
    for j, p in enumerate(D['planes']):
        if j % npart != ip:
            continue
        pos, nz = p['pos'].astype(float), p['noise'].astype(float)
        T = len(p['q']) * DT
        lo, hi = sorted((p['p0'], p['p0'] + p['w'] * T))
        dist = np.where(pos < lo, pos - lo, np.where(pos > hi, pos - hi, 0.0)) / PITCH
        if p['w'] < 0:
            dist = -dist                           # + = the cathode end's side
        hW = hits_v0(p['W'].astype(float), nz)
        hM = hits_v0(p['M'].astype(float), nz)
        ok = np.isfinite(hW[:, 0]) & np.isfinite(hM[:, 0])
        tan = float(p['w'] / v)
        for i in np.flatnonzero(ok):
            rows.append((p['view'], int(p.get('event_id', -1)), tan, float(dist[i]), float(hW[i, 0] - hM[i, 0]),
                         float(hW[i, 1] - hM[i, 1]), float(hW[i, 2] / hM[i, 2]), float(hW[i, 2] / nz[i])))
    df = pd.DataFrame(rows, columns=['view', 'event_id', 'tan', 'd', 'dt_le', 'dt_pk', 'ra', 'snr'])
    df.to_parquet(a.out)
    print(f'[far] wrote {a.out}: {len(df)} hits')


def summary(a):
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(a.summary))], ignore_index=True)
    out = {}
    ad = df.d.abs()
    for view in ('x', 'y'):
        for tl, th in zip(TANB[:-1], TANB[1:]):
            g0 = df[(df.view == view) & (df.tan.abs() >= tl) & (df.tan.abs() < th)]
            print(f'{view} |tan| {tl:.2f}-{th:.2f}  planes {g0.event_id.nunique()}')
            for dl, dh in zip(DB[:-1], DB[1:]):
                m = (ad[g0.index] >= dl) & (ad[g0.index] < dh) if dl else ad[g0.index] < 0.5
                g = g0[m]
                if len(g) < 30:
                    continue
                r = dict(n=len(g), dt_le=float(np.median(g.dt_le)), dt_pk=float(np.median(g.dt_pk)),
                         ra=float(np.median(g.ra)), snr=float(np.median(g.snr)))
                out[f'{view}/{tl}-{th}/{dl}-{dh}'] = r
                lab = 'in' if not dl else f'{dl}-{dh}'
                print(f'   |d| {lab:>4s} pitch  n {r["n"]:6d}  dt_le {r["dt_le"]:+7.1f}  dt_pk {r["dt_pk"]:+7.1f} ns'
                      f'  a(W)/a(M) {r["ra"]:.2f}  snr {r["snr"]:5.1f}')
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickle', nargs='?')
    ap.add_argument('--out')
    ap.add_argument('--part', default='0/1')
    ap.add_argument('--summary')
    ap.add_argument('--json')
    a = ap.parse_args()
    summary(a) if a.summary else run(a)


if __name__ == '__main__':
    main()
