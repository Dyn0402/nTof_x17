#!/usr/bin/env python3
"""sps_outliers.py <sps window pickle(s)> --bundle B --json OUT [--nmax N] -- the hits the robust ladder fit rejects on the
SPS inclined view, data against the pure model (sps_resim.py 'noise' mode, arms 'base' = the pass kernel and 'gas' = the
Magboltz diffusion).  Every leading-edge hit is placed in the track's own frame: dx = its strip position minus the
fitted track at the hit's time (in pitches; 0 = on the ladder), and the hit's time after t0.  A hit is an outlier when it
is more than 180 ns (3 samples) off the plane's robust ladder line.  Reported per row and source: hits and outliers per
plane, the dx distribution of the outliers (|dx| <= 1.5, 1.5-4, > 4 pitches), their side (ahead of / behind the track
along the drift) and their relative amplitude."""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import emulate as E  # noqa: E402
import sps_resim as SRS  # noqa: E402
from resim import PAD, padded  # noqa: E402

PITCH = 0.78


def hits_frame(W, nz, pos, p):
    h = E.dream(W, nz)
    t, a = h['t'], h['a']
    k = np.isfinite(t) & (a > 0)
    if k.sum() < 5:
        return None
    x, y, wt = pos[k], t[k], a[k]
    keep = np.ones(len(x), bool)
    for _ in range(4):                                   # the robust line of tiers.ladder_u, keeping the mask
        xm, ym = np.average(x[keep], weights=wt[keep]), np.average(y[keep], weights=wt[keep])
        s = np.sum(wt[keep] * (x[keep] - xm) * (y[keep] - ym)) / np.sum(wt[keep] * (x[keep] - xm) ** 2)
        r = y - (ym + s * (x - xm))
        sig = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= 3 * max(sig, 10.0)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    out = np.abs(r) > 180.0
    c = np.median((y - (p['t0'] + (x - p['p0']) / p['w']))[~out]) if (~out).sum() else 0.0   # leading-edge offset
    u = y - p['t0'] - c                                   # drift-time coordinate of the hit
    dx = (x - (p['p0'] + p['w'] * u)) / PITCH * np.sign(p['w'])     # > 0: on the cathode (late) side of the track
    return dict(n=len(x), out=out, dx=dx, u=u, a=wt / wt.max())


def summarise(recs):
    n = np.array([r['n'] for r in recs])
    out = np.concatenate([r['out'] for r in recs])
    dx = np.concatenate([r['dx'] for r in recs])
    a = np.concatenate([r['a'] for r in recs])
    u = np.concatenate([r['u'] for r in recs])
    o = dict(planes=len(recs), hits_per_plane=float(n.mean()), outliers_per_plane=float(out.sum() / len(recs)))
    for name, m in (('near', np.abs(dx) <= 1.5), ('mid', (np.abs(dx) > 1.5) & (np.abs(dx) <= 4)), ('far', np.abs(dx) > 4)):
        o[f'out_{name}_per_plane'] = float((out & m).sum() / len(recs))
    o['out_cathode_side_frac'] = float(np.mean(dx[out] > 0))
    o['out_rel_amp_med'] = float(np.median(a[out]))
    o['in_rel_amp_med'] = float(np.median(a[~out]))
    o['out_u_q'] = [float(x) for x in np.percentile(u[out], [10, 50, 90])]
    o['out_absdx_q'] = [float(x) for x in np.percentile(np.abs(dx[out]), [25, 50, 75, 90])]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--nmax', type=int, default=1500)
    a = ap.parse_args()
    res = {}
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = D['meta']
        sel = [p for p in D['planes'] if p['view'] == 'y' and p.get('q') is not None][:a.nmax]
        SRS.init(a.bundle, meta, len(sel[0]['q']), 'base:,gas:Dp=0.006', 5.0)
        wm, V = SRS.G['wm'], SRS.G['V']
        recs = dict(data=[], base=[], gas=[])
        for p in sel:
            rng = np.random.default_rng(p['event_id'] * 2 + 1)
            pos2, nz2, g2 = padded(p, rng)
            ns = p['W'].shape[1]
            Wd = np.vstack([np.zeros((PAD, ns)), p['W'].astype(float), np.zeros((PAD, ns))])
            N2 = rng.normal(0, 1, (len(pos2), ns)) * nz2[:, None]
            Z2 = Wd == 0
            src = dict(data=Wd)
            for name, hy in V.items():
                M2 = wm.model_waveforms('y', pos2, p['p0'], p['w'], p['t0'], p['q'].astype(float), hy) * g2[:, None]
                src[name] = np.where(Z2, 0.0, M2 + N2)       # the data's zero suppression applied to the model
            for name, W2 in src.items():
                r = hits_frame(W2, nz2, pos2, p)
                if r is not None:
                    recs[name].append(r)
        res[meta['key']] = {k: summarise(v) for k, v in recs.items()}
        print('==', meta['key'])
        for k, o in res[meta['key']].items():
            print(f"   {k:5s} hits/plane {o['hits_per_plane']:.1f}  outliers/plane {o['outliers_per_plane']:.2f}  "
                  f"(|dx|<=1.5: {o['out_near_per_plane']:.2f}, 1.5-4: {o['out_mid_per_plane']:.2f}, >4: {o['out_far_per_plane']:.2f})  "
                  f"cathode side {o['out_cathode_side_frac']:.2f}  rel amp out/in {o['out_rel_amp_med']:.2f}/{o['in_rel_amp_med']:.2f}  "
                  f"|dx| q25/50/75/90 {np.round(o['out_absdx_q'], 1)}  u q10/50/90 {np.round(o['out_u_q'], 0)}", flush=True)
    json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
