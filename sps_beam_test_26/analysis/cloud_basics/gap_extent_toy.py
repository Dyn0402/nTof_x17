#!/usr/bin/env python3
"""gap_extent_toy.py -- closure of gap_extent.py's estimator on synthetic inclined tracks.

Tracks at the data's |tan_x| (sampled from a cache), uniform ionisation in clusters (2.8 /mm along the
path) from z = 0 to the true gap G, each electron landing at x_mesh + z tan + N(0, sigma0^2 + 2 Dd z/v)
and arriving at u = z / v (+ longitudinal diffusion), strips at the 0.78 mm pitch, the chamber's X
template, 60 ns sampling with a random phase, Gaussian noise; reference position smeared by SIG_REF.
The same profile() and fit() as the data.  Optional field non-uniformity: the last DEAD_MM before the
cathode collect nothing (to see what a short charge column does to the estimator).

    gap_extent_toy.py [--det det3] [--G 30] [--n 2000]
"""
import argparse
import pickle

import numpy as np

import gap_extent as GE
from bench_attach import template

STEP = 30.0


def gen(n, G, tans, h, v, sig0, Dd, DL, sig_ref, noise, gain, rng, nsamp=32, dead_mm=0.0, t0=-30.0):
    pos = np.arange(-60, 61) * 0.78
    tg = np.arange(len(h)) * STEP
    evs = []
    for i in range(n):
        tn = rng.choice(tans)
        x0 = rng.uniform(-0.39, 0.39)
        L = G * np.sqrt(1 + tn ** 2)
        nc = rng.poisson(2.8 * L)
        z = rng.uniform(0, G, nc)
        ne = 1 + rng.geometric(0.35, nc) - 1
        z = np.repeat(z, ne)
        z = z[z < G - dead_mm]
        u = z / v
        sd = np.sqrt(sig0 ** 2 + 2 * Dd * u)
        x = x0 + z * tn + rng.normal(0, 1, len(z)) * sd
        t = t0 + u + rng.normal(0, 1, len(z)) * np.sqrt(2 * DL * u) / v
        g = rng.gamma(0.5, 2.0, len(z)) * gain
        ts = rng.uniform(0, 60) + np.arange(nsamp) * 60.0
        idx = np.round(x / 0.78).astype(int) + 60
        ok = (idx >= 0) & (idx < len(pos))
        W = np.zeros((len(pos), nsamp))
        for k in np.unique(idx[ok]):
            m = ok & (idx == k)
            dt = ts[None, :] - t[m][:, None]
            W[k] = (g[m][:, None] * np.interp(dt, tg, h, left=0, right=0)).sum(0)
        W += rng.normal(0, noise, W.shape)
        xm = x0 + rng.normal(0, sig_ref)
        a, b = xm + (-8) * tn, xm + 40 * tn
        sel = (pos >= min(a, b) - 5) & (pos <= max(a, b) + 5)
        evs.append({'tan_x': tn, 'ref_mesh_x': xm,
                    'x': dict(W=W[sel], pos=pos[sel], ch=np.arange(len(pos))[sel])})
    return evs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default='det3')
    ap.add_argument('--key', default='sat_det3')
    ap.add_argument('--G', type=float, default=30.0)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--v', type=float, default=0.0383)
    ap.add_argument('--sig-ref', type=float, default=0.4)
    ap.add_argument('--gain', type=float, default=6.0)
    ap.add_argument('--dead-mm', type=float, default=0.0)
    ap.add_argument('--t0', type=float, default=-30.0, help='arrival time of mesh charge [ns]; -30 puts the near-mesh strips\' peak at sample 8, as in the data (the template peaks 510 ns after arrival)')
    a = ap.parse_args()
    h, _ = template(a.det)
    ev = pickle.load(open(f'{GE.OUT}/incl_{a.key}.pkl', 'rb'))
    tans = np.array([e['tan_x'] for e in ev.values() if 'x' in e and 0.12 <= abs(e['tan_x']) <= 0.8])
    rng = np.random.default_rng(11)
    evs = gen(a.n, a.G, tans, h, a.v, 0.35, 4.3e-4, 4e-4, a.sig_ref, 8.0, a.gain, rng, dead_mm=a.dead_mm, t0=a.t0)
    sat = np.mean([e['x']['W'].max() >= GE.SAT for e in evs])
    for q in ('peak', 'int'):
        n, P, E = GE.profile(evs, 'x', set(), qmode=q)
        r = GE.fit(P, E)
        hh = GE.half(P, E)
        bt = []
        for lo, hi in zip(GE.TAN_BINS[:-1], GE.TAN_BINS[1:]):
            nt, Pt, Et = GE.profile(evs, 'x', set(), qmode=q, tbin=(lo, hi))
            if nt > 40:
                bt.append(GE.half(Pt, Et)['gap'])
        print(f'toy G {a.G} dead {a.dead_mm} {q:4s}: n {n} (sat {sat:.2f})  HALF gap {hh["gap"]:.2f} (z0 {hh["z0"]:+.2f})  erf gap {r["gap"]:.2f}  z0 {r["z0"]:+.2f} z1 {r["z1"]:.2f}  '
              f's0 {r["s0"]:.2f} s1 {r["s1"]:.2f} slope {r["slope"]:+.4f}   half by tan ' + ' '.join(f'{g:.2f}' for g in bt), flush=True)


if __name__ == '__main__':
    main()
