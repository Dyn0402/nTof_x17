#!/usr/bin/env python3
"""split_toy.py -- is the 'bright events lose more' trend of headon_split.py a selection effect?

With attachment, a cluster from shallow depth (early) arrives whole and one from
deep (late) arrives attenuated, so selecting events by TOTAL charge picks events
whose big clusters were early: the bright class has a lower late/early ratio
without any charge-dependent physics.  Without attachment the split is
time-symmetric and every class has the same ratio.

Toy head-on track, 30 mm at v = 12.76 um/ns (243 V/cm), Poisson primary
clusters (30 / cm), cluster sizes with a 1/n^2 tail (delta electrons), each
electron surviving exp(-r t), per-event gain scale (lognormal, the det4 gain
map) that cannot bias anything, bench X template as electronics, 60 ns
sampling with a random phase.  Split exactly as the data: total over the window,
5 % trimmed, terciles; R = level(2.4-2.7 us) / level(1.08-1.26 us), with the
drift start at the data's onset.

    split_toy.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMPL = '/home/dylan/.claude/jobs/a61ac7ca/tmp/bench/b3/calib_bundle_r06/arrays.npz'


def run(r, n=30000, seed=0, v=12.76, gap=30.0, onset=720.0, gsig=0.35):
    rng = np.random.default_rng(seed)
    A = np.load(TMPL); tg, tm = A['grid'], A['tmpl_x']
    dt = 10.0
    grid = np.arange(0, 64 * 60 + 1000, dt)
    W = np.zeros((n, 64))
    T = gap * 1e3 / v
    for i in range(n):
        nc = rng.poisson(3.0 * gap)
        z = rng.uniform(0, gap, nc)
        # cluster sizes: 1/n^2 tail truncated at 300
        u = rng.uniform(size=nc)
        size = np.minimum(np.floor(1.0 / (1.0 - u * 0.997)), 300).astype(int)
        t = z * 1e3 / v
        surv = rng.binomial(size, np.exp(-r * t))
        cur = np.bincount(np.clip(((onset + t) / dt).astype(int), 0, len(grid) - 1),
                          weights=surv, minlength=len(grid)).astype(float)
        sig = np.convolve(cur, tm)[:len(grid)]
        ph = rng.uniform(0, 60)
        samp = np.interp(np.arange(64) * 60 + ph, grid + tg[0], sig)
        W[i] = samp * rng.lognormal(0, gsig)
    tot = W[:, 12:58].sum(1)
    lo, hi = np.quantile(tot, [0.05, 0.95]); keep = (tot > lo) & (tot < hi)
    edges = np.quantile(tot[keep], [0, 1 / 3, 2 / 3, 1])
    t = np.arange(64) * 60.0
    out = []
    for c in range(3):
        m = keep & (tot >= edges[c]) & (tot <= edges[c + 1])
        S = W[m].mean(0)
        out.append(float(S[(t >= 2400) & (t < 2700)].mean() / S[(t >= 1080) & (t <= 1260)].mean()))
    S = W[keep].mean(0)
    allR = float(S[(t >= 2400) & (t < 2700)].mean() / S[(t >= 1080) & (t <= 1260)].mean())
    return out, allR


def main():
    res = {}
    for r in (0.0, 1.0e-4, 1.9e-4, 3.0e-4):
        terc, allR = run(r)
        res[str(r)] = dict(terciles=terc, all=allR)
        print(f'r = {r * 1e4:.1f}e-4/ns: R all {allR:.3f};  terciles faint/mid/bright ' +
              ' / '.join(f'{x:.3f}' for x in terc))
    print('data 243 V/cm X: all 0.769; terciles 0.863 / 0.796 / 0.752')
    json.dump(res, open(os.path.join(HERE, 'results', 'split_toy.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
