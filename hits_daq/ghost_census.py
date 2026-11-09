#!/usr/bin/env python3
"""ghost_census.py <window pickles> --out OUT.json -- how many hit strips lie beyond the track's column ends, data against the
pass model, per |tan| bin.

A strip whose centre is more than half a pitch outside the track's projection [x(mesh), x(cathode)] collects no charge from
the track itself; it fires only through lateral spreading (footprint, diffusion, RC), same-time coupling, electronics
cross-talk or noise.  These 'ghost' strips sit at the column ends and rotate the single-time ladder (the strip-end term).
Counted here with the V0 emulator on the data window W, on the noiseless pass model M, and on M + white noise at the strip
sigma; the column ends come from the fitted track (p0, w) and the charge profile q (first/last depth bin above 10 % of max).
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emulate as E  # noqa: E402

PITCH = 0.78
TB = (0.03, 0.08, 0.15, 0.25, 0.4, 0.7)
KW = {'x': 1.0368355217021707, 'y': 1.0149061257816572}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    acc = {}
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        v = D['meta']['v_drift'] / 1000.0
        for p in D['planes']:
            view = p['view']
            tan = abs(p['w']) / (KW[view] * v)
            W, M, nz, pos = p['W'].astype(float), p['M'].astype(float), p['noise'].astype(float), p['pos'].astype(float)
            q = p['q'].astype(float)
            nzq = np.flatnonzero(q > 0.1 * q.max()) if q.max() > 0 else []
            if len(nzq) < 3 or len(pos) < 5:
                continue
            ulo, uhi = nzq[0] * 60.0, (nzq[-1] + 1) * 60.0
            xa, xb = sorted((p['p0'] + p['w'] * ulo, p['p0'] + p['w'] * uhi))
            beyond_lo = pos < xa - 0.5 * PITCH          # beyond the end at lower position
            beyond_hi = pos > xb + 0.5 * PITCH
            mesh_side_lo = p['w'] > 0                   # the mesh end is at the lower position when w > 0
            rng = np.random.default_rng(int(p['event_id']))
            arms = {'data': W, 'model': M, 'model+noise': M + rng.normal(size=M.shape) * nz[:, None]}
            ib = int(np.digitize(tan, TB)) - 1
            if ib < 0 or ib >= len(TB) - 1:
                continue
            for name, X in arms.items():
                h = E.dream(X, nz)
                hit = np.isfinite(h['t'])
                g_lo, g_hi = int((hit & beyond_lo).sum()), int((hit & beyond_hi).sum())
                g_mesh, g_cath = (g_lo, g_hi) if mesh_side_lo else (g_hi, g_lo)
                far = int((hit & ((pos < xa - 1.5 * PITCH) | (pos > xb + 1.5 * PITCH))).sum())
                key = (view, ib, name)
                r = acc.setdefault(key, dict(n=0, hits=0, ghost=0, g_mesh=0, g_cath=0, far=0, any_ghost=0))
                r['n'] += 1
                r['hits'] += int(hit.sum())
                r['ghost'] += g_lo + g_hi
                r['g_mesh'] += g_mesh
                r['g_cath'] += g_cath
                r['far'] += far
                r['any_ghost'] += int((g_lo + g_hi) > 0)
    out = {}
    for (view, ib, name), r in sorted(acc.items()):
        o = dict(view=view, tan_lo=TB[ib], tan_hi=TB[ib + 1], arm=name, n_planes=r['n'], hits_per_plane=r['hits'] / r['n'],
                 ghosts_per_plane=r['ghost'] / r['n'], ghost_mesh=r['g_mesh'] / r['n'], ghost_cathode=r['g_cath'] / r['n'],
                 far_per_plane=r['far'] / r['n'], frac_planes_with_ghost=r['any_ghost'] / r['n'])
        out[f'{view}|{TB[ib]}-{TB[ib + 1]}|{name}'] = o
        print(f"{view} |tan| {TB[ib]:.2f}-{TB[ib + 1]:.2f} {name:12s} n {r['n']:5d}  hits/plane {o['hits_per_plane']:5.2f}  ghosts/plane {o['ghosts_per_plane']:.3f} (mesh {o['ghost_mesh']:.3f}, cathode {o['ghost_cathode']:.3f}; >1.5 pitch {o['far_per_plane']:.3f})  planes with ghost {o['frac_planes_with_ghost']:.2f}")
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
