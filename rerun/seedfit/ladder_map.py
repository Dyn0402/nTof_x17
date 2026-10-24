#!/usr/bin/env python3
"""ladder_map.py <det> <v_timing> -- the pivot fit's ladder v (robust slope of w on tan_ref) in 3x3 regions of the
chamber (M3 position at the mesh), per view, from sf_<det>_400.json + sp_<det>_rest.json (1300 held-out events)."""
import json, os, pickle, sys
import numpy as np
L = os.path.expanduser('~/x17/cosmic_bench/vladder/')
det, vt = sys.argv[1], float(sys.argv[2])
ev = pickle.load(open(L + det + '/cache.pkl', 'rb'))
R = {}
for f in (f'sf_{det}_400.json', f'sp_{det}_rest.json'):
    R.update(json.load(open(L + f))['events'])


def ladder(tr, w, nb=100):
    m = np.isfinite(w)
    if m.sum() < 25:
        return np.nan, np.nan, int(m.sum())
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27)
    rng = np.random.default_rng(0); idx = np.flatnonzero(m)
    e = np.std([np.polyfit(tr[j], w[j], 1)[0] for j in (rng.choice(idx, len(idx)) for _ in range(nb))])
    return s, e, int(m.sum())


edges = [0, 133, 266, 400]
out = {}
for p in ('x', 'y'):
    ks = [k for k in R if p in R[k] and 'pivot' in R[k][p]]
    tr = np.array([R[k][p]['tr'] for k in ks]); w = np.array([R[k][p]['pivot']['w'] for k in ks])
    xr = np.array([ev[int(k)]['ref_mesh_x'] for k in ks]); yr = np.array([ev[int(k)]['ref_mesh_y'] for k in ks])
    s, e, n = ladder(tr, w)
    print(f'{det} {p}-view  all: {s:.2f} ± {e:.2f}  ({100 * (s / vt - 1):+.1f}% vs timing {vt}), n {n}')
    grid = []
    for iy in (2, 1, 0):
        line = f'   y {edges[iy]:3d}-{edges[iy + 1]:3d} '
        for ix in range(3):
            m = (xr >= edges[ix]) & (xr < edges[ix + 1]) & (yr >= edges[iy]) & (yr < edges[iy + 1])
            s2, e2, n2 = ladder(tr[m], w[m])
            grid.append(dict(ix=ix, iy=iy, v=s2, err=e2, n=n2))
            line += f'| {s2:5.1f}±{e2:3.1f} (n{n2:3d}) ' if np.isfinite(s2) else '|      .              '
        print(line)
    vs = np.array([g['v'] for g in grid if np.isfinite(g['v'])]); es = np.array([g['err'] for g in grid if np.isfinite(g['v'])])
    wm_ = np.sum(vs / es ** 2) / np.sum(1 / es ** 2)
    print(f'   chi2 of the 9 regions about their mean: {np.sum(((vs - wm_) / es) ** 2):.1f} / {len(vs) - 1}   spread (max-min) {vs.max() - vs.min():.1f}')
    out[p] = dict(all=dict(v=s, err=e, n=n), grid=grid)
json.dump(out, open(L + f'ladder_map_{det}.json', 'w'), indent=1, default=float)
