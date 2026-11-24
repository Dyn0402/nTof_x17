#!/usr/bin/env python3
"""T5 pointing-map library: frames, the robust residual centre, cell maps, the cosmic map's prediction.

Frames (from map_view.FRAME / capsule_tof.VIEW, checked on the tables): the map acts on GLOBAL slopes.
  A, C: horizontal view x -> slope d_x/d_z, measuring coordinate global x, normal z; vertical d_y/d_z.
  D:    horizontal view x -> slope d_z/d_x, measuring coordinate global z, normal x; vertical d_y/d_x.
Expected horizontal slope of a capsule track: the line from the source (x_s, z_s) to the track's point p0 (strip plane):
  A, C: e = (p0_x - x_s) / (p0_z - z_s);  D: e = (p0_z - z_s) / (p0_x - x_s).
Vertical: e_v = (p0_y - y_c) / (p0_n - n_s), y_c = centre of the source line (-72 .. +15 mm) = -28.5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path.insert(0, str(REPO))

XS, ZS = -10.9, -4.5            # pass1_tracks.SOURCE_XZ_MM
Y_LO, Y_HI = -72.0, 15.0        # source line (README finding 18)
YC = 0.5 * (Y_LO + Y_HI)
FR = {'A': ('x', 'z'), 'C': ('x', 'z'), 'D': ('z', 'x')}
PRE_ACCESS = ('run_79', 'run_81')
MAP_FILE = REPO / 'rerun/ntof/stage3/angle_map/angle_map_r2.json'
LINES = Path('/media/ucla/work/ntof_pass1/ana_r2/track_quality/lines.parquet')
BDLINES = Path('/media/ucla/work/ntof_pass1/ana_r2/track_quality/bd_lines.parquet')
E25 = np.arange(-200.0, 201.0, 25.0)
E50 = np.arange(-200.0, 201.0, 50.0)


def load(ext=False, drop_pre=True):
    d = pd.read_parquet(HERE / ('late_tracks_ext06.parquet' if ext else 'late_tracks.parquet'))
    if drop_pre:
        d = d[~d.run.isin(PRE_ACCESS)]
    d = d[~((d.arm == 'D') & ~d.wall_D)]            # capsule_tof: D needs the wall
    d['rn'] = d.run.str.replace('run_', '').astype(int)
    d['ms'] = d.t_since_flash_ns / 1e6
    return d.reset_index(drop=True)


def arm_arrays(g, arm, xs=XS, zs=ZS, plane_dz=0.0, dist_scale=1.0):
    """Measured (unmapped and mapped) and expected global slopes, horizontal and vertical.

    plane_dz: move the point used for the expected line along the track by plane_dz mm in the normal coordinate
    towards the source (15 = mid-gap instead of the strip plane).  dist_scale: scale the chamber's distance from the
    source (normal coordinate about n_s)."""
    h, n = FR[arm]
    src = {'x': xs, 'z': zs}
    P = {k: g[f'p0_{k}'].to_numpy(float) for k in 'xyz'}
    Dn = {k: g[f'd_{k}_nomap'].to_numpy(float) for k in 'xyz'}
    Dm = {k: g[f'd_{k}'].to_numpy(float) for k in 'xyz'}
    sh, sv = Dn[h] / Dn[n], Dn['y'] / Dn[n]
    shm, svm = Dm[h] / Dm[n], Dm['y'] / Dm[n]
    # point on the track: p0 moved by -plane_dz*sign(p0_n) along the normal
    sg = np.sign(P[n] - src[n])
    dn = -plane_dz * sg
    ph, py, pn = P[h] + sh * dn, P['y'] + sv * dn, P[n] + dn
    L = (pn - src[n]) * dist_scale
    e = (ph - src[h]) / L
    ev = (py - YC) / L
    return dict(x=g.x_local.to_numpy(float), y=g.y_local.to_numpy(float), sh=sh, sv=sv, shm=shm, svm=svm, e=e, ev=ev,
                L=L, q=g.q_total.to_numpy(float), rn=g.rn.to_numpy(), ms=g.ms.to_numpy(float))


def wmedian(r, w=0.06, n_iter=8, m0=None):
    """Centre of the residual core: median within +-w of the current centre, iterated from the median of |r| < 0.3."""
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return np.nan
    m = np.median(r[np.abs(r) < 0.3]) if m0 is None else m0
    for _ in range(n_iter):
        k = np.abs(r - m) < w
        if k.sum() < 20:
            return np.nan
        m2 = np.median(r[k])
        if abs(m2 - m) < 1e-6:
            break
        m = m2
    return float(m)


def centre(r, w=0.06, nsub=8, rng=None):
    """(centre, stat error from nsub random sub-samples, n, core sigma (MAD within the window))."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 200:
        return np.nan, np.nan, n, np.nan
    m = wmedian(r, w)
    rng = rng or np.random.default_rng(1)
    lab = rng.integers(0, nsub, n)
    sub = np.array([wmedian(r[lab == i], w, m0=m) for i in range(nsub)])
    err = float(np.nanstd(sub, ddof=1) / np.sqrt(nsub))
    k = np.abs(r - m) < 0.3
    sig = 1.4826 * np.median(np.abs(r[k] - m))
    return m, err, n, float(sig)


def cell_index(x, y, edges):
    nc = len(edges) - 1
    ix = np.digitize(x, edges) - 1
    iy = np.digitize(y, edges) - 1
    ok = (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nc)
    return ix, iy, ok


def cell_map(x, y, r, edges, extra=None, w=0.06, nmin=300):
    """Per cell [ix, iy]: centre, err, n, sigma, and the cell means of the `extra` arrays."""
    nc = len(edges) - 1
    out = {k: np.full((nc, nc), np.nan) for k in ('c', 'err', 'sig')}
    out['n'] = np.zeros((nc, nc), int)
    for k in (extra or {}):
        out[k] = np.full((nc, nc), np.nan)
    ix, iy, ok = cell_index(x, y, edges)
    key = np.where(ok, ix * nc + iy, -1)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    bounds = np.searchsorted(ks, np.arange(nc * nc + 1))
    rng = np.random.default_rng(3)
    for kk in range(nc * nc):
        sl = order[bounds[kk]:bounds[kk + 1]]
        i, j = divmod(kk, nc)
        out['n'][i, j] = len(sl)
        if len(sl) < nmin:
            continue
        c, e, _, s = centre(r[sl], w, rng=rng)
        out['c'][i, j], out['err'][i, j], out['sig'][i, j] = c, e, s
        for k, v in (extra or {}).items():
            out[k][i, j] = float(np.nanmean(v[sl]))
    return out


# ---------------- the cosmic map ----------------
def cosmic_maps():
    return json.load(open(MAP_FILE))['maps']


def interp(mp, x, y):
    from rerun.ntof.stage3.map_view import interp as I
    return I(mp, x, y)


def clean_lines_df():
    from rerun.ntof.improve import angle_map as AM
    M = pd.read_parquet(LINES)
    return M, AM


def boot_cosmic(key, nboot=30, seed=5, edges=None):
    """Bootstrap replicas of the A/C cosmic map (angle_map.fit_map on resampled clean lines)."""
    from rerun.ntof.improve import angle_map as AM
    M = pd.read_parquet(LINES)
    arm, p = key.split('_')
    G = AM.clean_lines(M, arm.lower(), p)
    rng = np.random.default_rng(seed)
    reps = []
    for b in range(nboot):
        reps.append(AM.fit_map(G.iloc[rng.integers(0, len(G), len(G))]))
    return reps
