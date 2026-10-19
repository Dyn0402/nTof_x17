#!/usr/bin/env python3
"""
optical_transport.py -- the EAR2 neutron beam at the measuring station, from
n_TOF's own phase-4 FLUKA phase space and collimation, by the method of the
n_TOF optical transport code (simul/transport/transport.cc, Vlachoudis et al.).

Why this exists
---------------
Until 2026-10-14 every beam footprint in this repo (the Geant4 neutron gun,
sept26_prelim_analysis/capsule_pose.py) took the beam's radius from
`Lambda2D` in lamda2DvsEn_EAR2.root.  That histogram is the transport code's
HIST_LAMBDA output, lambda = (t*v - L)*100 cm, the moderation-length
resolution function -- not a radius.  See ear2_beam/README.md.

Method (as transport.cc)
------------------------
Each FLUKA neutron leaving the target-moderator towards EAR2 (the EAR2.bin
dumps: x, y, z in cm, y vertical; offsets -x -10.8 -z 9.86 put the beam axis
at 0) is treated as a point source.  End points Q are thrown uniformly over a
disk of radius RMAX at each scoring height L; the straight ray P -> Q is kept
if it passes every collimator aperture (circles at heights D_k, centres
(cx, cz), radii R_k, from simul/transport/input/*.inp).  The particle's own
direction is not used (transport.cc does the same: it only uses the source
point).  Weights are per unit area at L, so shapes and interception fractions
are absolute within a height; the overall normalisation is not used here --
the flux comes from the evaluated Phase-4 flux.

Outputs, per collimation variant, energy band and height L:
  profile_<variant>.npz : 2D histograms (1 mm bins, +-RMAX) and mean ray slopes
  summary_<variant>.csv : centroid, FWHM (x, z), r50/r90, BIF of centred disks,
                          local divergence

    python -m ear2_beam.optical_transport --ps DIR [--variant july2021] [--n-end 400]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('vx', '<f4'), ('vy', '<f4'), ('vz', '<f4'),
               ('E', '<f4'), ('t', '<f4'), ('w', '<f4'), ('ij', 'i1'), ('pad', 'V3')])
#: EAR2 transport offsets (README_instructions: "-D Y -t16 -x -10.8 -y37.2 -z 9.86"), cm
X0, Z0 = -10.8, 9.86
#: collimation, simul/transport/input: (cx cm, cz cm, height from target centre m, radius cm)
COLLIMATION = {
    # ear2_july2021.inp: the post-upgrade line, small second collimator (21.8 mm exit)
    'july2021': [(0, 0, 7.40, 10.0), (0, 0, 8.40, 10.0), (0, 0, 15.20, 3.5), (0, 0, 17.03, 1.84),
                 (0, 0, 17.63, 1.34), (0, 0, 17.83, 1.26), (0, 0, 17.93, 1.17), (0, 0, 18.03, 1.09),
                 (0, 0, 18.13, 1.09)],
    # ear2_misalign_revised2018.inp: first collimator found at (-1.4, -1.1) cm by the 2017 survey
    'survey2017': [(-1.4, -1.1, 7.54, 10.0), (-1.4, -1.1, 8.54, 10.0), (0, 0, 15.083, 3.5),
                   (0, 0, 16.923, 1.84), (0, 0, 17.523, 1.34), (0, 0, 17.723, 1.26), (0, 0, 17.823, 1.17),
                   (0, 0, 17.923, 1.09), (0, 0, 18.023, 1.09)],
    # ear2_jul21_miss2mm.inp: second-collimator exit 2 mm off in both axes (n_TOF's own sensitivity case)
    'miss2mm': [(0, 0, 7.40, 10.0), (0, 0, 8.40, 10.0), (0, 0, 15.20, 3.5), (0, 0, 17.03, 1.84),
                (0, 0, 17.63, 1.34), (0, 0, 17.83, 1.26), (0, 0, 17.93, 1.17), (0.2, 0.2, 18.03, 1.09),
                (0.2, 0.2, 18.13, 1.09)],
    # ear2_misalign_fission_Michi.inp: the BIG second collimator (60 mm exit), 2017 first-collimator survey
    'fission': [(-1.4, -1.1, 7.40, 10.0), (-1.4, -1.1, 8.40, 10.0), (0, 0, 15.023, 4.825), (0, 0, 18.023, 3.0)],
}
#: energy bands (eV).  'beam' = our 4-12 ms window at 19.5 m (14-124 meV)
BANDS = {'cold': (1e-3, 1.4e-2), 'beam': (1.4e-2, 0.124), 'thermal_25': (0.02, 0.03),
         'epi': (1.0, 1e3), 'fast': (1e5, 1e7)}
HEIGHTS = (18.5, 19.0, 19.5, 19.75, 19.95, 20.25, 20.5, 21.0)
RMAX = 4.5      # cm (the big collimator needs --rmax 8)
NB = 90         # 1 mm bins over +-RMAX
BIF_D = (1.5, 2.0, 3.0, 4.0)


def load(ps_dir: str) -> np.ndarray:
    fs = sorted(glob.glob(f'{ps_dir}/**/EAR2.bin', recursive=True))
    d = np.concatenate([np.fromfile(f, DT) for f in fs])
    d = d[d['ij'] == 8]
    print(f'{len(fs)} files, {len(d):,} neutrons', flush=True)
    return d


def transport(src: np.ndarray, coll, L: float, n_end: int, rng, chunk: int = 4000):
    """Accepted rays at height L (m): end x, z (cm), ray slopes dx/dy, dz/dy, weight."""
    px = (src['x'] - X0).astype(np.float64)
    pz = (src['z'] - Z0).astype(np.float64)
    py = src['y'].astype(np.float64) / 100.0             # m
    w = src['w'].astype(np.float64)
    C = np.array(coll, float)
    keep = C[:, 2] < L
    C = C[keep]
    out = []
    for k in range(0, len(px), chunk):
        sl = slice(k, k + chunk)
        n = len(px[sl])
        r = RMAX * np.sqrt(rng.random((n, n_end)))
        a = 2 * np.pi * rng.random((n, n_end))
        qx, qz = r * np.cos(a), r * np.sin(a)
        ok = np.ones((n, n_end), bool)
        for cx, cz, D, R in C:
            f = ((D - py[sl]) / (L - py[sl]))[:, None]
            ax = px[sl][:, None] + f * (qx - px[sl][:, None]) - cx
            az = pz[sl][:, None] + f * (qz - pz[sl][:, None]) - cz
            ok &= ax * ax + az * az <= R * R
            if not ok.any():
                break
        i, j = np.nonzero(ok)
        if len(i) == 0:
            continue
        span = (L - py[sl][i]) * 100.0                  # cm
        out.append(np.stack([qx[i, j], qz[i, j], (qx[i, j] - px[sl][i]) / span,
                             (qz[i, j] - pz[sl][i]) / span, w[sl][i]], 1))
    return np.concatenate(out) if out else np.zeros((0, 5))


def fwhm(edges, h):
    c = 0.5 * (edges[1:] + edges[:-1])
    m = h.max()
    above = np.nonzero(h >= m / 2)[0]
    if len(above) == 0:
        return np.nan
    lo, hi = above[0], above[-1]
    # linear interpolation on the outer edges
    xl = np.interp(m / 2, [h[lo - 1], h[lo]], [c[lo - 1], c[lo]]) if lo > 0 else c[lo]
    xh = np.interp(m / 2, [h[hi + 1], h[hi]], [c[hi + 1], c[hi]]) if hi < len(h) - 1 else c[hi]
    return xh - xl


def summarise(R: np.ndarray) -> dict:
    x, z, sx, sz, w = R.T
    W = w.sum()
    mx, mz = (w * x).sum() / W, (w * z).sum() / W
    e = np.linspace(-RMAX, RMAX, NB + 1)
    hx, _ = np.histogram(x, e, weights=w)
    hz, _ = np.histogram(z, e, weights=w)
    r = np.hypot(x - mx, z - mz)
    o = np.argsort(r)
    c = np.cumsum(w[o]) / W
    s = dict(n_rays=len(w), cx_mm=10 * mx, cz_mm=10 * mz,
             sx_mm=10 * np.sqrt((w * (x - mx) ** 2).sum() / W), sz_mm=10 * np.sqrt((w * (z - mz) ** 2).sum() / W),
             fwhm_x_mm=10 * fwhm(e, hx), fwhm_z_mm=10 * fwhm(e, hz),
             r50_mm=10 * r[o][np.searchsorted(c, 0.5)], r90_mm=10 * r[o][np.searchsorted(c, 0.9)],
             r99_mm=10 * r[o][min(np.searchsorted(c, 0.99), len(c) - 1)])
    for d in BIF_D:
        s[f'bif_{d:g}cm'] = float(w[r <= d / 2].sum() / W)
    # divergence: mean slope against position (a cone from an apex), and the local spread
    s['slope_x_per_cm_mrad'] = 1e3 * np.polyfit(x, sx, 1, w=w)[0]
    s['local_spread_mrad'] = 1e3 * np.sqrt(np.average((sx - np.polyval(np.polyfit(x, sx, 1, w=w), x)) ** 2, weights=w))
    return s


BAND_COLS = ('band', 'e_lo', 'e_hi', 'slope_x_per_cm_mrad', 'local_spread_mrad', 'r50_mm', 'cx_mm', 'cz_mm')


def write_bands(f, T: pd.DataFrame) -> None:
    """The band table as a classic TTree (a dict assignment makes an RNTuple,
    which the LCG ROOT the Geant4 build links cannot read)."""
    types = {c: (np.int64 if c == 'band' else np.float64) for c in BAND_COLS}
    f.mktree('bands', types)
    f['bands'].extend({c: T[c].to_numpy().astype(types[c]) for c in BAND_COLS})


def table(d: np.ndarray, variant: str, L: float, n_end: int, out: Path, per_decade: int = 4) -> Path:
    """Energy-binned 2D profiles at one height, for an event generator: a ROOT
    file with one TH2D per log-E band (x, z in cm, beam axis at 0), the band
    edges, and per band the cone slope (mrad/cm: ray angle = slope * position)
    and the local angular spread (mrad)."""
    import uproot
    E = d['E'].astype(np.float64) * 1e9
    edges = np.logspace(-3, 9, 12 * per_decade + 1)
    e2 = np.linspace(-RMAX, RMAX, NB + 1)
    rng = np.random.default_rng(2)
    rows, H = [], {}
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        src = d[(E >= lo) & (E < hi)]
        if len(src) > 200_000:                      # plenty: subsample the source points
            src = src[rng.choice(len(src), 200_000, replace=False)]
        R = transport(src, COLLIMATION[variant], L, n_end, rng) if len(src) else np.zeros((0, 5))
        if len(R) < 200:
            continue
        h, _, _ = np.histogram2d(R[:, 0], R[:, 1], [e2, e2], weights=R[:, 4])
        H[f'prof_{k:03d}'] = (h, e2, e2)
        s = summarise(R)
        rows.append(dict(band=k, e_lo=lo, e_hi=hi, n_src=len(src), **s))
        print(k, f'{lo:.3g}-{hi:.3g} eV', round(s['r50_mm'], 2), round(s['bif_2cm'], 3), flush=True)
    T = pd.DataFrame(rows)
    path = out / f'ear2_profile_{variant}_L{L:g}.root'
    with uproot.recreate(path) as f:
        for k, v in H.items():
            f[k] = v
        write_bands(f, T)
    T.to_csv(out / f'ear2_profile_{variant}_L{L:g}.csv', index=False)
    print('wrote', path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ps', required=True, help='directory tree holding EAR2.bin phase-space dumps')
    ap.add_argument('--variant', default='july2021', choices=sorted(COLLIMATION))
    ap.add_argument('--bands', default=','.join(BANDS))
    ap.add_argument('--heights', default=','.join(map(str, HEIGHTS)))
    ap.add_argument('--n-end', type=int, default=400)
    ap.add_argument('--table', type=float, default=None, metavar='L',
                    help='write the energy-binned profile table at height L (m) instead')
    ap.add_argument('--rmax', type=float, default=4.5, help='scoring radius, cm (1 mm bins)')
    ap.add_argument('--out', type=Path, default=Path('/media/dylan/data/x17/ear2_beam'))
    a = ap.parse_args()
    globals()['RMAX'], globals()['NB'] = a.rmax, int(round(20 * a.rmax))
    a.out.mkdir(parents=True, exist_ok=True)
    d = load(a.ps)
    if a.table is not None:
        table(d, a.variant, a.table, a.n_end, a.out)
        return 0
    E = d['E'].astype(np.float64) * 1e9
    rows, hists = [], {}
    rng = np.random.default_rng(1)
    for band in a.bands.split(','):
        lo, hi = BANDS[band]
        src = d[(E >= lo) & (E < hi)]
        for L in map(float, a.heights.split(',')):
            R = transport(src, COLLIMATION[a.variant], L, a.n_end, rng)
            if len(R) < 100:
                print(band, L, 'too few rays', len(R))
                continue
            s = dict(variant=a.variant, band=band, e_lo=lo, e_hi=hi, L=L, n_src=len(src), **summarise(R))
            rows.append(s)
            e = np.linspace(-RMAX, RMAX, NB + 1)
            h, _, _ = np.histogram2d(R[:, 0], R[:, 1], [e, e], weights=R[:, 4])
            hists[f'{band}_{L:g}'] = h
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    pd.DataFrame(rows).to_csv(a.out / f'summary_{a.variant}.csv', index=False)
    np.savez_compressed(a.out / f'profile_{a.variant}.npz', edges_cm=np.linspace(-RMAX, RMAX, NB + 1), **hists)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
