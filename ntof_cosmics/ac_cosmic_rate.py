#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ac_cosmic_rate.py -- is the observed rate of A-C through-going cosmics what the
muon flux predicts?

At EAR2 the neutron beam is vertical (global y), so the four arms lie in the
horizontal plane and an A-C through-goer is near-horizontal (the joined lines
sit 74-79 deg from vertical).  Monte Carlo: muon directions from the
Chirkin-corrected I0 cos^2(theta*) zenith distribution (I0 = 70 /m^2/s/sr,
E > ~1 GeV, open sky -- no building or bunker shielding), lines uniform on a
disk perpendicular to each direction.  Accepted = crosses A's and C's measured
active area (398.6 x 362 mm, `common/mx17_active_area.py`), passes the gate's
|tan| < 0.6 in both local planes of both chambers, and fires (wall AND
plastic) on A or C -- the hardware trigger.  Geometry from run_149's
run_config.json.

    python ntof_cosmics/ac_cosmic_rate.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

CFG = Path('/media/dylan/data/x17/beam_july/runs/run_149/run_config.json')
I0 = 70.0                       # m^-2 s^-1 sr^-1, vertical integral intensity
CHIRKIN = (0.102573, -0.068287, 0.958633, 0.0407253, 0.817285)
ACT_U, ACT_V = 398.6 / 2, (18.0 - 199.29, 379.9 - 199.29)
WALL_U, WALL_V = (-225.0, 175.0), 250.0      # structure frame (ana.WALL_EDGES), bar half-length
PLAS_HALF_U, PLAS_HALF_V = 100.0, 150.0
TAN_MAX = 0.6
R_DISK = 900.0                  # mm
ZEN_BINS = np.arange(50, 90.01, 2.5)
OUT = HERE / 'results' / 'ac_rate'


def cos_star(c):
    p1, p2, p3, p4, p5 = CHIRKIN
    return np.sqrt((c ** 2 + p1 ** 2 + p2 * c ** p3 + p4 * c ** p5) / (1 + p1 ** 2 + p2 + p4))


def geometry():
    det = {d['name']: d for d in json.loads(CFG.read_text())['detectors']}
    g = {}
    for arm, sgn in (('A', 1), ('C', -1)):
        c = det[f'mx17_{arm}']['det_center_coords']
        plas = [det[f'plastic_{arm}_{s}']['det_center_coords'] for s in 'LR']
        wall = det[f'sipm_{arm}_01']['det_center_coords']
        g[arm] = dict(sgn=sgn, x0=c['x'], z0=c['z'], wall_z=wall['z'], plas_z=plas[0]['z'],
                      plas_x=[p['x'] for p in plas])
    return g


def cross_z(P, D, z):
    s = (z - P[:, 2]) / D[:, 2]
    return P[:, 0] + s * D[:, 0], P[:, 1] + s * D[:, 1]


def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def simulate(n=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    g = geometry()
    # downward hemisphere, uniform in solid angle; y is vertical
    c = rng.uniform(1e-4, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - c ** 2)
    D = np.stack([s * np.cos(phi), -c, s * np.sin(phi)], 1)
    # a point on the disk perpendicular to D through the origin
    a = np.cross(D, np.array([1.0, 0, 0]))
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(D, a)
    r = R_DISK * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    P = (r * np.cos(t))[:, None] * a + (r * np.sin(t))[:, None] * b
    w = I0 * cos_star(c) ** 2 * (2 * np.pi) * (np.pi * (R_DISK / 1e3) ** 2) / n   # Hz per sample
    ok = np.abs(D[:, 2]) > 1e-6
    hit = {}
    for arm, G in g.items():
        x, y = cross_z(P, D, G['z0'])
        u = G['sgn'] * (x - G['x0'])            # C is rotated 180 deg about y
        act = (np.abs(u) < ACT_U) & (y > ACT_V[0]) & (y < ACT_V[1])
        tx, ty = D[:, 0] / D[:, 2], D[:, 1] / D[:, 2]
        gate = (np.abs(tx) < TAN_MAX) & (np.abs(ty) < TAN_MAX)
        xw, yw = cross_z(P, D, G['wall_z'])
        uw = G['sgn'] * xw
        wall = (uw > WALL_U[0]) & (uw < WALL_U[1]) & (np.abs(yw) < WALL_V)
        xp, yp = cross_z(P, D, G['plas_z'])
        plas = np.zeros(n, bool)
        for px in G['plas_x']:
            plas |= (np.abs(xp - px) < PLAS_HALF_U) & (np.abs(yp) < PLAS_HALF_V)
        hit[arm] = dict(act=act & ok, gate=gate, trig=wall & plas & ok)
    both = hit['A']['act'] & hit['C']['act']
    trig = hit['A']['trig'] | hit['C']['trig']
    gated = both & hit['A']['gate'] & hit['C']['gate']
    zen = np.degrees(np.arccos(c))
    acc = gated & trig
    hz, _ = np.histogram(zen[acc], bins=ZEN_BINS, weights=w[acc])
    return dict(zen_hist_hz=hz.tolist(),rate_both=float(w[both].sum()), rate_both_trig=float(w[both & trig].sum()),
                rate_gated_trig=float(w[gated & trig].sum()),
                zen_median=_wmedian(zen[gated & trig], w[gated & trig]),
                n_acc=int((gated & trig).sum()),
                trigA_rate=float(w[hit['A']['trig']].sum()))


def observed():
    from ntof_cosmics.inbeam_through_goers import ac_pairs, COLS
    T = pd.concat(pd.read_parquet(f, columns=[c for c in COLS if c not in ('run', 't_since_flash_ns')])
                  for f in glob.glob(str(HERE / 'results' / 'tracking' / 'k_run_147' / 'tracks_run_149_*.parquet')))
    C = ac_pairs(T)
    inv = pd.read_csv(HERE / 'results' / 'cosmic_subruns.csv')
    sec = float(inv[inv.run == 149].seconds.sum())
    ho, _ = np.histogram(C[C.sep < 60].vert_deg, bins=ZEN_BINS)
    return dict(zen_hist=ho.tolist(), hours=sec / 3600, n_pairs=len(C), n_sep60=int((C.sep < 60).sum()),
                rate_pairs_hz=len(C) / sec, rate_sep60_hz=float((C.sep < 60).sum() / sec),
                zen_median=float(np.median(C[C.sep < 60].vert_deg)))


def main() -> int:
    sims = [simulate(seed=s) for s in (1, 2)]
    m = {k: (np.mean([x[k] for x in sims], axis=0).tolist() if k == 'zen_hist_hz'
             else float(np.mean([x[k] for x in sims]))) for k in sims[0]}
    o = observed()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(dict(expected=m, observed=o, zen_bins=ZEN_BINS.tolist(),
                                                      I0=I0), indent=1))
    print('expected (open-sky flux, 100 % chamber efficiency):')
    for k, v in m.items():
        if isinstance(v, list):
            continue
        print(f'  {k:16s} {v:.4g}' + ('  /h ' + f'{v * 3600:.0f}' if 'rate' in k else ''))
    print('observed run_149:', json.dumps({k: round(v, 4) for k, v in o.items() if not isinstance(v, list)}))
    print(f'implied eps_A x eps_C (gated, sep<60) = {o["rate_sep60_hz"] / m["rate_gated_trig"]:.2f}; '
          f'(all A-C single pairs) = {o["rate_pairs_hz"] / m["rate_gated_trig"]:.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
