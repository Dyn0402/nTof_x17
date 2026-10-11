#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
residual_checks.py -- where does the data/sim capsule-band residual come from?
HANDOFF_TRACKING_2026-10-06.md §13, "Chasing the residual" (2026-10-08).

Data: run_145 stat090_0000 under the in-situ bundles (is2); sim: the digitised
Geant4 beam captures g4_{A,C}_is2_full.  Every check uses compare_data's
selection (gated, pointing coincidence, charge window 25-75 % of the view's own
q_sum).  The band is the robust slope of raw tan against position, with the
foot taken as the sample's own zero crossing, so it needs no capsule foot and
works on y.

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/residual_checks.py [time|miss|yview|charge|source|all]

  time    data band by time since the flash
  miss    x miss at the capsule plane (lever - D tan), data vs sim, and the band in the pointing core
  yview   x and y band, data vs sim, with the same x-pointing cut (|miss| < 40 mm)
  charge  band by charge quintile, data vs sim, x and y
  source  sim band by the TRUE source position at the capsule plane, and vs source width
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from ntof_cosmics.g4_digi import analyse as A
from ntof_cosmics.g4_digi import compare_data as C

warnings.filterwarnings('ignore', category=FutureWarning)
SIM = {'A': 'g4_A_is2_full', 'C': 'g4_C_is2_full'}
MISS_CUT = 40.0


def band(pos, tan, rng, x0=None, nb=100):
    """(band, bootstrap err, n, zero crossing, median response per compare_data bin)."""
    from ntof_tracking import run145_target_imaging as TI
    pos, tan = np.asarray(pos, float), np.asarray(tan, float)
    m = np.isfinite(pos) & np.isfinite(tan) & (np.abs(tan) > 1e-3) & (np.abs(tan) < 1.5)
    pos, tan = pos[m], tan[m]
    if x0 is None:
        s, b = TI._robust_line(pos, tan)
        x0 = -b / s
    lev = pos - x0
    m = (np.abs(lev) > 30) & (np.abs(lev) < 130)
    lev, tan = lev[m], tan[m]
    if len(lev) < 40:
        return np.nan, np.nan, len(lev), x0, [np.nan] * (len(C.E) - 1)
    s, _ = TI._robust_line(lev, tan)
    bs = []
    for _ in range(nb):
        i = rng.integers(0, len(lev), len(lev))
        si, _ = TI._robust_line(lev[i], tan[i])
        if si != 0:
            bs.append(abs((1 / A.D) / si))
    te = lev / A.D
    r = tan * np.sign(te * s) / np.abs(te)
    resp = [float(np.median(r[(np.abs(te) >= lo) & (np.abs(te) < hi)])) for lo, hi in zip(C.E[:-1], C.E[1:])]
    return abs((1 / A.D) / s), float(np.std(bs)), len(lev), x0, resp


def data_sel(t, arm, view='x'):
    g = t[(t.arm == arm) & t.gated & t.coinc_this_arm.astype(bool) & (t[f'{view}_q_sum'] > 0)
          & np.isfinite(t[f'tan_raw_{view}']) & np.isfinite(t.tan_raw_x)]
    lo, hi = np.percentile(g[f'{view}_q_sum'], [25, 75])
    g = g[g[f'{view}_q_sum'].between(lo, hi)]
    return g.assign(xmiss=np.abs(g.x_local - C.PINWHEEL[arm] - A.D * g.tan_raw_x))


def sim_sel(arm, view='x'):
    R = A.load(SIM[arm])
    f = A.foot(R)
    R = R[R.x_ok & R[f'{view}_ok'] & np.isfinite(R.x_tan_theta) & np.isfinite(R[f'{view}_tan_theta'])]
    lo, hi = np.percentile(R[f'{view}_q_sum'], [25, 75])
    R = R[R[f'{view}_q_sum'].between(lo, hi)]
    return R.assign(xmiss=np.abs(R.xl - f - A.D * R.x_tan_theta),
                    src_u=R.u_mesh - A.D * R.tan_u, src_v=R.v_mesh - A.D * R.tan_v)


def sim_pos(R, view):
    return R.xl if view == 'x' else R.y_p0


def fmt(c, e):
    return f'{c:.3f}±{e:.3f}'


def check_time(t, rng):
    print('== data band by time since the flash (x)')
    for arm in 'AC':
        g = data_sel(t, arm)
        ms = g.t_since_flash_ns / 1e6
        out = []
        for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 1e9)):
            m = ((ms >= lo) & (ms < hi)).to_numpy()
            c, e, n, *_ = band(g.x_local[m], g.tan_raw_x[m], rng)
            out.append(f'{lo:g}-{hi:g} ms n={n} {fmt(c, e)}')
        print(f'  {arm}: ' + ' | '.join(out))


def check_miss(t, rng):
    print('== x miss at the capsule plane, and the band in the pointing core')
    for arm in 'AC':
        g, R = data_sel(t, arm), sim_sel(arm)
        for name, d in (('data', g), ('sim reco', R)):
            a = d.xmiss.to_numpy()
            print(f'  {arm} {name:8s} |miss| q25/50/75/90 {np.percentile(a, [25, 50, 75, 90]).round(0)} '
                  f'>50mm {np.mean(a > 50):.3f}')
        for cut in (40, 60, 100):
            row = []
            for name, d, pos, tan in (('data', g, g.x_local, g.tan_raw_x), ('sim', R, R.xl, R.x_tan_theta)):
                m = (d.xmiss < cut).to_numpy()
                c, e, *_ = band(pos[m], tan[m], rng)
                row.append(f'{name} {fmt(c, e)}')
            print(f'  {arm} |miss|<{cut}: ' + '  '.join(row))


def check_yview(t, rng):
    print(f'== x and y band with the x-pointing cut |miss| < {MISS_CUT:g} mm (one y candidate in data)')
    for arm in 'AC':
        for view in 'xy':
            g, R = data_sel(t, arm, view), sim_sel(arm, view)
            mg = ((g.xmiss < MISS_CUT) & (g.n_cand_y == 1)).to_numpy()
            mr = (R.xmiss < MISS_CUT).to_numpy()
            cd, ed, nd, _, rd = band(g[f'{view}_local'][mg], g[f'tan_raw_{view}'][mg], rng)
            cs, es, ns, _, rs = band(sim_pos(R, view)[mr], R[f'{view}_tan_theta'][mr], rng)
            print(f'  {arm} {view}: data {fmt(cd, ed)} (n {nd}) resp {np.round(rd, 2)} | '
                  f'sim {fmt(cs, es)} (n {ns}) resp {np.round(rs, 2)} | data/sim {cd / cs:.3f}')


def check_charge(t, rng):
    print(f'== band by charge quintile (|miss| < {MISS_CUT:g} mm)')
    for arm in 'AC':
        for view in 'xy':
            g, R = data_sel(t, arm, view), sim_sel(arm, view)
            g, R = g[g.xmiss < MISS_CUT], R[R.xmiss < MISS_CUT]
            for name, d, pos, tan in (('data', g, g[f'{view}_local'], g[f'tan_raw_{view}']),
                                      ('sim ', R, sim_pos(R, view), R[f'{view}_tan_theta'])):
                q = d[f'{view}_q_sum']
                e = np.percentile(q, [0, 20, 40, 60, 80, 100])
                out = []
                for lo, hi in zip(e[:-1], e[1:]):
                    m = ((q >= lo) & (q < hi)).to_numpy()
                    c, er, *_ = band(pos[m], tan[m], rng, nb=50)
                    out.append(fmt(c, er))
                print(f'  {arm} {view} {name}: ' + ' '.join(out))


def check_source(rng):
    print('== sim band by the TRUE source position at the capsule plane (all tracks, no miss cut)')
    for arm in 'AC':
        R = sim_sel(arm, 'y')
        for view, src in (('x', R.src_u), ('y', R.src_v)):
            pos, tan = sim_pos(R, view), R[f'{view}_tan_theta']
            q = np.percentile(src, [0, 25, 50, 75, 100])
            quart = []
            for lo, hi in zip(q[:-1], q[1:]):
                m = ((src >= lo) & (src < hi)).to_numpy()
                c, e, *_ = band(pos[m], tan[m], rng, nb=40)
                quart.append(fmt(c, e))
            width = []
            for W in (5, 10, 20, 30, 45, 70, np.inf):
                m = (np.abs(src - np.median(src)) < W).to_numpy()
                c, e, *_ = band(pos[m], tan[m], rng, nb=30)
                width.append(f'rms {np.std(src[m]):.0f}: {c:.3f}')
            print(f'  {arm} {view} by source quartile: {" ".join(quart)}')
            print(f'  {arm} {view} vs source width:     {" | ".join(width)}')


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    rng = np.random.default_rng(5)
    t = pd.read_parquet(C.DATA)
    for k, fn in (('time', lambda: check_time(t, rng)), ('miss', lambda: check_miss(t, rng)),
                  ('yview', lambda: check_yview(t, rng)), ('charge', lambda: check_charge(t, rng)),
                  ('source', lambda: check_source(rng))):
        if what in (k, 'all'):
            fn()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
