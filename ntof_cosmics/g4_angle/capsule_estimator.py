#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capsule_estimator.py -- k_arm's capsule-pointing estimators (band, track, and
the response per |true tan| bin) on the Geant4 beam-capture population with an
IDEAL reconstruction, against the same estimators on the data.
HANDOFF_TRACKING_2026-10-06.md §12.

Why: the wall estimator is population-dominated (ideal 0.59 on beam electrons,
§10g), so it cannot say whether the reconstruction is right on beam.  The
capsule estimators can: on the simulated beam population they read ~1 for an
ideal line through the gap ionisation, so any departure in data is the
reconstruction (or a population the sim lacks), not scattering.

Sim geometry check: 1 GeV muon guns from the capsule centre give
u_mesh = 16.3 + 234.6 tan, i.e. the data's D_PERP and a foot of 16.3 mm.

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_angle/capsule_estimator.py [--data]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path('/media/dylan/data/x17/ntof_cosmics/g4_angle')
CAMPAIGN = Path('/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet')
D = 234.6
FOOT_SIM = 16.3
E = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def _resp(lev, tan):
    te = lev / D
    r = tan * np.sign(te) / np.abs(te)
    return [float(np.median(r[(np.abs(te) >= lo) & (np.abs(te) < hi)])) for lo, hi in zip(E[:-1], E[1:])]


def _k(lev, tan):
    from ntof_tracking import run145_target_imaging as TI
    s, _ = TI._robust_line(lev, tan)
    return (1 / D) / s, float(np.median((lev / D) / tan))


def sim() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(BASE / 'neutrons_nose' / '*.parquet')))])
    d['same'] = d.wall_same_track.astype(object).fillna(False).astype(bool)
    rows = []
    for arm, name in ((2, 'A'), (3, 'C')):
        b = d[(d.arm == arm) & (d.w_hi - d.w_lo > 20)]
        for lab, a in [('full gap', b), ('full gap, reaches wall (same track)', b[b.same]),
                       ('  KE < 2 MeV', b[b.same & (b.dom_ke_gap < 2)]),
                       ('  KE 2-4 MeV', b[b.same & b.dom_ke_gap.between(2, 4)]),
                       ('  KE > 4 MeV', b[b.same & (b.dom_ke_gap > 4)])]:
            lo, hi = np.percentile(a.edep_gap, [25, 75])
            a = a[a.edep_gap.between(lo, hi)]
            lev = (a.u_mesh - FOOT_SIM).to_numpy()
            tan = a.tan_gap_u.to_numpy()
            m = (np.abs(lev) > 30) & (np.abs(lev) < 130) & (np.abs(tan) > 1e-3)
            kb, kt = _k(lev[m], tan[m])
            rows.append(dict(source='Geant4 ideal', arm=name, sample=lab, n=int(m.sum()), band=kb, track=kt,
                             **{f'r{lo:.2f}': v for lo, v in zip(E, _resp(lev[m], tan[m]))}))
    return pd.DataFrame(rows)


def data() -> pd.DataFrame:
    import pyarrow.parquet as pq
    from ntof_tracking import run145_target_imaging as TI
    from sept26_prelim_analysis import k_arm as K
    cols = ['arm', 'gated', 'coinc_this_arm', 'x_q_sum', 'x_local', 'tan_raw_x', 't_since_flash_ns', 'is_flash']
    t = pq.read_table(CAMPAIGN, columns=cols, filters=[('arm', 'in', ['A', 'C']), ('gated', '==', True)]).to_pandas()
    t = t[t.coinc_this_arm.astype(bool) & (t.x_q_sum > 0) & np.isfinite(t.tan_raw_x) & ~t.is_flash.astype(bool)]
    rows = []
    for arm in 'AC':
        g = t[t.arm == arm]
        lo, hi = np.percentile(g.x_q_sum, K.CHARGE_WINDOW)
        g = g[g.x_q_sum.between(lo, hi)]
        ms = g.t_since_flash_ns / 1e6
        for lab, x in [('production raw, > 10 ms', g[ms > 10]), ('production raw, 10-20 ms', g[(ms > 10) & (ms < 20)]),
                       ('production raw, 30-60 ms', g[(ms > 30) & (ms < 60)]),
                       ('production raw, > 60 ms', g[ms > 60])]:
            lev = (x.x_local - TI.PINWHEEL[arm]).to_numpy()
            tan = x.tan_raw_x.to_numpy()
            m = (np.abs(lev) > K.LEVER_WINDOW_MM[0]) & (np.abs(lev) < K.LEVER_WINDOW_MM[1]) & (np.abs(tan) > 1e-3)
            kb, kt = _k(lev[m], tan[m])
            rows.append(dict(source='data campaign', arm=arm, sample=lab, n=int(m.sum()), band=kb, track=kt,
                             **{f'r{lo:.2f}': v for lo, v in zip(E, _resp(lev[m], tan[m]))}))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', action='store_true', help='also the campaign table (6 M rows, ~1 min)')
    a = ap.parse_args()
    R = sim()
    if a.data:
        R = pd.concat([R, data()], ignore_index=True)
    R.to_csv(BASE / 'capsule_estimator.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(R.round(3).to_string(index=False))
    print(f'-> {BASE / "capsule_estimator.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
