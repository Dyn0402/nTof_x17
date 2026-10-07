#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_data.py -- the capsule view, like for like: digitised Geant4 beam
electrons through the production reco vs the data, both under the same
in-situ bundle and k_arm's own selection (charge window 25-75 % of the reco
x_q_sum, lever 30-130 mm, reco position, raw reco tan).  Bootstrap errors.
HANDOFF_TRACKING_2026-10-06.md §13.

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/compare_data.py [--sim-a g4_A_is2] [--sim-c g4_C_is2]
        [--data is2|prod|<tracks.parquet>] [--out compare_data.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ntof_cosmics.g4_digi import analyse as A

DATA = Path.home() / 'scratch/ntof_insitu/beamseed/is2/run_145/stat090_0000/tracks/tracks.parquet'
#: run_145 stat090_0000 tracks per reco: the in-situ bundles, and the production pass (v 42.6)
DATAS = {'is2': DATA,
         'prod': Path.home() / 'scratch/ntof_insitu/beamseed/prod/run_145/stat090_0000/tracks/tracks.parquet'}
E = [0.10, 0.20, 0.30, 0.40, 0.55]
PINWHEEL = {'A': 16.35, 'C': 17.3}


def stats(lev, tan, rng, nboot=300):
    m = (np.abs(lev) > 30) & (np.abs(lev) < 130) & np.isfinite(tan) & (np.abs(tan) > 1e-3)
    lev, tan = lev[m], tan[m]

    def one(l, t):
        from ntof_tracking import run145_target_imaging as TI
        s, _ = TI._robust_line(l, t)
        te = l / A.D
        r = t * np.sign(te) / np.abs(te)
        o = dict(band=(1 / A.D) / s, track=float(np.median(te / t)))
        for lo, hi in zip(E[:-1], E[1:]):
            b = (np.abs(te) >= lo) & (np.abs(te) < hi)
            o[f'r{lo:.2f}'] = float(np.median(r[b]))
            o[f'w{lo:.2f}'] = float((r[b] <= 0).mean())
        return o
    c = one(lev, tan)
    B = pd.DataFrame([one(lev[i], tan[i]) for i in
                      (rng.integers(0, len(lev), len(lev)) for _ in range(nboot))])
    return c, B.std(), int(len(lev))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-a', default='g4_A_is2')
    ap.add_argument('--sim-c', default='g4_C_is2')
    ap.add_argument('--data', default='is2', help='is2 | prod | a tracks.parquet')
    ap.add_argument('--out', default='compare_data.csv')
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    t = pd.read_parquet(DATAS.get(a.data, a.data))
    rows = []
    for arm, lab in (('A', a.sim_a), ('C', a.sim_c)):
        R = A.load(lab)
        R = R[R.x_ok & np.isfinite(R.x_tan_theta)]
        lo, hi = np.percentile(R.x_q_sum, [25, 75])
        Q = R[R.x_q_sum.between(lo, hi)]
        f = A.foot(R)
        for name, lev, tan in (('G4 ideal line', Q.u_mesh - f, Q.tan_u),
                               ('G4 -> digitiser -> reco', Q.xl - f, Q.x_tan_theta)):
            c, s, n = stats(lev.to_numpy(), tan.to_numpy(), rng)
            rows.append(dict(arm=arm, sample=name, n=n, **c, **{f'{k}_err': v for k, v in s.items()}))
        g = t[(t.arm == arm) & t.gated & t.coinc_this_arm.astype(bool) & (t.x_q_sum > 0) & np.isfinite(t.tan_raw_x)]
        lo, hi = np.percentile(g.x_q_sum, [25, 75])
        g = g[g.x_q_sum.between(lo, hi)]
        c, s, n = stats((g.x_local - PINWHEEL[arm]).to_numpy(), g.tan_raw_x.to_numpy(), rng)
        rows.append(dict(arm=arm, sample=f'data run_145 ({a.data})', n=n, **c, **{f'{k}_err': v for k, v in s.items()}))
    T = pd.DataFrame(rows)
    T.to_csv(A.OUT / a.out, index=False)
    cols = ['band', 'track'] + [f'r{lo:.2f}' for lo in E[:-1]]
    with pd.option_context('display.width', 250):
        show = T[['arm', 'sample', 'n']].copy()
        for c in cols:
            show[c] = [f'{v:.3f}±{e:.3f}' for v, e in zip(T[c], T[f'{c}_err'])]
        for lo in E[:-1]:
            show[f'wrong{lo:.2f}'] = T[f'w{lo:.2f}'].round(2)
        print(show.to_string(index=False))
    print(f'-> {A.OUT / a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
