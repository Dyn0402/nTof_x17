#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bd_map.py -- chamber D's angle map from the B-D cosmic lines (B has positions only, so B is D's truth the way C is A's).

Needs the censored-window cut (rerun/ntof/stage3/censor_join.py): without it 69 % of D's beam-off tracks are fakes on
censored strips and the B-D lines do not close.  Sample: beam-off sub-runs, one gated_reco B track and one gated D track,
D's line passing within SEP_MM of B's point (after the median offset); truth = the joined B-D line, j = d(p)/d(x_global)
for p = z (D's x view) and y (D's y view), 468 mm lever arm.  Map as angle_map.py but 100 mm cells (~10x fewer lines):
r = a + s j per cell, bilinear, corrected slope (s - a)/(1 + s_).  "Clean": D's other view agrees with the line within
CLEAN (B has no angle to check).

    python -m rerun.ntof.improve.bd_map --view <dir with censored beam-off tables>   -> <out>/bd_lines.parquet,
                                                                                         bd_map.json, bd_map_heldout.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.improve import angle_map as AM  # noqa: E402
from rerun.ntof.improve.track_quality import out_dir  # noqa: E402

SEP_MM = 60.0
CLEAN = 0.10
EDGES = np.arange(-200.0, 201.0, 100.0)
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'gated', 'gated_reco', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
        'x_local', 'y_local', 't_since_flash_ns']
K = ['run', 'subrun', 'tag', 'event_id']
VIEWS = {'z': 'D_x', 'y': 'D_y'}         # global coordinate -> D's view name (D's x strips measure global z)


def lines(view: str) -> pd.DataFrame:
    t = pd.concat([pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(f'{view}/tracks_run_*.parquet'))])
    t = t[t.t_since_flash_ns.isna()]
    B = t[(t.arm == 'B') & t.gated_reco.fillna(False).astype(bool)]
    D = t[(t.arm == 'D') & t.gated.fillna(False).astype(bool)]
    B = B[B.groupby(K).arm.transform('size') == 1]
    D = D[D.groupby(K).arm.transform('size') == 1]
    M = B.merge(D, on=K, suffixes=('_b', '_d'))
    tt = (M.p0_x_b - M.p0_x_d) / M.d_x_d
    mz, my = M.p0_z_d + tt * M.d_z_d - M.p0_z_b, M.p0_y_d + tt * M.d_y_d - M.p0_y_b
    M['sep_mm'] = np.hypot(mz - np.median(mz), my - np.median(my))
    M = M[M.sep_mm < SEP_MM].copy()
    for p in 'zy':
        M[f'j{p}'] = (M[f'p0_{p}_d'] - M[f'p0_{p}_b']) / (M.p0_x_d - M.p0_x_b)
        M[f'res_{p}'] = M[f'd_{p}_d'] / M.d_x_d - M[f'j{p}']
    M.to_parquet(out_dir() / 'bd_lines.parquet', index=False)
    return M


def clean(M, p):
    q = 'y' if p == 'z' else 'z'
    ok = (M[f'res_{q}'].abs() < CLEAN) & (M[f'j{p}'].abs() < 0.45) & (M[f'res_{p}'].abs() < 0.15)
    return pd.DataFrame({'x': M.loc[ok, 'x_local_d'], 'y': M.loc[ok, 'y_local_d'], 'j': M.loc[ok, f'j{p}'],
                         'r': M.loc[ok, f'res_{p}'], 'run': M.loc[ok, 'run'], 'ev': M.loc[ok, 'event_id']})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', required=True)
    a = ap.parse_args()
    AM.EDGES, AM.CENTRES, AM.N_MIN = EDGES, 0.5 * (EDGES[1:] + EDGES[:-1]), 60
    M = lines(a.view)
    print(len(M), 'B-D lines from', M.run.nunique(), 'runs')
    maps, rows = {}, []
    for p, key in VIEWS.items():
        G = clean(M, p)
        maps[key] = AM.fit_map(G)
        gp = G.ev.astype(int) % 2      # event parity: the beam-off lines are 90 % in odd runs, so run parity is lopsided
        for tr in (0, 1):
            mp = AM.fit_map(G[gp == tr])
            Gt = G[gp != tr]
            cs = AM.correct(Gt.r.to_numpy() + Gt.j.to_numpy(), Gt.x.to_numpy(), Gt.y.to_numpy(), mp)
            # global scale only (a0, s0 of the training half), the fallback without a map
            g0 = AM.correct(Gt.r.to_numpy() + Gt.j.to_numpy(), Gt.x.to_numpy(), Gt.y.to_numpy(),
                            dict(mp, a=np.full_like(np.asarray(mp['a']), mp['a0']).tolist(),
                                 s=np.full_like(np.asarray(mp['s']), mp['s0']).tolist()))
            rows.append(dict(view=key, train=tr, n_test=len(Gt), sig_before=AM.mad(Gt.r),
                             sig_global=AM.mad(g0 - Gt.j.to_numpy()), sig_map=AM.mad(cs - Gt.j.to_numpy()),
                             a0=mp['a0'], s0=mp['s0']))
    json.dump(dict(meta=dict(built_from='bd_lines.parquet (beam-off, censored-window cut)', edges=EDGES.tolist(),
                             model='as angle_map.py; global slopes d_z/d_x (D x view) and d_y/d_x (D y view)'),
                   maps=maps), open(out_dir() / 'bd_map.json', 'w'), indent=1)
    T = pd.DataFrame(rows)
    T.to_csv(out_dir() / 'bd_map_heldout.csv', index=False)
    print(T.round(4).to_string())
    for k, m in maps.items():
        print(k, 'a0 %.4f s0 %.4f' % (m['a0'], m['s0']), 'cells n', np.asarray(m['n']).ravel().tolist())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
