#!/usr/bin/env python3
"""Level (a): Geant4 truth gap line (edep-weighted fit of every DriftGas step) and its mesh-plane point, the T5/W2b
estimator, per arm and library; source from the sample's own band crossings vs the generator centre (0, 0)."""
import sys

import numpy as np
import pandas as pd

import w3lib as L

rows = []
for lib in ('nose', 'ear2_L19p5'):
    R = L.load_reduced(lib)
    R = L.add_global(R)
    acc = R.full & (R.sh.abs() < 0.5) & (R.tan_gap_v.abs() < 0.5) & (R.v_mesh.abs() <= 125)
    acc &= (R.A != 'D') | R.wall          # data: D needs its wall
    S = R[acc].copy()
    xs, zs, cr = L.source_from_crossings(S)
    print(lib, 'crossings', {k: (round(v[0], 2), round(v[1], 2)) for k, v in cr.items()}, 'xs', round(xs, 2),
          'zs', round(zs, 2), flush=True)
    for arm in 'ACD':
        for wall in ((False, True) if arm in 'AC' else (True,)):
            g = S[(S.A == arm) & (S.wall | (not wall))]
            for lab, src in (('crossing', (xs, zs)), ('generator (0,0)', (0.0, 0.0)),
                             ('own crossing only', (cr[arm][0] if arm in 'AC' else xs, cr[arm][0] if arm == 'D' else zs))):
                e = L.expected(g, arm, src)
                k, ke, a = L.boot_scale(e, g.sh.to_numpy(), nboot=200)
                nb = int(((np.abs(e) >= 0.18) & (np.abs(e) <= 0.44)).sum())
                rows.append(dict(lib=lib, arm=arm, wall_req=wall, source=lab, xs=src[0], zs=src[1], n_sel=len(g),
                                 n_band=nb, k=k, k_err=ke, offset=a))
                print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(L.HERE / 'level_a.csv', index=False)
