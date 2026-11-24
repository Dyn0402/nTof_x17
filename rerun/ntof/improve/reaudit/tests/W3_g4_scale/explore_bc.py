#!/usr/bin/env python3
"""Why does the reconstruction read steeper than the truth gap line?  k for truth / reco by dom_share and energy,
and the residual widths."""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17')
import w3lib as L

warnings.simplefilter('ignore')
from ntof_cosmics.g4_digi import analyse as AN  # noqa: E402

AN.OUT = L.DIGI
for lib, src in (('nose', (0.46, 0.32)), ('ear2_L19p5', (0.35, 1.11))):
    for arm in 'AC':
        R = AN.load(f'g4_{arm}_{lib}_pass1'); R['A'] = arm
        R = L.add_global(R, tan='tan_u')
        h, n, s = L.to_global(arm, R.xl, np.full(len(R), 30.228), R.x_tan_theta)
        R['hr'], R['nr'], R['sr'] = h, n, s
        reco = R.x_ok & R.y_ok & (R.sr.abs() < 0.5) & (R.tan_v_reco.abs() < 0.5) & (R.vl.abs() <= 125)
        g = R[reco].copy()
        g['et'] = L.expected(g, arm, src)
        g['er'] = L.expected(g.rename(columns={'hr': '_h', 'nr': '_n'}), arm, src, h='_h', n='_n')
        print(f'--- {lib} {arm}  n {len(g)}  MAD truth resid {1.4826*np.median(np.abs(g.sh-g.et)):.3f} reco resid '
              f'{1.4826*np.median(np.abs(g.sr-g.er)):.3f}  reco-truth tan {1.4826*np.median(np.abs(g.sr-g.sh)):.3f} '
              f'pos {1.4826*np.median(np.abs(g.hr-g.h)):.2f} mm')
        for lab, m in (('dom_share>0.9', g.dom_share > 0.9), ('dom_share<=0.9', g.dom_share <= 0.9),
                       ('ke<2', g.dom_ke_gap < 2), ('ke 2-4', (g.dom_ke_gap >= 2) & (g.dom_ke_gap < 4)),
                       ('ke>=4', g.dom_ke_gap >= 4),
                       ('|reco-truth tan|<0.05', (g.sr - g.sh).abs() < 0.05)):
            x = g[m]
            kt, ket, _ = L.boot_scale(x.et.to_numpy(), x.sh.to_numpy(), nboot=100, nmin=25)
            kr, ker, _ = L.boot_scale(x.er.to_numpy(), x.sr.to_numpy(), nboot=100, nmin=25)
            # reco measured vs truth expected (isolates the angle part of the reconstruction)
            km, kme, _ = L.boot_scale(x.et.to_numpy(), x.sr.to_numpy(), nboot=100, nmin=25)
            print(f'  {lab:24s} n {len(x):5d} truth {kt:.3f}+-{ket:.3f} reco {kr:.3f}+-{ker:.3f} '
                  f'reco-angle/truth-pos {km:.3f}+-{kme:.3f}')
        # direct regression of reco tan on truth tan in the band
        b = (g.sh.abs() > 0.15) & (g.sh.abs() < 0.45)
        print('  median(sr/sh) in 0.15<|sh|<0.45:', round(float(np.median(g.sr[b] / g.sh[b])), 4))
