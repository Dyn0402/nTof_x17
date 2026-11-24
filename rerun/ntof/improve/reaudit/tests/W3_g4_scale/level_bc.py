#!/usr/bin/env python3
"""Levels (b) reconstruction and (c) truth on the reconstructed events: the digitised A / C Geant4 libraries through
pass 1 ($WORK_DATA/ntof_pass1/g4_digi/g4_<arm>_<lib>_pass1.parquet, wall-selected: full gap + own track reaches the
wall).  Reco frame: xl = -(x_p0 - 199.29) = local u at the strip plane, x_tan_theta = du/dw (same sign as truth)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17')
import w3lib as L

warnings.simplefilter('ignore')
from ntof_cosmics.g4_digi import analyse as AN  # noqa: E402

AN.OUT = L.DIGI
W_STRIP = 30.1 + 0.128          # strip plane just behind the mesh (only matters at the 1e-4 level)
rows = []
for lib in ('nose', 'ear2_L19p5'):
    Red = L.add_global(L.load_reduced(lib))
    base = Red.full & (Red.v_mesh.abs() <= 125) & (Red.sh.abs() < 0.5) & (Red.tan_gap_v.abs() < 0.5)
    _, zs_truth, crt = L.source_from_crossings(Red[base & ((Red.A != 'D') | Red.wall)])
    S = {}
    for arm in 'AC':
        R = AN.load(f'g4_{arm}_{lib}_pass1')
        R['A'] = arm
        R = L.add_global(R, tan='tan_u')   # truth: tan_u, u_mesh -> h, n, sh
        h, n, s = L.to_global(arm, R.xl, np.full(len(R), W_STRIP), R.x_tan_theta)
        R['hr'], R['nr'], R['sr'] = h, n, s
        R['tvr'] = R.tan_v_reco
        S[arm] = R
    # reco source x from the reco A / C crossings (data: XS from A/C), z from the D truth crossing (no D reco)
    cr = {}
    for arm in 'AC':
        g = S[arm]
        g = g[g.x_ok & g.y_ok & (g.sr.abs() < 0.5) & (g.tvr.abs() < 0.5) & (g.vl.abs() <= 125)]
        cr[arm] = L.crossing(g.hr.to_numpy(), -(g.sr * g.nr).to_numpy())
    xs_r = np.mean([cr[a][0] for a in 'AC'])
    xs_t = np.mean([crt[a][0] for a in 'AC'])
    print(lib, 'reco crossings', cr, 'xs reco', round(xs_r, 2), 'truth xs', round(xs_t, 2), 'zs (D truth)',
          round(zs_truth, 2), flush=True)
    for arm in 'AC':
        R = S[arm]
        reco = R.x_ok & R.y_ok & (R.sr.abs() < 0.5) & (R.tvr.abs() < 0.5) & (R.vl.abs() <= 125)
        tru = R.full if 'full' in R else (R.w_hi - R.w_lo > 20)
        tcut = tru & (R.sh.abs() < 0.5) & (R.tan_v.abs() < 0.5) & (R.v_mesh.abs() <= 125)
        cases = {
            '(a) truth, all digitised (wall) events, truth cuts': (tcut, 'h', 'n', 'sh'),
            '(b) reconstruction, reco cuts': (reco, 'hr', 'nr', 'sr'),
            '(c) truth on reconstructed events (reco cuts)': (reco, 'h', 'n', 'sh'),
            '(c2) truth on events passing reco AND truth cuts': (reco & tcut, 'h', 'n', 'sh'),
            '(b2) reconstruction on events passing reco AND truth cuts': (reco & tcut, 'hr', 'nr', 'sr'),
        }
        for lab, (m, hc, nc, sc) in cases.items():
            g = R[m]
            for slab, src in (('crossing', (xs_r if 'reco' in lab and lab.startswith('(b') else xs_t, zs_truth)),
                              ('generator (0,0)', (0.0, 0.0))):
                e = L.expected(g.rename(columns={hc: '_h', nc: '_n'}), arm, src, h='_h', n='_n')
                k, ke, a = L.boot_scale(e, g[sc].to_numpy(), nboot=200)
                rows.append(dict(lib=lib, arm=arm, case=lab, source=slab, xs=src[0], zs=src[1], n=int(m.sum()),
                                 k=k, k_err=ke, offset=a))
                print(f'{lib:11s} {arm} {lab:58s} {slab:16s} n {int(m.sum()):6d} k {k:.4f} +- {ke:.4f}', flush=True)
        # paired difference reco - truth on the same events (bootstrap of the difference)
        m = reco
        g = R[m]
        src = (xs_t, zs_truth)
        e_t = L.expected(g, arm, src)
        e_r = L.expected(g.rename(columns={'hr': '_h', 'nr': '_n'}), arm, (xs_r, zs_truth), h='_h', n='_n')
        rng = np.random.default_rng(7)
        d = []
        for _ in range(200):
            i = rng.integers(0, len(g), len(g))
            d.append(L.scale(e_r[i], g.sr.to_numpy()[i])[0] - L.scale(e_t[i], g.sh.to_numpy()[i])[0])
        d0 = L.scale(e_r, g.sr.to_numpy())[0] - L.scale(e_t, g.sh.to_numpy())[0]
        rows.append(dict(lib=lib, arm=arm, case='(b)-(c) paired difference', source='crossing', n=len(g), k=d0,
                         k_err=float(np.nanstd(d, ddof=1))))
        print(f'{lib:11s} {arm} (b)-(c) paired: {d0:+.4f} +- {np.nanstd(d, ddof=1):.4f}', flush=True)
pd.DataFrame(rows).to_csv(L.HERE / 'level_bc.csv', index=False)
