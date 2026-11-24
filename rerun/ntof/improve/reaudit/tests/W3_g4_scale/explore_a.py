#!/usr/bin/env python3
"""Level (a) explorations: residual profile per e bin; truncation (|measured tan| cut) and acceptance variants."""
import numpy as np
import pandas as pd

import w3lib as L

for lib in ('nose', 'ear2_L19p5'):
    R = L.add_global(L.load_reduced(lib))
    base = R.full & (R.v_mesh.abs() <= 125) & ((R.A != 'D') | R.wall)
    S5 = R[base & (R.sh.abs() < 0.5) & (R.tan_gap_v.abs() < 0.5)]
    xs, zs, cr = L.source_from_crossings(S5)
    for arm in 'ACD':
        for lab, cut in (('|t|<0.5 both', (R.sh.abs() < 0.5) & (R.tan_gap_v.abs() < 0.5)),
                         ('|t|<1.0 both', (R.sh.abs() < 1.0) & (R.tan_gap_v.abs() < 1.0)),
                         ('no tan cut', R.sh.abs() < 50),
                         ('|tv|<0.5 only', R.tan_gap_v.abs() < 0.5)):
            g = R[base & cut & (R.A == arm)]
            e = L.expected(g, arm, (xs, zs))
            k, ke, a = L.boot_scale(e, g.sh.to_numpy(), nboot=100)
            print(f'{lib:11s} {arm} {lab:14s} n {len(g):6d} k {k:.4f} +- {ke:.4f}')
        g = R[base & (R.sh.abs() < 0.5) & (R.tan_gap_v.abs() < 0.5) & (R.A == arm)]
        e = L.expected(g, arm, (xs, zs))
        k, a, B = L.scale(e, g.sh.to_numpy(), return_bins=True)
        print('   bins', ' '.join(f'{r.e:+.2f}:{r.c:+.4f}({r.n})' for r in B.itertuples()))
        # dominant-track line instead of all-steps line
        gg = g[np.isfinite(g.tan_dom_u)].copy()
        gg = L.add_global(gg, tan='tan_dom_u', u='u_mesh_dom', out=('hd', 'nd', 'sd'))
        e2 = L.expected(gg.rename(columns={'hd': 'h', 'nd': 'n'}).assign(), arm, (xs, zs)) if False else \
            (gg.hd - (xs if arm in 'AC' else zs)) / (gg.nd - (zs if arm in 'AC' else xs))
        k2, ke2, _ = L.boot_scale(e2.to_numpy(), gg.sd.to_numpy(), nboot=100)
        print(f'   dominant-track line: k {k2:.4f} +- {ke2:.4f}  (n {len(gg)})')
        # residual width
        r = g.sh.to_numpy() - e
        print(f'   residual MAD sigma {1.4826 * np.median(np.abs(r - np.median(r))):.4f}')
