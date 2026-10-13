#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
same_chamber_scatter.py -- why real aluminium e+e- pairs do not point back to
their vertex, from Geant4 truth, and what that means for X17 legs and for the
close-pair loss in the expected count.

Input: the per-lepton truth of MX17_Full_Geant scripts/same_arm_pairs.py over
neutrons_thermal_trig_2cm_nose (same parts as same_chamber_g4.py).  Every
number is on the TRUE first gap step (position and direction), so it is the
physics alone: no reconstruction, no resolution.

Outputs (paths.out('same_chamber')):

  scatter_legs.csv     per conversion-lepton kinetic-energy bin: miss of the
                       gap-entry line at the true vertex, net deflection between
                       the vertex->gap chord and the gap direction, p*beta*theta
  scatter_x17.csv      the same scaled to X17 legs (8 and 10 MeV) with the
                       measured 1/(p beta) law, checked on the 6-9 MeV bin
  vertex_truthline.csv the same-chamber vertex test (lines within 30 mm,
                       midpoint within 20 mm of the axis) applied to TRUE gap
                       lines of real e+e- pairs, by opening angle
  sep_loss.csv         share of accepted e+e- pairs closer than 20 mm (and
                       20-40 mm) in either view at mid-gap, by opening angle

    python -m sept26_prelim_analysis.same_chamber_scatter
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths
from sept26_prelim_analysis import same_chamber_g4 as G

PARTS = Path.home() / 'x17_local' / 'g4_same_arm' / 'parts'
PAIRS = Path.home() / 'x17_local' / 'g4_same_arm' / 'out' / 'g4_pairs.parquet'
ME = 0.511
KE_BINS = [(0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 6), (6, 9)]
OPEN_BINS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90)]
X17_E = (8.0, 10.0)


def conv_legs(L: pd.DataFrame) -> pd.DataFrame:
    T = L[L.arm.isin([2, 3]) & ((L.w_max - L.w_min) >= G.TRACK_W_MM) & (L.op == 'conv')].copy()
    g0 = T[['gx0', 'gy0', 'gz0']].to_numpy(float)
    p = T[['px0', 'py0', 'pz0']].to_numpy(float)
    p /= np.linalg.norm(p, axis=1)[:, None]
    o = T[['ox', 'oy', 'oz']].to_numpy(float)
    v = o - g0
    t = (v * p).sum(1)
    T['miss'] = np.linalg.norm(v - t[:, None] * p, axis=1)
    ch = g0 - o
    T['dist'] = np.linalg.norm(ch, axis=1)
    ch /= T['dist'].to_numpy()[:, None]
    T['defl'] = np.degrees(np.arccos(np.clip((ch * p).sum(1), -1, 1)))
    T['pb'] = T.ke0 * (T.ke0 + 2 * ME) / (T.ke0 + ME)
    return T


def legs_table(T):
    rows = []
    for lo, hi in KE_BINS:
        s = T[(T.ke0 >= lo) & (T.ke0 < hi)]
        rows.append(dict(ke_lo=lo, ke_hi=hi, n=len(s), miss_med=s.miss.median(),
                         miss_p68=s.miss.quantile(.68), defl_med=s.defl.median(),
                         defl_p68=s.defl.quantile(.68), pb_med=s.pb.median(),
                         pbtheta=float(np.median(s.pb * np.radians(s.defl))),
                         p_miss20=float((s.miss < 20).mean())))
    s = T[T.ke0 >= 0.5]
    rows.append(dict(ke_lo=0.5, ke_hi=99, n=len(s), miss_med=s.miss.median(),
                     miss_p68=s.miss.quantile(.68), defl_med=s.defl.median(),
                     defl_p68=s.defl.quantile(.68), pb_med=s.pb.median(),
                     pbtheta=float(np.median(s.pb * np.radians(s.defl))),
                     p_miss20=float((s.miss < 20).mean())))
    return pd.DataFrame(rows)


def x17_table(T):
    s = T[T.ke0 >= 0.5]
    rows = []
    for E in X17_E:
        f = s.pb / E
        ms, d = s.miss * f, s.defl * f
        p = float((ms < 20).mean())
        rows.append(dict(E_MeV=E, defl_med=d.median(), miss_med=ms.median(),
                         miss_p68=ms.quantile(.68), p_miss20=p, p_both20=p * p))
    return pd.DataFrame(rows)


def truthline_vertex(L):
    T = L[L.arm.isin([2, 3]) & ((L.w_max - L.w_min) >= G.TRACK_W_MM) & (L.op == 'conv')]
    rows = []
    for _, g in T.groupby(['file', 'event', 'arm']):
        if len(g) < 2:
            continue
        for i, j in itertools.combinations(range(len(g)), 2):
            x, y = g.iloc[i], g.iloc[j]
            if x.parent == y.parent:
                rows.append([x.gx0, x.gy0, x.gz0, x.px0, x.py0, x.pz0,
                             y.gx0, y.gy0, y.gz0, y.px0, y.py0, y.pz0, x.ox, x.oy, x.oz])
    P = np.array(rows, float)
    p1, r1, p2, r2, o = P[:, 0:3], P[:, 3:6], P[:, 6:9], P[:, 9:12], P[:, 12:15]
    w0 = p1 - p2
    aa, bb, cc = (r1 * r1).sum(1), (r1 * r2).sum(1), (r2 * r2).sum(1)
    dd, ee = (r1 * w0).sum(1), (r2 * w0).sum(1)
    den = aa * cc - bb * bb
    sc, tc = (bb * ee - cc * dd) / den, (aa * ee - bb * dd) / den
    c1, c2 = p1 + sc[:, None] * r1, p2 + tc[:, None] * r2
    dca = np.linalg.norm(c1 - c2, axis=1)
    mid = 0.5 * (c1 + c2)
    off = np.linalg.norm(mid - o, axis=1)
    vr = np.hypot(mid[:, 0], mid[:, 2])
    vr_true = np.hypot(o[:, 0], o[:, 2])
    op = np.degrees(np.arccos(np.clip(bb / np.sqrt(aa * cc), -1, 1)))
    ok = (dca < G.VERTEX_DCA_MM) & (vr < G.VERTEX_R_MM)
    out = [dict(open_lo=-1, open_hi=999, n=len(P), pass_frac=ok.mean(), mid_off_med=np.median(off),
                mid_off_p68=np.quantile(off, .68), dca_med=np.median(dca),
                true_in_r20=float((vr_true < 20).mean()))]
    for lo, hi in OPEN_BINS:
        m = (op >= lo) & (op < hi)
        out.append(dict(open_lo=lo, open_hi=hi, n=int(m.sum()), pass_frac=ok[m].mean(),
                        mid_off_med=np.median(off[m]), mid_off_p68=np.quantile(off[m], .68),
                        dca_med=np.median(dca[m]), true_in_r20=float((vr_true[m] < 20).mean())))
    return pd.DataFrame(out)


def sep_loss():
    P = pd.read_parquet(PAIRS)
    rows = []
    for arm in ('A', 'C'):
        g = P[(P.arm == arm) & (P.cls == 'e+e- (one gamma)') & (P.dw1 > 0) & (P.dw2 > 0)
              & (P[['tu1', 'tv1', 'tu2', 'tv2']].abs().max(axis=1) < 0.6)]
        du = np.abs((g.u01 + g.tu1 * (15 - g.w01)) - (g.u02 + g.tu2 * (15 - g.w02)))
        dv = np.abs((g.v01 + g.tv1 * (15 - g.w01)) - (g.v02 + g.tv2 * (15 - g.w02)))
        m = np.minimum(du, dv)
        for lo, hi in [(-1, 999)] + OPEN_BINS:
            k = (g.open_true >= lo) & (g.open_true < hi)
            rows.append(dict(arm=arm, open_lo=lo, open_hi=hi, n=int(k.sum()),
                             lt20=float((m[k] < 20).mean()) if k.any() else np.nan,
                             b20_40=float(((m[k] >= 20) & (m[k] < 40)).mean()) if k.any() else np.nan,
                             open_med=float(g.open_true[k].median()) if k.any() else np.nan))
    return pd.DataFrame(rows)


def main() -> int:
    out = paths.out('same_chamber')
    L, _ = G.load(PARTS)
    T = conv_legs(L)
    pd.set_option('display.width', 200)
    for name, df in (('scatter_legs', legs_table(T)), ('scatter_x17', x17_table(T)),
                     ('vertex_truthline', truthline_vertex(L)), ('sep_loss', sep_loss())):
        df.to_csv(out / f'{name}.csv', index=False)
        print(f'\n== {name}\n{df.round(3).to_string(index=False)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
