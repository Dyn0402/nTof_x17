#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wall_edge_scale.py -- the beam angle scale from the SiPM wall, binned in strip
position: no capsule assumption, no regression dilution (HANDOFF_TRACKING_
2026-10-06.md §10c).

Where the fired wall group switches across a surveyed boundary U_b (structure
frame), half the tracks cross on each side, so at that strip position u_b the
median TRUE tan is (U_b - u_b + foot) / L.  Compared with the median RAW reco
tan of the same tracks, that is the angle scale; the two outer boundaries
together also give it free of a rigid wall offset, and give the effective
source distance D_eff = du_b / dtan_true (a point source on the beam axis at
the strip plane's perpendicular distance predicts 234.6 mm).

Input: the scint-stack per-track tables (every late trigger, 34 runs),
`/media/dylan/data/x17/scint_stack/tracks/stack_<run>.parquet`.

    python ntof_cosmics/wall_edge_scale.py [--arms A C D] [--split t0]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

STACK = Path('/media/dylan/data/x17/scint_stack/tracks')
OUT = HERE / 'results' / 'wall_edge_scale'
L_WALL = 97.4
from ntof_scint_stack.ana import WALL_EDGES  # noqa: E402
BOUNDS = tuple(float(b) for b in WALL_EDGES[1:4])   # interior group boundaries, structure frame
EDGES = np.arange(-230.0, 230.1, 3.0)


def load(arm: str) -> pd.DataFrame:
    cols = ['run', 'arm', 'u_mm', 'tan_raw_x', 'n_trk', 'x_t0', 'y_t0', 'q_per_len'] + \
           [f'w{i}_amp_on' for i in range(1, 9)]
    T = []
    for f in sorted(glob.glob(str(STACK / 'stack_run_*.parquet'))):
        d = pd.read_parquet(f, columns=cols)
        T.append(d[(d.arm == arm) & (d.n_trk == 1)])
    d = pd.concat(T, ignore_index=True)
    # the stack keeps an amplitude only above the measured threshold (NaN
    # otherwise) -- `ntof_scint_stack.ana.wall_groups_fired`
    fired = np.stack([(d[f'w{2 * g + 1}_amp_on'].notna() | d[f'w{2 * g + 2}_amp_on'].notna()).to_numpy()
                      for g in range(4)], axis=1)
    d['n_grp'] = fired.sum(1)
    d['grp'] = np.where(d.n_grp == 1, fired.argmax(1), -1)
    return d[d.n_grp == 1].reset_index(drop=True)


def _f(u, u0, s):
    return 0.5 * (1 + erf((u - u0) / (np.sqrt(2) * s)))


def edge(d: pd.DataFrame, g: int):
    sel = d[d.grp.isin([g, g + 1])]
    idx = np.digitize(sel.u_mm, EDGES) - 1
    c = 0.5 * (EDGES[:-1] + EDGES[1:])
    p, uc, n = [], [], []
    for i in range(len(c)):
        m = idx == i
        if m.sum() < 20:
            continue
        p.append((sel.grp.to_numpy()[m] == g + 1).mean()); uc.append(c[i]); n.append(m.sum())
    p, uc, n = map(np.asarray, (p, uc, n))
    if len(p) < 6:
        return None
    if p[0] > p[-1]:
        p = 1 - p
    w = np.abs(p - 0.5) < 0.45
    try:
        (u0, s), cov = curve_fit(_f, uc[w], p[w], p0=[uc[w][np.argmin(np.abs(p[w] - 0.5))], 20],
                                 sigma=np.sqrt(np.clip(p[w] * (1 - p[w]), 0.01, 1) / n[w]))
    except Exception:
        return None
    near = sel[(sel.u_mm - u0).abs() < 3]
    return u0, float(np.sqrt(cov[0, 0])), s, float(np.median(near.tan_raw_x)), len(near)


def measure(d: pd.DataFrame, arm: str, label: str) -> list[dict]:
    # the stack's u_mm is already in the wall's structure frame (ana.fit_wall_u
    # compares u + L s tan with WALL_EDGES directly), so no pinwheel foot here
    foot = 0.0
    rows, E = [], {}
    for g, Ub in enumerate(BOUNDS):
        e = edge(d, g)
        if e is None:
            continue
        u0, ue, s, traw, n = e
        tt = (Ub - u0 + foot) / L_WALL
        E[g] = (u0, tt, traw)
        rows.append(dict(arm=arm, sample=label, boundary=f'{g}|{g + 1}', u_b=u0, u_b_err=ue, width=s,
                         tan_true=tt, tan_raw=traw, n=n, true_over_raw=tt / traw))
    if 0 in E and 2 in E:
        (u0, t0, r0), (u2, t2, r2) = E[0], E[2]
        rows.append(dict(arm=arm, sample=label, boundary='outer pair (offset-free)', u_b=np.nan,
                         tan_true=np.nan, tan_raw=np.nan, n=int(len(d)),
                         true_over_raw=(t2 - t0) / (r2 - r0), D_eff=(u2 - u0) / (t2 - t0)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    # validated on A only (reproduces det_a_scint to 0.1 %); C and D come out
    # inconsistent here -- use ntof_scint_stack ana pointing λ·k for them (§10c)
    ap.add_argument('--arms', nargs='+', default=['A'])
    a = ap.parse_args()
    rows = []
    for arm in a.arms:
        d = load(arm)
        t0 = np.maximum(d.x_t0, d.y_t0)
        q = d.q_per_len.median()
        for lab, m in (('all', np.ones(len(d), bool)), ('t0 <= 100', t0 <= 100),
                       ('100 < t0 <= 300', (t0 > 100) & (t0 <= 300)),
                       ('q_per_len < median', d.q_per_len < q), ('q_per_len >= median', d.q_per_len >= q)):
            rows += measure(d[m], arm, lab)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / 'wall_edge_scale.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_rows', 200):
        print(R[R.boundary == 'outer pair (offset-free)'][['arm', 'sample', 'n', 'true_over_raw', 'D_eff']]
              .round(3).to_string(index=False))
        print()
        print(R[R.boundary != 'outer pair (offset-free)'][R['sample'] == 'all'][
            ['arm', 'boundary', 'u_b', 'u_b_err', 'width', 'tan_true', 'tan_raw', 'n', 'true_over_raw']]
            .round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
