#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
y_guard.py -- refuse a stage-3 table whose global y is mirrored.

Every internal consistency check is blind to a global mirror of an axis
(../COORDINATES.md §3), so the guard uses an external, direction-carrying fact:
**late captures sit in the capsule's upstream aluminium nose, at the BOTTOM**
(the beam comes up from the spallation target).  The y band crossing of the
late (>= 20 ms) A and C tracks -- the zero of the robust line of y displacement
to the axis plane against y on the chamber -- is therefore NEGATIVE, ~ -30 mm
(Geant4 -28 mm; data -31/-35 mm after the 2026-10-13 fix).  Mirrored, it reads
~ +33 mm.  The estimator is a pointing one and scale-free in its zero, so a
wrong k does not move it.

It is the cheap secondary check.  The primary one is the cosmic-muon timing on
beam-off run_149 (``ntof_cosmics/y_orientation_check.py cosmic``: the truly
higher arm is hit first), which needs the n_TOF wall trees and is run by
``tests/test_y_orientation.py`` where the data are staged.

    python -m sept26_prelim_analysis.y_guard <tracks parquet or stage-3 dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: tracks after this time since the flash are thermal captures in the capsule
LATE_MS = 20.0
#: below this many tracks per arm the crossing is not computed
MIN_TRACKS = 50
#: a crossing this many sigma ABOVE zero is a mirror; refuse
FAIL_SIGMA = 3.0
#: which global coordinate an arm's x view measures, and the plane normal
VIEW = {'A': ('x', 'z'), 'C': ('x', 'z')}


def late_y_crossing(df: pd.DataFrame) -> dict:
    """Per arm (A, C): the late y band crossing and a verdict.

    ``df`` is stage-3 rows as stored (any subset of columns that includes
    p0_*, d_*, tanx, tany, gated(_reco), angle_calibrated, t_since_flash_ns).
    """
    from sept26_prelim_analysis.capsule_tof import crossing
    out = {}
    need = {'p0_y', 'd_y', 't_since_flash_ns', 'angle_calibrated', 'tany', 'tanx'}
    if not need <= set(df.columns):
        return {'verdict': 'not_checked', 'why': f'missing {sorted(need - set(df.columns))}'}
    gate = df['gated_reco'] if 'gated_reco' in df else df['gated']
    ok = (gate.astype('boolean').fillna(False).astype(bool)
          & df['angle_calibrated'].astype('boolean').fillna(False).astype(bool)
          & (df['tanx'].abs() < 0.6) & (df['tany'].abs() < 0.6)
          & (df['t_since_flash_ns'] >= LATE_MS * 1e6))
    for c in ('x_t0', 'y_t0'):
        if c in df:
            ok &= df[c] <= 300
    worst = 'insufficient'
    for arm, (_, n) in VIEW.items():
        g = df[ok & (df['arm'] == arm)]
        if len(g) < MIN_TRACKS:
            out[arm] = dict(n=int(len(g)), verdict='insufficient')
            continue
        t = -g[f'p0_{n}'].to_numpy(float) / g[f'd_{n}'].to_numpy(float)
        p0y = g['p0_y'].to_numpy(float)
        hgt = p0y + t * g['d_y'].to_numpy(float)
        f = np.isfinite(hgt) & np.isfinite(p0y)
        y0, err, nk = crossing(p0y[f], hgt[f] - p0y[f])
        if not np.isfinite(y0):
            v = 'insufficient'
        elif y0 > 0 and y0 > FAIL_SIGMA * err:
            v = 'FAIL'
        elif y0 < 0 and -y0 > FAIL_SIGMA * err:
            v = 'pass'
        else:
            v = 'inconclusive'
        out[arm] = dict(n=int(nk), y_cross_mm=round(y0, 2), err_mm=round(err, 2), verdict=v)
        rank = ['insufficient', 'inconclusive', 'pass', 'FAIL']
        if v == 'FAIL' or (worst != 'FAIL' and rank.index(v) > rank.index(worst)):
            worst = v
    out['verdict'] = worst
    out['rule'] = ('late (>= 20 ms) A/C y band crossing must be < 0 (the capsule nose, '
                   'bottom); > +3 sigma means stored y is mirrored (COORDINATES.md)')
    return out


def enforce(res: dict, where: str, allow_fail: bool = False) -> None:
    """Print the result; raise on a mirror unless explicitly allowed."""
    arms = ', '.join(f'{a} {r.get("y_cross_mm", "-")}±{r.get("err_mm", "-")} mm '
                     f'(n={r["n"]}, {r["verdict"]})'
                     for a, r in res.items() if a in VIEW)
    print(f'  [y-guard] {where}: {res["verdict"]}  {arms}')
    if res['verdict'] == 'FAIL':
        msg = (f'{where}: the late capsule y crossing is ABOVE zero -- stored y is '
               f'MIRRORED (COORDINATES.md, HANDOFF_Y_FLIP_FIX.md). Not writing.')
        if allow_fail:
            print('  !!!!!!!! ' + msg + ' (overridden)')
        else:
            raise RuntimeError(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('src', type=Path, help='stage-3 parquet, or a directory of them')
    a = ap.parse_args()
    from sept26_prelim_analysis.build_tracks import stored_y_sign
    files = sorted(a.src.glob('tracks_run_*.parquet')) if a.src.is_dir() else [a.src]
    cols = ['arm', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'tanx', 'tany', 'gated',
            'gated_reco', 'angle_calibrated', 't_since_flash_ns', 'x_t0', 'y_t0']
    import pyarrow.parquet as pq
    parts = []
    for f in files:
        have = set(pq.read_schema(f).names)
        parts.append(pd.read_parquet(f, columns=[c for c in cols if c in have],
                                     filters=[('arm', 'in', ['A', 'C'])]))
    df = pd.concat(parts, ignore_index=True)
    res = late_y_crossing(df)
    try:
        flag = stored_y_sign(a.src)
        print(f'  sidecar says stored y sign {flag:+.0f}')
    except (FileNotFoundError, ValueError) as exc:
        print(f'  sidecar: {exc}')
    enforce(res, str(a.src), allow_fail=True)
    return 1 if res['verdict'] == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
