#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse.py -- reco / truth per |true tan| bin for digitised samples, and the
capsule estimators on them.  Truth is the generated line (muons) or the ideal
edep-weighted gap line (Geant4).

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/analyse.py <label> [<label> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path('/media/dylan/data/x17/ntof_cosmics/g4_digi')
E = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55]
HALF = 199.29
D, FOOT = 234.6, 16.35
#: sim foot per arm (zero crossing of the ideal band; muon guns give 16.3 on A)
FOOT_SIM = {'A': 16.35, 'C': 16.4}


def load(label: str) -> pd.DataFrame:
    """<label>.parquet, or the condor parts <label>/*.parquet concatenated."""
    f = OUT / f'{label}.parquet'
    if f.exists():
        R = pd.read_parquet(f)
    else:
        import glob
        parts = sorted(glob.glob(str(OUT / label / '*.parquet')))
        if not parts:
            raise FileNotFoundError(f'{f} and {OUT / label}/*.parquet')
        R = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ('x_ok', 'y_ok', 'x_quality_ok', 'y_quality_ok'):
        if c in R:
            R[c] = R[c].astype(object).fillna(False).astype(bool)
    R['xl'] = -(R.x_p0 - HALF)
    return R


def foot(R):
    return FOOT_SIM.get(str(R['arm'].iloc[0]) if 'arm' in R else 'A', FOOT)


def response(R: pd.DataFrame, plane: str, truth: str) -> pd.DataFrame:
    t = R[truth].to_numpy(float)
    r = R[f'{plane}_tan_theta'].to_numpy(float)
    ok = R[f'{plane}_ok'].to_numpy(bool) & np.isfinite(t) & np.isfinite(r)
    rows = []
    for lo, hi in zip(E[:-1], E[1:]):
        m = ok & (np.abs(t) >= lo) & (np.abs(t) < hi)
        n_all = int(((np.abs(t) >= lo) & (np.abs(t) < hi)).sum())
        if m.sum() < 15:
            continue
        x = r[m] * np.sign(t[m])
        rows.append(dict(plane=plane, lo=lo, hi=hi, n=int(m.sum()), eff=m.sum() / max(n_all, 1),
                         ratio=float(np.median(x) / np.median(np.abs(t[m]))) if lo > 0 else np.nan,
                         med_ratio=float(np.median(x / np.abs(t[m]))) if lo > 0 else np.nan,
                         sigma=float(1.4826 * np.median(np.abs(r[m] - t[m] - np.median(r[m] - t[m])))),
                         wrong_sign=float((x < 0).mean())))
    return pd.DataFrame(rows)


def capsule(R: pd.DataFrame) -> dict:
    """k_arm band/track on the reco x tan against the reco position."""
    from ntof_tracking import run145_target_imaging as TI
    g = R[R.x_ok & np.isfinite(R.x_tan_theta)]
    lev = (g.xl - foot(R)).to_numpy()
    t = g.x_tan_theta.to_numpy()
    m = (np.abs(lev) > 30) & (np.abs(lev) < 130) & (np.abs(t) > 1e-3)
    if m.sum() < 50:
        return {}
    s, _ = TI._robust_line(lev[m], t[m])
    return dict(n=int(m.sum()), band=(1 / D) / s, track=float(np.median((lev[m] / D) / t[m])))


def main() -> int:
    for lab in sys.argv[1:]:
        R = load(lab)
        print(f'=== {lab}: {len(R)} events; x fit {R.x_ok.mean():.2f}, y fit {R.y_ok.mean():.2f}')
        for p, tr in (('x', 'tan_u'), ('y', 'tan_v')):
            T = response(R, p, tr)
            with pd.option_context('display.width', 200):
                print(T.round(3).to_string(index=False))
        if 'u_mesh' in R:
            print('  capsule (reco):', {k: round(v, 3) for k, v in capsule(R).items()})
        good = R.x_ok & (R.x_q_sum < 1e5)
        print(f'  median x_q_sum {R.x_q_sum[good].median():.0f} / y {R.y_q_sum[R.y_ok & (R.y_q_sum < 1e5)].median():.0f}'
              f' (data is2 A beam 1553 / 1358); q runaways (>1e5) {(R.x_ok & (R.x_q_sum >= 1e5)).mean():.3f}')
        if (R.kind == 'g4').any():
            # the data's view: truth = capsule pointing from the TRUE mesh position, lever 30-130 mm
            f = foot(R)
            lev = R.u_mesh - f
            C = R[(lev.abs() > 30) & (lev.abs() < 130)].assign(tan_cap=lambda z: (z.u_mesh - f) / D)
            print('  -- vs capsule-expected tan (lever 30-130 mm), x; ideal-line/capsule alongside')
            T = response(C, 'x', 'tan_cap')
            I = response(C.assign(x_tan_theta=C.tan_u, x_ok=True), 'x', 'tan_cap')
            T['ideal_ratio'] = I.set_index('lo').reindex(T.lo).med_ratio.to_numpy()
            with pd.option_context('display.width', 200):
                print(T.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
