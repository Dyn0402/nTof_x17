#!/usr/bin/env python3
"""focus_split.py -- is the capsule focus scale (capsule_focus.py, HANDOFF_TRACKING
§21) a property of the angle scale, or of the population that is focused?

capsule_focus gives one sharpest s per arm and |tan| bin.  On is2_v2 that is
1.05-1.12, against an in-situ k set by A-C cosmic lines.  A distance error common
to A and C cancels between the two: the A-C truth tan is dx / (D_A + D_C), so
s_A = (D_A'/D_A) (L/L') ~ 1 + (e_A - e_C)/2 and s_C ~ 1 + (e_C - e_A)/2.  A
beam-axis offset along z is antisymmetric in the same way, and a common
reference-depth (t0) offset cancels exactly.  So a COMMON s > 1 on both arms is
not geometry.  This script asks whether it is population: the same focus scan,
split by time after the flash, height along the capsule, fit quality, charge,
whether other arms fired, and the slope sign.

    python -m ntof_cosmics.focus_split [--src .../stage3_is2_v2] [--label is2_v2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'focus_split'
SRC = Path('/media/dylan/data/x17/sept26_prelim/stage3_is2_v2')
SCALES = np.round(np.arange(0.80, 1.3001, 0.005), 3)
TAN_BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
U_HAT = {'A': np.array([1., 0., 0.]), 'C': np.array([-1., 0., 0.])}
W_HAT = {'A': np.array([0., 0., 1.]), 'C': np.array([0., 0., -1.])}
PRE_ACCESS = ('run_79', 'run_81')
CAPSULE_Y = (-29.5, 50.7)          # He-3 gas extent along the beam, geometry.HE3_GAS_Y
COLS = ['run', 'arm', 'gated', 'angle_calibrated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
        'tanx', 'tany', 't_since_flash_ns', 'target_y_mm', 'chi2dof_x', 'q_per_len',
        'n_coinc_arms', 'x_n_strips', 'wall_A', 'wall_C']


def load(src: Path, tan_max: float) -> pd.DataFrame:
    f = src / 'tracks_campaign.parquet'
    d = ds.dataset(f).to_table(columns=COLS, filter=(ds.field('arm').isin(['A', 'C'])
                                                    & ds.field('gated'))).to_pandas()
    d = d[d.angle_calibrated.astype('boolean').fillna(False).astype(bool)]
    d = d[~d.run.astype(str).isin(PRE_ACCESS)]
    d = d[(d.tanx.abs() < tan_max) & (d.tany.abs() < tan_max)].reset_index(drop=True)
    pu = np.empty(len(d))
    pw = np.empty(len(d))
    tx = np.empty(len(d))
    for arm in ('A', 'C'):
        m = (d.arm == arm).to_numpy()
        P0 = d.loc[m, ['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
        D = d.loc[m, ['d_x', 'd_y', 'd_z']].to_numpy(float)
        pu[m], pw[m] = P0 @ U_HAT[arm], P0 @ W_HAT[arm]
        with np.errstate(divide='ignore', invalid='ignore'):
            tx[m] = (D @ U_HAT[arm]) / (D @ W_HAT[arm])
    d['pu'], d['pw'], d['tx'] = pu, pw, tx
    d = d[np.isfinite(d.pu) & np.isfinite(d.pw) & np.isfinite(d.tx)]
    d['wall'] = np.where(d.arm == 'A', d.wall_A, d.wall_C).astype(bool)
    d['ms'] = d.t_since_flash_ns / 1e6
    return d.reset_index(drop=True)


def best_s(pu, pw, tx, nboot=30, rng=None):
    """Sharpest s (min median |miss|), refined by a parabola; bootstrap error."""
    if len(pu) < 200:
        return np.nan, np.nan, np.nan, len(pu)

    def one(pu, pw, tx):
        med = np.array([np.median(np.abs((pu - pw * s * tx) / np.hypot(s * tx, 1.0))) for s in SCALES])
        j = int(np.argmin(med))
        if 0 < j < len(SCALES) - 1:
            y0, y1, y2 = med[j - 1:j + 2]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
            return SCALES[j] + off * (SCALES[1] - SCALES[0]), med[j]
        return SCALES[j], med[j]

    s0, m0 = one(pu, pw, tx)
    rng = rng or np.random.default_rng(0)
    n = len(pu)
    sub = min(n, 100_000)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, n, sub)
        bs.append(one(pu[i], pw[i], tx[i])[0])
    # the bootstrap is on at most 100k; scale its spread to the full sample
    err = float(np.std(bs)) * np.sqrt(sub / n)
    return float(s0), err, float(m0), n


def splits(d: pd.DataFrame) -> dict:
    yin = d.target_y_mm.between(*CAPSULE_Y)
    q = d.q_per_len
    qmed = d.groupby('arm').q_per_len.transform('median')
    cmed = d.groupby('arm').chi2dof_x.transform('median')
    return {
        'all': np.ones(len(d), bool),
        'ms 1-5': d.ms.between(1, 5), 'ms 5-10': d.ms.between(5, 10),
        'ms 10-20': d.ms.between(10, 20), 'ms 20-80': d.ms.between(20, 80),
        'y in capsule': yin, 'y outside capsule': ~yin,
        'wall fired': d.wall, 'wall not fired': ~d.wall,
        'chi2 < median': d.chi2dof_x < cmed, 'chi2 > median': d.chi2dof_x >= cmed,
        'q/len < median': q < qmed, 'q/len > median': q >= qmed,
        'only this arm': d.n_coinc_arms <= 1, 'other arms too': d.n_coinc_arms > 1,
        'tanx > 0': d.tx > 0, 'tanx < 0': d.tx < 0,
        '|tany| < 0.15': d.tany.abs() < 0.15, '|tany| > 0.15': d.tany.abs() >= 0.15,
        'x strips <= 6': d.x_n_strips <= 6, 'x strips > 6': d.x_n_strips > 6,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--label', default='is2_v2')
    ap.add_argument('--tan-max', type=float, default=0.6)
    a = ap.parse_args()
    d = load(a.src, a.tan_max)
    print(f'{a.label}: {len(d):,} gated A/C tracks (post-access, |tan| < {a.tan_max})')
    S = splits(d)
    rows = []
    rng = np.random.default_rng(1)
    for name, m in S.items():
        m = np.asarray(m, bool)
        for arm in ('A', 'C'):
            ma = m & (d.arm == arm).to_numpy()
            for lo, hi in TAN_BINS + [(0.1, 0.5)]:
                mb = ma & (d.tx.abs() >= lo).to_numpy() & (d.tx.abs() < hi).to_numpy()
                s, e, med, n = best_s(d.pu.to_numpy()[mb], d.pw.to_numpy()[mb], d.tx.to_numpy()[mb], rng=rng)
                rows.append(dict(chain=a.label, split=name, arm=arm, tan_lo=lo, tan_hi=hi,
                                 s_best=s, s_err=e, med_miss=med, n=n))
        r = [x for x in rows if x['split'] == name and x['tan_lo'] == 0.1 and x['tan_hi'] == 0.5]
        print(f'  {name:20s} ' + '  '.join(f"{x['arm']} s {x['s_best']:.3f}±{x['s_err']:.3f} "
                                         f"miss {x['med_miss']:5.1f} n {x['n']:>8,}" for x in r), flush=True)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / f'focus_split_{a.label}.csv', index=False)
    print(f'-> {OUT / f"focus_split_{a.label}.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
