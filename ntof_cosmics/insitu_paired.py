#!/usr/bin/env python3
"""insitu_paired.py -- paired comparison of two in-situ reco tables on the A-C
cosmic truth (insitu_calib.py), with bootstrap errors on the differences.

Each label gets its own per-plane scale from the TRAIN third (median w/tan_true,
|tan| 0.15-0.45, as `insitu_calib.compare_tables`); on the TEST two thirds, the
events both labels reconstructed are paired and resampled together.  Per
|tan_true| bin: MAD resolution and median response, and their differences
(arm - base) with a paired bootstrap.  Also the response slope across the core
(the S-shape: median ratio at 0.35-0.6 minus at 0.08-0.15).

    python ntof_cosmics/insitu_paired.py --work ~/scratch/ntof_insitu --arm A --base is2ts_s3 --arm-label rcm3ts_s3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BINS = ((0.0, 0.08), (0.08, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.6), (0.08, 0.6))
NBOOT = 1000


def _mad(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)), axis=-1)


def scaled(work: Path, arm: str, label: str) -> pd.DataFrame:
    T = pd.read_parquet(work / 'truth.parquet')
    T = T[T.arm == arm]
    M = T.merge(pd.read_parquet(work / f'reco_{label}_{arm}.parquet'), on=['subrun', 'event_id'])
    out = M[['subrun', 'event_id', 'train']].copy()
    for ax in 'xy':
        t, w = M[f'tan_{ax}'].to_numpy(), M[f'{ax}_w'].to_numpy() * 1e3
        tr = np.isfinite(w) & M.train.to_numpy() & (np.abs(t) > 0.15) & (np.abs(t) < 0.45)
        out[f't_{ax}'] = t
        out[f'r_{ax}'] = w / float(np.median(w[tr] / t[tr]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--arm-label', required=True)
    a = ap.parse_args()
    B = scaled(a.work, a.arm, a.base)
    C = scaled(a.work, a.arm, a.arm_label)
    P = B.merge(C, on=['subrun', 'event_id', 'train'], suffixes=('_b', '_a'))
    P = P[~P.train]
    rng = np.random.default_rng(0)
    rows = []
    for ax in 'xy':
        t = P[f't_{ax}_b'].to_numpy()
        rb, ra = P[f'r_{ax}_b'].to_numpy(), P[f'r_{ax}_a'].to_numpy()
        ok = np.isfinite(rb) & np.isfinite(ra)
        for lo, hi in BINS:
            m = ok & (np.abs(t) >= lo) & (np.abs(t) < hi)
            n = int(m.sum())
            eb, ea = rb[m] - t[m], ra[m] - t[m]
            idx = rng.integers(0, n, (NBOOT, n))
            dsig = _mad(ea[idx]) - _mad(eb[idx])
            row = dict(arm=a.arm, axis=ax, lo=lo, hi=hi, n=n,
                       sigma_base=float(_mad(eb)), sigma_arm=float(_mad(ea)),
                       dsigma=float(_mad(ea) - _mad(eb)), dsigma_err=float(np.std(dsig)))
            if lo > 0:
                qb, qa = rb[m] / t[m], ra[m] / t[m]
                dr = np.median(qa[idx], axis=1) - np.median(qb[idx], axis=1)
                row.update(ratio_base=float(np.median(qb)), ratio_arm=float(np.median(qa)),
                           dratio_err=float(np.std(dr)))
            rows.append(row)
        # S-shape: response at large angle minus at small, per label, paired
        lo_m = ok & (np.abs(t) >= 0.08) & (np.abs(t) < 0.15)
        hi_m = ok & (np.abs(t) >= 0.35) & (np.abs(t) < 0.6)
        s_b = np.median(rb[hi_m] / t[hi_m]) - np.median(rb[lo_m] / t[lo_m])
        s_a = np.median(ra[hi_m] / t[hi_m]) - np.median(ra[lo_m] / t[lo_m])
        il, ih = np.where(lo_m)[0], np.where(hi_m)[0]
        bs = []
        for _ in range(NBOOT):
            jl, jh = rng.choice(il, len(il)), rng.choice(ih, len(ih))
            bs.append((np.median(ra[jh] / t[jh]) - np.median(ra[jl] / t[jl]))
                      - (np.median(rb[jh] / t[jh]) - np.median(rb[jl] / t[jl])))
        rows.append(dict(arm=a.arm, axis=ax, lo=-1, hi=-1, n=int(len(il) + len(ih)),
                         sshape_base=float(s_b), sshape_arm=float(s_a), dsshape_err=float(np.std(bs))))
    R = pd.DataFrame(rows)
    out = a.work / f'paired_{a.arm}_{a.base}_vs_{a.arm_label}.csv'
    R.to_csv(out, index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 30):
        print(R.round(4).to_string(index=False))
    print(f'-> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
