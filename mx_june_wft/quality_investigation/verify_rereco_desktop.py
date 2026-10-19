#!/usr/bin/env python3
"""
verify_rereco_desktop.py — independent verification that the campaign det3
reconstruction reproduces: re-reconstruct a sample of golden events on a
DIFFERENT machine from a DIFFERENT copy of the raw data with the same frozen
code + bundle, and compare per-event against the promoted campaign parquet.

Run on the desktop, from the fleetcheck worktree:
    ~/PycharmProjects/nTof_x17/.venv/bin/python \
        mx_june_wft/quality_investigation/verify_rereco_desktop.py \
        --bundle ~/fleetcheck_data/calib_bundle_lp2_t0p \
        --reference ~/fleetcheck_data/events.parquet \
        --out ~/fleetcheck_data/rereco --limit 400 --jobs 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths, M3_CHI2_CUT, M3_MIN_NCLUS
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run
    from M3RefTracking import M3RefTracking, get_xy_angles

    cfg = get_config('sat_det3')
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    _xa, _ya, evn = get_xy_angles(rays.ray_data)
    filt = set(int(e) for e in evn)
    print(f'{len(filt):,} M3-matched events on this machine')

    cal = CalibrationBundle.load(os.path.expanduser(args.bundle))
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)
    out = os.path.join(os.path.expanduser(args.out), 'events.parquet')
    reconstruct_run(cfg, cal, out, event_filter=filt, jobs=args.jobs,
                    limit=args.limit,
                    bundle_path=os.path.expanduser(args.bundle))

    new = pd.read_parquet(out).set_index('event_id')
    ref = pd.read_parquet(os.path.expanduser(args.reference)) \
        .set_index('event_id')
    shared = new.index.intersection(ref.index)
    print(f'\n=== comparison on {len(shared)} shared events')
    n, r = new.loc[shared], ref.loc[shared]
    for col in ('x_p0', 'y_p0', 'x_w', 'y_w', 'x_t0', 'y_t0',
                'x_chi2', 'y_chi2', 'x_q_sum'):
        if col not in n or col not in r:
            continue
        a, b = n[col].astype(float), r[col].astype(float)
        both = a.notna() & b.notna()
        d = (a[both] - b[both]).abs()
        agree = a.isna().eq(b.isna()).mean()
        print(f'{col:8s} nan-pattern agree {agree*100:6.2f}%  '
              f'median|Δ| {d.median():.3e}  max|Δ| {d.max():.3e}  '
              f'frac|Δ|>1e-6 {(d > 1e-6).mean()*100:.2f}%')
    for col in ('x_ok', 'y_ok', 'x_quality_ok', 'y_quality_ok', 'n_tracks'):
        if col in n and col in r:
            m = (n[col] == r[col]).mean()
            print(f'{col:14s} identical {m*100:6.2f}%')


if __name__ == '__main__':
    main()
