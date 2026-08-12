#!/usr/bin/env python3
"""
10_cx0_refit.py — T1.2 model arm: kill the discrete X sharing kernel (cX = 0)
and let the transverse-spread terms absorb it.

F6's hypothesis: the X view cannot have resistive sharing (the ESL strips run
along y, grooves block transport across x), so X's ±1 copy should be diffusion
— already in the model as sigma_p0/Dp — and the constant discrete c1 on X is
the same physics booked twice. This refits (sigma_p0, Dp) with cX = 0 and the
rest of the kernel pinned at the production lp2 values, ref-pinned on the
calibration cache. (No per-plane tau: the naive kTauY port regressed —
KERNEL_ARMS_2026-08-12.md — so the Y kernel stays as lp2 shipped it.)

The result is meant for a bench --patch arm, judged on implied-velocity
flatness (ground rule 3), NOT adopted on chi2.

    ../.venv/bin/python mx_june_wft/10_cx0_refit.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/cx0_refit.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--maxiter', type=int, default=50)
    ap.add_argument('--n-train', type=int, default=180)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import calibrate as wc
    from wft.calib import CalibrationBundle, HYPER_NAMES

    cfg = get_config(args.run_key)
    bundle = args.bundle or os.path.join(cfg.OUT_BASE, 'wft', 'calib_bundle_lp2')
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')
    out_dir = cfg.out_dir('wft', 'kernel_arms')
    os.makedirs(out_dir, exist_ok=True)

    cal = CalibrationBundle.load(bundle)
    h = cal.hyper
    x0 = np.array([h[k] for k in HYPER_NAMES] + [cal.v_drift])
    fixed = {k: float(h[k]) for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s')}
    extra = {'share_lp': 1.0, 'cX': 0.0}

    with open(cache, 'rb') as f:
        eids = sorted(pickle.load(f).keys())
    train = eids[:args.n_train]
    print(f'[cx0] bundle {os.path.basename(bundle)}, {len(train)} train events')
    print(f'[cx0] fixed {fixed}')
    print(f'[cx0] extra (not fitted) {extra}; free: sigma_p0, Dp; '
          f'v pinned {cal.v_drift}')

    res = wc.fit_hypers(cache, bundle, train, jobs=args.jobs,
                        maxiter=args.maxiter, x0=x0, v_fixed=cal.v_drift,
                        fixed=fixed, extra_hyper=extra)
    res['extra_hyper'] = extra
    res['bundle'] = bundle
    res['x0'] = {k: float(v) for k, v in zip(HYPER_NAMES, x0[:7])}
    out = os.path.join(out_dir, 'cx0_refit.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'[cx0] sigma_p0 {res["sigma_p0"]:.4f} (was {h["sigma_p0"]:.4f}), '
          f'Dp {res["Dp"]:.5f} (was {h["Dp"]:.5f}), '
          f'chi2 {res["chi2"]:.4e} vs init {res["chi2_init"]:.4e}')
    print(f'[cx0] wrote {out}')


if __name__ == '__main__':
    main()
