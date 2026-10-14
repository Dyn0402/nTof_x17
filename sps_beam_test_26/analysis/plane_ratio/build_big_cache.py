#!/usr/bin/env python3
"""build_big_cache.py -- a larger reference-corridor waveform cache for the
per-view kernel study, written NEXT to the calibration cache, never over it.

The calibration cache holds 400 events (180 train + 220 held-out), which gives
~100 near-vertical events per view and a held-out angle sample too small for a
0.05 deg effect.  build_cache selects the lowest event ids first, so the first
400 events of this cache are the calibration cache's events; the study's
held-out set is everything here that is NOT in the calibration training set.

    build_big_cache.py <run_key> [--n 3000]
Output: <OUT_BASE>/wft/plane_ratio/big_cache_<n>.pkl
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--n', type=int, default=3000)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calibrate import build_cache
    cfg = get_config(a.run_key)
    out = os.path.join(cfg.out_dir('wft', 'plane_ratio'), f'big_cache_{a.n}.pkl')
    build_cache(cfg, a.n, out_path=out)
    print('wrote', out)


if __name__ == '__main__':
    main()
