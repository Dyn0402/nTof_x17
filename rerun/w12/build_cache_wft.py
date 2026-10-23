#!/usr/bin/env python3
"""build_cache_wft.py <run_key> --align <alignment.json> --out <cache.pkl> [--n 1500] [--cm 0]

W12: the calibration cache with the M3 reference placed by the WAVEFORM alignment (mx_june_wft/01_alignment.py on
the previous iteration's reconstruction) instead of the hits alignment (C8).  Same event selection and windows as
wft.calibrate.build_cache; --cm sets wft.io's common-mode mask (0 = plain median).  Run through rerun/stage_job.py.
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--align', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--cm', type=int, default=0)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio
    from wft.calibrate import build_cache
    wio.CM_MASK = a.cm
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ev = build_cache(get_config(a.run_key), a.n, out_path=a.out, align_path=a.align)
    print(f'[w12] {len(ev)} events, reference from {a.align} -> {a.out}')


if __name__ == '__main__':
    main()
