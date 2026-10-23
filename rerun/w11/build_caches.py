#!/usr/bin/env python3
"""build_caches.py <run_key> [--n 1500] -- W11: the same calibration cache twice, plain and masked common mode.

Same events, same M3 reference, same windows (wft.calibrate.build_cache); only wft.io's common mode differs:
CM_MASK = 0 (plain block median, production) and CM_MASK = 5 (signal-masked, with the coherent-event guard).
The first 180 events (lowest ids) are the calibration training set, the rest are held out.
Output: <OUT_BASE>/wft/w11/cache_cm0.pkl, cache_cm5.pkl.  Run through rerun/stage_job.py (needs decoded + the hits
alignment of 03_alignment_and_tpc.py).
"""
import argparse
import os
import pickle
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--n', type=int, default=1500)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft import io as wio
    from wft.calibrate import build_cache
    cfg = get_config(a.run_key)
    d = cfg.out_dir('wft', 'w11')
    ids = {}
    for m in (0, 5):
        wio.CM_MASK = m
        out = os.path.join(d, f'cache_cm{m}.pkl')
        build_cache(cfg, a.n, out_path=out)
        ids[m] = set(pickle.load(open(out, 'rb')))
        print(f'[w11] CM_MASK={m}: {len(ids[m])} events -> {out}', flush=True)
    print(f'[w11] events in both {len(ids[0] & ids[5])}, only plain {len(ids[0] - ids[5])}, '
          f'only masked {len(ids[5] - ids[0])}')


if __name__ == '__main__':
    main()
