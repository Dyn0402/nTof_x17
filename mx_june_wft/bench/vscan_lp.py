#!/usr/bin/env python3
"""
vscan_lp.py — ref-pinned chi2(v) scan under a given calibration bundle.

The ensemble drift-velocity measurement (R&D script 19): fix each plane's
(p0, w) to the M3 reference with w = tan_ref * v, profile t0 and the charge
amplitudes, and scan v. The minimum is the ensemble v under THIS kernel.
Everything the gap analysis quotes in mm scales with it.

    ../../.venv/bin/python mx_june_wft/bench/vscan_lp.py sat_det3 \
        --bundle <dir> [--n 200] [--jobs 8]
"""
import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--vgrid', default='33,40.5,0.5')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    import wft.calibrate as wc
    from wft.calib import CalibrationBundle
    cfg = get_config(args.run_key)
    cache = os.path.join(cfg.OUT_BASE, 'wft', 'calib_work', 'calib_cache.pkl')
    ids = sorted(pickle.load(open(cache, 'rb')))[:args.n]
    cal = CalibrationBundle.load(args.bundle)
    lo, hi, step = (float(x) for x in args.vgrid.split(','))
    vs = np.arange(lo, hi + 1e-9, step)

    warm = {e: {} for e in ids}
    print(f'{len(ids)} ref-pinned events, kernel share_lp='
          f'{cal.hyper.get("share_lp", 0)}')
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=wc._init_hyper,
                             initargs=(cache, args.bundle)) as pool:
        # two warm passes at the central v so the t0 grids are converged and
        # identical treatment applies at every scan point
        for _ in range(2):
            for eid, c, t0s in pool.map(
                    wc._event_chi2,
                    [(e, dict(cal.hyper), cal.v_drift, warm[e]) for e in ids],
                    chunksize=6):
                warm[eid] = t0s
        out = []
        for v in vs:
            tot = 0.0
            for eid, c, _t0s in pool.map(
                    wc._event_chi2,
                    [(e, dict(cal.hyper), float(v), warm[e]) for e in ids],
                    chunksize=6):
                tot += c
            out.append(tot)
            print(f'  v={v:5.2f}  chi2={tot:.6e}', flush=True)
    out = np.array(out)
    j = int(np.argmin(out))
    # parabolic refinement
    if 0 < j < len(vs) - 1:
        a, b, c = out[j - 1], out[j], out[j + 1]
        den = a - 2 * b + c
        vbest = vs[j] + (0.5 * (a - c) / den if den > 0 else 0.0) * step
    else:
        vbest = vs[j]
    print(f'chi2(v) minimum: v = {vbest:.2f} um/ns  '
          f'(bundle v = {cal.v_drift:.2f})')


if __name__ == '__main__':
    main()
