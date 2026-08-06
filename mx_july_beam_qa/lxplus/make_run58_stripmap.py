#!/usr/bin/env python3
"""
make_run58_stripmap.py — freeze the (detector, FEU) -> strip-position map into a
small npz so the condor worker needs no repo imports.

The worker (`run58_columns.py`) must be standalone: lxplus batch nodes have the
LCG view (numpy/pandas/uproot) but not this repo, and shipping `common/` plus
`mx17_m1_map.csv` plus a 180 kB run_config just to call `map_hit` 512 times is
silly. This writes `run58_stripmap.npz` with one 512-long float array per
(detector, plane), which is all the worker actually uses.

    ../../.venv/bin/python mx_july_beam_qa/lxplus/make_run58_stripmap.py \
        [--run-config <path>] [--out mx_july_beam_qa/lxplus/run58_stripmap.npz]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

# beam detector -> (FEU x, FEU y); run_config `dream_feus`, same for run_58/79
BEAM_DETS = {'A': (3, 4), 'B': (5, 6), 'C': (7, 8), 'D': (1, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-config', default=None,
                    help='a July run_config.json (default: the local run_79 one; '
                         'the mapping is identical across the July campaign)')
    ap.add_argument('--out', default=os.path.join(HERE, 'run58_stripmap.npz'))
    args = ap.parse_args()

    cfg = args.run_config or ('/media/dylan/data/x17/beam_july/runs/run_79/'
                              'run_config.json')
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(cfg, os.path.join(REPO, 'mx17_m1_map.csv'))
    out = {}
    for letter, (fx, fy) in BEAM_DETS.items():
        det = rc.get_detector(f'mx17_{letter}')
        for feu, axis, plane in ((fx, 0, 'x'), (fy, 1, 'y')):
            p = np.full(512, np.nan)
            for ch in range(512):
                h = det.map_hit(feu, ch)
                if h is not None and h[axis] is not None:
                    p[ch] = h[axis]
            out[f'{letter}{plane}'] = p
            out[f'{letter}{plane}_feu'] = np.array([feu])
            n = int(np.isfinite(p).sum())
            print(f'  mx17_{letter} {plane}: FEU {feu}, {n} mapped strips, '
                  f'{np.nanmin(p):.1f}-{np.nanmax(p):.1f} mm')
    np.savez_compressed(args.out, **out)
    print('wrote', args.out, os.path.getsize(args.out), 'bytes')


if __name__ == '__main__':
    main()
