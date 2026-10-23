#!/usr/bin/env python3
"""v_scan.py -- chi2(v) of the ref-pinned calibration objective, with p0 pinned to the M3 reference vs profiled.

Pre-run check (rerun/CHECKS_2026-10-23.md C8).  The RC-kernel calibration with v free (recal.py) returns v above dry
Magboltz on det3, while free fits read the charge moving at ~35 um/ns.  The calibration pins p0 at ``ref_mesh``, the
M3 track at the plane the HITS-chain alignment placed the detector; a depth offset dz there is absorbed as
dv ~ dz / <drift time>.  This scans v with every other hyper fixed (an arm JSON) and evaluates recal.py's cold
objective twice: p0 pinned (calibration as run) and p0 profiled +-1 mm (recal.py --p0-profile).

    v_scan.py --cache calib_cache.pkl --bundle <bundle> --arm arm.json --vs 33,35,37,39,42 --n 60 --out vscan.json
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'plane_ratio')]
import recal as R                                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--vs', default='31,33,35,37,39,41,43')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--jobs', type=int, default=7)
    ap.add_argument('--depth-bins', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    h = {k: float(q) for k, q in json.load(open(a.arm))['hyper'].items()}
    ids = sorted(pickle.load(open(a.cache, 'rb')))[:a.n]
    vs = [float(x) for x in a.vs.split(',')]
    out = dict(arm=a.arm, n=len(ids), vs=vs, pinned=[], profiled=[])
    for prof in (False, True):
        with ProcessPoolExecutor(a.jobs, initializer=R._init_worker,
                                 initargs=(a.cache, a.bundle, a.depth_bins, 120.0, 960.0, prof)) as pool:
            for v in vs:
                c = sum(t for _e, t in pool.map(R._event_chi2_cold, [(e, h, v) for e in ids], chunksize=2))
                out['profiled' if prof else 'pinned'].append(float(c))
                print(f'{"profiled" if prof else "pinned  "} v={v:5.1f} chi2={c:.6e}', flush=True)
    for k in ('pinned', 'profiled'):
        c = np.array(out[k])
        p = np.polyfit(vs, c, 2)
        out[f'{k}_vmin'] = float(-p[1] / (2 * p[0])) if p[0] > 0 else None
        print(k, 'parabola minimum v =', out[f'{k}_vmin'], ' grid minimum v =', vs[int(np.argmin(c))])
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
