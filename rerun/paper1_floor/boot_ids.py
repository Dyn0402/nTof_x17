#!/usr/bin/env python3
"""boot_ids.py --out D -- the clean events of the residual bootstrap (BOOTSTRAP_2026-11-01.md): per calibration row and
view, the event ids of analyze.py's 'topo' step (the clean sample of the §8 ladder: unsaturated, amplitude ranks
10-80 %, one candidate, one track, no band strip, strip-count excess and multiplicity below their 90th percentile),
with the event's tan_ref and dt (the residual after the slope line and the 2D 40 mm tilt map, FD's convention).

Run where the frames are (FLOOR_FRAMES, lxplus: .../x17rerun/paper1_floor/frames).
"""
import argparse
import json
import os

import numpy as np

import analyze as A
import floorlib as C

ROWS = dict(det2='o22_long_det2', det3='sat_det3', det4='g_det4', det7='g_det7_long')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for det, key in ROWS.items():
        F = C.load(key)
        rec = dict(det=det, key=key)
        for p in ('x', 'y'):
            g = A.derived(C.sample(F, p))
            topo = dict(A.ladder(g))['topo']
            s = g[topo]
            rec[p] = dict(event_id=[int(e) for e in s.event_id], tan_ref=[float(t) for t in s.tan_ref],
                          dt=[float(d) for d in s.dt], s68=C.s68(s.dt), n=int(len(s)))
            print(det, key, p, len(s), f's68 {rec[p]["s68"]:.3f}')
        json.dump(rec, open(os.path.join(a.out, f'{key}.json'), 'w'))


if __name__ == '__main__':
    main()
