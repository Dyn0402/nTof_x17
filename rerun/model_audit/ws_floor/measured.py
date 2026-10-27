#!/usr/bin/env python3
"""measured.py -- the measured per-event angle resolution vs angle, per chamber and view, for comparison with the
floor toy.  Source: the finalize tables pulled by the model audit (<det>/final_physmask/angles/per_event.npz:
M3-matched single tracks, the production fit's w on the promoted w0/kw against the M3 tan).  s68 (core, the
fleet-table convention: far > 5 deg set aside and quoted separately) of (theta_fit - theta_ref) in bins of
|theta_ref|; the M3 reference contributes <= 0.15 deg (pointing slope 0.02 deg + multiple scattering, memory
m3-self-resolution) and is not subtracted.

    measured.py   -> results/measured.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
EDGES = [0, 3, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 45]


def s68(a):
    """the fleet-table convention: 68th percentile of |d - median| after the far (> 5 deg) tail is set aside"""
    a = np.asarray(a)
    a = a[np.abs(a) <= 5]
    return float(np.percentile(np.abs(a - np.median(a)), 68))


def main():
    out = {}
    for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
        z = np.load(os.path.join(AUDIT, det, 'final_physmask', 'angles', 'per_event.npz'))
        b = json.load(open(os.path.join(AUDIT, det, 'calib_bundle', 'bundle.json')))
        for v in ('x', 'y'):
            # tan on the promoted constants, as fleet_tables.py (the table carries the pre-finalize w0/kw)
            tr = z[f'{v}_tan_ref']
            tf = (z[f'{v}_w'] * 1e3 - b['w0'][v]) / (b['kw'][v] * float(b['v_drift']))
            ok = np.isfinite(tr) & np.isfinite(tf)
            a_ref = np.degrees(np.arctan(tr[ok]))
            d = np.degrees(np.arctan(tf[ok])) - a_ref
            rows = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                m = (np.abs(a_ref) >= lo) & (np.abs(a_ref) < hi)
                if m.sum() < 30:
                    continue
                rows.append(dict(lo=lo, hi=hi, mid=float(np.median(np.abs(a_ref[m]))), n=int(m.sum()),
                                 s68=s68(d[m]), far5=float(np.mean(np.abs(d[m]) > 5))))
            out[f'{det}_{v}'] = dict(all_s68=s68(d), far5=float(np.mean(np.abs(d) > 5)), n=int(ok.sum()), bins=rows)
            print(det, v, f"all {s68(d):.3f}", ' '.join(f"{r['mid']:.0f}:{r['s68']:.2f}" for r in rows))
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'measured.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
