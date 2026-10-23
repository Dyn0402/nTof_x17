#!/usr/bin/env python3
"""set_w0kw.py <run_key> --bundle <src bundle> --angles <03_angles out dir> --out <bundle dir>   (W6)

Measure the angle-mapping constants of a bundle from a reconstruction made WITH that bundle, and write a copy of the
bundle carrying them.  Same estimator as mx_june_wft/bench/set_w0.py, but read from 03_angles' per-event table
(fitted w and the M3 angle through the waveform alignment):

    w0 = median(w * 1e3 - v * tan_ref)                     over |tan_ref| < 0.30
    kw = median((w * 1e3 - w0) / (v * tan_ref))            over 0.10 < |tan_ref| < 0.40

Under a kernel calibrated with the right v and reference, w0 ~ 0 and kw ~ 1; a large kw is a calibration failure,
not a constant (RERUN_PLAN W6).  The w0/kw an RC bundle inherits from its r06 source bundle are another kernel's.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--angles', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(a.bundle)
    v = float(cal.v_drift)
    z = np.load(os.path.join(a.angles, 'per_event.npz'))
    w0, kw, rep = {}, {}, {}
    for p in ('x', 'y'):
        w, t = z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref']
        ok = np.isfinite(w) & np.isfinite(t)
        s = ok & (np.abs(t) < 0.30)
        d = w[s] - v * t[s]
        w0[p] = float(np.median(d))
        s1 = ok & (np.abs(t) > 0.10) & (np.abs(t) < 0.40)
        kw[p] = float(np.median((w[s1] - w0[p]) / (v * t[s1]))) if s1.sum() >= 30 else 1.0
        rep[p] = dict(w0=w0[p], kw=kw[p], n_w0=int(s.sum()), n_kw=int(s1.sum()),
                      angle_bias_if_uncorrected_deg=float(np.degrees(w0[p] / v)),
                      old_w0=float(cal.w0.get(p, 0.0)), old_kw=float(cal.kw.get(p, 1.0)))
        print(f'{p}: w0 {w0[p]:+.3f} um/ns ({np.degrees(w0[p] / v):+.2f} deg), kw {kw[p]:.4f}  '
              f'[was w0 {rep[p]["old_w0"]:+.3f}, kw {rep[p]["old_kw"]:.4f}]  n={int(s.sum())}/{int(s1.sum())}')
    cal.w0, cal.kw = w0, kw
    prov = dict(cal.provenance)
    prov.update(w0_kw_stale=False, w0_kw_note=f'w0/kw measured by rerun/w12/set_w0kw.py from {a.angles} '
                f'(a reconstruction with this bundle, {a.run_key})', w0_kw_measurement=rep)
    cal.provenance = prov
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    cal.save(a.out, note=f'w0/kw measured ({a.run_key})')
    json.dump(rep, open(os.path.join(a.out, 'w0kw.json'), 'w'), indent=1)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
