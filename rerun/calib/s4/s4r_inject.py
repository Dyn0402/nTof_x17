#!/usr/bin/env python3
"""s4r_inject.py --cache headon.pkl --template DIR --T0 ns --out R.json [--scales 1,0.5,0.3] -- is the longer stack T of
low-amplitude events (det4 +60 ns below 1000 ADC, det3/det6 ~+85 ns) the estimator or the chamber?

Takes the unsaturated head-on events at 2000-3500 ADC (X window maximum), scales each event's X waveforms by s and adds
Gaussian noise of the strip's own sigma x sqrt(1 - s^2), so the injected event has the real noise level at a fraction s
of its amplitude (2000-3500 -> 600-1050 ADC at s = 0.3).  The drift is the same events' drift, so any change of the
fitted T with s is the stack estimator (threshold alignment, fit) on weaker signals, not the chamber.
Fits as s4f_fit.py (x_width.measure, |tan_x| < 0.03, bootstrap); 3 noise seeds per scale.
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_stacks  # noqa: E402  (sys.path for cloud_basics)
import x_width as xw  # noqa: E402


def inject(ev, s, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k, e in ev.items():
        W = np.asarray(e['x']['W'], float)
        sig = np.asarray(e['x']['noise'], float)[:, None] * np.sqrt(max(1 - s * s, 0.0))
        out[k] = dict(e, x=dict(e['x'], W=s * W + rng.normal(size=W.shape) * sig))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--template', required=True)
    ap.add_argument('--T0', type=float, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--scales', default='1,0.5,0.3'); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--lo', type=float, default=2000.0); ap.add_argument('--hi', type=float, default=3500.0)
    ap.add_argument('--nboot', type=int, default=20); ap.add_argument('--workers', type=int, default=7)
    a = ap.parse_args()
    ev = pickle.load(open(a.cache, 'rb'))
    base = {k: e for k, e in ev.items() if a.lo <= e['x']['W'].max() < a.hi}
    h = s4_stacks.pass_template(a.template)
    out = dict(cache=a.cache, template=a.template, lo=a.lo, hi=a.hi, arms={})
    with ProcessPoolExecutor(a.workers) as pool:
        for s in [float(x) for x in a.scales.split(',')]:
            for seed in range(1 if s == 1.0 else a.seeds):
                sub = base if s == 1.0 else inject(base, s, seed)
                amp = np.median([e['x']['W'].max() for e in sub.values() if abs(e['tan_x']) < xw.TAN])
                m = xw.measure(pool, h, sub, a.T0, a.nboot)
                m['Dp'] = float(np.sqrt(2 * m['Dd'])) if m['Dd'] > 0 else 0.0
                m['T_err'] = float(m['T'] * m['v_err'] / m['v']) if m['v'] > 0 else float('nan')
                m['amp_median'] = float(amp)
                out['arms'][f's{s:g}_seed{seed}'] = m
                print(f's {s:4.2f} seed {seed}: n {m["n"]:4d}  amp {amp:6.0f}  T {m["T"]:6.1f}±{m["T_err"]:4.1f}  '
                      f'Dp {m["Dp"]:.4f}  sig0 {m["sig0"]:.3f}', flush=True)
                json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
