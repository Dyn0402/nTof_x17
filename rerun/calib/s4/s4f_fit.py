#!/usr/bin/env python3
"""s4f_fit.py --spec SPEC.json --out RESULT.json [--nboot 20] [--workers 8] -- S4 follow-up: the S4 head-on stack fit
(s4_stacks.fit_stack: x_width.measure on the events whose X window stays below the saturation cut, |tan_x| < 0.03) on
a grid of arms, each arm one choice of

    cache      a headon.pkl (s4_stacks.py: rr1 decode; s4f_build.py --gen prod: the pre-rerun decode), both ref-only
               through the pass alignment;
    selection  'all' (S4) or a named event list from s4f_build.py's selections.json (the hits-chain events the pre-pass
               input read: has_x and has_y, radial residual < 10 mm);
    template   an arrays.npz whose tmpl_x is the electronics response (the pass bundle, or the pre-pass bundle the input
               used: det4 calib_bundle_lp, det7 calib_bundle_r06);
    sat        the per-event saturation cut on the X window maximum (ADC; null = no cut); amp_lo an optional lower
               bound on the same maximum (amplitude bins); sel_not = true takes the complement of the event list;
               ch_min keeps the events whose X peak strip is at or above that channel (det7 X: ch 256-511 only).

SPEC: {"T0": ns, "arms": {name: {"cache": path, "sel": [path, name] | null, "sel_not": bool, "template": path,
                                 "sat": float | null, "amp_lo": float}}}
Dp = sqrt(2 Dd); v_gap(g) = g / T.  Writes RESULT.json {name: {n, T, T_err, Dd, Dd_err, Dp, Dp_err, sig0, ...}}.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_stacks  # noqa: E402  (sys.path for cloud_basics)
import x_width as xw  # noqa: E402

_CACHE = {}


def load(path):
    if path not in _CACHE:
        _CACHE[path] = pickle.load(open(path, 'rb'))
    return _CACHE[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--nboot', type=int, default=20); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only', default=None, help='comma-separated arm names')
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    names = a.only.split(',') if a.only else list(spec['arms'])
    with ProcessPoolExecutor(a.workers) as pool:
        for name in names:
            arm = spec['arms'][name]
            ev = load(arm['cache'])
            if arm.get('sel'):
                keep = set(json.load(open(arm['sel'][0]))[arm['sel'][1]])
                ev = {k: e for k, e in ev.items() if (k in keep) != bool(arm.get('sel_not'))}
            if arm.get('ch_min') is not None:                  # X peak strip at or above ch_min (det7: ch 256-511)
                ev = {k: e for k, e in ev.items() if e['x']['ch'][np.argmax(e['x']['W'].max(axis=1))] >= arm['ch_min']}
            n_head = sum(abs(e['tan_x']) < xw.TAN for e in ev.values())
            sat, lo = arm.get('sat'), arm.get('amp_lo', 0.0)
            uns = {k: e for k, e in ev.items() if lo <= e['x']['W'].max() < (np.inf if sat is None else sat)}
            n = sum(abs(e['tan_x']) < xw.TAN for e in uns.values())
            r = dict(arm, n_headon=int(n_head), n=int(n))
            if n >= xw.MIN_EV:
                t0 = time.time()
                m = xw.measure(pool, s4_stacks.pass_template(os.path.dirname(arm['template'])), uns, spec['T0'], a.nboot)
                m['Dp'] = float(np.sqrt(2 * m['Dd']))
                m['Dp_err'] = float(m['Dd_err'] / np.sqrt(2 * m['Dd'])) if m['Dd'] > 0 else float('nan')
                m['T_err'] = float(m['T'] * m['v_err'] / m['v']) if m['v'] > 0 else float('nan')
                r.update(m)
                print(f'{name:28s} n {n:5d}/{n_head:5d}  T {m["T"]:6.1f}±{m["T_err"]:4.1f}  Dp {m["Dp"]:.4f}±{m["Dp_err"]:.4f}  '
                      f'sig0 {m["sig0"]:.3f}  ({time.time() - t0:.0f} s)', flush=True)
            else:
                print(f'{name:28s} n {n} < {xw.MIN_EV}: not fitted', flush=True)
            out[name] = r
            json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
