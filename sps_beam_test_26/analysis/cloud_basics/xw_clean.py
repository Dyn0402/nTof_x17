#!/usr/bin/env python3
"""xw_clean.py -- the X width fit and the neighbour/peak ratio on 'clean' events only (strip_gain_audit.py flags).

clean = no dead/weak X strip within +-4 of the cluster centre and |d ln g / dx| x 3 mm < GRAD_MAX at the
centre.  If det4/det7's amplitude-dependent D_T (FINDINGS §32-34) comes from the gain structure under
the window, it should vanish on clean events and concentrate in the rest.

    xw_clean.py [--keys g_det4,...] [--grad 0.15] [--cuts 3500,2000] [--sig0-fix 0.35]  -> results/xw_clean.json
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import x_width as xw
import xw_compress as XC

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ['sat_det3__long_run_resist_490V_drift_1000V', 'g_det4', 'g_det7_long', 'g_det7_longer', 'g_det6_long']


def subsets(ev, F, grad):
    def clean(k):
        f = F.get(k, {})
        return f.get('bad_x', 1) == 0 and abs(f.get('dlng_x', np.nan)) < grad
    good = {k: e for k, e in ev.items() if clean(k)}
    rest = {k: e for k, e in ev.items() if k not in good}
    return dict(all=ev, clean=good, rest=rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(KEYS))
    ap.add_argument('--grad', type=float, default=0.15)
    ap.add_argument('--cuts', default='3500,2000')
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--sig0-fix', type=float, help='pin sigma_0 (and T at T0) in the width fits; key suffix @s0<val>')
    a = ap.parse_args()
    path = f'{HERE}/results/xw_clean.json'
    out = json.load(open(path)) if os.path.exists(path) else {}
    with ProcessPoolExecutor(a.workers) as pool:
        for key in a.keys.split(','):
            det = xw.det_of(key)
            h, _ = xw.template(det)
            ev = pickle.load(open(f'{xw.CACHES}/headon_{key}.pkl', 'rb'))
            F = pickle.load(open(f'{xw.CACHES}/gainflags_{key}.pkl', 'rb'))
            T0 = xw.GAP / (xw.V_BUNDLE[det] * 1e-3)
            r = dict(det=det, grad_max=a.grad)
            for lab, sub in subsets(ev, F, a.grad).items():
                A, R = XC.ratios(sub)
                r[lab] = dict(n_headon=int(sum(abs(e['tan_x']) < xw.TAN for e in sub.values())),
                              ratio=XC.binned(A, R))
                for cut in [float(c) for c in a.cuts.split(',')]:
                    uns = {k: e for k, e in sub.items() if e['x']['W'].max() < cut}
                    n = sum(abs(e['tan_x']) < xw.TAN for e in uns.values())
                    if n < xw.MIN_EV:
                        r[lab][f'unsat{cut:.0f}'] = dict(n=int(n))
                        continue
                    m = xw.measure(pool, h, uns, T0, a.nboot, sig0_fix=a.sig0_fix)
                    r[lab][f'unsat{cut:.0f}'] = m
                    print(f'{key:44s} {lab:5s} cut {cut:4.0f}: n {m["n"]:4d}  v {m["v"]:5.2f}±{m["v_err"]:.2f}  '
                          f'sig0 {m["sig0"]:.3f}±{m["sig0_err"]:.3f}  D_T {m["DT"]:4.0f}±{m["DT_err"]:3.0f}', flush=True)
                rr = ' '.join(f'{q["lo"]}:{q["med"]:.2f}' for q in r[lab]['ratio'] if 'med' in q)
                print(f'{"":44s} {lab:5s} r by peak ADC  {rr}', flush=True)
            out[key + ('' if a.grad == 0.15 else f'@g{a.grad:g}') + ('' if a.sig0_fix is None else f'@s0{a.sig0_fix:g}')] = r
            json.dump(out, open(path, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
