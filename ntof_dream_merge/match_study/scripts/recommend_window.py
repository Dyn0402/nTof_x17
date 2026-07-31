#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recommend_window.py -- turn the scans into one number the merge can adopt.

WHY NOT JUST MAXIMISE efficiency x purity. On the wall-AND-plastic leg the purity
is above 99.9 % at every window worth considering, so that product is maximised
by the widest window on offer -- it recommends doing nothing. Purity is the wrong
objective once it is saturated; the thing that still costs is the ACCIDENTAL RATE
itself, because every accidental match is a wrong wall time, a wrong amplitude
and, 1 time in 3, a wrong arm for the Micromegas cross-check.

So the criterion here is the standard one for a plateau: take the TIGHTEST window
whose efficiency is still within `--tol` (relative) of the plateau value, and
report what that costs and what it buys. The plateau is measured at +-150 ns,
which every scan shows to be flat.

USAGE
    python recommend_window.py [--timebase perbunch] [--tol 0.005]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import DATA

T_BINS = ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80))
LEGNAME = {'wp': 'wall AND plastic', 'w': 'wall only'}


def knee(w, eff, tol, plateau_at=150.0):
    """Tightest half-width whose efficiency is within `tol` of the plateau."""
    plateau = float(np.interp(plateau_at, w, eff))
    ok = eff >= (1 - tol) * plateau
    if not ok.any():
        return np.nan, plateau
    return float(w[np.argmax(ok)]), plateau


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--timebase', default='perbunch')
    ap.add_argument('--tol', type=float, default=0.005)
    ap.add_argument('--compare', default='legacy')
    args = ap.parse_args()

    Z = np.load(DATA / f'window_scan_{args.timebase}.npz')
    Z0 = np.load(DATA / f'window_scan_{args.compare}.npz')
    out = dict(timebase=args.timebase, tol=args.tol, legs={})

    for leg in ('wp', 'w'):
        w = Z[f'{leg}/sym/w']
        eff, fal = Z[f'{leg}/sym/eff'], Z[f'{leg}/sym/false']
        wk, pl = knee(w, eff, args.tol)
        i = int(np.argmin(np.abs(w - wk)))
        j = int(np.argmin(np.abs(w - 150.0)))
        e0 = np.interp(150.0, Z0[f'{leg}/sym/w'], Z0[f'{leg}/sym/eff'])
        f0 = np.interp(150.0, Z0[f'{leg}/sym/w'], Z0[f'{leg}/sym/false'])
        print(f'\n=== {LEGNAME[leg]} ===')
        print(f'  plateau efficiency (+-150 ns)      {pl:7.2%}')
        print(f'  recommended half-width             {wk:7.0f} ns  '
              f'(within {args.tol:.1%} of plateau)')
        print(f'    efficiency                       {eff[i]:7.2%}')
        print(f'    accidental match rate            {fal[i]:7.3%}  '
              f'(was {fal[j]:.3%} at +-150 ns, '
              f'{f0:.3%} on the {args.compare} time base)')
        print(f'    background suppression           '
              f'{f0 / max(fal[i], 1e-9):7.1f}x vs {args.compare} at +-150 ns')
        print(f'    efficiency change                {eff[i] - e0:+7.2%} '
              f'vs {args.compare} at +-150 ns')
        row = dict(plateau_eff=float(pl), half_width_ns=float(wk),
                   eff=float(eff[i]), false=float(fal[i]),
                   false_at_150=float(fal[j]),
                   compare=dict(timebase=args.compare, eff=float(e0),
                                false=float(f0)))

        print('  per time bin:')
        print('    t (ms)     plateau   knee     eff@knee   false@knee')
        row['per_t'] = {}
        for lo, hi in T_BINS:
            n = f'sym_t{lo}_{hi}'
            if f'{leg}/{n}/w' not in Z:
                continue
            ww, ee, ff = Z[f'{leg}/{n}/w'], Z[f'{leg}/{n}/eff'], Z[f'{leg}/{n}/false']
            wkt, plt_ = knee(ww, ee, args.tol)
            k = int(np.argmin(np.abs(ww - wkt)))
            print(f'    {lo:3d}-{hi:<4d} {plt_:9.2%} {wkt:6.0f} ns {ee[k]:10.2%} '
                  f'{ff[k]:12.3%}')
            row['per_t'][f'{lo}-{hi}'] = dict(half_width_ns=float(wkt),
                                              eff=float(ee[k]), false=float(ff[k]))
        out['legs'][leg] = row

    p = DATA / 'recommended_window.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\n-> {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
