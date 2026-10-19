#!/usr/bin/env python3
"""fit_kernel.py -- which sharing-kernel FORM does the beam data pick, and
what are its constants?

Fits the three candidate forms of forms.py to the run_71 RAW head-on stacks
through the cross-relation, per drift plateau and per view, with a paired
bootstrap over events for the errors.  The three drift fields are the
invariance test: the geometric fractions q_j MUST move with field (diffusion),
the kernel constants (c, tau) must NOT.

    ../../../.venv/bin/python fit_kernel.py [--nboot 200]
writes fit_kernel.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

import forms

HERE = os.path.dirname(os.path.abspath(__file__))
PLATEAUS = [('raw700', 243.0), ('raw450', 156.0), ('raw275', 95.0)]
DLIST = (1, -1, 2, -2)
LO, HI = 22, 61          # fit samples (index 30 = the central peak)

# Every form carries the four geometric fractions q_{+-1}, q_{+-2} SEPARATELY.
# Forcing them symmetric is only right if the track is exactly normal; the X
# view carries the known 0.2-0.4 deg residual tilt of the flat mount, which
# shows up as q_{+1} != q_{-1}, and folding that into the kernel is what made
# the first pass of this fit blame X's poor rms on the kernel form.
SPEC = {
    'cascade': dict(names=('tau', 'c', 'q1', 'q1m', 'q2', 'q2m'),
                    x0=(350.0, 0.45, 0.25, 0.25, 0.05, 0.05),
                    lo=(30.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    hi=(3000.0, 0.95, 2.0, 2.0, 1.0, 1.0)),
    'ladder':  dict(names=('tau', 'c', 'c2', 'q1', 'q1m', 'q2', 'q2m'),
                    x0=(350.0, 0.45, 0.20, 0.25, 0.25, 0.05, 0.05),
                    lo=(30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    hi=(3000.0, 0.95, 1.5, 2.0, 2.0, 1.0, 1.0)),
    'delay':   dict(names=('tau', 'c1', 'c2', 'sigma_s', 'q1', 'q1m',
                           'q2', 'q2m'),
                    x0=(150.0, 0.30, 0.10, 60.0, 0.25, 0.25, 0.05, 0.05),
                    lo=(20.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0),
                    hi=(2000.0, 1.5, 1.5, 3000.0, 2.0, 2.0, 1.0, 1.0)),
    'geom':    dict(names=('q1', 'q1m', 'q2', 'q2m'), x0=(0.4, 0.4, 0.1, 0.1),
                    lo=(0.0, 0.0, 0.0, 0.0), hi=(3.0, 3.0, 2.0, 2.0)),
}


def trim_mean(A, rows=None, frac=0.20):
    B = A if rows is None else A[rows]
    out = np.full(B.shape[1], 0.0)
    for s in range(B.shape[1]):
        c = B[:, s]
        c = c[np.isfinite(c)]
        if len(c) == 0:
            continue
        c = np.sort(c)
        k = int(len(c) * frac)
        out[s] = (c[k:len(c) - k] if len(c) > 2 * k else c).mean()
    return out


def stacks_for(Z, lab, view, rows=None):
    return {d: trim_mean(Z[f'A_{lab}_{view}_{d:+d}'], rows)
            for d in (0,) + DLIST}


def unpack(form, x):
    nm = SPEC[form]['names']
    p = dict(zip(nm, x))
    q = {0: 1.0, 1: p['q1'], -1: p['q1m'], 2: p['q2'], -2: p['q2m']}
    return p, q


def fit(form, W, x0=None):
    S = SPEC[form]
    n = len(W[0])

    def res(x):
        p, q = unpack(form, x)
        nn = forms.build_n(form, 2, q, p, n)
        return forms.cross_resid(nn, W, DLIST, LO, HI)

    r = least_squares(res, np.array(x0 if x0 is not None else S['x0'], float),
                      bounds=(S['lo'], S['hi']), xtol=1e-12, ftol=1e-12)
    # residual as a percentage of the typical model-side amplitude, so the
    # three forms are compared on a scale a person can read
    p, q = unpack(form, r.x)
    nn = forms.build_n(form, 2, q, p, n)
    scale = max(np.abs(np.convolve(nn[0], W[1])[:n]).max(), 1e-12)
    return dict(par={k: float(v) for k, v in zip(S['names'], r.x)},
                rms=float(np.sqrt(np.mean(r.fun ** 2))),
                rms_pct=float(100 * np.sqrt(np.mean(r.fun ** 2)) / scale),
                cost=float(r.cost), x=[float(v) for v in r.x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=150)
    ap.add_argument('--views', default='y,x')
    args = ap.parse_args()
    Z = np.load(os.path.join(HERE, 'stacks_run71_raw.npz'))
    rng = np.random.default_rng(20260818)
    out = {'meta': dict(fit_window_samples=[LO, HI], dlist=list(DLIST),
                        nboot=args.nboot)}

    for view in args.views.split(','):
        print(f'\n########## view {view.upper()}')
        for lab, E in PLATEAUS:
            W = stacks_for(Z, lab, view)
            nev = len(Z[f'A_{lab}_{view}_+0'])
            print(f'--- {lab} ({E:.0f} V/cm, {nev} events)')
            rec = {'field_Vcm': E, 'n_events': int(nev)}
            for form in ('cascade', 'ladder', 'delay', 'geom'):
                f = fit(form, W)
                # bootstrap
                bs = []
                for _ in range(args.nboot):
                    rows = rng.integers(0, nev, nev)
                    try:
                        bs.append(fit(form, stacks_for(Z, lab, view, rows),
                                      x0=f['x'])['x'])
                    except Exception:
                        pass
                B = np.array(bs, float)
                err = {k: float(B[:, i].std())
                       for i, k in enumerate(SPEC[form]['names'])} if len(B) > 5 else {}
                f['err'] = err
                rec[form] = f
                txt = '  '.join(
                    f'{k}={v:.4g}' + (f'+-{err[k]:.3g}' if k in err else '')
                    for k, v in f['par'].items())
                print(f'  {form:8} rms {f["rms_pct"]:6.2f} %   {txt}')
            out.setdefault(view, {})[lab] = rec

    with open(os.path.join(HERE, 'fit_kernel.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\nwrote fit_kernel.json')


if __name__ == '__main__':
    main()
