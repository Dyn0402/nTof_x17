#!/usr/bin/env python3
"""systematics.py -- what the beam kernel measurement is, and is not, worth.

Four checks, in decreasing order of how much they move the answer:

  window   how long a fit window the constants need.  THE dominant systematic:
           a single one-pole cascade fitted over a SHORT window returns a
           short tau and over a long window a long one, because the measured
           tail is heavier than one exponential.  The FORM ranking is stable
           at every window; the absolute tau is not.
  basis    per-sample median vs 20 %-trimmed mean vs plain mean.
  gate     the central-strip amplitude window (400-3000 ADC by default), which
           is also the pile-up and saturation guard.
  align    peak-aligned stacks vs ABSOLUTE window time.  The alignment is a
           nonlinear operation on the central strip; it is applied identically
           to every strip of an event, so it cannot manufacture sharing, but
           this is the check that says so.

    ../../../.venv/bin/python systematics.py
writes systematics.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

import forms
from fit_kernel import SPEC, DLIST, LO, HI, trim_mean

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
sys.path.insert(0, ANA)
RW = ('/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17-mpgd26/'
      'cf7ef626-6174-476c-b483-f2699f32d221/scratchpad/rw/'
      'robust_library_run71_raw.npz')
LAB, VIEW = 'raw450', 'y'


def agg(A, how):
    if how == 'trim20':
        return trim_mean(A, None, 0.20)
    if how == 'median':
        return trim_mean(A, None, 0.499)
    if how == 'mean':
        out = np.zeros(A.shape[1])
        for s in range(A.shape[1]):
            c = A[:, s]
            c = c[np.isfinite(c)]
            if len(c):
                out[s] = c.mean()
        return out
    raise ValueError(how)


def fit(form, W, lo, hi):
    S = SPEC[form]
    n = len(W[0])

    def res(x):
        p = dict(zip(S['names'], x))
        q = {0: 1.0, 1: p['q1'], -1: p['q1m'], 2: p['q2'], -2: p['q2m']}
        return forms.cross_resid(forms.build_n(form, 2, q, p, n), W, DLIST,
                                 lo, hi)

    r = least_squares(res, np.array(S['x0'], float), bounds=(S['lo'], S['hi']),
                      xtol=1e-12, ftol=1e-12)
    p = dict(zip(S['names'], r.x))
    q = {0: 1.0, 1: p['q1'], -1: p['q1m'], 2: p['q2'], -2: p['q2m']}
    nn = forms.build_n(form, 2, q, p, n)
    sc = max(np.abs(np.convolve(nn[0], W[1])[:n]).max(), 1e-12)
    return p, 100 * np.sqrt(np.mean(r.fun ** 2)) / sc


def main():
    out = {}
    Z = np.load(os.path.join(HERE, 'stacks_run71_raw.npz'))
    T = Z['t_rel']
    A = {d: Z[f'A_{LAB}_{VIEW}_{d:+d}'] for d in (0,) + DLIST}
    W = {d: agg(A[d], 'trim20') for d in A}
    n = len(W[0])

    print('--- window: the dominant systematic')
    print(f"{'fit window':>22}{'cascade tau':>13}{'c':>8}{'rms':>8}"
          f"{'delay rms':>11}{'ratio c2/c1':>13}")
    rows = []
    for end in (600, 720, 900, 1200, 1500, 1800):
        hi = int(np.searchsorted(T, end + 1))
        pc, rc = fit('cascade', W, LO, hi)
        pd, rd = fit('delay', W, LO, hi)
        rows.append(dict(end_ns=end, tau=pc['tau'], c=pc['c'], rms_cascade=rc,
                         rms_delay=rd, delay_ratio=pd['c2'] / max(pd['c1'], 1e-9)))
        print(f"{T[LO]:+.0f} .. {end:+5.0f} ns{pc['tau']:11.0f}{pc['c']:8.3f}"
              f"{rc:7.2f} %{rd:10.2f} %{rows[-1]['delay_ratio']:13.3f}")
    out['window'] = rows

    print('\n--- aggregation basis')
    print(f"{'basis':>10}{'cascade tau':>13}{'c':>8}{'rms':>8}{'delay rms':>11}")
    out['basis'] = {}
    for how in ('trim20', 'median', 'mean'):
        Wb = {d: agg(A[d], how) for d in A}
        pc, rc = fit('cascade', Wb, LO, HI)
        _pd, rd = fit('delay', Wb, LO, HI)
        out['basis'][how] = dict(tau=pc['tau'], c=pc['c'], rms_cascade=rc,
                                 rms_delay=rd)
        print(f'{how:>10}{pc["tau"]:11.0f}{pc["c"]:8.3f}{rc:7.2f} %{rd:10.2f} %')

    print('\n--- central-strip amplitude gate (rebuilds the stacks)')
    from stacks import build as build_stacks
    out['gate'] = {}
    print(f"{'gate [ADC]':>14}{'events':>8}{'cascade tau':>13}{'c':>8}{'rms':>8}")
    for g in ((400, 1200), (400, 3000), (1200, 3000)):
        S, _m = build_stacks(q0=g)
        Ag = {d: S[f'A_{LAB}_{VIEW}_{d:+d}'] for d in (0,) + DLIST}
        Wg = {d: agg(Ag[d], 'trim20') for d in Ag}
        pc, rc = fit('cascade', Wg, LO, HI)
        out['gate'][f'{g[0]}-{g[1]}'] = dict(n=len(Ag[0]), tau=pc['tau'],
                                             c=pc['c'], rms=rc)
        print(f'{g[0]:6d}-{g[1]:<7d}{len(Ag[0]):8d}{pc["tau"]:11.0f}'
              f'{pc["c"]:8.3f}{rc:7.2f} %')

    if os.path.exists(RW):
        print('\n--- peak-aligned vs ABSOLUTE window time')
        R = np.load(RW)
        Wa = {d: np.nan_to_num(R[f'trim_{LAB}_{VIEW}_{d:+d}']) for d in (0,) + DLIST}
        na = len(Wa[0])
        pk = int(np.argmax(Wa[0]))
        lo_a = max(pk - 8, 1)
        pc, rc = fit('cascade', Wa, lo_a, na)
        pd, rd = fit('delay', Wa, lo_a, na)
        out['align'] = dict(absolute=dict(tau=pc['tau'], c=pc['c'],
                                          rms_cascade=rc, rms_delay=rd,
                                          n_samples=int(na - lo_a)))
        hi_eq = int(np.searchsorted(T, (na - 1 - pk) * 60.0 + 1))
        pa, ra = fit('cascade', W, LO, hi_eq)
        _p, rda = fit('delay', W, LO, hi_eq)
        out['align']['aligned_same_span'] = dict(tau=pa['tau'], c=pa['c'],
                                                 rms_cascade=ra, rms_delay=rda)
        print(f'{"absolute time":>22}{pc["tau"]:11.0f}{pc["c"]:8.3f}'
              f'{rc:7.2f} %{rd:10.2f} %')
        print(f'{"peak-aligned, same span":>22}{pa["tau"]:11.0f}{pa["c"]:8.3f}'
              f'{ra:7.2f} %{rda:10.2f} %')
    else:
        print(f'\n(absolute-time library not built: {RW})')

    with open(os.path.join(HERE, 'systematics.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote systematics.json')


if __name__ == '__main__':
    main()
