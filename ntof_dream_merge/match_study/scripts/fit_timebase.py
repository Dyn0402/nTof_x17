#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_timebase.py -- re-fit the DREAM -> n_TOF time map on the candidate processing.

    t_nTOF = t_DREAM (1 + K) + T0

K = 1.089e-4 and T0 = -197.5 ns were fitted while the merge was being built, on
the OFFICIAL processing of run 224572 and with the laptop-side tflash repair in
the chain. Neither is guaranteed to survive a reprocessing: the flash finder is
what defines t = 0 in the n_TOF trees, and v12 changes it. Measured here, the
residual peak sits ~43 ns away from zero, which is a third of the accept
half-width -- irrelevant while the window is +-150 ns, decisive as soon as it is
tightened.

The fit is a straight line through the residuals of the matched core, iterated so
that the core selection is centred on its own solution, and it is done per
sub-run and per arm as well as globally so that a shared constant can be told
apart from a per-arm cable difference.

USAGE
    python fit_timebase.py [--leg wp] [--json timebase.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import DATA, K, T0, SUBRUNS

ARMS = ('A', 'B', 'C', 'D')
CORE_NS = 250.0          # generous: the whole band lives inside +-200 ns
ITERS = 3


def raw_residuals(sub, leg, k, t0):
    """Residuals of every candidate within +-1 us of the prediction, with arm."""
    ev = np.load(DATA / f'events_{sub}.npz')
    cd = np.load(DATA / f'cand_{sub}_{leg}.npz')
    from window_scan import residuals
    import study_common
    k_old, t_old = study_common.K, study_common.T0
    study_common.K, study_common.T0 = k, t0
    try:
        ei, r, ci = residuals(ev['bunch'], ev['t'], cd['bunch'], cd['t'],
                              search=1000.0)
    finally:
        study_common.K, study_common.T0 = k_old, t_old
    return ei, r, cd['arm'][ci], ev['t']


def fit(ev_t, ei, r, core=CORE_NS, nbin=24):
    """Robust r = a + b t: per-time-bin MEDIAN residual, then a straight line.

    The residual band is strongly asymmetric (a fast rise on the early side, a
    tail on the late one -- it is a trigger-latency distribution, not a
    resolution), so a least-squares mean would be pulled by the tail and would
    drift with the local rate. The median of the core is stable against both.
    """
    m = np.abs(r) < core
    if m.sum() < 200:
        return np.nan, 0.0, int(m.sum()), np.nan
    t, rr = ev_t[ei[m]], r[m]
    edges = np.geomspace(max(t.min(), 1e5), t.max(), nbin + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbin - 1)
    tc, rc, wc = [], [], []
    for i in range(nbin):
        s = idx == i
        if s.sum() < 50:
            continue
        tc.append(np.median(t[s]))
        rc.append(np.median(rr[s]))
        wc.append(s.sum())
    if len(tc) < 3:
        return float(np.median(rr)), 0.0, int(m.sum()), float(rr.std())
    tc, rc, wc = np.array(tc), np.array(rc), np.array(wc, float)
    b, a = np.polyfit(tc, rc, 1, w=np.sqrt(wc))
    spread = float(np.std(rc - (a + b * tc)))
    return float(a), float(b), int(m.sum()), spread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', default='wp')
    ap.add_argument('--json', default=str(DATA / 'timebase.json'))
    args = ap.parse_args()

    out = dict(start=dict(K=K, T0=T0), leg=args.leg, per_sub={}, per_arm={})
    k, t0 = K, T0
    for it in range(ITERS):
        A, B, N = [], [], 0
        for sub in SUBRUNS:
            ei, r, arm, ev_t = raw_residuals(sub, args.leg, k, t0)
            a, b, n, s = fit(ev_t, ei, r)
            A.append(a * n)
            B.append(b * n)
            N += n
            out['per_sub'][sub] = dict(a=a, b=b, n=n, sigma=s)
            if it == ITERS - 1:
                for ai, arm_name in enumerate(ARMS):
                    sa = arm == ai
                    aa, bb, nn, ss = fit(ev_t, ei[sa], r[sa])
                    out['per_arm'].setdefault(arm_name, {})[sub] = dict(
                        a=aa, b=bb, n=nn, sigma=ss)
        da, db = sum(A) / N, sum(B) / N
        t0 = t0 + da
        k = k + db
        print(f'  iter {it}: shift {da:+7.2f} ns, slope {db:+.3e} '
              f'-> T0 = {t0:.2f} ns, K = {k:.6e}   (n = {N:,})')

    out['fitted'] = dict(K=k, T0=t0)
    print(f'\nfitted on {args.leg}: K = {k:.6e} (was {K:.6e}), '
          f'T0 = {t0:.2f} ns (was {T0:.2f})')
    print(f'  drift term at 80 ms: {k * 8e7 / 1000:.2f} us '
          f'(was {K * 8e7 / 1000:.2f} us)')
    print('\nper sub-run, at the fitted point:')
    for sub, v in out['per_sub'].items():
        print(f'  {sub}: residual offset {v["a"]:+6.2f} ns, slope {v["b"]:+.2e}, '
              f'core sigma {v["sigma"]:.1f} ns, n = {v["n"]:,}')
    print('\nper arm (offset a in ns, at the fitted point):')
    for arm, d in out['per_arm'].items():
        cells = '  '.join(f'{s.split("_")[-1]} {v["a"]:+6.2f} '
                          f'(sig {v["sigma"]:5.1f})' for s, v in d.items())
        print(f'  arm {arm}: {cells}')

    with open(args.json, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\n-> {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
