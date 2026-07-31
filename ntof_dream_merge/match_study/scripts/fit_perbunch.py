#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_perbunch.py -- the residual band is not resolution, it is clock drift.

After the global re-fit the match residual is still 9.6 ns wide (68 % half-width)
at 1-3 ms and 44 ns wide at 40-80 ms. A width that grows in proportion to the
time since the flash is not a timing resolution: it is a RATE error, and a rate
error that changes from bunch to bunch is what is left once one global K has been
removed. 44 ns over 60 ms is 7e-7, i.e. the DREAM timestamp clock wanders by
about a part per million between bursts.

So fit the map per bunch,

    t_nTOF = t_DREAM (1 + K + dk_b) + T0 + da_b

with (da_b, dk_b) least-squares from that bunch's own matched events, and the
window can be tightened to the actual coincidence resolution instead of to the
drift envelope.

HONESTY ABOUT THE FIT. It is fitted on the same events it is then used to match,
so the in-sample width is optimistic. Everything below is therefore quoted
CROSS-VALIDATED: the parameters come from the odd-numbered events of a bunch and
are evaluated on the even-numbered ones (and vice versa), which is what the real
use case looks like -- the correction is a property of the burst, not of the
event.

USAGE
    python fit_perbunch.py [--leg wp] [--core 200]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import DATA, SUBRUNS

MIN_EVENTS = 20          # below this the two-parameter fit is not worth making
TRIM_NS = 100.0          # one outlier-rejection pass


def _fit_bunch(t, r):
    """(offset, slope) least squares with a single trim; nan if not fittable."""
    if t.size < MIN_EVENTS:
        return np.nan, np.nan, 0
    b, a = np.polyfit(t, r, 1)
    keep = np.abs(r - (a + b * t)) < TRIM_NS
    if keep.sum() >= MIN_EVENTS and keep.sum() < t.size:
        b, a = np.polyfit(t[keep], r[keep], 1)
    return float(a), float(b), int(keep.sum())


def core_residuals(sub, leg, arm_off):
    """Per event: time, bunch, and the nearest candidate residual (nan if none)."""
    import window_scan as ws
    ev, cd = ws.load(sub, leg, '', arm_off)
    ei, r, ci = ws.residuals(ev['bunch'], ev['t'], cd['bunch'], cd['t'],
                             search=400.0)
    best = np.full(ev['t'].size, np.nan)
    if ei.size:
        o = np.argsort(np.abs(r))[::-1]        # last write wins -> smallest |r|
        best[ei[o]] = r[o]
    return ev['t'], ev['bunch'], best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', default='wp')
    ap.add_argument('--core', type=float, default=200.0)
    args = ap.parse_args()

    import window_scan as ws
    arm_off, tb = ws.apply_timebase('fitarm')
    print(f'starting from the fitarm map: K = {tb["K"]:.6e}, '
          f'T0 = {tb["T0"]:.2f} ns')

    out = {}
    stats = []
    for sub in SUBRUNS:
        t, bn, r = core_residuals(sub, args.leg, arm_off)
        good = np.isfinite(r) & (np.abs(r) < args.core)
        pars = {}
        # Per-event correction to ADD to the predicted n_TOF time. `corr_in` is
        # the bunch's own fit applied to its own events; `corr_cv` is the
        # cross-validated one -- fitted on half the bunch's MATCHED events and
        # applied to ALL events of the other half, matched or not, so that the
        # efficiency it produces is not the fit reading back its own input.
        corr_in = np.full(t.size, np.nan)
        corr_cv = np.full(t.size, np.nan)
        for b in np.unique(bn):
            all_idx = np.flatnonzero(bn == b)
            fit_idx = all_idx[good[all_idx]]
            if fit_idx.size < MIN_EVENTS:
                continue
            a, k, n = _fit_bunch(t[fit_idx], r[fit_idx])
            pars[int(b)] = (a, k, n)
            corr_in[all_idx] = a + k * t[all_idx]
            half = np.arange(all_idx.size) % 2
            for h in (0, 1):
                f = all_idx[(half == h) & good[all_idx]]
                e = all_idx[half == 1 - h]
                if f.size < MIN_EVENTS:
                    continue
                aa, kk, _ = _fit_bunch(t[f], r[f])
                if np.isfinite(aa):
                    corr_cv[e] = aa + kk * t[e]
        cv = r - corr_cv
        np.savez_compressed(DATA / f'perbunch_corr_{sub}_{args.leg}.npz',
                            corr_cv=corr_cv, corr_in=corr_in)

        ks = np.array([v[1] for v in pars.values()])
        as_ = np.array([v[0] for v in pars.values()])
        stats.append((sub, len(pars), as_, ks))
        print(f'\n{sub}: fitted {len(pars)} of {np.unique(bn).size} bunches '
              f'({good.sum():,} core events)')
        print(f'  per-bunch offset  da: median {np.median(as_):+7.2f} ns, '
              f'RMS {as_.std():6.2f} ns, p1-p99 '
              f'{np.percentile(as_, 1):+.1f} .. {np.percentile(as_, 99):+.1f}')
        print(f'  per-bunch rate    dk: median {np.median(ks):+.3e}, '
              f'RMS {ks.std():.3e}  (= {ks.std() * 6e7:.1f} ns of drift at 60 ms)')

        # what it buys, cross-validated
        print('  cross-validated residual width, 68 % half-width [ns]:')
        print('    t bin (ms)     before     after     events')
        for lo, hi in ((1, 3), (3, 10), (10, 20), (20, 40), (40, 80)):
            m = good & (t >= lo * 1e6) & (t < hi * 1e6)
            mc = m & np.isfinite(cv)
            if mc.sum() < 100:
                continue
            w0 = 0.5 * np.diff(np.percentile(r[m], [16, 84]))[0]
            w1 = 0.5 * np.diff(np.percentile(cv[mc], [16, 84]))[0]
            print(f'    {lo:4d}-{hi:<4d} {w0:10.1f} {w1:9.1f} {mc.sum():10,}')
        out[sub] = pars

    np.savez_compressed(
        DATA / f'perbunch_{args.leg}.npz',
        **{sub: np.array([[b, *v] for b, v in sorted(p.items())], float)
           for sub, p in out.items()})
    with open(DATA / f'perbunch_{args.leg}_summary.json', 'w') as f:
        json.dump({sub: dict(n_bunches=int(n), offset_rms=float(a.std()),
                             rate_rms=float(k.std()),
                             rate_median=float(np.median(k)))
                   for sub, n, a, k in stats}, f, indent=1)
    print(f'\n-> {DATA}/perbunch_{args.leg}.npz')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
