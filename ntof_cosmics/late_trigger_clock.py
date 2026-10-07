#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
late_trigger_clock.py -- what sets the time shape of the late (> 20 ms) triggers?

Direct beam captures in a thin 1/v absorber fall ~ t^-4 at these times (and
the evaluated EAR2 flux has ~nothing below 2.5 meV, i.e. after ~28 ms at
19.5 m).  A component with its own clock -- 12B beta decay (T1/2 = 20.20 ms,
from 12C(n,p) by flash neutrons > 13.6 MeV) or a thermal die-away in the hall
-- falls exponentially.  Fit the campaign's trigger-time distribution (unique
triggers with a gated track, 1 ms bins, empty comb bins masked):
    R(t) = A t^-n + B 2^(-t/T) + C
HANDOFF_TRACKING_2026-10-06.md §10f.

    python ntof_cosmics/late_trigger_clock.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import curve_fit

CAMPAIGN = '/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet'
T_12B = 20.20


def load(arm=None):
    f = [('gated', '==', True)] + ([('arm', '==', arm)] if arm else [])
    t = pq.read_table(CAMPAIGN, columns=['run', 'subrun', 'event_id', 't_since_flash_ns'], filters=f).to_pandas()
    t = t.drop_duplicates(['run', 'subrun', 'event_id'])
    return t.t_since_flash_ns.to_numpy() / 1e6


def fit(ms, lo=20.0, hi=80.0, fixT=None):
    e = np.arange(lo, hi + 0.01, 1.0)
    h, _ = np.histogram(ms, e)
    c = 0.5 * (e[1:] + e[:-1])
    ok = h > 0.2 * np.median(h[h > 0])           # the DAQ comb leaves empty bins
    c, h = c[ok], h[ok].astype(float)
    if fixT is None:
        f = lambda t, A, n, B, T, C: A * (t / 30) ** (-n) + B * 2 ** (-(t - 30) / T) + C  # noqa: E731
        p0 = [h[10] * 0.3, 4, h[10] * 0.7, 22, h[-1] * 0.1]
        bounds = ([0, 1, 0, 2, 0], [np.inf, 10, np.inf, 200, np.inf])
    else:
        f = lambda t, A, n, B, C: A * (t / 30) ** (-n) + B * 2 ** (-(t - 30) / fixT) + C  # noqa: E731
        p0 = [h[10] * 0.3, 4, h[10] * 0.7, h[-1] * 0.1]
        bounds = ([0, 1, 0, 0], [np.inf, 10, np.inf, np.inf])
    p, cv = curve_fit(f, c, h, p0=p0, sigma=np.sqrt(h), bounds=bounds, maxfev=50000)
    chi = float(np.sum(((h - f(c, *p)) / np.sqrt(h)) ** 2))
    return p, np.sqrt(np.diag(cv)), chi, len(c) - len(p), c, h, f


OUT = __import__('pathlib').Path(__file__).resolve().parent / 'results' / 'late_clock'


def main() -> int:
    import json
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {}
    for arm in (None, 'A', 'C'):
        ms = load(arm)
        lab = arm or 'all arms'
        e = np.arange(1, 80.01, 1.0)
        h, _ = np.histogram(ms, e)
        pf, ef, chif, ndff, *_ = fit(ms, lo=20.0)
        px, _ex, chix, ndfx, *_ = fit(ms, lo=20.0, fixT=T_12B)
        summ[lab] = dict(edges=e.tolist(), hist=h.tolist(), free=dict(p=pf.tolist(), err=ef.tolist(), chi2=chif, ndf=ndff),
                         fixed_12B=dict(p=px.tolist(), chi2=chix, ndf=ndfx))
        for lo in (20.0, 30.0):
            p, e, chi, ndf, *_ = fit(ms, lo=lo)
            print(f'{lab:8s} {lo:.0f}-80 ms  free T: T1/2 = {p[3]:.1f} ± {e[3]:.1f} ms, power n = {p[1]:.1f}, '
                  f'exp share at 40 ms = {p[2] * 2 ** (-10 / p[3]) / (p[0] * (40 / 30) ** -p[1] + p[2] * 2 ** (-10 / p[3]) + p[4]):.2f}, '
                  f'chi2/ndf {chi:.0f}/{ndf}')
            p, e, chi, ndf, *_ = fit(ms, lo=lo, fixT=T_12B)
            print(f'{"":8s} {lo:.0f}-80 ms  T fixed 20.2: chi2/ndf {chi:.0f}/{ndf}, const share at 70 ms '
                  f'= {p[3] / (p[0] * (70 / 30) ** -p[1] + p[2] * 2 ** (-40 / T_12B) + p[3]):.2f}')
    (OUT / 'summary.json').write_text(json.dumps(summ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
