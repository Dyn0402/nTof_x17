#!/usr/bin/env python3
"""gate_compare.py -- paired gate table: production vs per-view arms, one key.

Reads <wft>/angles_<tag>/per_event.npz (03_angles.py's per-event dump) for the
production reference and each arm, pairs events by id per plane, and reports
with a paired bootstrap (the arms share events, so unpaired errors are ~3x too
large):

  s68 (|theta| residual 68th percentile about the median), all and |theta| < 5,
  median bias, and the UNGATED implied-v spread (median v_fit/v_ref over
  |tan_ref| bins >= 0.08 -- the R06_GATE paired column, not 03's gated one),

plus the headline efficiency/position numbers from efficiency_<tag>.

    gate_compare.py <wft dir> --base prodref --arms pvA pvB [--json out.json]
"""
import argparse
import json
import os

import numpy as np

BINS = ((0.08, 0.15), (0.15, 0.25), (0.25, 0.4), (0.4, 0.7))
LT5 = 0.0875


def load(W, tag):
    z = np.load(os.path.join(W, f'angles_{tag}', 'per_event.npz'))
    return {p: {int(e): (r, f) for e, r, f in
                zip(z[f'{p}_eid'], z[f'{p}_tan_ref'], z[f'{p}_tan_fit'])}
            for p in ('x', 'y')}


def stats(tr, tf):
    dth = np.degrees(np.arctan(tf)) - np.degrees(np.arctan(tr))
    med = np.median(dth)
    s68 = np.percentile(np.abs(dth - med), 68)
    h = np.abs(tr) < LT5
    dh = dth[h]
    s68h = np.percentile(np.abs(dh - np.median(dh)), 68) if len(dh) > 10 else np.nan
    at = np.abs(tr)
    meds = [np.median(tf[m] / tr[m]) for lo, hi in BINS
            for m in [(at >= lo) & (at < hi)] if m.sum() >= 15]
    vs = (max(meds) - min(meds)) if len(meds) > 1 else np.nan
    return np.array([s68, s68h, med, vs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wft')
    ap.add_argument('--base', default='prodref')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261009)
    base = load(a.wft, a.base)
    out = {}
    names = ('s68', 's68_lt5', 'bias', 'vspread')
    for tag in a.arms:
        arm = load(a.wft, tag)
        out[tag] = {}
        for p in ('x', 'y'):
            common = sorted(set(base[p]) & set(arm[p]))
            B = np.array([base[p][e] for e in common])
            A = np.array([arm[p][e] for e in common])
            sb, sa = stats(*B.T), stats(*A.T)
            n = len(common)
            bs = []
            for _ in range(a.nboot):
                i = rng.integers(0, n, n)
                bs.append(stats(*A[i].T) - stats(*B[i].T))
            bs = np.array(bs)
            d = sa - sb
            err = np.nanstd(bs, axis=0)
            out[tag][p] = dict(n=n, base=dict(zip(names, map(float, sb))),
                               arm=dict(zip(names, map(float, sa))),
                               delta=dict(zip(names, map(float, d))),
                               delta_err=dict(zip(names, map(float, err))))
            print(f'{tag:12s} {p}  n={n:5d}  ' + '  '.join(
                f'{k} {sb[j]:.3f}->{sa[j]:.3f} ({d[j]:+.3f}±{err[j]:.3f}, '
                f'{d[j] / err[j] if err[j] > 0 else 0:+.1f}σ)'
                for j, k in enumerate(names)))
        for t in (a.base, tag):
            pth = os.path.join(a.wft, f'efficiency_{t}', 'efficiency_breakdown.json')
            if os.path.exists(pth):
                e = json.load(open(pth))
                out[tag].setdefault('efficiency', {})[t] = {
                    k: e.get(k) for k in ('within_R', 'reco_at_all', 'core_sigma_mm',
                                          'median_r_mm', 'has_any')}
        if 'efficiency' in out[tag]:
            print(f'{"":12s} efficiency: ' + json.dumps(out[tag]['efficiency']))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
