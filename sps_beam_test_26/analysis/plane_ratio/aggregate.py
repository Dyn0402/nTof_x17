#!/usr/bin/env python3
"""aggregate.py -- combine the per-(chamber, arm) held-out bench JSONs.

Each bench job scored production and ONE arm on the same held-out events and
kept per-event residuals.  Here every arm is PAIRED (same events, bootstrap)
against
  * production  -- the shipped bundle, and
  * its control -- the same calibration recipe with r06's single ratio
    (g06, g06_pp, g06_mf05, g06_mf05_pp): the per-view effect alone.

Metrics per plane: s68 (all), s68 at |theta| < 5 deg, median bias, angle slope.

    aggregate.py <results dir> [--start blind] [--json summary.json]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

LT5 = 0.0875


def load(path, start):
    r = json.load(open(path))
    out = {}
    for arm, v in r.items():
        ev = v.get('events', {}).get(start)
        if ev is None:
            continue
        out[arm] = {p: {int(e[0]): (e[1], e[2]) for e in ev[p]} for p in ('x', 'y')}
    return out


def stats(a):
    tr, tf = a[:, 0], a[:, 1]
    d = tf - tr
    k = np.abs(d) < 0.15
    dth = np.degrees(np.arctan(tf[k])) - np.degrees(np.arctan(tr[k]))
    h = np.abs(tr[k]) < LT5
    s = lambda x: np.percentile(np.abs(x - np.median(x)), 68) if len(x) > 10 else np.nan
    return np.array([s(dth), s(dth[h]), np.median(dth),
                     np.polyfit(tr[k], tf[k], 1)[0]])


def paired(A, B, nboot, rng):
    common = sorted(set(A) & set(B))
    a = np.array([A[e] for e in common])
    b = np.array([B[e] for e in common])
    d0 = stats(a) - stats(b)
    n = len(common)
    bs = np.array([stats(a[i]) - stats(b[i])
                   for i in (rng.integers(0, n, n) for _ in range(nboot))])
    return d0, np.nanstd(bs, axis=0), n


def control_of(tag):
    det, rest = tag.split('_', 1)
    m = re.match(r'(diag_pv_free|pv_all|pv_cx|pv_sp|pv|g06)(.*)', rest)
    return f'{det}_g06{m.group(2)}' if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--start', default='blind')
    ap.add_argument('--nboot', type=int, default=400)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261009)
    arms = {}
    prod = {}
    for f in sorted(glob.glob(os.path.join(a.results, 'bench_*.json'))):
        tag = os.path.basename(f)[6:-5]
        d = load(f, a.start)
        if not d:
            continue
        arms[tag] = d.get(tag) or d[[k for k in d if k != 'production'][0]]
        prod[tag.split('_')[0]] = d['production']
    names = ('s68', 's68_lt5', 'bias', 'slope')
    out = {}
    for tag in sorted(arms):
        det = tag.split('_')[0]
        row = {}
        ctl = control_of(tag)
        for ref_name, ref in (('vs_prod', prod[det]),
                              ('vs_ctl', arms.get(ctl) if ctl != tag else None)):
            if ref is None:
                continue
            row[ref_name] = {}
            for p in ('x', 'y'):
                d, e, n = paired(arms[tag][p], ref[p], a.nboot, rng)
                row[ref_name][p] = dict(n=n, **{k: float(d[i]) for i, k in enumerate(names)},
                                        **{k + '_err': float(e[i]) for i, k in enumerate(names)})
        for p in ('x', 'y'):
            row.setdefault('abs', {})[p] = dict(zip(names, map(float, stats(
                np.array(list(arms[tag][p].values()))))))
        out[tag] = row
        line = f'{tag:24s}'
        for p in ('x', 'y'):
            q = row['abs'][p]
            line += f' | {p} s68 {q["s68"]:.3f} h {q["s68_lt5"]:.3f}'
            for rn in ('vs_prod', 'vs_ctl'):
                if rn in row:
                    z = row[rn][p]
                    line += (f' {rn[3:]} {z["s68"]:+.3f}±{z["s68_err"]:.3f}'
                             f'/h{z["s68_lt5"]:+.3f}±{z["s68_lt5_err"]:.3f}')
        print(line)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
