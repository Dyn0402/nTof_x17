#!/usr/bin/env python3
"""compare.py base:arm [base:arm ...] [--prefix rmF_] [--nboot 500] [--json out.json]

Event-paired comparison of resid_map outputs (~/x17/cosmic_bench/det4dig/<prefix><tag>.npz): for each view the
ladder slope, s68 of the angle residual (|d| < 5 deg; each arm at its own robust ladder slope, as resid_show), the
far fraction and the median chi2/dof, for base and arm on the events both fitted, and arm - base with a bootstrap
over events (the same resampled events in both arms).  Classes: all, head-on (|tan_ref| < 0.1), inclined (> 0.1).
"""
import argparse, json, os
import numpy as np
from resid_show import slope

D = os.path.expanduser('~/x17/cosmic_bench/det4dig/')


def load(tag, prefix, vi):
    P = np.load(D + f'{prefix}{tag}.npz')['per']
    P = P[P[:, 1] == vi]
    return {int(r[0]): r for r in P}


def stats(R, s, c):
    tr, w, chi, dof = R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5
    return np.array([np.percentile(np.abs(d[~far]), 68) if (~far).sum() > 5 else np.nan, 100 * far.mean(),
                     np.median(chi / dof)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pairs', nargs='+'); ap.add_argument('--prefix', default='rmF_')
    ap.add_argument('--nboot', type=int, default=500); ap.add_argument('--json')
    a = ap.parse_args(); rng = np.random.default_rng(1); out = {}
    for pr in a.pairs:
        b, t = pr.split(':')
        for vi, view in enumerate('xy'):
            B, T = load(b, a.prefix, vi), load(t, a.prefix, vi)
            ids = sorted(set(B) & set(T)); RB = np.array([B[i] for i in ids]); RT = np.array([T[i] for i in ids])
            sb, cb = slope(RB[:, 2], RB[:, 3]); st, ct = slope(RT[:, 2], RT[:, 3])
            at = np.abs(RB[:, 2])
            print(f'{view}  {b} -> {t}  n {len(ids)}  ladder {sb:.2f} -> {st:.2f}')
            for cl, m in (('all', at < 9), ('head-on', at < 0.1), ('inclined', at >= 0.1)):
                idx = np.flatnonzero(m)
                x0, x1 = stats(RB[idx], sb, cb), stats(RT[idx], st, ct)
                bs = []
                for _ in range(a.nboot):
                    k = rng.choice(idx, len(idx))
                    bs.append(stats(RT[k], st, ct) - stats(RB[k], sb, cb))
                e = np.nanstd(np.array(bs), 0)
                dd = x1 - x0
                print(f'   {cl:8s} n{len(idx):5d}  s68 {x0[0]:.3f} -> {x1[0]:.3f} ({dd[0]:+.3f} +- {e[0]:.3f})  '
                      f'far {x0[1]:4.1f} -> {x1[1]:4.1f}% ({dd[1]:+.1f} +- {e[1]:.1f})  chi2/dof {x0[2]:6.1f} -> {x1[2]:6.1f}')
                out[f'{view}|{b}|{t}|{cl}'] = dict(n=len(idx), ladder=[sb, st], base=x0.tolist(), arm=x1.tolist(),
                                                     delta=dd.tolist(), err=e.tolist())
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
