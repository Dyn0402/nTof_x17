#!/usr/bin/env python3
"""finalize_compare.py [--dir ~/x17/cosmic_bench/det4dig/finalize] [--arms phys timeD] [--json out.json]

Event-paired comparison of two finalize arms (make_pass_dag finalize --out-tag; <det>_<arm>/per_event.npz copied
from <row>/wft/final_<arm>/angles).  Per chamber and view, on the events both arms reconstruct:
  ladder   robust slope of the fitted w [um/ns] against the M3 tan (the per-event fit never uses v), vs timing v
  s68      of the angle residual at that ladder, |d| < 5 deg; far = share with |d| > 5 deg
  prod     the production angle (tan_fit: w0/kw and the bundle v applied) - its bias (median) and s68
with a bootstrap over events for arm - base.  Classes: all, head-on (|tan_ref| < 0.1), inclined.
"""
import argparse, json, os
import numpy as np

V_TIMING = {'det2': 39.3, 'det3': 35.8, 'det4': 33.0, 'det6': 24.17, 'det7': 36.8}


def slope(tr, w):
    m = np.isfinite(w) & np.isfinite(tr) & (np.abs(tr) < 0.7)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27)
    return s, c


def stats(tr, w, tf, s, c):
    d = np.degrees(np.arctan((w - c) / s) - np.arctan(tr)); far = np.abs(d) > 5
    dp = np.degrees(np.arctan(tf) - np.arctan(tr)); fp = np.abs(dp) > 5
    return np.array([np.percentile(np.abs(d[~far]), 68), 100 * far.mean(),
                     np.median(dp[~fp]), np.percentile(np.abs(dp[~fp] - np.median(dp[~fp])), 68)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.expanduser('~/x17/cosmic_bench/det4dig/finalize'))
    ap.add_argument('--arms', nargs=2, default=['timeD', 'phys']); ap.add_argument('--json')
    ap.add_argument('--nboot', type=int, default=300)
    a = ap.parse_args(); b0, b1 = a.arms; rng = np.random.default_rng(2); out = {}
    names = ('s68', 'far%', 'prod bias', 'prod s68')
    for det in V_TIMING:
        Z = [np.load(os.path.join(a.dir, f'{det}_{t}', 'per_event.npz')) for t in (b0, b1)]
        for v in 'xy':
            e0, e1 = Z[0][f'{v}_eid'], Z[1][f'{v}_eid']
            com, i0, i1 = np.intersect1d(e0, e1, return_indices=True)
            tr = Z[0][f'{v}_tan_ref'][i0]
            W = [Z[0][f'{v}_w'][i0] * 1e3, Z[1][f'{v}_w'][i1] * 1e3]
            TF = [Z[0][f'{v}_tan_fit'][i0], Z[1][f'{v}_tan_fit'][i1]]
            L = [slope(tr, w) for w in W]
            print(f'{det} {v}  n {len(com)} (only {b0} {len(e0) - len(com)}, only {b1} {len(e1) - len(com)})  '
                  f'ladder {L[0][0]:.2f} -> {L[1][0]:.2f}  (timing {V_TIMING[det]})')
            for cl, m in (('all', np.abs(tr) < 9), ('head-on', np.abs(tr) < 0.1), ('inclined', np.abs(tr) >= 0.1)):
                idx = np.flatnonzero(m)
                x0 = stats(tr[idx], W[0][idx], TF[0][idx], *L[0]); x1 = stats(tr[idx], W[1][idx], TF[1][idx], *L[1])
                bs = []
                for _ in range(a.nboot):
                    k = rng.choice(idx, len(idx))
                    bs.append(stats(tr[k], W[1][k], TF[1][k], *L[1]) - stats(tr[k], W[0][k], TF[0][k], *L[0]))
                e = np.std(bs, 0)
                print(f'   {cl:8s} n{len(idx):5d}  ' + '  '.join(
                    f'{nm} {x0[j]:.3f}->{x1[j]:.3f} ({x1[j] - x0[j]:+.3f}+-{e[j]:.3f})' for j, nm in enumerate(names)))
                out[f'{det}|{v}|{cl}'] = dict(n=len(idx), ladder=[L[0][0], L[1][0]], v_timing=V_TIMING[det],
                                             base=x0.tolist(), arm=x1.tolist(), err=e.tolist(), fields=names)
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
