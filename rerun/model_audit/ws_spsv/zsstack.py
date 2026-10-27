#!/usr/bin/env python3
"""zsstack.py --bundle B --out o.json CACHE.pkl [...]

Column duration T of each cache from its stacked strip-sum waveform, four estimators side by side: the box (x) template
fit with a charge gradient (boxfit.py), the flat box, the erf edge pair (edges.py) and the 50 %-crossing width.  RAW
caches (name contains 'raw'): a sample missing on every strip is a dropped group, left out; ZS caches: a suppressed
zero is a measured zero (below threshold), kept.  Caches must be the baseline-fixed ones (sps_build_cache.py >= 7505338).
"""
import argparse, json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.abspath(os.path.join(HERE, '..', '..', '..'))]
import boxfit as B                                # noqa: E402
import edges as E                                 # noqa: E402
from wft.calib import CalibrationBundle          # noqa: E402

t = np.arange(64) * 60.0


def w50(st):
    pre = np.nanmedian(st[:8]); pl = np.nanmedian(st[(t > 1300) & (t < 2000)]); y = (st - pre) / (pl - pre)
    j = np.where((t > 2000) & (y < 0.5))[0]; jr = np.where((t > 300) & (y > 0.5))[0]
    if not len(j) or not len(jr):
        return np.nan, pre / pl
    k = j[0]; tf = t[k - 1] + (y[k - 1] - 0.5) / (y[k - 1] - y[k]) * 60
    k = jr[0]; tr = t[k - 1] + (0.5 - y[k - 1]) / (y[k] - y[k - 1]) * 60
    return tf - tr, pre / pl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('caches', nargs='+'); ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cal = CalibrationBundle.load(a.bundle); tm = np.asarray(cal.tmpl['x'], float)
    K = B.Kern(np.asarray(cal.grid, float), tm / tm.max(), 30.0); out = {}
    for f in a.caches:
        C = pickle.load(open(f, 'rb')); raw = 'raw' in os.path.basename(f); key = os.path.basename(f)[:-4]; out[key] = {}
        for v in 'xy':
            S = []
            for e in C.values():
                W = np.asarray(e[v]['W'], float)
                if W.shape[1] != 64:
                    continue
                s = W.sum(0); ok = ~(W == 0).all(0) if raw else np.ones(64, bool)
                if s.max() <= 0:
                    continue
                S.append(np.where(ok, s / s.max(), np.nan))
            if len(S) < 30:
                continue                                   # 32-sample caches: the column overruns the window
            st = np.nanmedian(np.array(S), 0); okk = np.isfinite(st)
            _, p = B.fit_two_stage(st, okk, K, True, T_hi=3700)
            _, p2 = B.fit_two_stage(st, okk, K, False, T_hi=3700)
            tr, tf = E.edge_times(st); W5, pre = w50(st)
            out[key][v] = dict(n=len(S), T_box=p[1], grad=p[3] / p[2], T_flat=p2[1], T_edge=tf - tr, T_50=W5, pre=pre,
                               box_t0=p[0], stack=[None if not np.isfinite(x) else float(x) for x in st],
                               model=[float(x) for x in K(p[0], p[1]) @ np.array(p[2:4])])
            print(f'{key:45s} {v} n{len(S)} box {p[1]:.0f} (g {p[3] / p[2]:+.2f}) flat {p2[1]:.0f} edge {tf - tr:.0f} '
                  f'w50 {W5:.0f} pre/pl {pre:.2f}', flush=True)
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
