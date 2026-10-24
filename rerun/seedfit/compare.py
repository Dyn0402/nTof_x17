#!/usr/bin/env python3
"""compare.py o1.json [o2.json ...] [--v V] -- per plane and fitter: share of events above the best χ² any fitter found
(>1 %), mean nfev/time, the ladder (robust slope of w on tan_ref) and the angle residual s68 with tan = w / ladder."""
import json, sys
import numpy as np


def ladder(tr, w):
    m = np.isfinite(w)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.isfinite(w) & (np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27))
    return s, c


def main():
    files = [f for f in sys.argv[1:] if f.endswith('.json')]
    ev = {}
    for f in files:
        for k, e in json.load(open(f))['events'].items():
            for p, row in e.items():
                ev.setdefault((k, p), {}).update(row)
    names = sorted({n for r in ev.values() for n, v in r.items() if isinstance(v, dict) and 'chi2' in v})
    for p in ('x', 'y'):
        rows = [r for (k, pp), r in ev.items() if pp == p and all(n in r and 'chi2' in r[n] for n in names)]
        tr = np.array([r['tr'] for r in rows])
        C = np.array([[r[n]['chi2'] for n in names] for r in rows]); cmin = C.min(1)
        print(f'--- plane {p}  n {len(rows)}')
        for j, n in enumerate(names):
            w = np.array([r[n]['w'] for r in rows])
            s, c = ladder(tr, w)
            th = np.degrees(np.arctan((w - c) / s)) - np.degrees(np.arctan(tr))
            far = np.abs(th) > 5
            dt = np.abs(th[~far] - np.median(th[~far]))
            wb = np.array([r[names[int(np.argmin([r[m]['chi2'] for m in names]))]]['w'] for r in rows])
            jump = np.abs(w - wb) > 1.0
            print(f'  {n:8s} w off best-chi2 w by >1: {jump.mean() * 100:5.1f}% (|tan|>0.15: {jump[np.abs(tr) > 0.15].mean() * 100:5.1f}%)  >1% above best {np.mean(C[:, j] > 1.01 * cmin) * 100:5.1f}%  >10% {np.mean(C[:, j] > 1.10 * cmin) * 100:5.1f}%  '
                  f'nfev {np.mean([r[n]["nfev"] for r in rows]):5.0f}  {np.mean([r[n]["sec"] for r in rows]):.2f}s  '
                  f'ladder {s:6.2f}  s68 {np.percentile(dt, 68.27):.3f} deg  far(>5deg) {far.mean() * 100:4.1f}%')


if __name__ == '__main__':
    main()
