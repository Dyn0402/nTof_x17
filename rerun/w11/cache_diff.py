#!/usr/bin/env python3
"""cache_diff.py <a.pkl> <b.pkl> -- compare two calibration caches window by window (same events expected).

Reports: events in common, windows bit-identical, and for windows that differ the change of the window-integrated
charge on the centre strip and its +-1..+-5 neighbours relative to the centre (b - a)."""
import pickle, sys
import numpy as np

A, B = (pickle.load(open(p, 'rb')) for p in sys.argv[1:3])
common = sorted(set(A) & set(B))
print(f'events a {len(A)} b {len(B)} common {len(common)}')
for plane in ('x', 'y'):
    same = 0; n = 0; rel = {k: [] for k in (1, 2, 3, 5)}; shape_mismatch = 0
    for e in common:
        if plane not in A[e] or plane not in B[e]:
            continue
        wa, wb = np.asarray(A[e][plane]['W'], float), np.asarray(B[e][plane]['W'], float)
        pa, pb = np.asarray(A[e][plane]['pos']), np.asarray(B[e][plane]['pos'])
        n += 1
        if wa.shape != wb.shape or not np.array_equal(pa, pb):
            shape_mismatch += 1
            continue
        if np.array_equal(wa, wb):
            same += 1
            continue
        qa, qb = wa.sum(1), wb.sum(1)
        c = int(np.argmax(qa))
        for k in rel:
            for d in (-k, k):
                if 0 <= c + d < len(qa):
                    rel[k].append((qb[c + d] - qa[c + d]) / max(qa[c], 1.0))
    print(f' {plane}: windows {n}, identical {same}, different-shape {shape_mismatch}; '
          + ', '.join(f'dQ(±{k})/Q0 median {100*np.median(v):+.2f}% p90|.| {100*np.percentile(np.abs(v),90):.2f}%'
                      for k, v in rel.items() if v))
