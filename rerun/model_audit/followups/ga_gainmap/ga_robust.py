#!/usr/bin/env python3
"""ga_robust.py <npz> -- robust split-half numbers of an in-fit gain map: bulk channels (|ln g| < 0.3 in both halves)
signal = sqrt(cov(A, B)), noise = std(A-B)/2, and the outlier channels (|ln g| > 0.5 in BOTH halves) listed."""
import sys
import numpy as np
z = np.load(sys.argv[1])
for p in 'xy':
    m = (z['info_A_' + p] > 50) & (z['info_B_' + p] > 50)
    a, b = np.log(z['map_A_' + p]), np.log(z['map_B_' + p])
    bulk = m & (np.abs(a) < 0.3) & (np.abs(b) < 0.3)
    x, y = a[bulk] - a[bulk].mean(), b[bulk] - b[bulk].mean()
    c = np.mean(x * y)
    out = np.flatnonzero(m & (np.abs(a) > 0.5) & (np.abs(b) > 0.5) & (np.sign(a) == np.sign(b)))
    print(f"{p}: channels {m.sum()}, bulk {bulk.sum()}: signal {np.sign(c) * np.sqrt(abs(c)):+.3f} noise {np.std(x - y) / 2:.3f} "
          f"corr {np.corrcoef(x, y)[0, 1]:.2f}; reproducible outliers |ln g|>0.5: {len(out)} "
          f"{[(int(ch), round(float(np.sqrt(z['map_A_' + p][ch] * z['map_B_' + p][ch])), 2)) for ch in out][:20]}")
