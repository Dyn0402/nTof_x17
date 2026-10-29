"""Summarise t9-format residual JSONs: per view, amplitude-weighted mean of the per-bin medians (bins >= 800 ADC) of
centre k=-1..+3, +-1 (mean of both) k=-2,0,+2, +-2 (mean) k=0,+4.  Usage: python -I t13_table.py file.json ..."""
import json, sys, os
import numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f)); K = np.array(d.get('K', list(range(-4, 9))))
    for v in 'xy':
        ms, ws = [], []
        for key, r in d['summary'].items():
            if key.startswith(v + '_') and r['lo'] >= 800:
                ms.append(np.array(r['median'] if 'median' in r else r['mean'])); ws.append(r['n'])
        if not ms:
            continue
        m = np.average(np.array(ms), axis=0, weights=ws); i = lambda k: int(np.where(K == k)[0][0])
        n1 = lambda k: 0.5 * (m[1][i(k)] + m[3][i(k)]); n2 = lambda k: 0.5 * (m[0][i(k)] + m[4][i(k)])
        print(f"{os.path.basename(f)[:-5]:28s} {v}: centre k-1..+3 " + ' '.join(f'{m[2][i(k)]:+.3f}' for k in (-1, 0, 1, 2, 3))
              + f" | ±1 k-2,0,+2 {n1(-2):+.3f} {n1(0):+.3f} {n1(2):+.3f} | ±2 k0,+4 {n2(0):+.3f} {n2(4):+.3f}")
