#!/usr/bin/env python3
"""boundary.py <window pickles> -- the same-time neighbour signal across an ASIC/connector boundary (channel jump) against inside one."""
import pickle, sys, numpy as np
sys.path.insert(0, '..')
R = []
for f in sys.argv[1:]:
    for p in pickle.load(open(f, 'rb'))['planes']:
        if abs(p['w']) > 1.2e-3 or len(p['pos']) < 7: continue
        W, ch, pos = p['W'].astype(float), p['ch'].astype(int), p['pos'].astype(float)
        ts = np.arange(W.shape[1]) * 60.0; t0 = p['t0']
        ek = (ts >= t0 - 60) & (ts <= t0 + 180)
        Q = W[:, ek].sum(1)
        pc = p['p0'] + p['w'] * 90
        c = int(np.argmin(np.abs(pos - pc)))
        if c < 2 or c >= len(pos) - 2 or Q[c] < 300: continue
        phi = (pc - pos[c]) / 0.78; s = 1 if phi >= 0 else -1
        if abs(phi) < 0.25: continue
        f1, f2 = c - s, c - 2 * s
        if p['censor'].astype(bool)[[c, f1, f2]][:, ek].any(): continue
        j1 = abs(int(ch[f1]) - int(ch[c])) != 1
        j2 = abs(int(ch[f2]) - int(ch[f1])) != 1
        R.append((p['view'], int(ch[c]), j1, j2, Q[f1] / Q[c], Q[f2] / Q[c]))
import pandas as pd
T = pd.DataFrame(R, columns='view ch j1 j2 r1 r2'.split())
import json, os
RES = {}
for v in 'xy':
    g = T[T.view == v]
    for (a, b), h in g.groupby(['j1', 'j2']):
        e = lambda x: np.std([np.median(np.random.choice(x, len(x))) for _ in range(300)])
        RES[f'{v}|{a}|{b}'] = dict(n=len(h), r1=float(h.r1.median()), r1e=float(e(h.r1.values)), r2=float(h.r2.median()), r2e=float(e(h.r2.values)))
        print(v, 'c|f1 jump', a, ' f1|f2 jump', b, 'n', len(h), 'r1 %.4f +- %.4f  r2 %.4f +- %.4f' % (h.r1.median(), e(h.r1.values), h.r2.median(), e(h.r2.values)))

json.dump(RES, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'coupling_boundary.json'), 'w'), indent=1)
