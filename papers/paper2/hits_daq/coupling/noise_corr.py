#!/usr/bin/env python3
"""noise_corr.py <window pickles> -- noise correlation between strips by distance, from post-signal samples (> t0 + 1300 ns), near-normal
planes, on strips > 3 pitches from the track, each strip-event detrended linearly.  Capacitance between strips would correlate their noise."""
import multiprocessing as mp, pickle, sys, numpy as np, pandas as pd
def one(p):
    if abs(p['w']) > 1.2e-3 or len(p['pos']) < 9: return []
    W, nz, pos, ch = p['W'].astype(float), p['noise'].astype(float), p['pos'].astype(float), p['ch'].astype(int)
    ts = np.arange(W.shape[1]) * 60.0
    k = ts > float(p['t0']) + 1300
    if k.sum() < 6: return []
    pc = p['p0'] + p['w'] * 400
    keep = np.abs(pos - pc) > 3 * 0.78
    X = W[:, k] / nz[:, None]
    tt = np.arange(k.sum()) - (k.sum() - 1) / 2
    X = X - X.mean(1, keepdims=True) - (X @ tt / (tt @ tt))[:, None] * tt
    out = []
    for i in range(len(pos)):
        for d in range(1, 6):
            j = i + d
            if j < len(pos) and keep[i] and keep[j] and np.sign(pos[i] - pc) == np.sign(pos[j] - pc):
                out.append((p['view'], d, bool(np.any(np.abs(np.diff(ch[i:j + 1])) != 1)), float(np.mean(X[i] * X[j])), float(np.mean(X[i] ** 2))))
    return out
P = [p for f in sys.argv[1:] for p in pickle.load(open(f, 'rb'))['planes']]
with mp.Pool(2) as pool: R = [x for a in pool.map(one, P, chunksize=64) for x in a]
N = pd.DataFrame(R, columns='view d jump cov var'.split())
g = N.groupby(['view', 'jump', 'd'])
print('n'); print(g.size().unstack(['view', 'jump']).to_string())
print('mean cov / mean var'); print((g['cov'].mean() / g['var'].mean()).unstack(['view', 'jump']).round(4).to_string())
se = g['cov'].std() / np.sqrt(g.size()) / g['var'].mean()
print('err'); print(se.unstack(['view', 'jump']).round(4).to_string())
