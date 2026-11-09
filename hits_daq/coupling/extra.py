#!/usr/bin/env python3
"""extra.py <window pickles> -- (1) noise correlation by strip distance, median over planes, inside one chip and across a channel jump;
(2) grid pinning: mesh-end charge centroid from the first samples only; (3) near-side share vs the 31.2 mm beat."""
import multiprocessing as mp, pickle, sys, numpy as np, pandas as pd
def one(p):
    W, nz, pos, ch = p['W'].astype(float), p['noise'].astype(float), p['pos'].astype(float), p['ch'].astype(int)
    ts = np.arange(W.shape[1]) * 60.0; t0 = float(p['t0']); v = p['view']
    N = []
    pre = ts < t0 - 180
    if pre.sum() >= 4 and len(pos) >= 5:
        X = W[:, pre] / nz[:, None]
        X = X - X.mean(1, keepdims=True)                  # slow baseline of the strip in this event removed
        for d in range(1, min(6, len(pos))):
            for i in range(len(pos) - d):
                jump = np.any(np.abs(np.diff(ch[i:i + d + 1])) != 1)
                N.append((v, d, bool(jump), float(np.mean(X[i] * X[i + d])), float(np.mean(X[i] * X[i]))))
    r = None
    if abs(p['w']) <= 1.2e-3 and len(pos) >= 7:
        pc = p['p0'] + p['w'] * 30
        c = int(np.argmin(np.abs(pos - pc)))
        if 2 <= c < len(pos) - 2:
            k = (ts >= t0 - 60) & (ts <= t0 + 60)
            Q = W[:, k].sum(1); s = np.abs(np.arange(len(pos)) - c) <= 2
            q = np.maximum(Q[s], 0)
            if Q[c] > 150 and q.sum() > 0 and not p['censor'].astype(bool)[s][:, k].any():
                phi = (pc - pos[c]) / 0.78; sgn = 1 if phi >= 0 else -1
                r = (v, float(np.sum(q * pos[s]) / q.sum()), float(pc), float(abs(phi)), Q[c + sgn] / Q[c])
    return N, r
fs = sys.argv[1:]
P = [p for f in fs for p in pickle.load(open(f, 'rb'))['planes']]
with mp.Pool(2) as pool: R = pool.map(one, P, chunksize=64)
N = pd.DataFrame([x for a, _ in R for x in a], columns='view d jump cov var'.split())
print('noise: mean cov / mean var by strip distance (after removing each strip-event mean of the pre-signal samples)')
g = N.groupby(['view', 'jump', 'd'])
print((g['cov'].mean() / g['var'].mean()).unstack(['view', 'jump']).round(4).to_string()); print(g.size().unstack(['view', 'jump']).to_string())
print('median cov/var'); print((g['cov'].median() / g['var'].median()).unstack(['view', 'jump']).round(4).to_string())
T = pd.DataFrame([r for _, r in R if r is not None], columns='view cen pc phi nearratio'.split())
Z = lambda x, per: abs(np.exp(2j * np.pi * x / per).mean()) ** 2 * len(x)
for v in 'xy':
    g = T[T.view == v]
    print(v, 'first-samples centroid n', len(g), 'Rayleigh Z:', ' '.join(f'{per}:{Z(g.cen.values, per):.1f}' for per in (.76, .78, .79, .80, .81, .84)))
g = T[T.view == 'x'].copy()
g['pb'] = np.floor(g.pc % 31.2 / 31.2 * 8).astype(int); g['pp'] = pd.cut(g.phi, [0, .15, .3, .45, .55])
print('X near-side ratio (first samples) vs beat bin, by phase bin (median; n)')
print(g.pivot_table(index='pp', columns='pb', values='nearratio', aggfunc='median', observed=True).round(3).to_string())
print(g.pivot_table(index='pp', columns='pb', values='nearratio', aggfunc='size', observed=True).to_string())
