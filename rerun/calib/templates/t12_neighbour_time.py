"""T12: when does the neighbour charge arrive?  Model-free, head-on unsaturated planes (|tan| < 0.05, max < 3550).
Per plane: centre = brightest strip; stack W[c+o](k) / W[c](peak) at sample offsets k from the centre's peak, o = 0,
+-1 (mean of both), +-2; medians.  Also the cumulative charge ratio sum_{k<=K} (q-1 + q+1) / sum_{k<=K} q0 vs K.
Usage: python -I t12_neighbour_time.py -> results/t12_neighbour_time.json"""
import json, os, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/{}.pkl')
K = np.arange(-4, 22)
out = {}
for det in ('det2', 'det3', 'det4', 'det6', 'det7'):
    ev = pickle.load(open(CACHE.format(det), 'rb'))
    for plane in 'xy':
        S = []
        for e in ev.values():
            if plane not in e or abs(e[f'tan_{plane}']) >= 0.05:
                continue
            W = np.asarray(e[plane]['W'], float)
            if W.max() >= 3550:
                continue
            c = int(np.argmax(W.max(axis=1))); j0 = int(np.argmax(W[c]))
            if c < 2 or c > len(W) - 3 or W[c, j0] < 400:
                continue
            A = np.full((3, len(K)), np.nan)
            for ik, k in enumerate(K):
                j = j0 + k
                if 0 <= j < W.shape[1]:
                    A[0, ik] = W[c, j]; A[1, ik] = 0.5 * (W[c - 1, j] + W[c + 1, j]); A[2, ik] = 0.5 * (W[c - 2, j] + W[c + 2, j])
            S.append(A / W[c, j0])
        S = np.array(S); m = np.nanmedian(S, axis=0)
        cum = np.nancumsum(np.nan_to_num(np.nanmean(S, axis=0)), axis=1)
        out[f'{det}_{plane}'] = dict(n=len(S), K=K.tolist(), centre=m[0].tolist(), n1=m[1].tolist(), n2=m[2].tolist())
        sel = [0, 2, 4, 6, 8, 10, 14, 20]
        print(f'{det} {plane} n={len(S)}  k: ' + ' '.join(f'{K[i]:+d}' for i in [4 + s for s in sel if 4 + s < len(K)]))
        for name, r in (('centre', m[0]), ('±1/2  ', m[1]), ('±2/2  ', m[2])):
            print(f'   {name} ' + ' '.join(f'{r[4 + s]:+.2f}' for s in sel if 4 + s < len(K)))
json.dump(out, open(f'{HERE}/results/t12_neighbour_time.json', 'w'), indent=1)
