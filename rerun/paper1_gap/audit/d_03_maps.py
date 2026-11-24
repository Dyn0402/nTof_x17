"""D parts 3 and 4: per-row reference-indexed maps of tau = w0/v and v (n = 6, 8): bootstrap + split-half sets."""
import json, sys, pickle
from multiprocessing import Pool
sys.path.insert(0, '.')
from d_lib import *
NB, NSPLIT = 200, 30
MINN = 40


def grids(U, W, w, tr, n):
    C = cells(U, W, w, tr, n, minn=MINN)
    tau = np.array([c['w0'] / c['v'] if np.isfinite(c['v']) else np.nan for c in C]).reshape(n, n)
    vv = np.array([c['v'] for c in C]).reshape(n, n)
    w0 = np.array([c['w0'] for c in C]).reshape(n, n)
    nn = np.array([c['n'] for c in C]).reshape(n, n)
    return tau, vv, w0, nn


def rowjob(a):
    key, xhalf = a
    d = load(key, xhalf)
    out = {}
    rng = np.random.default_rng(11)
    for p in 'xy':
        U, W, w, tr, _ = view(d, p, idx='ref')
        for n in (6, 8):
            tau, vv, w0, nn = grids(U, W, w, tr, n)
            bt = np.zeros((NB, n, n)); bv = np.zeros((NB, n, n))
            for b in range(NB):
                i = rng.integers(0, len(U), len(U))
                bt[b], bv[b], _, _ = grids(U[i], W[i], w[i], tr[i], n)
            sp = np.zeros((NSPLIT, 2, 2, n, n))
            for s in range(NSPLIT):
                perm = rng.permutation(len(U)); h = len(U) // 2
                for k, ii in enumerate((perm[:h], perm[h:2 * h])):
                    t_, v_, _, _ = grids(U[ii], W[ii], w[ii], tr[ii], n)
                    sp[s, k, 0], sp[s, k, 1] = t_, v_
            out[(p, n)] = dict(tau=tau, v=vv, w0=w0, n=nn, btau=bt, bv=bv, split=sp, N=len(U))
    return (key, xhalf), out


if __name__ == '__main__':
    jobs = [(k, False) for k in ROWS] + [(k, True) for k in ROWS if k.startswith('det7')]
    with Pool(3) as P:
        R = P.map(rowjob, jobs, chunksize=1)
    pickle.dump(dict(R), open(f'{OUT}/maps.pkl', 'wb'))
    print('done')
