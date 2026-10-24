"""Null test for the rigid fit: move the measured short pattern to other places on the pogo grid
(mirror rows/columns, cyclic row shifts) and refit. If a rigid displacement explained the real
pattern, the real data should fit clearly better than the relocated copies. Writes null_test.json."""
import json, numpy as np, multiprocessing as mp
import model as M
C = np.load('corr_feu6.npz')['C']; REAL = C[M._PA, M._PB] > 0.9
IDX = {}
for i, (a, b, k) in enumerate(M.PAIRS):
    IDX[(M.POS[a], M.POS[b])] = i; IDX[(M.POS[b], M.POS[a])] = i


def transformed(T):
    out = np.zeros(len(M.PAIRS), bool); n = 0
    for i, (a, b, k) in enumerate(M.PAIRS):
        if REAL[i]:
            j = IDX.get((T(M.POS[a]), T(M.POS[b])))
            if j is not None:
                out[j] = True; n += 1
    return out, n

TRANS = {'real': lambda p: p, 'mirror_rows': lambda p: (p[0], 29 - p[1]), 'mirror_cols': lambda p: (9 - p[0], p[1]),
         'mirror_both': lambda p: (9 - p[0], 29 - p[1]), 'rows+15': lambda p: (p[0], (p[1] + 15) % 30),
         'rows+15_mc': lambda p: (9 - p[0], (p[1] + 15) % 30)}
DX = np.arange(-0.7, 0.7001, 0.05); TH = np.arange(-3, 3.001, 0.2); RP = [0.15, 0.2, 0.25, 0.3]


def work(args):
    name, th = args
    meas, _ = transformed(TRANS[name])
    best = (0,)
    for rp in RP:
        for dx in DX:
            for dy in DX:
                l = M.fast_labels(dx, dy, th, rp)
                tp, fp, fn = M.fast_score(l[0], l[1], meas)
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best[0]: best = (f1, tp, fp, fn, float(th), rp, float(dx), float(dy))
    return name, best


if __name__ == '__main__':
    with mp.Pool(8) as p:
        res = p.map(work, [(n, th) for n in TRANS for th in TH])
    out = {}
    for n in TRANS:
        b = max((r for nn, r in res if nn == n), key=lambda r: r[0])
        out[n] = dict(n_meas=int(transformed(TRANS[n])[0].sum()), f1=b[0], tp=b[1], fp=b[2], fn=b[3], th=b[4], rp=b[5], dx=b[6], dy=b[7])
        print(n, out[n])
    json.dump(out, open('null_test.json', 'w'), indent=1)
