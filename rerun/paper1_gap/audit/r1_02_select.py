"""r1_02_select.py: choose K and the penalty by how well the gap field of one golden run predicts the tilt of the other."""
import sys, itertools
from r1_lib import *
D = {r: load(r) for r in ('run1', 'run2')}
NS = (6, 8)
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
MEAS = {}
for r in D:
    for p in 'xy':
        v = V[(r, p)]; vg = CO.line(v['tr'], v['w'])[1]
        for n in NS:
            M = CL.cellmap(v['U'], v['W'], v['w'], v['tr'], n)
            MEAS[(r, p, n)] = (M, (M['w0'] - np.nanmean(M['w0'])) / vg, M['w0e'] / vg, vg)

def fieldT(r, K, lam, beta=BETA_G):
    m = D[r]['m']
    return GapField(m.U.values, m.W.values, np.log(m.T_amp.values), K, lam, beta)

def predmap(F, r, p, n, hw='h', b=0.5, a=1.0):
    v = V[(r, p)]
    gh = hgrad(v['U'], v['W'], hw)[0 if p == 'x' else 1]
    gg = F.grad(v['U'], v['W'])[0 if p == 'x' else 1]
    return cellmean(v['U'], v['W'], a * gh + b * gg, n)

def rss(pred, A, Ae):
    ok = np.isfinite(pred) & np.isfinite(A) & np.isfinite(Ae)
    fl = 0.5 * np.median(Ae[ok]); w = 1 / (Ae[ok] ** 2 + fl ** 2)
    d = A[ok] - pred[ok]; d = d - np.sum(w * d) / np.sum(w)
    tot = A[ok] - np.sum(w * A[ok]) / np.sum(w)
    return float(np.sum(w * d ** 2)), float(np.sum(w * tot ** 2)), int(ok.sum())

if __name__ == '__main__':
    out = {}
    LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    Ks = [4, 6, 8, 10, 12]
    for K, lam in itertools.product(Ks, LAMS):
        res = {}
        tot = {n: [0, 0] for n in NS}
        for src, dst in (('run1', 'run2'), ('run2', 'run1')):
            F = fieldT(src, K, lam)
            for p in 'xy':
                for n in NS:
                    M, A, Ae, vg = MEAS[(dst, p, n)]
                    pr = predmap(F, dst, p, n)
                    r_, t_, nn = rss(pr, A, Ae)
                    tot[n][0] += r_; tot[n][1] += t_
        out[(K, lam)] = {n: 1 - tot[n][0] / tot[n][1] for n in NS}
        print(K, lam, {n: round(out[(K, lam)][n], 3) for n in NS}, flush=True)
    # also: h only (no gap)
    tot = {n: [0, 0] for n in NS}
    F = fieldT('run1', 6, 1e-2)
    for dst in ('run1', 'run2'):
        for p in 'xy':
            for n in NS:
                M, A, Ae, vg = MEAS[(dst, p, n)]
                r_, t_, nn = rss(predmap(F, dst, p, n, b=0.0), A, Ae); tot[n][0] += r_; tot[n][1] += t_
    print('mesh only', {n: round(1 - tot[n][0] / tot[n][1], 3) for n in NS})
    pickle.dump(out, open(f'{OUT}/select.pkl', 'wb'))
