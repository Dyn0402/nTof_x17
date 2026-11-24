import sys
from r2_lib import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
K, LAM = 10, 0.1
SP = Spl(K)
def design(d, tanterms=True):
    tx, ty = d.x_tan_theta.values, d.y_tan_theta.values
    cols = [d.lq.values - np.median(d.lq)]
    if tanterms: cols += [tx, ty, tx * tx, ty * ty, tx * ty]
    return np.vstack(cols).T
def fitmap(d, K=K, lam=LAM, tanterms=True, ext=None, nmax=250000, seed=1):
    if len(d) > nmax: d = d.sample(nmax, random_state=seed)
    y = np.log(d.Tm.values); y = y - y.mean()
    ex = design(d, tanterms) if ext is None else ext
    c, e, keep, rs = Spl(K).fit(d.U.values, d.Wc.values, y, lam, extra=ex)
    return c, e, rs, keep.mean()
if __name__ == '__main__':
    uu, ww = grid(60)
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.5))
    res = {}
    for i, arm in enumerate('AC'):
        for j, w in enumerate(['cosmic', 'late']):
            d = load(w, arm)
            c, e, rs, kp = fitmap(d)
            print(arm, w, len(d), 'extras', np.round(e, 3), 'rs', round(rs, 3), 'keep', round(kp, 3), flush=True)
            B = Spl(K).basis(uu.ravel(), ww.ravel())
            m = (B @ c).reshape(uu.shape)
            res[(arm, w)] = m
            a = ax[i, 2 * j]
            im = a.pcolormesh(uu * 100 + 200, ww * 100 + 200, m, vmin=-0.08, vmax=0.08, cmap='RdBu_r'); a.set_title(f'{arm} {w} ln T')
            a.set_aspect('equal')
        # difference cos-late correlation
    fig.savefig(f'{OUT}/explore_maps.png', dpi=90)
    for arm in 'AC':
        a, b = res[(arm, 'cosmic')].ravel(), res[(arm, 'late')].ravel()
        inside = (np.abs(uu.ravel()) < 1.8) & (np.abs(ww.ravel()) < 1.8)
        print(arm, 'corr cosmic vs late', np.corrcoef(a[inside], b[inside])[0, 1], 'sd', a[inside].std(), b[inside].std())
