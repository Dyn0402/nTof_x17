import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
D = {k: load(rel) for k, rel in ROWS.items()}
sets = {'gold': ('run1', 'run2'), 'r625': ('r625',), 'run1': ('run1',), 'run2': ('run2',)}
def getd(name, boot=None):
    d = pd.concat([D[k] for k in sets[name]], ignore_index=True)
    if boot is not None:
        d = d.iloc[boot.integers(0, len(d), len(d))].reset_index(drop=True)
    return d
def tm(a, cut=0.2):
    return GM.tmean(a, cut)
def cellvals(d, n, q_tan=0.3):
    ex, ey = GM.edges(n, *GM.IN_X), GM.edges(n, *GM.IN_Y)
    i, j = GM.cell_of(d.xm.values, d.ym.values, ex, ey)
    R = {k: np.full((n, n), np.nan) for k in ('vx', 'vy', 'vxe', 'vye', 'Q', 'hpx', 'hpy', 'hpxe', 'hpye', 'Uc', 'Wc', 'Nq')}
    Qev = 0.5 * (d.x_q_u90.values + d.y_q_u90.values)
    qs = (np.abs(d.tan_ref_x.values) <= q_tan) & (np.abs(d.tan_ref_y.values) <= q_tan)
    for a in range(n):
        for b in range(n):
            m = (i == a) & (j == b); dc = d[m]
            R['Uc'][a, b] = (0.5 * (ex[a] + ex[a + 1]) - 200) / 100; R['Wc'][a, b] = (0.5 * (ey[b] + ey[b + 1]) - 200) / 100
            for p in 'xy':
                tr = dc[f'tan_ref_{p}'].values
                s = np.abs(tr) <= GM.TAN_LADDER
                w0, v, w0e, ve, nn = CO.line(tr[s], dc[f'{p}_w'].values[s] * 1e3)
                R[f'v{p}'][a, b], R[f'v{p}e'][a, b] = v, ve
                _, h, _, he, _ = CO.line(tr, dc[f'{p}_p0'].values - dc[f'ref_mesh_{p}'].values)
                R[f'hp{p}'][a, b], R[f'hp{p}e'][a, b] = h, he
            mq = m & qs
            R['Q'][a, b] = tm(Qev[mq]) if mq.sum() > 20 else np.nan
            R['Nq'][a, b] = mq.sum()
    return R
def vrel(R):
    rx, ry = R['vx'] / np.nanmean(R['vx']), R['vy'] / np.nanmean(R['vy'])
    wx, wy = 1 / (R['vxe'] / np.nanmean(R['vx'])) ** 2, 1 / (R['vye'] / np.nanmean(R['vy'])) ** 2
    return (rx * wx + ry * wy) / (wx + wy)
def maps(R):
    v = vrel(R); Q = R['Q'] / np.nanmean(R['Q'])
    P = v * Q; P = P / np.nanmean(P)
    return dict(v=v, Q=Q, P=P, vx=R['vx'] / np.nanmean(R['vx']), vy=R['vy'] / np.nanmean(R['vy']))
def hp_plane_removed(R, p):
    h, he = R[f'hp{p}'], R[f'hpe'.replace('hpe', f'hp{p}e')]
    ok = np.isfinite(h) & np.isfinite(he)
    fl = 0.5 * np.median(he[ok]); w = 1 / (he[ok] ** 2 + fl ** 2)
    X = np.vstack([np.ones(ok.sum()), R['Uc'][ok], R['Wc'][ok]]).T
    c = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ h[ok])
    res = h - (c[0] + c[1] * R['Uc'] + c[2] * R['Wc'])
    return res
def dish(R, p):
    h, he = R[f'hp{p}'], R[f'hp{p}e']
    ok = np.isfinite(h) & np.isfinite(he) & (he > 0)
    U, W = R['Uc'][ok], R['Wc'][ok]
    X = np.vstack([np.ones(ok.sum()), U, W, U * U / 2, W * W / 2, U * W]).T
    w = 1 / he[ok] ** 2
    A = (X * w[:, None]).T @ X
    c = np.linalg.solve(A, (X * w[:, None]).T @ h[ok])
    r = h[ok] - X @ c; chi2 = float(np.sum(w * r ** 2)); dof = ok.sum() - 6
    cov = np.linalg.inv(A) * max(chi2 / dof, 1.0)
    return c, np.sqrt(np.diag(cov)), chi2 / dof
def ols(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    X = np.vstack([np.ones(len(x)), x]).T
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(c[1]), float(np.corrcoef(x, y)[0, 1])
out = {}; imgs = {}
rng = np.random.default_rng(21)
NB = 100
for n in (4, 6):
    base = {k: cellvals(getd(k), n) for k in sets}
    M = {k: maps(base[k]) for k in base}
    imgs[n] = {k: dict(M[k], hpx=base[k]['hpx'], hpy=base[k]['hpy'], Q=base[k]['Q'], Uc=base[k]['Uc'], Wc=base[k]['Wc'], Nq=base[k]['Nq']) for k in base}
    res = {}
    pairs = {'gold_vs_r625': ('gold', 'r625'), 'run1_vs_run2': ('run1', 'run2')}
    # point values
    def stats(Ra, Rb):
        Ma, Mb = maps(Ra), maps(Rb)
        s = {}
        for q in ('v', 'Q', 'P', 'vx', 'vy'):
            s[q] = ols(Ma[q].ravel() - 1, Mb[q].ravel() - 1)
        for p in 'xy':
            s[f'hp{p}'] = ols(hp_plane_removed(Ra, p).ravel(), hp_plane_removed(Rb, p).ravel())
        return s
    for nm, (a, b) in pairs.items():
        s0 = stats(base[a], base[b]); bs = []
        for _ in range(NB):
            bs.append(stats(cellvals(getd(a, rng), n), cellvals(getd(b, rng), n)))
        for q in s0:
            arr = np.array([x[q] for x in bs])
            res[f'{nm}:{q}'] = dict(slope=s0[q][0], slope_e=float(np.nanstd(arr[:, 0])), corr=s0[q][1], corr_e=float(np.nanstd(arr[:, 1])))
        print(n, nm, {q: (round(res[f'{nm}:{q}']['slope'], 2), round(res[f'{nm}:{q}']['slope_e'], 2), round(res[f'{nm}:{q}']['corr'], 2)) for q in s0}, flush=True)
    # amplitude of v and Q maps (std over cells)
    for k in base:
        res[f'amp:{k}'] = dict(sd_v=float(np.nanstd(M[k]['v'])), sd_Q=float(np.nanstd(M[k]['Q'])), sd_P=float(np.nanstd(M[k]['P'])), meanQ=float(np.nanmean(base[k]['Q'])))
    # dish
    for k in base:
        for p in 'xy':
            c, e, rc = dish(base[k], p)
            res[f'dish:{k}:{p}'] = dict(c=c.tolist(), e=e.tolist(), rchi2=rc)
    # dish bootstrap for gold and r625
    for k in ('gold', 'r625'):
        for p in 'xy':
            B = np.array([dish(cellvals(getd(k, rng), n), p)[0] for _ in range(NB)])
            res[f'dish:{k}:{p}']['boot_e'] = B.std(0).tolist()
    out[n] = res
    # proxy: golden Q vs colsum T (4x4 only)
    if n == 4:
        co = json.load(open(f'{WORK}/cellobs.json'))
        T = np.full((4, 4), np.nan)
        for r in ('run1', 'run2'):
            t = np.full((4, 4), np.nan)
            for o in co[r]:
                t[o['i'], o['j']] = o['T']
            T = t if r == 'run1' else 0.5 * (T + t)
        Tr = T / np.nanmean(T)
        pr = {}
        for k in ('run1', 'run2', 'gold'):
            pr[k] = dict(zip(('slope', 'corr'), ols(Tr.ravel() - 1, M[k]['Q'].ravel() - 1)))
        # also the product-like cross check: colsum T vs q_u90
        res['proxy_T_vs_Q'] = pr
        res['sd_T_colsum'] = float(np.nanstd(Tr)); res['T_rel'] = Tr.tolist()
        print('proxy', pr, 'sd T', np.nanstd(Tr), 'sd Q gold', np.nanstd(M['gold']['Q']))
np.savez(f'{OUT}/gap_imgs.npz', **{f'{n}__{s}__{q}': v for n, d in imgs.items() for s, dd in d.items() for q, v in dd.items()})
json.dump(out, open(f'{OUT}/gapdish.json', 'w'), indent=1, default=float)
