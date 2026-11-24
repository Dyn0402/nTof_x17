"""r1_04_part2.py: prediction against measurement (maps) and free (a, b).  -> part2.json"""
import sys
from r1_lib2 import *
NB = 150
rng = np.random.default_rng(21)
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
NCELL = (4, 6, 8)
J = {}

def wts(Ae):
    ok = np.isfinite(Ae)
    fl = 0.5 * np.median(Ae[ok]); return 1 / (Ae ** 2 + fl ** 2)

def wplane(M, y, Ae):
    ok = np.isfinite(y) & np.isfinite(Ae)
    w = wts(Ae)[ok]
    X = np.vstack([np.ones(ok.sum()), M['Uc'][ok], M['Wc'][ok]]).T
    c = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y[ok])
    return c

def measured(r, p, n):
    v = V[(r, p)]; vg = CO.line(v['tr'], v['w'])[1]
    M = CL.cellmap(v['U'], v['W'], v['w'], v['tr'], n)
    return M, M['w0'] / vg, M['w0e'] / vg, vg

def mstats(A, P, Ae, M, rng, nb=1000):
    """const-removed full map, plane and fine: Pearson r and slope of measured on predicted, cell bootstrap"""
    ok = np.isfinite(A) & np.isfinite(P) & np.isfinite(Ae)
    out = {}
    w = wts(Ae)
    def one(idx, kind):
        a, pp, ww = A[ok][idx], P[ok][idx], w[ok][idx]
        if kind == 'full':
            pass
        elif kind == 'fine':
            Uc, Wc = M['Uc'][ok][idx], M['Wc'][ok][idx]
            X = np.vstack([np.ones(len(a)), Uc, Wc]).T
            ca = np.linalg.solve((X * ww[:, None]).T @ X, (X * ww[:, None]).T @ a)
            cp = np.linalg.solve((X * ww[:, None]).T @ X, (X * ww[:, None]).T @ pp)
            a = a - X @ ca; pp = pp - X @ cp
        a = a - np.sum(ww * a) / ww.sum(); pp = pp - np.sum(ww * pp) / ww.sum()
        r = np.sum(ww * a * pp) / np.sqrt(np.sum(ww * a * a) * np.sum(ww * pp * pp))
        s = np.sum(ww * a * pp) / np.sum(ww * pp * pp)
        return r, s
    N = ok.sum()
    for kind in ('full', 'fine'):
        r0, s0 = one(np.arange(N), kind)
        bs = np.array([one(rng.integers(0, N, N), kind) for _ in range(nb)])
        out[kind] = dict(r=float(r0), r_err=float(bs[:, 0].std()), slope=float(s0), slope_err=float(bs[:, 1].std()), N=int(N))
    return out

def planes_event(r, p):
    """event-level plane of the measured tilt: tau = a + bU + cW (tan per 100 mm) from w = c0 + v tr + v(bU+cW), bootstrap errors"""
    v = V[(r, p)]
    X = np.vstack([np.ones(len(v['U'])), v['tr'], v['U'], v['W']]).T
    def f(idx):
        c, keep, rss, rs = CL.lsq_clip(X[idx], v['w'][idx])
        return np.array([c[2] / c[1], c[3] / c[1]])
    c0 = f(np.arange(len(v['w'])))
    bs = np.array([f(rng.integers(0, len(v['w']), len(v['w']))) for _ in range(60)])
    return c0, bs.std(axis=0)

def pplane(v, pred):
    X = np.vstack([np.ones(len(v['U'])), v['U'], v['W']]).T
    return np.linalg.lstsq(X, pred, rcond=None)[0][1:]

CASES = [('T', 'run1', 'run2'), ('T', 'run2', 'run1'), ('T', 'run1', 'run1'), ('T', 'run2', 'run2'),
         ('q', 'run1', 'run2'), ('q', 'run2', 'run1'), ('q', 'r625', 'r625')]
for kind, src, dst in CASES:
    fld = Fld(f'{kind}_{src}')
    for p in 'xy':
        v = V[(dst, p)]
        gh, gg = regressors(v, p, fld, 'h', BUNDLE)
        pred = gh + 0.5 * gg
        key = f'{kind}:{src}->{dst}:{p}'
        J[key] = dict(plane_pred={'mesh': pplane(v, gh).tolist(), 'gap_half': pplane(v, 0.5 * gg).tolist(), 'sum': pplane(v, pred).tolist()})
        for n in NCELL:
            M, A, Ae, vg = measured(dst, p, n)
            for nm, P_ in (('nominal', gh + 0.5 * gg), ('mesh', gh), ('gap', 0.5 * gg)):
                Pm = cellmean(v['U'], v['W'], P_, n)
                J[key][f'n{n}_{nm}'] = mstats(A, Pm, Ae, M, rng, 600)
        print(key, {k: (round(J[key][f'n8_nominal'][k]['r'], 3), round(J[key][f'n8_nominal'][k]['slope'], 3)) for k in ('full', 'fine')}, flush=True)
# measured event-level planes
for r in ('run1', 'run2', 'r625'):
    for p in 'xy':
        c, e = planes_event(r, p)
        J[f'plane_meas:{r}:{p}'] = dict(own_cross=c.tolist(), err=e.tolist())
        print('plane', r, p, c, e)
json.dump(J, open(f'{OUT}/part2_maps.json', 'w'), indent=1)
