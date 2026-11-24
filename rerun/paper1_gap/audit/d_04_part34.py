"""D parts 3 and 4 from maps.pkl (d_03_maps.py)."""
import json, pickle, sys, itertools
sys.path.insert(0, '.')
from d_lib import *
from scipy.interpolate import RegularGridInterpolator
R = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))
AL = {k: align(k) for k in ROWS}
NB = 200


def mset(key, p):
    return R[(key, key.startswith('det7') and p == 'x')]


def centres(n):
    eu, ew = np.linspace(*UL, n + 1), np.linspace(*WL, n + 1)
    return 0.5 * (eu[1:] + eu[:-1]), 0.5 * (ew[1:] + ew[:-1])


def plane_fit(g, n):
    Uc, Wc = centres(n)
    UU, WW = np.meshgrid(Uc, Wc, indexing='ij')
    ok = np.isfinite(g)
    if ok.sum() < 6:
        return None, np.full_like(g, np.nan)
    A = np.c_[np.ones(ok.sum()), UU[ok], WW[ok]]
    c = np.linalg.lstsq(A, g[ok], rcond=None)[0]
    res = np.full_like(g, np.nan)
    res[ok] = g[ok] - A @ c
    return c, res


def resid(g, n):
    return plane_fit(g, n)[1]


def pear(a, b, minc=8):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < minc:
        return np.nan, int(ok.sum())
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(ok.sum())


def tel_pts(U, W, al):
    th = np.radians(al['theta_deg']); x, y = 200 + 100 * U, 200 + 100 * W
    X = np.cos(th) * (x - 200) - np.sin(th) * (y - 200) + 200 + al['x_offset']
    Y = np.sin(th) * (x - 200) + np.cos(th) * (y - 200) + 200 + al['y_offset']
    return X, Y


def inv_pts(X, Y, al):
    th = np.radians(al['theta_deg']); dx, dy = X - 200 - al['x_offset'], Y - 200 - al['y_offset']
    x = np.cos(th) * dx + np.sin(th) * dy + 200; y = -np.sin(th) * dx + np.cos(th) * dy + 200
    return (x - 200) / 100, (y - 200) / 100


def r_tel(ga, ala, gb, alb, n, flip=None):
    """correlation of two residual grids at the same telescope positions, evaluated at A's cell centres and at B's,
    averaged.  flip: mirror grid B (null control)."""
    if flip == 'u':
        gb = gb[::-1, :]
    elif flip == 'w':
        gb = gb[:, ::-1]
    elif flip == 'uw':
        gb = gb[::-1, ::-1]
    Uc, Wc = centres(n)
    rs, ns = [], []
    for (g1, a1, g2, a2) in ((ga, ala, gb, alb), (gb, alb, ga, ala)):
        UU, WW = np.meshgrid(Uc, Wc, indexing='ij')
        X, Y = tel_pts(UU.ravel(), WW.ravel(), a1)
        u2, w2 = inv_pts(X, Y, a2)
        f = RegularGridInterpolator((Uc, Wc), np.where(np.isfinite(g2), g2, np.nan), bounds_error=False, fill_value=np.nan)
        v2 = f(np.c_[u2, w2])
        r, nc = pear(g1.ravel(), v2)
        rs.append(r); ns.append(nc)
    return float(np.nanmean(rs)) if np.isfinite(rs).any() else np.nan, int(min(ns))


def reliability(M, n, q):
    """split-half r (average over splits) and Spearman-Brown reliability of a row's residual map."""
    rr = []
    for s in range(M['split'].shape[0]):
        a, b = resid(M['split'][s, 0, q], n), resid(M['split'][s, 1, q], n)
        r, _ = pear(a, b)
        rr.append(r)
    rr = np.array(rr); r = np.nanmean(rr) if np.isfinite(rr).any() else np.nan
    return float(r), float(np.nanstd(rr)), float(2 * r / (1 + r)) if r > -0.9 else np.nan


QN = {'tau': 0, 'v': 1}


def pair_stats(kA, kB, p, n, q, frame):
    MA, MB = mset(kA, p)[(p, n)], mset(kB, p)[(p, n)]
    keyb = 'btau' if q == 'tau' else 'bv'
    gA = lambda b: resid(MA[q] if b < 0 else MA[keyb][b], n)
    gB = lambda b: resid(MB[q] if b < 0 else MB[keyb][b], n)
    def rr(a, b, flip=None):
        if frame == 'chamber':
            return pear(a, b)
        return r_tel(a, AL[kA], b, AL[kB], n, flip)
    r0, nc = rr(gA(-1), gB(-1))
    rb = np.array([rr(gA(b), gB(b))[0] for b in range(NB)])
    nulls = [rr(gA(-1), gB(-1), f)[0] for f in ('u', 'w', 'uw')] if frame == 'tel' else \
            [pear(gA(-1), gB(-1)[::-1, :])[0], pear(gA(-1), gB(-1)[:, ::-1])[0], pear(gA(-1), gB(-1)[::-1, ::-1])[0]]
    relA, relB = reliability(MA, n, QN[q] if False else (0 if q == 'tau' else 1)), reliability(MB, n, 0 if q == 'tau' else 1)
    ceil = float(np.sqrt(relA[2] * relB[2])) if relA[2] > 0 and relB[2] > 0 else np.nan
    return dict(r=r0, err=float(np.nanstd(rb)), ncell=nc, ceil=ceil, relA=relA[0], relB=relB[0], null=nulls,
                rms_resA=float(np.nanstd(gA(-1))), rms_resB=float(np.nanstd(gB(-1))))


if __name__ == '__main__':
    out = dict(same={}, cross={}, curl={})
    # part 3a: same chamber, different rows, both frames
    for ch, rows in CHAMBERS.items():
        for kA, kB in itertools.combinations(rows, 2):
            for p in 'xy':
                for n in (6, 8):
                    for q in ('tau', 'v'):
                        for frame in ('chamber', 'tel'):
                            out['same'][f'{kA}|{kB}|{p}|{n}|{q}|{frame}'] = pair_stats(kA, kB, p, n, q, frame)
        print('same', ch, flush=True)
    # part 3b: different chambers, same slot
    slot = {k: AL[k]['z_x'] > 500 for k in ROWS}
    for kA, kB in itertools.combinations(ROWS, 2):
        if kA.split(':')[0] == kB.split(':')[0] or slot[kA] != slot[kB]:
            continue
        for p in 'xy':
            for n in (6, 8):
                for q in ('tau', 'v'):
                    out['cross'][f'{kA}|{kB}|{p}|{n}|{q}|tel'] = pair_stats(kA, kB, p, n, q, 'tel')
    print('cross', len(out['cross']), flush=True)
    # part 4: curl test per row, n = 6 (reference-indexed): plane gradients of tau_x, tau_y
    for key in ROWS:
        for xh in ([False, True] if key.startswith('det7') else [False]):
            n = 6
            Uc, Wc = centres(n)
            def grads(gx, gy):
                cx, _ = plane_fit(gx, n); cy, _ = plane_fit(gy, n)
                if cx is None or cy is None:
                    return None
                return [cx[1], cx[2], cy[1], cy[2]]   # dtx/dU, dtx/dW, dty/dU, dty/dW   (per 100 mm)
            Mx = R[(key, xh)][('x', n)]; My = R[(key, False)][('y', n)]
            g0 = grads(Mx['tau'], My['tau'])
            if g0 is None:
                continue
            B = [grads(Mx['btau'][b], My['btau'][b]) for b in range(NB)]
            B = np.array([b for b in B if b is not None])
            names = ['dtx_dU', 'dtx_dW', 'dty_dU', 'dty_dW']
            o = {nm: (float(g0[i]), float(B[:, i].std())) for i, nm in enumerate(names)}
            o['curl'] = (float(g0[1] - g0[2]), float((B[:, 1] - B[:, 2]).std()))
            o['sum'] = (float(g0[1] + g0[2]), float((B[:, 1] + B[:, 2]).std()))
            o['ncell_x'] = int(np.isfinite(Mx['tau']).sum()); o['ncell_y'] = int(np.isfinite(My['tau']).sum())
            out['curl'][key + ('|xhalf' if xh else '')] = o
            print(key, xh, {k: ('%.2e±%.1e' % v if isinstance(v, tuple) else v) for k, v in o.items()}, flush=True)
    json.dump(out, open(f'{OUT}/part34.json', 'w'), indent=1, default=float)
