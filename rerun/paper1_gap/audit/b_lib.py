import pickle, numpy as np
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/b'
MAPS = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))
RUNS = ('run1', 'run2')


def basis(U, W, deg):
    if deg == 1:
        return np.vstack([np.ones_like(U), U, W]).T
    return np.vstack([np.ones_like(U), U, W, U * U, W * W, U * W]).T


def grid(n):
    M = MAPS['run1'][n]
    UU, WW = np.meshgrid(M['U'], M['W'], indexing='ij')
    return UU, WW


def wfit(y, e, n, deg=2):
    """weighted polynomial surface on the n x n grid; returns (fit, resid) (NaN where y is NaN)."""
    UU, WW = grid(n)
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    A = basis(UU[ok], WW[ok], deg)
    w = 1 / e[ok]
    c = np.linalg.lstsq(A * w[:, None], y[ok] * w, rcond=None)[0]
    full = basis(UU.ravel(), WW.ravel(), deg) @ c
    fit = full.reshape(n, n)
    return fit, y - fit, c


def cov(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return float(np.mean((a - a.mean()) * (b - b.mean()))) * len(a) / max(len(a) - 1, 1)


def pear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def iv_slope(y1, y2, x1, x2):
    """cross-run (noise-free) slope of y on x: (cov(y1,x2)+cov(y2,x1)) / (2 cov(x1,x2)); arrays flattened."""
    ok = np.isfinite(y1) & np.isfinite(y2) & np.isfinite(x1) & np.isfinite(x2)
    y1, y2, x1, x2 = y1[ok], y2[ok], x1[ok], x2[ok]
    return (cov(y1, x2) + cov(y2, x1)) / (2 * cov(x1, x2))


def boot_cells(f, arrs, nb=2000, seed=1):
    """bootstrap over cells of statistic f(*arrs) (arrays flattened); returns (value, std, p16, p84)."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a).ravel() for a in arrs]
    ok = np.all([np.isfinite(a) for a in arrs], axis=0)
    arrs = [a[ok] for a in arrs]
    N = len(arrs[0])
    v0 = f(*arrs)
    vs = []
    for _ in range(nb):
        i = rng.integers(0, N, N)
        with np.errstate(all='ignore'):
            vs.append(f(*[a[i] for a in arrs]))
    vs = np.array(vs)
    vs = vs[np.isfinite(vs)]
    return float(v0), float(np.std(vs)), float(np.percentile(vs, 16)), float(np.percentile(vs, 84)), N
