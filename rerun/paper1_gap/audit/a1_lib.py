"""a1_lib.py -- shared helpers of audit test A1 (cell maps of the drift tilt tau = w0/v and of v_rel; stats; planes).
Outputs of the audit live under $WORK/audit/a1/."""
import os
import sys

import numpy as np

REPO = '/home/dylan/PycharmProjects/nTof_x17'
sys.path.insert(0, f'{REPO}/rerun/paper1_gap')
import gapmap as GM  # noqa: E402
import cellobs as CO  # noqa: E402

WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/a1'
os.makedirs(OUT, exist_ok=True)
UL, WL = (-1.7, 1.7), (-1.51, 1.51)
NS = (4, 6, 8, 10, 12, 16)
NMIN = 30
DEG = 180 / np.pi
_T = {}


def tables():
    if not _T:
        for run, R in GM.RUNS.items():
            _T[run] = GM.load_table(WORK, R['rel'])
    return _T


def view_arrays(run, p, index='ref', tmax=GM.TAN_LADDER):
    """U, W (100 mm from (200,200)), t (tan_ref), w (um/ns) of the events with |tan_ref| <= tmax in view p."""
    d = tables()[run]
    if index == 'ref':
        U = (d.xm.values - 200) / 100; W = (d.ym.values - 200) / 100
    else:
        U = (d.x_p0.values + 14 * d.x_tan_theta.values - 200) / 100
        W = (d.y_p0.values + 14 * d.y_tan_theta.values - 200) / 100
    t = d[f'tan_ref_{p}'].values; w = d[f'{p}_w'].values * 1e3
    s = np.abs(t) <= tmax
    return U[s], W[s], t[s], w[s]


def rline(t, y, clip=3.5, cubic=False, wt=None, nmin=NMIN):
    """robust y = a + b t (+ k t^3): same iteration as CO.line (3 passes, 3.5 robust sigma), optional event weights,
    threshold nmin events.  -> (a, b, err_a, err_b, n_kept)"""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    wt = np.ones(len(t)) if wt is None else np.asarray(wt)[ok]
    if len(t) < nmin:
        return (np.nan,) * 4 + (0,)
    cols = [np.ones(len(t)), t] + ([t ** 3] if cubic else [])
    X = np.vstack(cols).T
    keep = np.ones(len(t), bool)
    for _ in range(3):
        sw = np.sqrt(wt[keep])
        c = np.linalg.lstsq(X[keep] * sw[:, None], y[keep] * sw, rcond=None)[0]
        r = y - X @ c
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < clip * rs
    sw = np.sqrt(wt[keep])
    Xk = X[keep] * sw[:, None]
    c = np.linalg.lstsq(Xk, y[keep] * sw, rcond=None)[0]
    s2 = np.sum((y[keep] * sw - Xk @ c) ** 2) / max(keep.sum() - X.shape[1], 1)
    cov = s2 * np.linalg.inv(Xk.T @ Xk)
    return float(c[0]), float(c[1]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])), int(keep.sum())


def cellmap(U, W, t, w, n, cubic=False, symmetrize=False, seed=1):
    """n x n cells over UL x WL.  Arrays [i (U), j (W)]: w0, v, ew0, ev, N (events in the cell)."""
    eu, ew = np.linspace(*UL, n + 1), np.linspace(*WL, n + 1)
    i, j = np.searchsorted(eu, U) - 1, np.searchsorted(ew, W) - 1
    shp = (n, n)
    out = {k: np.full(shp, np.nan) for k in ('w0', 'v', 'ew0', 'ev')}
    out['N'] = np.zeros(shp)
    rng = np.random.default_rng(seed)
    for a in range(n):
        for b in range(n):
            m = np.where((i == a) & (j == b))[0]
            out['N'][a, b] = len(m)
            if len(m) < NMIN:
                continue
            tt, ww = t[m], w[m]
            wt = None
            if symmetrize:
                # per |t| bin, weight the larger side down so that n(+t) = n(-t): the cell's tan_ref distribution is
                # symmetric about zero
                wt = np.ones(len(m))
                eb = np.linspace(0, GM.TAN_LADDER, 7)
                bi = np.clip(np.searchsorted(eb, np.abs(tt)) - 1, 0, 5)
                for k in range(6):
                    p_, m_ = (bi == k) & (tt > 0), (bi == k) & (tt <= 0)
                    npl, nmi = p_.sum(), m_.sum()
                    if npl == 0 or nmi == 0:
                        wt[p_ | m_] = 0
                    elif npl > nmi:
                        wt[p_] = nmi / npl
                    elif nmi > npl:
                        wt[m_] = npl / nmi
            a0, v, ea, ev, _ = rline(tt, ww, cubic=cubic, wt=wt)
            out['w0'][a, b], out['v'][a, b], out['ew0'][a, b], out['ev'][a, b] = a0, v, ea, ev
    out['U'] = np.broadcast_to(0.5 * (eu[:-1] + eu[1:])[:, None], shp).copy()
    out['W'] = np.broadcast_to(0.5 * (ew[:-1] + ew[1:])[None, :], shp).copy()
    return out


def global_line(U, W, t, w, cubic=False, tmax=None):
    return rline(t, w, cubic=cubic, nmin=40)


def to_tau(cm, vrun):
    """tau = w0 / v_run, v_rel = v / v_run, with errors."""
    return dict(tau=cm['w0'] / vrun, etau=cm['ew0'] / vrun, vrel=cm['v'] / vrun, evrel=cm['ev'] / vrun,
                N=cm['N'], U=cm['U'], W=cm['W'])


def wls(A, y, e, scale=True):
    """weighted LS: -> coef, cov (scaled by max(1, chi2/dof) if scale), chi2, dof"""
    w = 1 / e ** 2
    Aw = A * np.sqrt(w)[:, None]
    c, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    chi2 = float(np.sum(w * (y - A @ c) ** 2)); dof = len(y) - A.shape[1]
    if scale and dof > 0:
        cov = cov * max(1.0, chi2 / dof)
    return c, cov, chi2, dof


def design(U, W, kind):
    one = np.ones_like(U)
    if kind == 'mean':
        return one[:, None]
    if kind == 'plane':
        return np.vstack([one, U, W]).T
    if kind == 'quad':
        return np.vstack([one, U, W, U * U, U * W, W * W]).T
    raise ValueError(kind)


def residual(mp, q, eq, kind):
    """subtract the weighted fit of `kind` from map q (error eq).  -> R, e, coef, cov, chi2, dof (flat valid cells mask)"""
    ok = np.isfinite(mp[q]) & np.isfinite(mp[eq]) & (mp[eq] > 0)
    U, W = mp['U'][ok], mp['W'][ok]
    A = design(U, W, kind)
    c, cov, chi2, dof = wls(A, mp[q][ok], mp[eq][ok])
    R = np.full(mp[q].shape, np.nan)
    R[ok] = mp[q][ok] - A @ c
    return R, c, cov, chi2, dof


def cross_stats(R1, e1, R2, e2, nb=1000, seed=5):
    """run-1 x run-2 statistics of two residual maps (same shape)."""
    ok = np.isfinite(R1) & np.isfinite(R2) & np.isfinite(e1) & np.isfinite(e2)
    a, b, ea, eb = R1[ok], R2[ok], e1[ok], e2[ok]
    nn = len(a)
    res = dict(ncell=int(nn))
    if nn < 4:
        return res

    def one(a, b):
        am, bm = a - a.mean(), b - b.mean()
        cov = np.mean(am * bm)
        r = np.sum(am * bm) / np.sqrt(np.sum(am ** 2) * np.sum(bm ** 2))
        return cov, r

    cov, r = one(a, b)
    rng = np.random.default_rng(seed)
    bc, br = [], []
    for _ in range(nb):
        k = rng.integers(0, nn, nn)
        c_, r_ = one(a[k], b[k])
        bc.append(c_); br.append(r_)
    bc = np.array(bc)
    res.update(rms1=float(np.sqrt(np.sum(a ** 2 / ea ** 2) / np.sum(1 / ea ** 2))),
               rms2=float(np.sqrt(np.sum(b ** 2 / eb ** 2) / np.sum(1 / eb ** 2))),
               err1=float(np.mean(ea)), err2=float(np.mean(eb)),
               r=float(r), r_err=float(np.nanstd(br)), r_fisher_err=float(1 / np.sqrt(max(nn - 3, 1))),
               cov=float(cov), cov_err=float(np.std(bc)),
               sig=float(np.sqrt(max(cov, 0))), sig_hi=float(np.sqrt(max(cov + np.std(bc), 0))),
               sig_lo=float(np.sqrt(max(cov - np.std(bc), 0))),
               var_sig=float(cov))
    # inverse-variance weighted cross-covariance (fixed weights, so still free of noise bias)
    w = 1 / (ea ** 2 + eb ** 2)
    am, bm = a - np.sum(w * a) / w.sum(), b - np.sum(w * b) / w.sum()
    res['cov_w'] = float(np.sum(w * am * bm) / w.sum())
    res['sig_w'] = float(np.sqrt(max(res['cov_w'], 0)))
    return res


# ----------------------------------------------------------------------------------------------- Part 2 helpers
LU, LW = 340.0, 302.0      # mm, the span of the U and W ranges


def deming0(x, y, sx, sy):
    """slope s of y = s x through the origin with errors on both: minimise sum (y - s x)^2/(sy^2 + s^2 sx^2)."""
    from scipy.optimize import minimize_scalar
    def f(s):
        return np.sum((y - s * x) ** 2 / (sy ** 2 + s ** 2 * sx ** 2))
    s0 = np.sum(x * y) / np.sum(x * x)
    r = minimize_scalar(f, bounds=(s0 - 5 * abs(s0) - 1, s0 + 5 * abs(s0) + 1), method='bounded')
    return float(r.x)


def fd_corners(R, e, n):
    """finite differences of a residual map R[i(U), j(W)] to cell corners.  Returns Cx = dR/dW, Cy = dR/dU (tan per
    100 mm, as functions of the corner) from a map R and its error e; both arrays (n-1, n-1) (NaN if a cell is missing)."""
    dU, dW = (LU / n) / 100, (LW / n) / 100
    s = R[:-1, :-1] + R[1:, :-1] + R[:-1, 1:] + R[1:, 1:]          # unused; placeholders for shape
    Cx = (R[:-1, 1:] + R[1:, 1:] - R[:-1, :-1] - R[1:, :-1]) / (2 * dW)
    Cy = (R[1:, :-1] + R[1:, 1:] - R[:-1, :-1] - R[:-1, 1:]) / (2 * dU)
    e2 = e ** 2
    es = np.sqrt(e2[:-1, 1:] + e2[1:, 1:] + e2[:-1, :-1] + e2[1:, :-1])
    return Cx, Cy, es / (2 * dW), es / (2 * dU)


def pear(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def curl_stats(Rx, ex, Ry, ey, n):
    """Rx[run], ex[run]: plane-subtracted tau_x maps of both runs; same for y.  -> dict of r, slope, ceilings"""
    C = {}
    for r in ('run1', 'run2'):
        cx, _, sx, _ = fd_corners(Rx[r], ex[r], n)
        _, cy, _, sy = fd_corners(Ry[r], ey[r], n)
        C[r] = (cx, cy, sx, sy)
    ok = np.ones(C['run1'][0].shape, bool)
    for r in C:
        ok &= np.isfinite(C[r][0]) & np.isfinite(C[r][1])
    out = dict(ncorner=int(ok.sum()))
    if ok.sum() < 6:
        return out
    g = lambda r, k: C[r][k][ok]
    rxx, ryy = pear(g('run1', 0), g('run2', 0)), pear(g('run1', 1), g('run2', 1))
    rxy12, rxy21 = pear(g('run2', 0), g('run1', 1)), pear(g('run1', 0), g('run2', 1))   # Cx(a) vs Cy(b)
    rxy = 0.5 * (rxy12 + rxy21)
    s12 = deming0(g('run2', 0), g('run1', 1), g('run2', 2), g('run1', 3))
    s21 = deming0(g('run1', 0), g('run2', 1), g('run1', 2), g('run2', 3))
    # noise-free cross covariances (run a x run b, never the same run)
    cov = lambda a, b: float(np.mean((a - a.mean()) * (b - b.mean())))
    cxx, cyy = cov(g('run1', 0), g('run2', 0)), cov(g('run1', 1), g('run2', 1))
    cxy = 0.5 * (cov(g('run2', 0), g('run1', 1)) + cov(g('run1', 0), g('run2', 1)))
    out.update(rxx=rxx, ryy=ryy, rxy12=rxy12, rxy21=rxy21, rxy=rxy, frac=float(rxy / np.sqrt(max(rxx * ryy, 1e-9))),
               slope12=s12, slope21=s21, slope=0.5 * (s12 + s21), cxx=cxx, cyy=cyy, cxy=cxy,
               slope_cov=float(cxy / np.sqrt(max(cxx * cyy, 1e-12))) if cxx > 0 and cyy > 0 else np.nan,
               rms_cx=float(np.sqrt(max(cxx, 0))), rms_cy=float(np.sqrt(max(cyy, 0))))
    return out


def helmholtz_system(n, maps_list):
    """maps_list: list of (R_x, e_x, R_y, e_y) (n x n each).  Returns A (rows weighted), y (weighted), mask info."""
    N1 = n + 1
    nu, nw = LU / n, LW / n
    rows, ys, ids = [], [], []
    for mi, (Rx, ex, Ry, ey) in enumerate(maps_list):
        for comp, R, e in (('x', Rx, ex), ('y', Ry, ey)):
            for i in range(n):
                for j in range(n):
                    if not (np.isfinite(R[i, j]) and np.isfinite(e[i, j]) and e[i, j] > 0):
                        continue
                    row = np.zeros(N1 * N1)
                    k = lambda a, b: a * N1 + b
                    if comp == 'x':
                        for jj in (j, j + 1):
                            row[k(i + 1, jj)] += 1 / (2 * nu); row[k(i, jj)] -= 1 / (2 * nu)
                    else:
                        for ii in (i, i + 1):
                            row[k(ii, j + 1)] += 1 / (2 * nw); row[k(ii, j)] -= 1 / (2 * nw)
                    rows.append(row / e[i, j]); ys.append(R[i, j] / e[i, j]); ids.append((mi, comp, i, j, e[i, j]))
    return np.array(rows), np.array(ys), ids


def helmholtz_solve(A, y):
    """min-norm WLS: -> Phi, standardized residual, leverage, rank, cov (pinv of normal matrix)"""
    M = A.T @ A
    cov = np.linalg.pinv(M, rcond=1e-10, hermitian=True)
    phi = cov @ (A.T @ y)
    res = y - A @ phi
    H = np.einsum('ij,jk,ik->i', A, cov, A)
    rank = int(np.linalg.matrix_rank(M, tol=1e-9 * np.linalg.norm(M, 2)))
    return phi, res, H, rank, cov
