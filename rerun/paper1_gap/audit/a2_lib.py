"""a2_lib.py -- shared helpers of audit test a2: event datasets, tensor B-spline maps (indep / phi / phi+), application to a
test set exactly as track_grid.py does (constants refit with TT.robust_const), metrics, paired bootstrap."""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline
REPO = '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap'
sys.path.insert(0, REPO)
import gapmap as GM, track_test as TT, track_grid as TG  # noqa: E402
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/a2'
UD, WD = (-1.75, 1.75), (-1.55, 1.55)


class Data:
    """Events with both views.  Arrays: U, W, w{x,y} (um/ns), tr{x,y}, tp{x,y}; order key for time."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def sub(self, m):
        m = np.asarray(m)
        return Data(**{k: (v[m] if isinstance(v, np.ndarray) and len(v) == self.n else v) for k, v in self.__dict__.items() if k != 'n'}, n=int(np.sum(m) if m.dtype == bool else len(m)))


def load(run):
    R = GM.RUNS[run]
    Ux, Wx, wx, trx, tpx = TG.table(WORK, R, 'x')
    Uy, Wy, wy, tr_y, tpy = TG.table(WORK, R, 'y')
    assert len(Ux) == len(Uy) and np.allclose(Ux, Uy)
    d = GM.load_table(WORK, R['rel'])
    sel = np.abs(d['tan_ref_x'].values) <= GM.TAN_LADDER
    assert sel.sum() == len(Ux) == (np.abs(d['tan_ref_y'].values) <= GM.TAN_LADDER)[sel].sum() or True
    acq = d['acq_tag'].values[sel]; ts = d['timestamp'].values[sel].astype(float)
    # time rank: file order (acq_tag sorts chronologically), then timestamp inside the file
    o = np.lexsort((ts, acq)); rank = np.empty(len(o), int); rank[o] = np.arange(len(o))
    files = {a: i for i, a in enumerate(sorted(set(acq)))}
    fid = np.array([files[a] for a in acq])
    return Data(U=Ux, W=Wx, wx=wx, trx=trx, tpx=tpx, wy=wy, trY=tr_y, tpy=tpy, rank=rank, fid=fid,
                xm=d['xm'].values[sel], ym=d['ym'].values[sel], p0x=d['x_p0'].values[sel], p0y=d['y_p0'].values[sel], n=len(Ux))


def concat(a, b):
    kw = {}
    for k, v in a.__dict__.items():
        if k == 'n': continue
        kw[k] = np.concatenate([v, getattr(b, k)])
    return Data(**kw, n=a.n + b.n)


def s68(a):
    return TT.s68(a)


# ---------------------------------------------------------------- splines
def knots(K, lo, hi):
    return np.concatenate([[lo] * 3, np.linspace(lo, hi, K - 2), [hi] * 3])


_cache = {}
def basis(x, K, lo, hi, nu=0):
    x = np.clip(x, lo, hi)
    key = (K, lo, hi, nu)
    if key not in _cache:
        b = BSpline(knots(K, lo, hi), np.eye(K), 3)
        _cache[key] = b if nu == 0 else b.derivative(nu)
    B = _cache[key](x)
    B[np.abs(B) < 1e-14] = 0
    return sp.csr_matrix(B)


def tensor(A, B):
    K1, K2 = A.shape[1], B.shape[1]
    return A[:, np.repeat(np.arange(K1), K2)].multiply(B[:, np.tile(np.arange(K2), K1)]).tocsr()


def pen1(K):
    D = np.diff(np.eye(K), 2, axis=0)
    return D.T @ D


def pen2(K):
    P = pen1(K); I = np.eye(K)
    return np.kron(P, I) + np.kron(I, P)


VARIANTS = ('indep', 'phi', 'phiplus')


class Spec:
    """Column layout of a map: offset field a_x, a_y and tr-coefficient s_x, s_y (um/ns) as linear functions of coefs."""
    def __init__(self, variant, K, KS, kappa=1.0, k_extra=None, extra_fac=30.0):
        self.variant, self.K, self.KS, self.kappa = variant, K, KS, kappa
        self.k_extra = k_extra or K; self.extra_fac = extra_fac
        sizes = []   # (name, ncols, kind, Kdim)
        if variant == 'indep':
            sizes += [('ax', K * K, 'sp', K, 1.0), ('ay', K * K, 'sp', K, 1.0)]
        else:
            sizes += [('phi', K * K, 'sp', K, 1.0), ('cx', 1, 'c', 0, 0), ('cy', 1, 'c', 0, 0)]
            if variant == 'phiplus':
                ke = self.k_extra
                sizes += [('ax2', ke * ke, 'sp', ke, extra_fac), ('ay2', ke * ke, 'sp', ke, extra_fac)]
        if KS == 0:
            sizes += [('sx', 1, 'c', 0, 0), ('sy', 1, 'c', 0, 0)]
        else:
            sizes += [('sx', KS * KS, 'sp', KS, 1.0), ('sy', KS * KS, 'sp', KS, 1.0)]
        self.blocks = sizes
        self.off = {}; o = 0
        for nm, n, *_ in sizes:
            self.off[nm] = (o, o + n); o += n
        self.nc = o

    def penalty(self, lam):
        P = np.zeros((self.nc, self.nc))
        for nm, n, kind, Kd, fac in self.blocks:
            if kind == 'sp':
                a, b = self.off[nm]; P[a:b, a:b] = lam * fac * pen2(Kd)
        return sp.csr_matrix(P)

    def _blk(self, nm, M, n):
        a, b = self.off[nm]
        out = sp.lil_matrix((n, self.nc)) if False else None
        return sp.hstack([sp.csr_matrix((n, a)), M, sp.csr_matrix((n, self.nc - b))], format='csr')

    def fields(self, U, W):
        """-> FAx, FAy, FSx, FSy: csr (n, nc)."""
        n = len(U); K = self.K
        Bu, Bw = basis(U, K, *UD), basis(W, K, *WD)
        z = lambda: sp.csr_matrix((n, self.nc))
        one = sp.csr_matrix(np.ones((n, 1)))
        if self.variant == 'indep':
            T = tensor(Bu, Bw)
            FAx, FAy = self._blk('ax', T, n), self._blk('ay', T, n)
        else:
            dBu, dBw = basis(U, K, *UD, nu=1), basis(W, K, *WD, nu=1)
            FAx = self._blk('phi', tensor(dBu, Bw), n) + self._blk('cx', one, n)
            FAy = self._blk('phi', self.kappa * tensor(Bu, dBw), n) + self._blk('cy', one, n)
            if self.variant == 'phiplus':
                ke = self.k_extra
                T2 = tensor(basis(U, ke, *UD), basis(W, ke, *WD))
                FAx = FAx + self._blk('ax2', T2, n); FAy = FAy + self._blk('ay2', T2, n)
        if self.KS == 0:
            FSx, FSy = self._blk('sx', one, n), self._blk('sy', one, n)
        else:
            TS = tensor(basis(U, self.KS, *UD), basis(W, self.KS, *WD))
            FSx, FSy = self._blk('sx', TS, n), self._blk('sy', TS, n)
        return FAx.tocsr(), FAy.tocsr(), FSx.tocsr(), FSy.tocsr()


def fit_map(spec, D, lam, nclip=2, sig=3.5, ridge=1e-9):
    """Linear LSQ of w = a(U,W) + tr s(U,W) for both views at once, nclip rounds of 3.5 robust-sigma clipping."""
    FAx, FAy, FSx, FSy = spec.fields(D.U, D.W)
    Xx = FAx + sp.diags(D.trx) @ FSx; Xy = FAy + sp.diags(D.trY) @ FSy
    X = sp.vstack([Xx, Xy]).tocsr(); y = np.concatenate([D.wx, D.wy]); nx = D.n
    P = spec.penalty(lam)
    keep = np.ones(len(y), bool); c = None
    for it in range(nclip + 1):
        Xk = X[keep]
        M = (Xk.T @ Xk).toarray() + P.toarray(); b = Xk.T @ y[keep]
        M[np.diag_indices_from(M)] += ridge * np.mean(np.diag(M)) + 1e-12
        try:
            c = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(M, b, rcond=None)[0]
        r = y - X @ c
        if it < nclip:
            keep = np.ones(len(y), bool)
            for sl in (slice(0, nx), slice(nx, None)):
                rs = 1.4826 * np.median(np.abs(r[sl] - np.median(r[sl])))
                keep[sl] = np.abs(r[sl] - np.median(r[sl])) < sig * rs
    return dict(spec=spec, c=c, lam=lam, nkeep=int(keep.sum()))


def eval_map(m, U, W):
    FAx, FAy, FSx, FSy = m['spec'].fields(U, W)
    c = m['c']
    return FAx @ c, FAy @ c, FSx @ c, FSy @ c


def apply_arm(a, s, w, tr):
    dw = a - np.mean(a); vr = s / np.mean(s)
    c = TT.robust_const(tr, w, dw, vr)
    return (w - c[0] - dw) / (c[1] * vr)


def apply_map(m, T):
    """-> dict view -> predicted tan on test set T"""
    ax, ay, sx, sy = eval_map(m, T.U, T.W)
    return {'x': apply_arm(ax, sx, T.wx, T.trx), 'y': apply_arm(ay, sy, T.wy, T.trY)}


def dth(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))


def scores(pred, T):
    return {'x': s68(dth(pred['x'], T.trx)), 'y': s68(dth(pred['y'], T.trY))}


def cv_score(spec, D, lam, nfold=5, mode='time'):
    """inner CV inside D: held-out-fold angle residuals pooled; constants refit per fold.  -> (s68x, s68y)"""
    if mode == 'time':
        f = np.minimum((D.rank.argsort().argsort() * nfold) // D.n, nfold - 1)
    else:
        f = np.arange(D.n) % nfold
    dx, dy = [], []
    for k in range(nfold):
        tr_, te = D.sub(f != k), D.sub(f == k)
        m = fit_map(spec, tr_, lam)
        p = apply_map(m, te)
        dx.append(dth(p['x'], te.trx)); dy.append(dth(p['y'], te.trY))
    return s68(np.concatenate(dx)), s68(np.concatenate(dy))


def pilot_kappa(D):
    cx = TT.robust_const(D.trx, D.wx, 0.0, 1.0); cy = TT.robust_const(D.trY, D.wy, 0.0, 1.0)
    return float(cy[1] / cx[1])


# ---------------------------------------------------------------- baselines on a test set
def baseline_arms(T, Dtr, grids=(4, 6, 8, 10)):
    """pass1, const, pass1c (pass-1 angle with two constants refit), cells nxn (w0+v) -> {arm: {'x': tan, 'y': tan}}"""
    arms = {'pass1': {'x': T.tpx, 'y': T.tpy}}
    arms['const'] = {'x': apply_arm(np.zeros(T.n), np.ones(T.n), T.wx, T.trx), 'y': apply_arm(np.zeros(T.n), np.ones(T.n), T.wy, T.trY)}
    pc = {}
    for p, tp, tr in (('x', T.tpx, T.trx), ('y', T.tpy, T.trY)):
        c = TT.robust_const(tr, tp, 0.0, 1.0); pc[p] = (tp - c[0]) / c[1]
    arms['pass1c'] = pc
    for n in grids:
        out = {}
        for p, w, tr, wt, trt in (('x', T.wx, T.trx, Dtr.wx, Dtr.trx), ('y', T.wy, T.trY, Dtr.wy, Dtr.trY)):
            C = TG.cells(Dtr.U, Dtr.W, wt, trt, n)
            vr = TT.bilinear(C, 'vrel', T.U, T.W); dw = TT.bilinear(C, 'dw0', T.U, T.W)
            c = TT.robust_const(tr, w, dw, vr); out[p] = (w - c[0] - dw) / (c[1] * vr)
        arms[f'cells{n}'] = out
    return arms


def paired_boot(darr, ref, nb=1000, seed=1):
    """darr: {arm: angle residual array (deg)}; returns {arm: (s68, err)}, and diffs vs ref arm {arm: (diff, err)}"""
    rng = np.random.default_rng(seed)
    names = list(darr); n = len(darr[ref]); M = np.vstack([darr[k] for k in names])
    S = np.empty((nb, len(names)))
    for b in range(nb):
        i = rng.integers(0, n, n)
        q = np.percentile(M[:, i], [16, 84], axis=1)
        S[b] = (q[1] - q[0]) / 2
    j = names.index(ref)
    return ({k: (s68(darr[k]), float(S[:, i].std())) for i, k in enumerate(names)},
            {k: (s68(darr[k]) - s68(darr[ref]), float((S[:, i] - S[:, j]).std())) for i, k in enumerate(names)})
