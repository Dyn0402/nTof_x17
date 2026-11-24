"""lib.py -- production fitter of the per-mounting angle map ``bow['phi']`` (bench re-finalization, step 3).

The map is fitted DIRECTLY in the production form of ``wft.reco.bow_pairs`` (``wft.bow_phi``):

    tan_p = (w_p - w0_r,p) / (c1_r,p (1 + S_p(U, W))) - A_p(U, W),      c1 = kw v_drift of the row,

with A_x = dPhi/dU + cx, A_y = dPhi/dW + cy (one shared potential Phi, kappa = 1, tan units), S_x and S_y two scale maps,
and (U, W) = ((p0 + BOW_MID_MM z - 200)/100) the mid-gap position of the constant mapping z = (w - w0)/c1 of the row.
Per row and view (w0, c1) are refit on the row's own events with the map applied (robust line of w against
g = (1 + S)(tan_ref + A)); they are refit inside every cross-validation fold from that fold's training events only.

Fit: Gauss-Newton on the map coefficients at fixed constants (the product (1 + S)(tan_ref + A) is linearised, three to six
rounds, converged to 1e-5 in tan), then the means of A and 1 + S are moved into the row constants, then the constants are
refit; repeated.  Penalty: lam x second differences of the spline coefficients (Phi, S_x, S_y), as a2/r3.

Model selection (K, K_S, lam) by 4-block inner CV on the training part only, in the linear a2 form z = a + tan_ref s
(row constants from the inner training part; the S x A cross term of the production form is second order and equal for
all candidates).  The grids are K in {6, 9, 12, 16, 20}, K_S in {0, 4, 6, 8, 10}, lam in 10^{-4..3}; K^2 + 2 K_S^2 <= N/15.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
from wft import reco, bow_phi  # noqa: E402

MID = reco.BOW_MID_MM
UW_MAX = reco.BOW_UW_MAX
KGRID = (6, 9, 12, 16, 20, 24)
KSGRID = (0, 4, 6, 8, 10, 12, 14, 16)
LAMS = tuple(10.0 ** e for e in range(-6, 4))
NPEN = 15.0                  # K^2 + 2 K_S^2 <= N / NPEN
NIN = 4                      # inner CV blocks
CLIP = 3.5


# ------------------------------------------------------------------------------------------------ splines
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
    P = pen1(K)
    I = np.eye(K)
    return np.kron(P, I) + np.kron(I, P)


class Layout:
    """Coefficient vector c = [phi (K^2), cx, cy, sx (KS^2 or 1), sy (KS^2 or 1)]."""

    def __init__(self, K, KS, dom):
        self.K, self.KS, self.dom = K, KS, dom
        ns = KS * KS if KS else 1
        self.i_phi = (0, K * K)
        self.i_c = (K * K, K * K + 2)
        self.i_sx = (K * K + 2, K * K + 2 + ns)
        self.i_sy = (K * K + 2 + ns, K * K + 2 + 2 * ns)
        self.nc = K * K + 2 + 2 * ns

    def penalty(self, lam):
        P = np.zeros((self.nc, self.nc))
        a, b = self.i_phi
        P[a:b, a:b] = lam * pen2(self.K)
        if self.KS:
            for a, b in (self.i_sx, self.i_sy):
                P[a:b, a:b] = lam * pen2(self.KS)
        return P

    def fields(self, U, W):
        """FAx, FAy, FSx, FSy: sparse (n, nc); A = FA c, S (or s = 1 + S in the linear form) = FS c."""
        n = len(U)
        (ulo, uhi), (wlo, whi) = self.dom
        K, KS = self.K, self.KS
        Bu, Bw = basis(U, K, ulo, uhi), basis(W, K, wlo, whi)
        dBu, dBw = basis(U, K, ulo, uhi, 1), basis(W, K, wlo, whi, 1)
        Z = lambda m: sp.csr_matrix((n, m))
        one = sp.csr_matrix(np.ones((n, 1)))
        zero = Z(1)
        ns = KS * KS if KS else 1
        TS = tensor(basis(U, KS, ulo, uhi), basis(W, KS, wlo, whi)) if KS else one
        FAx = sp.hstack([tensor(dBu, Bw), one, zero, Z(ns), Z(ns)], format='csr')
        FAy = sp.hstack([tensor(Bu, dBw), zero, one, Z(ns), Z(ns)], format='csr')
        FSx = sp.hstack([Z(K * K), zero, zero, TS, Z(ns)], format='csr')
        FSy = sp.hstack([Z(K * K), zero, zero, Z(ns), TS], format='csr')
        return FAx, FAy, FSx, FSy


# ------------------------------------------------------------------------------------------------ constants
def robust_line(g, w, nit=3, clip=CLIP):
    """(w0, c1) of w = w0 + c1 g with MAD clipping (as track_test.robust_const)."""
    keep = np.ones(len(g), bool)
    c = np.array([0.0, 1.0])
    for _ in range(nit):
        A = np.vstack([np.ones(keep.sum()), g[keep]]).T
        c = np.linalg.lstsq(A, w[keep], rcond=None)[0]
        r = w - c[0] - c[1] * g
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < clip * rs
    return c


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def dth(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))


# ------------------------------------------------------------------------------------------------ events
class Ev:
    """Selected gate events of a mounting (arrays; ``row`` indexes the mounting's row list).  fx, fy: the event enters the
    fit/evaluation of that view (not railed, and for D X x >= 200 mm).  blk: time block 0..4 inside its row."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.n = len(self.row)

    def sub(self, m):
        m = np.asarray(m)
        kw = {k: v[m] for k, v in self.__dict__.items() if isinstance(v, np.ndarray) and len(v) == self.n}
        kw.update({k: v for k, v in self.__dict__.items() if not (isinstance(v, np.ndarray) and len(v) == self.n) and k != 'n'})
        return Ev(**kw)


def uw(ev, w0, c1):
    """(U, W, zx, zy) of the events for per-row constants w0, c1 (arrays [2, nrow], x then y), as bow_pairs."""
    zx = (ev.wx - w0[0][ev.row]) / c1[0][ev.row]
    zy = (ev.wy - w0[1][ev.row]) / c1[1][ev.row]
    U = ((ev.p0x + MID * zx) - 200.0) / 100.0
    W = ((ev.p0y + MID * zy) - 200.0) / 100.0
    U = np.where(np.isfinite(U) & (np.abs(U) <= UW_MAX), U, 0.0)
    W = np.where(np.isfinite(W) & (np.abs(W) <= UW_MAX), W, 0.0)
    return U, W, zx, zy


def ladder_constants(ev, nrow, rows_use=None):
    """Per row and view robust ladder constants (no map): w = w0 + c1 tan_ref, on events with fx / fy."""
    w0 = np.zeros((2, nrow))
    c1 = np.ones((2, nrow))
    for r in range(nrow):
        for v, (w, tr, f) in enumerate(((ev.wx, ev.trx, ev.fx), (ev.wy, ev.try_, ev.fy))):
            m = (ev.row == r) & f
            if m.sum() >= 30:
                c = robust_line(tr[m], w[m])
                w0[v, r], c1[v, r] = c
    return w0, c1


def fit_row_constants(ev, nrow, blk_out, A, S, fallback):
    """Per row and view, the robust line of w against g = (1 + S)(tan_ref + A) on that row's events (fx / fy, block != blk_out)
    -> (w0, c1) [2, nrow] and n events [2, nrow].  Rows with < 30 events keep ``fallback`` (w0, c1) and n = 0."""
    w0 = fallback[0].copy()
    c1 = fallback[1].copy()
    nn = np.zeros((2, nrow), int)
    for r in range(nrow):
        for v, (w, tr, f, a, s) in enumerate(((ev.wx, ev.trx, ev.fx, A[0], S[0]), (ev.wy, ev.try_, ev.fy, A[1], S[1]))):
            m = (ev.row == r) & f & (ev.blk != blk_out)
            nn[v, r] = int(m.sum())
            if m.sum() >= 30:
                c = robust_line((1 + s[m]) * (tr[m] + a[m]), w[m])
                w0[v, r], c1[v, r] = c
    return w0, c1, nn


# ------------------------------------------------------------------------------------------------ production-form fit
def design_stack(lay, U, W, A0, S0, trx, try_, fx, fy):
    """GN design for both views stacked (rows with fx / fy only): returns X (sparse), and the per-view row masks."""
    FAx, FAy, FSx, FSy = lay.fields(U, W)
    Xx = sp.diags(1.0 + S0[0]) @ FAx + sp.diags(trx + A0[0]) @ FSx
    Xy = sp.diags(1.0 + S0[1]) @ FAy + sp.diags(try_ + A0[1]) @ FSy
    return Xx, Xy


def eval_AS(lay, c, U, W):
    FAx, FAy, FSx, FSy = lay.fields(U, W)
    return (FAx @ c, FAy @ c), (FSx @ c, FSy @ c)


def renormalise(lay, c, ev, U, W):
    """Move the training means of A_p and 1 + S_p out of the map (into the row constants): A -= mean, 1 + S -> (1 + S)/mean."""
    (Ax, Ay), (Sx, Sy) = eval_AS(lay, c, U, W)
    c = c.copy()
    c[lay.i_c[0]] -= Ax[ev.fx].mean()
    c[lay.i_c[0] + 1] -= Ay[ev.fy].mean()
    for (a, b), S, f in ((lay.i_sx, Sx, ev.fx), (lay.i_sy, Sy, ev.fy)):
        m = (1.0 + S[f]).mean()
        c[a:b] = (1.0 + c[a:b]) / m - 1.0
    return c


def fit_gn(ev, nrow, lay, lam, w0, c1, n_it=6, tol=1e-5, blk_out=-1, verbose=False):
    """Production-form fit on the events of ``ev`` (those of block ``blk_out`` are left out; use -1 for none).  Starts from the
    row constants (w0, c1) [2, nrow] (ladder constants).  Returns dict(c, w0, c1, U, W, n_it, nkeep)."""
    tr_ = ev.blk != blk_out
    e = ev.sub(tr_)
    c = np.zeros(lay.nc)
    # start: constant S = 0 -> (1 + S) = 1; the cx, cy, sx, sy constants are zero
    P = lay.penalty(lam)
    keep = None
    prev = None
    w0, c1 = w0.copy(), c1.copy()
    for it in range(n_it):
        U, W, zx, zy = uw(e, w0, c1)
        (Ax, Ay), (Sx, Sy) = eval_AS(lay, c, U, W)
        Xx, Xy = design_stack(lay, U, W, (Ax, Ay), (Sx, Sy), e.trx, e.try_, e.fx, e.fy)
        tx = zx - e.trx + Ax * Sx
        ty = zy - e.try_ + Ay * Sy
        if it == 1:
            # clip from the residuals of the map after the first round, per view; the mask is then kept fixed
            kx, ky = e.fx.copy(), e.fy.copy()
            for r_, k_ in ((zx - (1 + Sx) * (e.trx + Ax), kx), (zy - (1 + Sy) * (e.try_ + Ay), ky)):
                md = np.median(r_[k_])
                rs = 1.4826 * np.median(np.abs(r_[k_] - md))
                k_ &= np.abs(r_ - md) < CLIP * rs
            keep = (kx, ky)
        elif keep is None:
            keep = (e.fx, e.fy)
        kx, ky = keep
        X = sp.vstack([Xx[kx], Xy[ky]]).tocsr()
        y = np.concatenate([tx[kx], ty[ky]])
        M = (X.T @ X).toarray() + P
        M[np.diag_indices_from(M)] += 1e-9 * np.mean(np.diag(M)) + 1e-12
        c = np.linalg.solve(M, X.T @ y)
        c = renormalise(lay, c, e, U, W)
        (Ax, Ay), (Sx, Sy) = eval_AS(lay, c, U, W)
        w0n, c1n, _ = fit_row_constants(e, nrow, -1, (Ax, Ay), (Sx, Sy), (w0, c1))
        # change of the predicted tan on the fit events
        tx_new = (e.wx - w0n[0][e.row]) / (c1n[0][e.row] * (1 + Sx)) - Ax
        d = np.abs(tx_new - prev).max() if prev is not None else np.inf
        prev = tx_new
        w0, c1 = w0n, c1n
        if verbose:
            print('  GN', it, 'max dtan', d, 'nkeep', kx.sum(), ky.sum())
        if d < tol and it >= 2:
            break
    return dict(c=c, w0=w0, c1=c1, n_it=it + 1, lay=lay)


def block_from_fit(fit, name_prov=None, tilt_only=False):
    """The ``bow['phi']`` block (wft/bow_phi.py) of a fit.  tilt_only: S = 0 (KS = 0, coefficient 0)."""
    lay, c = fit['lay'], fit['c']
    K, KS = lay.K, lay.KS
    (ulo, uhi), (wlo, whi) = lay.dom
    a, b = lay.i_phi
    blk = dict(version=bow_phi.PHI_VERSION, units='tan', degree=3, K=K, KS=(0 if tilt_only else KS),
               U_domain=[ulo, uhi], W_domain=[wlo, whi],
               knots_U=knots(K, ulo, uhi).tolist(), knots_W=knots(K, wlo, whi).tolist(),
               knots_S_U=(knots(KS, ulo, uhi).tolist() if (KS and not tilt_only) else None),
               knots_S_W=(knots(KS, wlo, whi).tolist() if (KS and not tilt_only) else None),
               kappa=1.0, phi=c[a:b].tolist(), cx=float(c[lay.i_c[0]]), cy=float(c[lay.i_c[0] + 1]),
               sx=([0.0] if tilt_only else (c[lay.i_sx[0]:lay.i_sx[1]]).tolist()),
               sy=([0.0] if tilt_only else (c[lay.i_sy[0]:lay.i_sy[1]]).tolist()),
               provenance=name_prov or {})
    bow_phi.validate(blk)
    return blk


# ------------------------------------------------------------------------------------------------ prediction (= bow_pairs)
def make_cal(bundle, w0, kw, bow_extra=None, phi=None):
    """A calibration-like object for ``wft.reco.bow_pairs``: the row's bundle with a ``phi`` block, constants (w0, kw)."""
    bow = dict(bundle.get('bow') or {}) if bow_extra is None else dict(bow_extra)
    if phi is not None:
        bow['phi'] = phi
    return SimpleNamespace(bow=bow, v_map={}, w0={'x': float(w0[0]), 'y': float(w0[1])}, kw={'x': float(kw[0]), 'y': float(kw[1])},
                           v_drift=float(bundle['v_drift']), angle_map={})


def predict(phi, v_drift, w0, c1, wx, p0x, wy, p0y, bow=None):
    """tan_x, tan_y of pairs through the production conversion (bow_pairs) for constants w0 [x, y] (um/ns), c1 = kw v (um/ns)."""
    cal = SimpleNamespace(bow=dict(bow or {}, phi=phi) if phi is not None else dict(bow or {}), v_map={},
                          w0={'x': float(w0[0]), 'y': float(w0[1])}, kw={'x': float(c1[0]) / v_drift, 'y': float(c1[1]) / v_drift},
                          v_drift=float(v_drift), angle_map={})
    r = reco.bow_pairs(cal, wx, p0x, wy, p0y)
    return r['tan_x'], r['tan_y']


# ------------------------------------------------------------------------------------------------ model selection
def inner_blocks(ev, nin=NIN):
    """4 time blocks inside each row (events are in time order inside a row)."""
    b = np.zeros(ev.n, int)
    for r in np.unique(ev.row):
        i = np.where(ev.row == r)[0]
        b[i] = np.minimum((np.arange(len(i)) * nin) // len(i), nin - 1)
    return b


def select_model(ev, nrow, kgrid=KGRID, ksgrid=KSGRID, lams=LAMS, tilt_only=False, verbose=False):
    """(K, KS, lam, score, info) by inner CV on ``ev`` (the training part).  Linear a2 form; row constants from the inner
    training part."""
    ib = inner_blocks(ev)
    nfit = int(ev.fx.sum() + ev.fy.sum()) // 2
    ks = (0,) if tilt_only else ksgrid
    cands = [(K, KS) for K in kgrid for KS in ks if K * K + 2 * KS * KS <= nfit / NPEN]
    if not cands:
        cands = [(5, 0)]
    dom = ev.dom
    prep = []
    for j in range(NIN):
        tr_, te = ev.sub(ib != j), ev.sub(ib == j)
        w0, c1 = ladder_constants(tr_, nrow)
        Ut, Wt, zxt, zyt = uw(tr_, w0, c1)
        Ue, We, zxe, zye = uw(te, w0, c1)
        # pilot clip: residual of z against tan_ref (ladder) -> drop the far tails from the fit
        keepx = tr_.fx & (np.abs(zxt - tr_.trx - np.median((zxt - tr_.trx)[tr_.fx])) < 6 * 1.4826 * np.median(np.abs((zxt - tr_.trx)[tr_.fx] - np.median((zxt - tr_.trx)[tr_.fx]))))
        keepy = tr_.fy & (np.abs(zyt - tr_.try_ - np.median((zyt - tr_.try_)[tr_.fy])) < 6 * 1.4826 * np.median(np.abs((zyt - tr_.try_)[tr_.fy] - np.median((zyt - tr_.try_)[tr_.fy]))))
        prep.append((tr_, te, Ut, Wt, zxt, zyt, Ue, We, zxe, zye, keepx, keepy))
    score = {}
    for (K, KS) in cands:
        lay = Layout(K, KS, dom)
        mats = []
        for (tr_, te, Ut, Wt, zxt, zyt, Ue, We, zxe, zye, kx, ky) in prep:
            FAx, FAy, FSx, FSy = lay.fields(Ut, Wt)
            Xx = (FAx + sp.diags(tr_.trx) @ FSx)[kx]
            Xy = (FAy + sp.diags(tr_.try_) @ FSy)[ky]
            X = sp.vstack([Xx, Xy]).tocsr()
            y = np.concatenate([zxt[kx], zyt[ky]])
            G = (X.T @ X).toarray()
            b = X.T @ y
            EAx, EAy, ESx, ESy = lay.fields(Ue, We)
            mats.append((G, b, EAx, EAy, ESx, ESy))
        for lam in lams:
            P = lay.penalty(lam)
            ex, ey = [], []
            for (G, b, EAx, EAy, ESx, ESy), (tr_, te, *_r) in zip(mats, prep):
                M = G + P
                M[np.diag_indices_from(M)] += 1e-9 * np.mean(np.diag(M)) + 1e-12
                c = np.linalg.solve(M, b)
                zxe, zye = _r[6], _r[7]
                # tan = (z - a) / s, with a = FA c, s = FS c; remove the training-mean offset/scale (constants absorb them)
                ax, ay, sx_, sy_ = EAx @ c, EAy @ c, ESx @ c, ESy @ c
                tx = (zxe - ax) / sx_
                ty = (zye - ay) / sy_
                ex.append(dth(tx, te.trx)[te.fx])
                ey.append(dth(ty, te.try_)[te.fy])
            sc = s68(np.concatenate(ex)) + s68(np.concatenate(ey))
            score[(K, KS, lam)] = sc
        if verbose:
            best = min((v, k) for k, v in score.items() if k[:2] == (K, KS))
            print('   K', K, 'KS', KS, 'best lam', best[1][2], 'score', round(best[0], 4))
    (K, KS, lam), sc = min(score.items(), key=lambda kv: kv[1])
    info = dict(n_cand=len(cands), kmax_allowed=max(c[0] for c in cands), ksmax_allowed=max(c[1] for c in cands),
                lam_edge=('low' if lam == lams[0] else 'high' if lam == lams[-1] else ''),
                k_edge=bool(K == max(c[0] for c in cands) and K == kgrid[-1]),
                ks_edge=bool(KS == max(c[1] for c in cands) and KS == ksgrid[-1] and not tilt_only),
                k_limited=bool(K == max(c[0] for c in cands) and K < kgrid[-1]))
    return K, KS, lam, sc, info
