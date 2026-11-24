#!/usr/bin/env python3
"""r3lib.py -- penalised tensor B-spline angle maps (tilt + scale) fitted to cosmic lines, with smoothing chosen by
cross-validation over runs, and optional pointing-cell constraints (joint fit).  Test W2a_map_r3.

Rows: t6lib.rows_ac / rows_d (one row per line and view; r = global slope - line slope j; r_loc = g r, j_loc = g j).
Model, local tan convention:    r_loc = a_view(x, y) + s_view(x, y) * j_loc
  tilt 'pot'  : a_x = d phi/dx, a_y = d phi/dy, phi a cubic tensor B-spline (knot spacing ht)
  tilt 'free' : a_x, a_y two independent cubic tensor B-splines
  scale 'common' : one s(x, y) for both views;  'sep' : s_x, s_y (knot spacing hs)
Penalty: discrete thin-plate energy of every field on a 10 mm grid over [-200, 200]^2
  sum (f_xx^2 + 2 f_xy^2 + f_yy^2) -- linear fields are free, so the edges extrapolate smoothly and the means are not
  shrunk.  Two smoothing parameters, lam_t (tilt) and lam_s (scale), chosen by grouped K-fold CV over runs.
Robustness: iterated 3 sigma clipping per view, as t6lib.Poly.
Pointing rows (joint fit): per valid cell, c_loc = kappa (a_x(cell) + s_x(cell) e_loc) + alpha + gamma e_loc, weight
  sigma_line^2 / (err^2 + floor^2); alpha, gamma linear nuisances, kappa scanned (grid) per fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'T6_surface_map'))
from apply_r3 import local_basis  # noqa: E402
from t6lib import mad  # noqa: E402

LO, HI = -200.0, 200.0
GRID = np.arange(-200.0, 200.01, 10.0)


def nbasis(h):
    return int(round((HI - LO) / h)) + 3


def tensor_rows(x, y, h, dx=0, dy=0, w=None):
    """sparse (n, nb^2) matrix of B_i^(dx)(x) B_j^(dy)(y) [* w], column index i * nb + j."""
    nb = nbasis(h)
    ix, Bx = local_basis(x, LO, HI, h, dx)
    iy, By = local_basis(y, LO, HI, h, dy)
    n = len(ix)
    rows = np.repeat(np.arange(n), 16)
    cols = ((ix[:, None, None] + np.arange(4)[None, :, None]) * nb
            + (iy[:, None, None] + np.arange(4)[None, None, :])).reshape(n, 16).ravel()
    val = (Bx[:, :, None] * By[:, None, :]).reshape(n, 16)
    if w is not None:
        val = val * np.asarray(w, float)[:, None]
    return sp.csr_matrix((val.ravel(), (rows, cols)), shape=(n, nb * nb))


class Spline:
    def __init__(self, tilt='pot', scale='sep', ht=20.0, hs=40.0, lam_t=None, lam_s=None,
                 lam_grid_t=None, lam_grid_s=None, kfold=5, clip=3.0, n_iter=4, point=None):
        self.tilt, self.scale, self.ht, self.hs = tilt, scale, ht, hs
        self.lam_t, self.lam_s = lam_t, lam_s
        self.lam_grid_t = lam_grid_t if lam_grid_t is not None else 10.0 ** np.arange(-1, 5.1, 1.0)
        self.lam_grid_s = lam_grid_s if lam_grid_s is not None else 10.0 ** np.arange(-1, 5.1, 1.0)
        self.kfold, self.clip, self.n_iter = kfold, clip, n_iter
        self.point = point          # dict(x, y, e_loc, c_loc, err, floor, kappas) or None
        self.nt, self.ns = nbasis(ht), nbasis(hs)
        self.ntc = self.nt ** 2 * (1 if tilt == 'pot' else 2)
        self.nsc = self.ns ** 2 * (1 if scale == 'common' else 2)
        self.npar = self.ntc + self.nsc
        self._pen = None

    # ---------------------------------------------------------------- design
    def design(self, x, y, view, jl):
        x = np.nan_to_num(np.asarray(x, float)); y = np.nan_to_num(np.asarray(y, float))
        view = np.asarray(view); jl = np.asarray(jl, float)
        v0, v1 = (view == 0).astype(float), (view == 1).astype(float)
        if self.tilt == 'pot':
            T = tensor_rows(x, y, self.ht, dx=1, w=v0) + tensor_rows(x, y, self.ht, dy=1, w=v1)
        else:
            T = sp.hstack([tensor_rows(x, y, self.ht, w=v0), tensor_rows(x, y, self.ht, w=v1)])
        if self.scale == 'common':
            S = tensor_rows(x, y, self.hs, w=jl)
        else:
            S = sp.hstack([tensor_rows(x, y, self.hs, w=jl * v0), tensor_rows(x, y, self.hs, w=jl * v1)])
        return sp.hstack([T, S]).tocsr()

    def field_mats(self, x, y):
        """dict of sparse matrices mapping the coefficient vector to a_x, a_y, s_x, s_y at points (x, y)."""
        n = len(x)
        Z = lambda k: sp.csr_matrix((n, k))  # noqa: E731
        nt2, ns2 = self.nt ** 2, self.ns ** 2
        if self.tilt == 'pot':
            ax = tensor_rows(x, y, self.ht, dx=1); ay = tensor_rows(x, y, self.ht, dy=1)
        else:
            B = tensor_rows(x, y, self.ht)
            ax = sp.hstack([B, Z(nt2)]); ay = sp.hstack([Z(nt2), B])
        Bs = tensor_rows(x, y, self.hs)
        if self.scale == 'common':
            sx = sy = Bs
        else:
            sx = sp.hstack([Bs, Z(ns2)]); sy = sp.hstack([Z(ns2), Bs])
        return dict(a_x=sp.hstack([ax, Z(self.nsc)]).tocsr(), a_y=sp.hstack([ay, Z(self.nsc)]).tocsr(),
                    s_x=sp.hstack([Z(self.ntc), sx]).tocsr(), s_y=sp.hstack([Z(self.ntc), sy]).tocsr())

    def penalties(self):
        if self._pen is not None:
            return self._pen
        g = GRID
        X, Y = np.meshgrid(g, g, indexing='ij')
        F = self.field_mats(X.ravel(), Y.ravel())
        m = len(g)
        I = sp.identity(m)
        d2 = sp.diags([1, -2, 1], [0, 1, 2], shape=(m - 2, m)) / 100.0
        d1 = sp.diags([-1, 1], [0, 1], shape=(m - 1, m)) / 10.0
        Dxx, Dyy, Dxy = sp.kron(d2, I), sp.kron(I, d2), sp.kron(d1, d1)
        P = {}
        for grp, keys in (('t', ('a_x', 'a_y')), ('s', ('s_x', 's_y') if self.scale == 'sep' else ('s_x',))):
            acc = sp.csr_matrix((self.npar, self.npar))
            for k in keys:
                for Dm, wt in ((Dxx, 1.0), (Dyy, 1.0), (Dxy, 2.0)):
                    E = Dm @ F[k]
                    acc = acc + wt * (E.T @ E)
            P[grp] = (acc * 1e6).toarray()      # mm^-4 -> per (100 mm)^... only the relative scale matters
        self._pen = P
        return P

    # ---------------------------------------------------------------- fit
    def _solve(self, XtX, Xty, lt, ls):
        P = self.penalties()
        A = XtX + lt * P['t'] + ls * P['s'] + 1e-9 * np.eye(self.npar) * max(np.trace(XtX) / self.npar, 1.0)
        try:
            return np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, Xty, rcond=None)[0]

    def _point_block(self, kappa):
        p = self.point
        F = self.field_mats(p['x'], p['y'])
        Xp = kappa * (F['a_x'] + sp.diags(p['e_loc']) @ F['s_x'])
        Xp = sp.hstack([Xp, sp.csr_matrix(np.column_stack([np.ones(len(p['x'])), p['e_loc']]))]).tocsr()
        w = p['sig_line'] ** 2 / (p['err'] ** 2 + p['floor'] ** 2)
        return Xp, np.asarray(p['c_loc'], float), w

    def _fit_fixed(self, X, y, vw, lt, ls):
        """robust penalised fit at fixed lambdas; X rows of lines.  Returns beta (lines part), keep mask."""
        keep = np.ones(len(y), bool)
        nuis = 2 if self.point is not None else 0
        for it in range(self.n_iter):
            Xk = X[keep]
            XtX = (Xk.T @ Xk).toarray()
            Xty = Xk.T @ y[keep]
            if self.point is not None:
                best = None
                for kap in self.point['kappas']:
                    Xp, cp, wp = self._point_block(kap)
                    XtXf = np.zeros((self.npar + nuis,) * 2); XtXf[:self.npar, :self.npar] = XtX
                    XtXf += (Xp.T @ sp.diags(wp) @ Xp).toarray()
                    Xtyf = np.concatenate([Xty, np.zeros(nuis)]) + Xp.T @ (wp * cp)
                    P = self.penalties()
                    A = XtXf.copy()
                    A[:self.npar, :self.npar] += lt * P['t'] + ls * P['s']
                    A += 1e-9 * np.eye(len(A)) * max(np.trace(XtX) / self.npar, 1.0)
                    b = np.linalg.solve(A, Xtyf)
                    rl = y[keep] - Xk @ b[:self.npar]
                    rp = cp - Xp @ b
                    obj = (rl @ rl + wp @ rp ** 2 + lt * b[:self.npar] @ P['t'] @ b[:self.npar]
                           + ls * b[:self.npar] @ P['s'] @ b[:self.npar])
                    if best is None or obj < best[0]:
                        best = (obj, b, kap)
                _, bf, self.kappa = best
                beta = bf[:self.npar]
                self.nuis = bf[self.npar:]
            else:
                beta = self._solve(XtX, Xty, lt, ls)
            res = y - X @ beta
            nk = keep.copy()
            for w in (0, 1):
                m = vw == w
                s = 1.4826 * np.median(np.abs(res[m & keep]))
                nk[m] = np.abs(res[m]) < self.clip * s
            keep = nk
        return beta, keep

    def _rows(self, D):
        X = self.design(D.x.to_numpy(), D.y.to_numpy(), D.view.to_numpy(), (D.j * D.g).to_numpy())
        return X, (D.r * D.g).to_numpy(), D.view.to_numpy()

    def cv(self, D, grid=None):
        """grouped K-fold over runs on clean rows; criterion = sum over views of the pooled held-out MAD."""
        runs = D.run.to_numpy()
        sizes = pd.Series(runs).value_counts()
        fold_of, load = {}, np.zeros(self.kfold)
        for r, n in sizes.items():
            k = int(np.argmin(load)); fold_of[r] = k; load[k] += n
        fold = np.array([fold_of[r] for r in runs])
        X, y, vw = self._rows(D)
        jl = (D.j * D.g).to_numpy()
        grid = grid if grid is not None else [(a, b) for a in self.lam_grid_t for b in self.lam_grid_s]
        out = []
        pred = {g: np.full(len(y), np.nan) for g in grid}
        for k in range(self.kfold):
            tr, te = fold != k, fold == k
            for g in grid:
                beta, _ = self._fit_fixed(X[tr], y[tr], vw[tr], *g)
                # corrected residual in local convention: (t - a)/(1+s) - j  ==  (r_loc + jl - a)/(1+s) - jl
                a_, s_ = self._a_s_rows(D[te], beta)
                pred[g][te] = (y[te] + jl[te] - a_) / (1 + s_) - jl[te]
        for g in grid:
            e = pred[g]
            sx, sy = mad(e[vw == 0]), mad(e[vw == 1])
            out.append(dict(lam_t=g[0], lam_s=g[1], cv_x=sx, cv_y=sy, cv=sx + sy))
        return pd.DataFrame(out)

    def _a_s_rows(self, D, beta):
        F = self.field_mats(np.nan_to_num(D.x.to_numpy(float)), np.nan_to_num(D.y.to_numpy(float)))
        vw = D.view.to_numpy()
        a = np.where(vw == 0, F['a_x'] @ beta, F['a_y'] @ beta)
        s = np.where(vw == 0, F['s_x'] @ beta, F['s_y'] @ beta)
        return a, s

    def fit(self, D):
        D = D[D.clean] if 'clean' in D else D
        if self.lam_t is None or self.lam_s is None:
            T = self.cv(D)
            b = T.loc[T.cv.idxmin()]
            self.cvtab = T
            self.lam_t_sel, self.lam_s_sel = float(b.lam_t), float(b.lam_s)
        else:
            self.lam_t_sel, self.lam_s_sel = self.lam_t, self.lam_s
        X, y, vw = self._rows(D)
        self.beta, keep = self._fit_fixed(X, y, vw, self.lam_t_sel, self.lam_s_sel)
        self.nkeep = int(keep.sum())
        self.sig = {w: float(np.std((y - X @ self.beta)[keep & (vw == w)])) for w in (0, 1)}
        self.train_xy = (D.x.to_numpy(float), D.y.to_numpy(float))
        return self

    # ---------------------------------------------------------------- use
    def fields(self, x, y):
        F = self.field_mats(np.nan_to_num(np.asarray(x, float)), np.nan_to_num(np.asarray(y, float)))
        return {k: v @ self.beta for k, v in F.items()}

    def corrected_resid(self, D):
        a, s = self._a_s_rows(D, self.beta)
        g = D.g.to_numpy()
        return (D.slope.to_numpy() - g * a) / (1 + s) - D.j.to_numpy()

    def coef_arrays(self):
        b = self.beta
        nt2, ns2 = self.nt ** 2, self.ns ** 2
        t, s = b[:self.ntc], b[self.ntc:]
        sh = lambda v, n: v.reshape(n, n).tolist()  # noqa: E731
        tilt = (dict(kind='potential', phi=sh(t, self.nt)) if self.tilt == 'pot'
                else dict(kind='free', a_x=sh(t[:nt2], self.nt), a_y=sh(t[nt2:], self.nt)))
        scale = (dict(kind='common', s=sh(s, self.ns)) if self.scale == 'common'
                 else dict(kind='per_view', s_x=sh(s[:ns2], self.ns), s_y=sh(s[ns2:], self.ns)))
        return tilt, scale


def support_mask(x, y, step=20.0, nmin=3):
    """grid points (GRID) whose 20 mm cell holds >= nmin training lines."""
    e = np.arange(LO - 5, HI + 5.01, step)
    H = np.histogram2d(x, y, [e, e])[0]
    gi = np.clip(np.digitize(GRID, e) - 1, 0, len(e) - 2)
    return H[np.ix_(gi, gi)] >= nmin
