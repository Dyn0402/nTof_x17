#!/usr/bin/env python3
"""spline_map_lib.py -- the fitting library of the n_TOF spline angle map (r3).  Used by spline_map_fit.py.

Moved into the repo on 2026-11-24 from the prototype of the re-audit (rerun/ntof/improve/reaudit/tests/W2a_map_r3/:
r3lib.py, task1.py, task2.py, build_maps.py, make_candidate.py; the line readers of T6_surface_map/t6lib.py; the
pointing cells of T5_pointing_map/pm.py and hmap.py).  The algorithms are unchanged.  The evaluator of a fitted map is
rerun/ntof/stage3/spline_map.py; this module builds the same fields from sparse design matrices for the fit.

Training rows (``rows_ac``, ``rows_d``): one row per cosmic line and view.  r = (global slope of the chamber's track)
- (slope j of the joined line).  The fit works in the LOCAL tan convention: r_loc = g r and j_loc = g j, with
g = A (+1, +1), C (+1, -1), D (-1, +1) (x view, y view; spline_map.G).  Model:

    r_loc = a_view(x, y) + s_view(x, y) * j_loc,   a_x = d phi/dx, a_y = d phi/dy   (tilt 'pot', scale 'sep')

phi and s_x, s_y are tensor cubic B-splines (knots every ht = 20 mm and hs = 40 mm on [-200, 200] mm).  The penalty is
the discrete thin-plate energy of every field on a 10 mm grid (sum f_xx^2 + 2 f_xy^2 + f_yy^2), so linear fields are
free.  The two smoothing parameters lam_t (tilt) and lam_s (scale) are chosen by 5-fold cross-validation over runs
(folds balanced in rows), on the pooled held-out MAD of the corrected residual.  Robustness: 4 iterations of 3 sigma
clipping per view.

Pointing rows (chamber D): the horizontal capsule pointing of late beam tracks per 25 mm cell.  c = (unmapped global
slope) - (slope e expected from the source to the track's point).  Each valid cell adds one row
c_loc = kappa (a_x(cell) + s_x(cell) e_loc) + alpha + gamma e_loc, weight sigma_line^2 / (err^2 + floor^2), with
alpha, gamma free nuisances and kappa scanned on a grid.  The floor makes the cells scatter with chi2/ndf = 1 about
the lines-only map.  The nuisances are not part of the map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.stage3.spline_map import local_basis, G  # noqa: E402

LO, HI = -200.0, 200.0
GRID = np.arange(-200.0, 200.01, 10.0)
HT, HS = 20.0, 40.0
CLEAN = 0.10                       # line_residuals / angle_map / bd_map "clean" rule
#: smoothing grids searched by the CV (the prototype's; A, C and D differ because D has ~10x fewer lines)
LT_AC, LS_AC = np.array([0.003, 0.01, 0.03, 0.1]), np.array([1e-3, 3e-3, 1e-2, 0.1, 1.0])
LT_D, LS_D = 10.0 ** np.arange(-2, 3.1, 1.0), 10.0 ** np.arange(-3, 2.1, 1.0)
#: kappa grid of the joint (lines + pointing) fit
KAP = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0])
CLAMP_PCT = (0.1, 99.9)
FRAMES = {'A': 'x view -> d_x/d_z, y view -> d_y/d_z (global); local x_local = +x, y_local = +y',
          'C': 'x view -> d_x/d_z, y view -> d_y/d_z (global); local x_local = -x, y_local = +y',
          'D': 'x view -> d_z/d_x, y view -> d_y/d_x (global); local x_local = -z, y_local = +y'}


def lam_grids(arm):
    return (LT_D, LS_D) if arm == 'D' else (LT_AC, LS_AC)


# =============================================================================================== statistics
def mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 10 else np.nan


def widths(v):
    """(w68, w95): half widths of the central 68 % and 95 % intervals."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    q = np.percentile(v, [2.5, 16, 84, 97.5])
    return (q[2] - q[1]) / 2, (q[3] - q[0]) / 2


# =============================================================================================== training rows
def rows_ac(M: pd.DataFrame, arm: str) -> pd.DataFrame:
    """All A-C lines (line_residuals.py lines.parquet) as (line, view) rows for chamber arm; clean as
    angle_map.clean_lines; part = run parity."""
    s, o = arm.lower(), ('c' if arm == 'A' else 'a')
    out = []
    for vi, (p, q) in enumerate((('x', 'y'), ('y', 'x'))):
        j, jq = M[f'j{p}'], M[f'j{q}']
        slope = M[f'd_{p}_{s}'] / M[f'd_z_{s}']
        slope_q = M[f'd_{q}_{s}'] / M[f'd_z_{s}']
        r = slope - j
        clean = (((M[f'd_{p}_{o}'] / M[f'd_z_{o}'] - j).abs() < CLEAN) & ((slope_q - jq).abs() < CLEAN)
                 & (j.abs() < 0.45) & (r.abs() < 0.15))
        gp, gq = G[arm][vi], G[arm][1 - vi]
        out.append(pd.DataFrame(dict(line=np.arange(len(M)), view=vi, x=M[f'x_local_{s}'], y=M[f'y_local_{s}'],
                                     j=j, r=r, slope=slope, jo=jq, slope_o=slope_q, g=gp, go=gq, run=M.run,
                                     part=M.run.str.replace('run_', '').astype(int) % 2, clean=clean,
                                     allok=(j.abs() < 0.45) & np.isfinite(r))))
    return pd.concat(out, ignore_index=True)


def rows_d(M: pd.DataFrame) -> pd.DataFrame:
    """All B-D lines (bd_map.py bd_lines.parquet) as (line, view) rows for chamber D; clean as bd_map.clean;
    part = event parity (bd_map's own protocol; the held-out uses run parity)."""
    out = []
    for vi, (p, q) in enumerate((('z', 'y'), ('y', 'z'))):
        j, jq = M[f'j{p}'], M[f'j{q}']
        r = M[f'res_{p}']
        slope = M[f'd_{p}_d'] / M.d_x_d
        slope_q = M[f'd_{q}_d'] / M.d_x_d
        clean = (M[f'res_{q}'].abs() < CLEAN) & (j.abs() < 0.45) & (r.abs() < 0.15)
        gp, gq = G['D'][vi], G['D'][1 - vi]
        out.append(pd.DataFrame(dict(line=np.arange(len(M)), view=vi, x=M.x_local_d, y=M.y_local_d, j=j, r=r,
                                     slope=slope, jo=jq, slope_o=slope_q, g=gp, go=gq, run=M.run,
                                     part=M.event_id.astype(int) % 2, clean=clean,
                                     allok=(j.abs() < 0.45) & np.isfinite(r))))
    return pd.concat(out, ignore_index=True)


# =============================================================================================== the spline model
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
    """Penalised tensor B-spline tilt + scale map of one chamber (both views).  See the module docstring."""

    def __init__(self, tilt='pot', scale='sep', ht=HT, hs=HS, lam_t=None, lam_s=None,
                 lam_grid_t=None, lam_grid_s=None, kfold=5, clip=3.0, n_iter=4, point=None):
        self.tilt, self.scale, self.ht, self.hs = tilt, scale, ht, hs
        self.lam_t, self.lam_s = lam_t, lam_s
        self.lam_grid_t = lam_grid_t if lam_grid_t is not None else 10.0 ** np.arange(-1, 5.1, 1.0)
        self.lam_grid_s = lam_grid_s if lam_grid_s is not None else 10.0 ** np.arange(-1, 5.1, 1.0)
        self.kfold, self.clip, self.n_iter = kfold, clip, n_iter
        self.point = point          # dict(x, y, e_loc, c_loc, err, floor, sig_line, kappas) or None
        self.nt, self.ns = nbasis(ht), nbasis(hs)
        self.ntc = self.nt ** 2 * (1 if tilt == 'pot' else 2)
        self.nsc = self.ns ** 2 * (1 if scale == 'common' else 2)
        self.npar = self.ntc + self.nsc
        self._pen = None

    # ---------------------------------------------------------------- design
    def design(self, x, y, view, jl):
        x = np.nan_to_num(np.asarray(x, float))
        y = np.nan_to_num(np.asarray(y, float))
        view = np.asarray(view)
        jl = np.asarray(jl, float)
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
            ax = tensor_rows(x, y, self.ht, dx=1)
            ay = tensor_rows(x, y, self.ht, dy=1)
        else:
            B = tensor_rows(x, y, self.ht)
            ax = sp.hstack([B, Z(nt2)])
            ay = sp.hstack([Z(nt2), B])
        Bs = tensor_rows(x, y, self.hs)
        if self.scale == 'common':
            sx = sy = Bs
        else:
            sx = sp.hstack([Bs, Z(ns2)])
            sy = sp.hstack([Z(ns2), Bs])
        return dict(a_x=sp.hstack([ax, Z(self.nsc)]).tocsr(), a_y=sp.hstack([ay, Z(self.nsc)]).tocsr(),
                    s_x=sp.hstack([Z(self.ntc), sx]).tocsr(), s_y=sp.hstack([Z(self.ntc), sy]).tocsr())

    def penalties(self):
        if self._pen is not None:
            return self._pen
        g = GRID
        X, Y = np.meshgrid(g, g, indexing='ij')
        F = self.field_mats(X.ravel(), Y.ravel())
        m = len(g)
        I = sp.identity(m)  # noqa: E741
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
            P[grp] = (acc * 1e6).toarray()      # only the relative scale of lam matters
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
                    XtXf = np.zeros((self.npar + nuis,) * 2)
                    XtXf[:self.npar, :self.npar] = XtX
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
            k = int(np.argmin(load))
            fold_of[r] = k
            load[k] += n
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


# =============================================================================================== comparison models
class NoMap:
    def fit(self, D):
        return self

    def corrected_resid(self, D):
        return D.slope.to_numpy() - D.j.to_numpy()


class Free:
    """The cell algorithm of angle_map.py (free a, s per cell, bilinear), per view (edges, n_min)."""

    def __init__(self, edges=None, n_min=None):
        from rerun.ntof.improve import angle_map as AM
        self.edges = np.asarray(AM.EDGES if edges is None else edges)
        self.n_min = AM.N_MIN if n_min is None else n_min

    def fit(self, D):
        from rerun.ntof.improve import angle_map as AM
        old = AM.EDGES, AM.CENTRES, AM.N_MIN
        AM.EDGES, AM.CENTRES, AM.N_MIN = self.edges, 0.5 * (self.edges[1:] + self.edges[:-1]), self.n_min
        try:
            self.maps = {}
            for w in (0, 1):
                Dw = D[D.view == w]
                self.maps[w] = AM.fit_map(pd.DataFrame(dict(x=Dw.x, y=Dw.y, j=Dw.j, r=Dw.r)))
        finally:
            AM.EDGES, AM.CENTRES, AM.N_MIN = old
        return self

    def corrected_resid(self, D):
        from rerun.ntof.improve import angle_map as AM
        out = np.full(len(D), np.nan)
        for w in (0, 1):
            m = (D.view == w).to_numpy()
            Dw = D[m]
            out[m] = AM.correct(Dw.slope.to_numpy(), Dw.x.to_numpy(), Dw.y.to_numpy(), self.maps[w]) - Dw.j.to_numpy()
        return out


# =============================================================================================== capsule pointing
#: horizontal view of each chamber: (measuring global coordinate, normal coordinate)
FR = {'A': ('x', 'z'), 'C': ('x', 'z'), 'D': ('z', 'x')}
E25 = np.arange(-200.0, 201.0, 25.0)
LATE_NS = 25e6
E_REACH, HOLE_PAD, NMIN_CELL = 0.44, 10.0, 1000
LATE_COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'gated', 'censored_window', 'x_local', 'y_local',
             'd_x', 'd_y', 'd_z', 'p0_x', 'p0_y', 'p0_z', 'q_total', 't_since_flash_ns', 'low_charge', 't0_railed',
             'x_split_child', 'y_split_child', 'wall_D', 'x_t0', 'y_t0', 'x_ftst', 'y_ftst']


def _bool(t, c):
    return t[c].astype('boolean').fillna(False).astype(bool).to_numpy() if c in t else np.zeros(len(t), bool)


def _late_one(f, rows, canon):
    """late (>= 25 ms) A/C/D tracks of one table: gated on the UNMAPPED angles, not censored, one per (sub-run, event,
    arm), not low_charge / t0_railed / split child.  A mapped table is read through its _nomap twins and also keeps
    the mapped direction (d_<k>_map)."""
    import pyarrow.parquet as pq
    names = set(pq.read_schema(f).names)
    mapped = 'gated_nomap' in names
    cols = [c for c in LATE_COLS if c in names]
    if mapped:
        cols += [c + '_nomap' for c in ('gated', 'd_x', 'd_y', 'd_z', 'x_local', 'y_local', 'p0_x', 'p0_y', 'p0_z')
                 if c + '_nomap' in names]
    t = pq.read_table(f, columns=cols, filters=[('arm', 'in', ['A', 'C', 'D']),
                                                ('t_since_flash_ns', '>=', LATE_NS)]).to_pandas()
    if t.empty:
        return None
    if mapped:
        for k in 'xyz':
            t[f'd_{k}_map'] = t[f'd_{k}']
        tw = [c[:-6] for c in t.columns if c.endswith('_nomap')]
        t = t.drop(columns=tw).rename(columns={c + '_nomap': c for c in tw})
    g = _bool(t, 'gated') & ~_bool(t, 'censored_window')
    t = t[g]
    n = t.groupby(['subrun', 'event_id', 'arm']).arm.transform('size')
    t = t[(n == 1).to_numpy()]
    if 't0_railed' not in t:
        t = t.assign(t0_railed=_t0_railed(t, rows, canon))
    t = t[~_bool(t, 'low_charge') & ~_bool(t, 't0_railed') & ~_bool(t, 'x_split_child') & ~_bool(t, 'y_split_child')]
    keep = ['run', 'subrun', 'tag', 'event_id', 'arm', 'x_local', 'y_local', 'd_x', 'd_y', 'd_z', 'p0_x', 'p0_y',
            'p0_z', 'q_total', 't_since_flash_ns', 'wall_D'] + [f'd_{k}_map' for k in 'xyz' if mapped]
    t = t[[c for c in keep if c in t]].copy()
    t['wall_D'] = _bool(t, 'wall_D')
    for c in ('run', 'subrun', 'tag', 'arm'):
        t[c] = t[c].astype(str)
    return t


def _t0_railed(t, rows, canon):
    """quality_join's t0_railed (|t0 - pin| >= T0_RAIL_NS in either view), for a view that does not carry it yet."""
    from rerun.ntof.stage3 import quality_join as QJ
    pins = QJ._pins(rows, canon)
    arm = t['arm'].to_numpy()
    rail = np.zeros(len(t), bool)
    for p in 'xy':
        t0 = t[f'{p}_t0'].to_numpy(float)
        ft = t[f'{p}_ftst'].fillna(-1).astype(int).to_numpy()
        pin = np.array([pins.get((tg, a), {}).get(p, {}).get(int(f), np.nan) if a != 'B' else np.nan
                        for tg, a, f in zip(t['tag'], arm, ft)])
        with np.errstate(invalid='ignore'):
            rail |= np.abs(t0 - pin) >= QJ.T0_RAIL_NS
    return rail


def build_late_tracks(view, jobs=3, out=None):
    """The late-track sample of the pointing cells from every table of a view (unmapped, or mapped via _nomap)."""
    import csv
    from concurrent.futures import ProcessPoolExecutor
    NT = REPO / 'rerun' / 'ntof'
    sys.path[:0] = [str(NT), str(NT / 'stage3')]
    import ntof_ledger as L
    canon = L.load_canon(str(NT / 'ledger_canon.json'))
    reg = [r for r in csv.DictReader(open(NT / 'ntof_registry.csv')) if r['excluded'] != 'True']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    fs = sorted(Path(view).glob('tracks_run_*.parquet'))
    if not fs:
        raise SystemExit(f'no tables in {view}')
    args = []
    for f in fs:
        run, sub = f.stem.split('_')[2], '_'.join(f.stem.split('_')[3:])
        args.append((f, groups.get((run, sub), []), canon))
    parts = []
    with ProcessPoolExecutor(jobs) as ex:
        for i, t in enumerate(ex.map(_late_one, *zip(*args))):
            if t is not None:
                parts.append(t)
            if i % 100 == 0:
                print(f'  late tracks {i}/{len(fs)}: {sum(len(p) for p in parts)}', flush=True)
    d = pd.concat(parts, ignore_index=True)
    if out is not None:
        d.to_parquet(out, index=False)
    return d


def load_late(d: pd.DataFrame) -> pd.DataFrame:
    """The pointing sample: pre-access runs out (A x connector 8 dead), D needs its wall."""
    from sept26_prelim_analysis.pass1_tracks import PRE_ACCESS_RUNS
    d = d[~d.run.isin(PRE_ACCESS_RUNS)]
    d = d[~((d.arm == 'D') & ~d.wall_D.astype(bool))]
    d = d.copy()
    d['rn'] = d.run.str.replace('run_', '').astype(int)
    d['ms'] = d.t_since_flash_ns / 1e6
    return d.reset_index(drop=True)


def arm_arrays(g, arm, mapped=False):
    """Horizontal measured global slope sh (unmapped, or the mapped d_<k>_map when mapped) and the expected capsule
    slope e (from the source (x_s, z_s) to the track's point p0) of one chamber."""
    from sept26_prelim_analysis.pass1_tracks import SOURCE_XZ_MM
    h, n = FR[arm]
    src = {'x': SOURCE_XZ_MM[0], 'z': SOURCE_XZ_MM[1]}
    P = {k: g[f'p0_{k}'].to_numpy(float) for k in 'xyz'}
    suf = '_map' if mapped else ''
    Dn = {k: g[f'd_{k}{suf}'].to_numpy(float) for k in 'xyz'}
    sh, sv = Dn[h] / Dn[n], Dn['y'] / Dn[n]
    L = P[n] - src[n]
    e = (P[h] - src[h]) / L
    return dict(x=g.x_local.to_numpy(float), y=g.y_local.to_numpy(float), sh=sh, sv=sv, e=e, L=L)


def wmedian(r, w=0.06, n_iter=8, m0=None):
    """Centre of the residual core: median within +-w of the current centre, iterated from the median of |r| < 0.3."""
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return np.nan
    m = np.median(r[np.abs(r) < 0.3]) if m0 is None else m0
    for _ in range(n_iter):
        k = np.abs(r - m) < w
        if k.sum() < 20:
            return np.nan
        m2 = np.median(r[k])
        if abs(m2 - m) < 1e-6:
            break
        m = m2
    return float(m)


def centre(r, w=0.06, nsub=8, rng=None):
    """(centre, stat error from nsub random sub-samples, n, core sigma (MAD within the window))."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 200:
        return np.nan, np.nan, n, np.nan
    m = wmedian(r, w)
    rng = rng or np.random.default_rng(1)
    lab = rng.integers(0, nsub, n)
    sub = np.array([wmedian(r[lab == i], w, m0=m) for i in range(nsub)])
    err = float(np.nanstd(sub, ddof=1) / np.sqrt(nsub))
    k = np.abs(r - m) < 0.3
    sig = 1.4826 * np.median(np.abs(r[k] - m))
    return m, err, n, float(sig)


def cell_index(x, y, edges):
    nc = len(edges) - 1
    ix = np.digitize(x, edges) - 1
    iy = np.digitize(y, edges) - 1
    ok = (ix >= 0) & (ix < nc) & (iy >= 0) & (iy < nc)
    return ix, iy, ok


def cell_map(x, y, r, edges, w=0.06, nmin=300):
    """Per cell [ix, iy]: centre, err, n, sigma of the residual r."""
    nc = len(edges) - 1
    out = {k: np.full((nc, nc), np.nan) for k in ('c', 'err', 'sig')}
    out['n'] = np.zeros((nc, nc), int)
    ix, iy, ok = cell_index(x, y, edges)
    key = np.where(ok, ix * nc + iy, -1)
    order = np.argsort(key, kind='stable')
    ks = key[order]
    bounds = np.searchsorted(ks, np.arange(nc * nc + 1))
    rng = np.random.default_rng(3)
    for kk in range(nc * nc):
        sl = order[bounds[kk]:bounds[kk + 1]]
        i, j = divmod(kk, nc)
        out['n'][i, j] = len(sl)
        if len(sl) < nmin:
            continue
        c, e, _, s = centre(r[sl], w, rng=rng)
        out['c'][i, j], out['err'][i, j], out['sig'][i, j] = c, e, s
    return out


def cell_means(x, y, v, edges):
    nc = len(edges) - 1
    ix, iy, ok = cell_index(x, y, edges)
    key = np.where(ok, ix * nc + iy, nc * nc)
    cnt = np.bincount(key, minlength=nc * nc + 1)[:-1]
    s = np.bincount(key, weights=np.nan_to_num(v), minlength=nc * nc + 1)[:-1]
    return (s / np.where(cnt > 0, cnt, np.nan)).reshape(nc, nc)


def holes(x, y):
    """x_local of low-density 5 mm strip bands (< 0.6 x running median), |y| < 100: dead bands."""
    from scipy.ndimage import median_filter
    eb = np.arange(-200, 201, 5.)
    h = np.histogram(x[np.abs(y) < 100], eb)[0]
    ref = median_filter(h, 11, mode='nearest')
    return eb[:-1][h < 0.6 * ref]


def normal_halfwidth(sh, y):
    """Half-width of the near-normal deficit of MEASURED slopes (0.01 bins below 0.85 x the |s| 0.12-0.25 level,
    contiguous around 0), central y."""
    eb = np.arange(-0.3, 0.3001, 0.01)
    h = np.histogram(sh[np.abs(y) < 100], eb)[0].astype(float)
    mid = 0.5 * (eb[1:] + eb[:-1])
    ref = np.interp(mid, mid[(np.abs(mid) > 0.12) & (np.abs(mid) < 0.25)],
                    h[(np.abs(mid) > 0.12) & (np.abs(mid) < 0.25)])
    low = h < 0.85 * ref
    i0 = int(np.argmin(np.abs(mid)))
    lo = hi = i0
    while lo > 0 and low[lo - 1]:
        lo -= 1
    while hi < len(mid) - 1 and low[hi + 1]:
        hi += 1
    if not low[i0]:
        return 0.0
    return float(max(abs(eb[lo]), abs(eb[hi + 1])))


def track_ok(A, hole_x):
    """track-level masks: expected |e| within the reach, outside the near-normal deficit, away from dead bands."""
    ae = np.abs(A['e'])
    ok = (ae >= A['e_normal']) & (ae <= E_REACH)
    for h in hole_x:
        ok &= ~((A['x'] > h - HOLE_PAD) & (A['x'] < h + 5 + HOLE_PAD))
    return ok


def pointing_cells(d: pd.DataFrame, arm: str, edges=E25) -> dict:
    """Horizontal pointing cells of one chamber (the unmapped slopes).  A cell is valid when its centre is finite,
    its core sigma < 1.3 x the chamber's median over cells and it holds >= 1000 tracks."""
    g = d[d.arm == arm]
    A = arm_arrays(g, arm)
    r = A['sh'] - A['e']
    hx = holes(A['x'], A['y'])
    A['e_normal'] = normal_halfwidth(A['sh'], A['y']) + 0.06
    tk = track_ok(A, hx)
    X, Y = np.where(tk, A['x'], np.nan), np.where(tk, A['y'], np.nan)
    m = cell_map(X, Y, r, edges, nmin=300)
    e = cell_means(X, Y, A['e'], edges)
    reach = np.isfinite(m['c'])
    sig_med = np.nanmedian(m['sig'][reach & (m['n'] >= NMIN_CELL)])
    valid = reach & (m['sig'] < 1.3 * sig_med) & (m['n'] >= NMIN_CELL)
    return dict(edges=edges, c=m['c'], err=m['err'], n=m['n'], sig=m['sig'], e=e, valid=valid,
                e_normal=A['e_normal'], holes_x=hx, n_tracks=int(len(g)))


def point_rows(pc: dict, arm: str) -> dict:
    """The pointing rows of the joint fit, in the local tan convention."""
    e = pc['edges']
    c = 0.5 * (e[1:] + e[:-1])
    X, Y = np.meshgrid(c, c, indexing='ij')
    v = pc['valid'] & np.isfinite(pc['c']) & np.isfinite(pc['err'])
    gx = G[arm][0]
    return dict(x=X[v], y=Y[v], c_loc=gx * pc['c'][v], e_loc=gx * pc['e'][v], err=pc['err'][v], valid=v, edges=e)


def floor_from(mdl, P):
    """kappa, alpha, gamma by WLS of c on [pred, 1, e] and the floor giving chi2/ndf = 1 about the lines-only map."""
    f = mdl.fields(P['x'], P['y'])
    pred = f['a_x'] + f['s_x'] * P['e_loc']
    X = np.column_stack([pred, np.ones(len(pred)), P['e_loc']])
    fl = 0.0
    for _ in range(30):
        w = 1 / (P['err'] ** 2 + fl ** 2)
        b = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * P['c_loc']))
        res = P['c_loc'] - X @ b
        ndf = len(res) - 3
        lo, hi = 0.0, 0.05
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if np.sum(res ** 2 / (P['err'] ** 2 + mid ** 2)) > ndf:
                lo = mid
            else:
                hi = mid
        if abs(mid - fl) < 1e-6:
            break
        fl = mid
    chi0 = float(np.sum(res ** 2 / P['err'] ** 2) / ndf)
    return dict(floor=float(fl), kappa=float(b[0]), alpha=float(b[1]), gamma=float(b[2]), chi2ndf_stat=chi0,
                rms_res=float(np.std(res)), n_cells=len(res), corr=float(np.corrcoef(pred, P['c_loc'])[0, 1]))


# =============================================================================================== one arm's map
def fit_arm(arm, D, pc=None, floor_mult=1.0, log=None):
    """Fit one chamber.  D: rows_ac / rows_d; pc: pointing cells (pointing_cells) for the joint fit, or None.
    Returns (arm entry of the map file, the fitted Spline)."""
    log = log or (lambda msg: print(msg, flush=True))
    Dc = D[D.clean]
    LT, LS = lam_grids(arm)
    lines = Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS).fit(Dc)
    log(f'{arm}: lines-only CV lam_t {lines.lam_t_sel:g} lam_s {lines.lam_s_sel:g}, rows kept {lines.nkeep}, '
        f'sigma {lines.sig[0]:.5f} / {lines.sig[1]:.5f}')
    mdl, extra = lines, dict(pointing=None)
    if pc is not None:
        P = point_rows(pc, arm)
        fl = floor_from(lines, P)
        pt = dict(P, floor=fl['floor'] * floor_mult, sig_line=np.mean(list(lines.sig.values())), kappas=KAP)
        mdl = Spline(tilt='pot', scale='sep', lam_t=lines.lam_t_sel, lam_s=lines.lam_s_sel, point=pt).fit(Dc)
        extra = dict(pointing=dict(n_cells=int(len(P['x'])), floor=pt['floor'], floor_mult=floor_mult,
                                   floor_fit=fl, kappa=float(mdl.kappa), alpha=float(mdl.nuis[0]),
                                   gamma=float(mdl.nuis[1]), kappa_grid=KAP.tolist(),
                                   n_tracks=pc.get('n_tracks'), e_normal=float(pc['e_normal']),
                                   note='nuisances alpha, gamma, kappa are fitted with the map and not applied'))
        log(f'{arm}: joint with {len(P["x"])} pointing cells, floor {pt["floor"]:.5f}, kappa {mdl.kappa:g}, '
            f'alpha {mdl.nuis[0]:+.5f}, gamma {mdl.nuis[1]:+.5f}')
    tilt, scale = mdl.coef_arrays()
    sup = support_mask(Dc.x.to_numpy(float), Dc.y.to_numpy(float))
    entry = dict(arm=arm, frame=FRAMES[arm], g=[int(v) for v in G[arm]],
                 tilt_basis=dict(type='uniform cubic B-spline, tensor product', lo=LO, hi=HI, h=mdl.ht, nb=mdl.nt),
                 scale_basis=dict(type='uniform cubic B-spline, tensor product', lo=LO, hi=HI, h=mdl.hs, nb=mdl.ns),
                 tilt=tilt, scale=scale, clamp=None,
                 support=dict(grid_mm=GRID.tolist(), mask=sup.astype(int).tolist(),
                              rule='10 mm grid point whose 20 mm cell holds >= 3 clean training lines'),
                 fit=dict(lam_t=mdl.lam_t_sel, lam_s=mdl.lam_s_sel, n_rows_kept=mdl.nkeep,
                          sigma_fit={'xy'[k]: v for k, v in mdl.sig.items()},
                          n_lines=int(Dc.line.nunique()), n_rows_clean=int(len(Dc)),
                          cv=None if not hasattr(lines, 'cvtab') else lines.cvtab.to_dict(orient='list'), **extra))
    entry['clamp'] = clamp_block(entry, Dc)
    entry['clamp_rule'] = f'{list(CLAMP_PCT)} percentile of the field over the clean training lines'
    return entry, mdl


def clamp_block(entry, Dc):
    """Each field clamped to the [0.1, 99.9] percentile of its (position-clamped, unclamped) values at the positions
    of the clean training rows."""
    from rerun.ntof.stage3 import spline_map as SM
    x, y = Dc.x.to_numpy(float), Dc.y.to_numpy(float)
    bad = ~(np.isfinite(x) & np.isfinite(y))
    f = SM.raw_fields(entry, np.where(bad, 0.0, x), np.where(bad, 0.0, y))
    f = [np.where(bad, 0.0, np.clip(v, lo, hi)) for v, (lo, hi) in
         zip(f, ((-9.0, 9.0), (-9.0, 9.0), (-0.99, 9.0), (-0.99, 9.0)))]
    return {k: [float(np.percentile(v, CLAMP_PCT[0])), float(np.percentile(v, CLAMP_PCT[1]))]
            for k, v in zip(('a_x', 'a_y', 's_x', 's_y'), f)}
