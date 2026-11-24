#!/usr/bin/env python3
"""t6lib.py -- data rows and constrained (surface) angle-map models for test T6_surface_map.

Frames (verified from the code and on the line cache, see report):
  The maps are fitted on GLOBAL slopes: A, C view p -> d_p/d_z; D x view -> d_z/d_x, D y view -> d_y/d_x.
  Chamber local axes (geometry.detector_transforms, local_to_global uses (xl, yl, -w)):
    A  u=+x  v=+y  w(depth, strips->cathode)=-z     C  u=-x  v=+y  w=+z     D  u=-z  v=+y  w=-x
  local reco tan t = -d(local p)/dw (build_tracks.local_and_global), and global slope = g t with
    g: A (+1, +1), C (+1, -1), D (-1, +1)  (x view, y view) -- checked: (d_p/d_z)/tan = +1,+1,+1,-1 on lines.parquet.
  So the residual in the LOCAL tan convention is r_loc = g r (and j_loc = g j); the scale s is unchanged.
  Physical drift-line deflection delta = d(local p)/dw offset = -r_loc.  Models below are written in r_loc:
      r_loc_x = dphi/dx_l + s j_loc,   r_loc_y = dphi/dy_l + s j_loc      (potential model)
  with x_l = x_local, y_l = y_local (mm); polynomials in u = x/200, v = y/200.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path.insert(0, str(REPO))
from rerun.ntof.improve import angle_map as AM  # noqa: E402

WD = Path('/media/ucla/work/ntof_pass1/ana_r2/track_quality')
OUT = Path('/media/ucla/work/claude-jobs/eb8200b3/T6_surface_map')
G = {'A': (1, 1), 'C': (1, -1), 'D': (-1, 1)}
L = 200.0
CLEAN = 0.10


def rows_ac(arm: str) -> pd.DataFrame:
    """All A-C lines as (line, view) rows for chamber arm; flag clean exactly as angle_map.clean_lines."""
    M = pd.read_parquet(WD / 'lines.parquet')
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


def rows_d() -> pd.DataFrame:
    M = pd.read_parquet(WD / 'bd_lines.parquet')
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


# ----------------------------------------------------------------------------------------------- polynomial models
def mono(n):
    return [(i, k) for t in range(n + 1) for i in range(t, -1, -1) for k in [t - i]]


class Poly:
    """Linear model in r_loc.  spec keys: n (potential order, or tilt-poly order+1 when pot=False), pot (bool),
    m (scale order), sep (separate scale per view), soff (common scale + per-view constant), rot (in-plane rotation),
    cellA (free 50 mm cells for the tilt instead of a polynomial; edges), ridge."""

    def __init__(self, **spec):
        self.spec = dict(n=2, pot=True, m=0, sep=False, soff=False, rot=False, cellA=None, ridge=1e-3)
        self.spec.update(spec)

    def X(self, D: pd.DataFrame, jo_meas=False):
        sp = self.spec
        u, v = D.x.to_numpy() / L, D.y.to_numpy() / L
        u, v = np.clip(np.nan_to_num(u), -1.1, 1.1), np.clip(np.nan_to_num(v), -1.1, 1.1)
        vw = D.view.to_numpy()
        jl = (D.j * D.g).to_numpy()
        cols, names = [], []
        if sp['cellA'] is not None:
            e = np.asarray(sp['cellA'])
            nc = len(e) - 1
            ix = np.clip(np.digitize(D.x, e) - 1, 0, nc - 1)
            iy = np.clip(np.digitize(D.y, e) - 1, 0, nc - 1)
            for w in (0, 1):
                cols.append((vw == w).astype(float)); names.append(f'a0_{w}')
                for a in range(nc):
                    for b in range(nc):
                        cols.append(((vw == w) & (ix == a) & (iy == b)).astype(float)); names.append(f'cell_{w}_{a}_{b}')
        elif sp['pot']:
            for (i, k) in mono(sp['n']):
                if i + k == 0:
                    continue
                dx = i * u ** max(i - 1, 0) * v ** k if i > 0 else np.zeros_like(u)
                dy = k * u ** i * v ** max(k - 1, 0) if k > 0 else np.zeros_like(u)
                cols.append(np.where(vw == 0, dx, dy)); names.append(f'phi_{i}{k}')
        else:
            for w in (0, 1):
                for (i, k) in mono(sp['n'] - 1):
                    cols.append((vw == w) * u ** i * v ** k); names.append(f'a{w}_{i}{k}')
        if sp['sep']:
            for w in (0, 1):
                for (i, k) in mono(sp['m']):
                    cols.append((vw == w) * jl * u ** i * v ** k); names.append(f's{w}_{i}{k}')
        else:
            for (i, k) in mono(sp['m']):
                cols.append(jl * u ** i * v ** k); names.append(f's_{i}{k}')
            if sp['soff']:
                cols.append((vw == 1) * jl); names.append('s_off_y')
        if sp['rot']:
            jo = ((D.slope_o if jo_meas else D.jo) * D.go).to_numpy()
            cols.append(np.where(vw == 0, -jo, jo)); names.append('psi')
        self.names = names
        return np.column_stack(cols)

    def fit(self, D: pd.DataFrame, n_iter=5):
        X = self.X(D)
        y = (D.r * D.g).to_numpy()
        vw = D.view.to_numpy()
        keep = np.ones(len(y), bool)
        lam = self.spec['ridge'] if self.spec['cellA'] is not None else 0.0
        pen = np.array([1.0 if n.startswith('cell') else 0.0 for n in self.names]) if lam else None
        for _ in range(n_iter):
            Xk, yk = X[keep], y[keep]
            A = Xk.T @ Xk
            if lam:
                A = A + np.diag(pen) * len(yk) * lam
            beta = np.linalg.lstsq(A, Xk.T @ yk, rcond=None)[0]
            res = y - X @ beta
            nk = keep.copy()
            for w in (0, 1):
                m = vw == w
                sig = 1.4826 * np.median(np.abs(res[m & keep]))
                nk[m] = np.abs(res[m]) < 3 * sig
            keep = nk
        self.beta = beta
        res = y - X @ beta
        sig2 = np.mean(res[keep] ** 2)
        A = X[keep].T @ X[keep]
        if lam:
            A = A + np.diag(pen) * keep.sum() * lam
        self.cov = sig2 * np.linalg.pinv(A)
        self.nkeep = int(keep.sum())
        self.sig = float(np.sqrt(sig2))
        return self

    def tilt_scale(self, D):
        """r_loc = tilt + scale * j_loc (+ rot); returns tilt(loc), scale, rot_term(loc, with measured other slope)."""
        X = self.X(D, jo_meas=True)
        b = self.beta.copy()
        isS = np.array([n.startswith('s') for n in self.names])
        isR = np.array([n == 'psi' for n in self.names])
        tilt = X[:, ~isS & ~isR] @ b[~isS & ~isR]
        jl = (D.j * D.g).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = (X[:, isS] @ b[isS]) / np.where(jl == 0, np.nan, jl)
        # recompute scale without dividing (j may be 0): evaluate scale columns with j=1
        D1 = D.copy(); D1['j'] = D1.g  # j_loc = 1
        X1 = self.X(D1, jo_meas=True)
        scale = X1[:, isS] @ b[isS]
        rot = X[:, isR] @ b[isR] if isR.any() else np.zeros(len(D))
        return tilt, scale, rot

    def corrected_resid(self, D):
        tilt, scale, rot = self.tilt_scale(D)
        g = D.g.to_numpy()
        a_glob = g * (tilt + rot)
        return (D.slope.to_numpy() - a_glob) / (1 + scale) - D.j.to_numpy()


class Free:
    """angle_map.py free cells, per view (edges, n_min)."""

    def __init__(self, edges=AM.EDGES, n_min=AM.N_MIN, globalonly=False):
        self.edges, self.n_min, self.globalonly = np.asarray(edges), n_min, globalonly

    def fit(self, D):
        AM.EDGES, AM.CENTRES, AM.N_MIN = self.edges, 0.5 * (self.edges[1:] + self.edges[:-1]), self.n_min
        self.maps = {}
        for w in (0, 1):
            Dw = D[D.view == w]
            mp = AM.fit_map(pd.DataFrame(dict(x=Dw.x, y=Dw.y, j=Dw.j, r=Dw.r)))
            if self.globalonly:
                mp = dict(mp, a=np.full_like(np.asarray(mp['a']), mp['a0']).tolist(),
                          s=np.full_like(np.asarray(mp['s']), mp['s0']).tolist())
            self.maps[w] = mp
        AM.EDGES, AM.CENTRES, AM.N_MIN = np.arange(-200.0, 201.0, 50.0), None, 80
        AM.CENTRES = 0.5 * (AM.EDGES[1:] + AM.EDGES[:-1])
        return self

    def corrected_resid(self, D):
        out = np.full(len(D), np.nan)
        for w in (0, 1):
            m = (D.view == w).to_numpy()
            Dw = D[m]
            out[m] = AM.correct(Dw.slope.to_numpy(), Dw.x.to_numpy(), Dw.y.to_numpy(), self.maps[w]) - Dw.j.to_numpy()
        return out


class NoMap:
    def fit(self, D):
        return self

    def corrected_resid(self, D):
        return D.slope.to_numpy() - D.j.to_numpy()


def mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 10 else np.nan


def widths(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    q = np.percentile(v, [2.5, 16, 84, 97.5])
    return (q[2] - q[1]) / 2, (q[3] - q[0]) / 2
