"""r2_lib.py -- audit R2: selection of n_TOF tracks, drift-time field (P-spline over strip coordinates), helpers.
Frames.  Strip frame: U = (x_p0 + 14 x_tan_theta - 200)/100, W = (y_p0 + 14 y_tan_theta - 200)/100 (100 mm units), like the bench.
n_TOF local frame (G): x_local = -(x_p0 - 199.29), y_local = y_p0 - 199.29; a_ntof = SGN * a_strip, SGN = A:(+1,-1) C:(+1,+1)."""
import sys, json
import numpy as np, pandas as pd
from scipy.interpolate import BSpline
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17')
W_ = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{W_}/audit/r2'
HALF = 199.29
SGN = {('A', 'x'): 1.0, ('A', 'y'): -1.0, ('C', 'x'): 1.0, ('C', 'y'): 1.0, ('B', 'x'): np.nan, ('D', 'x'): np.nan}
G0 = 28.2
UD = (-2.05, 2.05)
BETA = 0.68
MID = 14.0


def load(which, arm, railed_ok=False, chi2=3.0, single=True, nstr=6):
    d = pd.read_parquet(f'{OUT}/tr_{which}.parquet')
    d = d[d.arm == arm]
    ok = (d.x_quality_ok & d.y_quality_ok & (d.chi2dof_x < chi2) & (d.chi2dof_y < chi2)
          & (d.x_n_strips >= nstr) & (d.y_n_strips >= nstr))
    if single:
        ok &= (d.n_cand_x == 1) & (d.n_cand_y == 1)
    if not railed_ok:
        ok &= ~d.drift_railed.fillna(False).astype(bool)
    ok &= np.isfinite(d.x_q_u90) & np.isfinite(d.y_q_u90) & (d.x_q_sum > 50) & (d.y_q_sum > 50) & (d.x_q_sum < 1e6) & (d.y_q_sum < 1e6)
    d = d[ok].copy()
    d['U'] = (d.x_p0 + MID * d.x_tan_theta - 200) / 100
    d['Wc'] = (d.y_p0 + MID * d.y_tan_theta - 200) / 100
    d['Tm'] = 0.5 * (d.x_q_u90 + d.y_q_u90)
    d['lq'] = np.log(0.5 * (d.x_q_sum + d.y_q_sum))
    d['absx'] = d.x_tan_theta.abs()
    d['absy'] = d.y_tan_theta.abs()
    d['run_i'] = d.run.str.replace('run_', '').astype(int)
    return d


def knots(K, lo, hi):
    ni = K - 4
    return np.concatenate([[lo] * 4, np.linspace(lo, hi, ni + 2)[1:-1], [hi] * 4])


class Spl:
    def __init__(self, K, ud=UD, wd=UD):
        self.K, self.ud, self.wd = K, ud, wd
        self.tu, self.tw = knots(K, *ud), knots(K, *wd)
        self.bu = BSpline(self.tu, np.eye(K), 3, extrapolate=False)
        self.bw = BSpline(self.tw, np.eye(K), 3, extrapolate=False)
        D = np.diff(np.eye(K), 2, axis=0)
        P1 = D.T @ D
        I = np.eye(K)
        self.P = np.kron(P1, I) + np.kron(I, P1)

    def basis(self, U, W, du=0, dw=0):
        U = np.clip(U, self.ud[0] + 1e-9, self.ud[1] - 1e-9)
        W = np.clip(W, self.wd[0] + 1e-9, self.wd[1] - 1e-9)
        bu = (self.bu if du == 0 else self.bu.derivative(du))(U)
        bw = (self.bw if dw == 0 else self.bw.derivative(dw))(W)
        return (bu[:, :, None] * bw[:, None, :]).reshape(len(U), -1)

    def fit(self, U, W, y, lam, wts=None, clip=3.5, passes=3, extra=None):
        """penalised robust LS.  extra: (n, m) design columns fitted jointly (unpenalised); returns coef(spline), coef(extra), keep"""
        B = self.basis(U, W)
        n = len(y)
        wts = np.ones(n) if wts is None else wts
        keep = wts > 0
        lam_eff = lam * wts.sum() / self.K ** 2
        m = 0 if extra is None else extra.shape[1]
        Bf = B if extra is None else np.hstack([B, extra])
        Pf = np.zeros((Bf.shape[1],) * 2); Pf[:B.shape[1], :B.shape[1]] = self.P
        for it in range(passes + 1):
            w = wts * keep
            A = (Bf * w[:, None]).T @ Bf + lam_eff * Pf + 1e-9 * np.eye(Bf.shape[1])
            c = np.linalg.solve(A, (Bf * w[:, None]).T @ y)
            r = y - Bf @ c
            rs = 1.4826 * np.median(np.abs(r[wts > 0]))
            keep = np.abs(r) < clip * rs
        return c[:B.shape[1]], c[B.shape[1]:], keep, rs


def grid(n=80, lo=-1.9, hi=1.9):
    u = np.linspace(lo, hi, n)
    return np.meshgrid(u, u, indexing='ij')


def mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 5 else np.nan
