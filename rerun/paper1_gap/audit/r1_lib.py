"""r1_lib: data loading, tensor-product P-spline gap fields, tilt prediction helpers (audit R1)."""
import sys, json, pickle, os
import numpy as np, pandas as pd
from scipy.interpolate import BSpline
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap')
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import gapmap as GM, cellobs as CO, track_test as TT
import c_lib as CL
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/r1'
G0 = 28.2
BETA_G = 0.68
UD, WD = (-1.75, 1.75), (-1.56, 1.56)
BUNDLE = json.load(open(f'{WORK}/pass1/{GM.RUNS["run1"]["rel"]}/calib_bundle/bundle.json'))['bow']
BUNDLE625 = json.load(open(f'{WORK}/pass1_all/mx17_det3_day_6-25-26/long_run/mx17_3/calib_bundle/bundle.json'))['bow']
REL = {'run1': GM.RUNS['run1']['rel'], 'run2': GM.RUNS['run2']['rel'], 'r625': 'mx17_det3_day_6-25-26/long_run'}


def addUW(d):
    d['U'] = (d.x_p0 + 14 * d.x_tan_theta - 200) / 100
    d['W'] = (d.y_p0 + 14 * d.y_tan_theta - 200) / 100
    return d


def qwalk(d):
    """q_u90 (mean of x and y) with the amplitude walk removed (one global slope against ln mean q_sum)."""
    q = 0.5 * (d.x_q_u90.values + d.y_q_u90.values)
    lq = np.log(0.5 * (d.x_q_sum.values + d.y_q_sum.values))
    ok = np.isfinite(q) & np.isfinite(lq) & (q > 0)
    keep = ok.copy()
    for _ in range(3):
        A = np.vstack([np.ones(keep.sum()), lq[keep] - np.median(lq[ok])]).T
        c = np.linalg.lstsq(A, q[keep], rcond=None)[0]
        r = q - c[0] - c[1] * (lq - np.median(lq[ok]))
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = ok & (np.abs(r) < 4 * rs)
    return q - c[1] * (lq - np.median(lq[ok])), ok & keep, float(c[1])


def load(run):
    """dict: tab (selected table with U, W), qamp (q_u90 walk-removed), qok, T (golden: colsum matched frame)"""
    if run == 'r625':
        tab = addUW(CL.load(REL[run]))
    else:
        tab = addUW(GM.load_table(WORK, REL[run]))
    qa, qok, qs = qwalk(tab)
    D = dict(tab=tab, qamp=qa, qok=qok, q_slope=qs)
    if run != 'r625':
        m, info = GM.colsum_T(WORK, run, tab)
        D['m'] = addUW(m); D['info'] = info
    return D


# ----------------------------------------------------------------------- splines
def knots(K, lo, hi):
    ni = K - 4
    return np.concatenate([[lo] * 4, np.linspace(lo, hi, ni + 2)[1:-1], [hi] * 4])


class Spl:
    """tensor-product cubic B-spline K x K over (UD, WD) with second-difference penalty (P-spline)."""
    def __init__(self, K):
        self.K = K
        self.tu, self.tw = knots(K, *UD), knots(K, *WD)
        self.bu = BSpline(self.tu, np.eye(K), 3, extrapolate=False)
        self.bw = BSpline(self.tw, np.eye(K), 3, extrapolate=False)
        D = np.diff(np.eye(K), 2, axis=0)
        P1 = D.T @ D
        I = np.eye(K)
        self.P = np.kron(P1, I) + np.kron(I, P1)      # coefficient index = i * K + j (U major)

    def basis(self, U, W, du=0, dw=0):
        U = np.clip(U, UD[0] + 1e-9, UD[1] - 1e-9); W = np.clip(W, WD[0] + 1e-9, WD[1] - 1e-9)
        bu = (self.bu if du == 0 else self.bu.derivative(du))(U)
        bw = (self.bw if dw == 0 else self.bw.derivative(dw))(W)
        return (bu[:, :, None] * bw[:, None, :]).reshape(len(U), -1)

    def fit(self, U, W, y, lam, wts=None, clip=3.5, passes=3, B=None):
        """penalised robust LS; wts = integer resample multiplicities (bootstrap); returns (coef, keep)."""
        B = self.basis(U, W) if B is None else B
        n = len(y)
        wts = np.ones(n) if wts is None else wts
        keep = wts > 0
        lam_eff = lam * wts.sum() / self.K ** 2
        for it in range(passes + 1):
            w = wts * keep
            A = (B * w[:, None]).T @ B + lam_eff * self.P + 1e-9 * np.eye(B.shape[1])
            c = np.linalg.solve(A, (B * w[:, None]).T @ y)
            r = y - B @ c
            rs = 1.4826 * np.median(np.abs(r[wts > 0]))
            keep = np.abs(r) < clip * rs
        return c, keep


def gradU(spl, c, U, W):
    return spl.basis(U, W, 1, 0) @ c


def gradW(spl, c, U, W):
    return spl.basis(U, W, 0, 1) @ c


class GapField:
    """delta g(U, W) [mm] = G0 * S / (1 + beta); S spline of ln(amplitude-walk-free T or q_u90)."""
    def __init__(self, U, W, lnT, K, lam, beta, wts=None, g0=G0):
        self.spl = Spl(K); self.beta = beta; self.g0 = g0
        self.c, self.keep = self.spl.fit(U, W, lnT - np.mean(lnT), lam, wts)
        self.fac = g0 / (1 + beta)

    def dg(self, U, W):
        return self.fac * (self.spl.basis(U, W) @ self.c)

    def grad(self, U, W):
        """(d dg/dx, d dg/dy) dimensionless (mm per mm)."""
        return self.fac * gradU(self.spl, self.c, U, W) / 100, self.fac * gradW(self.spl, self.c, U, W) / 100


# ----------------------------------------------------------------------- mesh term
def hgrad(U, W, which='h', bundle=None):
    """(dh/dx, dh/dy) of the bundle's bow block (dimensionless); which = h | h_x | h_y | (hx, hy) per view"""
    b = (bundle or BUNDLE)
    def one(k):
        h = b[k]
        return (h['gu'] + h['kuu'] * U + h['kuw'] * W) / 100, (h['gw'] + h['kww'] * W + h['kuw'] * U) / 100
    if which == 'perview':
        return one('h_x')[0], one('h_y')[1]
    return one(which)


# ----------------------------------------------------------------------- views / cells
def view(d, p, idx=None):
    s = (np.abs(d[f'tan_ref_{p}'].values) <= GM.TAN_LADDER)
    return dict(U=d.U.values[s], W=d.W.values[s], w=d[f'{p}_w'].values[s] * 1e3, tr=d[f'tan_ref_{p}'].values[s],
                tp=d[f'{p}_tan_theta'].values[s], sel=s)


def cellmean(U, W, x, n):
    eu, ew = np.linspace(*CL.UL, n + 1), np.linspace(*CL.WL, n + 1)
    i, j = np.searchsorted(eu, U) - 1, np.searchsorted(ew, W) - 1
    out = np.full((n, n), np.nan)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    for a in range(n):
        for b in range(n):
            m = ok & (i == a) & (j == b)
            if m.sum() >= 1:
                out[a, b] = np.mean(x[m])
    return out


def plane_w(M, A, Ae):
    return CL.plane_fit(M, A, Ae)
