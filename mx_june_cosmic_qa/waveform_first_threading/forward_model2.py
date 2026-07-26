#!/usr/bin/env python3
"""Forward model v2.

Upgrades over v1 (wf4_forward):
  - per-plane impulse templates (X vs Y undershoot differ)
  - per-channel gain correction (1.5% flat-field, from gainmap.npz)
  - saturation censoring: masked in NNLS, one-sided penalty in the outer chi2
  - per-plane sharing scale kY (Y kernel stronger, hit-level 0.52 vs 0.45)
  - optional joint two-plane fit: shared charge profile q, t0 tied via the
    measured FEU offset (dt_xy.json, keyed by ftst_x - ftst_y)

Hyper dict: c1, c2, kY, tau_s, sigma_s, sigma_p0, Dp  (+ v used by callers).
"""
import os, json
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
_tz = np.load(os.path.join(BASE, 'templates_perplane.npz'))
TGRID = _tz['grid']
TMPL = {'x': _tz['tmpl_x'], 'y': _tz['tmpl_y']}
_gz = np.load(os.path.join(BASE, 'gainmap.npz'))
GAIN = {'x': _gz['gain_x'], 'y': _gz['gain_y']}
DT_XY = {int(k): v for k, v in json.load(open(os.path.join(BASE, 'dt_xy.json'))).items()}

SNS = 60.0
NSAMP = 32
TS = np.arange(NSAMP) * SNS


def set_nsamp(ns):
    """Adapt to a different DAQ window length (e.g. det4 = 37 samples)."""
    global NSAMP, TS
    NSAMP = int(ns)
    TS = np.arange(NSAMP) * SNS
    _smear_cache.clear()
SAT = 3550.0
DT = 60.0
K = 18
UK = (np.arange(K) + 0.5) * DT
PITCH = 0.78

HYPER0 = dict(c1=0.306, c2=0.057, kY=1.0, tau_s=47.0, sigma_s=87.0,
              sigma_p0=0.098, Dp=0.0114)

_smear_cache = {}

def _templates(plane, sigma_s):
    key = (plane, round(float(sigma_s), 1))
    if key not in _smear_cache:
        base = TMPL[plane]
        _smear_cache[key] = (base, gaussian_filter1d(base, max(sigma_s, 1.0) / 10.0))
    return _smear_cache[key]


def strip_fractions(pos, p0, w, sigma_p0, Dp):
    F = np.zeros((len(pos), K))
    for k in range(K):
        ua, ub = k * DT, (k + 1) * DT
        pa, pb = p0 + w * ua, p0 + w * ub
        pc, half = 0.5 * (pa + pb), 0.5 * abs(pb - pa)
        sig = np.sqrt(sigma_p0 ** 2 + Dp ** 2 * UK[k] + half ** 2 / 3.0)
        F[:, k] = 0.5 * (erf((pos + PITCH / 2 - pc) / (np.sqrt(2) * sig))
                         - erf((pos - PITCH / 2 - pc) / (np.sqrt(2) * sig)))
    return F


def build_matrix(plane, pos, p0, w, t0, hyper):
    c1 = hyper['c1'] * (hyper['kY'] if plane == 'y' else 1.0)
    c2 = hyper['c2'] * (hyper['kY'] if plane == 'y' else 1.0)
    tau = hyper['tau_s']
    tmpl, sm = _templates(plane, hyper['sigma_s'])
    F = strip_fractions(pos, p0, w, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    M = np.zeros((n, NSAMP, K))
    for k in range(K):
        h0 = np.interp(TS - (t0 + UK[k]), TGRID, tmpl, left=0, right=0)
        h1 = np.interp(TS - (t0 + UK[k] + tau), TGRID, sm, left=0, right=0)
        h2 = np.interp(TS - (t0 + UK[k] + 2 * tau), TGRID, sm, left=0, right=0)
        Fk = F[:, k]
        M[:, :, k] += Fk[:, None] * h0[None, :]
        M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
        M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
        M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
        M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
    return M.reshape(n * NSAMP, K)


def prep_plane(P, plane):
    """Gain-correct waveforms; return (W, noise, pos, satmask)."""
    W = P['W'].astype(np.float64).copy()
    g = GAIN[plane][P['ch'].astype(int)]
    W /= g[:, None]
    noise = np.maximum(P['noise'].astype(np.float64), 3.0) / g
    sat = P['W'].astype(np.float64) >= SAT
    return W, noise, P['pos'].astype(np.float64), sat


def chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=True):
    M = build_matrix(plane, pos, p0, w, t0, hyper)
    ok = ~sat.reshape(-1)
    Wt = np.repeat(1.0 / noise, NSAMP)
    A = (M * Wt[:, None])[ok]
    y = (W / noise[:, None]).reshape(-1)[ok]
    try:
        q, rn = nnls(A, y, maxiter=50 * K)
    except Exception:
        return np.inf, None
    chi = rn * rn
    if censor and sat.any():
        model = (M @ q).reshape(W.shape)
        clip = W[sat]
        pen = np.maximum(0.0, clip - model[sat]) / np.repeat(
            noise, NSAMP).reshape(W.shape)[sat]
        chi += float((pen ** 2).sum())
    return chi, q


def fit_plane(P, plane, p0_init, w_init, t0_init, hyper=HYPER0, fix_p0w=None):
    W, noise, pos, sat = prep_plane(P, plane)
    dof = int((~sat).sum())
    if fix_p0w is not None:
        p0f, wf = fix_p0w

        def obj(t0v):
            return chi2_plane(plane, W, noise, pos, sat, p0f, wf, float(t0v),
                              hyper)[0]
        grid = np.arange(t0_init - 240, t0_init + 241, 40.0)
        cs = [obj(t) for t in grid]
        t0b = grid[int(np.argmin(cs))]
        r = minimize(lambda v: obj(v[0]), [t0b], method='Nelder-Mead',
                     options=dict(xatol=2.0, fatol=0.5, maxiter=60))
        chi, q = chi2_plane(plane, W, noise, pos, sat, p0f, wf, float(r.x[0]), hyper)
        return dict(chi2=chi, dof=dof, p0=p0f, w=wf, t0=float(r.x[0]), q=q)

    def obj(v):
        return chi2_plane(plane, W, noise, pos, sat, v[0], v[1], v[2], hyper)[0]

    v0 = np.array([p0_init, w_init, t0_init])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.3, maxiter=300,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0], [0.8, 0, 0], [0, 0.004, 0], [0, 0, 50]])))
    r = minimize(obj, r.x, method='Nelder-Mead',
                 options=dict(xatol=5e-4, fatol=0.2, maxiter=200))
    chi, q = chi2_plane(plane, W, noise, pos, sat, r.x[0], r.x[1], r.x[2], hyper)
    return dict(chi2=chi, dof=dof, p0=float(r.x[0]), w=float(r.x[1]),
                t0=float(r.x[2]), q=q, nfev=r.nfev)


def fit_joint(ev, p0x_i, wx_i, p0y_i, wy_i, t0_i, hyper=HYPER0):
    """Joint two-plane fit: shared q (per-plane scalar), t0y = t0x - dt."""
    dt = DT_XY.get(int(ev['ftst_x'] - ev['ftst_y']), -18.8)
    Wx, nx, px_, sx = prep_plane(ev['x'], 'x')
    Wy, ny, py_, sy = prep_plane(ev['y'], 'y')
    dof = int((~sx).sum() + (~sy).sum())

    def solve(p0x, wx, p0y, wy, t0):
        Mx = build_matrix('x', px_, p0x, wx, t0, hyper)
        My = build_matrix('y', py_, p0y, wy, t0 - dt, hyper)
        okx = ~sx.reshape(-1); oky = ~sy.reshape(-1)
        Ax = (Mx * np.repeat(1.0 / nx, NSAMP)[:, None])[okx]
        Ay = (My * np.repeat(1.0 / ny, NSAMP)[:, None])[oky]
        yx = (Wx / nx[:, None]).reshape(-1)[okx]
        yy = (Wy / ny[:, None]).reshape(-1)[oky]
        # per-plane scale: solve NNLS on x, get alpha from y projection, iterate
        alpha = 1.0
        for _ in range(2):
            A = np.vstack([Ax, alpha * Ay])
            y = np.concatenate([yx, yy])
            try:
                q, rn = nnls(A, y, maxiter=50 * K)
            except Exception:
                return np.inf, None, 1.0
            my = Ay @ q
            den = float(my @ my)
            alpha = float(my @ yy / den) if den > 0 else 1.0
        chi = float(((Ax @ q - yx) ** 2).sum() + ((alpha * my - yy) ** 2).sum())
        return chi, q, alpha

    def obj(v):
        return solve(*v)[0]

    v0 = np.array([p0x_i, wx_i, p0y_i, wy_i, t0_i])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.5, maxiter=500,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0, 0, 0], [0.8, 0, 0, 0, 0],
                                   [0, 0.004, 0, 0, 0], [0, 0, 0.8, 0, 0],
                                   [0, 0, 0, 0.004, 0], [0, 0, 0, 0, 50]])))
    r = minimize(obj, r.x, method='Nelder-Mead',
                 options=dict(xatol=5e-4, fatol=0.3, maxiter=300))
    chi, q, alpha = solve(*r.x)
    return dict(chi2=chi, dof=dof, p0x=float(r.x[0]), wx=float(r.x[1]),
                p0y=float(r.x[2]), wy=float(r.x[3]), t0=float(r.x[4]),
                q=q, alpha=alpha, nfev=r.nfev)


def init_guess(P, plane, tan_ref, p0_ref, v_scale=36.6e-3):
    W = P['W'].astype(np.float64)
    noise = np.maximum(P['noise'].astype(np.float64), 3.0)
    amax = W.max(axis=1)
    thr = np.maximum(6 * noise, 60)
    sel = amax > thr
    if sel.sum() == 0:
        sel = amax == amax.max()
    leads = []
    for wv in W[sel]:
        ipk = int(np.argmax(wv)); a = wv.max()
        for kk in range(1, ipk + 1):
            if wv[kk] >= 0.5 * a > wv[kk - 1]:
                leads.append(SNS * (kk - 1 + (0.5 * a - wv[kk - 1]) /
                                    (wv[kk] - wv[kk - 1])))
                break
    t0g = (min(leads) if leads else 400.0)
    return p0_ref, tan_ref * v_scale, t0g
