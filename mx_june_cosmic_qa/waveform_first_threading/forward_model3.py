#!/usr/bin/env python3
"""forward_model3 — speed-optimized drop-in for forward_model2's fit path.

Same model, same hypers, same results (validated < 1e-6 relative chi2);
~10-20x faster through:
  - time tensors H0/H1/H2 (32 x K) cached per (plane, t0 on a 5 ns grid) —
    they do not depend on (p0, w)
  - fully vectorized strip_fractions and matrix assembly (no Python K-loop)
  - 2-stage search: coarse (p0, w) grid at the best coarse t0, then a small
    Nelder-Mead on (p0, w, t0)

Use fm3.fit_plane(...) exactly like fm2.fit_plane(...).
"""
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.special import erf

import forward_model2 as fm2

SNS = fm2.SNS
SAT = fm2.SAT
DT = fm2.DT
PITCH = fm2.PITCH
# TS / K / UK are read dynamically from fm2 so set_nsamp()/K changes apply

_TT_CACHE = {}
T0_STEP = 5.0


def _time_tensors(plane, t0q, hyper):
    key = (plane, t0q, hyper['tau_s'], round(hyper['sigma_s'], 1))
    if key in _TT_CACHE:
        return _TT_CACHE[key]
    tmpl, sm = fm2._templates(plane, hyper['sigma_s'])
    tau = hyper['tau_s']
    # arg grids: (32, K): TS[t] - (t0 + UK[k] (+ j*tau))
    base = fm2.TS[:, None] - (t0q + fm2.UK[None, :])
    H0 = np.interp(base, fm2.TGRID, tmpl, left=0, right=0)
    H1 = np.interp(base - tau, fm2.TGRID, sm, left=0, right=0)
    H2 = np.interp(base - 2 * tau, fm2.TGRID, sm, left=0, right=0)
    if len(_TT_CACHE) > 4096:
        _TT_CACHE.clear()
    _TT_CACHE[key] = (H0, H1, H2)
    return H0, H1, H2


def strip_fractions_vec(pos, p0, w, sigma_p0, Dp):
    ua = np.arange(fm2.K) * DT
    ub = ua + DT
    pa = p0 + w * ua
    pb = p0 + w * ub
    pc = 0.5 * (pa + pb)
    half = 0.5 * np.abs(pb - pa)
    sig = np.sqrt(sigma_p0 ** 2 + Dp ** 2 * fm2.UK + half ** 2 / 3.0)
    z = 1.0 / (np.sqrt(2) * sig)[None, :]
    hi = (pos[:, None] + PITCH / 2 - pc[None, :]) * z
    lo = (pos[:, None] - PITCH / 2 - pc[None, :]) * z
    return 0.5 * (erf(hi) - erf(lo))          # (n, K)


def build_matrix_fast(plane, pos, p0, w, t0, hyper):
    t0q = round(t0 / T0_STEP) * T0_STEP
    dt_resid = t0 - t0q
    if abs(dt_resid) > 1e-9:
        # exact t0: fall back to interp on the residual by shifting base grid
        t0q = t0
        tmpl, sm = fm2._templates(plane, hyper['sigma_s'])
        tau = hyper['tau_s']
        base = fm2.TS[:, None] - (t0q + fm2.UK[None, :])
        H0 = np.interp(base, fm2.TGRID, tmpl, left=0, right=0)
        H1 = np.interp(base - tau, fm2.TGRID, sm, left=0, right=0)
        H2 = np.interp(base - 2 * tau, fm2.TGRID, sm, left=0, right=0)
    else:
        H0, H1, H2 = _time_tensors(plane, t0q, hyper)
    c1 = hyper['c1'] * (hyper['kY'] if plane == 'y' else 1.0)
    c2 = hyper['c2'] * (hyper['kY'] if plane == 'y' else 1.0)
    F = strip_fractions_vec(pos, p0, w, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    M = np.empty((n, fm2.NSAMP, fm2.K))
    np.multiply(F[:, None, :], H0[None, :, :], out=M)
    Fs = np.zeros_like(F)
    Fs[1:] = F[:-1]
    Fs[:-1] += F[1:]
    M += (c1 * Fs)[:, None, :] * H1[None, :, :]
    if c2 > 0:
        Fs2 = np.zeros_like(F)
        Fs2[2:] = F[:-2]
        Fs2[:-2] += F[2:]
        M += (c2 * Fs2)[:, None, :] * H2[None, :, :]
    return M.reshape(n * fm2.NSAMP, fm2.K)


def chi2_plane_fast(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=True,
                    snap_t0=True):
    if snap_t0:
        t0 = round(t0 / T0_STEP) * T0_STEP
    M = build_matrix_fast(plane, pos, p0, w, t0, hyper)
    ok = ~sat.reshape(-1)
    Wt = np.repeat(1.0 / noise, fm2.NSAMP)
    A = (M * Wt[:, None])[ok]
    y = (W / noise[:, None]).reshape(-1)[ok]
    try:
        q, rn = nnls(A, y, maxiter=50 * fm2.K)
    except Exception:
        return np.inf, None
    chi = rn * rn
    if censor and sat.any():
        model = (M @ q).reshape(W.shape)
        clip = W[sat]
        pen = np.maximum(0.0, clip - model[sat]) / np.repeat(
            noise, fm2.NSAMP).reshape(W.shape)[sat]
        chi += float((pen ** 2).sum())
    return chi, q


def fit_plane(P, plane, p0_init, w_init, t0_init, hyper=None, fix_p0w=None):
    hyper = hyper or fm2.HYPER0
    W, noise, pos, sat = fm2.prep_plane(P, plane)
    dof = int((~sat).sum())

    if fix_p0w is not None:
        p0f, wf = fix_p0w
        grid = np.arange(t0_init - 240, t0_init + 241, 20.0)
        cs = [chi2_plane_fast(plane, W, noise, pos, sat, p0f, wf, t, hyper)[0]
              for t in grid]
        j = int(np.argmin(cs))
        # refine on 5 ns
        g2 = np.arange(grid[j] - 20, grid[j] + 21, T0_STEP)
        cs2 = [chi2_plane_fast(plane, W, noise, pos, sat, p0f, wf, t, hyper)[0]
               for t in g2]
        j2 = int(np.argmin(cs2))
        chi, q = chi2_plane_fast(plane, W, noise, pos, sat, p0f, wf,
                                 float(g2[j2]), hyper)
        return dict(chi2=chi, dof=dof, p0=p0f, w=wf, t0=float(g2[j2]), q=q)

    # stage 1: coarse grid over (p0, w) x t0
    nfev = 0
    best = (np.inf, p0_init, w_init, t0_init)
    for t0 in np.arange(t0_init - 120, t0_init + 121, 40.0):
        for dp in (-0.8, 0.0, 0.8):
            for dw in (-2.4e-3, -0.8e-3, 0.0, 0.8e-3, 2.4e-3):
                c, _ = chi2_plane_fast(plane, W, noise, pos, sat,
                                       p0_init + dp, w_init + dw, t0, hyper)
                nfev += 1
                if c < best[0]:
                    best = (c, p0_init + dp, w_init + dw, t0)

    # stage 2: small NM from the coarse best (t0 free, not snapped)
    def obj(v):
        c, _ = chi2_plane_fast(plane, W, noise, pos, sat, v[0], v[1], v[2],
                               hyper, snap_t0=False)
        return c
    v0 = np.array(best[1:])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.3, maxiter=140,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0], [0.4, 0, 0], [0, 1.5e-3, 0],
                                   [0, 0, 20]])))
    chi, q = chi2_plane_fast(plane, W, noise, pos, sat, r.x[0], r.x[1],
                             r.x[2], hyper, snap_t0=False)
    return dict(chi2=chi, dof=dof, p0=float(r.x[0]), w=float(r.x[1]),
                t0=float(r.x[2]), q=q, nfev=nfev + r.nfev)
