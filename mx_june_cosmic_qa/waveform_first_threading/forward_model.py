#!/usr/bin/env python3
"""Forward-model waveform fit for one plane of one event.

Model
-----
Track in (pos, t): charge arriving at drift time u (since mesh crossing at t0)
lands at pos p(u) = p0 + w*u   (w = tan * v_drift, mm/ns, signed).
Charge profile: non-negative amplitudes q_k in K time bins of DT=60 ns.
Geometric spread: within-bin segment boxcar ⊗ Gaussian sigma_p(u).
Resistive sharing: strip-level kernel [c2, c1, 1, c1, c2] with delay tau_s
per neighbor step (shared copies use the same template).
Strip i waveform: sum_k q_k * sum_j coef_j * F[i,j,k] * h(t - t0 - u_k - |j| tau_s)
where F = fraction of bin-k charge geometrically on strip i-j.

Fit: NNLS over q_k (weighted by per-strip noise), Nelder-Mead over (p0, w, t0).
"""
import os, pickle
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.special import erf

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
_tz = np.load(os.path.join(BASE, 'template.npz'))
TGRID, TMPL = _tz['grid'], _tz['tmpl']
SNS = 60.0
TS = np.arange(32) * SNS
SAT = 3550.0
DT = 60.0
K = 18                       # charge bins: u in [0, 1080)
UK = (np.arange(K) + 0.5) * DT
PITCH = 0.78

HYPER = dict(c1=0.48, c2=0.08, tau_s=69.0, sigma_p0=0.35, Dp=0.012)
# sigma_p(u) = sqrt(sigma_p0^2 + Dp^2 * u)   [mm], u in ns


def strip_fractions(pos, p0, w, sigma_p0, Dp):
    """F[i, k]: fraction of bin-k charge on strip i (geometric, pre-sharing)."""
    n = len(pos)
    F = np.zeros((n, K))
    for k in range(K):
        ua, ub = k * DT, (k + 1) * DT
        pa, pb = p0 + w * ua, p0 + w * ub
        pc, half = 0.5 * (pa + pb), 0.5 * abs(pb - pa)
        sig = np.sqrt(sigma_p0 ** 2 + Dp ** 2 * UK[k] + half ** 2 / 3.0)
        lo = pos - PITCH / 2.0
        hi = pos + PITCH / 2.0
        F[:, k] = 0.5 * (erf((hi - pc) / (np.sqrt(2) * sig))
                         - erf((lo - pc) / (np.sqrt(2) * sig)))
    return F


def template_comb(shift):
    """Template evaluated on the 32-sample comb for arrival time `shift`.
    Template grid is aligned so t50 (50% rise) is at 0; shift = arrival time
    of the charge; we take t50 ≈ arrival + fixed offset absorbed by t0."""
    return np.interp(TS - shift, TGRID, TMPL, left=0.0, right=0.0)


def build_matrix(pos, p0, w, t0, hyper):
    """M[(i*32+t), k] model matrix."""
    c1, c2, tau = hyper['c1'], hyper['c2'], hyper['tau_s']
    F = strip_fractions(pos, p0, w, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    M = np.zeros((n, 32, K))
    # precompute per-k templates for the 3 delay classes
    for k in range(K):
        h0 = template_comb(t0 + UK[k])
        h1 = template_comb(t0 + UK[k] + tau)
        h2 = template_comb(t0 + UK[k] + 2 * tau)
        Fk = F[:, k]
        # direct
        M[:, :, k] += Fk[:, None] * h0[None, :]
        # first neighbours: strip i receives c1 * charge on i-1 and i+1
        M[1:, :, k] += c1 * Fk[:-1, None] * h1[None, :]
        M[:-1, :, k] += c1 * Fk[1:, None] * h1[None, :]
        # second neighbours
        if c2 > 0:
            M[2:, :, k] += c2 * Fk[:-2, None] * h2[None, :]
            M[:-2, :, k] += c2 * Fk[2:, None] * h2[None, :]
    return M.reshape(n * 32, K)


def fit_qs(W, noise, pos, p0, w, t0, hyper, mask=None):
    """Weighted NNLS solve for charge profile; returns (chi2, q, model)."""
    n = len(pos)
    y = (W / noise[:, None]).reshape(-1)
    M = build_matrix(pos, p0, w, t0, hyper)
    Wt = np.repeat(1.0 / noise, 32)
    A = M * Wt[:, None]
    if mask is not None:
        mflat = mask.reshape(-1)
        A = A[mflat]; y = y[mflat]
    q, rnorm = nnls(A, y)
    chi2 = rnorm ** 2
    return chi2, q, M


def fit_event_plane(W, noise, pos, p0_init, w_init, t0_init, hyper=HYPER,
                    fix_p0w=None, refine=True):
    """Full fit. fix_p0w=(p0,w) to constrain track line (t0 still free)."""
    W = W.astype(np.float64)
    mask = W < SAT
    dof = int(mask.sum())

    if fix_p0w is not None:
        p0f, wf = fix_p0w

        def obj_t0(t0v):
            c, _, _ = fit_qs(W, noise, pos, p0f, wf, float(t0v), hyper, mask)
            return c
        # coarse t0 grid then golden refine
        t0s = np.arange(t0_init - 240, t0_init + 241, 40.0)
        cs = [obj_t0(t) for t in t0s]
        t0b = t0s[int(np.argmin(cs))]
        r = minimize(lambda v: obj_t0(v[0]), [t0b], method='Nelder-Mead',
                     options=dict(xatol=2.0, fatol=0.5, maxiter=60))
        chi, q, _ = fit_qs(W, noise, pos, p0f, wf, float(r.x[0]), hyper, mask)
        return dict(chi2=chi, dof=dof, p0=p0f, w=wf, t0=float(r.x[0]), q=q)

    def obj(v):
        p0v, wv, t0v = v
        c, _, _ = fit_qs(W, noise, pos, p0v, wv, t0v, hyper, mask)
        return c

    v0 = np.array([p0_init, w_init, t0_init])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.3, maxiter=300,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0], [0.8, 0, 0], [0, 0.004, 0], [0, 0, 50]])))
    if refine:
        r = minimize(obj, r.x, method='Nelder-Mead',
                     options=dict(xatol=5e-4, fatol=0.2, maxiter=200))
    chi, q, _ = fit_qs(W, noise, pos, r.x[0], r.x[1], r.x[2], hyper, mask)
    return dict(chi2=chi, dof=dof, p0=float(r.x[0]), w=float(r.x[1]),
                t0=float(r.x[2]), q=q, nfev=r.nfev)


def model_waveforms(pos, p0, w, t0, q, hyper=HYPER):
    M = build_matrix(pos, p0, w, t0, hyper)
    return (M @ q).reshape(len(pos), 32)


def init_guess(W, noise, pos, tan_ref, p0_ref):
    """Data-driven initial guess: amp-weighted centroid & simple lead times."""
    amax = W.max(axis=1)
    thr = np.maximum(6 * noise, 60)
    sel = amax > thr
    if sel.sum() == 0:
        sel = amax == amax.max()
    # crude t0: earliest 50% crossing among selected strips minus template lead
    leads = []
    for wv in W[sel]:
        ipk = int(np.argmax(wv))
        a = wv.max()
        for kk in range(1, ipk + 1):
            if wv[kk] >= 0.5 * a > wv[kk - 1]:
                leads.append(SNS * (kk - 1 + (0.5 * a - wv[kk - 1]) / (wv[kk] - wv[kk - 1])))
                break
    t0g = (min(leads) if leads else 400.0)
    w_g = 0.034 * tan_ref
    return p0_ref, w_g, t0g
