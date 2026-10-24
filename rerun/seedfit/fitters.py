"""fitters.py -- candidate single-track plane fitters for the starting-angle study (rerun/seedfit, 2026-10-24).

All take (P, plane, t0p) with the model already configured (wm.use_calibration ...) and return a dict with
p0, w [mm/ns], t0, chi2, nfev.  'prod' is wft.model.fit_plane_raw from the blind init_guess start (what the
pass uses); the others try to find the global minimum without any reference information.
"""
import numpy as np
from scipy.optimize import nnls, minimize

from wft import model as wm

_B = {}


def _basis(plane, ns, step):
    key = (plane, ns, step, id(wm.TMPL))
    if key not in _B:
        tp = 'x' if wm.HYPER.get('rc_tmpl', 'x') in ('x', 1, 1.0) else plane
        ts = np.arange(ns) * wm.SNS
        ta = np.arange(-300, ns * wm.SNS, step)
        _B[key] = (ta, np.stack([np.interp(ts - a, wm.TGRID, wm.TMPL[tp], left=0, right=0) for a in ta], 1))
    return _B[key]


def arrival_image(W, noise, sat, plane, step=60.0, nsig=4.0):
    """Per-strip time deconvolution with the electronics template only (no sharing kernel): Q[strip, arrival bin]."""
    ta, Bm = _basis(plane, W.shape[1], step)
    Q = np.zeros((W.shape[0], len(ta)))
    for i in np.flatnonzero(W.max(1) > nsig * noise):
        g = ~sat[i]
        if g.sum() < 4:
            continue
        try:
            Q[i], _ = nnls(Bm[g] / noise[i], W[i, g] / noise[i], maxiter=3000)
        except RuntimeError:
            pass
    return ta, Q


def centroid(W, noise, pos, sat, plane):
    ta, Q = arrival_image(W, noise, sat, plane)
    q = Q.sum()
    if q <= 0:
        return None
    return float((Q * pos[:, None]).sum() / q), float((Q * ta[None, :]).sum() / q)


def _start(P, plane, t0p):
    p0i, wi, t0g = wm.init_guess(P, plane, 0.0, None, wm.CAL.v_drift)
    t0 = float(t0p) if t0p is not None else t0g
    prior = (t0, 5.0) if t0p is not None else None
    return p0i, wi, t0, prior


def fit_prod(P, plane, t0p, **_):
    p0i, wi, t0, prior = _start(P, plane, t0p)
    return wm.fit_plane_raw(P, plane, p0i, wi, t0, t0_prior=prior)


def fit_seeded(P, plane, t0p, tr=None, ref=None, **_):
    p0i, wi, t0, prior = _start(P, plane, t0p)
    p0i, wi, _ = wm.init_guess(P, plane, tr, ref, wm.CAL.v_drift)
    return wm.fit_plane_raw(P, plane, p0i, wi, t0, t0_prior=prior)


def fit_pivot(P, plane, t0p, wgrid=np.arange(-32, 32.1, 2.0), nref=2, **_):
    """Scan w with the line through the deconvolved charge centroid, refine the best wells with fit_plane_raw."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    p0i, wi, t0, prior = _start(P, plane, t0p)
    c = centroid(W, noise, pos, sat, plane)
    if c is None:
        return fit_prod(P, plane, t0p)
    xc, tc = c
    chis = np.array([wm.chi2_plane(plane, W, noise, pos, sat, xc - w * 1e-3 * (tc - t0), w * 1e-3, t0, wm.HYPER,
                                   t0_prior=prior)[0] for w in wgrid])
    n = len(wgrid)
    lm = [i for i in range(n) if (i == 0 or chis[i] <= chis[i - 1]) and (i == n - 1 or chis[i] <= chis[i + 1])]
    best = None
    for i in sorted(lm, key=lambda i: chis[i])[:nref]:
        w = wgrid[i] * 1e-3
        r = wm.fit_plane_raw(P, plane, xc - w * (tc - t0), w, t0, t0_prior=prior)
        if best is None or r['chi2'] < best['chi2']:
            best = r
    best['nfev'] += n
    return best


def _local_pivot(W, noise, pos, sat, plane, xc, tc, w, t0, prior, rounds=2, tstep=5.0):
    """Alternate: (x_c, w) by Nelder-Mead at fixed t0 (smooth there), then t0 on its 5 ns grid at a fixed line."""
    nfev = 0
    H = wm.HYPER

    def chi(xc_, w_, t0_):
        return wm.chi2_plane(plane, W, noise, pos, sat, xc_ - w_ * (tc - t0_), w_, t0_, H, t0_prior=prior)[0]

    c = chi(xc, w, t0); nfev += 1
    for _ in range(rounds):
        r = minimize(lambda v: chi(v[0], v[1], t0), [xc, w], method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=0.5, maxiter=80,
                                  initial_simplex=[[xc, w], [xc + 0.3, w], [xc, w + 1.2e-3]]))
        nfev += r.nfev
        if r.fun < c:
            xc, w, c = float(r.x[0]), float(r.x[1]), float(r.fun)
        tb = t0
        for dt in (-20, -15, -10, -5, 5, 10, 15, 20):
            cc = chi(xc, w, t0 + dt); nfev += 1
            if cc < c:
                c, tb = cc, t0 + dt
        t0 = tb
    return xc, w, t0, c, nfev


def fit_pivot2(P, plane, t0p, wgrid=np.arange(-32, 32.1, 2.0), tgrid=(-60, -30, 0, 30, 60), nref=3, **_):
    """Global (w, t0) scan with the line through the deconvolved centroid, then pivot-coordinate local refinement
    of the best nref wells."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    p0i, wi, t0r, prior = _start(P, plane, t0p)
    c = centroid(W, noise, pos, sat, plane)
    if c is None:
        return fit_prod(P, plane, t0p)
    xc, tc = c
    S = np.array([[wm.chi2_plane(plane, W, noise, pos, sat, xc - w * 1e-3 * (tc - (t0r + dt)), w * 1e-3, t0r + dt,
                                 wm.HYPER, t0_prior=prior)[0] for dt in tgrid] for w in wgrid])
    nfev = S.size
    prof = S.min(1)
    n = len(wgrid)
    lm = [i for i in range(n) if (i == 0 or prof[i] <= prof[i - 1]) and (i == n - 1 or prof[i] <= prof[i + 1])]
    best = None
    for i in sorted(lm, key=lambda i: prof[i])[:nref]:
        t0 = t0r + tgrid[int(np.argmin(S[i]))]
        x, w, t, cc, nf = _local_pivot(W, noise, pos, sat, plane, xc, tc, wgrid[i] * 1e-3, t0, prior)
        nfev += nf
        if best is None or cc < best[3]:
            best = (x, w, t, cc)
    x, w, t, cc = best
    return dict(p0=x - w * (tc - t), w=w, t0=t, chi2=cc, nfev=nfev)


FITTERS = dict(prod=fit_prod, seeded=fit_seeded, pivot=fit_pivot, pivot2=fit_pivot2)


def centroid_fast(W, noise, pos, sat, plane):
    """(x_c, t_c) without deconvolution: convolution adds means, so the charge-weighted time centroid of the
    waveforms minus the template's own time centroid is the arrival-time centroid.  Strips above 4 sigma only;
    each strip's window integral is its weight (the template has a non-zero area)."""
    tp = 'x' if wm.HYPER.get('rc_tmpl', 'x') in ('x', 1, 1.0) else plane
    tm = wm.TMPL[tp]
    t_tmpl = float(np.sum(wm.TGRID * tm) / np.sum(tm))
    rows = W.max(1) > 4 * noise
    if not rows.any():
        return None
    Wr = W[rows]
    ts = np.arange(W.shape[1]) * wm.SNS
    q = Wr.sum(1)
    Q = q.sum()
    if Q <= 0:
        return None
    return float((q * pos[rows]).sum() / Q), float((Wr * ts[None, :]).sum() / Q) - t_tmpl


def fit_pivot_fast(P, plane, t0p, wgrid=np.arange(-32, 32.1, 2.0), nref=2, **_):
    """fit_pivot with the undeconvolved centroid."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    p0i, wi, t0, prior = _start(P, plane, t0p)
    c = centroid_fast(W, noise, pos, sat, plane)
    if c is None:
        return fit_prod(P, plane, t0p)
    xc, tc = c
    chis = np.array([wm.chi2_plane(plane, W, noise, pos, sat, xc - w * 1e-3 * (tc - t0), w * 1e-3, t0, wm.HYPER,
                                   t0_prior=prior)[0] for w in wgrid])
    n = len(wgrid)
    lm = [i for i in range(n) if (i == 0 or chis[i] <= chis[i - 1]) and (i == n - 1 or chis[i] <= chis[i + 1])]
    best = None
    for i in sorted(lm, key=lambda i: chis[i])[:nref]:
        w = wgrid[i] * 1e-3
        r = wm.fit_plane_raw(P, plane, xc - w * (tc - t0), w, t0, t0_prior=prior)
        if best is None or r['chi2'] < best['chi2']:
            best = r
    best['nfev'] += n
    return best


FITTERS['pivot_fast'] = fit_pivot_fast


def _reco_fit(P, plane, t0p, shear=False, two_sided=False):
    """wft.reco's production plane fit: _global_start (p0, w grid around the charge centre) + fit_plane_raw
    [+ the two-sided re-search].  P0_SHEAR anchors the scan's lines half a drift column from the centre."""
    from wft import reco as wr
    p0i, wi, t0, prior = _start(P, plane, t0p)
    old = wr.P0_SHEAR
    wr.P0_SHEAR = shear
    try:
        p0s, ws, t0s = wr._global_start(P, plane, p0i, t0, wm.HYPER, t0_prior=prior)
    finally:
        wr.P0_SHEAR = old
    r = wm.fit_plane_raw(P, plane, p0s, ws, t0s, t0_prior=prior)
    if two_sided:
        r = wr._two_sided(P, plane, r, wm.CAL, wm.HYPER, t0_prior=prior)
    return r


FITTERS['reco'] = lambda P, plane, t0p, **k: _reco_fit(P, plane, t0p)
FITTERS['reco_shear'] = lambda P, plane, t0p, **k: _reco_fit(P, plane, t0p, shear=True)
FITTERS['reco_2s'] = lambda P, plane, t0p, **k: _reco_fit(P, plane, t0p, two_sided=True)


def fit_reco_pivot(P, plane, t0p, **_):
    """The packaged version: wft.reco._pivot_fit (PIVOT_START), falling back to the production start."""
    from wft import reco as wr
    p0i, wi, t0, prior = _start(P, plane, t0p)
    r = wr._pivot_fit(P, plane, wm.HYPER, t0, t0_prior=prior)
    return r if r is not None else _reco_fit(P, plane, t0p)


FITTERS['reco_pivot'] = fit_reco_pivot
