"""
The forward model.

For one plane of one event, the model says: charge ``q_k >= 0`` arrives in each
60 ns slice ``k`` of drift at transverse position ``p0 + w * u_k``; each slice's
charge is shared onto the strips by the geometric strip integral, then onto
their neighbours by the resistive kernel (``c1`` at delay ``tau_s`` to +-1
strip, ``c2`` at ``2 tau_s`` to +-2, scaled by ``kY`` on Y), and finally folded
with the measured per-plane impulse response. Fitting ``(p0, w, t0)`` with the
charge profile solved by NNLS at each step gives the track's position at the
mesh and its transverse speed ``w``; the angle is ``tan(theta) = w / v_drift``.

Because the neighbours' delayed copies are *in the model*, they stop being
contamination — which is exactly what a per-strip hit time cannot do.

This is the packaged form of ``forward_model2.py`` (model v2) and
``forward_model3.py`` (the vectorised fitter) from the R&D directory, with the
module-level calibration replaced by an explicit
:class:`~wft.calib.CalibrationBundle`. The numerics are unchanged and are
regression-tested against the R&D code (``tests/test_model_regression.py``).

Calibration is module-global state, set once per process by
``use_calibration()``; worker processes set it in their initializer.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls, minimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

from .calib import CalibrationBundle

# ---------------------------------------------------------------- module state
CAL: CalibrationBundle | None = None
TGRID: np.ndarray | None = None
TMPL: dict | None = None
GAIN: dict | None = None
DT_XY: dict = {}
PITCH = 0.78
SNS = 60.0
SAT = 3550.0
DT = 60.0                 # width of one charge/depth bin [ns]
NSAMP = 32
K = 18
TS = np.arange(NSAMP) * SNS
UK = (np.arange(K) + 0.5) * DT
HYPER: dict | None = None
# Fractional model-error term in the chi2 weights (0 = per-strip noise only,
# the pre-2026-07-29 behaviour). Settable via env so calibration worker
# processes and the reco driver pick it up consistently.
MODEL_FRAC = float(__import__('os').environ.get('WFT_MODEL_FRAC', '0'))

_smear_cache: dict = {}
_tt_cache: dict = {}
T0_STEP = 5.0             # t0 quantisation for the cached time tensors


def use_calibration(cal: CalibrationBundle) -> None:
    """Install a calibration bundle as the model's calibration."""
    global CAL, TGRID, TMPL, GAIN, DT_XY, PITCH, SNS, SAT, DT, K, UK, HYPER
    CAL = cal
    TGRID = np.asarray(cal.grid, float)
    TMPL = {p: np.asarray(cal.tmpl[p], float) for p in ('x', 'y')}
    GAIN = {p: np.asarray(cal.gain[p], float) for p in ('x', 'y')}
    DT_XY = dict(cal.dt_xy)
    PITCH = float(cal.pitch_mm)
    SNS = float(cal.sample_ns)
    SAT = float(cal.sat_adc)
    DT = float(cal.sample_ns)
    K = int(cal.n_depth_bins)
    UK = (np.arange(K) + 0.5) * DT
    HYPER = dict(cal.hyper)
    HYPER.setdefault('kY', 1.0)
    _smear_cache.clear()
    _tt_cache.clear()
    set_nsamp(NSAMP)


def set_nsamp(ns: int) -> None:
    """Adapt to a different DAQ window length (det4 mixes 32 and 37 samples)."""
    global NSAMP, TS
    NSAMP = int(ns)
    TS = np.arange(NSAMP) * SNS
    _smear_cache.clear()
    _tt_cache.clear()


def set_depth_bins(k: int) -> None:
    """Extend/shrink the charge basis (K=26 is used at low drift field, where
    the column takes longer than the nominal window to arrive)."""
    global K, UK
    K = int(k)
    UK = (np.arange(K) + 0.5) * DT
    _tt_cache.clear()


def set_depth_binning(k: int, dt_ns: float) -> None:
    """Switch to a coarser/finer charge basis covering the same drift span
    (used by the fast global pre-scan: K=9 x 120 ns instead of 18 x 60 ns).
    Does NOT clear the time-tensor cache — the cache key carries (K, DT), so
    both binnings coexist and switching back keeps warm tensors."""
    global K, UK, DT
    K = int(k)
    DT = float(dt_ns)
    UK = (np.arange(K) + 0.5) * DT


def _require_cal():
    if CAL is None:
        raise RuntimeError('wft.model has no calibration: call use_calibration()')


# ------------------------------------------------------------------- pieces
def _templates(plane: str, sigma_s: float):
    """(impulse response, dispersion-smeared impulse response) for a plane."""
    key = (plane, round(float(sigma_s), 1))
    if key not in _smear_cache:
        base = TMPL[plane]
        _smear_cache[key] = (base, gaussian_filter1d(base, max(sigma_s, 1.0) / 10.0))
    return _smear_cache[key]


def _tau_eff(plane: str, hyper: dict) -> float:
    """Per-plane sharing delay: tau_s, optionally scaled on Y by kTauY (the
    resistive/capacitive timescale need not be the same on both planes; only
    the amplitude asymmetry kY was modelled before)."""
    t = hyper['tau_s']
    return t * hyper.get('kTauY', 1.0) if plane == 'y' else t


def _tau2_delay(plane: str, hyper: dict, tau: float) -> float:
    """Arrival delay of the +-2-strip copy. Default 2*tau (the historical
    linear assumption). On the resistive-strip axis the spread is RC
    *diffusion*, where delay grows quadratically with distance — set
    'tau2_fac_y' (e.g. 4.0) to model that on Y."""
    fac = hyper.get('tau2_fac_y', 2.0) if plane == 'y' else 2.0
    return fac * tau


def _lp_copies(plane: str, hyper: dict):
    """Neighbour copy shapes for the RC-ladder ('share_lp') mode: the +-1 copy
    is the template through one lateral RC stage (exp kernel, time constant
    tau_s), the +-2 copy through two cascaded stages. Measured directly on
    near-vertical det3 tracks (bench/rc_line_step3.py): the copy is a
    low-passed template — broader, ~100 ns later, with a long late tail —
    not the delayed+smeared template of the historical kernel."""
    tau = _tau_eff(plane, hyper)
    sig = (hyper.get('sigma_sY', hyper['sigma_s']) if plane == 'y'
           else hyper['sigma_s'])
    key = ('lp', plane, tau, round(sig, 1))
    hit = _smear_cache.get(key)
    if hit is not None:
        return hit
    base = TMPL[plane]
    step = float(TGRID[1] - TGRID[0])
    n = max(3, int(np.ceil(6 * tau / step)))
    tg = np.arange(n) * step
    e = np.exp(-tg / max(tau, 1e-3))
    e /= e.sum()
    sm1 = np.convolve(base, e)[:len(base)]
    sm2 = np.convolve(sm1, e)[:len(base)]
    if sig > 1.0:
        sm1 = gaussian_filter1d(sm1, sig / step)
        sm2 = gaussian_filter1d(sm2, sig / step)
    _smear_cache[key] = (sm1, sm2)
    return sm1, sm2


def _time_tensors(plane: str, t0q: float, hyper: dict):
    """(K, NSAMP) impulse responses of each depth bin, cached on a 5 ns t0 grid."""
    tau = _tau_eff(plane, hyper)
    d2 = _tau2_delay(plane, hyper, tau)
    lp = bool(hyper.get('share_lp'))
    sig_key = (hyper.get('sigma_sY', hyper['sigma_s']) if plane == 'y'
               else hyper['sigma_s'])
    key = (plane, t0q, tau, d2, round(sig_key, 1), NSAMP, K, DT, lp)
    hit = _tt_cache.get(key)
    if hit is not None:
        return hit
    base = TS[:, None] - (t0q + UK[None, :])          # (NSAMP, K)
    if lp:
        tmpl = TMPL[plane]
        sm1, sm2 = _lp_copies(plane, hyper)
        H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
        H1 = np.interp(base, TGRID, sm1, left=0, right=0)
        H2 = np.interp(base, TGRID, sm2, left=0, right=0)
    else:
        tmpl, sm = _templates(plane, hyper['sigma_s'])
        H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
        H1 = np.interp(base - tau, TGRID, sm, left=0, right=0)
        H2 = np.interp(base - d2, TGRID, sm, left=0, right=0)
    if len(_tt_cache) > 4096:
        _tt_cache.clear()
    _tt_cache[key] = (H0, H1, H2)
    return H0, H1, H2


def strip_fractions(pos, p0, w, sigma_p0, Dp):
    """Fraction of each depth bin's charge landing on each strip: the bin's
    transverse extent (p0 + w*u over the bin) smeared by the initial cloud size
    and diffusion, integrated over the strip pitch."""
    ua = np.arange(K) * DT
    ub = ua + DT
    pa, pb = p0 + w * ua, p0 + w * ub
    pc = 0.5 * (pa + pb)
    half = 0.5 * np.abs(pb - pa)
    sig = np.sqrt(sigma_p0 ** 2 + Dp ** 2 * UK + half ** 2 / 3.0)
    z = 1.0 / (np.sqrt(2) * sig)[None, :]
    hi = (pos[:, None] + PITCH / 2 - pc[None, :]) * z
    lo = (pos[:, None] - PITCH / 2 - pc[None, :]) * z
    return 0.5 * (erf(hi) - erf(lo))                  # (n_strip, K)


def build_matrix(plane, pos, p0, w, t0, hyper):
    """Design matrix: column k = the (strip, sample) waveform produced by unit
    charge in depth bin k, sharing and impulse response included."""
    t0q = round(t0 / T0_STEP) * T0_STEP
    if abs(t0 - t0q) > 1e-9:
        base = TS[:, None] - (t0 + UK[None, :])
        if hyper.get('share_lp'):
            sm1, sm2 = _lp_copies(plane, hyper)
            H0 = np.interp(base, TGRID, TMPL[plane], left=0, right=0)
            H1 = np.interp(base, TGRID, sm1, left=0, right=0)
            H2 = np.interp(base, TGRID, sm2, left=0, right=0)
        else:
            tmpl, sm = _templates(plane, hyper['sigma_s'])
            tau = _tau_eff(plane, hyper)
            H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
            H1 = np.interp(base - tau, TGRID, sm, left=0, right=0)
            H2 = np.interp(base - _tau2_delay(plane, hyper, tau), TGRID, sm,
                           left=0, right=0)
    else:
        H0, H1, H2 = _time_tensors(plane, t0q, hyper)
    kY = hyper.get('kY', 1.0) if plane == 'y' else 1.0
    c1, c2 = hyper['c1'] * kY, hyper['c2'] * kY
    # aY (aX): left/right asymmetry of the sharing kernel — copy toward higher
    # strip positions weighted (1+a), toward lower (1-a). 0 = symmetric.
    a = hyper.get('aY' if plane == 'y' else 'aX', 0.0)
    if plane == 'y':
        sp0 = hyper.get('sigma_p0Y', hyper['sigma_p0'])
        dp = hyper.get('DpY', hyper['Dp'])
    else:
        sp0, dp = hyper['sigma_p0'], hyper['Dp']
    F = strip_fractions(pos, p0, w, sp0, dp)
    n = len(pos)
    M = np.empty((n, NSAMP, K))
    np.multiply(F[:, None, :], H0[None, :, :], out=M)
    Fs = np.zeros_like(F)
    Fs[1:] = (1.0 + a) * F[:-1]
    Fs[:-1] += (1.0 - a) * F[1:]
    M += (c1 * Fs)[:, None, :] * H1[None, :, :]
    if c2 > 0:
        Fs2 = np.zeros_like(F)
        Fs2[2:] = (1.0 + a) * F[:-2]
        Fs2[:-2] += (1.0 - a) * F[2:]
        M += (c2 * Fs2)[:, None, :] * H2[None, :, :]
    return M.reshape(n * NSAMP, K)


def prep_plane(P, plane):
    """Gain-correct one plane's waveform window. P: dict with W (nstrip, nsamp),
    pos [mm], noise per strip, ch (channel numbers).
    Returns (W, noise, pos, saturation mask)."""
    W = np.asarray(P['W'], dtype=np.float64).copy()
    g = GAIN[plane][np.asarray(P['ch'], dtype=int)]
    W /= g[:, None]
    noise = np.maximum(np.asarray(P['noise'], dtype=np.float64), 3.0) / g
    sat = np.asarray(P['W'], dtype=np.float64) >= SAT
    return W, noise, np.asarray(P['pos'], dtype=np.float64), sat


def sample_weights(W, noise):
    """Per-sample chi2 weights (1/sigma). With MODEL_FRAC > 0, a fractional
    model-error term is added in quadrature so that percent-level template
    mismatch on bright samples stops dominating the fit; at 0 this is exactly
    the production per-strip-noise weighting."""
    if MODEL_FRAC <= 0:
        return np.repeat(1.0 / noise, NSAMP)
    sig = np.sqrt(noise[:, None] ** 2 +
                  (MODEL_FRAC * np.maximum(W, 0.0)) ** 2)
    return (1.0 / sig).reshape(-1)


def chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=True,
               snap_t0=True):
    """chi2 of the model at (p0, w, t0), with the charge profile profiled out by
    NNLS. Saturated samples are censored: excluded from the fit, and penalised
    only if the model falls *below* the clipped value."""
    if snap_t0:
        t0 = round(t0 / T0_STEP) * T0_STEP
    M = build_matrix(plane, pos, p0, w, t0, hyper)
    ok = ~sat.reshape(-1)
    if MODEL_FRAC <= 0:
        # production path — kept expression-for-expression identical to the
        # regression-tested numerics (division vs reciprocal matters at ulp
        # level, and the NM trajectory is chaotic in the last ulp)
        Wt = np.repeat(1.0 / noise, NSAMP)
        A = (M * Wt[:, None])[ok]
        y = (W / noise[:, None]).reshape(-1)[ok]
    else:
        Wt = sample_weights(W, noise)
        A = (M * Wt[:, None])[ok]
        y = (W.reshape(-1) * Wt)[ok]
    try:
        q, rn = nnls(A, y, maxiter=50 * K)
    except Exception:
        return np.inf, None
    chi = rn * rn
    if censor and sat.any():
        model = (M @ q).reshape(W.shape)
        if MODEL_FRAC <= 0:
            pen = np.maximum(0.0, W[sat] - model[sat]) / np.repeat(
                noise, NSAMP).reshape(W.shape)[sat]
        else:
            pen = (np.maximum(0.0, W[sat] - model[sat]) *
                   Wt.reshape(W.shape)[sat])
        chi += float((pen ** 2).sum())
    return chi, q


def model_waveforms(plane, pos, p0, w, t0, q, hyper):
    return (build_matrix(plane, pos, p0, w, t0, hyper) @ q).reshape(len(pos), NSAMP)


# --------------------------------------------------------------------- fits
def fit_plane_raw(P, plane, p0_init, w_init, t0_init, hyper=None, fix_p0w=None):
    """Two-stage fit: coarse (p0, w, t0) grid, then Nelder-Mead. With
    ``fix_p0w=(p0, w)`` only t0 and the charge profile are fitted — that is the
    'ref-pinned' configuration used for calibration and for the chi2(v) scan."""
    _require_cal()
    hyper = hyper or HYPER
    W, noise, pos, sat = prep_plane(P, plane)
    dof = int((~sat).sum())

    if fix_p0w is not None:
        p0f, wf = fix_p0w
        grid = np.arange(t0_init - 240, t0_init + 241, 20.0)
        cs = [chi2_plane(plane, W, noise, pos, sat, p0f, wf, t, hyper)[0]
              for t in grid]
        j = int(np.argmin(cs))
        g2 = np.arange(grid[j] - 20, grid[j] + 21, T0_STEP)
        cs2 = [chi2_plane(plane, W, noise, pos, sat, p0f, wf, t, hyper)[0]
               for t in g2]
        t0b = float(g2[int(np.argmin(cs2))])
        chi, q = chi2_plane(plane, W, noise, pos, sat, p0f, wf, t0b, hyper)
        return dict(chi2=chi, dof=dof, p0=p0f, w=wf, t0=t0b, q=q, nfev=len(cs) + len(cs2))

    nfev = 0
    best = (np.inf, p0_init, w_init, t0_init)
    for t0 in np.arange(t0_init - 120, t0_init + 121, 40.0):
        for dp in (-0.8, 0.0, 0.8):
            for dw in (-2.4e-3, -0.8e-3, 0.0, 0.8e-3, 2.4e-3):
                c, _ = chi2_plane(plane, W, noise, pos, sat,
                                  p0_init + dp, w_init + dw, t0, hyper)
                nfev += 1
                if c < best[0]:
                    best = (c, p0_init + dp, w_init + dw, t0)

    def obj(v):
        return chi2_plane(plane, W, noise, pos, sat, v[0], v[1], v[2], hyper,
                          snap_t0=False)[0]

    v0 = np.array(best[1:])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.3, maxiter=140,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0], [0.4, 0, 0], [0, 1.5e-3, 0],
                                   [0, 0, 20]])))
    chi, q = chi2_plane(plane, W, noise, pos, sat, r.x[0], r.x[1], r.x[2],
                        hyper, snap_t0=False)
    return dict(chi2=chi, dof=dof, p0=float(r.x[0]), w=float(r.x[1]),
                t0=float(r.x[2]), q=q, nfev=nfev + r.nfev)


def fit_joint(evx, evy, ftst_diff, p0x_i, wx_i, p0y_i, wy_i, t0_i, hyper=None):
    """Joint two-plane fit: one shared charge profile (per-plane scale), t0
    tied through the measured FEU offset. Its value is stabilising a
    near-vertical plane, where timing carries no slope information."""
    _require_cal()
    hyper = hyper or HYPER
    dt = DT_XY.get(int(ftst_diff), -18.8)
    Wx, nx, px_, sx = prep_plane(evx, 'x')
    Wy, ny, py_, sy = prep_plane(evy, 'y')
    dof = int((~sx).sum() + (~sy).sum())

    def solve(p0x, wx, p0y, wy, t0):
        Mx = build_matrix('x', px_, p0x, wx, t0, hyper)
        My = build_matrix('y', py_, p0y, wy, t0 - dt, hyper)
        okx, oky = ~sx.reshape(-1), ~sy.reshape(-1)
        Ax = (Mx * np.repeat(1.0 / nx, NSAMP)[:, None])[okx]
        Ay = (My * np.repeat(1.0 / ny, NSAMP)[:, None])[oky]
        yx = (Wx / nx[:, None]).reshape(-1)[okx]
        yy = (Wy / ny[:, None]).reshape(-1)[oky]
        alpha = 1.0
        my = None
        for _ in range(2):
            try:
                q, _rn = nnls(np.vstack([Ax, alpha * Ay]),
                              np.concatenate([yx, yy]), maxiter=50 * K)
            except Exception:
                return np.inf, None, 1.0
            my = Ay @ q
            den = float(my @ my)
            alpha = float(my @ yy / den) if den > 0 else 1.0
        chi = float(((Ax @ q - yx) ** 2).sum() + ((alpha * my - yy) ** 2).sum())
        return chi, q, alpha

    v0 = np.array([p0x_i, wx_i, p0y_i, wy_i, t0_i])
    r = minimize(lambda v: solve(*v)[0], v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.5, maxiter=500,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0, 0, 0], [0.8, 0, 0, 0, 0],
                                   [0, 0.004, 0, 0, 0], [0, 0, 0.8, 0, 0],
                                   [0, 0, 0, 0.004, 0], [0, 0, 0, 0, 50]])))
    r = minimize(lambda v: solve(*v)[0], r.x, method='Nelder-Mead',
                 options=dict(xatol=5e-4, fatol=0.3, maxiter=300))
    chi, q, alpha = solve(*r.x)
    return dict(chi2=chi, dof=dof, p0x=float(r.x[0]), wx=float(r.x[1]),
                p0y=float(r.x[2]), wy=float(r.x[3]), t0=float(r.x[4]),
                q=q, alpha=alpha)


def init_guess(P, plane, tan_seed=0.0, p0_seed=None, v_drift=None):
    """Starting point for the fit. Deliberately crude and reference-free: the
    brightest strip for position, the earliest half-maximum crossing for t0."""
    W = np.asarray(P['W'], dtype=np.float64)
    noise = np.maximum(np.asarray(P['noise'], dtype=np.float64), 3.0)
    pos = np.asarray(P['pos'], dtype=np.float64)
    amax = W.max(axis=1)
    if p0_seed is None:
        p0_seed = float(pos[int(np.argmax(amax))])
    sel = amax > np.maximum(6 * noise, 60)
    if sel.sum() == 0:
        sel = amax == amax.max()
    leads = []
    for wv in W[sel]:
        ipk = int(np.argmax(wv))
        a = wv.max()
        for kk in range(1, ipk + 1):
            if wv[kk] >= 0.5 * a > wv[kk - 1]:
                leads.append(SNS * (kk - 1 + (0.5 * a - wv[kk - 1]) /
                                    (wv[kk] - wv[kk - 1])))
                break
    t0g = min(leads) if leads else 400.0
    v = v_drift if v_drift is not None else (CAL.v_drift if CAL else 36.6)
    return float(p0_seed), float(tan_seed * v * 1e-3), float(t0g)
