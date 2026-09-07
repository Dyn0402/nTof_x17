"""
The forward model.

For one plane of one event, the model says: charge ``q_k >= 0`` arrives in each
60 ns slice ``k`` of drift at transverse position ``p0 + w * u_k``; each slice's
charge is shared onto the strips by the geometric strip integral, then onto
their neighbours by the resistive kernel (scaled by ``kY`` on Y), and finally
folded with the measured per-plane impulse response. Fitting ``(p0, w, t0)``
with the charge profile solved by NNLS at each step gives the track's position
at the mesh and its transverse speed ``w``; the angle is
``tan(theta) = w / v_drift``.

The resistive kernel has two forms (``share_mode`` on the bundle):

``delay``   the original parameterisation: a copy of the impulse response,
            amplitude ``c1``, delayed by ``tau_s`` to the +-1 strips (``c2``
            at ``2 tau_s`` to +-2), smeared by ``sigma_s``.
``lp``      the H4-beam-measured structure (M70V_FLAT_ANALYSIS.md §3,
            RAW_RUN71_REANALYSIS §4): the neighbour sees an RC-*dispersed*
            copy — the impulse response convolved with a one-pole low-pass of
            time constant ``tau_s``, cascaded once more for +-2. The copy
            peaks essentially WITH the central strip (shifted only by the RC
            rise, ~+30-60 ns for tau_s of a few hundred ns) and carries the
            long tail the delayed-copy form cannot represent without an
            unphysical ``sigma_p0``. ``c1``/``c2`` keep their meaning as the
            copies' amplitude (area) fractions.

Because the neighbours' delayed copies are *in the model*, they stop being
contamination — which is exactly what a per-strip hit time cannot do.

PRIOR ART (``REFERENCES.md`` in this package has the annotated list). This model
was re-derived from our own data, not lifted from a paper, but it is not new:
the resistive layer as a distributed RC network whose neighbour signal carries
position is Dixit et al., NIM A 518 (2004) 721; the resistive-*strip*
transmission line that makes the copy a cascaded one-pole — and that forces
``c2 < c1`` — is Galan et al., JINST 7 (2012) C04009; and fitting neighbouring
channels *simultaneously* against a spreading-times-electronics model is the
T2K ND280 ERAM analysis, Attie et al., NIM A 1056 (2023) 168534. What is ours
is solving the drift-depth charge profile inside that fit, so the result is a
micro-TPC rather than a sharpened centroid.

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

from .calib import CalibrationBundle, check_kernel_ordering

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
SHARE_MODE = 'delay'      # 'delay' | 'lp' — see module docstring
DEAD: dict = {}           # per-plane dead-channel arrays (T1.3); see prep_plane

_smear_cache: dict = {}
_lp_cache: dict = {}
_tt_cache: dict = {}
T0_STEP = 5.0             # t0 quantisation for the cached time tensors


def use_calibration(cal: CalibrationBundle) -> None:
    """Install a calibration bundle as the model's calibration."""
    global CAL, TGRID, TMPL, GAIN, DT_XY, PITCH, SNS, SAT, DT, K, UK, HYPER, \
        SHARE_MODE, DEAD
    check_kernel_ordering(cal.hyper, where=f'bundle {cal.detector}/{cal.run_key}')
    CAL = cal
    DEAD = {p: np.asarray(sorted(ch), dtype=int)
            for p, ch in getattr(cal, 'dead', {}).items() if len(ch)}
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
    SHARE_MODE = getattr(cal, 'share_mode', 'delay') or 'delay'
    _smear_cache.clear()
    _lp_cache.clear()
    _tt_cache.clear()
    set_nsamp(NSAMP)


def set_share_mode(mode: str) -> None:
    """Override the sharing-kernel form ('delay' | 'lp')."""
    global SHARE_MODE
    if mode not in ('delay', 'lp'):
        raise ValueError(f'share_mode must be delay|lp, got {mode!r}')
    SHARE_MODE = mode
    _lp_cache.clear()
    _tt_cache.clear()


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


def _lp_copies(plane: str, sigma_s: float, tau_s: float):
    """(extended grid, once-, twice-RC-convolved smeared template) for the
    ``lp`` share mode. The template grid stops at +1.4 us, but an RC tail with
    tau of a few hundred ns is still alive there, so the template is
    zero-padded to +6 us before convolving. Discrete one-pole with the grid
    step preserves the area, so c1/c2 stay area fractions."""
    key = (plane, round(float(sigma_s), 1), round(float(tau_s), 1))
    hit = _lp_cache.get(key)
    if hit is not None:
        return hit
    _, sm = _templates(plane, sigma_s)
    step = float(TGRID[1] - TGRID[0])
    n_pad = max(0, int(round((6000.0 - TGRID[-1]) / step)))
    ge = np.concatenate([TGRID, TGRID[-1] + step * (1 + np.arange(n_pad))])
    x = np.concatenate([sm, np.zeros(n_pad)])
    a = np.exp(-step / max(float(tau_s), 1.0))
    l1 = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = acc * a + x[i] * (1.0 - a)
        l1[i] = acc
    l2 = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = acc * a + l1[i] * (1.0 - a)
        l2[i] = acc
    if len(_lp_cache) > 256:
        _lp_cache.clear()
    _lp_cache[key] = (ge, l1, l2)
    return ge, l1, l2


def _copy_responses(plane: str, base: np.ndarray, hyper: dict):
    """(H1, H2) neighbour-copy responses on the (NSAMP, K) time offsets in
    ``base``, per the active SHARE_MODE.

    ``tau_y_fac`` scales the RC constant on the Y plane only: the resistive
    strips run along y, so Y's copy is slower as well as stronger (measured
    directly, tau_X 230 / tau_Y 410 ns). NOTE the key is deliberately NOT the
    bundles' ``kTauY``: that constant belongs to the archived RC-ladder
    representation, and switching it on under this kernel form regressed Y
    badly (sigma_Y 1.14 -> 1.57 deg, bench 2026-08-12) — the F19 lesson, RC
    constants are representation-dependent. A per-plane tau must enter here
    through a recalibration that fits/validates ``tau_y_fac`` under THIS
    kernel; no existing bundle carries the key, so nothing changes silently."""
    tau = hyper['tau_s'] * (hyper.get('tau_y_fac', 1.0) if plane == 'y' else 1.0)
    if SHARE_MODE == 'lp':
        ge, l1, l2 = _lp_copies(plane, hyper['sigma_s'], tau)
        H1 = np.interp(base, ge, l1, left=0, right=0)
        H2 = np.interp(base, ge, l2, left=0, right=0)
    else:
        _, sm = _templates(plane, hyper['sigma_s'])
        H1 = np.interp(base - tau, TGRID, sm, left=0, right=0)
        H2 = np.interp(base - 2 * tau, TGRID, sm, left=0, right=0)
    return H1, H2


def _time_tensors(plane: str, t0q: float, hyper: dict):
    """(K, NSAMP) impulse responses of each depth bin, cached on a 5 ns t0 grid."""
    key = (plane, t0q, hyper['tau_s'], round(hyper['sigma_s'], 1), NSAMP, K,
           SHARE_MODE, hyper.get('tau_y_fac', 1.0) if plane == 'y' else 1.0)
    hit = _tt_cache.get(key)
    if hit is not None:
        return hit
    tmpl, _sm = _templates(plane, hyper['sigma_s'])
    base = TS[:, None] - (t0q + UK[None, :])          # (NSAMP, K)
    H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
    H1, H2 = _copy_responses(plane, base, hyper)
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
        tmpl, _sm = _templates(plane, hyper['sigma_s'])
        base = TS[:, None] - (t0 + UK[None, :])
        H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
        H1, H2 = _copy_responses(plane, base, hyper)
    else:
        H0, H1, H2 = _time_tensors(plane, t0q, hyper)
    # per-plane amplitude on the discrete (RC) sharing kernel. kY is the
    # long-standing Y multiplier; cX (default 1, i.e. no change) scales the X
    # side — the resistive strips run along y, so X cannot have resistive
    # sharing and its +-1 copy should be diffusion (F6): cX = 0 with Dp
    # refit is the physically motivated test arm (handoff T1.2).
    kY = hyper.get('kY', 1.0) if plane == 'y' else hyper.get('cX', 1.0)
    c1, c2 = hyper['c1'] * kY, hyper['c2'] * kY
    r = hyper.get('c2_over_c1')
    if r is not None:
        # SLAVE c2 TO c1.  The +-2 strip is reached only through the +-1
        # strip, so c2 < c1 always -- yet the shipped bundles carry c2 > c1 on
        # every detector (det3 1.14, det2 1.53, det7 1.75, det4 2.12).  That is
        # not a bound artefact: the ref-pinned cosmic chi2 is genuinely flat in
        # this direction (sloppy-mode analysis 2026-08-17), so the fit is free
        # to walk there and does.  The H4 head-on beam data measures the ratio
        # directly and model-free, at 0.45 +- 0.03 over a 2.6x range of drift
        # field (sps_beam_test_26/analysis/sharing_kernel); near-vertical bench
        # cosmics give 0.63 +- 0.10 on det3.  Pinning it costs one hyper and
        # makes the ordering structural.
        # Applied to the BASE hypers, before the per-plane kY/cX scaling, so
        # the ratio is plane-independent.  No existing bundle carries the key.
        c2 = float(r) * c1
    F = strip_fractions(pos, p0, w, hyper['sigma_p0'], hyper['Dp'])
    n = len(pos)
    M = np.empty((n, NSAMP, K))
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
    return M.reshape(n * NSAMP, K)


def prep_plane(P, plane):
    """Gain-correct one plane's waveform window. P: dict with W (nstrip, nsamp),
    pos [mm], noise per strip, ch (channel numbers).
    Returns (W, noise, pos, censor mask).

    Dead channels (bundle ``dead``, T1.3) are censored samples — a broken
    connection reads baseline, not zero charge, so their rows are excluded
    from the fit and the dof exactly like saturated samples, and their noise
    is inflated so the one-sided saturation penalty cannot pull on them
    either: no information in either direction."""
    W = np.asarray(P['W'], dtype=np.float64).copy()
    ch = np.asarray(P['ch'], dtype=int)
    g = GAIN[plane][ch]
    W /= g[:, None]
    noise = np.maximum(np.asarray(P['noise'], dtype=np.float64), 3.0) / g
    sat = np.asarray(P['W'], dtype=np.float64) >= SAT
    d = DEAD.get(plane)
    if d is not None and len(d):
        rows = np.isin(ch, d)
        if rows.any():
            sat[rows] = True
            noise[rows] = 1e9
    return W, noise, np.asarray(P['pos'], dtype=np.float64), sat


def chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=True,
               snap_t0=True, t0_prior=None):
    """chi2 of the model at (p0, w, t0), with the charge profile profiled out by
    NNLS. Saturated samples are censored: excluded from the fit, and penalised
    only if the model falls *below* the clipped value.

    ``t0_prior=(t0_pred, sigma)`` adds a Gaussian penalty pinning t0 to an
    external per-event prediction (the scintillator trigger through the ftst
    phase). The chi2 surface has near-degenerate minima 60 ns (one depth bin)
    apart — the profile shifts a bin and p0 slides by w*60 — and only ~35 % of
    free fits land in the physical one (T1.1 gate, 2026-08-11); the prior is
    what selects it."""
    if snap_t0:
        t0 = round(t0 / T0_STEP) * T0_STEP
    M = build_matrix(plane, pos, p0, w, t0, hyper)
    ok = ~sat.reshape(-1)
    if not ok.any():
        return np.inf, None
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
        pen = np.maximum(0.0, W[sat] - model[sat]) / np.repeat(
            noise, NSAMP).reshape(W.shape)[sat]
        chi += float((pen ** 2).sum())
    if t0_prior is not None:
        chi += ((t0 - t0_prior[0]) / t0_prior[1]) ** 2
    return chi, q


def model_waveforms(plane, pos, p0, w, t0, q, hyper):
    return (build_matrix(plane, pos, p0, w, t0, hyper) @ q).reshape(len(pos), NSAMP)


# --------------------------------------------------------------------- fits
def fit_plane_raw(P, plane, p0_init, w_init, t0_init, hyper=None, fix_p0w=None,
                  t0_prior=None):
    """Two-stage fit: coarse (p0, w, t0) grid, then Nelder-Mead. With
    ``fix_p0w=(p0, w)`` only t0 and the charge profile are fitted — that is the
    'ref-pinned' configuration used for calibration and for the chi2(v) scan.
    ``t0_prior=(t0_pred, sigma)`` is passed through to :func:`chi2_plane`."""
    _require_cal()
    hyper = hyper or HYPER
    W, noise, pos, sat = prep_plane(P, plane)
    dof = int((~sat).sum())

    if fix_p0w is not None:
        p0f, wf = fix_p0w
        grid = np.arange(t0_init - 240, t0_init + 241, 20.0)
        cs = [chi2_plane(plane, W, noise, pos, sat, p0f, wf, t, hyper,
                         t0_prior=t0_prior)[0]
              for t in grid]
        j = int(np.argmin(cs))
        g2 = np.arange(grid[j] - 20, grid[j] + 21, T0_STEP)
        cs2 = [chi2_plane(plane, W, noise, pos, sat, p0f, wf, t, hyper,
                          t0_prior=t0_prior)[0]
               for t in g2]
        t0b = float(g2[int(np.argmin(cs2))])
        chi, q = chi2_plane(plane, W, noise, pos, sat, p0f, wf, t0b, hyper,
                            t0_prior=t0_prior)
        return dict(chi2=chi, dof=dof, p0=p0f, w=wf, t0=t0b, q=q, nfev=len(cs) + len(cs2))

    nfev = 0
    best = (np.inf, p0_init, w_init, t0_init)
    for t0 in np.arange(t0_init - 120, t0_init + 121, 40.0):
        for dp in (-0.8, 0.0, 0.8):
            for dw in (-2.4e-3, -0.8e-3, 0.0, 0.8e-3, 2.4e-3):
                c, _ = chi2_plane(plane, W, noise, pos, sat,
                                  p0_init + dp, w_init + dw, t0, hyper,
                                  t0_prior=t0_prior)
                nfev += 1
                if c < best[0]:
                    best = (c, p0_init + dp, w_init + dw, t0)

    def obj(v):
        return chi2_plane(plane, W, noise, pos, sat, v[0], v[1], v[2], hyper,
                          snap_t0=False, t0_prior=t0_prior)[0]

    v0 = np.array(best[1:])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.3, maxiter=140,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0], [0.4, 0, 0], [0, 1.5e-3, 0],
                                   [0, 0, 20]])))
    chi, q = chi2_plane(plane, W, noise, pos, sat, r.x[0], r.x[1], r.x[2],
                        hyper, snap_t0=False, t0_prior=t0_prior)
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
