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

Two channel wildcards ride the same bundle (``prep_plane``): ``DEAD``
channels are censored samples (no signal in either direction, dropped from
the chi2 like a saturated one); ``HOT`` channels DO carry signal (coherent or
correlated noise fit as a cluster -- HANDOFF_D_NOISY_CHANNELS.md) and stay in
the fit, just down-weighted by inflating their noise (``HOT_NOISE_INFLATION``)
rather than dropped, so a real track that happens to cross one is not
penalised for the crossing.
"""
from __future__ import annotations

import os

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
HOT: dict = {}            # per-plane hot-channel arrays; see prep_plane
#: How far a hot channel's noise is inflated in the chi2 (HANDOFF_D_NOISY_
#: CHANNELS.md item 3: "cap their influence... a per-channel weight in the
#: NNLS design matrix"). Deliberately finite, unlike DEAD's 1e9: a hot
#: channel DOES carry signal and stays in the fit (rows are not censored --
#: `sat` is untouched), just down-weighted by 1/HOT_NOISE_INFLATION**2 in the
#: chi2 sum, so it cannot dominate but a real track crossing it is not
#: penalised for the crossing.
#:
#: **Scanned 2026-09-08 on D/run_145 and it does not matter.** 10, 30 and 100
#: were fitted over one fixed set of 3 600 real triggers and agree to three
#: decimals on every metric, in every stratum -- convergence, chi2/dof, angle,
#: p0. HANDOFF_HOT_WILDCARD_TUNING.md Sec. 4.1 proposed that 10 was "probably
#: too weak" and that this was why the first hot-masked re-run regressed; it
#: was not. Paired per-event, a window whose SEED is unchanged by the mask
#: fits identically with it (chi2/dof 39.45 -> 38.54, median |dp0| 0.00 mm):
#: everything the wildcard does, it does through seeding, not weighting.
#: 10 is kept because nothing argues for anything else.
#: ``WFT_HOT_NOISE_INFLATION`` overrides it, which is how that scan is run.
HOT_NOISE_INFLATION = float(os.environ.get('WFT_HOT_NOISE_INFLATION', 10.0))

_smear_cache: dict = {}
_lp_cache: dict = {}
_tt_cache: dict = {}
T0_STEP = 5.0             # t0 quantisation for the cached time tensors


def use_calibration(cal: CalibrationBundle) -> None:
    """Install a calibration bundle as the model's calibration."""
    global CAL, TGRID, TMPL, GAIN, DT_XY, PITCH, SNS, SAT, DT, K, UK, HYPER, \
        SHARE_MODE, DEAD, HOT
    check_kernel_ordering(cal.hyper, where=f'bundle {cal.detector}/{cal.run_key}')
    CAL = cal
    DEAD = {p: np.asarray(sorted(ch), dtype=int)
            for p, ch in getattr(cal, 'dead', {}).items() if len(ch)}
    HOT = {p: np.asarray(sorted(ch), dtype=int)
          for p, ch in getattr(cal, 'hot', {}).items() if len(ch)}
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
    either: no information in either direction.

    Hot channels (bundle ``hot``, HANDOFF_D_NOISY_CHANNELS.md) are NOT
    censored -- they carry real signal, just noise/correlated-noise-shaped
    signal, and a real track can cross one. Their rows stay live in the fit
    (``sat`` untouched) with noise inflated by a bounded
    ``HOT_NOISE_INFLATION`` rather than to 1e9, so they are down-weighted --
    capped influence, not zero -- and cannot pull the fit on their own."""
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
    h = HOT.get(plane)
    if h is not None and len(h):
        rows = np.isin(ch, h)
        if rows.any():
            noise[rows] *= HOT_NOISE_INFLATION
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
    chi, q = _solve_nnls(M, W, noise, sat, censor)
    if q is None:
        return np.inf, None
    if t0_prior is not None:
        chi += ((t0 - t0_prior[0]) / t0_prior[1]) ** 2
    return chi, q


def _solve_nnls(M, W, noise, sat, censor=True):
    """(chi2, q) for a design matrix of any width: the charge profile solved by
    NNLS with the per-strip noise weighting, saturated samples censored.

    Factored out of :func:`chi2_plane` so the two-track model (:func:`chi2_plane_two`)
    uses exactly the same weighting, censoring and dead/hot handling rather than a
    second copy of it. Numerics are unchanged — ``tests/test_model_regression.py``
    pins them."""
    ok = ~sat.reshape(-1)
    if not ok.any():
        return np.inf, None
    Wt = np.repeat(1.0 / noise, NSAMP)
    A = (M * Wt[:, None])[ok]
    y = (W / noise[:, None]).reshape(-1)[ok]
    try:
        q, rn = nnls(A, y, maxiter=50 * M.shape[1])
    except Exception:
        return np.inf, None
    chi = rn * rn
    if censor and sat.any():
        model = (M @ q).reshape(W.shape)
        pen = np.maximum(0.0, W[sat] - model[sat]) / np.repeat(
            noise, NSAMP).reshape(W.shape)[sat]
        chi += float((pen ** 2).sum())
    return chi, q


def model_waveforms(plane, pos, p0, w, t0, q, hyper):
    return (build_matrix(plane, pos, p0, w, t0, hyper) @ q).reshape(len(pos), NSAMP)


# ------------------------------------------------------- two tracks, one plane
# Nothing in the forward model assumes one track except build_matrix being
# called once: the charge profile is already a free non-negative vector, so a
# second track is a second block of columns in the SAME linear solve.
#
#     W  ~  M(theta_a) qa + M(theta_b) qb ,   qa, qb >= 0
#
# The outer parameters are (p0, w, t0) per track, 6 in all (5 with t0 tied).
# Design and acceptance criteria: sept26_prelim_analysis/HANDOFF_JOINT_TWO_TRACK_FIT.md.

#: Below this the two tracks are not distinguishable and the charge split
#: between their (identical) column blocks is arbitrary. Transverse: a
#: charge-weighted r.m.s. over the drift column, so two tracks that CROSS in
#: this plane still count as separated (a mid-column distance would call them
#: degenerate at the crossing). Temporal: one depth bin, since two tracks at
#: the same place but a bin apart in time are resolved by the profile alone.
TWO_MIN_SEP_MM = 1.2          # ~1.5 strip pitches
TWO_MIN_DT_NS = 60.0          # one depth bin
#: Backstop only. The end-to-end failure -- two straight lines fitting ONE
#: track's column by taking half of it each in depth -- is handled by measuring
#: the separation *where both children have charge* (see
#: :func:`two_track_separation`), not by a threshold on the overlap itself. A
#: threshold there is a bad instrument: with the same quantity in the
#: optimiser's barrier the fit parks exactly on it, and the guard then decides
#: genuine 15 mm pairs on the fourth decimal (measured 2026-09-16).
TWO_MIN_OVERLAP = 0.05
#: Multiplicative barrier on the collapsed basin. Scale-free on purpose: chi2
#: here runs 1e3-1e5 and an additive constant would be a threshold in disguise.
#: It is optimisation hygiene only -- the real protection is the ``guards_ok``
#: verdict that ``wft.reco.fit_plane_two`` applies to the result.
TWO_BARRIER = 2.0


def two_track_separation(pa, pb, wgt=None) -> float:
    """R.m.s. transverse distance between two tracks over the drift column [mm],
    weighted by ``wgt`` -- one weight per depth bin.

    The weight is what makes this the right question. Weighted by the depths
    where BOTH children carry charge, it is large for two tracks side by side
    (including two that cross: the crossing is one depth out of eighteen) and
    zero for one track cut in half end to end, where there is no depth at which
    both children exist. A zero-sum weight therefore returns 0 -- not a fallback
    to the unweighted r.m.s., which is exactly the number the end-to-end failure
    makes look enormous."""
    d = (pa[0] - pb[0]) + (pa[1] - pb[1]) * UK
    if wgt is None:
        return float(np.sqrt(np.mean(d * d)))
    w = np.asarray(wgt, float)[:len(d)]
    s = w.sum()
    return float(np.sqrt((w * d * d).sum() / s)) if s > 0 else 0.0


def two_track_distinguishability(pa, pb, wgt=None) -> float:
    """How far apart the two tracks are, in units of the resolvability floor;
    < 1 means the pair is degenerate. Transverse and temporal separation add
    in quadrature: either one alone is enough."""
    sep = two_track_separation(pa, pb, wgt)
    dt = abs(pa[2] - pb[2])
    return float(np.hypot(sep / TWO_MIN_SEP_MM, dt / TWO_MIN_DT_NS))


def constrained_bins(t0: float) -> np.ndarray:
    """Which depth bins the DAQ window actually constrains, for a track at t0.

    A bin whose charge arrives after the last sample contributes an almost-zero
    column, so NNLS is free to park an arbitrary amount of charge in it -- which
    it does: run_145 tracks carry fitted ``q_sum`` up to 1e26, and ``q_uend``
    reads the last bin for nearly every track. That is a property of the
    production model, not of this fit, but any guard built on the charge profile
    has to look only where the profile means something."""
    last = float(TS[-1]) if len(TS) else 0.0
    arr = t0 + UK
    return (arr >= -0.5 * SNS) & (arr <= last)


def profile_overlap(qa, qb, bins=None) -> float:
    """How much of the drift column the two children share, 0 to 1: the
    histogram intersection of their normalised charge profiles. Two real
    coincident tracks each cross the whole gap and overlap almost completely;
    two halves of one column do not overlap at all."""
    qa, qb = _masked(qa, bins), _masked(qb, bins)
    sa, sb = qa.sum(), qb.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.minimum(qa / sa, qb / sb).sum())


def _masked(q, bins=None) -> np.ndarray:
    q = np.asarray(q, float)
    return q if bins is None else q * np.asarray(bins, float)[:len(q)]


def common_weight(qa, qb, bins=None) -> np.ndarray:
    """Per depth bin, how much of the column the two children SHARE: the
    pointwise minimum of their normalised charge profiles. Zero everywhere for
    one track cut in half end to end."""
    ma, mb = _masked(qa, bins), _masked(qb, bins)
    sa, sb = ma.sum(), mb.sum()
    return (np.minimum(ma / sa, mb / sb) if sa > 0 and sb > 0
            else np.zeros_like(ma))


def build_matrix_two(plane, pos, pa, pb, hyper):
    """(strip x sample, 2K) design matrix of two straight tracks."""
    return np.hstack([build_matrix(plane, pos, pa[0], pa[1], pa[2], hyper),
                      build_matrix(plane, pos, pb[0], pb[1], pb[2], hyper)])


def chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper, censor=True,
                   snap_t0=True):
    """chi2 of the two-track model, both charge profiles profiled out together.

    ``pa``/``pb`` are ``(p0, w, t0)``. Returns ``(chi2, qa, qb)``. Setting
    ``qb = 0`` reproduces the one-track model exactly, so chi2 here can never
    exceed the one-track chi2 at the same ``pa`` — which is what makes
    ``dchi2 = chi2_one - chi2_two`` a model-selection statistic and not a fit
    artefact."""
    if snap_t0:
        pa = (pa[0], pa[1], round(pa[2] / T0_STEP) * T0_STEP)
        pb = (pb[0], pb[1], round(pb[2] / T0_STEP) * T0_STEP)
    M = build_matrix_two(plane, pos, pa, pb, hyper)
    chi, q = _solve_nnls(M, W, noise, sat, censor)
    if q is None:
        return np.inf, None, None
    return chi, q[:K], q[K:]


def _two_pack(pa, pb, tie_t0):
    return (np.array([pa[0], pa[1], pb[0], pb[1], 0.5 * (pa[2] + pb[2])])
            if tie_t0 else np.array([pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]]))


def _two_unpack(v, tie_t0):
    if tie_t0:
        return (v[0], v[1], v[4]), (v[2], v[3], v[4])
    return (v[0], v[1], v[2]), (v[3], v[4], v[5])


_TWO_SIMPLEX_FREE = np.array([[0, 0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0, 0],
                              [0, 1.5e-3, 0, 0, 0, 0], [0, 0, 20, 0, 0, 0],
                              [0, 0, 0, 0.4, 0, 0], [0, 0, 0, 0, 1.5e-3, 0],
                              [0, 0, 0, 0, 0, 20]], float)
_TWO_SIMPLEX_TIED = np.array([[0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0],
                              [0, 1.5e-3, 0, 0, 0], [0, 0, 0.4, 0, 0],
                              [0, 0, 0, 1.5e-3, 0], [0, 0, 0, 0, 20]], float)


#: How many of the offered starting points are actually refined. All of them
#: are scored first (one chi2 each, ~1 ms); Nelder-Mead, which costs a few
#: hundred of those, runs only on the best few.
TWO_N_REFINE = 2


def fit_plane_two_raw(W, noise, pos, sat, plane, starts, hyper=None,
                      wgt=None, maxiter=220, maxiter_polish=140,
                      n_refine=TWO_N_REFINE, bins=None):
    """Fit two tracks to one prepared plane window from several starting points.

    ``starts``: list of ``(pa, pb, tie_t0)``. All are scored, the best
    ``n_refine`` are refined by Nelder-Mead on (p0, w, t0) per track — 6
    parameters, or 5 with ``tie_t0`` — with the collapsed basin held off by a
    multiplicative barrier, and the winner is polished once more. Returns the
    best result as a dict, or None.

    The caller prepares the window (``prep_plane``) once and passes it in: the
    two-track fit is always a SECOND look at a window a one-track fit has
    already seen, so re-preparing it here would be waste."""
    _require_cal()
    hyper = hyper or HYPER
    dof = int((~sat).sum())

    def obj(v, tie_t0):
        pa, pb = _two_unpack(v, tie_t0)
        c, qa, qb = chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper,
                                   snap_t0=False)
        if not np.isfinite(c):
            return np.inf
        # the barrier measures exactly what the guard will: separation at the
        # depths where both children have charge. One quantity, so the
        # optimiser is not pushed onto a threshold the guard then rules on.
        d = two_track_distinguishability(pa, pb, common_weight(qa, qb, bins))
        return c * (1.0 + TWO_BARRIER * (1.0 - d) ** 2) if d < 1.0 else c

    scored = sorted((obj(_two_pack(pa, pb, tie), tie), i, (pa, pb, tie))
                    for i, (pa, pb, tie) in enumerate(starts))
    nfev = len(starts)
    best = None
    for c0, _i, (pa, pb, tie) in scored[:max(1, n_refine)]:
        if not np.isfinite(c0):
            continue
        v0 = _two_pack(pa, pb, tie)
        simplex = v0 + (_TWO_SIMPLEX_TIED if tie else _TWO_SIMPLEX_FREE)
        r = minimize(obj, v0, args=(tie,), method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=0.3, maxiter=maxiter,
                                  initial_simplex=simplex))
        nfev += r.nfev
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x, tie)
    if best is None:
        return None
    r = minimize(obj, best[1], args=(best[2],), method='Nelder-Mead',
                 options=dict(xatol=5e-4, fatol=0.15, maxiter=maxiter_polish))
    nfev += r.nfev
    pa, pb = _two_unpack(r.x, best[2])
    chi, qa, qb = chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper,
                                 snap_t0=False)
    if qa is None:
        return None
    # label order: the track at the smaller position in the middle of the drift
    # column is 'a'. Mid-column rather than at the mesh because crossing tracks
    # swap order at the mesh but rarely at mid-column.
    u_mid = 0.5 * K * DT
    if pa[0] + pa[1] * u_mid > pb[0] + pb[1] * u_mid:
        pa, pb, qa, qb = pb, pa, qb, qa
    # the guards are measured on the FITTED profiles, not the parent's: the
    # separation that matters is the one at the depths where both tracks have
    # charge, and that is also what makes an end-to-end pair fail.
    common = common_weight(qa, qb, bins)
    return dict(chi2=float(chi), dof=dof, tie_t0=bool(best[2]), nfev=int(nfev),
                pa=tuple(float(x) for x in pa), pb=tuple(float(x) for x in pb),
                qa=qa, qb=qb, overlap=profile_overlap(qa, qb, bins),
                sep=two_track_separation(pa, pb, common),
                dist=two_track_distinguishability(pa, pb, common))


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
